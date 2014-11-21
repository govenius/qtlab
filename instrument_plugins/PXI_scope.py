# PXI_scope.py
# Joonas Govenius <joonas.goveius@aalto.fi>, 2014
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA

from instrument import Instrument
import types
import logging
import numpy as np
from scipy import interpolate
import datetime
import pytz
from dateutil import tz
import os
import qt
import time
import itertools
import re
import ftplib
import StringIO
import pyTDMS
from jinja2 import Template
from collections import OrderedDict

class PXI_scope(Instrument):
    '''
    This is a driver for arming and reading out traces from a "scope"
    implemented in an NI PXI system.
    
    Basic usage:
    
    pxi = instruments.create('pxi', 'PXI_scope', address='<IP address>', reset=False)
    pxi.set_points(300)
    pxi.set_average_power(4) # average 2**4 triggers
    
    # Optional: call arm() separately first and do other stuff
    # here if the acquisition is expected to take a long time
    
    traces = pxi.get_traces(arm_first=True) # blocks until data is returned.
    
    # plot the traces...
    from plot import plotbridge_plot as plt
    p = plt(name='PXI scope test', overwrite=True)
    p.set_xlabel('time (us)')
    p.set_ylabel('V (mV)')

    p.clear()
    for i,ch in enumerate([ 'AI0', 'AI1' ]):
      p.add_trace(traces['timeaxis'], traces[ch].real,
        x_plot_units=1e-6, y_plot_units=1e-3,
        points=True, lines=True,
        color=i, linetype=1, title=('Re[%s]'%ch))
      p.add_trace(traces['timeaxis'], traces[ch].imag,
        x_plot_units=1e-6, y_plot_units=1e-3,
        points=True, lines=True,
        color=i, linetype=2, title=('Im[%s]'%ch))

    p.update()
    p.run()

    '''

    def __init__(self, name, address, reset=False):
        '''
        Initializes the PXI_scope.

        Input:
          name (string)    : name of the instrument
          address (string) : path to log files
          reset (bool)     : resets to default values, default=False
        '''
        logging.info(__name__ + ' : Initializing instrument PXI_scope')
        Instrument.__init__(self, name, tags=['physical'])

        # Add some global constants
        self._address = address

        self._channels = (0,1)
        
        self.add_parameter('average_power',
            flags=Instrument.FLAG_GETSET, minval=0, maxval=31, type=types.IntType)

        self.add_parameter('points',
            flags=Instrument.FLAG_GETSET, minval=3, maxval=16000, type=types.IntType) # Limited by averaging buffer sizes in FPGA

        self.add_parameter('most_recent_raw_datafile',
            flags=Instrument.FLAG_GET, type=types.StringType)

        self.add_parameter('datafile_prefix',
            flags=Instrument.FLAG_GETSET, type=types.StringType)

        self.add_function('get_all')
        self.add_function('arm')
        self.add_function('get_traces')

        # Must reset since parameters do not persist
        self.reset()
        
        self.get_all()
        
    def get_all(self):
        '''
        Reads all implemented parameters from the instrument,
        and updates the wrapper.

        Input:
            None

        Output:
            None
        '''
        logging.info(__name__ + ' : get all')

        self.get_points()
        self.get_average_power()
        self.get_datafile_prefix()
        self.get_most_recent_raw_datafile()
        
    def reset(self):
        '''
        Resets the instrument to default values

        Input:
            None

        Output:
            None
        '''
        logging.info(__name__ + ' : resetting instrument')
        self._average_power = 0
        self._points = 1024
        self._most_recent_trace = None
        self._datafile_prefix = 'PXI_scope_data'
        self.__regenerate_config_xml()
      
    def arm(self):
        '''
        Arm the scope.
        '''

        logging.debug(__name__ + ' : arming...')
        
        ftp = self.__get_connection()
        try:
          buffer = StringIO.StringIO()
          try:
            # First get the name of the previously acquired trace
            ftp.retrbinary('RETR most_recent_trace.txt', buffer.write)
            self._most_recent_trace = buffer.getvalue()
          except ftplib.error_perm:
            self._most_recent_trace = None # there is no previous trace
          finally:
            buffer.close()
          
          # Create an empty file which the PXI takes as a signal to arm
          buffer = StringIO.StringIO()
          buffer.write('\n')
          ftp.storbinary('STOR arm.signal', buffer)
          buffer.close()
        finally:
          ftp.quit()
      
    def estimate_min_acquisition_time(self):
        ''' Give a lower bound for the acquisition time.
        This is close to exact if the time spent waiting for triggers can be ignored. '''

        boxcar_filter_length_in_samples = 256 # ATM, hard coded in the RT code
        adc_sample_rate = 250e6 # hard coded in the FPGA
        return (self.get_points() * 2**(self.get_average_power())
                              * boxcar_filter_length_in_samples/adc_sample_rate)

    def get_traces(self, arm_first=False):
        '''
        Get the traces.
        '''

        logging.debug(__name__ + ' : getting traces...')
        
        estimated_min_time = self.estimate_min_acquisition_time()
        
        for attempt in range(5):
        
          try:
            if arm_first:
              self.arm()
              time_armed = time.time()
              qt.msleep( estimated_min_time )
          
            ftp = self.__get_connection()
            while time.time() < time_armed + 60 + 4*estimated_min_time:
              buffer = StringIO.StringIO()
              try:
                # Get the name of the most recently acquired data
                ftp.retrbinary('RETR most_recent_trace.txt', buffer.write)
                most_recent_trace = buffer.getvalue()
              except ftplib.error_perm:
                most_recent_trace = None # there is no previous trace or new trace (yet)
              finally:
                buffer.close()
              
              if self._most_recent_trace != most_recent_trace:
                self._most_recent_trace = most_recent_trace
                
                # download the data to a local file
                most_recent_trace = os.path.split(most_recent_trace)[-1] # strip path, keep filename only
                local_datafilepath = os.path.join(qt.config.get('tempdir'), most_recent_trace)
                with open(local_datafilepath, 'wb') as fdata:
                  ftp.retrbinary('RETR %s' % most_recent_trace, fdata.write)
                ftp.delete(most_recent_trace) # delete from PXI
                self.update_value('most_recent_raw_datafile', local_datafilepath)
                
                # read the data file
                objs, rawdata = pyTDMS.read(local_datafilepath)
                
                # The sample rate is the same for both channels...
                samplerate = objs[objs.keys()[1]][3]['Sample rate'][1]
                
                channels = [
                       np.array(rawdata["/'Time Domain'/'AI%d_real'" % ch])
                  + 1j*np.array(rawdata["/'Time Domain'/'AI%d_imag'" % ch])
                  for ch in range(2) ]
                
                # There is a large DC offset at the inputs of the ADCs, if not properly matched to 50 Ohm at DC.
                # This shows up in the down-converted data at negatice DDC freq.
                # We almost never care about that, so filter it out.
                pts = len(channels[0])
                for ch in range(2):
                  ft = np.fft.fft(channels[ch])
                  ft[pts/2] = 0  # Assuming that the ratio of down-sampled data frequency and DDC freq is 2
                  channels[ch] = np.fft.ifft(ft)
                
                return OrderedDict( [
                  ("AI0", channels[0]),
                  ("AI1", channels[1]),
                  ('timeaxis', np.arange(pts) / samplerate ) # time axis
                ]) # convert data to a numpy array
                
              qt.msleep(2.) # try again in a little bit

          except:
            logging.exception('Attempt %d to get traces from scope failed!', attempt)
            qt.msleep(10. + attempt*(estimated_min_time))
          finally:
            try: ftp.quit()
            except: pass
        
        assert False, 'Failed to get traces from PXI scope. Are triggers being sent?'

    def do_get_most_recent_raw_datafile(self):
        '''
        Return the number of points per trace.
        '''
        return self._most_recent_trace

    def do_get_datafile_prefix(self):
        '''
        Return the number of points per trace.
        '''
        return self._datafile_prefix

    def do_set_datafile_prefix(self, val):
        '''
        Set the datafile_prefix. Only affects the naming of the raw data file.
        '''
        self._datafile_prefix = val
        self.__regenerate_config_xml()

    def do_get_points(self):
        '''
        Return the number of points per trace.
        '''
        return self._points

    def do_set_points(self, val):
        '''
        Set the number of points per trace.
        '''
        self._points = val
        self.__regenerate_config_xml()

    def do_get_average_power(self):
        '''
        Return the number of averaged triggers.
        Specified as a power of two! E.g. 8 means 2**8 averages.
        '''
        return self._average_power

    def do_set_average_power(self, val):
        '''
        Set the number of averaged triggers.
        Specified as a power of two! E.g. 8 means 2**8 averages.
        '''
        self._average_power = val
        self.__regenerate_config_xml()

    def __get_connection(self, change_to_datadir=True):
        ftp = ftplib.FTP(self._address, timeout=3)
        ftp.login('USER', 'PASS')
        if change_to_datadir: ftp.cwd('data')
        return ftp

    def __regenerate_config_xml(self):
        '''
        Regenerate config.xml and upload it to the PXI.
        '''

        # Regenerate the config file (as string)
        template = Template(PXI_scope._config_template)
        config = template.render(points=self._points,
                        average_power=self._average_power,
                        datafile_prefix=self._datafile_prefix)

        # Upload it to the target
        ftp = self.__get_connection(change_to_datadir=False)
        try:
          # Create an empty file which the PXI takes as a signal to arm
          buffer = StringIO.StringIO(config)
          ftp.storbinary('STOR Config.xml', buffer)
          buffer.close()
        finally:
          ftp.quit()



    _config_template = (
'''<?xml version='1.0' standalone='yes' ?>
<LVData xmlns="http://www.ni.com/LVData">
<Version>13.0</Version>
<Cluster>
<Name>Acquisition and Logging Configuration</Name>
<NumElts>14</NumElts>
<DBL>
<Name>Data Rate (Hz)</Name>
<Val>1000.00000000000000</Val>
</DBL>
<I32>
<Name>Samples per Read</Name>
<Val>{{ points }}</Val>
</I32>
<I32>
<Name>Number of Reads to Save</Name>
<Val>{{ average_power }}</Val>
</I32>
<String>
<Name>Data File Name</Name>
<Val>{{ datafile_prefix }}</Val>
</String>
<Array>
<Name>TDMS Properties</Name>
<Dimsize>8</Dimsize>
<Cluster>
<Name>TDMS Properties</Name>
<NumElts>2</NumElts>
<String>
<Name>Name</Name>
<Val>Aalto University</Val>
</String>
<String>
<Name>Value</Name>
<Val>JMG</Val>
</String>
</Cluster>
<Cluster>
<Name>TDMS Properties</Name>
<NumElts>2</NumElts>
<String>
<Name>Name</Name>
<Val>Author</Val>
</String>
<String>
<Name>Value</Name>
<Val>[DeveloperName]</Val>
</String>
</Cluster>
<Cluster>
<Name>TDMS Properties</Name>
<NumElts>2</NumElts>
<String>
<Name>Name</Name>
<Val>Operator</Val>
</String>
<String>
<Name>Value</Name>
<Val>[OperatorName]</Val>
</String>
</Cluster>
<Cluster>
<Name>TDMS Properties</Name>
<NumElts>2</NumElts>
<String>
<Name>Name</Name>
<Val>TestSystem</Val>
</String>
<String>
<Name>Value</Name>
<Val>[SystemName]</Val>
</String>
</Cluster>
<Cluster>
<Name>TDMS Properties</Name>
<NumElts>2</NumElts>
<String>
<Name>Name</Name>
<Val>Site</Val>
</String>
<String>
<Name>Value</Name>
<Val>[Location]</Val>
</String>
</Cluster>
<Cluster>
<Name>TDMS Properties</Name>
<NumElts>2</NumElts>
<String>
<Name>Name</Name>
<Val>Line</Val>
</String>
<String>
<Name>Value</Name>
<Val>[Line]</Val>
</String>
</Cluster>
<Cluster>
<Name>TDMS Properties</Name>
<NumElts>2</NumElts>
<String>
<Name>Name</Name>
<Val>Machine</Val>
</String>
<String>
<Name>Value</Name>
<Val>[MachineName]</Val>
</String>
</Cluster>
<Cluster>
<Name>TDMS Properties</Name>
<NumElts>2</NumElts>
<String>
<Name>Name</Name>
<Val>Description</Val>
</String>
<String>
<Name>Value</Name>
<Val>[Description]</Val>
</String>
</Cluster>
</Array>
<I32>
<Name>Number of Channels</Name>
<Val>2</Val>
</I32>
<Array>
<Name>Channel Scale Array</Name>
<Dimsize>4</Dimsize>
<DBL>
<Name>Ch0 Scale</Name>
<Val>1.00000000000000</Val>
</DBL>
<DBL>
<Name>Ch0 Scale</Name>
<Val>1.00000000000000</Val>
</DBL>
<DBL>
<Name>Ch0 Scale</Name>
<Val>1.00000000000000</Val>
</DBL>
<DBL>
<Name>Ch0 Scale</Name>
<Val>1.00000000000000</Val>
</DBL>
</Array>
<Array>
<Name>Channel Names</Name>
<Dimsize>4</Dimsize>
<String>
<Name>String</Name>
<Val>AI0</Val>
</String>
<String>
<Name>String</Name>
<Val>AI1</Val>
</String>
<String>
<Name>String</Name>
<Val>unused2</Val>
</String>
<String>
<Name>String</Name>
<Val>unused3</Val>
</String>
</Array>
<DBL>
<Name>Trigger - RMS Value</Name>
<Val>2.00000000000000</Val>
</DBL>
<I32>
<Name>Trigger - Channel</Name>
<Val>0</Val>
</I32>
<I32>
<Name>Trigger - Time (Minute)</Name>
<Val>30</Val>
</I32>
<Boolean>
<Name>Enable RMS Trigger</Name>
<Val>1</Val>
</Boolean>
<Boolean>
<Name>Enable Time Trigger</Name>
<Val>0</Val>
</Boolean>
<Boolean>
<Name>Always Log</Name>
<Val>0</Val>
</Boolean>
</Cluster>
</LVData>
'''
        )
