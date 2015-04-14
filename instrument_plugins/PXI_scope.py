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
    
    traces = pxi.get_traces(arm_first=True) # with arm_first, this blocks until data is returned.
    
    # plot the traces...
    from plot import plotbridge_plot as plt
    p = plt(name='PXI scope test', overwrite=True)
    p.set_xlabel('time (us)')
    p.set_ylabel('V (mV)')

    p.clear()
    for i,ch in enumerate([ 'AI0', 'AI1', 'AI0_stddev', 'AI1_stddev' ]):
      # "stddev" is the pointwise standard deviation of the ensemble of averaged traces
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
            flags=Instrument.FLAG_GETSET, minval=3, maxval=8194, type=types.IntType) # Limited by averaging buffer sizes in FPGA

        self.add_parameter('boxcar_power',
            flags=Instrument.FLAG_GETSET, minval=1, maxval=20, type=types.IntType)
        self.add_parameter('ddc_per_clock_scaled',
            flags=Instrument.FLAG_GETSET, minval=0, maxval=2**16, type=types.IntType)
        self.add_parameter('int_trigger_period_in_clockcycles',
            flags=Instrument.FLAG_GETSET, minval=1, type=types.IntType)
        self.add_parameter('ext_trigger',
            flags=Instrument.FLAG_GETSET, type=types.BooleanType)
        self.add_parameter('dio_default_value',
            flags=Instrument.FLAG_GETSET, minval=0, maxval=2**8-1, type=types.IntType)
        self._DIO_BITS = 8
        self.add_parameter('flip_times', flags=Instrument.FLAG_GETSET, type=types.ListType,
                           channels=tuple(range(self._DIO_BITS)), channel_prefix='dio_bit%d_')

        self._DIO_FLIPS = 20 # max number of DIO flips (size of "dio_values" and "dio_set_times"), hard-coded in FPGA
        self._ADC_SAMPLE_RATE = 250e6 # hard coded in the FPGA
        self._CLOCK_RATE = 125e6 # hard coded in the FPGA

        self.add_parameter('most_recent_raw_datafile',
            flags=Instrument.FLAG_GET, type=types.StringType)

        self.add_parameter('datafile_prefix',
            flags=Instrument.FLAG_GETSET, type=types.StringType)

        self.add_function('get_all')
        self.add_function('arm')
        self.add_function('get_traces')
        self.add_function('clear_triggers')
        self.add_function('plot_outputs')

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

        self.get_ddc_per_clock_scaled()
        self.get_boxcar_power()
        self.get_int_trigger_period_in_clockcycles()
        self.get_ext_trigger()

        self.get_dio_default_value()
        for i in range(self._DIO_BITS):
            self.get('dio_bit%d_flip_times' % i)

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
        self._ddc_per_clock_scaled = 768
        self._boxcar_power = 9
        self._int_trigger_period_in_clockcycles = 100
        self._ext_trigger = True

        self._dio_default_value = 0
        self._dio_flip_times = [ tuple() for i in range(self._DIO_BITS) ]

        self._most_recent_trace = None
        self._datafile_prefix = 'PXI_scope_data'

        self.__regenerate_config_xml()

    def plot_outputs(self, time_unit=1e-6):
        '''
        Visualize the triggers.
        '''

        dio_set_times, dio_values = self._combine_flip_times()

        nflips = np.where(np.array(dio_set_times) < 0)[0][0]

        # duplicate each point
        if nflips > 0:
          times = np.array([ dio_set_times[:nflips], dio_set_times[:nflips] ], dtype=np.float).T.reshape((-1))
          triggers = np.array([ dio_values[:nflips], dio_values[:nflips] ], dtype=np.int).T.reshape((-1))
        else:
          assert  nflips == 0
          times = np.array([ 0 ], dtype=np.float)
          triggers = np.array([ dio_defaul_value ], dtype=np.int)

        times /= self._CLOCK_RATE

        print times * 1e6
        print triggers

        # add points at the beginning and end
        total = self.get_digitized_time()
        times = np.append([-0.05*total, times[0]], times[1:])
        triggers = np.append([self._dio_default_value, self._dio_default_value], triggers[:-1])
        times = np.append(times, [total, total, 1.05*total])
        triggers = np.append(triggers, [triggers[-1], self._dio_default_value, self._dio_default_value])

        print times * 1e6
        print triggers

        # convert bit masks to lists
        triggers = np.array([
            [ (1+i)*( (v&(1<<i)) > 0 ) for i in range(8) ]
            for v in triggers ])

        import plot
        p = plot.get_plot('PXI outputs').get_plot()
        p.clear()
        p.set_xlabel('time (%.1e s)' % time_unit)
        p.set_ylabel('channel')

        for i,trig in enumerate(triggers.T):
            p.add_trace(times, trig.astype(np.float),
                        x_plot_units=time_unit,
                        points=True, lines=True,
                        title=str(i))

        p.update()
        p.run()

    def clear_triggers(self):
        '''
        Clear all trigger flips. (But don't change default values.)
        '''
        for i in range(self._DIO_BITS):
            self.set('dio_bit%d_flip_times' % i, tuple())

    def do_get_flip_times(self, channel):
        ''' See set_dio_bitX_flip_times() '''
        return np.array(self._dio_flip_times[channel]).copy()

    def do_set_flip_times(self, flip_times, channel):
        '''
        Flip the trigger output for the given DIO bit (0-based index)
        at times specified in the 'flip_times' sequence.

        The value before the first flip will be the
        value specified in dio_default_value bitmask.

        The flip times are assumed to be in seconds
        and are rounded to closest clock cycle.
        '''
        assert channel >= 0
        assert channel < self._DIO_BITS
        assert len(flip_times) < self._DIO_FLIPS

        self._dio_flip_times[channel] = np.round(np.array(flip_times)*self._CLOCK_RATE).astype(np.int).copy()
        self.__regenerate_config_xml()

    def _combine_flip_times(self):
        '''
        Convert the individual flip_times + dio_default_value into
        one set times and one values array, as expected by the PXI code.
        '''

        last_flip = -1
        current_value = self._dio_default_value

        flip_times = [ list(self._dio_flip_times[i]) for i in range(self._DIO_BITS) ]
        combined_set_times = []
        combined_set_values = []

        while True:
            # first find when next_flip_occurs
            next_flip_at = min( (times[0] if len(times)>0 else np.inf) for times in flip_times )
            #print 'next flip at clock cycle %s' % next_flip_at
            if np.isinf(next_flip_at): break # no more flips
            
            # flip all bits with the same flip time
            for i in range(self._DIO_BITS):
                if len(flip_times[i]) > 0 and flip_times[i][0] == next_flip_at:
                    current_value ^= (1<<i)
                    del flip_times[i][0]

            combined_set_times.append(next_flip_at)
            combined_set_values.append(current_value)
            

        assert len(combined_set_times) < self._DIO_FLIPS-1, 'Too many flips in total. (Hard coded limit in PXI.)'

        # no more flips, pad with -1 set times at the end.
        # the PXI code requires at least one -1 at the end!
        while len(combined_set_times) < self._DIO_FLIPS:
          combined_set_times.append(-1)
          combined_set_values.append(0)

        assert len(combined_set_times) == len(combined_set_values)
        assert len(combined_set_times) == self._DIO_FLIPS, 'Wrong number of flips. (Hard coded limit in PXI.)'

        return combined_set_times, combined_set_values
      
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

    def get_digitized_time(self):
        ''' The total amount of time digitized. '''
        boxcar_filter_length_in_samples = 2**self.get_boxcar_power()
        return (1 + self.get_points()) * boxcar_filter_length_in_samples/self._ADC_SAMPLE_RATE
      
    def estimate_min_acquisition_time(self):
        ''' Give a lower bound for the acquisition time.
        This is close to exact if the time spent waiting for triggers can be ignored. '''
        return self.get_digitized_time() * 2**(self.get_average_power())

    def get_traces(self, arm_first=False):
        '''
        Get the traces.
        '''

        logging.debug(__name__ + ' : getting traces...')
        
        estimated_min_time = self.estimate_min_acquisition_time()
        
        for attempt in range(5):
        
          try:
            time_armed = time.time()
            if arm_first:
              self.arm()
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
                local_datafilepath = os.path.abspath(local_datafilepath)
                with open(local_datafilepath, 'wb') as fdata:
                  ftp.retrbinary('RETR %s' % most_recent_trace, fdata.write)
                ftp.delete(most_recent_trace) # delete from PXI
                self.update_value('most_recent_raw_datafile', local_datafilepath)
                
                # read the data file
                objs, rawdata = pyTDMS.read(local_datafilepath)
                
                # The sample rate is the same for both channels...
                samplerate = objs["/'Time Domain'/'AI0_real'"][3]['Sample rate'][1]
                
                means = [
                       np.array(rawdata["/'Time Domain'/'AI%d_real'" % ch])
                  + 1j*np.array(rawdata["/'Time Domain'/'AI%d_imag'" % ch])
                  for ch in range(2) ]
                stddevs = [
                       np.array(rawdata["/'Time Domain'/'AI%d_real_std'" % ch])
                  + 1j*np.array(rawdata["/'Time Domain'/'AI%d_imag_std'" % ch])
                  for ch in range(2) ]
                
                # There is a large DC offset at the inputs of the ADCs, if not properly matched to 50 Ohm at DC.
                # This shows up in the down-converted data at negatice DDC freq.
                # We almost never care about that, so filter it out.
                pts = len(means[0])
                for ch in range(2):
                  ft = np.fft.fft(means[ch])
                  ft[pts/2] = 0  # Assuming that the ratio of down-sampled data frequency and DDC freq is 2
                  means[ch] = np.fft.ifft(ft)
                
                return OrderedDict( [
                  ("AI0", means[0]), # mean value
                  ("AI1", means[1]),
                  ('timeaxis', np.arange(pts) / samplerate ), # time axis
                  ("AI0_stddev", stddevs[0]), # pointwise standard deviation
                  ("AI1_stddev", stddevs[1])
                ]) # convert data to a numpy array
                
              qt.msleep(2.) # try again in a little bit

            raise Exception('Acquisition taking too long. Estimated %g s, waited %g s.' % (
                estimated_min_time, time.time() - time_armed) )

          except Exception as e:
            if str(e).strip().lower() == 'human abort': raise
            logging.exception('Attempt %d to get traces from scope failed!', attempt)
            qt.msleep(10. + attempt*(estimated_min_time))
          finally:
            try: ftp.quit()
            except: pass

        assert False, 'All attempts to acquire data failed.'
        
    def do_get_most_recent_raw_datafile(self):
        return self._most_recent_trace

    def do_get_datafile_prefix(self):
        return self._datafile_prefix

    def do_set_datafile_prefix(self, val):
        '''
        Set the datafile_prefix. Only affects the naming of the raw data file.
        '''
        self._datafile_prefix = val
        self.__regenerate_config_xml()

    def do_get_points(self):
        '''
        Return the number of returned points per trace.
        '''
        return self._points
    def do_set_points(self, val):
        '''
        Set the number of returned points per trace.
        '''
        self._points = val
        self.__regenerate_config_xml()

    def do_get_boxcar_power(self):
        '''
        2^boxcar_power points from ADC are averaged into a single point
        right after DDC.
        '''
        return self._boxcar_power
    def do_set_boxcar_power(self, val):
        '''
        2^boxcar_power points from ADC are averaged into a single point
        right after DDC.
        '''
        self._boxcar_power = val
        self.__regenerate_config_xml()

    def do_get_ddc_per_clock_scaled(self):
        '''
        Specifies the digital downconversion frequency
        as (DDC freq./clock freq.)*2^16.
        '''
        return self._ddc_per_clock_scaled
    def do_set_ddc_per_clock_scaled(self, val):
        '''
        Specifies the digital downconversion frequency
        as (DDC freq./clock freq.)*2^16.
        '''
        self._ddc_per_clock_scaled = val
        self.__regenerate_config_xml()


    def do_get_int_trigger_period_in_clockcycles(self):
        '''
        Specifies the period for the internal trigger generator
        in clock cycles.
        '''
        return self._int_trigger_period_in_clockcycles
    def do_set_int_trigger_period_in_clockcycles(self, val):
        '''
        Specifies the period for the internal trigger generator
        in clock cycles.
        '''
        self._int_trigger_period_in_clockcycles = val
        self.__regenerate_config_xml()

    def do_get_ext_trigger(self):
        '''
        Use a trigger fed to the TRIG port on the transceiver.
        Otherwise the internal trigger generator is used.
        '''
        return self._ext_trigger
    def do_set_ext_trigger(self, val):
        '''
        Use a trigger fed to the TRIG port on the transceiver.
        Otherwise the internal trigger generator is used.
        '''
        self._ext_trigger = val
        self.__regenerate_config_xml()


    def do_get_dio_default_value(self):
        '''
        The value of the DIO outputs when idle (not digitizing).
        Specified as an 8-bit mask.
        '''
        return self._dio_default_value
    def do_set_dio_default_value(self, val):
        '''
        The value of the DIO outputs when idle (not digitizing).
        Specified as an 8-bit mask.
        '''
        self._dio_default_value = val
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

        dio_set_times, dio_values = self._combine_flip_times()

        # Regenerate the config file (as string)
        template = Template(PXI_scope._config_template)
        config = template.render(points=self._points,
                                 average_power=self._average_power,
                                 datafile_prefix=self._datafile_prefix,
                                 boxcar_width=self._boxcar_power,
                                 ddc_per_clock=self._ddc_per_clock_scaled,
                                 int_trigger_period=self._int_trigger_period_in_clockcycles,
                                 ext_trigger=int(bool(self._ext_trigger)),
                                 dio_set_times=dio_set_times,
                                 dio_values=dio_values,
                                 dio_default=self._dio_default_value,
                                 dio_size=self._DIO_FLIPS)

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
<Name>PXI scope configuration</Name>
<NumElts>11</NumElts>
<U16>
<Name>ADC samples per saved sample (as power of 2)</Name>
<Val>{{ boxcar_width }}</Val>
</U16>
<U32>
<Name>Saved samples</Name>
<Val>{{ points }}</Val>
</U32>
<I16>
<Name>ddc freq per clock freq * 2**16</Name>
<Val>{{ ddc_per_clock }}</Val>
</I16>
<U16>
<Name>Triggers to average (as power of 2)</Name>
<Val>{{ average_power }}</Val>
</U16>
<String>
<Name>Data File Name</Name>
<Val>{{ datafile_prefix }}</Val>
</String>
<U64>
<Name>Internal trigger period</Name>
<Val>{{int_trigger_period}}</Val>
</U64>
<Boolean>
<Name>External Trigger</Name>
<Val>{{ext_trigger}}</Val>
</Boolean>
<Boolean>
<Name>Force trigger</Name>
<Val>0</Val>
</Boolean>
<Array>
<Name>DIO set times</Name>
<Dimsize>{{dio_size}}</Dimsize>
{% for t in dio_set_times %}
<I64>
<Name>Numeric</Name>
<Val>{{t}}</Val>
</I64>
{% endfor %}
</Array>
<Array>
<Name>DIO values</Name>
<Dimsize>{{dio_size}}</Dimsize>
{% for v in dio_values %}
<U8>
<Name>Numeric</Name>
<Val>{{v}}</Val>
</U8>
{% endfor %}
</Array>
<U8>
<Name>DIO when not digitizing</Name>
<Val>{{dio_default}}</Val>
</U8>
</Cluster>
</LVData>
'''
        )
