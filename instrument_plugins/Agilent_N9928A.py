# Agilent_N9928A.py class, for commucation with an Agilent N9928A microwave vector network analyzer.
# Russell Lake <russell.lake@aalto.fi>, 2012
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
import visa
from visa import VisaIOError
import types
import logging
import numpy as np
import struct
import time
import qt
from collections import OrderedDict

class Agilent_N9928A(Instrument):
    '''
    This is the driver for the Agilent N9928A microwave vector network analyzer

    Usage:
    Initialize with
    <name> = instruments.create('<name>', 'Agilent_N9928A', address='<GBIP address>, reset=<bool>')
    '''
    def __init__(self, name, address, reset=False):
        '''
        Initializes the Agilent_N9928A, and communicates with the wrapper.

        Input:
          name (string)    : name of the instrument
          address (string) : GPIB address
          reset (bool)     : resets to default values, default=False
        '''
        logging.info(__name__ + ' : Initializing instrument Agilent_N1913A')
        Instrument.__init__(self, name, tags=['physical'])

        # Add some global constants
        self._address = address
        self._visainstrument = visa.ResourceManager().open_resource(self._address, timeout=30000) # timeout is in milliseconds
        self._visainstrument.read_termination = '\n'
        self._visainstrument.write_termination = '\n'

        # parameters 
        self.add_parameter('operating_mode',
            flags=Instrument.FLAG_GETSET, type=types.StringType,
                           format_map={"NA" : "network analyzer (NA)",
                                       "CPM" : "channel power meter (CPM)"})

        self.add_parameter('average_count',
            flags=Instrument.FLAG_GETSET, minval=1, maxval=100, type=types.IntType)

        self.add_parameter('frequency_center',
            flags=Instrument.FLAG_GETSET, minval=30E3, maxval=26.5E9, units = 'Hz', type=types.FloatType)

        self.add_parameter('frequency_start',
            flags=Instrument.FLAG_GETSET, minval=30E3, maxval=26.5E9, units = 'Hz', type=types.FloatType)

        self.add_parameter('frequency_stop',
            flags=Instrument.FLAG_GETSET, minval=30E3, maxval=26.5E9, units = 'Hz', type=types.FloatType)

        self.add_parameter('frequency_span',
            flags=Instrument.FLAG_GETSET, minval=1, maxval=26.5E9, units = 'Hz', type=types.FloatType)

        self.add_parameter('IF_bandwidth',
            flags=Instrument.FLAG_GETSET, minval=10, maxval=30E3, units = 'Hz', type=types.FloatType)

        self.add_parameter('points',
            flags=Instrument.FLAG_GETSET, minval=3, maxval=10001, type=types.IntType)

        self.add_parameter('sweep_mode', type=types.StringType,
                           flags=Instrument.FLAG_GETSET)

        self.add_parameter('velocity_factor', type=types.FloatType, minval=0, maxval=1,
                           flags=Instrument.FLAG_GETSET)
                           
        self.add_parameter('source_power', type=types.FloatType, minval=-45, maxval=0, units='dBm',
                           flags=Instrument.FLAG_GETSET)
                                              
        self.add_parameter('average_mode',
            flags=Instrument.FLAG_GETSET, type=types.StringType, 
                           format_map={"SWEep" : "sweep",
                                       "POINT" : "point"}
                           )

        self.add_parameter('source_power_mode',
            flags=Instrument.FLAG_GETSET, type=types.StringType, 
                           format_map={"HIGH" : "highest flat power level",
                                       "LOW" : "low flat power level",
                                       "MAN" : "manual setting"}
                           )                   
                           
        self.add_parameter('smoothing_mode',
            flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET, type=types.StringType, 
                           channels=(1,4), channel_prefix='ch%d_')

        self.add_parameter('current_measurement',
            flags=Instrument.FLAG_GETSET, type=types.StringType, channels=(1,4), 
                           channel_prefix='ch%d_',
                           format_map={"S11" : "forward reflection (S11)",
                                       "S21" : "forward transmission (S21)",
                                       "S12" : "reverse transmission (S12) ",
                                       "S22" : "reverse reflection (S22) ",
                                       "A" : "A receiver measurement",
                                       "B" : "B receiver measurement",
                                       "R1" : "Port 1 reference receiver measurement",
                                       "R2" : "Port 2 reference receiver measurement"}                              
                           )

        self.add_function('reset')
        self.add_function ('get_all')

        if (reset):
            self.reset()
        else:
            self.get_all()

    def reset(self):
        '''
        Resets the instrument to default values

        Input:
            None

        Output:
            None
        '''
        logging.info(__name__ + ' : resetting instrument')
        self._visainstrument.write('*RST')
        self.set_multi_trace()
        self.get_all()

    def set_multi_trace(self,param='D12_34'):    
        '''
        Set the multitrace configuration.
                
        Input:
            param (str): D1 - x 1
                         D2 - x 2
                         D12H - x2H
                         D1123 - x3H (NA mode only)
                         D12_34 - x4 (NA mode only)                                    
        Output:
            True / False.
        '''
        self._visainstrument.write('DISP:WIND:SPL %s' % param)

    def get_multi_trace(self):
      '''
      Get the multitrace window configuration.
      Used in modes: NA.

      Input:
          None

      Output:
           param (str): D1 - x 1
                         D2 - x 2
                         D12H - x2H
                         D1123 - x3H (NA mode only)
                         D12_34 - x4 (NA mode only) 
      '''
      logging.debug(__name__ + ' : Getting the window configuration.')
      return str(self._visainstrument.ask('DISPlay:WINDow:SPLit?'))

 
    def select_trace(self,trace):
        '''
        Select (make active) the current trace.
        See also set multitrace option. You cannot select a trace that isn't on the screen.
        Used in modes: CAT and NA.
        
        Input:
            in Network Analyzer (NA) mode channel (int): 1 - 4
            in CAble and Antenna (CAT) test mode channel (int): 1, 2
            
        Output:
            True / False.
        '''
        logging.debug(__name__ + ' : selecting trace #%u' % trace)
        self._visainstrument.write('CALC:PAR%u:SEL' % trace)

    def clear_average_count(self):
        '''
        Resets the sweep average count to zero so that the next sweep
        performed will be back to AVG 1.  Does NOT trigger the sweep.
        Used in modes: CAT and NA.

        Input:
            None

        Output:
            None
        '''
        logging.debug(__name__ + ' : Clear averages.')
        self._visainstrument.write('AVER:CLE')
        
    def trigger_average_count_times(self, count=None, autoscale_after_first_sweep=False):
        '''
        Calls single() average_count (or count, if specified) times.

        Input:
            None

        Output:
            None
        '''

        t0 = time.time()
        t1 = t0
        for i in range(self.get_average_count() if count==None else count):
          # measure the time it took to complete the loop
          t0 = t1
          t1 = time.time()
          
          self.single(block_until_done = False)

          if i>0:
            # sleep until the sweep is almost done
            sleeping = np.max(( 0., (t1 - t0) - 10. ))
            logging.debug('Sleeping %.2e seconds before asking for OPC.' % sleeping)
            qt.msleep(sleeping)

          self.block_until_operation_complete()

          if i==0 and autoscale_after_first_sweep:
            for j in range(4): self.autoscale(1+j)
          

    def autoscale(self,trace):
        '''
        Set autoscale on the spcified channel.
        Used in modes: NA.

        Input:
            channel (int): channel from 1 - 4.

        Output:
            None
        '''
        logging.debug(__name__ + 'Setting autoscale on trace %u.' % trace)
        self._visainstrument.write('DISP:WIND:TRAC%u:Y:AUTO' % trace)

    def single(self, block_until_done=True):
        '''
        Perform a single sweep, then hold.  This is only valid when in
        single sweep mode. The command is ignored if in continuous
        mode.
        
        Used in modes: all modes.

        Input:
           None

        Output:
            None
        '''
        logging.debug(__name__ + 'Do a single sweep and %swait for the command to be completed.' % ('' if block_until_done else 'DO NOT '))
        self._visainstrument.write('INIT:IMM')
        qt.msleep(5.)
        if block_until_done: self.block_until_operation_complete()

    def block_until_operation_complete(self):
        '''
        Send *OPC? to the device and block until 1 is received. Used to determine whether a (single) sweep is complete.
        
        Used in modes: all modes.

        Input:
           None

        Output:
            None
        '''
        while True:
          try:
            if int(self._visainstrument.ask('*OPC?')) == 1: break
          except VisaIOError as e:
            logging.debug(__name__ + ' operation complete query taking longer than the visa timeout: %s'  % (str(e)))
            self._visainstrument.timeout = 30.  # This sometimes gets reset, not sure where...
            qt.msleep(0.1) # allow interruption

    def get_data(self,trace):
        '''
        Get unformatted measured data from a given channel (1 - 4)
.
        Used in modes: Network Analyzer (NA).

        Input:
           trace (int): select the trace (1-4) from which to get the data.

        Output:
            None
        '''
        logging.debug(__name__ + ' : Get trace %d data.' % trace)
        
        self._visainstrument.write('FORM:DATA REAL,32')
        self.select_trace(trace)
        data = self._visainstrument.query_binary_values('CALC:DATA:SDATA?',
                                                        datatype='f',
                                                        is_big_endian=False,
                                                        container=np.array,
                                                        header_fmt='ieee')
        return np.array([ r + 1j*i for r,i in data.reshape((-1,2)) ])

    def get_frequency_data(self):
        '''
        Get get the current x-axis frequency values.
        Used in modes: Network Analyzer (NA).

        Input:
            None

        Output:
            None
        '''
        logging.debug(__name__ + 'Get the current x-axis values.')
        
        self._visainstrument.write('FORM:DATA REAL,32')
        return self._visainstrument.query_binary_values('FREQ:DATA?',
                                                        datatype='f',
                                                        is_big_endian=False,
                                                        container=np.array,
                                                        header_fmt='ieee')

###  begin the parameter functions--------------    
    def do_get_sweep_mode(self):
        '''
        Query whether the instrument is in continuous mode (1) or single sweep mode (0). 
        Used in modes: all modes.

        Input:
            none.

        Output:
            mode (string):
              0: single sweep mode
              1: continuous mode         
        '''
        logging.debug(__name__ + 'Getting the trigger mode (cont. vs. single).')
        outp = self._visainstrument.ask('INIT:CONT?')
        if (outp=='0'):
            return 'single sweep mode'
        elif (outp=='1'):
            return 'continuous mode'
        
    def do_set_sweep_mode(self, mode):
        '''
        Set the sweep mode. 1 = continuous, 0 = single sweep.
        Used in modes: NA.

        Input:
            mode (int): sweep mode.
        Output:
            None
        '''
        logging.debug(__name__ + 'Setting the sweep mode to state %s' % mode)
        self._visainstrument.write('INIT:CONT %s' % mode)

        
    def do_get_velocity_factor(self):
        '''
        Get the velocity factor.
        Used in modes: NA.

        Input:
            None

        Output:
            velocity factor (float)
        '''
        logging.debug(__name__ + 'Getting the velocity factor.')
        return float(self._visainstrument.ask('CORR:RVEL:COAX?'))

    def do_set_velocity_factor(self, factor):
        '''
        Set the velocity factor.
        Used in modes: NA.

        Input:
            factor (float): the center frequency in Hz. 

        Output:
            None
        '''
        logging.debug(__name__ + 'Setting the center frequency to %f. Hz' % freq)
        self._visainstrument.write('CORR:RVEL:COAX %f' % freq)
        

    def do_get_frequency_center(self):
        '''
        Get the center frequency.
        Used in modes: NA.

        Input:
            None

        Output:
            Center frequency in Hz.
        '''
        logging.debug(__name__ + 'Getting the center frequency.')
        return str(self._visainstrument.ask('FREQ:CENT?'))

    def do_set_frequency_center(self,freq):
        '''
        Set the center frequency.
        Used in modes: NA.

        Input:
            freq (float): the center frequency in Hz. 

        Output:
            None
        '''
        logging.debug(__name__ + 'Setting the center frequency to %f. Hz' % freq)
        self._visainstrument.write('FREQ:CENT %f' % freq)

        
    def do_get_source_power(self):
        '''
        Get the source power.
        Used in modes: NA.

        Input:
            None

        Output:
            Center frequency in Hz.
        '''
        logging.debug(__name__ + ' : Getting the source power.')
        return float(self._visainstrument.ask(':SOURce:POWer?'))
        
    def do_set_source_power(self,pow):
        '''
        Set the source power in 0.1 dB steps.
        Used in modes: NA.

        NOTE: Output power is NOT flat across the frequency range. To set flat source power, 
        use SOURce:POWer:ALC[:MODE].
        
        Input:
            freq (float): the center frequency in Hz. 

        Output:
            None
        '''
        logging.debug(__name__ + ' : Setting the source power manually to %f dBm.' % pow)
        self._visainstrument.write('SOUR:POW %f' % pow)
   
        
        
        
    def do_get_IF_bandwidth(self):
        '''
        Get the IF bandwidth.
        Used in modes: NA.

        Input:
            None

        Output:
            IF bandwidth for the measurement.
        '''
        logging.debug(__name__ + 'Getting the center frequency.')
        return str(self._visainstrument.ask('SENS:BWID?'))

    def do_set_IF_bandwidth(self,freq):
        '''
        Set the IF bandwidth.
        Used in modes: NA.

        Input:
            freq (float): the center frequency in Hz. 

        Output:
            None
        '''
        logging.debug(__name__ + 'Setting the center frequency to %f. Hz' % freq)
        self._visainstrument.write('SENS:BWID %f' % freq)

    def do_get_points(self):
        '''
        Get the number of data points.
        Used in modes: NA.

        Input:
            None

        Output:
            # of points.
        '''
        logging.debug(__name__ + 'Getting the number of points.')
        return int(self._visainstrument.ask('SWE:POIN?'))

    def do_set_points(self,points):
        '''
        Set the # of points.
        Used in modes: NA.

        Input:
            points (int): # of points.

        Output:
            None
        '''
        logging.debug(__name__ + 'Setting the number of points to %u.' % points)
        self._visainstrument.write('SWE:POIN %u' % points)

    def do_get_frequency_start(self):
        '''
        Get the start frequency.
        Used in modes: NA.

        Input:
            None

        Output:
            Center frequency in Hz.
        '''
        logging.debug(__name__ + 'Getting the start frequency.')
        return str(self._visainstrument.ask('FREQ:STAR?'))

    def do_set_frequency_start(self,freq):
        '''
        Set the start frequency.
        Used in modes: NA.

        Input:
            freq (float): the center frequency in Hz. 

        Output:
            None
        '''
        logging.debug(__name__ + 'Setting the start frequency to %f. Hz' % freq)
        self._visainstrument.write('FREQ:STAR %f' % freq)
        
    def do_get_frequency_stop(self):
        '''
        Get the stop frequency.
        Used in modes: NA.

        Input:
            None

        Output:
            Stop frequency in Hz.
        '''
        logging.debug(__name__ + 'Getting the stop frequency.')
        return str(self._visainstrument.ask('FREQ:STOP?'))

    def do_set_frequency_stop(self,freq):
        '''
        Set the stop frequency.
        Used in modes: NA.

        Input:
            freq (float): the center frequency in Hz. 

        Output:
            None
        '''
        logging.debug(__name__ + 'Setting the stop frequency to %f. Hz' % freq)
        self._visainstrument.write('FREQ:STOP %f' % freq)

    def do_get_frequency_span(self):
        '''
        Get the frequency span.
        Used in modes: NA.

        Input:
            None

        Output:
            Span frequency in Hz.
        '''
        logging.debug(__name__ + 'Getting the center frequency.')
        return str(self._visainstrument.ask('FREQ:SPAN?'))


    def do_set_frequency_span(self,freq):
        '''
        Set the frequency span.  Default is the maximum minus minimum frequency range of the FieldFox.
        Used in modes: NA.

        Input:
            freq (float): the center frequency in Hz. 

        Output:
            None
        '''
        logging.debug(__name__ + 'Setting the stop frequency to %f. Hz' % freq)
        self._visainstrument.write('FREQ:SPAN %f' % freq)

    def get_number_of_traces(self):
        '''
        Return the number of traces currently enabled.
        '''
        return {'D1': 1, 'D2': 2, 'D12H': 2, 'D1123': 3, 'D12_34': 4, 'D1234': 4}[self.get_multi_trace()]

    def get_s_parameters_on_screen(self, sparams=None):
        '''
        Return all traces on screen as {'S11': np.array(...), 'S21': ...}
        If sparams == None, all traces on screen will be returned. Otherwise, only the specified ones (e.g. 'S11').
        '''
        all_traces = OrderedDict()
        for i in range(1,1 + self.get_number_of_traces()):
          par = self.get('ch%d_current_measurement' % i)
          if sparams == None or par in sparams: all_traces[par] = self.get_data(i)

        if sparams != None and len(all_traces) != len(sparams):
          logging.warn('Fewer traces returned (%s) than requested (%s).', all_traces.keys(), sparams)

        return all_traces

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
        self.get_average_count()
        self.get_average_mode()  
        self.get_operating_mode()
        self.get_frequency_center()
        self.get_frequency_start()
        self.get_frequency_span()
        self.get_frequency_stop()
        self.get_points()
        self.get_IF_bandwidth()
        self.get_sweep_mode()
        self.get_velocity_factor()
        self.get_source_power()
        self.get_source_power_mode()

        # Get the measurement and smoothing parameters for the windows on the screen.
        for i in range(1,1 + self.get_number_of_traces()):
          self.get('ch%d_smoothing_mode' % i)
          self.get('ch%d_current_measurement' % i)
                   
    def do_get_operating_mode(self):
        '''
        Reads the averaging of the signal from the instrument

        Input:
            None

        Output:
            len (int) : averaging length
        '''
        logging.debug(__name__ + ' : get operating mode.')
        return str(self._visainstrument.ask('INST:SEL?'))

    def do_set_operating_mode(self, val):
        '''
        Set the operating mode.  You can choose from the results of ':INST:CAT?'.

        Input:
            len (int) : operating mode. see choices in context help or ':INST:CAT?'.

        Output:
            None
        '''
        logging.debug(__name__ + ' : set operating mode to %s' % val)
        self._visainstrument.write('INST "%s";*OPC?' % val)

    def do_get_average_count(self):
        '''
        Reads the averaging of the signal from the instrument

        Input:
            None

        Output:
            len (int) : averaging length
        '''
        logging.debug(__name__ + ' : get averaging')
        return float(self._visainstrument.ask('SENS:AVER:COUN?'))

    def do_set_average_count(self, amp):
        '''
        Set the averaging of the signal (relavant to ALL modes).

        Input:
            len (int) : averaging length

        Output:
            None
        '''
        logging.debug(__name__ + ' : set averaging to %u' % amp)
        self._visainstrument.write('SENS:AVER:COUN %u' % amp)

    def do_get_average_mode(self):
        '''
        Gets the average mode (POINT or SWEEP) within NA mode.

        Input:
            None

        Output:
            len (int) : averaging length
        '''
        logging.debug(__name__ + ' : get averaging mode')
        return str(self._visainstrument.ask(':SENS:AVER:MODE?'))

    def do_set_average_mode(self, val):
        '''
        Set the averaging mode within Network Analyzer (NA) mode.

        Input:
            len (int) : averaging length

        Output:
            None
        '''
        logging.debug(__name__ + ' : set averaging mode to %f' % val)
        self._visainstrument.write('SENS:AVER:MODE %s' % val)

        
    def do_get_source_power_mode(self):
        '''
        Gets the source power mode.

        Input:
            None

        Output:
            len (int) : averaging length
        '''
        logging.debug(__name__ + ' : getting source power mode')
        return str(self._visainstrument.ask('SOUR:POW:ALC?'))

    def do_set_source_power_mode(self, val):
        '''
        Set the averaging mode within Network Analyzer (NA) mode.

        Input:
            len (int) : averaging lengths

        Output:
            None
        '''
        logging.debug(__name__ + ' : set averaging mode to %s' % val)
        self._visainstrument.write('SOUR:POW:ALC %s' % val)
        
        
    def do_get_current_measurement(self,channel):
        '''
        Query the current measurement.
        
        need split window if asking about 1,2,3,
        
        Input:
            None

        Output:
            len (int) : averaging length
        '''
        logging.debug(__name__ + ' : get current measurement mode')
        return str(self._visainstrument.ask(':CALC:PAR%u:DEF?' % channel))

    def do_set_current_measurement(self, mode, channel):
        '''
        Set the current measurement.  Options depend on which mode (CAT, NA, VVM) is currently selected.
        If param/chan is 2,3,4 then an appropriate multitrace configuration must be created first.
        For NA Mode:

        Reverse measurements are available ONLY with full S-parameter option.
        S11 - Forward reflection measurement
        S21 - Forward transmission measurement
        S12 - Reverse transmission
        S22 - Reverse reflection
        A - A receiver measurement
        B - B receiver measurement
        R1 - Port 1 reference receiver measurement
        R2 - Port 2 reference receiver measurement

        Input:
            len (int) : 
                channel (int): 1,2,3,4
                mode (str): see format map in context help.
        Output:
            None
        '''
        logging.debug(__name__ + ' : set current measurment mode to %s' % mode)
        self.select_trace(channel)
        self._visainstrument.write(':CALC:PAR%u:DEF %s' % (channel, mode))

    def do_get_smoothing_mode(self,channel):
        '''
        Gets the smoothing mode (on or off).

        Input:
            None

        Output:
            len (int) : averaging length
        '''
        logging.debug(__name__ + ' : get averaging mode')

        self.select_trace(channel)
        return str(self._visainstrument.ask('CALC:SMO?'))

    def do_set_smoothing_mode(self,state,channel):
        '''
        Sets the smoothing mode (on or off).

        Input:
            #channel (int): select a trace to toggle the smoothing.
            state (boolean): 
        Output:
            len (int) : averaging length
        '''
        logging.debug(__name__ + ' : get averaging mode')
        
        self.select_trace(channel)
        self._visainstrument.write('CALC:SMO %s' % state)

    def do_set_average_mode(self, val):
        '''
        Set the averaging mode within Network Analyzer (NA) mode.

        Input:
            len (int) : averaging length

        Output:
            None
        '''
        logging.debug(__name__ + ' : set averaging mode to %s' % val)
        self._visainstrument.write('SENS:AVER:MODE %s' % val)







