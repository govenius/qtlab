# Agilent_E4426B.py class, to perform the communication between the Wrapper and the device
# Pieter de Groot <pieterdegroot@gmail.com>, 2008
# Martijn Schaafsma <qtlab@mcschaafsma.nl>, 2008
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
import types
import logging
import numpy

class Agilent_E4426B(Instrument):
    '''
    This is the driver for the Agilent E4426B Signal Genarator

    Usage:
    Initialize with
    <name> = instruments.create('<name>', 'Agilent_E4426B', address='<GBIP address>, reset=<bool>')
    '''

    def __init__(self, name, address, reset=False):
        '''
        Initializes the Agilent_E4426B, and communicates with the wrapper.

        Input:
          name (string)    : name of the instrument
          address (string) : GPIB address
          reset (bool)     : resets to default values, default=False
        '''
        logging.info(__name__ + ' : Initializing instrument Agilent_E4426B')
        Instrument.__init__(self, name, tags=['physical'])

        # Add some global constants
        self._address = address
        self._visainstrument = visa.ResourceManager().open_resource(self._address)
        self._visainstrument.read_termination = '\n'
        self._visainstrument.write_termination = '\n'

        self.add_parameter('power',
            flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET, units='dBm', minval=-135, maxval=20, type=types.FloatType)
        self.add_parameter('phase',
            flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET, units='rad', minval=-numpy.pi, maxval=numpy.pi, type=types.FloatType)
        self.add_parameter('frequency', format='%.09e',
            flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET, units='Hz', minval=1e5, maxval=4e9, type=types.FloatType)
        self.add_parameter('status',
            flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET, type=types.StringType,
            format_map={'on': 'output on',
                        'off': 'output off'})
        #self.add_parameter('sweep_time',
        #    flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET, units='s', minval=0.01, maxval=99., type=types.FloatType)
        #self.add_parameter('frequency_center',
        #    flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET, units='Hz', minval=30e3, maxval=20e9, type=types.FloatType)
        #self.add_parameter('frequency_start',
        #    flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET, units='Hz', minval=30e3, maxval=20e9, type=types.FloatType)
        #self.add_parameter('frequency_stop',
        #    flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET, units='Hz', minval=30e3, maxval=20e9, type=types.FloatType)
        #self.add_parameter('frequency_span',
        #    flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET, units='Hz', minval=0, maxval=10e9, type=types.FloatType)
        #self.add_parameter('frequency_step',
        #    flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET, units='Hz', minval=0.01, maxval=99e9, type=types.FloatType)
        
        self.add_parameter('trigger_source',
            flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET, type=types.StringType)
        

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
        self.get_power()
        self.get_phase()
        self.get_frequency()
        self.get_status()
        #self.get_sweep_time()
        #self.get_frequency_center()
        #self.get_frequency_start()
        #self.get_frequency_stop()
        #self.get_frequency_span()
        #self.get_frequency_step()
        self.get_trigger_source()
        

    def do_get_power(self):
        '''
        Reads the power of the signal from the instrument

        Input:
            None

        Output:
            ampl (?) : power in ?
        '''
        logging.debug(__name__ + ' : get power')
        return float(self._visainstrument.ask('POW:AMPL?'))

    def do_set_power(self, amp):
        '''
        Set the power of the signal

        Input:
            amp (float) : power in ??

        Output:
            None
        '''
        logging.debug(__name__ + ' : set power to %f' % amp)
        self._visainstrument.write('POW:AMPL %s' % amp)


    def do_get_phase(self):
        '''
        Reads the phase of the signal from the instrument

        Input:
            None

        Output:
            phase (float) : Phase in radians
        '''
        logging.debug(__name__ + ' : get phase')
        return float(self._visainstrument.ask('PHASE?'))

    def do_set_phase(self, phase):
        '''
        Set the phase of the signal

        Input:
            phase (float) : Phase in radians

        Output:
            None
        '''
        logging.debug(__name__ + ' : set phase to %f' % phase)
        self._visainstrument.write('PHASE %s' % phase)

    def do_get_frequency(self):
        '''
        Reads the frequency of the signal from the instrument

        Input:
            None

        Output:
            freq (float) : Frequency in Hz
        '''
        logging.debug(__name__ + ' : get frequency')
        return float(self._visainstrument.ask('FREQ:CW?'))

    def do_set_frequency(self, freq):
        '''
        Set the frequency of the instrument

        Input:
            freq (float) : Frequency in Hz

        Output:
            None
        '''
        logging.debug(__name__ + ' : set frequency to %f' % freq)
        rounded = '%.3f' % numpy.round(freq,decimals=3)
        if numpy.abs(float(rounded) - freq) > numpy.finfo(numpy.float).tiny:
          logging.warn('Rounding the requested frequency (%.15e Hz) to %s Hz (i.e. by %.6e Hz).' % (freq, rounded, float(rounded) - freq))
        self._visainstrument.write('FREQ:CW %s' % rounded)

    
    
    # def do_get_frequency_center(self):
        # '''
        # Reads the frequency center point for a step sweep or ramp sweep of the instrument

        # Input:
            # None

        # Output:
            # freq (float) : Frequency in Hz
        # '''
        # logging.debug(__name__ + ' : get frequency_center')
        # return float(self._visainstrument.ask('FREQ:CENT?'))

    # def do_set_frequency_center(self, freq):
        # '''
        # Set the frequency center point for a step sweep or ramp sweep of the instrument

        # Input:
            # freq (float) : Frequency in Hz

        # Output:
            # None
        # '''
        # logging.debug(__name__ + ' : set frequency_center to %f' % freq)
        # self._visainstrument.write('FREQ:CENT %s' % freq)
        
    # def do_get_frequency_start(self):
        # '''
        # Reads the frequency start point for a step sweep or ramp sweep of the instrument

        # Input:
            # None

        # Output:
            # freq (float) : Frequency in Hz
        # '''
        # logging.debug(__name__ + ' : get frequency_start')
        # return float(self._visainstrument.ask('FREQ:STAR?'))

    # def do_set_frequency_start(self, freq):
        # '''
        # Set the frequency start point for a step sweep or ramp sweep of the instrument

        # Input:
            # freq (float) : Frequency in Hz

        # Output:
            # None
        # '''
        # logging.debug(__name__ + ' : set frequency_start to %f' % freq)
        # self._visainstrument.write('FREQ:STAR %s' % freq)
        
    # def do_get_frequency_stop(self):
        # '''
        # Reads the frequency stop point for a step sweep or ramp sweep of the signal from the instrument

        # Input:
            # None

        # Output:
            # freq (float) : Frequency in Hz
        # '''
        # logging.debug(__name__ + ' : get frequency_stop')
        # return float(self._visainstrument.ask('FREQ:STOP?'))

    # def do_set_frequency_stop(self, freq):
        # '''
        # Set the frequency stop point for a step sweep or ramp sweep of the instrument

        # Input:
            # freq (float) : Frequency in Hz

        # Output:
            # None
        # '''
        # logging.debug(__name__ + ' : set frequency_stop to %f' % freq)
        # self._visainstrument.write('FREQ:STOP %s' % freq)
        
    # def do_get_frequency_span(self):
        # '''
        # Reads the frequency span of the signal from the instrument

        # Input:
            # None

        # Output:
            # freq (float) : Frequency in Hz
        # '''
        # logging.debug(__name__ + ' : get frequency_span')
        # return float(self._visainstrument.ask('FREQ:SPAN?'))

    # def do_set_frequency_span(self, freq):
        # '''
        # Set the frequency span of the instrument

        # Input:
            # freq (float) : Frequency in Hz

        # Output:
            # None
        # '''
        # logging.debug(__name__ + ' : set frequency_span to %f' % freq)
        # self._visainstrument.write('FREQ:SPAN %s' % freq)
    
    # def do_get_frequency_step(self):
        # '''
        # Reads the frequency step of the signal from the instrument

        # Input:
            # None

        # Output:
            # freq (float) : Frequency in Hz
        # '''
        # logging.debug(__name__ + ' : get frequency_step')
        # return float(self._visainstrument.ask('FREQ:STEP?'))

    # def do_set_frequency_step(self, freq):
        # '''
        # Set the frequency step of of the signal from the instrument

        # Input:
            # freq (float) : Frequency in Hz

        # Output:
            # None
        # '''
        # logging.debug(__name__ + ' : set frequency_step to %f' % freq)
        # self._visainstrument.write('FREQ:STEP %s' % freq)
        
    def do_get_status(self):
        '''
        Reads the output status from the instrument

        Input:
            None

        Output:
            status (string) : 'On' or 'Off'
        '''
        logging.debug(__name__ + ' : get status')
        stat = self._visainstrument.ask('OUTP?')

        if (stat=='1'):
          return 'on'
        elif (stat=='0'):
          return 'off'
        else:
          raise ValueError('Output status not specified : %s' % stat)
        return

    def do_set_status(self, status):
        '''
        Set the output status of the instrument

        Input:
            status (string) : 'On' or 'Off'

        Output:
            None
        '''
        logging.debug(__name__ + ' : set status to %s' % status)
        if status.upper() in ('ON', 'OFF'):
            status = status.upper()
        else:
            raise ValueError('set_status(): can only set on or off')
        self._visainstrument.write('OUTP %s' % status)


    def do_get_sweep_time(self):
        '''
        Reads the power of the signal from the instrument

        Input:
            None

        Output:
            sweep time in seconds
        '''
        logging.debug(__name__ + ' : get sweep time')
        return float(self._visainstrument.ask(':SWEEP:TIME?'))

    def do_set_sweep_time(self, amp):
        '''
        Set the power of the signal

        Input:
            sweep time (float) : sweep time in seconds.

        Output:
            None
        '''
        logging.debug(__name__ + ' : set sweep time to %f' % amp)
        self._visainstrument.write(':SWEEP:TIME %s' % amp)





    # shortcuts
    def off(self):
        '''
        Set status to 'off'

        Input:
            None

        Output:
            None
        '''
        self.set_status('off')

    def on(self):
        '''
        Set status to 'on'

        Input:
            None

        Output:
            None
        '''
        self.set_status('on')
    
    def do_get_trigger_source(self):
        '''

        Input:
            None

        Output:
            source ['KEY', 'EXT', 'BUS']
        '''
        r = self._visainstrument.ask(':TRIG:SOUR?')
        logging.debug(__name__ + ' : get trigger_source: %s' % r)
        return r

    def do_set_trigger_source(self, val):
        '''

        Input:
            source ['KEY', 'EXT', 'BUS']

        Output:
            None
        '''
        if val not in ['IMM', 'EXT', 'BUS']:
          raise Exception('Invalid trigger source "%s". Should be "IMM", "EXT" or "BUS".')
        logging.debug(__name__ + ' : set trigger_source to %s' % val)
        self._visainstrument.write(':TRIG:SOUR %s' % val)
