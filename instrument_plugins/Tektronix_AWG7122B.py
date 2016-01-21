# Tektronix_AWG7122B.py class, to perform the communication between the Wrapper and the device
# Joonas Govenius <joonas.govenius@aalto.fi>, 2013
#
# Based on the AWG520 driver by:
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
import numpy as np
import struct

class Tektronix_AWG7122B(Instrument):
    '''
    This is the python driver for the Tektronix AWG7122B
    Arbitrary Waveform Generator

    Usage:
    Initialize with
    <name> = instruments.create('name', 'Tektronix_AWG7122B', address='<GPIB address>',
        reset=<bool>)

    
		############################################################
		# Example script that generates a 10 MHz sine with 64 points
		############################################################
		
		import qt
		import numpy as np

		awg1 = qt.instruments['awg1']

		freq = 10e6
		pts_per_wf = 64

		wf1 = np.sin(2*np.pi*np.arange(pts_per_wf,dtype=np.int)/float(pts_per_wf)) # RF sent to the sample
		awg1.send_waveform(wf1, 'ch1_wf')    # The waveform name here is arbitrary
		awg1.set_ch1_waveform_name('ch1_wf')

		awg1.set_ch1_amplitude(0.5)

		awg1.set_clock(freq * pts_per_wf)

		awg1.set_ch1_status('on')
		awg1.set_trigger_mode('cont')
		awg1.set_run_state('run')
		
		
    '''

    def __init__(self, name, address, reset=False):
        '''
        Initializes the AWG7122B.

        Input:
            name (string)    : name of the instrument
            address (string) : GPIB address
            reset (bool)     : resets to default values, default=false

        Output:
            None
        '''
        logging.debug(__name__ + ' : Initializing instrument')
        Instrument.__init__(self, name, tags=['physical'])


        self._address = address
#>>>>>>>>>>>>>>
        assert False, "pyvisa syntax has changed, tweak the line below according to the instructions in qtlab/instrument_plugins/README_PYVISA_API_CHANGES"
        #self._visainstrument = visa.instrument(self._address)
#<<<<<<<<<<<<<<
        self._values = {}
        self._values['files'] = {}

        # Add parameters
        self.add_parameter('run_state', type=types.StringType,
            flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET,
            format_map={'run': 'running',
                        'stop': 'stopped',
                        'arm': 'waiting for trigger'})
        self.add_parameter('trigger_mode', type=types.StringType,
            flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET,
            format_map={'cont': 'continuous',
                        'trig': 'triggered',
                        'gat': 'gated',
                        'seq': 'sequence'})
        self.add_parameter('trigger_impedance', type=types.FloatType,
            flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET,
            minval=49, maxval=2e3, units='Ohm')
        self.add_parameter('trigger_level', type=types.FloatType,
            flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET,
            minval=-5, maxval=5, units='Volts')
        self.add_parameter('clock', type=types.FloatType,
            flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET,
            minval=1e6, maxval=12e9, units='Hz')
        self.add_parameter('waveform_name', type=types.StringType,
            flags=Instrument.FLAG_SET, channels=(1, 2),
            channel_prefix='ch%d_')
        self.add_parameter('amplitude', type=types.FloatType,
            flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET,
            channels=(1, 2), minval=0.5, maxval=1.0, units='Volts', channel_prefix='ch%d_')
        #self.add_parameter('offset', type=types.FloatType,
        #    flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET,
        #    channels=(1, 2), minval=-2, maxval=2, units='Volts', channel_prefix='ch%d_')
        self.add_parameter('marker1_low', type=types.FloatType,
            flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET,
            channels=(1, 2), minval=-2, maxval=2, units='Volts', channel_prefix='ch%d_')
        self.add_parameter('marker1_high', type=types.FloatType,
            flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET,
            channels=(1, 2), minval=-2, maxval=2, units='Volts', channel_prefix='ch%d_')
        self.add_parameter('marker2_low', type=types.FloatType,
            flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET,
            channels=(1, 2), minval=-2, maxval=2, units='Volts', channel_prefix='ch%d_')
        self.add_parameter('marker2_high', type=types.FloatType,
            flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET,
            channels=(1, 2), minval=-2, maxval=2, units='Volts', channel_prefix='ch%d_')
        self.add_parameter('status', type=types.StringType,
            flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET,
            channels=(1, 2), channel_prefix='ch%d_',
            format_map={'on': 'output on',
                        'off': 'output off'})

        # Add functions
        self.add_function('reset')
        self.add_function('get_all')
        self.add_function('clear_waveforms')
        self.add_function('set_trigger_impedance_1e3')
        self.add_function('set_trigger_impedance_50')

        if reset:
            self.reset()
        else:
            self.get_all()

    # Functions
    def reset(self):
        '''
        Resets the instrument to default values

        Input:
            None

        Output:
            None
        '''
        logging.info(__name__ + ' : Resetting instrument')
        self._visainstrument.write('*RST')

    def get_all(self):
        '''
        Reads all implemented parameters from the instrument,
        and updates the wrapper.

        Input:
            None

        Output:
            None
        '''
        logging.info(__name__ + ' : Reading all data from instrument')

        self.get_run_state()
        self.get_trigger_mode()
        self.get_trigger_impedance()
        self.get_trigger_level()
        self.get_clock()

        for i in range(1,2):
            self.get('ch%d_amplitude' % i)
            #self.get('ch%d_offset' % i)
            self.get('ch%d_marker1_low' % i)
            self.get('ch%d_marker1_high' % i)
            self.get('ch%d_marker2_low' % i)
            self.get('ch%d_marker2_high' % i)
            self.get('ch%d_status' % i)

    def peak_to_peak_amplitude_to_dbm(self, amplitude):
      return 10.*np.log10( 1000. * (amplitude/(2*np.sqrt(2)))**2 / (50.) )

    def dbm_to_peak_to_peak_amplitude(self, dbm):
      return np.sqrt(10.**(dbm/10.) / 1000. * 50.) * (2*np.sqrt(2))

    def clear_waveforms(self):
        '''
        Clears the waveform on both channels.

        Input:
            None

        Output:
            None
        '''
        logging.debug(__name__ + ' : Clear waveforms from channels')
        self._visainstrument.write('SOURCE1:WAVEFORM ""')
        self._visainstrument.write('SOURCE2:WAVEFORM ""')

    def set_trigger_impedance_1e3(self):
        '''
        Sets the trigger impedance to 1 kOhm

        Input:
            None

        Output:
            None
        '''
        logging.debug(__name__  + ' : Set trigger impedance to 1e3 Ohm')
        self._visainstrument.write('TRIG:IMP 1e3')

    def set_trigger_impedance_50(self):
        '''
        Sets the trigger impedance to 50 Ohm

        Input:
            None

        Output:
            None
        '''
        logging.debug(__name__  + ' : Set trigger impedance to 50 Ohm')
        self._visainstrument.write('TRIG:IMP 50')

    # Parameters
    def do_get_trigger_mode(self):
        '''
        Reads the trigger mode from the instrument
        '''
        return self._visainstrument.ask('AWGC:RMOD?').lower().strip()

    def do_set_trigger_mode(self, mod):
        '''
        Sets trigger mode of the instrument
        '''
        self._visainstrument.write('AWGC:RMOD %s' % (mod.upper().strip()))

    def do_get_run_state(self):
        '''
        Reads the trigger mode from the instrument
        '''
        int_to_state = {0: 'stop', 1: 'arm', 2: 'run'}
        return int_to_state[int(self._visainstrument.ask('AWGC:RST?'))]

    def do_set_run_state(self, state):
        '''
        Sets trigger mode of the instrument. Must be 'run' or 'stop'.
        '''
        assert state != 'arm', '"arm" cannot be set here. Use trigger_mode and set this to "run".'
        if state == 'run':
          self._visainstrument.write('AWGC:RUN')
        elif  state == 'stop':
          self._visainstrument.write('AWGC:STOP')
        else:
          raise Exception('Unknown run state: %s' % state)

    def do_get_trigger_impedance(self):
        '''
        Reads the trigger impedance from the instrument

        Input:
            None

        Output:
            impedance (??) : 1e3 or 50 depending on the mode
        '''
        logging.debug(__name__  + ' : Get trigger impedance from instrument')
        return self._visainstrument.ask('TRIG:IMP?')

    def do_set_trigger_impedance(self, mod):
        '''
        Sets the trigger impedance of the instrument

        Input:
            mod (int) : Either 1e3 of 50 depending on the mode

        Output:
            None
        '''
        if (mod==1e3):
            self.set_trigger_impedance_1e3()
        elif (mod==50):
            self.set_trigger_impedance_50()
        else:
            logging.error(__name__ + ' : Unable to set trigger impedance to %s, expected "1e3" or "50"' %mod)

    def do_get_trigger_level(self):
        '''
        Reads the trigger level from the instrument

        Input:
            None

        Output:
            None
        '''
        logging.debug(__name__  + ' : Get trigger level from instrument')
        return float(self._visainstrument.ask('TRIG:LEV?'))

    def do_set_trigger_level(self, level):
        '''
        Sets the trigger level of the instrument

        Input:
            level (float) : trigger level in volts
        '''
        logging.debug(__name__  + ' : Trigger level set to %.3f' %level)
        self._visainstrument.write('TRIG:LEV %.3f' %level)


    def do_get_clock(self):
        '''
        Returns the clockfrequency, which is the rate at which the datapoints are
        sent to the designated output

        Input:
            None

        Output:
            clock (float) : frequency in Hz
        '''
        return float(self._visainstrument.ask('SOUR:FREQ?'))

    def do_set_clock(self, clock):
        '''
        Sets the rate at which the datapoints are sent to the designated output channel

        Input:
            clock (float) : frequency in Hz

        Output:
            None
        '''
        self._visainstrument.write('SOUR:FREQ %f' % clock)

    def do_set_waveform_name(self, name, channel):
        '''
        Specifies which file has to be set on which channel
        Make sure the file exists, and the numpoints and clock of the file
        matches the instrument settings.

        If file doesn't exist an error is raised, if the numpoints doesn't match
        the command is neglected

        Input:
            name (string) : name of the waveform to be set for the channel
            channel (int) : 1 or 2, the number of the designated channel

        Output:
            None
        '''
        # check that the waveform exists
        try:
          wf_len = self._visainstrument.ask('WLIST:WAVEFORM:LENGTH? "%s"' % (name))
          if wf_len < 1: raise Exception('Empty waveform.')
        except:
          logging.warn('Waveform "%s" does not exist or is invalid.', name)
          raise
        self._visainstrument.write('SOURCE%d:WAVEFORM "%s"' % (channel, name))

    def do_get_amplitude(self, channel):
        '''
        Reads the amplitude of the designated channel from the instrument

        Input:
            channel (int) : 1 or 2, the number of the designated channel

        Output:
            amplitude (float) : the amplitude of the signal in Volts
        '''
        logging.debug(__name__ + ' : Get amplitude of channel %s from instrument'
            %channel)
        return float(self._visainstrument.ask('SOUR%s:VOLT:LEV:IMM:AMPL?' % channel))

    def do_set_amplitude(self, amp, channel):
        '''
        Sets the amplitude of the designated channel of the instrument

        Input:
            amp (float)   : amplitude in Volts
            channel (int) : 1 or 2, the number of the designated channel

        Output:
            None
        '''
        logging.debug(__name__ + ' : Set amplitude of channel %s to %.6f'
            %(channel, amp))
        self._visainstrument.write('SOUR%s:VOLT:LEV:IMM:AMPL %.6f' % (channel, amp))

    # def do_get_offset(self, channel):
        # '''
        # Reads the offset of the designated channel of the instrument

        # Input:
            # channel (int) : 1 or 2, the number of the designated channel

        # Output:
            # offset (float) : offset of designated channel in Volts
        # '''
        # logging.debug(__name__ + ' : Get offset of channel %s' %channel)
        # return float(self._visainstrument.ask('SOUR%s:VOLT:LEV:IMM:OFFS?' % channel))

    # def do_set_offset(self, offset, channel):
        # '''
        # Sets the offset of the designated channel of the instrument

        # Input:
            # offset (float) : offset in Volts
            # channel (int)  : 1 or 2, the number of the designated channel

        # Output:
            # None
        # '''
        # logging.debug(__name__ + ' : Set offset of channel %s to %.6f' %(channel, offset))
        # self._visainstrument.write('SOUR%s:VOLT:LEV:IMM:OFFS %.6f' % (channel, offset))

    def do_get_marker1_low(self, channel):
        '''
        Gets the low level for marker1 on the designated channel.

        Input:
            channel (int) : 1 or 2, the number of the designated channel

        Output:
            low (float) : low level in Volts
        '''
        logging.debug(__name__ + ' : Get lower bound of marker1 of channel %s' %channel)
        return float(self._visainstrument.ask('SOUR%s:MARK1:VOLT:LEV:IMM:LOW?' % channel))

    def do_set_marker1_low(self, low, channel):
        '''
        Sets the low level for marker1 on the designated channel.

        Input:
            low (float)   : low level in Volts
            channel (int) : 1 or 2, the number of the designated channel

        Output:
            None
         '''
        logging.debug(__name__ + ' : Set lower bound of marker1 of channel %s to %.3f'
            %(channel, low))
        self._visainstrument.write('SOUR%s:MARK1:VOLT:LEV:IMM:LOW %.3f' % (channel, low))

    def do_get_marker1_high(self, channel):
        '''
        Gets the high level for marker1 on the designated channel.

        Input:
            channel (int) : 1 or 2, the number of the designated channel

        Output:
            high (float) : high level in Volts
        '''
        logging.debug(__name__ + ' : Get upper bound of marker1 of channel %s' %channel)
        return float(self._visainstrument.ask('SOUR%s:MARK1:VOLT:LEV:IMM:HIGH?' % channel))

    def do_set_marker1_high(self, high, channel):
        '''
        Sets the high level for marker1 on the designated channel.

        Input:
            high (float)   : high level in Volts
            channel (int) : 1 or 2, the number of the designated channel

        Output:
            None
         '''
        logging.debug(__name__ + ' : Set upper bound of marker1 of channel %s to %.3f'
            %(channel,high))
        self._visainstrument.write('SOUR%s:MARK1:VOLT:LEV:IMM:HIGH %.3f' % (channel, high))

    def do_get_marker2_low(self, channel):
        '''
        Gets the low level for marker2 on the designated channel.

        Input:
            channel (int) : 1 or 2, the number of the designated channel

        Output:
            low (float) : low level in Volts
        '''
        logging.debug(__name__ + ' : Get lower bound of marker2 of channel %s' %channel)
        return float(self._visainstrument.ask('SOUR%s:MARK2:VOLT:LEV:IMM:LOW?' % channel))

    def do_set_marker2_low(self, low, channel):
        '''
        Sets the low level for marker2 on the designated channel.

        Input:
            low (float)   : low level in Volts
            channel (int) : 1 or 2, the number of the designated channel

        Output:
            None
         '''
        logging.debug(__name__ + ' : Set lower bound of marker2 of channel %s to %.3f'
            %(channel, low))
        self._visainstrument.write('SOUR%s:MARK2:VOLT:LEV:IMM:LOW %.3f' % (channel, low))

    def do_get_marker2_high(self, channel):
        '''
        Gets the high level for marker2 on the designated channel.

        Input:
            channel (int) : 1 or 2, the number of the designated channel

        Output:
            high (float) : high level in Volts
        '''
        logging.debug(__name__ + ' : Get upper bound of marker2 of channel %s' %channel)
        return float(self._visainstrument.ask('SOUR%s:MARK2:VOLT:LEV:IMM:HIGH?' % channel))

    def do_set_marker2_high(self, high, channel):
        '''
        Sets the high level for marker2 on the designated channel.

        Input:
            high (float)   : high level in Volts
            channel (int) : 1 or 2, the number of the designated channel

        Output:
            None
         '''
        logging.debug(__name__ + ' : Set upper bound of marker2 of channel %s to %.3f'
            %(channel,high))
        self._visainstrument.write('SOUR%s:MARK2:VOLT:LEV:IMM:HIGH %.3f' % (channel, high))

    def do_get_status(self, channel):
        '''
        Gets the status of the designated channel.

        Input:
            channel (int) : 1 or 2, the number of the designated channel

        Output:
            None
        '''
        logging.debug(__name__ + ' : Get status of channel %s' %channel)
        outp = self._visainstrument.ask('OUTP%s?' %channel)
        if (outp=='0'):
            return 'off'
        elif (outp=='1'):
            return 'on'
        else:
            logging.debug(__name__ + ' : Read invalid status from instrument %s' %outp)
            return 'an error occurred while reading status from instrument'

    def do_set_status(self, status, channel):
        '''
        Sets the status of designated channel.

        Input:
            status (string) : 'On' or 'Off'
            channel (int)   : channel number

        Output:
            None
        '''
        logging.debug(__name__ + ' : Set status of channel %s to %s'
            %(channel, status))
        if (status.upper()=='ON'):
            self._visainstrument.write('OUTP%s ON' %channel)
        elif (status.upper()=='OFF'):
            self._visainstrument.write('OUTP%s OFF' %channel)
        else:
            logging.debug(__name__ + ' : Try to set status to invalid value %s' % status)
            print 'Tried to set status to invalid value %s' %status

    #  Ask for string with filenames
    def get_filenames(self):
        logging.debug(__name__ + ' : Read filenames from instrument')
        return self._visainstrument.ask('MMEM:CAT? "MAIN"')

    # Send waveform to the device
    def send_waveform(self,w,waveform_name):
        '''
        Sends a complete waveform. All parameters need to be specified.
        See also: resend_waveform()

        Input:
            w (float[numpoints]) : waveform
            waveform_name (string)    : waveform_name

        Output:
            None
        '''
        logging.debug(__name__ + ' : Sending waveform %s to instrument' % waveform_name)
        
        if len(w) < 1: raise exception('Empty waveform.')
        if np.abs(w).max() > 1.0: raise exception('Waveform values must be between \pm 1.0 (inclusive).')

        wf_header = 'WLIST:WAVEFORM:DATA "%s",0,%d,#%d%d' % (waveform_name, len(w), len(str(2*len(w))), 2*len(w))
        wf_data = bytearray(2*len(w))
        for i in range(len(w)):
          struct.pack_into('<H', wf_data, 2*i, int(np.round((1+w[i]) * (2**13 - 1))))

        self._visainstrument.write('WLIST:WAVEFORM:DELETE "%s"' % waveform_name)
        self._visainstrument.write('WLIST:WAVEFORM:NEW "%s",%d,INTEGER' % (waveform_name,len(w)))
        self._visainstrument.write('%s%s' % (wf_header,str(wf_data)))
