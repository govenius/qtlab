# Gigatronics_2550B.py class, for commucation with a Gigatronics 2550B microwave source.
# Joonas Govenius <joonas.govenius@aalto.fi>, 2012
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
from lib import tcpinstrument
import visa
from visa import VisaIOError
import types
import logging
import numpy
import math
import time
import re


class Gigatronics_2550B(Instrument):
    '''
    This is the driver for the Gigatronics 2550B Signal Genarator

    Usage:
    Initialize with
    <name> = instruments.create('<name>', 'Gigatronics_2550B', address='<IPv4-address>:<TCP-port>, reset=<bool>')
    '''

    def __init__(self, name, address, reset=False):
        '''
        Initializes the Gigatronics_2550B, and communicates with the wrapper.

        Input:
          name (string)    : name of the instrument
          address (string) : GPIB address
          reset (bool)     : resets to default values, default=False
        '''
        logging.info(__name__ + ' : Initializing instrument Gigatronics_2550B')
        Instrument.__init__(self, name, tags=['physical'])

        self._address = address
        
        if re.match('(?i)gpib', address):
          logging.info('Initializing Gigatronics using VISA.')
          self._visainstrument = visa.ResourceManager().open_resource(self._address, timeout=2000)
          self._visainstrument.read_termination = '\n'
          self._visainstrument.write_termination = '\n'
        else:
          logging.info('Initializing Gigatronics using LAN.')
          self._visainstrument = tcpinstrument.TCPInstrument(address)

        try:

          self.add_parameter('idn', type=types.StringType, flags=Instrument.FLAG_GET, format='%.10s')

          self.add_parameter('power',
              flags=Instrument.FLAG_GETSET, units='dBm', minval=-135, maxval=25, type=types.FloatType)
          self.add_parameter('phase',
              flags=Instrument.FLAG_GETSET, units='rad', minval=-numpy.pi, maxval=numpy.pi, type=types.FloatType)
          self.add_parameter('frequency', format='%.09e',
              flags=Instrument.FLAG_GETSET, units='Hz', minval=1e5, maxval=50e9, type=types.FloatType) #, cache_time=1.)
          self.add_parameter('alc_source',
              flags=Instrument.FLAG_GETSET, type=types.StringType)
          self.add_parameter('trigger_source',
              flags=Instrument.FLAG_GETSET, type=types.StringType)
          self.add_parameter('pulse_modulation',
              flags=Instrument.FLAG_GETSET, type=types.BooleanType)
          self.add_parameter('pulse_modulation_source',
              flags=Instrument.FLAG_GETSET, type=types.StringType)
          self.add_parameter('pulse_modulation_inverted_polarity',
              flags=Instrument.FLAG_GETSET, type=types.BooleanType)
          self.add_parameter('power_correction_offset',
              flags=Instrument.FLAG_GETSET, units='dB', minval=-100., maxval=100., type=types.FloatType)
          self.add_parameter('power_correction_slope',
              flags=Instrument.FLAG_GETSET, units='dB/GHz', minval=0., maxval=0.5, type=types.FloatType)
          self.add_parameter('reference_clock_source',
              flags=Instrument.FLAG_GET, type=types.StringType)
          self.add_parameter('mode',
              flags=Instrument.FLAG_GET, type=types.StringType)
          self.add_parameter('status',
              flags=Instrument.FLAG_GETSET, type=types.StringType)

          self.add_function('reset')
          self.add_function ('get_all')

          if (reset):
              self.reset()
          else:
              self.get_all()

        except:
          self._visainstrument.close()
          raise


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
        self.get_idn()
        self.get_power()
        self.get_phase()
        self.get_frequency()
        self.get_status()
        self.get_alc_source()
        self.get_trigger_source()
        self.get_pulse_modulation()
        self.get_pulse_modulation_source()
        self.get_pulse_modulation_inverted_polarity()
        self.get_power_correction_offset()
        self.get_power_correction_slope()
        self.get_reference_clock_source()
        self.get_mode()

    def do_get_idn(self): return self._visainstrument.query('*IDN?')

    def do_get_mode(self):
        '''
        Get the source mode of operation.

        Input:
            None

        Output:
            string : "CW", "LIST", "FSWEEP", "PSWEEP"
        '''
        logging.debug(__name__ + ' : get source mode')
        m = self._visainstrument.ask('MODE?')
        if m=='FIX':
            return 'CW'
        else:
            return m

    def do_get_alc_source(self):
        '''
        Get the Automatic Load Control source

        Input:
            None

        Output:
            string : "INT", "DIOD", "PMET", "DPOS"
        '''
        logging.debug(__name__ + ' : get ALC source')
        s = self._visainstrument.ask('POW:ALC:SOUR?')
        if s[:3] == 'INT':
            return 'INT'
        else:
            return s[:4]

    def do_set_alc_source(self, val):
        '''
        Set the Automatic Load Control source
        
        Input:
            val : "INT", "DIOD", "PMET", "DPOS"

        Output:
            None
        '''
        logging.debug(__name__ + ' : set ALC source to %s' % val)
        self._visainstrument.write('POW:ALC:SOUR %s' % val)

    def do_get_trigger_source(self):
        '''
        Get the trigger source

        Input:
            None

        Output:
            string : "INT", "EXT", "NOT IN SWEEP MODE"
        '''
        logging.debug(__name__ + ' : get trigger source')
        s = self._visainstrument.ask('TRIG:SOUR?')
        if s[:3] == 'INT' or s[:3] == 'EXT':
            return s[:3]
        else:
            return s

    def do_set_trigger_source(self, val):
        '''
        Set the Automatic Load Control source
        
        Input:
            val : "INT", "EXT"

        Output:
            None
        '''
        logging.debug(__name__ + ' : set trigger source to %s' % val)
        self._visainstrument.write('TRIG:SOUR %s' % val)

    def do_get_pulse_modulation(self):
        '''
        Whether pulse modulation is on or off.
        
        Input:
            None

        Output:
            boolean
        '''
        logging.debug(__name__ + ' : get pulse modulation.')
        state = self._visainstrument.ask('PULM:STAT?')
        return state == '1' or state == 'ON'

    def do_set_pulse_modulation(self, val):
        '''
        Turn pulse modulation on/off.
        
        Input:
            val : boolean

        Output:
            None
        '''
        logging.debug(__name__ + ' : set pulse modulation to %s' % val)
        self._visainstrument.write('PULM:STAT %s' % str(int(val)))

    def do_get_pulse_modulation_inverted_polarity(self):
        '''
        Whether the external pulse modulation polarity is inverted or not.
        
        Input:
            None

        Output:
            boolean
        '''
        logging.debug(__name__ + ' : get pulse modulation inverted polarity')
        state = self._visainstrument.ask('PULM:EXT:POL?')
        return state[:3] == 'INV'

    def do_set_pulse_modulation_inverted_polarity(self, val):
        '''
        Turn inverting external pulse modulation polarity on/off.
        
        Input:
            val : boolean

        Output:
            None
        '''
        logging.debug(__name__ + ' : set  to %s' % val)
        self._visainstrument.write('PULM:EXT:POL %s' % ('INV' if val else 'NORM'))

    def do_get_pulse_modulation_source(self):
        '''
        Get the pulse modulation source.
        
        Input:
            None

        Output:
            string : "INT", "EXT"
        '''
        logging.debug(__name__ + ' : get pulse modulation source')
        return self._visainstrument.ask('PULM:SOUR?')[:3]

    def do_set_pulse_modulation_source(self, val):
        '''
        Set the pulse modulation source
        
        Input:
            val : "INT", "EXT"

        Output:
            None
        '''
        logging.debug(__name__ + ' : set  to %s' % val)
        self._visainstrument.write('PULM:SOUR %s' % val)

    def do_get_power_correction_offset(self):
        '''
        Get the offset power (loss correction).
                
        Input:
            None

        Output:
           offset power in dB
        '''
        logging.debug(__name__ + ' : get loss correction')
        return float(self._visainstrument.ask('CORR:LOSS?'))

    def do_set_power_correction_offset(self, val):
        '''
        Set the offset power (loss correction).

        Input:
            val (float) : offset power in dB

        Output:
            None
        '''
        logging.debug(__name__ + ' : set loss correction offset to %f dB' % val)
        self._visainstrument.write('CORR:LOSS %s' % val)


    def do_get_power_correction_slope(self):
        '''
        Get the power slope (loss correction).
                
        Input:
            None

        Output:
           slope dB/GHz
        '''
        logging.debug(__name__ + ' : get loss correction slope')
        return float(self._visainstrument.ask('CORR:SLOP?'))

    def do_set_power_correction_slope(self, val):
        '''
        Set the slope power (loss correction).

        Input:
            val (float) : slope power in dB/GHz

        Output:
            None
        '''
        logging.debug(__name__ + ' : set loss correction to %f dB/GHz' % val)
        self._visainstrument.write('CORR:SLOP %s' % val)


    def do_get_reference_clock_source(self):
        '''
        Get the reference clock source currently in use.
        
        Input:
            None

        Output:
            string: 'INT', 'EXT'
        '''
        logging.debug(__name__ + ' : get reference clock source.')
        return self._visainstrument.ask('ROSC:SOUR?')[:3]
        
    def do_get_power(self):
        '''
        Reads the power of the signal from the instrument

        Input:
            None

        Output:
            ampl (?) : power in dBm
        '''
        logging.debug(__name__ + ' : get power')
        return float(self._visainstrument.ask('POW:AMPL?'))

    def do_set_power(self, amp):
        '''
        Set the power of the signal

        Input:
            amp (float) : power in dBm

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

        rep = self._visainstrument.ask('PHASE?')
        m = re.match(r"^([\d\.]+) (RAD|DEG)", rep)
        if m == None or len(m.groups()) != 2: raise Exception('Failed to parse {0} in reply to PHASE?.'.format())

        phase = float(m.groups()[0])
        if m.groups()[1] == 'DEG': phase = phase * math.pi/180.
        return phase

    def do_set_phase(self, phase):
        '''
        Set the phase of the signal

        Input:
            phase (float) : Phase in radians

        Output:
            None
        '''
        logging.debug(__name__ + ' : set phase to %f' % phase)
        self._visainstrument.write('PHASE %s RAD' % phase)

    def do_get_frequency(self):
        '''
        Reads the frequency of the signal from the instrument

        Input:
            None

        Output:
            freq (float) : Frequency in Hz
        '''
        logging.debug(__name__ + ' : get frequency')
        f = float(self._visainstrument.ask('FREQ:CW?'))
        return f

    def do_set_frequency(self, freq):
        '''
        Set the frequency of the instrument

        Input:
            freq (float) : Frequency in Hz

        Output:
            None
        '''
        
        f0 = self.do_get_frequency()
        
        logging.debug(__name__ + ' : set frequency to %f' % freq)
        rounded = '%.3f' % numpy.round(freq,decimals=3)
        if numpy.abs(float(rounded) - freq) > numpy.finfo(numpy.float).tiny:
          logging.warn('Rounding the requested frequency (%.15e) to %s (i.e. by %.6e).' % (freq, rounded, float(rounded) - freq))
        self._visainstrument.write('FREQ:CW %s' % rounded)

        # the power takes a much longer time to stabilize for frequencies below 2 GHz
        if freq <= 2e9: time.sleep(.1) # sleep 500ms so that power stabilizes

        # the output power drops for a short time when changing
        # frequency past certain boundaries
        boundaries = [
            #0.71e9, 0.18e9, 0.27e9, 0.36e9, 0.675e9, 0.88e9, 1.01e9, 1.42e9,
            2e9, 3.2e9, 4e9, 5.1e9, 8e9,
            10.1e9, 12.7e9, 16.01e9,
            20.2e9, 25.4e9, 28.2e9, 28.57e9,
            30.09e9, 32.02e9, 39.6e9,
            47.99e9]
        if any([ (f0 - b)*(freq - b) <= 0. for b in boundaries ]): time.sleep(.2) # sleep 200ms so that power stabilizes
        
    def do_get_status(self):
        '''
        Reads the output status from the instrument

        Input:
            None

        Output:
            status (string) : 'On' or 'Off'
        '''
        logging.debug(__name__ + ' : get status')
        stat = self._visainstrument.ask('OUTP?').replace("\r", '').replace("\n", '')

        if (stat=='1'):
          return 'on'
        elif (stat=='0'):
          return 'off'
        else:
          raise ValueError('Output status not specified : "%s"' % stat)
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

