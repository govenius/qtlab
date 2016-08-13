# Agilent_E8257D.py class, to perform the communication between the Wrapper and the device
# Pieter de Groot <pieterdegroot@gmail.com>, 2008
# Martijn Schaafsma <qtlab@mcschaafsma.nl>, 2008
# Joonas Govenius <joonas.govenius@aalto.fi>, 2014
# Mate Jenei <mate.jenei@aalto.fi>, 2016
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

class Agilent_E8257D(Instrument):
    '''
    This is the driver for the Agilent E8257D Signal Genarator

    Usage:
    Initialize with
    <name> = instruments.create('<name>', 'Agilent_E8257D', address='<GBIP address>, reset=<bool>')
    '''

    def __init__(self, name, address, reset=False):
        '''
        Initializes the Agilent_E8257D, and communicates with the wrapper.

        Input:
          name (string)    : name of the instrument
          address (string) : GPIB address
          reset (bool)     : resets to default values, default=False
        '''
        logging.info(__name__ + ' : Initializing instrument Agilent_E8257D')
        Instrument.__init__(self, name, tags=['physical'])

        # Add some global constants
        self._address = address
        self._visainstrument = visa.ResourceManager().open_resource(self._address)

        self.add_parameter('power',
            flags=Instrument.FLAG_GETSET, units='dBm', minval=-135, maxval=16, type=types.FloatType)
        self.add_parameter('phase',
            flags=Instrument.FLAG_GETSET, units='rad', minval=-np.pi, maxval=np.pi, type=types.FloatType)
        self.add_parameter('frequency', format='%.09e',
            flags=Instrument.FLAG_GETSET, units='Hz', minval=1e5, maxval=20e9, type=types.FloatType)

        self.add_parameter('idn', flags=Instrument.FLAG_GET, type=types.StringType)
        self.add_parameter('options', flags=Instrument.FLAG_GET, type=types.StringType)

        self.add_parameter('status',
            flags=Instrument.FLAG_GETSET, type=types.StringType,
            format_map={'on': 'output on',
                        'off': 'output off'})
        #self.add_parameter('modulation',
        #    flags=Instrument.FLAG_GETSET, type=types.StringType,
        #    format_map={'on': 'AM/FM/phiM on',
        #                'off': 'AM/FM/phiM off'})

        self.add_parameter('mod_pulse',
            flags=Instrument.FLAG_GETSET, type=types.StringType,
            format_map={'on': 'pulsing/gating on',
                        'off': 'pulsing/gating off'})
        self.add_parameter('mod_am',
            flags=Instrument.FLAG_GETSET, type=types.StringType,
            format_map={'on': 'AM on',
                        'off': 'AM off'})
        self.add_parameter('mod_fm',
            flags=Instrument.FLAG_GETSET, type=types.StringType,
            format_map={'on': 'FM on',
                        'off': 'FM off'})
        self.add_parameter('mod_phim',
            flags=Instrument.FLAG_GETSET, type=types.StringType,
            format_map={'on': 'phase modulation on',
                        'off': 'phase modulation off'})

                        
        self.add_parameter('pulse_source',
            flags=Instrument.FLAG_GETSET, type=types.StringType,
            format_map={'int': 'internal',
                        'ext': 'external',
                        'scal': 'scalar network analyzer'})
        self.add_parameter('pulse_inverted_polarity', flags=Instrument.FLAG_GETSET, type=types.BooleanType)


        self.add_parameter('internal_pulsesource',
            flags=Instrument.FLAG_GETSET, type=types.StringType,
            format_map={'squ': 'square',
                        'frun': 'free running',
                        'trig': 'externally triggered',
                        'doub': 'doublet',
                        'gate': 'externally gated internal pulse generator'})
        self.add_parameter('internal_pulsesource_pulsewidth', format='%.09e',
                           flags=Instrument.FLAG_GETSET|Instrument.FLAG_GET_AFTER_SET,
                           units='s', minval=70e-9, maxval=42., type=types.FloatType)
        self.add_parameter('internal_pulsesource_pulseperiod', format='%.09e',
                           flags=Instrument.FLAG_GETSET|Instrument.FLAG_GET_AFTER_SET,
                           units='s', minval=70e-9, maxval=42., type=types.FloatType)

                        
        self.add_parameter('am_type',
            flags=Instrument.FLAG_GETSET, type=types.StringType,
            format_map={'linear': 'linear AM (depth = percent/Volt)',
                        'exponential': 'exponential AM (depth = dB/Volt)'})
        self.add_parameter('am_linear_depth',
            flags=Instrument.FLAG_GETSET, units='%', minval=0, maxval=100., type=types.FloatType)
        self.add_parameter('am_exponential_depth',
            flags=Instrument.FLAG_GETSET, units='dB/Volt', minval=0, maxval=40., type=types.FloatType)
        self.add_parameter('am_mode',
            flags=Instrument.FLAG_GETSET, type=types.StringType,
            format_map={'normal': 'normal (using ALC)',
                        'deep': 'deep modulation (ALC disabled for < -10 dBm amplitude)'})
        self.add_parameter('am_inverted_polarity',
            flags=Instrument.FLAG_GETSET, type=types.BooleanType)
        self.add_parameter('ext1_input_impedance',
            flags=Instrument.FLAG_GETSET, type=types.IntType,
            format_map={50: '50 Ohm',
                        600: '600 Ohm'})
        self.add_parameter('ext2_input_impedance',
            flags=Instrument.FLAG_GETSET, type=types.IntType,
            format_map={50: '50 Ohm',
                        600: '600 Ohm'})
        self.add_parameter('am_source',
            flags=Instrument.FLAG_GETSET, type=types.StringType,
            format_map={'int1': 'internal generator 1',
                        'int2': 'internal generator 2',
                        'ext1': 'external modulation (EXT1 input)',
                        'ext2': 'external modulation (EXT2 input)'})
        
        self.add_parameter('fm_dev',
            flags=Instrument.FLAG_GETSET, units='Hz', minval=1, maxval=32e+6, type=types.FloatType)
        self.add_parameter('fm_rate',
            flags=Instrument.FLAG_GETSET, units='Hz', minval=0, maxval=100e+3, type=types.FloatType)

        self.add_parameter('frequency_sweep_mode',
            flags=Instrument.FLAG_GETSET, type=types.StringType,
            format_map={'fixed': 'fixed frequency',
                        'sweep': 'regular sweep',
                        'list': 'list sweep'})
        self.add_parameter('sweep_time',
            flags=Instrument.FLAG_GETSET, units='s', minval=0.01, maxval=99., type=types.FloatType)
        self.add_parameter('frequency_center', format='%.09e',
            flags=Instrument.FLAG_GETSET, units='Hz', minval=30e3, maxval=20e9, type=types.FloatType)
        self.add_parameter('frequency_start', format='%.09e',
            flags=Instrument.FLAG_GETSET, units='Hz', minval=30e3, maxval=20e9, type=types.FloatType)
        self.add_parameter('frequency_stop', format='%.09e',
            flags=Instrument.FLAG_GETSET, units='Hz', minval=30e3, maxval=20e9, type=types.FloatType)
        self.add_parameter('frequency_span', format='%.09e',
            flags=Instrument.FLAG_GETSET, units='Hz', minval=0, maxval=10e9, type=types.FloatType)
        self.add_parameter('frequency_step', format='%.09e',
            flags=Instrument.FLAG_GETSET, units='Hz', minval=0.01, maxval=99e9, type=types.FloatType)
        
        self.add_parameter('trigger_source',
            flags=Instrument.FLAG_GETSET|Instrument.FLAG_GET_AFTER_SET, type=types.StringType)
        

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
        self.get_idn()
        self.get_options()
        self.get_power()
        self.get_phase()
        self.get_frequency()
        self.get_status()
        
        self.get_mod_pulse()
        self.get_mod_am()
        self.get_mod_fm()
        self.get_mod_phim()
        
        self.get_pulse_source()
        self.get_pulse_inverted_polarity()
        
        self.get_am_source()
        self.get_am_mode()
        self.get_am_type()
        self.get_am_inverted_polarity()
        self.get_am_exponential_depth()
        self.get_am_linear_depth()

        self.get_fm_dev()
        self.get_fm_rate()

        self.get_ext1_input_impedance()
        self.get_ext2_input_impedance()

        self.get_internal_pulsesource()
        self.get_internal_pulsesource_pulsewidth()
        self.get_internal_pulsesource_pulseperiod()

        self.get_frequency_sweep_mode()
        self.get_sweep_time()
        self.get_frequency_center()
        self.get_frequency_start()
        self.get_frequency_stop()
        self.get_frequency_span()
        self.get_frequency_step()
        self.get_trigger_source()
        

    def __to_rounded_string(self, x, decimals, significant_figures):
        ''' Round x to the specified number of decimals and significant figures.
            Output a warning if rounded value is not equal to x. '''
        rounded = ('%.{0}e'.format(significant_figures-1)) % ( np.round(x, decimals=decimals) )
        if np.abs(float(rounded) - x) > np.finfo(np.float).tiny:
          logging.warn('Rounding the requested value (%.20e) to %s (i.e. by %.20e).' % (x, rounded, x - float(rounded)))
        return rounded

    def do_get_idn(self):
        '''
        Get a string identifying the instrument.
        '''
        return self._visainstrument.ask('*IDN?')

    def do_get_options(self):
        '''
        Get a string identifying the installed options.
        '''
        return self._visainstrument.ask(':DIAG:INFO:OPT:DET?')

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
        
        self._visainstrument.write('FREQ:CW %s' % (
            self.__to_rounded_string(freq, decimals=3, significant_figures=20) ))
    
    
    def do_get_frequency_center(self):
        '''
        Reads the frequency center point for a step sweep or ramp sweep of the instrument

        Input:
            None

        Output:
            freq (float) : Frequency in Hz
        '''
        logging.debug(__name__ + ' : get frequency_center')
        return float(self._visainstrument.ask('FREQ:CENT?'))

    def do_set_frequency_center(self, freq):
        '''
        Set the frequency center point for a step sweep or ramp sweep of the instrument

        Input:
            freq (float) : Frequency in Hz

        Output:
            None
        '''
        logging.debug(__name__ + ' : set frequency_center to %f' % freq)
        self._visainstrument.write('FREQ:CENT %s' % freq)
        self.get_frequency_start()
        self.get_frequency_stop()
        self.get_frequency_span()
        
    def do_get_frequency_start(self):
        '''
        Reads the frequency start point for a step sweep or ramp sweep of the instrument

        Input:
            None

        Output:
            freq (float) : Frequency in Hz
        '''
        logging.debug(__name__ + ' : get frequency_start')
        return float(self._visainstrument.ask('FREQ:STAR?'))

    def do_set_frequency_start(self, freq):
        '''
        Set the frequency start point for a step sweep or ramp sweep of the instrument

        Input:
            freq (float) : Frequency in Hz

        Output:
            None
        '''
        logging.debug(__name__ + ' : set frequency_start to %f' % freq)
        self._visainstrument.write('FREQ:STAR %s' % freq)
        self.get_frequency_stop()
        self.get_frequency_center()
        self.get_frequency_span()
        
    def do_get_frequency_stop(self):
        '''
        Reads the frequency stop point for a step sweep or ramp sweep of the signal from the instrument

        Input:
            None

        Output:
            freq (float) : Frequency in Hz
        '''
        logging.debug(__name__ + ' : get frequency_stop')
        return float(self._visainstrument.ask('FREQ:STOP?'))

    def do_set_frequency_stop(self, freq):
        '''
        Set the frequency stop point for a step sweep or ramp sweep of the instrument

        Input:
            freq (float) : Frequency in Hz

        Output:
            None
        '''
        logging.debug(__name__ + ' : set frequency_stop to %f' % freq)
        self._visainstrument.write('FREQ:STOP %s' % freq)
        self.get_frequency_start()
        self.get_frequency_center()
        self.get_frequency_span()
        
    def do_get_frequency_span(self):
        '''
        Reads the frequency span of the signal from the instrument

        Input:
            None

        Output:
            freq (float) : Frequency in Hz
        '''
        logging.debug(__name__ + ' : get frequency_span')
        return float(self._visainstrument.ask('FREQ:SPAN?'))

    def do_set_frequency_span(self, freq):
        '''
        Set the frequency span of the instrument

        Input:
            freq (float) : Frequency in Hz

        Output:
            None
        '''
        logging.debug(__name__ + ' : set frequency_span to %f' % freq)
        self._visainstrument.write('FREQ:SPAN %s' % freq)
        self.get_frequency_start()
        self.get_frequency_stop()
        self.get_frequency_center()
    
    def do_get_frequency_step(self):
        '''
        Reads the frequency step of the signal from the instrument

        Input:
            None

        Output:
            freq (float) : Frequency in Hz
        '''
        logging.debug(__name__ + ' : get frequency_step')
        return float(self._visainstrument.ask('FREQ:STEP?'))

    def do_set_frequency_step(self, freq):
        '''
        Set the frequency step of of the signal from the instrument

        Input:
            freq (float) : Frequency in Hz

        Output:
            None
        '''
        logging.debug(__name__ + ' : set frequency_step to %f' % freq)
        self._visainstrument.write('FREQ:STEP %s' % freq)
        self.get_sweep_time()
        
    def do_get_status(self):
        '''
        Reads the output status from the instrument

        Input:
            None

        Output:
            status (string) : 'On' or 'Off'
        '''
        logging.debug(__name__ + ' : get status')
        stat = self._visainstrument.ask('OUTP?').strip().lower()
        return 'on' if (stat in ['1', 'on']) else 'off'

    def do_set_status(self, status):
        '''
        Set the output status of the instrument

        Input:
            status (string) : 'On' or 'Off'

        Output:
            None
        '''
        logging.debug(__name__ + ' : set status to %s' % status)
        self._visainstrument.write('OUTP %s' % status.upper())
        
    def do_get_frequency_sweep_mode(self):
        stat = self._visainstrument.ask(':FREQ:MODE?').lower().strip()
        if stat.startswith('fix') or stat.startswith('cw'): return 'fixed'
        if stat.startswith('swe'): return 'sweep'
        if stat.startswith('lis'): return 'list'
        raise Exception('Unknown sweep mode: %s' % stat)
    def do_set_frequency_sweep_mode(self, mode):
        self._visainstrument.write(':FREQ:MODE %s' % (mode.upper()))


    def do_get_sweep_time(self):
        '''
        Reads the sweep time of the ramp.

        Input:
            None

        Output:
            sweep time in seconds
        '''
        logging.debug(__name__ + ' : get sweep time')
        return float(self._visainstrument.ask(':SWEEP:TIME?'))

    def do_set_sweep_time(self, amp):
        '''
        Set the sweep time of the ramp.

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

        
    def do_get_mod_pulse(self):
        stat = self._visainstrument.ask(':PULM:STAT?')
        return 'on' if (stat.lower().strip() in ['1', 'on']) else 'off'
    def do_set_mod_pulse(self, status):
        self._visainstrument.write(':PULM:STAT %s' % (status.upper()))
        
    def do_get_mod_am(self):
        stat = self._visainstrument.ask(':AM:STAT?')
        return 'on' if stat.lower().strip() in ['1', 'on'] else 'off'
    def do_set_mod_am(self, status):
        self._visainstrument.write(':AM:STAT %s' % (status.upper()))
        
    def do_get_mod_fm(self):
        stat = self._visainstrument.ask(':FM:STAT?')
        return 'on' if stat.lower().strip() in ['1', 'on'] else 'off'
    def do_set_mod_fm(self, status):
        self._visainstrument.write(':FM:STAT %s' % (status.upper()))

    def do_get_mod_phim(self):
        stat = self._visainstrument.ask(':PM:STAT?')
        return 'on' if stat.lower().strip() in ['1', 'on'] else 'off'
    def do_set_mod_phim(self, status):
        self._visainstrument.write(':PM:STAT %s' % (status.upper()))
        
    def do_get_pulse_source(self):
        stat = self._visainstrument.ask(':PULM:SOUR?').lower()
        return stat
    def do_set_pulse_source(self, state):
        self._visainstrument.write( ':PULM:SOUR %s' % (state.upper()) )

    def do_get_internal_pulsesource(self):
        stat = self._visainstrument.ask(':PULM:SOUR:INT?').lower()
        return stat
    def do_set_internal_pulsesource(self, state):
        self._visainstrument.write( ':PULM:SOUR:INT %s' % (state.upper()) )
        self.get_pulse_source() # changes from 'ext' to 'int' (if not 'int' already)

    def do_get_internal_pulsesource_pulsewidth(self):
        return float( self._visainstrument.ask(':PULM:INT:PWID?') )
    def do_set_internal_pulsesource_pulsewidth(self, w):
        self._visainstrument.write(':PULM:INT:PWID %s' % (
            self.__to_rounded_string(w, decimals=9, significant_figures=9) ))

    def do_get_internal_pulsesource_pulseperiod(self):
        return float( self._visainstrument.ask(':PULM:INT:PER?') )
    def do_set_internal_pulsesource_pulseperiod(self, w):
        self._visainstrument.write(':PULM:INT:PER %s' % (
            self.__to_rounded_string(w, decimals=9, significant_figures=9) ))

    def do_get_pulse_inverted_polarity(self):
        stat = self._visainstrument.ask(':PULM:EXT:POL?')
        return stat.lower().strip().startswith('inv')
    def do_set_pulse_inverted_polarity(self, val):
        self._visainstrument.write(':PULM:EXT:POL %s' % ('INV' if val else 'NORM'))
 
    def do_get_am_source(self):
        stat = self._visainstrument.ask(':AM:SOUR?')
        if stat.lower() == 'ext': return 'ext1' # These are synonyms
        return stat.lower()
    def do_set_am_source(self, status):
        self._visainstrument.write(':AM:SOUR %s' % (status.upper()))
        
    def do_get_am_mode(self):
        stat = self._visainstrument.ask(':AM:MODE?')
        return stat.lower()
    def do_set_am_mode(self, status):
        self._visainstrument.write(':AM:MODE %s' % (status.upper()))
        
    def do_get_am_type(self):
        stat = self._visainstrument.ask(':AM:TYPE?')
        return stat.lower()
    def do_set_am_type(self, status):
        self._visainstrument.write(':AM:TYPE %s' % (status.upper()))
        
    def do_get_am_linear_depth(self):
        stat = self._visainstrument.ask(':AM:DEPTH:LINEAR?')
        return stat.lower()
    def do_set_am_linear_depth(self, val):
        self._visainstrument.write(':AM:DEPTH:LINEAR %g' % val)
        
    def do_get_am_exponential_depth(self):
        stat = self._visainstrument.ask(':AM:DEPTH:EXP?')
        return stat.lower()
    def do_set_am_exponential_depth(self, val):
        self._visainstrument.write(':AM:DEPTH:EXP %g' % val)
        
    def do_get_am_inverted_polarity(self):
        stat = self._visainstrument.ask(':AM:POL?')
        return stat.lower().strip().startswith('inv')
    def do_set_am_inverted_polarity(self, val):
        self._visainstrument.write(':AM:POL %s' % ('INV' if val else 'NORM'))
        
    def do_get_fm_dev(self):
        stat = self._visainstrument.ask(':FM:DEV?')
        return stat.lower()

    def do_set_fm_dev(self, val):
        self._visainstrument.write(':FM:DEV %g' % val)

    def do_get_fm_rate(self):
        stat = self._visainstrument.ask(':FM:INT:FREQ?')
        return stat.lower()
    def do_set_fm_rate(self, val):
        self._visainstrument.write(':FM:INT:FREQ %g' % val)

    def do_get_ext1_input_impedance(self):
        stat = self._visainstrument.ask(':AM:EXT1:IMP?')
        return int(float(stat))
    def do_set_ext1_input_impedance(self, val):
        self._visainstrument.write(':AM:EXT1:IMP %d' % val)
        
    def do_get_ext2_input_impedance(self):
        stat = self._visainstrument.ask(':AM:EXT2:IMP?')
        return int(float(stat))
    def do_set_ext2_input_impedance(self, val):
        self._visainstrument.write(':AM:EXT2:IMP %d' % val)
