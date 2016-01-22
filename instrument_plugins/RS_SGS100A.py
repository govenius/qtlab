# RS_SGS100A.py class, to perform the communication between the Wrapper and the device
# Joonas Govenius <joonas.govenius@aalto.fi>, 2015
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

class RS_SGS100A(Instrument):
    '''
    RF generator.

    Usage:
    Initialize with
    <name> = instruments.create('<name>', 'RS_SGS100A', address='<VISA address>, reset=<bool>')
    '''

    def __init__(self, name, address, reset=False):
        '''
        Initializes the RS_SGS100A, and communicates with the wrapper.

        Input:
          name (string)    : name of the instrument
          address (string) : VISA address
          reset (bool)     : resets to default values, default=False
        '''
        logging.info(__name__ + ' : Initializing instrument RS_SGS100A')
        Instrument.__init__(self, name, tags=['physical'])

        # Add some global constants
        self._address = address
        self._visainstrument = visa.ResourceManager().open_resource(self._address, timeout=2000)

        try:
          self._visainstrument.read_termination = '\n'
          self._visainstrument.write_termination = '\n'

          self.add_parameter('idn', flags=Instrument.FLAG_GET, type=types.StringType)

          self.add_parameter('power',
              flags=Instrument.FLAG_GETSET|Instrument.FLAG_GET_AFTER_SET, units='dBm', minval=-20, maxval=25, type=types.FloatType)
          self.add_parameter('frequency', format='%.09e',
              flags=Instrument.FLAG_GETSET|Instrument.FLAG_GET_AFTER_SET, units='Hz', minval=1e6, maxval=12.75e9, type=types.FloatType)
          self.add_parameter('timebase', type=types.StringType,
                             flags=Instrument.FLAG_GETSET|Instrument.FLAG_GET_AFTER_SET,
                             format_map={'INT':'internal',
                                         'EXT':'external'})
          self.add_parameter('timebase_external_input_frequency', type=types.IntType,
                             flags=Instrument.FLAG_GETSET|Instrument.FLAG_GET_AFTER_SET,
                             format_map={10:'10 MHz',
                                         100:'100 MHz',
                                         1000:'1000 MHz'})

          self.add_parameter('status',
              flags=Instrument.FLAG_GETSET|Instrument.FLAG_GET_AFTER_SET, type=types.StringType,
              format_map={'on': 'output on',
                          'off': 'output off'})

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
        self.get_frequency()
        self.get_status()
        self.get_timebase()
        self.get_timebase_external_input_frequency()

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
        return self._visainstrument.query('*IDN?')

    def do_get_power(self):
        '''
        Reads the power of the signal from the instrument.
        '''
        logging.debug(__name__ + ' : get power')
        return float(self._visainstrument.query('POW?'))

    def do_set_power(self, amp):
        '''
        Set the power of the signal.
        '''
        logging.debug(__name__ + ' : set power to %f' % amp)
        self._visainstrument.write('POW %s' % self.__to_rounded_string(amp, 2, 20))

    def do_get_frequency(self):
        '''
        Reads the frequency of the signal from the instrument.
        '''
        logging.debug(__name__ + ' : get frequency')
        return float(self._visainstrument.query('FREQ?'))

    def do_set_frequency(self, freq):
        '''
        Set the frequency of the instrument.
        '''
        logging.debug(__name__ + ' : set frequency to %f' % freq)
        
        self._visainstrument.write('FREQ %s' % (
            self.__to_rounded_string(freq, decimals=3, significant_figures=20) ))
        
    def do_get_status(self):
        '''
        Whether the RF output is on or not.
        '''
        logging.debug(__name__ + ' : get status')
        stat = self._visainstrument.query('OUTP?').strip().lower()
        return 'on' if (stat in ['1', 'on']) else 'off'

    def do_set_status(self, status):
        '''
        Set whether the RF output is on or not.
        '''
        logging.debug(__name__ + ' : set status to %s' % status)
        self._visainstrument.write('OUTP %s' % status.upper())

    def do_get_timebase(self):
        '''
        Reference oscillator (internal/external).
        '''
        logging.debug(__name__ + ' : get timebase')
        tbase = self._visainstrument.query('ROSC:SOUR?').upper()[:3]
        return tbase

    def do_set_timebase(self, val):
        '''
        Reference oscillator (internal/external).
        '''
        logging.debug(__name__ + ' : set timebase to %s' % val)
        tbase = val.upper()[:3]
        assert tbase in ['INT', 'EXT']
        self._visainstrument.write('ROSC:SOUR %s' % tbase)

    def do_get_timebase_external_input_frequency(self):
        '''
        Reference oscillator frequency (10/100/1000 MHz).
        '''
        logging.debug(__name__ + ' : get timebase freq')
        f = self._visainstrument.query('ROSC:EXT:FREQ?').strip().upper()
        assert f[-3:] == 'MHZ', f
        return f[:-3]

    def do_set_timebase_external_input_frequency(self, val):
        '''
        Reference oscillator frequency (10/100/1000 MHz).
        '''
        logging.debug(__name__ + ' : set timebase freq')
        self._visainstrument.write('ROSC:EXT:FREQ %sMHZ' % val)
