# Agilent_N1913A.py class, for commucation with an Agilent N1913A RF power sensor
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
import visa
import types
import logging
import numpy

class Agilent_N1913A(Instrument):
    '''
    This is the driver for the Agilent N1913A RF power sensor

    Usage:
    Initialize with
    <name> = instruments.create('<name>', 'Agilent_N1913A', address='<GBIP address>, reset=<bool>')
    '''

    def __init__(self, name, address, reset=False):
        '''
        Initializes the Agilent_N1913A, and communicates with the wrapper.

        Input:
          name (string)    : name of the instrument
          address (string) : GPIB address
          reset (bool)     : resets to default values, default=False
        '''
        logging.info(__name__ + ' : Initializing instrument Agilent_N1913A')
        Instrument.__init__(self, name, tags=['physical'])

        # Add some global constants
        self._address = address
#>>>>>>>>>>>>>>
        assert False, "pyvisa syntax has changed, tweak the line below according to the instructions in qtlab/instrument_plugins/README_PYVISA_API_CHANGES"
        #self._visainstrument = visa.instrument(self._address)
#<<<<<<<<<<<<<<

        self.add_parameter('averaging',
            flags=Instrument.FLAG_GETSET, units='arb', minval=1, maxval=1024, type=types.IntType)
        self.add_parameter('status',
            flags=Instrument.FLAG_GETSET, type=types.StringType)

        self.add_function('reset')
        self.add_function ('get_all')
        self.add_function('get_reading')


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

    def get_reading(self):
        '''
        Gets the current power reading.

        Input:
            None

        Output:
            output power in dBm
        '''
        logging.debug(__name__ + ' : getting current power reading')
        p = self._visainstrument.ask('FETC?')
        try:
            return float(p)
        except ValueError:
            print "Could not convert {0} to float.".format(p)
            return 0.
		
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
        self.get_averaging()
        self.get_status()

    def do_get_averaging(self):
        '''
        Reads the averaging of the signal from the instrument

        Input:
            None

        Output:
            len (int) : averaging length
        '''
        logging.debug(__name__ + ' : get averaging')
        return float(self._visainstrument.ask('SENS:AVER:COUN?'))

    def do_set_averaging(self, amp):
        '''
        Set the averaging of the signal

        Input:
            len (int) : averaging length

        Output:
            None
        '''
        logging.debug(__name__ + ' : set averaging to %f' % amp)
        self._visainstrument.write('SENS:AVER:COUN %s' % amp)

    def do_get_status(self):
        '''
        Reads the output status from the instrument

        Input:
            None

        Output:
            status (string) : 'On' or 'Off'
        '''
        logging.debug(__name__ + ' : get status')
        stat = self._visainstrument.ask('*STB?')
        logging.debug(__name__ + ' : get status == ' + stat)

#        if (stat=='1'):
#          return 'on'
#        elif (stat=='0'):
#          return 'off'
#        else:
#          raise ValueError('Output status not specified : %s' % stat)
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

