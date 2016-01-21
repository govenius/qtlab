# Agilent_34410A.py class, for commucation with an Agilent 34410A multimeter
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

class Agilent_34410A(Instrument):
    '''
    This is the driver for the Agilent 34410A multimeter

    Usage:
    Initialize with
    <name> = instruments.create('<name>', 'Agilent_34410A', address='<GBIP address>, reset=<bool>')
    '''

    def __init__(self, name, address, reset=False):
        '''
        Initializes the Agilent_34410A, and communicates with the wrapper.

        Input:
          name (string)    : name of the instrument
          address (string) : GPIB address
          reset (bool)     : resets to default values, default=False
        '''
        logging.info(__name__ + ' : Initializing instrument Agilent_34410A')
        Instrument.__init__(self, name, tags=['physical'])

        # Add some global constants
        self._address = address
        self._visainstrument = visa.ResourceManager().open_resource(self._address, timeout=30000) # timeout is in seconds
        self._visainstrument.read_termination = '\n'
        self._visainstrument.write_termination = '\n'

        self.add_parameter('measurement_function',
            flags=Instrument.FLAG_GETSET|Instrument.FLAG_GET_AFTER_SET, type=types.StringType)
        self.add_parameter('res_nplc',
            flags=Instrument.FLAG_GETSET|Instrument.FLAG_GET_AFTER_SET, units='', minval=0.006, maxval=100, type=types.FloatType)
        self.add_parameter('res_autorange',
            flags=Instrument.FLAG_GETSET|Instrument.FLAG_GET_AFTER_SET, units='bool', type=types.BooleanType)
        self.add_parameter('res_range',
            flags=Instrument.FLAG_GETSET|Instrument.FLAG_GET_AFTER_SET, units='Ohm', minval=100., maxval=1e9, type=types.FloatType)
        self.add_parameter('v_nplc',
            flags=Instrument.FLAG_GETSET|Instrument.FLAG_GET_AFTER_SET, units='', minval=0.006, maxval=100, type=types.FloatType)
        self.add_parameter('v_autorange',
            flags=Instrument.FLAG_GETSET|Instrument.FLAG_GET_AFTER_SET, units='bool', type=types.BooleanType)
        self.add_parameter('v_range',
            flags=Instrument.FLAG_GETSET|Instrument.FLAG_GET_AFTER_SET, units='V', minval=0.1, maxval=1e3, type=types.FloatType)
        self.add_parameter('offset_compensation',
            flags=Instrument.FLAG_GETSET|Instrument.FLAG_GET_AFTER_SET, units='bool', type=types.BooleanType)
        self.add_parameter('res_null_value',
            flags=Instrument.FLAG_GETSET|Instrument.FLAG_GET_AFTER_SET, units='Ohm', minval=-1e-20, maxval=1e-20, type=types.FloatType)
        self.add_parameter('subtract_res_null_value',
            flags=Instrument.FLAG_GETSET|Instrument.FLAG_GET_AFTER_SET, units='bool', type=types.BooleanType)

        self.add_parameter('vac_autorange',
            flags=Instrument.FLAG_GETSET|Instrument.FLAG_GET_AFTER_SET, units='bool', type=types.BooleanType)
        self.add_parameter('vac_range',
            flags=Instrument.FLAG_GETSET|Instrument.FLAG_GET_AFTER_SET, units='V', minval=0.1, maxval=1e3, type=types.FloatType)
        self.add_parameter('vac_bandwidth',
            flags=Instrument.FLAG_GETSET|Instrument.FLAG_GET_AFTER_SET, units='Hz', minval=3, maxval=200, type=types.FloatType)

        self.add_parameter('trigger_source',
            flags=Instrument.FLAG_GETSET|Instrument.FLAG_GET_AFTER_SET, type=types.StringType)
        self.add_parameter('samples_per_trigger',
            flags=Instrument.FLAG_GETSET|Instrument.FLAG_GET_AFTER_SET, units='', minval=1, maxval=50000, type=types.IntType)
        self.add_parameter('trigger_delay',
            flags=Instrument.FLAG_GETSET|Instrument.FLAG_GET_AFTER_SET, units='s', minval=0., maxval=3600., type=types.FloatType)

        self.add_function('reset')
        self.add_function ('get_all')
        self.add_function('get_reading')
        self.add_function('measurement_init')
        self.add_function('measurement_trigger')
        self.add_function('measurement_fetch')


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
        Get a new reading or readings (essentially, INIT and FETCH).
        Only use this if trigger source is "IMM" or "EXT".

        Input:
            None

        Output:
            resistance in Ohms or voltage in Volts
        '''
        logging.debug(__name__ + ' : getting a new reading')
        p = self._visainstrument.ask('READ?')
        try:
            return float(p)
        except ValueError:
            print "Could not convert {0} to float.".format(p)
            return float('nan')

    def measurement_init(self):
        '''
        Go from the idle to waiting-for-trigger state.

        Input:
            None

        Output:
            None
        '''
        logging.debug(__name__ + ' : INIT')
        self._visainstrument.write('INIT')

    def measurement_trigger(self):
        '''
        Send a software trigger.
        (Device needs to be in the waiting-for-trigger state,
         i.e. need to call init first. The software trigger
         oly makes sense when trigger source is 'BUS'.)

        Input:
            None

        Output:
            None
        '''
        logging.debug(__name__ + ' : TRG')
        self._visainstrument.write('*TRG')

    def measurement_fetch(self):
        '''
        Fetch the data obtained after the last trigger.
        (Typically you need to call init and trigger first.)

        Input:
            None

        Output:
            np.ndarray of resistance in Ohms or voltage in Volts
        '''
        logging.debug(__name__ + ' : FETCH')
        s = self._visainstrument.ask('FETC?')
        return numpy.array([float(n) for n in s.split(',')])

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
        self.get_measurement_function()
        self.get_res_nplc()
        self.get_res_autorange()
        self.get_res_range()
        self.get_v_nplc()
        self.get_v_autorange()
        self.get_v_range()
        self.get_vac_autorange()
        self.get_vac_range()
        self.get_vac_bandwidth()
        self.get_offset_compensation()
        self.get_subtract_res_null_value()
        self.get_res_null_value()
        self.get_samples_per_trigger()
        self.get_trigger_delay()

    def do_get_measurement_function(self):
        '''
        Ask what is being measured.

        Input:
            None

        Output:
            function (string) : what is being measured ('RES', 'VOLT:DC')
        '''
        r = self._visainstrument.ask('SENS:FUNC?')
        logging.debug(__name__ + ' : get measurement function: ' + r)
        r = r.replace('"','')
        if r == 'VOLT': r = 'VOLT:DC'
        return r

    def do_set_measurement_function(self, val):
        '''
        Set what is being measured.

        Input:
            function (string) : what is being measured ('RES', 'VOLT:DC')

        Output:
            None
        '''
        if val == 'VOLT': val = 'VOLT:DC'
        logging.debug(__name__ + ' : set measurement_function to %s' % val)
        if val not in ['RES', 'VOLT:DC', 'VOLT:AC']:
                raise Exception('Unknown measurement function: %s' % val)
        self._visainstrument.write('SENS:FUNC "%s"' % val)
        
    def do_get_res_nplc(self):
        '''

        Input:
            None

        Output:
            len (int) : averaging length
        '''
        logging.debug(__name__ + ' : get res_nplc')
        return float(self._visainstrument.ask('SENS:RES:NPLC?'))

    def do_set_res_nplc(self, val):
        '''

        Input:
            len (int) : averaging length

        Output:
            None
        '''
        logging.debug(__name__ + ' : set res_nplc to %f' % val)
        self._visainstrument.write('SENS:RES:NPLC %s' % val)

    def do_get_v_nplc(self):
        '''

        Input:
            None

        Output:
            len (int) : averaging length
        '''
        logging.debug(__name__ + ' : get v_nplc')
        return float(self._visainstrument.ask('SENS:VOLT:DC:NPLC?'))

    def do_set_v_nplc(self, val):
        '''

        Input:
            len (int) : averaging length

        Output:
            None
        '''
        logging.debug(__name__ + ' : set v_nplc to %f' % val)
        self._visainstrument.write('SENS:VOLT:DC:NPLC %s' % val)

    def do_get_res_range(self):
        '''

        Input:
            None

        Output:
            len (int) : averaging length
        '''
        logging.debug(__name__ + ' : get res_range')
        return float(self._visainstrument.ask('SENS:RES:RANG?'))

    def do_set_res_range(self, val):
        '''

        Input:
            len (int) : averaging length

        Output:
            None
        '''
        logging.debug(__name__ + ' : set res_range to %f' % val)
        self._visainstrument.write('SENS:RES:RANG %s' % val)

    def do_get_v_range(self):
        '''

        Input:
            None

        Output:
            range (V)
        '''
        logging.debug(__name__ + ' : get v_range')
        return float(self._visainstrument.ask('SENS:VOLT:DC:RANG?'))

    def do_set_v_range(self, val):
        '''

        Input:
            range (V)

        Output:
            None
        '''
        logging.debug(__name__ + ' : set v_range to %f' % val)
        self._visainstrument.write('SENS:VOLT:DC:RANG %s' % val)

    def do_get_vac_range(self):
        '''

        Input:
            None

        Output:
            range (V)
        '''
        logging.debug(__name__ + ' : get vac_range')
        return float(self._visainstrument.ask('SENS:VOLT:AC:RANG?'))

    def do_set_vac_range(self, val):
        '''

        Input:
            range (V)

        Output:
            None
        '''
        logging.debug(__name__ + ' : set vac_range to %f' % val)
        self._visainstrument.write('SENS:VOLT:AC:RANG %s' % val)

    def do_get_vac_bandwidth(self):
        '''

        Input:
            None

        Output:
            range (V)
        '''
        logging.debug(__name__ + ' : get vac_bandwidth')
        return float(self._visainstrument.ask('SENS:VOLT:AC:BAND?'))

    def do_set_vac_bandwidth(self, val):
        '''

        Input:
            range (V)

        Output:
            None
        '''
        logging.debug(__name__ + ' : set vac_bandwidth to %f' % val)
        self._visainstrument.write('SENS:VOLT:AC:BAND %s' % val)
        
    def do_get_res_null_value(self):
        '''
        Null value is subtracted from the measurement (if option turned on).

        Input:
            None

        Output:
            len (int) : averaging length
        '''
        logging.debug(__name__ + ' : get res_null value')
        return float(self._visainstrument.ask('SENS:RES:NULL:VAL?'))

    def do_set_res_null_value(self, val):
        '''
        Null value is subtracted from the measurement (if option turned on).

        Input:
            len (int) : averaging length

        Output:
            None
        '''
        logging.debug(__name__ + ' : set res_null value to %f' % val)
        self._visainstrument.write('SENS:RES:NULL:VAL %s' % val)

    def do_get_res_autorange(self):
        '''
        Is res_autorange on?

        Input:
            None

        Output:
            len (int) : averaging length
        '''
        r = self._visainstrument.ask('SENS:RES:RANG:AUTO?')
        logging.debug(__name__ + ' : get res_autorange: ' + r)
        return bool(int(r))

    def do_set_res_autorange(self, val):
        '''
        Sets res_autorange on/off.

        Input:
            len (int) : averaging length

        Output:
            None
        '''
        logging.debug(__name__ + ' : set res_autorange to %f' % val)
        self._visainstrument.write('SENS:RES:RANG:AUTO %s' % int(val))

    def do_get_v_autorange(self):
        '''
        Is v_autorange on?

        Input:
            None

        Output:
            len (int) : averaging length
        '''
        r = self._visainstrument.ask('SENS:VOLT:DC:RANG:AUTO?')
        logging.debug(__name__ + ' : get v_autorange: ' + r)
        return bool(int(r))

    def do_set_v_autorange(self, val):
        '''
        Sets v_autorange on/off.

        Input:
            len (int) : averaging length

        Output:
            None
        '''
        logging.debug(__name__ + ' : set v_autorange to %f' % val)
        self._visainstrument.write('SENS:VOLT:DC:RANG:AUTO %s' % int(val))
        
    def do_get_vac_autorange(self):
        '''
        Is vac_autorange on?

        Input:
            None

        Output:
            len (int) : averaging length
        '''
        r = self._visainstrument.ask('SENS:VOLT:AC:RANG:AUTO?')
        logging.debug(__name__ + ' : get vac_autorange: ' + r)
        return bool(int(r))

    def do_set_vac_autorange(self, val):
        '''
        Sets vac_autorange on/off.

        Input:
            len (int) : averaging length

        Output:
            None
        '''
        logging.debug(__name__ + ' : set vac_autorange to %f' % val)
        self._visainstrument.write('SENS:VOLT:AC:RANG:AUTO %s' % int(val))
        
    def do_get_offset_compensation(self):
        '''

        Input:
            None

        Output:
            len (int) : averaging length
        '''
        r = self._visainstrument.ask('SENS:RES:OCOM?')
        logging.debug(__name__ + ' : get offset_compensation: ' + r)
        return bool(int(r))

    def do_set_offset_compensation(self, val):
        '''

        Input:
            len (int) : averaging length

        Output:
            None
        '''
        logging.debug(__name__ + ' : set offset_compensation to %f' % val)
        self._visainstrument.write('SENS:RES:OCOM %s' % int(val))
        
    def do_get_subtract_res_null_value(self):
        '''

        Input:
            None

        Output:
            len (int) : averaging length
        '''
        r = self._visainstrument.ask('SENS:RES:NULL?')
        logging.debug(__name__ + ' : get res_null value subtraction: ' + r)
        return bool(int(r))

    def do_set_subtract_res_null_value(self, val):
        '''

        Input:
            len (int) : averaging length

        Output:
            None
        '''
        logging.debug(__name__ + ' : set subtract_res_null_value to %f' % val)
        self._visainstrument.write('SENS:RES:NULL %s' % int(val))

    def do_get_samples_per_trigger(self):
        '''

        Input:
            None

        Output:
            samples (int)
        '''
        r = self._visainstrument.ask('SAMP:COUN?')
        logging.debug(__name__ + ' : get samples_per_trigger: %s' % r)
        return int(r)

    def do_set_samples_per_trigger(self, val):
        '''

        Input:
            samples (int) [1, 5e4]

        Output:
            None
        '''
        logging.debug(__name__ + ' : set samples_per_trigger to %f' % val)
        self._visainstrument.write('SAMP:COUN %u' % val)

    def do_get_trigger_delay(self):
        '''

        Input:
            None

        Output:
            delay in seconds (float)
        '''
        r = self._visainstrument.ask('TRIG:DEL?')
        logging.debug(__name__ + ' : get trigger_delay: %s' % r)
        return float(r)

    def do_set_trigger_delay(self, val):
        '''

        Input:
            delay in seconds (float) [0., 3600.]

        Output:
            None
        '''
        logging.debug(__name__ + ' : set trigger_delay to %f' % val)
        self._visainstrument.write('TRIG:DEL %s' % val)


    def do_get_trigger_source(self):
        '''

        Input:
            None

        Output:
            source ['IMM', 'EXT', 'BUS']
        '''
        r = self._visainstrument.ask('TRIG:SOUR?')
        logging.debug(__name__ + ' : get trigger_source: %s' % r)
        return r

    def do_set_trigger_source(self, val):
        '''

        Input:
            source ['IMM', 'EXT', 'BUS']

        Output:
            None
        '''
        if val not in ['IMM', 'EXT', 'BUS']:
          raise Exception('Invalid trigger source "%s". Should be "IMM", "EXT" or "BUS".')
        logging.debug(__name__ + ' : set trigger_source to %s' % val)
        self._visainstrument.write('TRIG:SOUR %s' % val)