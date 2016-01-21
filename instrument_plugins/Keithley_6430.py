# Keithley_6430.py driver for Keithley 6430 SMU
# Russell Lake <russell.lake@aalto.fi>, 2012
#
# Based on driver for Keithley 2700 by:
# Pieter de Groot <pieterdegroot@gmail.com>, 2008
# Martijn Schaafsma <qtlab@mcschaafsma.nl>, 2008
# Reinier Heeres <reinier@heeres.eu>, 2008
#
# Update december 2009:
# Michiel Jol <jelle@michieljol.nl>
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

import qt

def bool_to_str(val):
    '''
    Function to convert boolean to 'ON' or 'OFF'
    '''
    if val == True:
        return "ON"
    else:
        return "OFF"

class Keithley_6430(Instrument):
    '''
    This is the driver for the Keithley 2700 Multimeter

    Usage:
    Initialize with
    <name> = instruments.create('<name>', 'Keithley_6430',
        address='<GBIP address>',
        reset=<bool>,
        change_display=<bool>,
        change_autozero=<bool>)
    '''

    def __init__(self, name, address, reset=False
        #, change_display=True, change_autozero=True
        ):
        '''
        Initializes the Keithley_6430, and communicates with the wrapper.

        Input:
            name (string)           : name of the instrument
            address (string)        : GPIB address
            reset (bool)            : resets to default values
            change_display (bool)   : If True (default), automatically turn off
                                        display during measurements.
            change_autozero (bool)  : If True (default), automatically turn off
                                        autozero during measurements.
        Output:
            None
        '''
        # Initialize wrapper functions
        logging.info('Initializing instrument Keithley_6430')
        Instrument.__init__(self, name, tags=['physical'])

        # Add some global constants
        self._address = address
        self._visainstrument = visa.ResourceManager().open_resource(self._address, timeout=30000) # timeout is in ms (should be > source delay)
        self._visainstrument.read_termination = '\n'
        self._visainstrument.write_termination = '\n'

        self._source_modes = ['VOLT', 'CURR']
        self._sense_modes = ['VOLT:DC', 'CURR:DC', 'RES']

        #self._change_display = change_display
        #self._change_autozero = change_autozero
        #self._trigger_sent = False

        # Add parameters to wrapper
        self.add_parameter('source_current_compliance',
            flags=Instrument.FLAG_GETSET,
            units='A', minval=1e-9, maxval=105e-3, type=types.FloatType)
        self.add_parameter('source_voltage_compliance',
            flags=Instrument.FLAG_GETSET,
            units='V', minval=1e-12, maxval=210., type=types.FloatType)

        self.add_parameter('source_current_compliance_tripped',
            flags=Instrument.FLAG_GET,
            type=types.BooleanType)
        self.add_parameter('source_voltage_compliance_tripped',
            flags=Instrument.FLAG_GET,
            type=types.BooleanType)

        self.add_parameter('source_current_level',
            flags=Instrument.FLAG_GETSET,
            units='A', minval=-105e-3, maxval=105e-3, type=types.FloatType)
        self.add_parameter('source_current_range',
            flags=Instrument.FLAG_GETSET|Instrument.FLAG_GET_AFTER_SET,
            units='A', minval=1e-12, maxval=105e-3, type=types.FloatType, format='%.3e')
        self.add_parameter('source_voltage_level',
            flags=Instrument.FLAG_GETSET,
            units='V', minval=-210., maxval=210., type=types.FloatType)
        self.add_parameter('source_voltage_range',
            flags=Instrument.FLAG_GETSET|Instrument.FLAG_GET_AFTER_SET,
            units='V', minval=200e-3, maxval=200, type=types.FloatType, format='%.3e')

        self.add_parameter('source_delay_auto',
            flags=Instrument.FLAG_GETSET,
            type=types.BooleanType)
        self.add_parameter('source_delay',
            flags=Instrument.FLAG_GETSET,
            units='s', minval=0., maxval=9999.998, type=types.FloatType)

        self.add_parameter('output_on',
            flags=Instrument.FLAG_GETSET,
            type=types.BooleanType)
        self.add_parameter('output_auto_off',
            flags=Instrument.FLAG_GETSET,
            type=types.BooleanType)
        self.add_parameter('source_mode',
            flags=Instrument.FLAG_GETSET,
            type=types.StringType, units='')
        self.add_parameter('sense_mode',
            flags=Instrument.FLAG_GETSET,
            type=types.StringType, units='')

        self.add_parameter('sense_autorange',
            flags=Instrument.FLAG_GETSET,
            type=types.BooleanType)
        self.add_parameter('sense_current_range',
            flags=Instrument.FLAG_GETSET,
            units='A',
            type=types.FloatType)
        self.add_parameter('sense_voltage_range',
            flags=Instrument.FLAG_GETSET,
            units='V',
            type=types.FloatType)
        self.add_parameter('sense_resistance_range',
            flags=Instrument.FLAG_GETSET,
            units='Ohm',
            type=types.FloatType)

        self.add_parameter('sense_resistance_ocomp', flags=Instrument.FLAG_GETSET,
            type=types.BooleanType)

        self.add_parameter('trigger_source',
            flags=Instrument.FLAG_GETSET,
            type=types.StringType)
        self.add_parameter('arm_source',
            flags=Instrument.FLAG_GETSET,
            type=types.StringType)

        self.add_parameter('trigger_count',
            flags=Instrument.FLAG_GETSET,
            type=types.IntType)
        self.add_parameter('arm_count',
            flags=Instrument.FLAG_GETSET,
            type=types.IntType)

        # self.add_parameter('trigger_inline',
        #     flags=Instrument.FLAG_GETSET,
        #     type=types.BooleanType)
        # self.add_parameter('trigger_count',
        #     flags=Instrument.FLAG_GETSET,
        #     units='#', type=types.IntType)
        # self.add_parameter('trigger_delay',
        #     flags=Instrument.FLAG_GETSET,
        #     units='s', minval=0., maxval=999.9999, type=types.FloatType)
        # self.add_parameter('trigger_source',
        #     flags=Instrument.FLAG_GETSET,
        #     units='')
        # self.add_parameter('trigger_timer',
        #     flags=Instrument.FLAG_GETSET,
        #     units='s', minval=0.001, maxval=99999.999, type=types.FloatType)

        # self.add_parameter('readval', flags=Instrument.FLAG_GET,
        #     units='AU',
        #     type=types.FloatType,
        #     tags=['measure'])
        # self.add_parameter('readlastval', flags=Instrument.FLAG_GET,
        #     units='AU',
        #     type=types.FloatType,
        #     tags=['measure'])
        # self.add_parameter('readnextval', flags=Instrument.FLAG_GET,
        #     units='AU',
        #     type=types.FloatType,
        #     tags=['measure'])

        self.add_parameter('nplc',
            flags=Instrument.FLAG_GETSET,
            units='#', type=types.FloatType, minval=0.01, maxval=10)
        # self.add_parameter('display', flags=Instrument.FLAG_GETSET,
        #     type=types.BooleanType)
        self.add_parameter('digits',
            flags=Instrument.FLAG_GETSET,
            units='', minval=4, maxval=7, type=types.IntType)            
        self.add_parameter('autozero', flags=Instrument.FLAG_GETSET,
            type=types.BooleanType)

        self.add_parameter('filter_auto',
            flags=Instrument.FLAG_GETSET,
            type=types.BooleanType)
        self.add_parameter('filter_repeat_enabled',
            flags=Instrument.FLAG_GETSET,
            type=types.BooleanType)
        self.add_parameter('filter_median_enabled',
            flags=Instrument.FLAG_GETSET,
            type=types.BooleanType)
        self.add_parameter('filter_moving_enabled',
            flags=Instrument.FLAG_GETSET,
            type=types.BooleanType)
        self.add_parameter('filter_repeat',
            flags=Instrument.FLAG_GETSET,
            units='',
            type=types.IntType)
        self.add_parameter('filter_median',
            flags=Instrument.FLAG_GETSET,
            units='',
            type=types.IntType)
        self.add_parameter('filter_moving',
            flags=Instrument.FLAG_GETSET,
            units='',
            type=types.IntType)

        # Add functions to wrapper
        
        self.add_function('reset')
        self.add_function('get_all')

        self.add_function('read')

        self.add_function('send_init')
        self.add_function('fetch_last')

        #self.add_function('send_trigger')
        #self.add_function('fetch')

        # Connect to measurement flow to detect start and stop of measurement
        qt.flow.connect('measurement-start', self._measurement_start_cb)
        qt.flow.connect('measurement-end', self._measurement_end_cb)

        if reset:
            self.reset()
        else:
            self.get_all()
            #self.set_defaults()

# --------------------------------------
#           functions
# --------------------------------------

    def reset(self):
        '''
        Resets instrument to default values

        Input:
            None

        Output:
            None
        '''
        logging.debug('Resetting instrument')
        self._visainstrument.write('*RST')
        self.get_all()

    def set_defaults(self):
        '''
        Set to driver defaults:

        Sense voltage and source current
        Output=data only
        source_mode=curr
        sense_mode=volt
        Digits=7
        NPLC=10
        Averaging=off
        source current compliance = 1 uA
        source voltage compliance = 1 mV

        '''

        self._visainstrument.write('SYST:PRES')
#        self._visainstrument.write(':FORM:ELEM READ')
            # Sets the format to only the read out, all options are:
            # READing = DMM reading, UNITs = Units,
            # TSTamp = Timestamp, RNUMber = Reading number,
            # CHANnel = Channel number, LIMits = Limits reading
        self.set_source_mode('CURR')
        self.set_sense_mode('VOLT:DC,CURR:DC')
        #self.set_trigger_cont(True)
        self.set_source_current_range(1e-6)
        self.set_source_current_level(0e-6)
        self.set_source_current_compliance(1e-6)
        self.set_source_voltage_range(200e-3)
        self.set_source_voltage_compliance(1e-3)
        self.set_digits(7)
        self.set_nplc(10)
#        self.set_averaging(False)


    def get_all(self):
        '''
        Reads all relevant parameters from instrument

        Input:
            None

        Output:
            None
        '''
        logging.info('Get all relevant data from device')
        self.get_output_on()
        self.get_output_auto_off()

        self.get_source_mode()
        self.get_source_current_compliance()
        self.get_source_voltage_compliance()
        self.get_source_current_compliance_tripped()
        self.get_source_voltage_compliance_tripped()
        self.get_source_current_range()
        self.get_source_voltage_range()
        self.get_source_current_level()
        self.get_source_voltage_level()
        self.get_source_delay_auto()
        self.get_source_delay()

        #self.get_trigger_inline()
        #self.get_trigger_count()
        #self.get_trigger_delay()
        #self.get_trigger_source()
        #self.get_trigger_timer()

        self.get_digits()
        self.get_nplc()
#        self.get_display()
        self.get_autozero()

        self.get_sense_mode()

        self.get_sense_autorange()
        self.get_sense_current_range()
        self.get_sense_voltage_range()
        self.get_sense_resistance_range()

        self.get_sense_resistance_ocomp()

        self.get_filter_auto()
        self.get_filter_repeat_enabled()
        self.get_filter_median_enabled()
        self.get_filter_moving_enabled()
        self.get_filter_repeat()
        self.get_filter_median()
        self.get_filter_moving()

        self.get_trigger_source()
        self.get_arm_source()

        self.get_trigger_count()
        self.get_arm_count()

# Link old read and readlast to new routines:
    # Parameters are for states of the machnine and functions
    # for non-states. In principle the reading of the Keithley is not
    # a state (it's just a value at a point in time) so it should be a
    # function, technically. The GUI, however, requires an parameter to
    # read it out properly, so the reading is now done as if it is a
    # parameter, and the old functions are redirected.

    def read(self):
        '''
        Arm, trigger, and readout (voltage (V), current (A), resistance (Ohm)).

        Note that the values may not be valid if sense mode doesn't include them.
        '''
        if not (self.get_output_on(query=False) or self.get_output_auto_off(query=False)):
            raise Exception('Either source must be turned on manually or auto_off has to be enables before calling read().')
        s = self._visainstrument.ask(':READ?')
        logging.debug('Read: %s' % s)
        
        return [float(n) for n in s.split(',')][:3] # We don't know what [3:5] are!!!

    def send_init(self):
        '''
        Go into the arm/trigger layers from the idle mode.
        '''
        logging.debug('Init.')
        self._visainstrument.write(':INIT')


    def fetch_last(self):
        '''
        Fetch the last measured value. Typically used after send_init.

        Note that the values may not be valid if sense mode doesn't include them.
        '''
        s = self._visainstrument.ask(':FETC?')
        logging.debug('Read: %s' % s)
        
        return [float(n) for n in s.split(',')][-5:-2] # We don't know what [-2:] are!!!

    # def readlast(self):
    #     '''
    #     Old function for read-out, links to get_readlastval()
    #     '''
    #     logging.debug('Link to get_readlastval()')
    #     return self.get_readlastval()

    # def readnext(self):
    #     '''
    #     Links to get_readnextval
    #     '''
    #     logging.debug('Link to get_readnextval()')
    #     return self.get_readnextval()

    # def send_trigger(self):
    #     '''
    #     Send trigger to Keithley, use when triggering is not continous.
    #     '''
    #     trigger_status = self.__is_arming_and_triggering_cont()
    #     if (trigger_status):
    #         logging.warning('Trigger is set to continous, sending trigger impossible')
    #     elif (not trigger_status):
    #         logging.debug('Sending trigger')
    #         self._visainstrument.write('INIT')
    #         self._trigger_sent = True
    #     else:
    #         logging.error('Error in retrieving triggering status, no trigger sent.')

    # def fetch(self):
    #     '''
    #     Get data at this instance, not recommended, use get_readlastval.
    #     Use send_trigger() to trigger the device.
    #     Note that Readval is not updated since this triggers itself.
    #     '''

    #     trigger_status = self.__is_arming_and_triggering_cont()
    #     if self._trigger_sent and (not trigger_status):
    #         logging.debug('Fetching data')
    #         reply = self._visainstrument.ask('FETCH?')
    #         self._trigger_sent = False
    #         return float(reply[0:15])
    #     elif (not self._trigger_sent) and (not trigger_status):
    #         logging.warning('No trigger sent, use send_trigger')
    #     else:
    #         logging.error('Triggering is on continous!')



    def set_trigger_cont(self):
        '''
        Set trigger and arm modes to immediate.

        Input:
            None

        Output:
            None
        '''
        self.set_trigger_source('IMM')
        self.set_arm_source('IMM')


    # def set_trigger_disc(self):
    #     '''
    #     Set trigger mode to BUS

    #     Input:
    #         None

    #     Output:
    #         None
    #     '''
    #     logging.debug('Set Trigger to discrete mode')
    #     self.set_trigger_inline('BUS')


    # def reset_trigger(self):
    #     '''
    #     Reset trigger status

    #     Input:
    #         None

    #     Output:
    #         None
    #     '''
    #     logging.debug('Resetting trigger')
    #     self._visainstrument.write(':ABOR')


# --------------------------------------
#           parameters
# --------------------------------------

    # def do_get_readnextval(self):
    #     '''
    #     Waits for the next value available and returns it as a float.
    #     Note that if the reading is triggered manually, a trigger must
    #     be send first to avoid a time-out.

    #     Input:
    #         None

    #     Output:
    #         value(float) : last triggerd value on input
    #     '''
    #     logging.debug('Read next value')

    #     trigger_status = self.__is_arming_and_triggering_cont()
    #     if (not trigger_status) and (not self._trigger_sent):
    #         logging.error('No trigger has been send, return 0')
    #         return float(0)
    #     self._trigger_sent = False

    #     text = self._visainstrument.ask(':DATA:FRESH?')
    #         # Changed the query to from Data?
    #         # to Data:FRESH? so it will actually wait for the
    #         # measurement to finish.
    #     return float(text[0:15])

    # def do_get_readlastval(self):
    #     '''
    #     Returns the last measured value available and returns it as a float.
    #     Note that if this command is send twice in one integration time it will
    #     return the same value.

    #     Example:
    #     If continually triggering at 1 PLC, don't use the command within 1 PLC
    #     again, but wait 20 ms. If you want the Keithley to wait for a new
    #     measurement, use get_readnextval.

    #     Input:
    #         None

    #     Output:
    #         value(float) : last triggerd value on input
    #     '''
    #     logging.debug('Read last value')

    #     text = self._visainstrument.ask(':DATA?')
    #     return float(text[0:15])

    # def do_get_readval(self, ignore_error=False):
    #     '''
    #     Aborts current trigger and sends a new trigger
    #     to the device and reads float value.
    #     Do not use when trigger mode is 'CONT'
    #     Instead use readlastval

    #     Input:
    #         ignore_error (boolean): Ignore trigger errors, default is 'False'

    #     Output:
    #         value(float) : currrent value on input
    #     '''
    #     trigger_status = self.__is_arming_and_triggering_cont()
    #     if trigger_status:
    #         if ignore_error:
    #             logging.debug('Trigger=continuous, can\'t trigger, return 0')
    #         else:
    #             logging.error('Trigger=continuous, can\'t trigger, return 0')
    #         text = '0'
    #         return float(text[0:15])
    #     elif not trigger_status:
    #         logging.debug('Read current value')
    #         text = self._visainstrument.ask('READ?')
    #         self._trigger_sent = False
    #         return float(text[0:15])
    #     else:
    #         logging.error('Error in retrieving triggering status, no trigger sent.')


    def do_set_source_current_compliance(self, val):
        '''
        Set source_current compliance to the specified value.

        Input:
            val (float)   : source_current_compliance in amps

        Output:
            None
        '''
        logging.debug('Set source_current_compliance to %s' % val)
        self._visainstrument.write('SENS:CURR:PROT %s' % str(val).upper())


    def do_get_source_current_compliance(self):
        '''
        Get source_current_level.

        Output:
            source_current_compliance (float) : source_current_compliance in amps
        '''
        r = self._visainstrument.ask('SENS:CURR:PROT?')
        logging.debug('Get source_current_compliance: %s' % r)
        return float(r)

    def do_get_source_current_compliance_tripped(self):
        '''
        Ask whether the voltage is limited by the current compliance.

        Output:
            tripped (boolean) : whether compliance limited
        '''
        r = self._visainstrument.ask('SENS:CURR:PROT:TRIP?')
        logging.debug('Get source_current_compliance_tripped: %s' % r)
        return bool(int(r))

    def do_set_source_current_range(self, val):
        '''
        Set source_current range to the specified value.

        Input:
            val (float)   : source_current_range in amps

        Output:
            None
        '''
        logging.debug('Set source_current_level to %s' % val)
        self._visainstrument.write('SOUR:CURR:RANG %.3e' % val)

    def do_get_source_current_range(self):
        '''
        Get source range for the current mode.
        '''
        r = self._visainstrument.ask('SOUR:CURR:RANG?')
        logging.debug('Get source_current_level: %s' % r)
        return float(r)
        return bool(int(r))

    def do_set_source_voltage_range(self, val):
        '''
        Set source_voltage range to the specified value.

        Input:
            val (float)   : source_voltage_range in amps

        Output:
            None
        '''
        logging.debug('Set source_voltage_level to %s' % val)
        self._visainstrument.write('SOUR:VOLT:RANG %.3e' % val)

    def do_get_source_voltage_range(self):
        '''
        Get source range for the voltage mode.
        '''
        r = self._visainstrument.ask('SOUR:VOLT:RANG?')
        logging.debug('Get source_voltage_level: %s' % r)
        return float(r)
    def do_set_source_current_level(self, val):
        '''
        Set source_current level to the specified value for the
        designated mode. If mode=None, the current mode is assumed

        Input:
            val (float)   : source_current_level in amps
            mode (string) : mode to set property for. Choose from self._modes

        Output:
            None
        '''
        logging.debug('Set source_current_level to %s' % val)
        self._visainstrument.write('SOUR:CURR:LEV %s' % str(val).upper())


    def do_get_source_current_level(self):
        '''
        Get source_current_level for the specified mode.
        If mode=None, the current mode is assumed.

        Input:
            mode (string) : mode to set property for. Choose from self._modes

        Output:
            source_current_compliance (float) : source_current_compliance in amps
        '''
        r = self._visainstrument.ask('SOUR:CURR:LEV?')
        logging.debug('Get source_current_level: %s' % r)
        return float(r)


    def do_set_source_voltage_level(self, val):
        '''
        Set source_voltage_level for the designated mode.

        Input:
            val (float)   : source_current_compliance in amps

        Output:
            None
        '''
        logging.debug('Set source_current_compliance to %s' % val)
        self._visainstrument.write('SOUR:VOLT:LEV %s' % str(val).upper())

    def do_get_source_voltage_level(self):
        '''
        Get source_voltage_level.

        Input:
            none

        Output:
            source_voltage_level (float) : source_voltage_level in volts
        '''
        r = self._visainstrument.ask('SOUR:VOLT:LEV?')
        logging.debug('Get source_current_level: %s' % r)
        return float(r)

    def do_set_source_voltage_compliance(self, val):
        '''
        Set source_voltage_compliance to the specified value.

        Input:
            val (float)   : source_voltage_compliance in volts

        Output:
            None
        '''
        logging.debug('Set source_voltage_compliance to %s' % val)
        self._visainstrument.write('SENS:VOLT:PROT %s' % str(val).upper())

    def do_get_source_voltage_compliance(self):
        '''
        Get source_voltage_compliance.

        Input:
            none

        Output:
            source_voltage_compliance (float) : source_voltage_compliance in volts
        '''
        r = self._visainstrument.ask('SENS:VOLT:PROT?')
        logging.debug('Get source_voltage_compliance: %s' % r)
        return float(r)

    def do_get_source_voltage_compliance_tripped(self):
        '''
        Ask whether the current is limited by the voltage compliance.

        Output:
            tripped (boolean) : whether compliance limited
        '''
        r = self._visainstrument.ask('SENS:VOLT:PROT:TRIP?')
        logging.debug('Get source_voltage_compliance_tripped: %s' % r)
        return bool(int(r))

    def do_set_output_on(self, val):
        '''
        Set source output on/off.

        Input:
            val (boolean) : on/off

        Output:
            None
        '''
        #val = bool_to_str(val)
        logging.debug('Set output_on to %s' % val)
        self._visainstrument.write('OUTP %s' % int(val))

    def do_get_output_on(self):
        '''
        Ask whether the output is on/off

        Input:
            None

        Output:
            val (bool) : returns true if output is on
        '''
        r = self._visainstrument.ask('OUTP?')
        logging.debug('Read output_on from instrument: %s' % r)
        return bool(int(r))

    def do_set_output_auto_off(self, val):
        '''
        Set source output auto-off on/off.

        Input:
            val (boolean) : on/off

        Output:
            None
        '''
        #val = bool_to_str(val)
        logging.debug('Set output_auto_off to %s' % val)
        self._visainstrument.write(':SOUR:CLE:AUTO %s' % int(val))

    def do_get_output_auto_off(self):
        '''
        Ask whether output auto-off is on/off

        Input:
            None

        Output:
            val (bool) : returns true if output is on
        '''
        r = self._visainstrument.ask('OUTP?')
        logging.debug('Read output_auto_off from instrument: %s' % r)
        return bool(int(r))

    def do_set_source_delay_auto(self, val):
        '''
        Set source auto delay on/off.

        Input:
            val (boolean) : on/off

        Output:
            None
        '''
        val = bool_to_str(val)
        logging.debug('Set source auto delay to %s' % val)
        self._visainstrument.write(':SOUR:DEL:AUTO %s' % val)

    def do_get_source_delay_auto(self):
        '''
        Ask whether the auto delay is on/off

        Input:
            None

        Output:
            val (bool) : returns true if auto delay is on
        '''
        r = self._visainstrument.ask(':SOUR:DEL:AUTO?')
        logging.debug('Read auto delay from instrument: %s' % r)
        return bool(int(r))

    def do_set_sense_resistance_ocomp(self, val):
        '''
        Set offset compensation on/off for resistance measurements.

        Input:
            val (boolean) : on/off

        Output:
            None
        '''
        val = bool_to_str(val)
        logging.debug('Set resistance measurement offset compensation to %s' % val)
        self._visainstrument.write(':SENS:RES:OCOM %s' % val)

    def do_get_sense_resistance_ocomp(self):
        '''
        Ask whether offset compensation is on/off for resistance measurements.

        Input:
            None

        Output:
            val (bool) : returns true if offset compensation is on
        '''
        r = self._visainstrument.ask(':SENS:RES:OCOM?')
        logging.debug('Read auto delay from instrument: %s' % r)
        return bool(int(r))

    def do_set_source_delay(self, val):
        '''
        Set source delay.

        Input:
            val (boolean) : on/off

        Output:
            None
        '''
        logging.debug('Set source delay to %s' % val)
        self._visainstrument.write(':SOUR:DEL %s' % val)

    def do_get_source_delay(self):
        '''
        Returns the delay between turning source on and measuring.

        Input:
            None

        Output:
            val (float) : delay in seconds
        '''
        r = self._visainstrument.ask(':SOUR:DEL?')
        logging.debug('Read delay from instrument: %s' % r)
        return float(r)

    def do_set_digits(self, val):
        '''
        Set digits to the specified value.

        Input:
            val (int)     : Number of digits = 4, 5, 6, or 7

        Output:
            None
        '''
        logging.debug('Set digits to %s' % val)
        self._visainstrument.write('DISP:DIG %s' % val)

    def do_get_digits(self):
        '''
        Get digits.

        Input:

        Output:
            digits (int) : Number of digits
        '''
        r = self._visainstrument.ask('DISP:DIG?')
        logging.debug('Getting digits: %s' % r)
        return int(r)

    def do_set_nplc(self, val):
        '''
        Set integration time to the specified value in Number of Powerline Cycles.

        Input:
            val (float)   : Integration time in nplc.

        Output:
            None
        '''
        logging.debug('Set NPLC to %s PLC' % val)
        self._visainstrument.write(':NPLC %s' % val)

    def do_get_nplc(self):
        '''
        Get integration time in Number of PowerLine Cycles.

        Input:
            None

        Output:
            time (float) : Integration time in PLCs
        '''
        logging.debug('Read integration time in PLCs')
        return float(self._visainstrument.ask(':NPLC?'))

    def do_set_trigger_source(self, val):
        '''
        Set the trigger source.

        Input:
            val (string)   : IMM or TLIN

        Output:
            None
        '''
        logging.debug('Set trigger source to %s' % val)
        self._visainstrument.write(':TRIG:SOUR %s' % val)

    def do_get_trigger_source(self):
        '''
        Get the trigger source.

        Input:
            None

        Output:
            src (string) : IMM or TLIN
        '''
        r = self._visainstrument.ask(':TRIG:SOUR?')
        logging.debug('Get the trigger source: %s' % r)
        return r

    def do_set_arm_source(self, val):
        '''
        Set the arm source.

        Input:
            val (string)   : IMM or TLIN

        Output:
            None
        '''
        logging.debug('Set arm source to %s' % val)
        self._visainstrument.write(':ARM:SOUR %s' % val)

    def do_get_arm_source(self):
        '''
        Get the arm source.

        Input:
            None

        Output:
            src (string) : IMM or TLIN
        '''
        r = self._visainstrument.ask(':ARM:SOUR?')
        logging.debug('Get the arm source: %s' % r)
        return r


    def do_set_trigger_count(self, val):
        '''
        Set the triggers to execute after initiating a measurement.

        Input:
            val (int)   :  no. of triggers

        Output:
            None
        '''
        logging.debug('Set trigger count to %s' % val)
        self._visainstrument.write(':TRIG:COUN %s' % val)

    def do_get_trigger_count(self):
        '''
        Get the triggers to execute after initiating a measurement.

        Input:
            None

        Output:
            number of triggers (int)
        '''
        r = self._visainstrument.ask(':TRIG:COUN?')
        logging.debug('Read trigger count: %s' % r)
        return int(r)

    def do_set_arm_count(self, val):
        '''
        Set the arms to execute after initiating a measurement.

        Input:
            val (int)   :  no. of arms

        Output:
            None
        '''
        logging.debug('Set arm count to %s' % val)
        self._visainstrument.write(':ARM:COUN %s' % val)

    def do_get_arm_count(self):
        '''
        Get the arms to execute after initiating a measurement.

        Input:
            None

        Output:
            number of arms (int)
        '''
        r = self._visainstrument.ask(':ARM:COUN?')
        logging.debug('Read arm count: %s' % r)
        return int(r)



    # def do_set_trigger_inline(self, val):
    #     '''
    #     Set trigger input line.

    #     Input:
    #         val (string) : 'IMM', 'BUS'

    #     Output:
    #         None
    #     '''
    #     val = bool_to_str(val)
    #     logging.debug('Set trigger mode to %s' % val)
    #     self._visainstrument.write('TRIG:ILIN %s' % val)

    # def do_get_trigger_inline(self):
    #     '''
    #     Get trigger mode from instrument

    #     Input:
    #         None

    #     Output:
    #         val (bool) : returns if triggering is continuous.
    #     '''
    #     logging.debug('Read trigger mode from instrument')
    #     return bool(int(self._visainstrument.ask('TRIG:CONT?')))

    # def do_set_trigger_count(self, val):
    #     '''
    #     Set trigger count
    #     if val>9999 count is set to INF

    #     Input:
    #         val (int) : trigger count

    #     Output:
    #         None
    #     '''
    #     logging.debug('Set trigger count to %s' % val)
    #     if val > 9999:
    #         val = 'INF'
    #     self._visainstrument.write('TRIG:COUN %s' % val)

    # def do_get_trigger_count(self):
    #     '''
    #     Get trigger count

    #     Input:
    #         None

    #     Output:
    #         count (int) : Trigger count
    #     '''
    #     logging.debug('Read trigger count from instrument')
    #     ans = self._visainstrument.ask('TRIG:COUN?')
    #     try:
    #         ret = int(ans)
    #     except:
    #         ret = 0

    #     return ret

    # def do_set_trigger_delay(self, val):
    #     '''
    #     Set trigger delay to the specified value

    #     Input:
    #         val (float) : Trigger delay in seconds

    #     Output:
    #         None
    #     '''
    #     logging.debug('Set trigger delay to %s' % val)
    #     self._visainstrument.write('TRIG:DEL %s' % val)

    # def do_get_trigger_delay(self):
    #     '''
    #     Read trigger delay from instrument

    #     Input:
    #         None

    #     Output:
    #         delay (float) : Delay in seconds
    #     '''
    #     logging.debug('Get trigger delay')
    #     return float(self._visainstrument.ask('TRIG:DEL?'))

    # def do_set_trigger_source(self, val):
    #     '''
    #     Set trigger source

    #     Input:
    #         val (string) : Trigger source

    #     Output:
    #         None
    #     '''
    #     logging.debug('Set Trigger source to %s' % val)
    #     self._visainstrument.write('TRIG:SOUR %s' % val)

    # def do_get_trigger_source(self):
    #     '''
    #     Read trigger source from instrument

    #     Input:
    #         None

    #     Output:
    #         source (string) : The trigger source
    #     '''
    #     logging.debug('Getting trigger source')
    #     return self._visainstrument.ask('TRIG:SOUR?')

    # def do_set_trigger_timer(self, val):
    #     '''
    #     Set the trigger timer

    #     Input:
    #         val (float) : the value to be set

    #     Output:
    #         None
    #     '''
    #     logging.debug('Set trigger timer to %s' % val)
    #     self._visainstrument.write('TRIG:TIM %s' % val)

    # def do_get_trigger_timer(self):
    #     '''
    #     Read the value for the trigger timer from the instrument

    #     Input:
    #         None

    #     Output:
    #         timer (float) : Value of timer
    #     '''
    #     logging.debug('Get trigger timer')
    #     return float(self._visainstrument.ask('TRIG:TIM?'))

    def do_set_sense_mode(self, mode):
        '''
        Set the sense_mode to the specified value

        Input:
            mode (string) : mode(s) to be set. Choose from self._sense_modes.
                            Use comma to separate multiple modes.

        Output:
            None
        '''

        mode = [ m.strip(' ') for m in mode.split(',') ]

        if not all([ m in self._sense_modes for m in mode ]):
            raise Exception('invalid sense_mode %s' % mode)
                
        mode = '"' + '","'.join(mode) + '"'

        logging.debug('Set sense_mode to %s', mode)

        string = ':SENS:FUNC %s' % mode

        logging.debug('%s', string)
        self._visainstrument.write(':SENS:FUNC:OFF:ALL')
        self._visainstrument.write(string)

        #self.get_all()
        # Get all values again because some paramaters depend on sense_mode

    def do_get_sense_mode(self):
        '''
        Read the sense_mode from the device

        Input:
            None

        Output:
            mode (string) : Current mode
        '''
        string = 'SENS:FUNC?'
        ans = self._visainstrument.ask(string).replace('"','')
        logging.debug('Getting sense_mode: %s' % ans)
        return ans

    def do_set_source_mode(self, mode):
        '''
        Set the source_mode to the specified value

        Input:
            mode (string) : mode to be set. Choose from self._source_modes

        Output:
            None
        '''

        logging.debug('Set source_mode to %s', mode)
        if mode in self._source_modes:
            string = 'SOUR:FUNC %s' % mode
            self._visainstrument.write(string)

        else:
            logging.error('invalid source_mode %s' % source_mode)

    def do_get_source_mode(self):
        '''
        Read the source_mode from the device

        Input:
            None

        Output:
            mode (string) : Current mode
        '''
        string = 'SOUR:FUNC?'
        ans = self._visainstrument.ask(string).strip('"')
        logging.debug('Getting source_mode: %s' % ans)
        return ans

    # def do_get_display(self):
    #     '''
    #     Read the staturs of diplay

    #     Input:
    #         None

    #     Output:
    #         True = On
    #         False= Off
    #     '''
    #     logging.debug('Reading display from instrument')
    #     reply = self._visainstrument.ask('DISP:ENAB?')
    #     return bool(int(reply))

    # def do_set_display(self, val):
    #     '''
    #     Switch the diplay on or off.

    #     Input:
    #         val (boolean) : True for display on and False for display off

    #     Output

    #     '''
    #     logging.debug('Set display to %s' % val)
    #     val = bool_to_str(val)
    #     return self._visainstrument.write('DISP:ENAB %s' % val)

    def do_get_autozero(self):
        '''
        Read the staturs of the autozero function

        Input:
            None

        Output:
            reply (boolean) : Autozero status.
        '''
        reply = self._visainstrument.ask('SYST:AZER:STAT?')
        logging.debug('Reading autozero status from instrument: %s' % reply)
        return bool(int(reply))

    def do_set_autozero(self, val):
        '''
        Switch the diplay on or off.

        Input:
            val (boolean) : True for display on and False for display off

        Output

        '''
        logging.debug('Set autozero to %s' % val)
        val = bool_to_str(val)
        return self._visainstrument.write('SYST:AZER:STAT %s' % val)

    def do_set_sense_autorange(self, val):
        '''
        Switch sense_autorange on or off for all modes.

        Input:
            val (boolean)

        Output:
            None
        '''
        logging.debug('Set sense_autorange to %s ' % val)
        val = bool_to_str(val)
        self._visainstrument.write('SENS:CURR:RANG:AUTO %s' % val)
        self._visainstrument.write('SENS:VOLT:RANG:AUTO %s' % val)
        self._visainstrument.write('SENS:RES:RANG:AUTO %s' % val)

    def do_get_sense_autorange(self):
        '''
        Get status of sense_autorange. Returns true iff true for all modes

        Input:

        Output:
            result (boolean)
        '''
        reply0 = bool(int(self._visainstrument.ask('SENS:CURR:RANG:AUTO?')))
        reply1 = bool(int(self._visainstrument.ask('SENS:VOLT:RANG:AUTO?')))
        reply2 = bool(int(self._visainstrument.ask('SENS:RES:RANG:AUTO?')))
        logging.debug('Get sense_autorange: %s %s %s' % (reply0, reply1, reply2))
        return reply0 and reply1 and reply2

    def do_set_sense_current_range(self, val):
        '''
        Set the range for current measurements, in amps.

        Input:
            val (float)   :  1e-1, ..., 1e-6,  (w/o remote amp)
                             1e-7, ..., 1e-12  (w/  remote amp)

        Output:
            None
        '''
        logging.debug('Set sense_current_range to %s' % val)
        self._visainstrument.write(':SENS:CURR:RANG %s' % val)

    def do_get_sense_current_range(self):
        '''
        Get the range for current measurements, in amps.

        Input:
            None

        Output:
            range in amps (float)
        '''
        r = self._visainstrument.ask(':SENS:CURR:RANG?')
        logging.debug('Read sense_current_range: %s, %s' % (r, float(r)))
        return float(r)

    def do_set_sense_voltage_range(self, val):
        '''
        Set the range for voltage measurements, in volts.

        Input:
            val (float)   :  200, 20, 2, 0.2

        Output:
            None
        '''
        logging.debug('Set sense_voltage_range to %s' % val)
        self._visainstrument.write(':SENS:VOLT:RANG %s' % val)

    def do_get_sense_voltage_range(self):
        '''
        Get the range for voltage measurements, in volts.

        Input:
            None

        Output:
            range in volts (float)
        '''
        r = self._visainstrument.ask(':SENS:VOLT:RANG?')
        logging.debug('Read sense_voltage_range: %s' % r)
        return float(r)

    def do_set_sense_resistance_range(self, val):
        '''
        Set the range for resistance measurements, in Ohms.

        Input:
            val (float)   :  2e0, ..., 2e7,  (w/o remote amp)
                             2e8, ..., 2e13  (w/  remote amp)

        Output:
            None
        '''
        logging.debug('Set sense_resistance_range to %s' % val)
        self._visainstrument.write(':SENS:RES:RANG %s' % val)

    def do_get_sense_resistance_range(self):
        '''
        Get the range for resistance measurements, in Ohms.

        Input:
            None

        Output:
            range in Ohms (float)
        '''
        r = self._visainstrument.ask(':SENS:RES:RANG?')
        logging.debug('Read sense_resistance_range: %s' % r)
        return float(r)

    def do_set_filter_auto(self, val):
        '''
        Switch filter_auto on/off.

        Input:
            val (boolean)

        Output:
            None
        '''
        logging.debug('Set filter_auto to %s' % val)
        val = bool_to_str(val)
        self._visainstrument.write('AVER:AUTO %s' % val)

    def do_get_filter_auto(self):
        '''
        Get status of filter_auto.

        Input:

        Output:
            result (boolean)
        '''
        reply0 = self._visainstrument.ask('AVER:AUTO?')
        logging.debug('Get filter_auto: %s' % reply0)
        return bool(int(reply0))

    def do_set_filter_repeat_enabled(self, val):
        '''
        Switch filter_repeat_enabled on/off.

        Input:
            val (boolean)

        Output:
            None
        '''
        logging.debug('Set filter_repeat_enabled to %s ' % val)
        val = bool_to_str(val)
        self._visainstrument.write(':AVER:REP:STAT %s' % val)

    def do_get_filter_repeat_enabled(self):
        '''
        Get status of filter_repeat_enabled.

        Input:

        Output:
            result (boolean)
        '''
        reply0 = self._visainstrument.ask(':AVER:REP:STAT?')
        logging.debug('Get filter_repeat_enabled: %s' % reply0)
        return bool(int(reply0))

    def do_set_filter_median_enabled(self, val):
        '''
        Switch filter_median_enabled on/off.

        Input:
            val (boolean)

        Output:
            None
        '''
        logging.debug('Set filter_median_enabled to %s ' % val)
        val = bool_to_str(val)
        self._visainstrument.write(':MED:STAT %s' % val)

    def do_get_filter_median_enabled(self):
        '''
        Get status of filter_median_enabled.

        Input:

        Output:
            result (boolean)
        '''
        reply0 = self._visainstrument.ask(':MED:STAT?')
        logging.debug('Get filter_median_enabled: %s' % reply0)
        return bool(int(reply0))

    def do_set_filter_moving_enabled(self, val):
        '''
        Switch filter_moving_enabled on/off.

        Input:
            val (boolean)

        Output:
            None
        '''
        logging.debug('Set filter_moving_enabled to %s ' % val)
        val = bool_to_str(val)
        self._visainstrument.write(':AVER:STAT %s' % val)

    def do_get_filter_moving_enabled(self):
        '''
        Get status of filter_moving_enabled.

        Input:

        Output:
            result (boolean)
        '''
        reply0 = self._visainstrument.ask(':AVER:STAT?')
        logging.debug('Get filter_moving_enabled: %s' % reply0)
        return bool(int(reply0))

    def do_set_filter_repeat(self, val):
        '''
        Set the number of samples that are averaged before passing
        the value to the median filter.

        Input:
            val (int)   :  no. of samples to average

        Output:
            None
        '''
        logging.debug('Set filter_repeat to %s' % val)
        self._visainstrument.write(':AVER:REP:COUN %s' % val)

    def do_get_filter_repeat(self):
        '''
        Get the number of samples that are averaged before passing
        the value to the median filter.

        Input:
            None

        Output:
            number of samples (int)
        '''
        r = self._visainstrument.ask(':AVER:REP:COUN?')
        logging.debug('Read filter_repeat: %s' % r)
        return int(r)

    def do_set_filter_median(self, val):
        '''
        Set the number of samples that are averaged before passing
        the value to the moving filter.

        Input:
            val (int)   :  no. of samples to average

        Output:
            None
        '''
        logging.debug('Set filter_median to %s' % val)
        self._visainstrument.write(':MED:RANK %s' % val)

    def do_get_filter_median(self):
        '''
        Get the number of samples that are averaged before passing
        the value to the median filter.

        Input:
            None

        Output:
            number of samples (int)
        '''
        r = self._visainstrument.ask(':MED:RANK?')
        logging.debug('Read filter_median: %s' % r)
        return int(r)

    def do_set_filter_moving(self, val):
        '''
        Set the number of samples that are averaged before passing
        the value to the median filter.

        Input:
            val (int)   :  no. of samples to average

        Output:
            None
        '''
        logging.debug('Set filter_moving to %s' % val)
        self._visainstrument.write(':AVER:COUN %s' % val)

    def do_get_filter_moving(self):
        '''
        Get the number of samples that are averaged before passing
        the value to the median filter.

        Input:
            None

        Output:
            number of samples (int)
        '''
        r = self._visainstrument.ask(':AVER:COUN?')
        logging.debug('Read filter_moving: %s' % r)
        return int(r)

# --------------------------------------
#           Internal Routines
# --------------------------------------

    # def _is_arming_and_triggering_cont(self):
    #     return (self.get_trigger_inline(query=False)[:3] == 'IMM') and (self.get_arm_inline(query=False)[:3] == 'IMM')


    def _measurement_start_cb(self, sender):
        '''
        Things to do at starting of measurement
        '''
        # if self._change_display:
        #     self.set_display(False)
        #     #Switch off display to get stable timing
        # if self._change_autozero:
        #     self.set_autozero(False)
        #     #Switch off autozero to speed up measurement

    def _measurement_end_cb(self, sender):
        '''
        Things to do after the measurement
        '''
        # if self._change_display:
        #     self.set_display(True)
        # if self._change_autozero:
        #     self.set_autozero(True)

