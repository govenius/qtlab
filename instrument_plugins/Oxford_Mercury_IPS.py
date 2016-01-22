# Oxford_Mercury_IPS.py class, to perform the communication between the Wrapper and the device
# Guenevere Prawiroatmodjo <guen@vvtp.tudelft.nl>, 2009
# Pieter de Groot <pieterdegroot@gmail.com>, 2009
# Joonas Govenius <joonas.govenius@aalto.fi>, 2012
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
from time import time
import qt
import visa
import types
import logging
import re

class Oxford_Mercury_IPS(Instrument):
    '''
    This is the python driver for the Oxford Instruments IPS Magnet Power Supply

    Choose remote --> ethernet --> SCPI on the device.
    The VISA address should then look something like this: TCPIP0::<IP address>::7020::SOCKET
    '''

    def __init__(self, name, address):
        '''
        Initializes the Oxford Instruments IPS 120 Magnet Power Supply.

        Input:
            name (string)    : name of the instrument
            address (string) : instrument address [TCPIP0::<IP address>::7020::SOCKET]

        Output:
            None
        '''
        logging.debug(__name__ + ' : Initializing instrument')
        Instrument.__init__(self, name, tags=['physical'])


        self._address = address
        self._visainstrument = visa.ResourceManager().open_resource(self._address, timeout=2000)
        try:
          self._visainstrument.read_termination = '\n'
          self._visainstrument.write_termination = '\n'

          self._values = {}

          #Add parameters
          self.add_parameter('idn', type=types.StringType, flags=Instrument.FLAG_GET, format='%.10s')

          self.add_parameter('activity', type=types.StringType,
              flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET,
              format_map = {
              "HOLD" : "Hold",
              "RTOS" : "To set point",
              "RTOZ" : "To zero",
              "CLMP" : "Clamped"})

          self.add_parameter('current_setpoint', type=types.FloatType, units='A',
               flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET,
               minval=-126., maxval=126.)

          self.add_parameter('current', type=types.FloatType, units='A',
               flags=Instrument.FLAG_GET)

          self.add_parameter('persistent_current', type=types.FloatType, units='A',
               flags=Instrument.FLAG_GET)

          self.add_parameter('voltage', type=types.FloatType, units='V',
               flags=Instrument.FLAG_GET)

          self.add_parameter('ramp_rate_setpoint', type=types.FloatType, units='A/min',
               flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET,
               minval=0., maxval=1200.)

          self.add_parameter('switch_heater_on', type=types.BooleanType,
               flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET)

          # Add functions
          self.add_function('get_all')
          self.get_all()

        except:
          self._visainstrument.close()
          raise

    def get_all(self):
        '''
        Reads all implemented parameters from the instrument,
        and updates the wrapper.

        Input:
            None

        Output:
            None
        '''
        logging.info(__name__ + ' : reading all settings from instrument')
        self.get_idn()
        self.get_voltage()
        self.get_current_setpoint()
        self.get_current()
        self.get_persistent_current()
        self.get_activity()
        self.get_ramp_rate_setpoint()
        self.get_switch_heater_on()


    # Functions
    def _query(self, message):
        '''
        Write a command to the device

        Input:
            message (str) : write command for the device

        Output:
            None
        '''
        logging.debug(__name__ + ' : Send the following command to the device: %s' % message)
        result = self._visainstrument.query('%s' % (message))
        qt.msleep(0.1)
        return result

    def do_get_idn(self):
        result = self._query('*IDN?').strip()
        return result[4:] if result.startswith('IDN:') else result

    def do_get_current_setpoint(self):
        '''
        Return the set point (target current)

        Input:
            None

        Output:
            result (float) : Target current in Amp
        '''
        result = self._query('READ:DEV:GRPZ:PSU:SIG:CSET')
        logging.debug(__name__ + ' : Read set point (target current): %s')

        m = re.match(r'STAT:DEV:GRPZ:PSU:SIG:CSET:(.+?)A', result)
        if m == None or len(m.groups()) != 1: raise Exception('Could not parse the reply: %s' % result)

        try:        
          return float(m.groups()[0])
        except Exception as e:
          raise e

    def do_set_current_setpoint(self, current):
        '''
        Set current setpoint (target current)
        Input:
            current (float) : target current in Amp

        Output:
            None
        '''
        logging.debug(__name__ + ' : Setting target current to %s' % current)
        cmd = 'SET:DEV:GRPZ:PSU:SIG:CSET:%s' % current
        result = self._query(cmd)
        
        # verify that the command was correctly parsed
        m = re.match(r'STAT:SET:DEV:GRPZ:PSU:SIG:CSET:(.+?):VALID', result)
        if m == None or len(m.groups()) != 1: raise Exception('The IPS did not acknowledge parsing %s correctly, instead got: %s' % (cmd, result))


    def do_get_current(self):
        '''
        Return the current.

        Input:
            None

        Output:
            result (float) : Current in amp.
        '''
        result = self._query('READ:DEV:GRPZ:PSU:SIG:CURR')
        logging.debug(__name__ + ' : Read current: %s')

        m = re.match(r'STAT:DEV:GRPZ:PSU:SIG:CURR:(.+?)A', result)
        if m == None or len(m.groups()) != 1: raise Exception('Could not parse the reply: %s' % result)

        try:        
          return float(m.groups()[0])
        except Exception as e:
          raise e

    def do_get_persistent_current(self):
        '''
        Return the persistent current.

        Input:
            None

        Output:
            result (float) : Current in amp.
        '''
        result = self._query('READ:DEV:GRPZ:PSU:SIG:PCUR')
        logging.debug(__name__ + ' : Read persistent current: %s')

        m = re.match(r'STAT:DEV:GRPZ:PSU:SIG:PCUR:(.+?)A', result)
        if m == None or len(m.groups()) != 1: raise Exception('Could not parse the reply: %s' % result)

        try:        
          return float(m.groups()[0])
        except Exception as e:
          raise e



    def do_get_activity(self):
        '''
        Return the action status  (activity).

        Input:
            None

        Output:
            result (float) : action status.
        '''
        result = self._query('READ:DEV:GRPZ:PSU:ACTN')
        logging.debug(__name__ + ' : Read activity: %s' % result)

        m = re.match(r'STAT:DEV:GRPZ:PSU:ACTN:(.+?)$', result)
        if m == None or len(m.groups()) != 1: raise Exception('Could not parse the reply: %s' % result)

        try:
          logging.debug("parsed '%s'" % (m.groups()[0]))

          return m.groups()[0]
        except Exception as e:
          raise e


    def do_set_activity(self, activity):
        '''
        Set action status:[HOLD | RTOS | RTOZ | CLMP]
        Input:
            status: [HOLD | RTOS | RTOZ | CLMP]

        Output:
            None
        '''
        logging.debug(__name__ + ' : Setting activity to %s' % activity)
        cmd = 'SET:DEV:GRPZ:PSU:ACTN:%s' % activity
        result = self._query(cmd)
        
        # verify that the command was correctly parsed
        m = re.match(r'STAT:SET:DEV:GRPZ:PSU:ACTN:(.+?):VALID', result)
        if m == None or len(m.groups()) != 1: raise Exception('The IPS did not acknowledge parsing %s correctly, instead got: %s' % (cmd, result))


    def do_set_ramp_rate_setpoint(self, ramp_rate):
        '''
        current rate to set (A/min)

        Output:
            None
        '''
        logging.debug(__name__ + ' : Setting current rate  current to %s' % ramp_rate)
        cmd = 'SET:DEV:GRPZ:PSU:SIG:RCST:%s' % ramp_rate
        result = self._query(cmd)
        
        # verify that the command was correctly parsed
        m = re.match(r'STAT:SET:DEV:GRPZ:PSU:SIG:RCST:(.+?):VALID', result)
        if m == None or len(m.groups()) != 1: raise Exception('The IPS did not acknowledge parsing %s correctly, instead got: %s' % (cmd, result))

    def do_get_ramp_rate_setpoint(self):
        '''
        Return the current rate set point

        Input:
            None

        Output:
            result (float) : current rate in A / min
        '''
        result = self._query('READ:DEV:GRPZ:PSU:SIG:RCST')
        logging.debug(__name__ + ' : Read current rate set point: %s')

        m = re.match(r'STAT:DEV:GRPZ:PSU:SIG:RCST:(.+?)A/m', result)
        if m == None or len(m.groups()) != 1: raise Exception('Could not parse the reply: %s' % result)

        try:        
          return float(m.groups()[0])
        except Exception as e:
          raise e


    def do_get_voltage(self):
        '''
        Return the voltage.

        Input:
            None

        Output:
            result (float) : Voltage in volts.
        '''
        result = self._query('READ:DEV:GRPZ:PSU:SIG:VOLT')
        logging.debug(__name__ + ' : Read voltage: %s')

        m = re.match(r'STAT:DEV:GRPZ:PSU:SIG:VOLT:(.+?)V', result)
        if m == None or len(m.groups()) != 1: raise Exception('Could not parse the reply: %s' % result)

        try:        
          return float(m.groups()[0])
        except Exception as e:
          raise e

    def do_get_switch_heater_on(self):
        '''
        Is the switch heater on.
        '''
        result = self._query('READ:DEV:GRPZ:PSU:SIG:SWHT')
        logging.debug(__name__ + ' : Read switch heater status: %s')

        m = re.match(r'STAT:DEV:GRPZ:PSU:SIG:SWHT:(OFF|ON)', result)
        if m == None or m.groups()[0].strip().upper() not in ['ON', 'OFF']:
            raise Exception('Could not parse the reply: %s' % result)

        try:
          return m.groups()[0] == 'ON'
        except Exception as e:
          raise e

    def do_set_switch_heater_on(self, val):
        '''
        Set the switch heater on/off.
        '''
        logging.debug(__name__ + ' : Setting switch heater status to %s' % val)
        cmd = 'SET:DEV:GRPZ:PSU:SIG:SWHN:%s' % ('ON' if val else 'OFF')
        result = self._query(cmd)

        # verify that the command was correctly parsed
        m = re.match(r'STAT:SET:DEV:GRPZ:PSU:SIG:SWHN:(.+?):VALID', result)
        if m == None or len(m.groups()) != 1: raise Exception('The IPS did not acknowledge parsing %s correctly, instead got: %s' % (cmd, result))
