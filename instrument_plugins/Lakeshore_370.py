# Lakeshore 370, Lakeshore 370 temperature controller driver
# Joonas Govenius <joonas.govenius@aalto.fi>, 2014
# Based on Lakeshore 340 driver by Reinier Heeres <reinier@heeres.eu>, 2010.
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
import re
import math
import time
import numpy as np
import qt
import os
import random
import hashlib
from lib.config import get_config
config = get_config()


class Lakeshore_370(Instrument):

    def __init__(self, name, address, reset=False, **kwargs):
        Instrument.__init__(self, name)

        self._address = address

        self._visainstrument = visa.ResourceManager().open_resource(self._address,
                                                                    timeout=5000.) # milliseconds
        try:
          if address.lower().startswith('asrl'):
            # These are for an RS-232 connection, values found in the LS manual
            self._visainstrument.parity = visa.constants.Parity.odd
            self._visainstrument.data_bits = 7
            self._visainstrument.stop_bits = visa.constants.StopBits.one
          self._visainstrument.read_termination = '\r\n'
          self._visainstrument.write_termination = '\n'

          self._channels = kwargs.get('channels', (1, 2, 5, 6))

          self._logger = kwargs.get('logger', None)

          self.add_parameter('identification',
              flags=Instrument.FLAG_GET)

          self.add_parameter('common_mode_reduction',
              flags=Instrument.FLAG_GET,
              type=types.BooleanType)

          self.add_parameter('guard_drive',
              flags=Instrument.FLAG_GET,
              type=types.BooleanType)

          self.add_parameter('scanner_auto',
              flags=Instrument.FLAG_GETSET,
              type=types.BooleanType)

          self.add_parameter('scanner_channel',
              flags=Instrument.FLAG_GETSET,
              type=types.IntType,
              format_map=dict(zip(self._channels,self._channels)))

          self.add_parameter('kelvin',
              flags=Instrument.FLAG_GET,
              type=types.FloatType,
              channels=self._channels,
              units='K')

          self.add_parameter('resistance',
              flags=Instrument.FLAG_GET,
              type=types.FloatType,
              channels=self._channels,
              units='Ohm')

          self.add_parameter('resistance_range',
              flags=Instrument.FLAG_GET,
              type=types.StringType,
              channels=self._channels,
              format_map={
                  1: '2 mOhm',
                  2: '6.32 mOhm',
                  3: '20 mOhm',
                  4: '63.2 mOhm',
                  5: '200 mOhm',
                  6: '632 mOhm',
                  7: '2 Ohm',
                  8: '6.32 Ohm',
                  9: '20 Ohm',
                  10: '63.2 Ohm',
                  11: '200 Ohm',
                  12: '632 Ohm',
                  13: '2 kOhm',
                  14: '6.32 kOhm',
                  15: '20 kOhm',
                  16: '63.2 kOhm',
                  17: '200 kOhm',
                  18: '632 kOhm',
                  19: '2 MOhm',
                  20: '6.32 MOhm',
                  21: '20 MOhm',
                  22: '63.2 MOhm'
                  })

          self.add_parameter('excitation_mode',
              flags=Instrument.FLAG_GET,
              type=types.IntType,
              channels=self._channels,
              format_map={
                  0: 'voltage',
                  1: 'current'
                  })

          self.add_parameter('excitation_on',
              flags=Instrument.FLAG_GET,
              type=types.BooleanType,
              channels=self._channels)

          self.add_parameter('excitation_range',
              flags=Instrument.FLAG_GETSET,
              type=types.StringType,
              channels=self._channels,
              format_map={
                  1: '2 uV or 1 pA',
                  2: '6.32 uV or 3.16 pA',
                  3: '20 uV or 10 pA',
                  4: '63.2 uV or 31.6 pA',
                  5: '200 uV or 100 pA',
                  6: '632 uV or 316 pA',
                  7: '2 mV or 1 nA',
                  8: '6.32 mV or 3.16 nA',
                  9: '20 mV or 10 nA',
                  10: '63.2 mV or 31.6 nA',
                  11: '200 mV or 100 nA',
                  12: '632 mV or 316nA',
                  13: '1 uA',
                  14: '3.16 uA',
                  15: '10 uA',
                  16: '31.6 uA',
                  17: '100 uA',
                  18: '316 uA',
                  19: '1 mA',
                  20: '3,16 mA',
                  21: '10 mA',
                  22: '31.6 mA'
                  })

          self.add_parameter('autorange',
              flags=Instrument.FLAG_GET,
              type=types.BooleanType,
              channels=self._channels)

          self.add_parameter('scanner_dwell_time',
              flags=Instrument.FLAG_GET,
              type=types.FloatType,
              units='s',
              channels=self._channels)

          self.add_parameter('scanner_pause_time',
              flags=Instrument.FLAG_GET,
              type=types.FloatType,
              units='s',
              channels=self._channels)

          self.add_parameter('filter_on',
              flags=Instrument.FLAG_GET,
              type=types.BooleanType,
              channels=self._channels)

          self.add_parameter('filter_settle_time',
              flags=Instrument.FLAG_GETSET,
              type=types.FloatType,
              units='s',
              channels=self._channels)

          self.add_parameter('filter_reset_threshold',
              flags=Instrument.FLAG_GET,
              type=types.FloatType,
              units='%',
              channels=self._channels)

          self._heater_ranges = {
              0: 'off',
              1: '31.6 uA',
              2: '100 uA',
              3: '316 uA',
              4: '1 mA',
              5: '3.16 mA',
              6: '10 mA',
              7: '31.6 mA',
              8: '100 mA' }
          self.add_parameter('heater_range',
              flags=Instrument.FLAG_GETSET,
              type=types.IntType,
              format_map=self._heater_ranges)

          self.add_parameter('heater_power',
              flags=Instrument.FLAG_GET,
              type=types.FloatType,
              units='W or %')

          self.add_parameter('heater_status',
              flags=Instrument.FLAG_GET,
              type=types.IntType,
              format_map={
                  0: 'OK',
                  1: 'heater open error'
                  })

          self.add_parameter('mode',
              flags=Instrument.FLAG_GETSET,
              type=types.IntType,
              format_map={0: 'Local', 1: 'Remote', 2: 'Remote, local lock'})

          self.add_parameter('temperature_control_mode',
              flags=Instrument.FLAG_GETSET,
              type=types.IntType,
              format_map={
                  1: 'closed loop PID',
                  2: 'Zone tuning',
                  3: 'open loop',
                  4: 'off'
                  })

          self.add_parameter('temperature_control_pid',
              flags=Instrument.FLAG_GETSET,
              type=types.TupleType)

          self.add_parameter('temperature_control_setpoint',
              flags=Instrument.FLAG_GETSET,
              type=types.FloatType)

          self.add_parameter('temperature_control_channel',
              flags=Instrument.FLAG_GET,
              type=types.IntType,
              format_map=dict(zip(self._channels,self._channels)))

          self.add_parameter('temperature_control_use_filtered_reading',
              flags=Instrument.FLAG_GET,
              type=types.BooleanType)

          self.add_parameter('temperature_control_setpoint_units',
              flags=Instrument.FLAG_GET,
              type=types.IntType,
              format_map={1: 'K', 2: 'Ohm'})

          self.add_parameter('temperature_control_heater_max_range',
              flags=Instrument.FLAG_GET,
              type=types.IntType,
              format_map=self._heater_ranges)

          self.add_parameter('temperature_control_heater_load_resistance',
              flags=Instrument.FLAG_GET,
              type=types.FloatType,
              units='Ohm')

          self.add_parameter('autoupdate_interval',
              flags=Instrument.FLAG_GETSET,
              type=types.IntType,
              units='s')

          self.add_parameter('still_heater',
              flags=Instrument.FLAG_GETSET|Instrument.FLAG_GET_AFTER_SET,
              type=types.FloatType,
              minval=0, maxval=100,
              units='%')

          self.add_parameter('autoupdate_while_measuring',
              flags=Instrument.FLAG_GETSET|Instrument.FLAG_PERSIST,
              type=types.BooleanType)
          if self.get_autoupdate_while_measuring() == None: self.update_value('autoupdate_while_measuring', False)

          self.add_function('local')
          self.add_function('remote')


          ### Auto-updating (useful mostly if you are also logging temperatures) ####
          self._autoupdater_handle = "lakeshore_autoupdater_%s" % (hashlib.md5(address).hexdigest()[:8])
          self.set_autoupdate_interval(kwargs.get('autoupdate_interval', 60. if self._logger != None else -1)) # seconds

          if reset:
              self.reset()
          else:
              self.get_all()

        except:
          self._visainstrument.close()
          raise

    def reset(self):
        self.__write('*RST')
        qt.msleep(.5)

    def get_all(self):
        self.get_identification()
        self.get_mode()        

        self.get_scanner_auto()
        self.get_scanner_channel()

        self.get_temperature_control_mode()
        self.get_temperature_control_pid()
        self.get_temperature_control_setpoint()
        self.get_temperature_control_setpoint_units()
        self.get_temperature_control_channel()
        self.get_temperature_control_use_filtered_reading()
        self.get_temperature_control_heater_max_range()
        self.get_temperature_control_heater_load_resistance()
        self.get_still_heater()
        
        self.get_heater_range()
        self.get_heater_status()
        self.get_heater_power()
        self.get_common_mode_reduction()
        self.get_guard_drive()
        
        for ch in self._channels:
          getattr(self, 'get_kelvin%s' % ch)()
          getattr(self, 'get_resistance%s' % ch)()
          getattr(self, 'get_resistance_range%s' % ch)()
          getattr(self, 'get_excitation_on%s' % ch)()
          getattr(self, 'get_excitation_mode%s' % ch)()
          getattr(self, 'get_excitation_range%s' % ch)()
          getattr(self, 'get_autorange%s' % ch)()
          getattr(self, 'get_scanner_dwell_time%s' % ch)()
          getattr(self, 'get_scanner_pause_time%s' % ch)()
          getattr(self, 'get_filter_on%s' % ch)()
          getattr(self, 'get_filter_settle_time%s' % ch)()
          getattr(self, 'get_filter_reset_threshold%s' % ch)()
          
          

    def __ask(self, msg):
        max_attempts = 5
        for attempt in range(max_attempts):
          try:
            m = self._visainstrument.ask("%s" % msg).replace('\r','')
            qt.msleep(.01)
            break
          except Exception as e:
            if attempt >= 0: logging.exception('Attempt #%d to communicate with LakeShore failed.', 1+attempt)
            if attempt < max_attempts-1 and not e.message.strip().lower().startswith('human abort'):
              qt.msleep((1+attempt)**2 * (0.1 + random.random()))
            else:
              raise
        return m

    def __write(self, msg):
        max_attempts = 5
        for attempt in range(max_attempts):
          try:
            self._visainstrument.write("%s" % msg)
            qt.msleep(.5)
            break
          except:
            if attempt > 0: logging.exception('Attempt #%d to communicate with LakeShore failed.', 1+attempt)
            if attempt < max_attempts-1 and not e.message.strip().lower().startswith('human abort'):
              qt.msleep((1+attempt)**2 * (0.1 + random.random()))
            else:
              raise

    def __query_auto_updated_quantities(self):

      if self not in qt.instruments.get_instruments().values():
        logging.debug('Old timer for Lakeshore auto-update. Terminating thread...')
        return False # stop the timer
      
      if not (self._autoupdate_interval != None and self._autoupdate_interval > 0):
        logging.debug('Auto-update interval not > 0. Terminating thread...')
        return False # stop the timer

      if (not self.get_autoupdate_while_measuring()) and qt.flow.is_measuring():
        return True # don't interfere with the measurement

      try:
        ch = self.do_get_scanner_channel()
        logging.debug('Auto-updating temperature reading (channel %s)...' % ch)
        getattr(self, 'get_kelvin%s' % ch)()
        getattr(self, 'get_resistance%s' % ch)()

      except Exception as e:
        logging.debug('Failed to auto-update temperature/resistance: %s' % (str(e)))

      return True # keep calling back
          
    def do_get_autoupdate_interval(self):
        return self._autoupdate_interval

    def do_set_autoupdate_interval(self, val):
        self._autoupdate_interval = val

        from qtflow import get_flowcontrol
        
        get_flowcontrol().remove_callback(self._autoupdater_handle, warn_if_nonexistent=False)
        
        if self._autoupdate_interval != None and self._autoupdate_interval > 0:

          if self._logger == None:
            logging.warn('You have enabled auto-updating, but not log file writing, which is a bit odd.')

          get_flowcontrol().register_callback(int(np.ceil(1e3 * self._autoupdate_interval)),
                                              self.__query_auto_updated_quantities,
                                              handle=self._autoupdater_handle)

    def do_get_autoupdate_while_measuring(self):
        return self.get('autoupdate_while_measuring', query=False)
    def do_set_autoupdate_while_measuring(self, v):
        self.update_value('autoupdate_while_measuring', bool(v))

    def do_get_identification(self):
        return self.__ask('*IDN?')
        
    def do_get_common_mode_reduction(self):
        ans = self.__ask('CMR?')
        return bool(int(ans))
        
    def do_get_guard_drive(self):
        ans = self.__ask('GUARD?')
        return bool(int(ans))
        
    def do_get_scanner_auto(self):
        ans = self.__ask('SCAN?')
        return bool(int(ans.split(',')[1]))

    def do_set_scanner_auto(self, val):
        ch = self.get_scanner_channel()
        cmd = 'SCAN %d,%d' % (ch, 1 if val else 0)
        self.__write(cmd)
        self.get_scanner_auto()
        self.get_scanner_channel()
        
    def do_get_scanner_channel(self):
        ans = self.__ask('SCAN?')
        return int(ans.split(',')[0])

    def do_set_scanner_channel(self, val):
        auto = self.get_scanner_auto()
        cmd = 'SCAN %d,%d' % (val, 1 if auto else 0)
        self.__write(cmd)
        self.get_scanner_auto()
        self.get_scanner_channel()

    def do_get_kelvin(self, channel):
        ans = float(self.__ask('RDGK? %s' % channel))
        if self._logger != None:
          try: self._logger('kelvin', channel, ans)
          except: logging.exception('Could not log kelvin%s', channel)
        return ans
        
    def do_get_resistance(self, channel):
        ans = float(self.__ask('RDGR? %s' % channel))
        if self._logger != None:
          try: self._logger('resistance', channel, ans)
          except: logging.exception('Could not log resistance%s', channel)
        return ans
        
    def do_get_resistance_range(self, channel):
        ans = self.__ask('RDGRNG? %s' % channel)
        return int(ans.split(',')[2])
        
    def do_get_excitation_mode(self, channel):
        ans = self.__ask('RDGRNG? %s' % channel)
        return int(ans.split(',')[0])
        
    def do_get_excitation_range(self, channel):
        ans = self.__ask('RDGRNG? %s' % channel)
        return int(ans.split(',')[1])
    def do_set_excitation_range(self, val, channel):
        s = self.__ask('RDGRNG? %s' % channel)
        s = s.split(',')
        s[1] = str(val)
        s = np.append([ str(channel) ], s)
        self.__write('RDGRNG %s' % (','.join(s)))
        
    def do_get_autorange(self, channel):
        ans = self.__ask('RDGRNG? %s' % channel)
        return bool(int(ans.split(',')[3]))
        
    def do_get_excitation_on(self, channel):
        ans = self.__ask('RDGRNG? %s' % channel)
        return (int(ans.split(',')[4]) == 0)
        
    def do_get_scanner_dwell_time(self, channel):
        ans = self.__ask('INSET? %s' % channel)
        return float(ans.split(',')[1])
        
    def do_get_scanner_pause_time(self, channel):
        ans = self.__ask('INSET? %s' % channel)
        return float(ans.split(',')[2])
        
    def do_get_filter_on(self, channel):
        ans = self.__ask('FILTER? %s' % channel)
        return bool(int(ans.split(',')[0]))
        
    def do_get_filter_settle_time(self, channel):
        ans = self.__ask('FILTER? %s' % channel)
        return float(ans.split(',')[1])
        
    def do_set_filter_settle_time(self, val, channel):
        cmd = 'FILTER %s,1,%d,80' % (channel,int(np.round(val)))
        self.__write(cmd)
        getattr(self, 'get_filter_settle_time%s' % channel)()
        getattr(self, 'get_filter_on%s' % channel)()
        getattr(self, 'get_filter_reset_threshold%s' % channel)()
        
    def do_get_filter_reset_threshold(self, channel):
        ans = self.__ask('FILTER? %s' % channel)
        return float(ans.split(',')[2])
        
    def do_get_heater_range(self):
        ans = self.__ask('HTRRNG?')
        return int(ans)
        
    def do_get_heater_power(self):
        ans = self.__ask('HTR?')
        return float(ans)
        
    def do_set_heater_range(self, val):
        self.__write('HTRRNG %d' % val)
        self.get_heater_range()
        
    def do_get_heater_status(self):
        ans = self.__ask('HTRST?')
        return ans
        
    def do_get_mode(self):
        ans = self.__ask('MODE?')
        return int(ans)

    def do_set_mode(self, mode):
        self.__write('MODE %d' % mode)
        self.get_mode()

    def local(self):
        self.set_mode(1)

    def remote(self):
        self.set_mode(2)
        
    def do_get_temperature_control_mode(self):
        ans = self.__ask('CMODE?')
        return int(ans)

    def do_set_temperature_control_mode(self, mode):
        setp = self.get_temperature_control_setpoint()

        self.__write('CMODE %d' % mode)
        self.get_temperature_control_mode()

        new_setp = self.get_temperature_control_setpoint()
        if new_setp != setp:
          logging.info('setpoint changed from %g to %g when changing CMODE to %s. Setting it back...' % (setp, new_setp, mode))
          self.set_temperature_control_setpoint(setp)

    def do_get_temperature_control_pid(self):
        ans = self.__ask('PID?')
        fields = ans.split(',')
        if len(fields) != 3:
            return None
        fields = [float(f) for f in fields]
        return fields

    def do_set_temperature_control_pid(self, val):
        assert len(val) == 3, 'PID parameter must be a triple of numbers.'
        assert val[0] >= 0.001 and val[0] < 1000, 'P out of range.'
        assert val[1] >= 0 and val[1] < 10000, 'I out of range.'
        assert val[2] >= 0 and val[2] < 2500, 'D out of range.'
        cmd = 'PID %.5g,%.5g,%.5g' % (val[0], val[1], val[2])
        self.__write(cmd)
        self.get_temperature_control_pid()

    def do_get_still_heater(self):
        ans = self.__ask('STILL?')
        return float(ans)

    def do_set_still_heater(self, val):
        self.__write('STILL %.2F' % (val))

    def do_get_temperature_control_setpoint(self):
        ans = self.__ask('SETP?')
        return float(ans)
        
    def do_set_temperature_control_setpoint(self, val):
        for attempts in range(5):
          self.__write('SETP %.3E' % (val))
          setp = self.get_temperature_control_setpoint()
          if np.abs(setp - val) < 1e-5: return # all is well
          logging.warn('Failed to change setpoint to %g (instead got %g). Retrying...' % (val, setp))
          qt.msleep(5.)
        logging.warn('Final attempt to change setpoint to %g failed (instead got %g).' % (val, setp))
        
    def do_get_temperature_control_channel(self):
        ans = self.__ask('CSET?')
        return int(ans.split(',')[0])
        
    def do_get_temperature_control_use_filtered_reading(self):
        ans = self.__ask('CSET?')
        return bool(ans.split(',')[1])
        
    def do_get_temperature_control_setpoint_units(self):
        ans = self.__ask('CSET?')
        return ans.split(',')[2]
        
    def do_get_temperature_control_heater_max_range(self):
        ans = self.__ask('CSET?')
        return int(ans.split(',')[5])
        
    def do_get_temperature_control_heater_load_resistance(self):
        ans = self.__ask('CSET?')
        return float(ans.split(',')[6])
