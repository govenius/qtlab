# Agilent V750, Agilent V750 turbo controller
# Joonas Govenius <joonas.govenius@aalto.fi>, 2014
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
#import visa
import serial # Requires pyserial >= 3.0
import types
import logging
import re
import math
import time
import numpy as np
import qt
import struct
import random
import hashlib

class Agilent_V750(Instrument):
  '''
  Driver for querying turbo status.
  
  logger can be any function that accepts a triple (quantity, 'turbo', value) as an input.
  
  Assumes an RS-232 connection and a baudrate of 9600.
  '''

  def __init__(self, name, address, reset=False, **kwargs):
    Instrument.__init__(self, name)

    self._address = address
    self._serial_min_time_between_commands = 0.020 # s
    self._serial_last_access_time = 0
    self._serial_reservation_counter = 0

    m = re.match(r'(?i)((com)|(/dev/ttys))(\d+)', address)
    try:
      self._serialportno = int(m.group(4)) - (1 if m.group(1).lower() == 'com' else 0)
      logging.debug('serial port number = %d', self._serialportno)
    except:
      logging.warn('Only local serial ports supported. Address must be of the form COM1 or /dev/ttyS0, not %s' % address)
      raise

    self._etx = 0x03 # end of transmission symbol
      
    self._logger = kwargs.get('logger', None)
    
    self.add_parameter('on',
      flags=Instrument.FLAG_GET,
      type=types.BooleanType)
    self.add_parameter('active_stop',
      flags=Instrument.FLAG_GET,
      type=types.BooleanType)
    self.add_parameter('water_cooling',
      flags=Instrument.FLAG_GET,
      type=types.BooleanType)
    self.add_parameter('low_speed_mode',
      flags=Instrument.FLAG_GET,
      type=types.BooleanType)
    self.add_parameter('soft_start',
      flags=Instrument.FLAG_GETSET,
      type=types.BooleanType)
    self.add_parameter('frequency_readable_after_stop',
      flags=Instrument.FLAG_GET,
      type=types.BooleanType)
    self.add_parameter('vent_valve_open',
      flags=Instrument.FLAG_GET,
      type=types.BooleanType)
    
    self.add_parameter('remote_configuration',
      flags=Instrument.FLAG_GETSET,
      type=types.IntType,
      format_map={0: 'serial',
                  1: 'remote',
                  2: 'front'}) # <-- According to the manual only 0 and 1 should be possible but that's not the case.

    error_codes ={0: 'no error',
                  1: 'no connection',
                  2: 'pump overtemp.',
                  4: 'controller overtemp.',
                  8: 'power fail',
                  16: 'output fail',
                  32: 'overvoltage',
                  64: 'short circuit',
                  128: 'too high load'}
    all_error_combinations = dict(
        (error, ', '.join( error_codes[2**j] for j in range(0,8) if ((error>>j)&1) ))
        for error in range(1, 2**8)
      )
    all_error_combinations[0] = error_codes[0]
    self.add_parameter('error_code',
      flags=Instrument.FLAG_GET,
      type=types.IntType,
      format_map=all_error_combinations)

    self.add_parameter('status',
      flags=Instrument.FLAG_GET,
      type=types.IntType,
      format_map={0: 'stop',
                  1: 'waiting interlock',
                  2: 'ramp-up',
                  3: 'auto-tuning',
                  4: 'braking',
                  5: 'normal',
                  6: 'fail',
                  7: 'leak check'})
    self.add_parameter('gas_load_type',
      flags=Instrument.FLAG_GET,
      type=types.IntType,
      format_map={0: 'Ar',
                  1: 'N2'})
    self.add_parameter('hours_of_operation',
      flags=Instrument.FLAG_GET,
      type=types.FloatType,
      units='h', format='%.1f')

    self.add_parameter('firmware_program_listing',
      flags=Instrument.FLAG_GET,
      type=types.StringType)
    self.add_parameter('firmware_parameter_listing',
      flags=Instrument.FLAG_GET,
      type=types.StringType)

    self.add_parameter('speed_target_low',
      flags=Instrument.FLAG_GET,
      type=types.FloatType,
      units='Hz', format='%.1f')
    self.add_parameter('speed_target_high',
      flags=Instrument.FLAG_GET,
      type=types.FloatType,
      units='Hz', format='%.1f')
    self.add_parameter('frequency',
      flags=Instrument.FLAG_GET,
      type=types.FloatType,
      units='Hz', format='%.1f')
    self.add_parameter('power',
      flags=Instrument.FLAG_GET,
      type=types.FloatType,
      units='W', format='%.1f')
    self.add_parameter('voltage',
      flags=Instrument.FLAG_GET,
      type=types.FloatType,
      units='V', format='%.1f')
    self.add_parameter('current',
      flags=Instrument.FLAG_GET,
      type=types.FloatType,
      units='mA', format='%.1f')

    self.add_parameter('temperature_controller',
      flags=Instrument.FLAG_GET,
      type=types.FloatType,
      units='C', format='%.1f')
    self.add_parameter('temperature_bearing',
      flags=Instrument.FLAG_GET,
      type=types.FloatType,
      units='C', format='%.1f')
    self.add_parameter('temperature_body',
      flags=Instrument.FLAG_GET,
      type=types.FloatType,
      units='C', format='%.1f')

    self.add_parameter('pressure_gauge1',
      flags=Instrument.FLAG_GET,
      type=types.FloatType,
      units='mBar', format='%.2f')
    self.add_parameter('pressure_gauge2',
      flags=Instrument.FLAG_GET,
      type=types.FloatType,
      units='mBar', format='%.2f')


    ### Auto-updating (useful mostly if you are also logging) ####

    self.add_parameter('autoupdate_interval',
        flags=Instrument.FLAG_GETSET,
        type=types.IntType,
        units='s')
    self.add_parameter('autoupdate_while_measuring',
        flags=Instrument.FLAG_GETSET|Instrument.FLAG_PERSIST,
        type=types.BooleanType)

    self._autoupdater_handle = "turbo_autoupdater_%s" % (hashlib.md5(address).hexdigest()[:8])
    if self.get_autoupdate_while_measuring() == None: self.update_value('autoupdate_while_measuring', False)
    self.set_autoupdate_interval(kwargs.get('autoupdate_interval', 60. if self._logger != None else -1)) # seconds

    if reset:
      self.reset()
    else:
      self.get_all()

  def __query_auto_updated_quantities(self):

    if self not in qt.instruments.get_instruments().values():
      logging.debug('Old timer for turbo auto-update. Terminating thread...')
      return False # stop the timer
    
    if not (self._autoupdate_interval != None and self._autoupdate_interval > 0):
      logging.debug('Auto-update interval not > 0. Terminating thread...')
      return False # stop the timer

    if (not self.get_autoupdate_while_measuring()) and qt.flow.is_measuring():
      return True # don't interfere with the measurement

    try:
      logging.debug('Auto-updating turbo readings...')
      getattr(self, 'get_all')()

    except Exception as e:
      logging.debug('Failed to auto-update turbo readings: %s', str(e))

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

  def reset(self):
    pass

  def get_all(self):
    try:
      self._reserve_serial()
      self.get_on()
      self.get_remote_configuration()
      self.get_active_stop()
      self.get_hours_of_operation()
      self.get_error_code()
      self.get_status()
      self.get_water_cooling()
      self.get_gas_load_type()
      self.get_low_speed_mode()
      self.get_soft_start()
      self.get_firmware_program_listing()
      self.get_firmware_parameter_listing()
      self.get_frequency_readable_after_stop()
      self.get_speed_target_low()
      self.get_speed_target_high()
      self.get_vent_valve_open()
      self.get_frequency()
      self.get_power()
      self.get_voltage()
      self.get_current()
      self.get_temperature_controller()
      self.get_temperature_bearing()
      self.get_temperature_body()
      self.get_pressure_gauge1()
      self.get_pressure_gauge2()
    finally:
      self._release_serial()


  def _reserve_serial(self):
    '''
    Counter based opening/closing of the serial connection.

    Using _reserve_serial() and _release_serial explicitly
    prevents the session from being closed between each command.
    Used, e.g., in get_all().
    '''
    self._serial_reservation_counter += 1
    logging.debug('connection reserved. counter = %s', self._serial_reservation_counter)
    time_to_sleep = ( self._serial_min_time_between_commands
                      - (time.time() - self._serial_last_access_time) )
    if time_to_sleep > 0: qt.msleep(time_to_sleep)
    if self._serial_reservation_counter == 1:
      #logging.info('turbo: opening connection. counter = %s', self._serial_reservation_counter)
      self._serial_connection = serial.Serial(self._address,
          baudrate=9600,
          bytesize=serial.EIGHTBITS,
          parity=serial.PARITY_NONE,
          stopbits=serial.STOPBITS_ONE,
          #dsrdtr=False,
          #interCharTimeout=None,
          #rtscts=False,
          #writeTimeout=None,
          timeout=1.)

  def _release_serial(self):
    ''' Counter based opening/closing of the serial connection. '''
    assert self._serial_reservation_counter > 0, 'Trying to release a serial session that has not been reserved! (counter = %s)' % (self._serial_reservation_counter)
    self._serial_reservation_counter -= 1
    logging.debug('connection released. counter = %s', self._serial_reservation_counter)
    self._serial_last_access_time = time.time()
    if self._serial_reservation_counter == 0:
      #logging.info('turbo: closing connection. counter = %s', self._serial_reservation_counter)
      self._serial_connection.close()

  def _ask(self, msg):
    logging.debug('Sending %s', ["0x%02x" % ord(c) for c in msg])
    
    #for attempt in range(2):
    try:
      self._reserve_serial()
      self._serial_connection.write(msg)
      m = ''
      while len(m) < 3 or (ord(m[-3]) != self._etx):
        lastlen = len(m)
        m += self._serial_connection.read()
        if lastlen == len(m): assert False, 'Timeout on serial port read.'
      logging.debug('Got %s', ["0x%02x" % ord(c) for c in m])
      return m

    except:
      #logging.exception('Attempt %d to communicate with turbo failed', attempt)
      #qt.msleep(1. + attempt**2)
      qt.msleep(1.)
      raise

    finally:
      self._release_serial()

    assert False, 'All attempts to communicate with the turbo failed.'


  def __read_value(self, value, log=False, write_instead=None):
    '''
    Read the specified value.
    If write_instead != None, it will be written instead
    '''
    # possible data types: 'L' == logic/boolean, 'N' == numeric, 'A' == alphanumeric
    window_datatype = {
        'firmware_program_listing': (406, 'A'),
        'firmware_parameter_listing': (407, 'A'),
        'remote_configuration': (8, 'A'), # <-- According to manual this should be 'L', but in reality it returns 0 (serial), 1 (remote), or 2 (front).
        'on': (0, 'L'),
        'active_stop': (107, 'L'),
        'water_cooling': (106, 'L'),
        'gas_load_type': (157, 'L'),
        'low_speed_mode': (1, 'L'),
        'soft_start': (100, 'L'),
        'speed_target_low': (117, 'N'),
        'speed_target_high': (120, 'N'),
        'vent_valve_open': (122, 'L'),
        'current': (200, 'N'),
        'voltage': (201, 'N'),
        'power': (202, 'N'),
        'frequency': (232, 'N'),
        'frequency_readable_after_stop': (167, 'L'),
        'temperature_bearing': (204, 'N'),
        'status': (205, 'N'),
        'error_code': (206, 'N'),
        'temperature_controller': (211, 'N'),
        'temperature_body': (222, 'N'),
        'pressure_gauge1': (224, 'N'),
        'pressure_gauge2': (254, 'N'),
        'hours_of_operation': (302, 'N')
    }
    assert value in window_datatype.keys(), 'Value must be one of: %s' % str(window_datatype.keys())

    stx = 0x02
    addr = 0x80 # for (RS-232)
    window = "%03d" % (window_datatype[value][0])
    command = 0x30 # read operation
    data = write_instead
    if data != None:
      if   window_datatype[value][1] == 'L': data = '1' if data else '0'
      elif window_datatype[value][1] == 'N': data = "%03d" % data
      elif window_datatype[value][1] == 'A':
        assert len(data) <= 10, 'Alphanumeric data for %s too long.' % value
        data = '%10s' % data

    crc = (addr ^ ord(window[0]) ^ ord(window[1]) ^ ord(window[2]) ^ command ^ self._etx)
    if data != None:
      for b in data: crc ^= ord(b)
    crc = "%02X" % crc
    logging.debug('crc = 0x%s', crc)
    
    if data == None:
      msg = struct.pack('>BB3sBB2s',
                    stx, # start of transmission
                    addr,
                    window,
                    command,
                    self._etx,
                    crc # checksum
                    )
    else:
      data_packing = {'L': '1s', 'N': '6s', 'A': '10s'}[window_datatype[value][1]]
      msg = struct.pack('>BB3sB%sB2s' % data_packing,
                    stx, # start of transmission
                    addr,
                    window,
                    command,
                    data,
                    self._etx,
                    crc # checksum
                    )
    #logging.warn('Asking %s', ' '.join([ '0x%02x' % ord(c) for c in msg ]))

    # query instrument
    r = self._ask(msg)

    # parse response
    datalen = {'L': 1, 'N': 6, 'A': 10}[window_datatype[value][1]]
    assert len(r) == 9 + datalen, 'Unexpected length of reply: %d != %d' % (len(r), 9 + datalen)
    rstx, raddr, rwindow, rcommand, rval, retx, rcrc = struct.unpack_from('>BB3sB%dsB2s' % datalen, r)
    crc = (raddr ^ ord(rwindow[0]) ^ ord(rwindow[1]) ^ ord(rwindow[2]) ^ rcommand ^ retx)
    for c in rval: crc ^= ord(c)
    crc = "%02X" % crc
    assert rcrc.upper() == crc.upper(), 'Invalid check bytes: 0x%s != 0x%s' % (crc, rcrc)
    assert rstx == stx
    assert rwindow == window
    assert rcommand == command
    assert retx == self._etx

    if window_datatype[value][1] == 'L':
      rval = (rval == '1')
    elif window_datatype[value][1] == 'N':
      rval = float(rval)
    elif window_datatype[value][1] == 'A':
      rval = rval
      if value == 'remote_configuration': # Compensate for the firmware bug. (This should be 'N' or 'L' type.)
        if   rval.strip() == '0': rval = 0
        elif rval.strip() == '1': rval = 1
        elif rval.strip() == '2': rval = 2
        else: logging.warn('Uknown response to "remote_configuration" (WIN 008): "%s"', rval)
    else:
      assert False, 'Invalid datatype: %s' % window_datatype[value][1]

    if log and self._logger != None:
      try: self._logger(value, 'turbo', rval)
      except Exception as e: logging.debug('Could not log turbo %s: %s', value, str(e))

    return rval

  def do_get_remote_configuration(self): return self.__read_value('remote_configuration')
  def do_set_remote_configuration(self, val): return self.__read_value('remote_configuration', write_instead=val)
  def do_get_on(self): return self.__read_value('on')
  def do_get_active_stop(self): return self.__read_value('active_stop')
  def do_get_water_cooling(self): return self.__read_value('water_cooling')
  def do_get_gas_load_type(self): return self.__read_value('gas_load_type')
  def do_get_low_speed_mode(self): return self.__read_value('low_speed_mode')
  def do_get_soft_start(self): return self.__read_value('soft_start')
  def do_set_soft_start(self, val): return self.__read_value('soft_start', write_instead=val)
  def do_get_frequency_readable_after_stop(self): return self.__read_value('frequency_readable_after_stop')
  def do_get_speed_target_low(self): return self.__read_value('speed_target_low')
  def do_get_speed_target_high(self): return self.__read_value('speed_target_high')
  def do_get_vent_valve_open(self): return self.__read_value('vent_valve_open')
  def do_get_frequency(self): return self.__read_value('frequency', log=True)
  def do_get_current(self): return self.__read_value('current', log=True)
  def do_get_voltage(self): return self.__read_value('voltage', log=True)
  def do_get_power(self): return self.__read_value('power', log=True)
  def do_get_hours_of_operation(self): return self.__read_value('hours_of_operation', log=True)
  def do_get_status(self): return self.__read_value('status', log=True)
  def do_get_error_code(self): return self.__read_value('error_code', log=True)
  def do_get_temperature_controller(self): return  self.__read_value('temperature_controller', log=True)
  def do_get_temperature_body(self): return  self.__read_value('temperature_body', log=True)
  def do_get_temperature_bearing(self): return self.__read_value('temperature_bearing', log=True)
  def do_get_firmware_program_listing(self): return self.__read_value('firmware_program_listing')
  def do_get_firmware_parameter_listing(self): return self.__read_value('firmware_parameter_listing')

  def do_get_pressure_gauge1(self):
    try: return self.__read_value('pressure_gauge1', log=True)
    except: return np.nan # normal if gauge isn't connected
  def do_get_pressure_gauge2(self):
    try: return self.__read_value('pressure_gauge2', log=True)
    except: return np.nan # normal if gauge isn't connected
