# Cryomech CP2800, Cryomech CP2800 pulsetube compressor
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
import serial # Requires pyserial >= 3.0.
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

class Cryomech_CP2800(Instrument):
  '''
  Driver for querying compressor status.
  
  logger can be any function that accepts a triple (quantity, 'compressor', value) as an input.
  
  
  Set SMDP address to 16 and the baudrate to 9600 in the compressor settings.
  Use an RS-232 cable.
  
  The protocol is a PITA and you don't get any reply at all if you make a mistake.
  To test that your connection works at all, you can try:
  
  import serial
  import struct
  ser = serial.Serial('COM5', timeout=2.) # with your COM<n> instead of COM5
  print ser.getSettingsDict()
  # You should get:
  #  {'baudrate': 9600,
  #  'bytesize': 8,
  #  'dsrdtr': False,
  #  'interCharTimeout': None,
  #  'parity': 'N',
  #  'rtscts': False,
  #  'stopbits': 1,
  #  'timeout': 2.,
  #  'writeTimeout': None,
  #  'xonxoff': False}

  msg = struct.pack('>BBBcBBBBBBc',0x02,0x10,0x80,'c',0x5f,0x95,0x00,0x15,0x4f,0x4c,'\r')
  ser.write(msg)
  reply = [ ser.read() for i in range(15) ]
  print reply
  # You should get: ['\x02', '\x10', '\x89', 'c', '_', '\x95', '\x00', '\x00', '\x00', '\x00', '\x01', '\x15', '@', 'F', '\r']
  # The 11th byte is \x01 (\x00) if the compressor is on (off).
  
  '''

  def __init__(self, name, address, reset=False, **kwargs):
    Instrument.__init__(self, name)

    self._address = address
    self._serial_min_time_between_commands = 0.100 # s
    self._serial_last_access_time = 0
    self._serial_reservation_counter = 0

    m = re.match(r'(?i)((com)|(/dev/ttys))(\d+)', address)
    try:
      self._serialportno = int(m.group(4)) - (1 if m.group(1).lower() == 'com' else 0)
      logging.debug('serial port number = %d', self._serialportno)
    except:
      logging.warn('Only local serial ports supported. Address must be of the form COM1 or /dev/ttyS0, not %s' % address)
      raise

    self._lastpacketserialno = max(0x10,
                                   ( 0x10 + int(random.random() * (0xff-0x10)) )%0x100
                                   )
    
    self._logger = kwargs.get('logger', None)
    
    self.add_parameter('on',
      flags=Instrument.FLAG_GET,
      type=types.BooleanType)
    
    self.add_parameter('clock_battery_ok',
      flags=Instrument.FLAG_GET,
      type=types.BooleanType)
    
    self.add_parameter('error_code',
      flags=Instrument.FLAG_GET,
      type=types.IntType)
    
    self.add_parameter('hours_of_operation',
      flags=Instrument.FLAG_GET,
      type=types.FloatType,
      units='h', format='%.1f')
    
    self.add_parameter('water_in_temperature',
      flags=Instrument.FLAG_GET,
      type=types.FloatType,
      units='C', format='%.1f')
    self.add_parameter('water_out_temperature',
      flags=Instrument.FLAG_GET,
      type=types.FloatType,
      units='C', format='%.1f')
    self.add_parameter('helium_temperature',
      flags=Instrument.FLAG_GET,
      type=types.FloatType,
      units='C', format='%.1f')
    self.add_parameter('oil_temperature',
      flags=Instrument.FLAG_GET,
      type=types.FloatType,
      units='C', format='%.1f')
    self.add_parameter('cpu_temperature',
      flags=Instrument.FLAG_GET,
      type=types.FloatType,
      units='C', format='%.1f')

    self.add_parameter('pressure_high_side',
      flags=Instrument.FLAG_GET,
      type=types.FloatType,
      units='psi (absolute)', format='%.1f')
    self.add_parameter('pressure_low_side',
      flags=Instrument.FLAG_GET,
      type=types.FloatType,
      units='psi (absolute)', format='%.1f')
    self.add_parameter('pressure_high_side_average',
      flags=Instrument.FLAG_GET,
      type=types.FloatType,
      units='psi (absolute)', format='%.1f')
    self.add_parameter('pressure_low_side_average',
      flags=Instrument.FLAG_GET,
      type=types.FloatType,
      units='psi (absolute)', format='%.1f')


    ### Auto-updating (useful mostly if you are also logging) ####

    self.add_parameter('autoupdate_interval',
        flags=Instrument.FLAG_GETSET,
        type=types.IntType,
        units='s')
    self.add_parameter('autoupdate_while_measuring',
        flags=Instrument.FLAG_GETSET|Instrument.FLAG_PERSIST,
        type=types.BooleanType)

    self._autoupdater_handle = "compressor_autoupdater_%s" % (hashlib.md5(address).hexdigest()[:8])
    if self.get_autoupdate_while_measuring() == None: self.update_value('autoupdate_while_measuring', False)
    self.set_autoupdate_interval(kwargs.get('autoupdate_interval', 60. if self._logger != None else -1)) # seconds

    if reset:
      self.reset()
    else:
      self.get_all()

  def __query_auto_updated_quantities(self):

    if self not in qt.instruments.get_instruments().values():
      logging.debug('Old timer for compressor auto-update. Terminating thread...')
      return False # stop the timer
    
    if not (self._autoupdate_interval != None and self._autoupdate_interval > 0):
      logging.debug('Auto-update interval not > 0. Terminating thread...')
      return False # stop the timer

    if (not self.get_autoupdate_while_measuring()) and qt.flow.is_measuring():
      return True # don't interfere with the measurement

    try:
      logging.debug('Auto-updating compressor readings...')
      getattr(self, 'get_all')()

    except Exception as e:
      logging.debug('Failed to auto-update compressor readings: %s' % (str(e)))

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
    self.__write('*RST')
    qt.msleep(.5)

  def get_all(self):
    try:
      self._reserve_serial()
      self.get_on()
      self.get_hours_of_operation()
      self.get_clock_battery_ok()
      self.get_error_code()
      self.get_water_in_temperature()
      self.get_water_out_temperature()
      self.get_helium_temperature()
      self.get_oil_temperature()
      self.get_cpu_temperature()
      self.get_pressure_high_side()
      self.get_pressure_low_side()
      self.get_pressure_high_side_average()
      self.get_pressure_low_side_average()
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
    time_to_sleep = ( self._serial_min_time_between_commands
                      - (time.time() - self._serial_last_access_time) )
    if time_to_sleep > 0: qt.msleep(time_to_sleep)
    if self._serial_reservation_counter == 1:
      #logging.info('cryo: opening connection. counter = %s', self._serial_reservation_counter)
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
    self._serial_last_access_time = time.time()
    if self._serial_reservation_counter == 0:
      #logging.info('cryo: closing connection. counter = %s', self._serial_reservation_counter)
      self._serial_connection.close()

  def _ask(self, msg):
    logging.debug('Sending %s', ["0x%02x" % ord(c) for c in msg])

    #for attempt in range(3):
    try:
      self._reserve_serial()
      self._serial_connection.write(msg)
      m = ''
      while len(m) < 1 or m[-1] != '\r':
        lastlen = len(m)
        m += self._serial_connection.read()
        if lastlen == len(m): assert False, 'Timeout on serial port read.'
      logging.debug('Got %s', ["0x%02x" % ord(c) for c in m])
      return m

    except:
      #logging.exception('Attempt %d to communicate with compressor failed', attempt)
      #qt.msleep(1. + attempt**2)
      qt.msleep(1.)
      raise

    finally:
      self._release_serial()

    assert False, 'All attempts to communicate with the compressor failed.'

  def __read_int(self, hash_code, array_index=0):
    assert len(hash_code) == 2, 'Specify hash code as a tuple of ints, e.g. (0x35, 0x74) for oil temperature (given in hex in the manual).'

    addr = 16 # address (16 == point-to-point (RS-232), >16 single-master--multi-slave (RS-485))
    cmdrsp = 0x80 # 8 --> CMD for accessing compressor dictionary variables (data_dictionary(V1.8).pdf)
                  # 0 --> NOT a response
    operation = 'c' # read operation
    self._lastpacketserialno = max(0x10, (self._lastpacketserialno+1) % 0x100)
    
    # ADDR + CMD_RSP + DATA
    checksum = (addr + cmdrsp + ord(operation) + hash_code[0] + hash_code[1]
                + array_index + self._lastpacketserialno) % 0x100
    checksum = [ 0x40 + ((checksum & 0xf0)>>4), 0x40 + (checksum & 0x0f) ]
    logging.debug('Checksum = %s', [ '0x%02x' % c for c in checksum ])
    
    msg = struct.pack('>BBBcBBBBBBc',
                    0x02, # start of Sycon multidrop packet (Sycon Multidrop Protocol II.pdf)
                    addr,
                    cmdrsp, 
                    operation,
                    hash_code[0], hash_code[1], # ID of the read quantity
                    array_index,
                    self._lastpacketserialno,
                    checksum[0],
                    checksum[1],
                    '\r' # end of Sycon multidrop packet
                    )
    #logging.warn('Asking %s', ' '.join([ '0x%02x' % ord(c) for c in msg ]))
    
    # escape special bytes
    esc = {
      struct.pack('>B', 0x02): struct.pack('>Bc', 0x07, '0'),
      struct.pack('>c', '\r'): struct.pack('>Bc', 0x07, '1'),
      struct.pack('>B', 0x07): struct.pack('>Bc', 0x07, '2')
      }
      
    for i in range(len(msg)-2,0,-1):
      if msg[i] in esc.keys(): msg = msg[:i] + esc[msg[i]] + msg[i+1:]

    # query instrument
    r = self._ask(msg)

    # unescape special bytes
    invesc = dict( (v,k) for k,v in esc.iteritems() )
    i = 1
    while i < len(r)-2:
      if r[i:i+2] in invesc.keys(): r = r[:i] + invesc[r[i:i+2]] + r[i+2:]
      i += 1
    assert len(r) == 15, 'The length of the (unescaped) data should be exactly 15, not %d: %s' % (len(r),
                              ' '.join([ '0x%02x' % ord(c) for c in r ]))
    logging.debug('Response after unescaping (%d bytes): %s', 
                  len(r), ' '.join([ '0x%02x' % ord(c) for c in r ]) )

    # parse response
    stx, addr, cmdrsp, c_char, hash_code0, hash_code1, array_index, data, serialno, chk1, chk2, cr = struct.unpack_from('>BBBcBBBlBBBc', r)
    
    checksum = (addr + cmdrsp + ord(c_char) + hash_code0 + hash_code1
                + array_index + serialno
                + (data & 0xff)
                + (data>>8 & 0xff)
                + (data>>16 & 0xff)
                + (data>>24 & 0xff)
                ) % 0x100
    assert (chk1 & 0xf0) == 0x40 and (chk2 & 0xf0) == 0x40, 'Invalid check byte format: 0x%02x 0x%02x' % (chk1, chk2)
    assert checksum == ((chk1 & 0x0f)<<4) + (chk2 & 0x0f), 'Invalid check sum: 0x%02x 0x%02x does not match data (0x%02x).' % (chk1, chk2, checksum)
    assert c_char == 'c', c_char
    assert serialno == self._lastpacketserialno, '0x%02x != 0x%02x' % (serialno, self._lastpacketserialno)
    assert hash_code0 == hash_code[0], '0x%02x != 0x%02x' % (ord(hash_code0), ord(hash_code[0]))
    assert hash_code1 == hash_code[1], '0x%02x != 0x%02x' % (ord(hash_code1), ord(hash_code[1]))
    return data

  def __log(self, quantity, value):
    if self._logger != None:
      try: self._logger(quantity, 'compressor', value)
      except Exception as e: logging.debug('Could not log compressor %s: %s', quantity, str(e))

  def do_get_on(self):
    return (self.__read_int((0x5F, 0x95)) == 1)

  def do_get_hours_of_operation(self):
    ans = self.__read_int((0x45, 0x4C)) / 60.
    self.__log('hours_of_operation', ans)
    return ans

  def do_get_error_code(self):
    ans = self.__read_int((0x65, 0xA4))
    self.__log('error_code', ans)
    return ans

  def do_get_clock_battery_ok(self):
    return (self.__read_int((0xA3, 0x7A)) == 1)

  def do_get_water_in_temperature(self):
    ans = 0.1 * self.__read_int((0x0D, 0x8F), 0)
    self.__log('water_in_temperature', ans)
    return ans

  def do_get_water_out_temperature(self):
    ans = 0.1 * self.__read_int((0x0D, 0x8F), 1)
    self.__log('water_out_temperature', ans)
    return ans

  def do_get_helium_temperature(self):
    ans = 0.1 * self.__read_int((0x0D, 0x8F), 2)
    self.__log('helium_temperature', ans)
    return ans

  def do_get_oil_temperature(self):
    ans = 0.1 * self.__read_int((0x0D, 0x8F), 3)
    self.__log('oil_temperature', ans)
    return ans

  def do_get_cpu_temperature(self):
    ans = 0.1 * self.__read_int((0x35, 0x74))
    self.__log('cpu_temperature', ans)
    return ans

  def do_get_pressure_high_side(self):
    ans = 0.1 * self.__read_int((0xAA, 0x50), 0)
    self.__log('pressure_high', ans)
    return ans

  def do_get_pressure_low_side(self):
    ans = 0.1 * self.__read_int((0xAA, 0x50), 1)
    self.__log('pressure_low', ans)
    return ans

  def do_get_pressure_high_side_average(self):
    return 0.1 * self.__read_int((0x7E, 0x90))

  def do_get_pressure_low_side_average(self):
    return 0.1 * self.__read_int((0xBB, 0x94))
