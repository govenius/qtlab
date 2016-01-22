# TTI EL302P, current source controller
# Joonas Govenius <joonas.govenius@aalto.fi>, 2016
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
import struct
import random
import hashlib

class TTI_EL302P(Instrument):
  '''
  Driver for querying turbo status.
  
  logger can be any function that accepts a triple (quantity, 'current_source', value) as an input.
  
  Assumes an RS-232 connection and a baudrate of 9600.
  '''

  def __init__(self, name, address, reset=False, **kwargs):
    Instrument.__init__(self, name)

    self._address = address

    self.add_parameter('on',
      flags=Instrument.FLAG_GETSET,
      type=types.BooleanType)

    self.add_parameter('voltage',
       flags=Instrument.FLAG_GET,
       type=types.FloatType,
       units='V', format='%.2f')
    self.add_parameter('current',
       flags=Instrument.FLAG_GET,
       type=types.FloatType,
       units='A', format='%.2f')

    self.add_parameter('voltage_limit',
       flags=Instrument.FLAG_GET,
       type=types.FloatType,
       units='V', format='%.2f')
    self.add_parameter('current_limit',
       flags=Instrument.FLAG_GET,
       type=types.FloatType,
       units='A', format='%.2f')

    self._logger = kwargs.get('logger', None)
    
    ### Auto-updating (useful mostly if you are also logging) ####

    self.add_parameter('autoupdate_interval',
        flags=Instrument.FLAG_GETSET,
        type=types.IntType,
        units='s')
    self.add_parameter('autoupdate_while_measuring',
        flags=Instrument.FLAG_GETSET|Instrument.FLAG_PERSIST,
        type=types.BooleanType)
    
    self._autoupdater_handle = "current_source_autoupdater_%s" % (hashlib.md5(address).hexdigest()[:8])
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
    self.get_on()
    self.get_voltage()
    self.get_current()
    self.get_voltage_limit()
    self.get_current_limit()

  def _write(self, msg): return self._ask(msg, write_instead=True)
  def _ask(self, msg, write_instead=False):
    logging.debug('Sending %s', msg)
    
    for attempt in range(3):
      try:
        assert self._address.lower().startswith('asrl')
        self._visainstrument = visa.ResourceManager().open_resource(self._address,
                                 timeout=2000.) # milliseconds
        try:
          self._visainstrument.parity = visa.constants.Parity.none
          self._visainstrument.data_bits = 8
          self._visainstrument.stop_bits = visa.constants.StopBits.one
          
          self._visainstrument.read_termination = '\r\n'
          self._visainstrument.write_termination = '\n'
          
          r = self._visainstrument.write(msg) if write_instead else self._visainstrument.query(msg)

          qt.msleep(0.020) # minimum 10 ms between queries specified in the manual
          
          self._visainstrument.close()
          return r

        except:
          self._visainstrument.close()
          raise
        
      except:
        logging.exception('Attempt %d to communicate with turbo failed', attempt)
        qt.msleep(1. + attempt**2)

    assert False, 'All attempts to communicate with the current source failed.'

  def _log(self, param, value):
    if self._logger != None:
      try: self._logger(param, 'current_source', value)
      except Exception as e: logging.debug('Could not log current source %s = %s. %s', param, value, str(e))
    
  def do_get_on(self):
    rval = self._ask('OUT?').strip().upper()
    assert rval[:3] == 'OUT'
    rval = (rval[3:].strip() == 'ON')
    self._log('on', int(rval))
    return rval

  def do_set_on(self, val):
    self._write('ON' if val else 'OFF')
    self.get_on()

  def do_get_voltage(self):
    rval = self._ask('VO?').strip()
    assert rval[0] == 'V'
    rval = float(rval[1:])
    self._log('voltage', rval)
    return rval

  def do_get_current(self):
    rval = self._ask('IO?').strip()
    assert rval[0] == 'I'
    rval = float(rval[1:])
    self._log('current', rval)
    return rval

  def do_get_voltage_limit(self):
    rval = self._ask('V?').strip()
    assert rval[0] == 'V'
    rval = float(rval[1:])
    return rval

  def do_get_current_limit(self):
    rval = self._ask('I?').strip()
    assert rval[0] == 'I'
    rval = float(rval[1:])
    return rval
