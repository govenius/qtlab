# Pieter de Groot <pieterdegroot@gmail.com>, 2008
# Martijn Schaafsma <qtlab@mcschaafsma.nl>, 2008
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
import qt

class SRS_SG396(Instrument):
  '''
  This is the driver for the SRS SG396 Vector Signal Genarator

  Usage:
  Initialize with
  <name> = instruments.create('<name>', 'SRS_SG396', address='<GBIP address>, reset=<bool>')
  '''

  def __init__(self, name, address, reset=False):
    '''
    Initializes the SRS_SG396, and communicates with the wrapper.

    Input:
      name (string)    : name of the instrument
      address (string) : GPIB address
      reset (bool)     : resets to default values, default=False
    '''
    logging.info(__name__ + ' : Initializing instrument SRS_SG396')
    Instrument.__init__(self, name, tags=['physical'])

    # Add some global constants
    self._address = address
    self._visainstrument = visa.ResourceManager().open_resource(self._address, timeout=2000) # timeout is in milliseconds
    try:
      self._visainstrument.read_termination = '\n'
      self._visainstrument.write_termination = '\n'

      self.MAX_BNC_FREQ = 62.5e6
      self.MIN_N_FREQ = 950e3

      self.add_parameter('power',
                         flags=Instrument.FLAG_GETSET|Instrument.FLAG_GET_AFTER_SET,
                         units='dBm', minval=-110, maxval=16.5, type=types.FloatType)
      self.add_parameter('phase',
                         flags=Instrument.FLAG_GETSET|Instrument.FLAG_GET_AFTER_SET,
                         units='deg', minval=-360, maxval=360, type=types.FloatType)
      self.add_parameter('frequency', format='%.09e',
                         flags=Instrument.FLAG_GETSET|Instrument.FLAG_GET_AFTER_SET,
                         units='Hz', minval=0, maxval=6.075e9, type=types.FloatType)
                         #cache_time=1.) # <-- cache because this is queried a lot when setting other params
      self.add_parameter('dc_offset',
                         flags=Instrument.FLAG_GETSET|Instrument.FLAG_GET_AFTER_SET,
                         units='V', minval=-1.5, maxval=1.5, type=types.FloatType)

      self.add_parameter('idn', flags=Instrument.FLAG_GET, type=types.StringType)
      self.add_parameter('temperature', flags=Instrument.FLAG_GET, units='deg C', type=types.FloatType)

      self.add_parameter('status',
          flags=Instrument.FLAG_GETSET|Instrument.FLAG_GET_AFTER_SET, type=types.StringType,
          format_map={'on': 'output on',
                      'off': 'output off'})

      self.add_parameter('modulation',
          flags=Instrument.FLAG_GETSET|Instrument.FLAG_GET_AFTER_SET, type=types.IntType,
          format_map={-1: 'no modulation',
                      0: 'AM / ASK',
                      1: 'FM / FSK',
                      2: 'phase / PSK',
                      3: 'sweep',
                      4: 'pulse',
                      5: 'blank',
                      7: 'QAM',
                      8: 'CPM',
                      9: 'VSB'})
      self.add_parameter('modulation_subtype',
          flags=Instrument.FLAG_GETSET|Instrument.FLAG_GET_AFTER_SET, type=types.IntType,
          format_map={0: 'analog (no constellation mapping)',
                      1: 'vector (no constellation mapping)',
                      2: 'default 1-bit constellation',
                      3: 'default 2-bit constellation',
                      4: 'default 3-bit constellation',
                      5: 'default 4-bit constellation',
                      6: 'default 5-bit constellation',
                      7: 'default 6-bit constellation',
                      8: 'default 7-bit constellation',
                      9: 'default 8-bit constellation',
                      10: 'default 9-bit constellation',
                      1: 'user constellation',
                      1: 'factory OQPSK constellation',
                      1: 'factory QPSK constellation',
                      1: 'factory pi/4 DQPSK constellation',
                      1: 'factor 3pi/8 8PSK constellation'})

      self.add_parameter('external_modulation_coupling',
          flags=Instrument.FLAG_GETSET|Instrument.FLAG_GET_AFTER_SET, type=types.IntType,
          format_map={0: 'AC (4 Hz high-pass)',
                      1: 'DC'})

      self.add_parameter('pulse_modulation',
          flags=Instrument.FLAG_GETSET|Instrument.FLAG_GET_AFTER_SET, type=types.IntType,
          format_map={3: 'square',
                      4: 'noise (PRBS)',
                      5: 'external',
                      11: 'user waveform'})
      self.add_parameter('pulse_modulation_width', format='%.09e',
                         flags=Instrument.FLAG_GETSET|Instrument.FLAG_GET_AFTER_SET,
                         units='s', minval=1e-6, maxval=10., type=types.FloatType)
      self.add_parameter('pulse_modulation_period', format='%.09e',
                         flags=Instrument.FLAG_GETSET|Instrument.FLAG_GET_AFTER_SET,
                         units='s', minval=1e-6, maxval=10., type=types.FloatType)

      self.add_parameter('am_modulation_depth',
          flags=Instrument.FLAG_GETSET|Instrument.FLAG_GET_AFTER_SET, units='%', minval=0, maxval=100., type=types.FloatType)

      self.add_parameter('timebase',
          flags=Instrument.FLAG_GETSET|Instrument.FLAG_GET_AFTER_SET, type=types.IntType,
          format_map={0: 'crystal',
                      1: 'OCXO',
                      2: 'rubidium',
                      3: 'external'})

      self.add_parameter('noise_mode',
          flags=Instrument.FLAG_GETSET|Instrument.FLAG_GET_AFTER_SET, type=types.IntType,
          format_map={0: 'optimized for less than 100 kHz from carrier',
                      1: 'optimized for more than 100 kHz from carrier'})

      status_codes ={0: 'normal',
                    1: '20 MHz PLL unlocked',
                    2: '100 MHz PLL unlocked',
                    4: '19 MHz PLL unlocked',
                    8: '1 GHz PLL unlocked',
                    16: '4 GHz PLL unlocked',
                    32: 'no timebase',
                    64: 'rubidium oscillator unlocked',
                    128: 'reserved status code',
                    256: 'modulation overloaded',
                    512: 'IQ modulation overloaded'}
      self.__all_status_combinations = dict(
          (status, ', '.join( status_codes[2**j] for j in range(0,8) if ((status>>j)&1) ))
          for status in range(1, 2**8)
        )
      self.__all_status_combinations[0] = status_codes[0]
      self.add_parameter('status_code',
          flags=Instrument.FLAG_GET, type=types.IntType,
          format_map=self.__all_status_combinations)

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
    Resets the instrument to default values.
    '''
    logging.info(__name__ + ' : resetting instrument')
    self._visainstrument.write('*RST')
    self.set_status('off')
    self.set_power(-30)
    self.get_all()

  def self_test(self):
    '''
    Run the self test and report errors.
    '''
    logging.info(__name__ + ' : self testing')
    errors = self._visainstrument.ask('*TST?')
    if errors.strip() != '0':
      logging.warn('Self test returned the following errors!:')
      assert errors.strip() == '17', 'Self test should return either 0 or 17.'
      self.check_for_errors()

    self.set_status('off')
    self.set_power(-30)

    logging.warn('Self test done. The status byte will show that the PLLs came unlocked but that\'s normal (on first read).')
    self.get_all()

  def check_for_errors(self):
    '''
    Check the error queue.
    '''
    for i in range(1000):
      err = self._visainstrument.ask('LERR?')
      if err.strip() == '0':
        return
      else:
        logging.warn('error_buffer[-%d] = %s (see manual p. 126)', i, err)
        qt.msleep(0.2)

  def get_all(self):
    '''
    Reads all implemented parameters.
    '''
    logging.info(__name__ + ' : get all')
    self.get_idn()
    self.get_power()
    self.get_dc_offset()
    self.get_phase()
    self.get_frequency()
    self.get_status()

    self.get_external_modulation_coupling()
    self.get_modulation()
    self.get_modulation_subtype()

    self.get_am_modulation_depth()

    self.get_pulse_modulation()
    self.get_pulse_modulation_width()
    self.get_pulse_modulation_period()

    self.get_noise_mode()
    self.get_timebase()
    self.get_status_code()
    self.get_temperature()

    self.check_for_errors()

  def __to_rounded_string(self, x, decimals, significant_figures):
    ''' Round x to the specified number of decimals and significant figures.
        Output a warning if rounded value is not equal to x. '''
    rounded = ('%.{0}e'.format(significant_figures-1)) % ( np.round(x, decimals=decimals) )
    if np.abs(float(rounded) - x) > np.finfo(np.float).tiny:
      logging.warn('Rounding the requested value (%.20e) to %s (i.e. by %.20e).',
                   x, rounded, x - float(rounded))
    return rounded

  def do_get_idn(self):
    '''
    Get a string identifying the instrument.
    '''
    return self._visainstrument.ask('*IDN?')

  def do_get_temperature(self):
    '''
    Temperature of the RF output block in deg C.
    '''
    return self._visainstrument.ask('TEMP?')

  def do_get_power(self):
    '''
    Reads the power of the signal in dBm.
    '''
    logging.debug(__name__ + ' : get power')
    return float(self._visainstrument.ask('AMPR?'))

  def do_set_power(self, amp):
    '''
    Set the power of the signal in dBm.
    '''
    p = self.__to_rounded_string(amp, 2, 6)
    logging.debug(__name__ + ' : set power to %s' % p)

    max_power = min(16.5, 16.5 - (self.get_frequency() - 4e9)/1e9 * 3.25)
    if float(p) > max_power:
      logging.warn('Trying to set %s dBm but the maximum power at %g Hz is %g dBm',
                   p, self.get_frequency(), max_power)
      
    if self.get_frequency() >= self.MIN_N_FREQ: self._visainstrument.write('AMPR %s' % p)
    if self.get_frequency() <= self.MAX_BNC_FREQ: self._visainstrument.write('AMPL %s' % p)

  def do_get_dc_offset(self):
    '''
    Reads the DC offset of the BNC output in V.
    '''
    logging.debug(__name__ + ' : get dc_offset')
    return float(self._visainstrument.ask('OFSL?'))

  def do_set_dc_offset(self, off):
    '''
    Set the DC offset of the BNC output in V.
    '''
    p = self.__to_rounded_string(off, 2, 6)
    logging.debug(__name__ + ' : set dc_offset to %s' % p)

    power_in_Vrms = 1.00 * 10**((self.get_power()-13.01) / 20)
    max_dc_offset = 1.5 * min(1, 1 - (power_in_Vrms-0.224)/(1-0.224) )
    if np.abs(float(p)) > max_dc_offset:
      logging.warn('Trying to set %s V but the maximum DC offset at %g dBm AC power is %g V',
                   p, self.get_power(), max_dc_offset)
    
    self._visainstrument.write('OFSL %s' % p)

  def do_get_phase(self):
    '''
    Reads the phase of the signal in degrees.
    '''
    logging.debug(__name__ + ' : get phase')
    return float(self._visainstrument.ask('PHAS?'))

  def do_set_phase(self, v):
    '''
    Set the phase of the signal in degress.
    '''
    if   self.get_frequency() <= 100e6: p = self.__to_rounded_string(v, 2, 10)
    elif self.get_frequency() <= 1e9:   p = self.__to_rounded_string(v, 1, 10)
    else:                               p = self.__to_rounded_string(v, 0, 10)
    logging.debug(__name__ + ' : set phase to %s' % p)
    self._visainstrument.write('PHAS %s' % p)

  def do_get_frequency(self):
    '''
    Reads the frequency of the signal in Hz.
    '''
    logging.debug(__name__ + ' : get frequency')
    return float(self._visainstrument.ask('FREQ?'))

  def do_set_frequency(self, freq):
    '''
    Set the frequency in Hz.
    '''
    logging.debug(__name__ + ' : set frequency to %s' % freq)

    self._visainstrument.write('FREQ %s' %
                               self.__to_rounded_string(freq, decimals=6, significant_figures=17))

  def do_get_status(self):
    '''
    Reads the output status ("on" or "off" for the BNC and N outputs).
    '''
    logging.debug(__name__ + ' : get status')
    stat_l = self._visainstrument.ask('ENBL?').strip() == '1'
    stat_r = self._visainstrument.ask('ENBR?').strip() == '1'
    return 'BNC %s, N %s' % ('on' if stat_l else 'off', 'on' if stat_r else 'off')

  def do_set_status(self, status):
    '''
    Sets the output status ("on" or "off").
    Sets both the BNC and N outputs whenever the frequency allows it.
    '''
    s = int(bool( (status.lower().strip() in ['1', 'on']) ))
    logging.debug(__name__ + ' : set status to %s' % s)
    if self.get_frequency() >= self.MIN_N_FREQ: self._visainstrument.write('ENBR %s' % s)
    if self.get_frequency() <= self.MAX_BNC_FREQ: self._visainstrument.write('ENBL %s' % s)

  def do_get_modulation(self):
    '''
    Gets the modulation mode (and whether modulation is enabled at all).
    '''
    if not (self._visainstrument.ask('MODL?').strip() == '1'): return -1
    return self._visainstrument.ask('TYPE?')

  def do_set_modulation(self, v):
    '''
    Sets the modulation mode (and turns modulation on), unless v == -1 which disables modulation.
    '''
    assert v in [-1,0,1,2,3,4,5,7,8,9], 'Unknown modulation type %s (see manual p. 114).' % v
    if v != -1: self._visainstrument.write('TYPE %d' % v)
    self._visainstrument.write('MODL %d' % (0 if v == -1 else 1))

  def do_get_modulation_subtype(self):
    '''
    Gets the modulation subtype.
    '''
    return self._visainstrument.ask('STYP?')

  def do_set_modulation_subtype(self, v):
    '''
    Sets the modulation subtype.
    '''
    assert v in range(16), 'Unknown modulation subtype %s (see manual p. 113).' % v
    self._visainstrument.write('STYP %d', v)

  def do_get_pulse_modulation(self): return self._visainstrument.ask('PFNC?')
  def do_set_pulse_modulation(self, v): return self._visainstrument.write('PFNC %d' % v)

  def do_get_pulse_modulation_width(self):
    return float( self._visainstrument.ask('PWID?') )
  def do_set_pulse_modulation_width(self, w):
    self._visainstrument.write('PWID %s' % (
        self.__to_rounded_string(w, decimals=9, significant_figures=9) ))

  def do_get_pulse_modulation_period(self):
    return float( self._visainstrument.ask('PPER?') )
  def do_set_pulse_modulation_period(self, w):
    self._visainstrument.write('PPER %s' % (
        self.__to_rounded_string(w, decimals=9, significant_figures=9) ))

  def do_get_am_modulation_depth(self):
    stat = self._visainstrument.ask('ADEP?')
    return stat.lower()
  def do_set_am_modulation_depth(self, val):
    self._visainstrument.write('ADEP %s' % self.__to_rounded_string(w, decimals=2, significant_figures=5))

  def do_get_external_modulation_coupling(self): return self._visainstrument.ask('COUP?')
  def do_set_external_modulation_coupling(self, v): return self._visainstrument.write('COUP %d' % v)

  def do_get_noise_mode(self): return self._visainstrument.ask('NOIS?')
  def do_set_noise_mode(self, v): return self._visainstrument.write('NOIS %d' % v)

  def do_get_timebase(self): return self._visainstrument.ask('TIMB?')
  def do_set_timebase(self, v): return self._visainstrument.write('TIMB %d' % v)

  def do_get_status_code(self):
    stat = self._visainstrument.ask('INSR?')
    if stat.strip() != '0':
      try: explanation = self.__all_status_combinations[int(stat)]
      except: explanation = 'unknown status code'
      logging.warn('The instrument status register was %s: %s. (The status is reset on each read.)',
                   stat, explanation)
    return stat
