# Arbitrary function generator driver
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
import qt
import visa
import types
import logging
import time
import numpy as np
import base64
import struct
import itertools
import re
from scipy import interpolate

class Siglent_SDG5082(Instrument):
  '''
  Driver for the Siglent SDG5082 arbitrary function generator.

  Usage:
  Initialize with
  <name> = instruments.create('name', 'Siglent_SDG5082', address='<VISA address>',
      reset=<bool>)
  '''

  def __init__(self, name, address, reset=False):
    '''
    Initializes the Siglent_SDG5082.

    Input:
        name (string)    : name of the instrument
        address (string) : VISA address
        reset (bool)     : resets to default values, default=false

    Output:
        None
    '''
    logging.info(__name__ + ' : Initializing instrument Siglent_SDG5082 (%s)', address)
    Instrument.__init__(self, name, tags=['physical'])
    self._address = address
    self._visa_default_timeout = 10000 # ms
    self._visa_min_time_between_commands = 0.020 # s
    self._visa_last_access_time = 0
    self._visa_reservation_counter = 0
    self._channels = (1,2)
    self._memory_location = {1: 'M36', 2: 'M60'} # fixed arb. wave memory locations for each channel

    self.add_parameter('idn', type=types.StringType, flags=Instrument.FLAG_GET, format='%.10s')

    self.add_parameter('output', type=types.BooleanType,
                       flags=Instrument.FLAG_GETSET|Instrument.FLAG_GET_AFTER_SET,
                       channels=self._channels, channel_prefix='ch%d_')
    self.add_parameter('modulation', type=types.BooleanType,
                       flags=Instrument.FLAG_GETSET|Instrument.FLAG_GET_AFTER_SET,
                       channels=self._channels, channel_prefix='ch%d_')
    self.add_parameter('load_impedance', type=types.FloatType,
                       flags=Instrument.FLAG_GETSET|Instrument.FLAG_GET_AFTER_SET,
                       channels=self._channels, minval=50, units='Ohm', channel_prefix='ch%d_')
    self.add_parameter('polarity', type=types.StringType,
                       flags=Instrument.FLAG_GETSET|Instrument.FLAG_GET_AFTER_SET,
                       channels=self._channels, channel_prefix='ch%d_',
                       format_map={"INV" : "inverted","NORM" : "normal"})
    self.add_parameter('frequency', type=types.FloatType, format='%.6e',
                       flags=Instrument.FLAG_GETSET|Instrument.FLAG_GET_AFTER_SET,
                       channels=self._channels, minval=1e-6, maxval=80e6, units='Hz', channel_prefix='ch%d_')
    self.add_parameter('amplitude', type=types.FloatType,
                       flags=Instrument.FLAG_GETSET|Instrument.FLAG_GET_AFTER_SET,
                       channels=self._channels, minval=0.004, maxval=10, units='V', channel_prefix='ch%d_')
    self.add_parameter('offset', type=types.FloatType,
                       flags=Instrument.FLAG_GETSET|Instrument.FLAG_GET_AFTER_SET,
                       channels=self._channels, minval=-10, maxval=10, units='V', channel_prefix='ch%d_')
    self.add_parameter('shape', type=types.StringType,
                       flags=Instrument.FLAG_GETSET|Instrument.FLAG_GET_AFTER_SET,
                       channels=self._channels, channel_prefix='ch%d_',
                       format_map = {
                         "SIN" : "Sine",
                         "SQU" : "Square",
                         "RAMP" : "Ramp",
                         "PULS" : "Pulse",
                         "NOISE" : "Noise",
                         "DC" : "DC",
                         "ARB" :"Arbitrary" } )
    self.add_parameter('waveform_data', type=types.StringType, format='%.10s',
                       flags=Instrument.FLAG_GET,
                       channels=self._channels, channel_prefix='ch%d_')
    self.add_parameter('ref_clock_mode', type=types.StringType,
                       flags=Instrument.FLAG_GETSET|Instrument.FLAG_GET_AFTER_SET,
                       format_map = { "INT" : "internal",
                                      "EXT" : "external" })

    self.add_function('reset')
    self.add_function('get_all')

    if reset: self.reset()

    self.get_all()

  #def release(self):
  #  self._visainstrument.close()

  def reset(self):
    '''
    Resets the instrument to default values

    Input:
        None

    Output:
        None
    '''
    logging.info(__name__ + ' : Resetting instrument')
    self._write('*RST')
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
    logging.info(__name__ + ' : reading all settings from instrument')

    try:
      self._reserve_visa()
      self.get_idn()
      self.get_ref_clock_mode()

      for c in self._channels:
        getattr(self, 'get_ch%s_output' % c)()
        getattr(self, 'get_ch%s_modulation' % c)()
        getattr(self, 'get_ch%s_load_impedance' % c)()
        getattr(self, 'get_ch%s_polarity' % c)()
        getattr(self, 'get_ch%s_frequency' % c)()
        getattr(self, 'get_ch%s_amplitude' % c)()
        getattr(self, 'get_ch%s_offset' % c)()
        getattr(self, 'get_ch%s_shape' % c)()
        getattr(self, 'get_ch%s_waveform_data' % c)()
    finally:
      self._release_visa()

  def _reserve_visa(self):
    '''
    Counter based opening/closing of the visa connection.

    Using _reserve_visa() and _release_visa explicitly
    prevents the session from being closed between each command.
    Used, e.g., in get_all().
    '''
    self._visa_reservation_counter += 1
    time_to_sleep = ( self._visa_min_time_between_commands
                      - (time.time() - self._visa_last_access_time) )
    if time_to_sleep > 0: qt.msleep(time_to_sleep)
    if self._visa_reservation_counter == 1:
      self._visainstrument = visa.ResourceManager().open_resource(self._address, timeout=self._visa_default_timeout)
      self._visainstrument.read_termination = '\n'
      self._visainstrument.write_termination = '\n'

  def _release_visa(self):
    ''' Counter based opening/closing of the visa connection. '''
    assert self._visa_reservation_counter > 0, 'Trying to release a visa session that has not been reserved! (counter = %s)' % (self._visa_reservation_counter)
    self._visa_reservation_counter -= 1
    self._visa_last_access_time = time.time()
    if self._visa_reservation_counter == 0:
      self._visainstrument.close()

  def _query(self, cmd, raw=False, expected_raw_bytes=(lambda m: np.inf)):
    '''
    expected_raw_bytes -- an optional function that tells how many bytes
                          are to be expected given the bytes received so
                          far. Only applicable if raw=True.
    '''
    return self._write(cmd, query_instead=True, raw=raw,
                       expected_raw_bytes=expected_raw_bytes)

  def _write(self, cmd, query_instead=False, raw=False, expected_raw_bytes=None):
    max_attempts = 1

    try:
      self._reserve_visa()

      try:
        if query_instead:
          if raw:
            self._visainstrument.write(cmd)
            r = ''
            max_read_attempts = 10000

            #self._visainstrument.timeout = 1000
            for read_attempt in range(max_read_attempts):
              try:
                r += self._visainstrument.read_raw()
                expected_len = expected_raw_bytes(r)
                if len(r) >= expected_len: break
                if read_attempt == max_read_attempts-1: raise Exception('still getting data after %s read attempts!' % read_attempt)

              except visa.VisaIOError as e:
                if e.message.strip().startswith('VI_ERROR_TMO'):
                  if len(r) > 0 and np.isinf(expected_len):
                    break # normal if no expected length is specified
                  else:
                    logging.warn('%d bytes received' % len(r))
                    raise # response shorter than expected
                else:
                  raise # non-timeout error

            return r
          else:
            return self._visainstrument.query(cmd)

        else:
          if raw:
            self._visainstrument.write_raw(cmd)
          else:
            self._visainstrument.write(cmd)
          return

      except Exception as e:
        raise
        #logging.exception('Failed to %s.' % ('query' if query_instead else 'write'))
        #if attempt == max_attempts-1: raise
        #time.sleep( .5 )

    finally:
      self._visainstrument.timeout = self._visa_default_timeout
      self._release_visa()

  def _round(self, x, decimals, significant_figures):
    ''' Round x to the specified number of decimals and significant figures.
        Output a warning if rounded value is not equal to x. '''
    rounded = ('%.{0}e'.format(significant_figures-1)) % ( np.round(x, decimals=decimals) )
    if np.abs(float(rounded) - x) > 10*np.finfo(np.float).tiny:
      logging.warn('Rounding the requested value (%.20e) to %s (i.e. by %.20e).' % (x, rounded, x - float(rounded)))
    return rounded

  def _get_outp_state(self, channel):
    r = self._query('C%s:OUTP?'%channel).strip().upper().strip('C%s:OUTP'%channel).strip().split(',')
    r = [ rr.strip().upper() for rr in r ]
    assert r[0] in ['ON', 'OFF'], r
    assert r[1] == 'LOAD', r
    assert r[3] == 'PLRT', r
    return {'state': r[0], 'load': r[2], 'polarity': r[4]}

  def _get_waveform_params(self, channel):
    r = self._query('C%s:BSWV?'%channel).strip().upper().strip('C%s:BSWV'%channel).strip().split(',')
    r = [ rr.strip().upper() for rr in r ]
    assert r[0] == 'WVTP', r
    assert r[2] == 'FRQ', r
    assert r[4] == 'PERI', r
    assert r[6] == 'AMP', r
    assert r[8] == 'OFST', r
    assert r[10] == 'HLEV', r
    assert r[12] == 'LLEV', r
    assert r[14] == 'PHSE', r
    return {'type': r[1], 'frequency': r[3].strip('HZ'), 'period': r[5].strip('S'),
            'amplitude': r[7].strip('V'), 'offset': r[9].strip('V'),
            'hlevel': r[11].strip('V'), 'llevel': r[13].strip('V'), 'phase': r[15]}

  def do_get_idn(self): return self._query('*IDN?').strip().strip('*IDN').strip()

  def do_set_output(self, state, channel):
    '''
    Whether the output is on.
    '''
    logging.debug(__name__ + ' : Set channel %s output state to %s', channel, state)
    self._write('C%s:OUTP %s' % (channel, 'ON' if state else 'OFF'))
  def do_get_output(self, channel):
    '''
    Whether the output is on.
    '''
    logging.debug(__name__ + ' : Get channel %s output state', channel)
    return self._get_outp_state(channel)['state'] in ['1', 'ON']

  def do_set_modulation(self, state, channel):
    '''
    Whether the modulation is on.
    '''
    logging.debug(__name__ + ' : Set channel %s modulation state to %s', channel, state)
    self._write('C%s:MDWV STATE,%s' % (channel, 'ON' if state else 'OFF'))
  def do_get_modulation(self, channel):
    '''
    Whether the modulation is on.
    '''
    logging.debug(__name__ + ' : Get channel %s modulation state', channel)
    r = self._query('C%s:MDWV?'%channel).strip().upper().strip('C%s:MDWV'%channel).strip().split(',')
    r = [ rr.strip().upper() for rr in r ]
    assert r[0] == 'STATE', r
    assert r[1] in ['ON','OFF'], r
    return r[1] in ['1', 'ON']

  def do_set_load_impedance(self, val, channel):
    '''
    Output load impedance.
    '''
    logging.debug(__name__ + ' : Set channel %s load_impedance to %s', channel, val)
    assert np.abs(val - 50) < 1e-6 or (np.isinf(val) and val>0), 'Specify 50 or np.inf (Ohms).'
    self._write('C%s:OUTP LOAD,%s' % (channel, 'HZ' if np.isinf(val) else '50'))
  def do_get_load_impedance(self, channel):
    '''
    Output load impedance.
    '''
    logging.debug(__name__ + ' : Get channel %s load_impedance', channel)
    return np.inf if self._get_outp_state(channel)['load'] == 'HZ' else float(r[1])

  def do_set_polarity(self, val, channel):
    '''
    Output signal polarity.
    '''
    logging.debug(__name__ + ' : Set channel %s polarity to %s', channel, val)
    pol = val.strip().upper()[3]
    self._write('C%s:OUTP PLRT,%s' % (channel, 'NOR' if pol=='NOR' else 'INVT'))
  def do_get_polarity(self, channel):
    '''
    Output signal polarity.
    '''
    logging.debug(__name__ + ' : Get channel %s polarity', channel)
    return {'NOR': 'normal', 'INV': 'inverted'}[ self._get_outp_state(channel)['polarity'][:3] ]

  def do_set_frequency(self, val, channel):
    '''
    Output signal frequency.
    '''
    logging.debug(__name__ + ' : Set channel %s frequency to %s', channel, val)
    self._write('C%s:BSWV FRQ,%.6fHZ' % (channel, float(self._round(val, 6, 20))))
  def do_get_frequency(self, channel):
    '''
    Output signal frequency.
    '''
    logging.debug(__name__ + ' : Get channel %s frequency', channel)
    return float(self._get_waveform_params(channel)['frequency'])

  def do_set_amplitude(self, val, channel):
    '''
    Output signal amplitude.
    '''
    logging.debug(__name__ + ' : Set channel %s amplitude to %s', channel, val)
    self._write('C%s:BSWV AMP,%.3fV' % (channel, float(self._round(val/2., 3, 20))))
  def do_get_amplitude(self, channel):
    '''
    Output signal amplitude.
    '''
    logging.debug(__name__ + ' : Get channel %s amplitude', channel)
    return 2*float(self._get_waveform_params(channel)['amplitude'])

  def do_set_offset(self, val, channel):
    '''
    Output signal offset.
    '''
    logging.debug(__name__ + ' : Set channel %s offset to %s', channel, val)
    self._write('C%s:BSWV OFST,%.3fV' % (channel, float(self._round(val, 3, 20))))
  def do_get_offset(self, channel):
    '''
    Output signal offset.
    '''
    logging.debug(__name__ + ' : Get channel %s offset', channel)
    return float(self._get_waveform_params(channel)['offset'])

  def do_set_shape(self, val, channel):
    '''
    Output signal shape.
    '''
    logging.debug(__name__ + ' : Set channel %s shape to %s', channel, val)
    shape = {'SIN':'SINE', 'SQU':'SQUARE', 'RAMP':'RAMP',
            'PULS':'PULSE', 'NOISE': 'NOISE',
             'DC':'DC', 'ARB':'ARB'}[val]
    self._write('C%s:BSWV WVTP,%s' % (channel, shape))
  def do_get_shape(self, channel):
    '''
    Output signal shape.
    '''
    logging.debug(__name__ + ' : Get channel %s shape', channel)
    return {'SINE': 'SIN', 'SQUARE': 'SQU', 'RAMP': 'RAMP',
            'PULSE': 'PULS', 'NOISE': 'NOISE',
            'DC': 'DC', 'ARB': 'ARB'}[ self._get_waveform_params(channel)['type'] ]

  def do_get_ref_clock_mode(self):
    '''
    Reference clock mode (internal/external).
    '''
    r = self._query('ROSC?').strip()
    logging.debug(__name__ + ' : Get the reference clock mode: %s' % r)
    return r.strip('ROSC').strip().upper()[:3]

  def do_set_ref_clock_mode(self,val):
    '''
    Reference clock mode (internal/external).
    '''
    logging.debug(__name__ + ' : Set the reference clock mode. %s' % val)
    self._write('ROSC %s' % val.upper().strip()[:3])

  def do_get_waveform_data(self, channel):
    '''
    Waveform data encoded in base64.
    '''
    assert channel in [1,2]

    def expected_raw_bytes(x):
      if len(x) < 100:
        # definetly still expecting more data
        # so return a large but finite value
        return 2**21
      m = re.search('LENGTH,\s*(\d+?)\s*KB', x)
      assert m != None
      bytes_before_data = x.find('WAVEDATA,') + len('WAVEDATA,')
      if int(m.group(1)) == 32: return 2**15 + bytes_before_data
      elif int(m.group(1)) == 512: return 2**20 + bytes_before_data
      else: raise Exception('Unknown LENGTH = %s' % (m.group(1)))

    r = self._query('WVDT %s?' % (self._memory_location[channel]),
                    raw=True, expected_raw_bytes=expected_raw_bytes)
    assert 'WAVEDATA,' in r, 'unexpected reply: %s' % r
    header = r[:r.find('WAVEDATA,')]
    data = r[len(header) + len('WAVEDATA,'):]
    assert data[-1] == '\n', header
    data = data[:-1]

    return base64.b64encode(data)

  def set_ch1_waveform(self, waveform, resample=False): self.set_waveform(1, waveform, resample)
  def set_ch2_waveform(self, waveform, resample=False): self.set_waveform(2, waveform, resample)
  def set_waveform(self, channel, waveform, resample=False):
    '''
    Load the specified waveform (array of floats between plus/minus one)
    and update the waveform_data parameter.

    resample --- accept any waveform length > 1 and resample the waveform
                 to an acceptable number of points.
    '''
    if resample:
      assert len(waveform) > 2
      waveform = interpolate.interp1d(np.linspace(0,1,len(waveform)), waveform)(
        np.linspace(0, 1, 2**19 if channel==2 else 2**14)) # TODO: support 512K on CH2
        #np.linspace(0, 1, 2**14))

    if channel in [ 1 ]:
       # TODO: support 512K on CH2
      assert len(waveform) == 2**14, 'Only 16K point waveforms supported on CH1. (Use resample=True if you want to automatically interpolate between the provided points.)'
    if channel in [ 2 ]:
      assert len(waveform) == 2**14 or len(waveform) == 2**19, 'Only 16K or 512K point waveforms supported. (Use resample=True if you want to automatically interpolate between the provided points.)'

    buf = self._waveform_to_byte_array(waveform)
    assert len(buf) == 2*len(waveform), '%s != %s' % (len(buf), len(waveform))

    try:
      self._reserve_visa()
      params = self._get_waveform_params(channel)

      assert len(buf)/1024 in [32, 1024], len(buf)
      cmd = ('C%s:WVDT %s,WVNM,USER%d,TYPE,5,LENGTH,%dKB,FREQ,%.9f,AMPL,%.9f,OFST,%.9f,PHASE,%.9f,WAVEDATA,'
             % (channel,
                self._memory_location[channel],
                channel,
                len(buf)/1024,
                float(params['frequency']),
                float(params['amplitude']),
                float(params['offset']),
                float(params['phase']) ))
      logging.warn(cmd + ('[%d bytes of waveform data]'%len(buf)))
      self._write(b''.join([cmd, bytes(buf)]), raw=True)
      cmd = 'C%s:ARWV INDEX, %s' % (channel, self._memory_location[channel].upper().strip('M'))
      #self._write(cmd)

      self.get('ch%s_waveform_data' % channel)
    finally:
      self._release_visa()

  def waveform_data_to_waveform(self, waveform_data):
    '''
    Converts the base64 encoded array of bytes
    obtained by get_waveform_data() into a list
    of real numbers between plus and minus one.
    '''
    wf = base64.decodestring(waveform_data)
    wf = np.array(struct.unpack('<%uH' % (len(wf)/2), wf))
    wf = np.mod(wf - 2**13, 2**14)
    zero = 2**13
    return (wf - zero).astype(np.float) / 2**13

  def _waveform_to_byte_array(self, waveform):
    '''
    Converts an array of real numbers between plus/minus one to a byte array
    accepted by the AFG.
    '''
    assert min(waveform) >= -1
    assert max(waveform) <= 1
    buf = bytearray(2*len(waveform))
    zero = 2**13
    for i in range(len(waveform)):
      struct.pack_into('<H', buf, 2*i, np.mod(zero + min(2**14 - 1,
                                                         int(np.round((1+waveform[i]) * (2**13)))),
                                              2**14))
    return buf
