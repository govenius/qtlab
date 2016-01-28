# DC power supply driver
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

class Rigol_DP1308A(Instrument):
  '''
  Driver for Rigol_DP1308A DC power supply.

  Usage:
  Initialize with
  <name> = instruments.create('name', 'Rigol_DP1308A', address='<VISA address>',
      reset=<bool>)
  '''

  def __init__(self, name, address, reset=False):
    '''
    Initializes the Rigol_DP1308A.

    Input:
        name (string)    : name of the instrument
        address (string) : GPIB address
        reset (bool)     : resets to default values, default=false

    Output:
        None
    '''
    logging.info(__name__ + ' : Initializing instrument Rigol_DP1308A (%s)', address)
    Instrument.__init__(self, name, tags=['physical'])
    self._address = address
    self._visa_min_time_between_commands = 0.020 # s
    self._visa_last_access_time = 0
    self._visa_reservation_counter = 0
    self._port_names = [ 'P6V', 'P25V', 'N25V' ]

    self.add_parameter('idn', type=types.StringType, flags=Instrument.FLAG_GET, format='%.10s')

    self.add_parameter('on', type=types.BooleanType, flags=Instrument.FLAG_GETSET,
                        channels=self._port_names, channel_prefix='%s_')
    self.add_parameter('ovp_on', type=types.BooleanType, flags=Instrument.FLAG_GETSET,
                        channels=self._port_names, channel_prefix='%s_')
    self.add_parameter('ocp_on', type=types.BooleanType, flags=Instrument.FLAG_GETSET,
                        channels=self._port_names, channel_prefix='%s_')

    self.add_parameter('voltage', type=types.FloatType, flags=Instrument.FLAG_GETSET,
                        minval=-25., maxval=25.,
                        units='V', format='%.3f',
                        channels=self._port_names, channel_prefix='%s_')

    self.add_parameter('current', type=types.FloatType, flags=Instrument.FLAG_GETSET,
                        minval=-5., maxval=5.,
                        units='A', format='%.3f',
                        channels=self._port_names, channel_prefix='%s_')

    self.add_parameter('measured_voltage', type=types.FloatType, flags=Instrument.FLAG_GET,
                        units='V', format='%.4f',
                        channels=self._port_names, channel_prefix='%s_')

    self.add_parameter('measured_current', type=types.FloatType, flags=Instrument.FLAG_GET,
                        units='A', format='%.4f',
                        channels=self._port_names, channel_prefix='%s_')

    self._voltage_ramp_step_size = 0.010
    self._current_ramp_step_size = 0.001
    self._ramp_step_duration = 0.500 # s
    self.add_parameter('voltage_ramp_step_size', type=types.FloatType,
        flags=Instrument.FLAG_GETSET,
        minval=0.001, maxval=100., units='V', format='%.4f')
    self.add_parameter('current_ramp_step_size', type=types.FloatType,
        flags=Instrument.FLAG_GETSET,
        minval=0.001, maxval=100., units='A', format='%.4f')
    self.add_parameter('ramp_step_duration', type=types.FloatType,
        flags=Instrument.FLAG_GETSET,
        minval=0., maxval=10., units='s', format='%.4f')

    self.add_function('reset')
    self.add_function('get_all')
    self.add_function('set_voltages')

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
      self.get_voltage_ramp_step_size()
      self.get_current_ramp_step_size()
      self.get_ramp_step_duration()

      for c in self._port_names:
        getattr(self, 'get_%s_on' % c)()
        getattr(self, 'get_%s_voltage' % c)()
        getattr(self, 'get_%s_current' % c)()
        getattr(self, 'get_%s_measured_voltage' % c)()
        getattr(self, 'get_%s_measured_current' % c)()
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
      self._visainstrument = visa.ResourceManager().open_resource(self._address, timeout=2000)
      self._visainstrument.read_termination = '\n'
      self._visainstrument.write_termination = '\n'

  def _release_visa(self):
    ''' Counter based opening/closing of the visa connection. '''
    assert self._visa_reservation_counter > 0, 'Trying to release a visa session that has not been reserved! (counter = %s)' % (self._visa_reservation_counter)
    self._visa_reservation_counter -= 1
    self._visa_last_access_time = time.time()
    if self._visa_reservation_counter == 0:
      self._visainstrument.close()

  def _ask(self, cmd):
    return self._write(cmd, ask_instead=True)

  def _write(self, cmd, ask_instead=False):
    max_attempts = 3
    for attempt in range(max_attempts):

      try:
        self._reserve_visa()

        try:
          if ask_instead:
            return self._visainstrument.query(cmd)
          else:
            self._visainstrument.write(cmd)
            return
        except Exception as e:
          logging.exception('Failed to write.')
          if attempt == max_attempts-1: raise
          time.sleep( .5+attempt )

      finally:
        self._release_visa()

  def set_currents(self, port_to_current, update_attribute_value=True):
    '''
      Ramp current limits on multiple port in parallel.

      port_to_current --- should be a dictionary { port name: voltage },
                          e.g. {'P6V': 0.5, 'N25V': -0.2}
    '''
    self.set_voltages(port_to_current, update_attribute_value, currents_instead=True)

  def set_voltages(self, port_to_voltage, update_attribute_value=True, currents_instead=False):
    '''
      Ramp voltage limits on multiple port in parallel.

      port_to_voltage --- should be a dictionary { port name: current },
                          e.g. {'P6V': 0.5, 'N25V': -15.}
    '''

    try:
      self._reserve_visa()

      if currents_instead:
        for port in port_to_voltage.keys():
          if port == 'N25V': assert port_to_voltage[port] <= 0, 'N25V current must be negative.'
          else: assert port_to_voltage[port] >= 0, '%s current must be negative.' % port
          assert np.abs(port_to_voltage[port]) <= {'P6V': 5., 'P25V': 1., 'N25V': 1.}[port], '%s A is too high for %s' % (port_to_voltage[port], port)
      else:
        for port in port_to_voltage.keys():
          if port == 'N25V': assert port_to_voltage[port] <= 0, 'Voltage must be negative for %s.' % port
          else: assert port_to_voltage[port] >= 0, 'Voltage must be positive for %s.' % port
          assert np.abs(port_to_voltage[port]) <= {'P6V': 6., 'P25V': 25., 'N25V': 25.}[port], '%s V is too high for %s' % (port_to_voltage[port], port)

      stepsize = self.get_current_ramp_step_size() if currents_instead else self.get_voltage_ramp_step_size()
      delay = self.get_ramp_step_duration()
      prev_step_started_at = time.time()

      target = {}
      old = {}
      old_voltage = {}
      old_current = {}
      steps = {}
      done = {}
      for port in port_to_voltage.keys():

        target[port] = np.round(port_to_voltage[port], decimals=3)
        if np.abs(target[port]) < 1e-4: target[port] = 0.   # remove minus sign in front of zero
        logging.debug(__name__ + ' : setting port %s voltage to %s V' % (port, target[port]))

        old_voltage[port] = self._get_voltage(port)
        old_current[port] = self._get_current(port)
        old[port] = old_current[port] if currents_instead else old_voltage[port]
        steps[port] = np.linspace(old[port], target[port], 2 + int(np.abs(target[port]-old[port])/stepsize))
        done[port] = False

      max_steps = max( len(s) for s in steps.values() )
      assert max_steps >= 2
      for i in range(1, max_steps):
        for port in port_to_voltage.keys():
          if done[port]: continue # This port is already done ramping

          if i < len(steps[port]):
            # Both V and I must be positive in APPL ...,V,I. Even for N25V.
            cmd = ( ('APPL %s,%.3f,%.3f' % (port, np.abs(old_voltage[port]), np.abs(steps[port][i])))
                    if currents_instead else
                    ('APPL %s,%.3f' % (port, np.abs(steps[port][i]))) )
            self._write(cmd)

            if i == len(steps[port]) - 1:
              done[port] = True
              if update_attribute_value:
                if currents_instead: self.update_value('%s_current' % port, float(steps[port][i]))
                else: self.update_value('%s_voltage' % port, float(steps[port][i]))

        if i < max_steps-1:
          time_to_sleep = ( delay
                            - (time.time() - prev_step_started_at) )
          qt.msleep(max(0.001, time_to_sleep))
          prev_step_started_at = time.time()

      if update_attribute_value:
        for port in port_to_voltage.keys():
          new = getattr(self, 'get_%s_current' % port)() if currents_instead else getattr(self, 'get_%s_voltage' % port)()
          assert ( np.abs(new - target[port]) < 1e-6 )

    finally:
      self._release_visa()

  def _get_voltage_and_current(self, port):
    r = self._ask('APPL? %s' % port)
    r = r.split(',')
    #logging.debug(r)
    assert len(r) == 4, r
    assert r[2].strip()[-1] == 'V', r
    assert r[3].strip()[-1] == 'A', r
    r = np.array([ float(x.strip()[:-1]) for x in r[2:] ])

    # Note that APPL? returns a positive current but a NEGATIVE voltage for N25V
    # (even though you have to specify a positive voltage when you're setting it with APPL)
    if port=='N25V':
      r[0] = -np.abs(r[0])
      r[1] = -np.abs(r[1])
      if np.abs(r[0]) == 0: r[0] = 0 # convert -0 to 0
      if np.abs(r[1]) == 0: r[1] = 0 # convert -0 to 0
    
    return r

  def _get_voltage(self, port): return self._get_voltage_and_current(port)[0]
  def _get_current(self, port): return self._get_voltage_and_current(port)[1]
  
  def do_get_idn(self): return self._ask('*IDN?')

  def do_get_voltage(self, channel): return self._get_voltage(channel)
  def do_set_voltage(self, val, channel): self.set_voltages({channel: val}, update_attribute_value=False)

  def do_get_current(self, channel): return self._get_current(channel)
  def do_set_current(self, val, channel): self.set_currents({channel: val}, update_attribute_value=False)

  def do_get_measured_voltage(self, channel):
    r = float(self._ask('MEAS:VOLT? %s' % channel))
    if channel=='N25V': r *= -1
    return r

  def do_get_measured_current(self, channel):
    r = float(self._ask('MEAS:CURR? %s' % channel))
    if channel=='N25V': r *= -1
    return r

  def do_set_voltage_ramp_step_size(self, stepsize): self._voltage_ramp_step_size = stepsize
  def do_get_voltage_ramp_step_size(self): return self._voltage_ramp_step_size

  def do_set_current_ramp_step_size(self, stepsize): self._current_ramp_step_size = stepsize
  def do_get_current_ramp_step_size(self): return self._current_ramp_step_size

  def do_set_ramp_step_duration(self, delay): self._ramp_step_duration = delay
  def do_get_ramp_step_duration(self): return self._ramp_step_duration

  def do_get_on(self, channel): return self._ask('OUTP? %s' % channel).strip().upper() == 'ON'
  def do_set_on(self, val, channel): self._write('OUTP %s,%s' % (channel, 'ON' if val else 'OFF'))

  def do_get_ovp_on(self, channel): return self._ask('OUTP:OVP:STAT? %s' % channel).strip().upper() == 'ON'
  def do_set_ovp_on(self, val, channel):
    assert self.do_get_on(channel), 'OVP can be toggled only when the channel (%s) is on.' % channel
    self._write('OUTP:OVP:STAT %s,%s' % (channel, 'ON' if val else 'OFF'))

  def do_get_ocp_on(self, channel): return self._ask('OUTP:OCP:STAT? %s' % channel).strip().upper() == 'ON'
  def do_set_ocp_on(self, val, channel):
    assert self.do_get_on(channel), 'OCP can be toggled only when the channel (%s) is on.' % channel
    self._write('OUTP:OCP:STAT %s,%s' % (channel, 'ON' if val else 'OFF'))

  #def do_get_ovp_limit(self, channel): return float(self._ask('OUTP:OVP? %s' % channel))
  #def do_get_ocp_limit(self, channel): return float(self._ask('OUTP:OCP? %s' % channel))
