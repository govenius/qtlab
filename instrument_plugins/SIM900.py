# SIM_900.py, Stanford Research 900 Mainframe (for SIM928 voltage sources) driver
# Kuan Yen Tan <kuan.tan@aalto.fi>, 2012
# Joonas Govenius <joonas.govenius@aalto.fi>, 2013
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
import time
import numpy as np

class SIM900(Instrument):
  '''
  Driver for SIM900 mainframe from Stanford Research Systems.
  Only SIM928 voltage source modules are supported.

  Usage:
  Initialize with
  <name> = instruments.create('name', 'SIM900', address='<GPIB address>',
      reset=<bool>)
  '''

  def __init__(self, name, address, reset=False):
    '''
    Initializes the SIM900.

    Input:
        name (string)    : name of the instrument
        address (string) : GPIB address
        reset (bool)     : resets to default values, default=false

    Output:
        None
    '''
    logging.info(__name__ + ' : Initializing instrument SIM900')
    Instrument.__init__(self, name, tags=['physical'])
    self._address = address
    self._visainstrument = visa.ResourceManager().open_resource(self._address, timeout=2000)
    
    self._last_communication_time = time.time()
    
    # This is crucial, if you set this too low (e.g. 10ms) the SIM will simply stop responding sooner or later
    #   10 ms  --> SIM will stop responding after a few commands sent at this rate
    #   50 ms  --> may be OK (worked for a few hundred commands)
    #   150 ms --> problems still occur ~daily/weekly
    #   200 ms --> seems OK
    self._min_time_between_commands = 0.100  # in seconds

    for port in range(1,9):
      self._clear_output_buffer(port)

    self._ch_enabled = [True for i in range(8)]
    self.add_parameter('enabled', type=types.BooleanType, flags=Instrument.FLAG_GETSET,
                        channels=range(1,9), channel_prefix='port%d_')

    self.add_parameter('idn', type=types.StringType, flags=Instrument.FLAG_GET,
                        channels=range(1,9), channel_prefix='port%d_')

    self.add_parameter('voltage', type=types.FloatType, flags=Instrument.FLAG_GETSET,
                        minval=-20., maxval=20.,
                        units='V', format='%.04e',
                        channels=range(1,9), channel_prefix='port%d_')
                        
    self.add_parameter('on', type=types.BooleanType, flags=Instrument.FLAG_GETSET,
                        channels=range(1,9), channel_prefix='port%d_')

    self.add_parameter('battery_status', type=types.StringType, flags=Instrument.FLAG_GET,
                        channels=range(1,9), channel_prefix='port%d_')
                        
    self.add_parameter('PID_propotional_on', type=types.StringType, flags=Instrument.FLAG_GETSET,
                        channels=range(1,9), channel_prefix='port%d_')
    
    self.add_parameter('PID_propotional_gain', type=types.FloatType, flags=Instrument.FLAG_GETSET,
                        minval=0.1,maxval=1000.,
                        format='%.02e',
                        channels=range(1,9), channel_prefix='port%d_')
                        
    self.add_parameter('PID_integral_on', type=types.StringType, flags=Instrument.FLAG_GETSET,
                        channels=range(1,9), channel_prefix='port%d_')
    
    self.add_parameter('PID_integral_gain', type=types.FloatType, flags=Instrument.FLAG_GETSET,
                        minval=1e-2,maxval=5e5,
                        format='%.02e',
                        channels=range(1,9), channel_prefix='port%d_')
                        
                        
    self.add_parameter('PID_derivative_on', type=types.StringType, flags=Instrument.FLAG_GETSET,
                        channels=range(1,9), channel_prefix='port%d_')
    
    self.add_parameter('PID_derivative_gain', type=types.FloatType, flags=Instrument.FLAG_GETSET,
                        minval=1e-6,maxval=1e0,
                        format='%.02e',
                        channels=range(1,9), channel_prefix='port%d_')
		
    self.add_parameter('PID_offset_on', type=types.StringType, flags=Instrument.FLAG_GETSET,
                        channels=range(1,9), channel_prefix='port%d_')
    
    self.add_parameter('PID_offset_gain', type=types.FloatType, flags=Instrument.FLAG_GETSET,
                        minval=-10.,maxval=10,
                        format='%.02e',
                        channels=range(1,9), channel_prefix='port%d_')
    
    self.add_parameter('PID_manual_output_on', type=types.StringType, flags=Instrument.FLAG_GETSET,
                        channels=range(1,9), channel_prefix='port%d_')
    
    self.add_parameter('PID_manual_output', type=types.FloatType, flags=Instrument.FLAG_GETSET,
                        minval=-10.,maxval=10,
                        format='%.02e',
                        channels=range(1,9), channel_prefix='port%d_')
                         
    self.add_parameter('PID_upper_limit', type=types.FloatType, flags=Instrument.FLAG_GETSET,
                        minval=-10.,maxval=10,
                        format='%.02e',
                        channels=range(1,9), channel_prefix='port%d_')
                        
    self.add_parameter('PID_lower_limit', type=types.FloatType, flags=Instrument.FLAG_GETSET,
                        minval=-10.,maxval=10,
                        format='%.02e',
                        channels=range(1,9), channel_prefix='port%d_')
    
    self._ramp_stepsize = 2.0
    self._ramp_delaytime = self._min_time_between_commands
    self.add_parameter('ramp_stepsize', type=types.FloatType,
        flags=Instrument.FLAG_GETSET,
        minval=0., maxval=50., units='V', format='%.04e')
    self.add_parameter('ramp_delaytime', type=types.FloatType,
        flags=Instrument.FLAG_GETSET,
        minval=0., maxval=100., units='s', format='%.04e')

    self._retries_on_set_error = 3
    self.add_parameter('retries_on_set_error', type=types.IntType,
        flags=Instrument.FLAG_GETSET,
        minval=0, maxval=10)

    self.add_function('reset')
    self.add_function('get_all')
    self.add_function('set_port_voltage')
    self.add_function('get_port_voltage')
    self.add_function('set_port_on')
    self.add_function('get_port_on')
    self.add_function('set_port_PID_propotional_on')
    self.add_function('get_port_PID_propotional_on')
    self.add_function('set_port_PID_propotional_gain')
    self.add_function('get_port_PID_propotional_gain')
    self.add_function('set_port_PID_integral_on')
    self.add_function('get_port_PID_integral_on')
    self.add_function('set_port_PID_integral_gain')
    self.add_function('get_port_PID_integral_gain')
    self.add_function('set_port_PID_derivative_on')
    self.add_function('get_port_PID_derivative_on')
    self.add_function('set_port_PID_derivative_gain')
    self.add_function('get_port_PID_derivative_gain')
    self.add_function('set_port_PID_offset_on')
    self.add_function('get_port_PID_offset_on')
    self.add_function('set_port_PID_offset_gain')
    self.add_function('get_port_PID_offset_gain')
    self.add_function('set_port_PID_manual_output_on')
    self.add_function('get_port_PID_manual_output_on')
    self.add_function('set_port_PID_manual_output')
    self.add_function('get_port_PID_manual_output')
    self.add_function('set_port_PID_upper_limit')
    self.add_function('get_port_PID_upper_limit')
    self.add_function('set_port_PID_lower_limit')
    self.add_function('get_port_PID_lower_limit')

    if reset: self.reset()
    
    self._set_baud_rate()
    self.disable_disconnected_ports()
    self.get_all()

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
    #self.get_all()

  def disable_disconnected_ports(self):
    '''
    Try to infer which ports in the mainframe are unused.

    Input:
        None

    Output:
        None
    '''
    for c in range(1,9):
      try:
        idn = getattr(self, 'get_port%d_idn' % c)()
        if len(idn) < 3: raise Exception('Invalid response to *IDN? query: "%s"' % idn)
      except Exception as e:
        logging.info('Disabling port %d. Reason: %s' % (c, str(e)))
        getattr(self, 'set_port%d_enabled' % c)(False)

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

    self.get_ramp_stepsize()
    self.get_ramp_delaytime()
    self.get_retries_on_set_error()

    for c in range(1,9):
      if getattr(self, 'get_port%d_enabled' % c)():
        getattr(self, 'get_port%d_idn' % c)()
        getattr(self, 'get_port%d_battery_status' % c)()
        getattr(self, 'get_port%d_on' % c)()
        getattr(self, 'get_port%d_voltage' % c)()

  def _ask(self, cmd):
    max_attempts = 3
    for attempt in range(max_attempts):
      try:
        t = time.time()
        t_to_sleep = self._min_time_between_commands - (t - self._last_communication_time)
        if t_to_sleep > 0: time.sleep(t_to_sleep)
        
        r = self._visainstrument.ask(cmd)
        self._last_communication_time = time.time()
    
        return r
      except Exception as e:
        logging.warn('Failed to get a reply from SIM: %s' % str(e))
        if attempt == max_attempts-1: raise
        self._last_communication_time = time.time()
        time.sleep( .2 )
        self._clear_mainframe_output_buffer()
        time.sleep( .2+attempt )

  def _write(self, cmd):
    max_attempts = 3
    for attempt in range(max_attempts):
      try:
        t = time.time()
        t_to_sleep = self._min_time_between_commands - (t - self._last_communication_time)
        if t_to_sleep > 0: time.sleep(t_to_sleep)
        
        self._visainstrument.write(cmd)
        self._last_communication_time = time.time()
        return
      except Exception as e:
        logging.warn('Failed to write to SIM: %s' % str(e))
        if attempt == max_attempts-1: raise
        self._last_communication_time = time.time()
        time.sleep( .2 )
        self._clear_mainframe_output_buffer()
        time.sleep( .2+attempt )
      
  def _set_baud_rate(self, port=None):
    ports = [port] if port != None else range(1,9)
    #self._clear_mainframe_output_buffer()
    for p in ports:
      self._wait_until_input_read(p)
      self._clear_output_buffer(p)
      baud_rate = 9600 #38400
      self._write("SNDT %d,'BAUD %d'" % (p, baud_rate))
      self._write("BAUD %d,%d" % (p, baud_rate))

  def _clear_mainframe_output_buffer(self):
    # check if something is ~immediately available
    old_timeout = self._visainstrument.timeout
    self._visainstrument.timeout = np.max(( .5, 3*self._min_time_between_commands ))
    
    try:
      for attempt in range(3):
        t = time.time()
        t_to_sleep = self._min_time_between_commands - (t - self._last_communication_time)
        if t_to_sleep > 0: time.sleep(t_to_sleep)
      
        r = self._visainstrument.read()
        self._last_communication_time = time.time()
        if len(r) == 0: break
    except Exception as e:
      self._last_communication_time = time.time()
      pass # Nothing is buffered anymore
    
    # restore the old timeout
    self._visainstrument.timeout = old_timeout
    
  def _clear_output_buffer(self, port):
    for j in range(8):
      resp = self._ask('GETN? %s,80' % port)
      if resp.strip() == "#3000":   # we expect "#3000" if there is nothing in the buffer
        return
        
      if j>0: logging.warn('Still getting %d bytes of output after %d GETN? %d,80 queries.' % (len(resp)-5, 1+j, port))
      time.sleep(0.2 * (1+j))
        # bytes_waiting = self._ask('NOUT? %s' % port)

        # if int(bytes_waiting) == 0:
            # return
        # else:
            # self._ask('GETN? %s,80' % port)
            # self._write('FLSO %s' % port)
            # if j%10 == 0: logging.debug(__name__ + ' : output bytes waiting for port %s: %s' % (port, bytes_waiting))
    
    raise Exception('Could not clear output buffer. Still getting: %s' % resp)

  def _wait_until_input_read(self, port):
    for j in range(4):
        bytes_waiting = self._ask('NINP? %s' % port)
        if int(bytes_waiting) == 0:
            return
        else:
            if j > 0: logging.warn(__name__ + ' : input bytes waiting for port %s: %s' % (port, bytes_waiting))
            time.sleep(0.2 * (1+j))
    
    #assert(False)

  def set_voltages(self, port_to_voltage, update_attribute_value=True):
    '''
      port_to_voltage --- should be a dictionary { port no: voltage }, e.g. {1: -0.5, 5: 0.0}
    '''
    for attempt in range(np.max(( 1, self._retries_on_set_error ))):
      stepsize = self.get_ramp_stepsize()
      delay = self.get_ramp_delaytime()
      
      target_voltage = {}
      old = {}
      steps = {}
      done = {}
      for port in port_to_voltage.keys():
        if not self._ch_enabled[port-1]:
          logging.warn('Port %d is disabled.' % port)
          return

        target_voltage[port] = np.round(port_to_voltage[port], decimals=3)
        if np.abs(target_voltage[port]) < 1e-4: target_voltage[port] = 0.   # the SIM doesn't like the minus in front of zero.
        logging.debug(__name__ + ' : setting port %s voltage to %s V' % (port, target_voltage[port]))
        
        old[port] = self._get_voltage(port)
        steps[port] = np.linspace(old[port], target_voltage[port], 2 + int(np.abs(target_voltage[port]-old[port])/stepsize))
        done[port] = False
      
      for i in range(max( len(s) for s in steps.values() )):
        for port in port_to_voltage.keys():
          if done[port]: continue # This port is already done ramping

          if np.isnan(old[port]): # previous value invalid?
            old[port] = target_voltage[port]
            done[port] = True
      
          if i < len(steps[port]):
            self._write('SNDT %s,"VOLT %s"' % (port, steps[port][i]))
            if i == len(steps[port]) - 1:
              done[port] = True
              if update_attribute_value:
                self.update_value('port%d_voltage' % port, float(steps[port][i]))

        time.sleep(delay)  # wait time between steps
          
      if self._retries_on_set_error > 0: # verify that the correct voltage was set
        success = {}
        for port in port_to_voltage.keys():
          new_voltage = getattr(self, 'get_port%d_voltage' % port)()
          success[port] = ( np.abs(new_voltage - target_voltage[port]) < 0.0005 )
          if not success[port]:
            logging.warn('Attempt #%d to set voltage to %g for port %d failed.' % (attempt, target_voltage[port], port))
        if all(success.values()):
          return
        else:
          time.sleep(.05*(1+attempt)) # go back to beginning and try again
      else:
        return # don't check for success

    logging.warn('The desired output voltage %g for port %d was not set!' % (target_voltage[port], port))

  def _set_voltage(self, port, voltage):
    self.set_voltages({port: voltage}, update_attribute_value=False)
  
  def _set_PID_propotional_on(self,port,onoff):
    if not self._ch_enabled[port-1]:
      logging.warn('Port %d is disabled.' % port)
      return
    for attempt in range(np.max(( 1, self._retries_on_set_error ))):
      try:
        self._clear_output_buffer(port)
        self._write('SNDT %s,"PCTL %s"' % (port,onoff))
        logging.debug(__name__ + ' : setting port %s PID propotional on %s' % (port, onoff))
      except Exception as e:
        logging.warn('Attempt #%d to turn port %d propotional on SIM failed: %s' % (1+attempt, port, str(e)))
        time.sleep( .5 )
        self._clear_mainframe_output_buffer()
        self._wait_until_input_read(port)
        self._clear_output_buffer(port)
        time.sleep( .5+attempt )
    return
  
  def _get_PID_propotional_on(self,port):
    if not self._ch_enabled[port-1]:
      logging.warn('Port %d is disabled.' % port)
      return np.nan
    for attempt in range(np.max(( 1, self._retries_on_set_error ))):
      try:
        self._clear_output_buffer(port)
        self._write('SNDT %s,"PCTL?"' % port)
        r = self._ask('GETN? %s,80' % port)
        logging.debug(__name__ + ' : getting port %s PID propotional on %s' % (port, r))
        if (r[:2]!="#3"): 
          raise Exception('Response %s is not in the expected format' % r)
        
        nbytes = int(r[2:5])

        if (nbytes < 1):
          time.sleep((1+attempt)*0.1)
          continue
        else:
          bytes = r[5:5+nbytes].replace("\n","").replace("\r","")
          logging.debug(__name__ + ' : parsed output on response: %s' % bytes)
          if bytes == '0':
            return False
          else:
            return True
      except Exception as e:
        logging.warn('Attempt #%d to turn port %d propotional on SIM failed: %s' % (1+attempt, port, str(e)))
        time.sleep( .5 )
        self._clear_mainframe_output_buffer()
        self._wait_until_input_read(port)
        self._clear_output_buffer(port)
        time.sleep( .5+attempt )
    return np.nan
  
  #######
  # propotional gains
  #####
  
  def _set_PID_propotional_gain(self,port,val):
    if not self._ch_enabled[port-1]:
      logging.warn('Port %d is disabled.' % port)
      return
    for attempt in range(np.max(( 1, self._retries_on_set_error ))):
      try:
        self._clear_output_buffer(port)
        self._write('SNDT %s,"GAIN %s"' % (port,val))
        logging.debug(__name__ + ' : setting port %s PID propotional gain %s' % (port, val))
      except Exception as e:
        logging.warn('Attempt #%d to set port %d propotional gain SIM failed: %s' % (1+attempt, port, str(e)))
        time.sleep( .5 )
        self._clear_mainframe_output_buffer()
        self._wait_until_input_read(port)
        self._clear_output_buffer(port)
        time.sleep( .5+attempt )
    return
  
  def _get_PID_propotional_gain(self,port):
    if not self._ch_enabled[port-1]:
      logging.warn('Port %d is disabled.' % port)
      return np.nan
    for attempt in range(np.max(( 1, self._retries_on_set_error ))):
      try:
        self._clear_output_buffer(port)
        self._write('SNDT %s,"GAIN?"' % port)
        r = self._ask('GETN? %s,80' % port)
        logging.debug(__name__ + ' : getting port %s PID propotional on %s' % (port, r))
        if (r[:2]!="#3"): 
          raise Exception('Response %s is not in the expected format' % r)
        
        nbytes = int(r[2:5])

        if (nbytes < 1):
          time.sleep((1+attempt)*0.1)
          continue
        else:
          bytes = r[5:5+nbytes].replace("\n","").replace("\r","")
          logging.debug(__name__ + ' : parsed output on response: %s' % bytes)
          return float(bytes)
      except Exception as e:
        logging.warn('Attempt #%d to turn port %d propotional on SIM failed: %s' % (1+attempt, port, str(e)))
        time.sleep( .5 )
        self._clear_mainframe_output_buffer()
        self._wait_until_input_read(port)
        self._clear_output_buffer(port)
        time.sleep( .5+attempt )
    return np.nan
  
  def _set_PID_integral_on(self,port,onoff):
    if not self._ch_enabled[port-1]:
      logging.warn('Port %d is disabled.' % port)
      return
    for attempt in range(np.max(( 1, self._retries_on_set_error ))):
      try:
        self._clear_output_buffer(port)
        self._write('SNDT %s,"ICTL %s"' % (port,onoff))
        logging.debug(__name__ + ' : setting port %s PID integral on %s' % (port, onoff))
      except Exception as e:
        logging.warn('Attempt #%d to turn port %d integral on SIM failed: %s' % (1+attempt, port, str(e)))
        time.sleep( .5 )
        self._clear_mainframe_output_buffer()
        self._wait_until_input_read(port)
        self._clear_output_buffer(port)
        time.sleep( .5+attempt )
    return
  
  def _get_PID_integral_on(self,port):
    if not self._ch_enabled[port-1]:
      logging.warn('Port %d is disabled.' % port)
      return np.nan
    for attempt in range(np.max(( 1, self._retries_on_set_error ))):
      try:
        self._clear_output_buffer(port)
        self._write('SNDT %s,"ICTL?"' % port)
        r = self._ask('GETN? %s,80' % port)
        logging.debug(__name__ + ' : getting port %s PID integral on %s' % (port, r))
        if (r[:2]!="#3"): 
          raise Exception('Response %s is not in the expected format' % r)
        
        nbytes = int(r[2:5])

        if (nbytes < 1):
          time.sleep((1+attempt)*0.1)
          continue
        else:
          bytes = r[5:5+nbytes].replace("\n","").replace("\r","")
          logging.debug(__name__ + ' : parsed output on response: %s' % bytes)
          if bytes == '0':
            return False
          else:
            return True
      except Exception as e:
        logging.warn('Attempt #%d to turn port %d integral on SIM failed: %s' % (1+attempt, port, str(e)))
        time.sleep( .5 )
        self._clear_mainframe_output_buffer()
        self._wait_until_input_read(port)
        self._clear_output_buffer(port)
        time.sleep( .5+attempt )
    return np.nan
  
  #######
  # integral gains
  #####
  
  def _set_PID_integral_gain(self,port,val):
    if not self._ch_enabled[port-1]:
      logging.warn('Port %d is disabled.' % port)
      return
    for attempt in range(np.max(( 1, self._retries_on_set_error ))):
      try:
        self._clear_output_buffer(port)
        self._write('SNDT %s,"INTG %s"' % (port,val))
        logging.debug(__name__ + ' : setting port %s PID integral gain %s' % (port, val))
      except Exception as e:
        logging.warn('Attempt #%d to set port %d integral gain SIM failed: %s' % (1+attempt, port, str(e)))
        time.sleep( .5 )
        self._clear_mainframe_output_buffer()
        self._wait_until_input_read(port)
        self._clear_output_buffer(port)
        time.sleep( .5+attempt )
    return
  
  def _get_PID_integral_gain(self,port):
    if not self._ch_enabled[port-1]:
      logging.warn('Port %d is disabled.' % port)
      return np.nan
    for attempt in range(np.max(( 1, self._retries_on_set_error ))):
      try:
        self._clear_output_buffer(port)
        self._write('SNDT %s,"INTG?"' % port)
        r = self._ask('GETN? %s,80' % port)
        logging.debug(__name__ + ' : getting port %s PID integral on %s' % (port, r))
        if (r[:2]!="#3"): 
          raise Exception('Response %s is not in the expected format' % r)
        
        nbytes = int(r[2:5])

        if (nbytes < 1):
          time.sleep((1+attempt)*0.1)
          continue
        else:
          bytes = r[5:5+nbytes].replace("\n","").replace("\r","")
          logging.debug(__name__ + ' : parsed output on response: %s' % bytes)
          return float(bytes)
      except Exception as e:
        logging.warn('Attempt #%d to turn port %d integral on SIM failed: %s' % (1+attempt, port, str(e)))
        time.sleep( .5 )
        self._clear_mainframe_output_buffer()
        self._wait_until_input_read(port)
        self._clear_output_buffer(port)
        time.sleep( .5+attempt )
    return np.nan
    
    
  def _set_PID_derivative_on(self,port,onoff):
    if not self._ch_enabled[port-1]:
      logging.warn('Port %d is disabled.' % port)
      return
    for attempt in range(np.max(( 1, self._retries_on_set_error ))):
      try:
        self._clear_output_buffer(port)
        self._write('SNDT %s,"DCTL %s"' % (port,onoff))
        logging.debug(__name__ + ' : setting port %s PID derivative on %s' % (port, onoff))
      except Exception as e:
        logging.warn('Attempt #%d to turn port %d derivative on SIM failed: %s' % (1+attempt, port, str(e)))
        time.sleep( .5 )
        self._clear_mainframe_output_buffer()
        self._wait_until_input_read(port)
        self._clear_output_buffer(port)
        time.sleep( .5+attempt )
    return
  
  def _get_PID_derivative_on(self,port):
    if not self._ch_enabled[port-1]:
      logging.warn('Port %d is disabled.' % port)
      return np.nan
    for attempt in range(np.max(( 1, self._retries_on_set_error ))):
      try:
        self._clear_output_buffer(port)
        self._write('SNDT %s,"DCTL?"' % port)
        r = self._ask('GETN? %s,80' % port)
        logging.debug(__name__ + ' : getting port %s PID derivative on %s' % (port, r))
        if (r[:2]!="#3"): 
          raise Exception('Response %s is not in the expected format' % r)
        
        nbytes = int(r[2:5])

        if (nbytes < 1):
          time.sleep((1+attempt)*0.1)
          continue
        else:
          bytes = r[5:5+nbytes].replace("\n","").replace("\r","")
          logging.debug(__name__ + ' : parsed output on response: %s' % bytes)
          return bytes.strip().lower() in ['1', 'on']
      except Exception as e:
        logging.warn('Attempt #%d to turn port %d derivative on SIM failed: %s' % (1+attempt, port, str(e)))
        time.sleep( .5 )
        self._clear_mainframe_output_buffer()
        self._wait_until_input_read(port)
        self._clear_output_buffer(port)
        time.sleep( .5+attempt )
    return np.nan
  
  #######
  # derivative gains
  #####
  
  def _set_PID_derivative_gain(self,port,val):
    if not self._ch_enabled[port-1]:
      logging.warn('Port %d is disabled.' % port)
      return
    for attempt in range(np.max(( 1, self._retries_on_set_error ))):
      try:
        self._clear_output_buffer(port)
        self._write('SNDT %s,"DERV %s"' % (port,val))
        logging.debug(__name__ + ' : setting port %s PID derivative gain %s' % (port, val))
      except Exception as e:
        logging.warn('Attempt #%d to set port %d derivative gain SIM failed: %s' % (1+attempt, port, str(e)))
        time.sleep( .5 )
        self._clear_mainframe_output_buffer()
        self._wait_until_input_read(port)
        self._clear_output_buffer(port)
        time.sleep( .5+attempt )
    return
  
  def _get_PID_derivative_gain(self,port):
    if not self._ch_enabled[port-1]:
      logging.warn('Port %d is disabled.' % port)
      return np.nan
    for attempt in range(np.max(( 1, self._retries_on_set_error ))):
      try:
        self._clear_output_buffer(port)
        self._write('SNDT %s,"DERV?"' % port)
        r = self._ask('GETN? %s,80' % port)
        logging.debug(__name__ + ' : getting port %s PID derivative on %s' % (port, r))
        if (r[:2]!="#3"): 
          raise Exception('Response %s is not in the expected format' % r)
        
        nbytes = int(r[2:5])

        if (nbytes < 1):
          time.sleep((1+attempt)*0.1)
          continue
        else:
          bytes = r[5:5+nbytes].replace("\n","").replace("\r","")
          logging.debug(__name__ + ' : parsed output on response: %s' % bytes)
          return float(bytes)
      except Exception as e:
        logging.warn('Attempt #%d to turn port %d derivative on SIM failed: %s' % (1+attempt, port, str(e)))
        time.sleep( .5 )
        self._clear_mainframe_output_buffer()
        self._wait_until_input_read(port)
        self._clear_output_buffer(port)
        time.sleep( .5+attempt )
    return np.nan
  
  def _set_PID_offset_on(self,port,onoff):
    if not self._ch_enabled[port-1]:
      logging.warn('Port %d is disabled.' % port)
      return
    for attempt in range(np.max(( 1, self._retries_on_set_error ))):
      try:
        self._clear_output_buffer(port)
        self._write('SNDT %s,"OCTL %s"' % (port,onoff))
        logging.debug(__name__ + ' : setting port %s PID offset on %s' % (port, onoff))
      except Exception as e:
        logging.warn('Attempt #%d to turn port %d offset on SIM failed: %s' % (1+attempt, port, str(e)))
        time.sleep( .5 )
        self._clear_mainframe_output_buffer()
        self._wait_until_input_read(port)
        self._clear_output_buffer(port)
        time.sleep( .5+attempt )
    return
  
  def _get_PID_offset_on(self,port):
    if not self._ch_enabled[port-1]:
      logging.warn('Port %d is disabled.' % port)
      return np.nan
    for attempt in range(np.max(( 1, self._retries_on_set_error ))):
      try:
        self._clear_output_buffer(port)
        self._write('SNDT %s,"OCTL?"' % port)
        r = self._ask('GETN? %s,80' % port)
        logging.debug(__name__ + ' : getting port %s PID offset on %s' % (port, r))
        if (r[:2]!="#3"): 
          raise Exception('Response %s is not in the expected format' % r)
        
        nbytes = int(r[2:5])

        if (nbytes < 1):
          time.sleep((1+attempt)*0.1)
          continue
        else:
          bytes = r[5:5+nbytes].replace("\n","").replace("\r","")
          logging.debug(__name__ + ' : parsed output on response: %s' % bytes)
          if bytes == '0':
            return False
          else:
            return True
      except Exception as e:
        logging.warn('Attempt #%d to turn port %d offset on SIM failed: %s' % (1+attempt, port, str(e)))
        time.sleep( .5 )
        self._clear_mainframe_output_buffer()
        self._wait_until_input_read(port)
        self._clear_output_buffer(port)
        time.sleep( .5+attempt )
    return np.nan
  
  #######
  # offset gains
  #####
  
  def _set_PID_offset_gain(self,port,val):
    if not self._ch_enabled[port-1]:
      logging.warn('Port %d is disabled.' % port)
      return
    for attempt in range(np.max(( 1, self._retries_on_set_error ))):
      try:
        self._clear_output_buffer(port)
        self._write('SNDT %s,"OFST %s"' % (port,val))
        logging.debug(__name__ + ' : setting port %s PID offset gain %s' % (port, val))
      except Exception as e:
        logging.warn('Attempt #%d to set port %d offset gain SIM failed: %s' % (1+attempt, port, str(e)))
        time.sleep( .5 )
        self._clear_mainframe_output_buffer()
        self._wait_until_input_read(port)
        self._clear_output_buffer(port)
        time.sleep( .5+attempt )
    return
  
  def _get_PID_offset_gain(self,port):
    if not self._ch_enabled[port-1]:
      logging.warn('Port %d is disabled.' % port)
      return np.nan
    for attempt in range(np.max(( 1, self._retries_on_set_error ))):
      try:
        self._clear_output_buffer(port)
        self._write('SNDT %s,"OFST?"' % port)
        r = self._ask('GETN? %s,80' % port)
        logging.debug(__name__ + ' : getting port %s PID offset on %s' % (port, r))
        if (r[:2]!="#3"): 
          raise Exception('Response %s is not in the expected format' % r)
        
        nbytes = int(r[2:5])

        if (nbytes < 1):
          time.sleep((1+attempt)*0.1)
          continue
        else:
          bytes = r[5:5+nbytes].replace("\n","").replace("\r","")
          logging.debug(__name__ + ' : parsed output on response: %s' % bytes)
          return float(bytes)
      except Exception as e:
        logging.warn('Attempt #%d to turn port %d offset on SIM failed: %s' % (1+attempt, port, str(e)))
        time.sleep( .5 )
        self._clear_mainframe_output_buffer()
        self._wait_until_input_read(port)
        self._clear_output_buffer(port)
        time.sleep( .5+attempt )
    return np.nan
				
	
  
  def _set_PID_manual_output_on(self,port,onoff):
    if not self._ch_enabled[port-1]:
      logging.warn('Port %d is disabled.' % port)
      return
    for attempt in range(np.max(( 1, self._retries_on_set_error ))):
      try:
        self._clear_output_buffer(port)
        self._write('SNDT %s,"AMAN %s"' % (port,onoff))
        logging.debug(__name__ + ' : setting port %s PID manual_output on %s' % (port, onoff))
      except Exception as e:
        logging.warn('Attempt #%d to turn port %d manual_output on SIM failed: %s' % (1+attempt, port, str(e)))
        time.sleep( .5 )
        self._clear_mainframe_output_buffer()
        self._wait_until_input_read(port)
        self._clear_output_buffer(port)
        time.sleep( .5+attempt )
    return
  
  def _get_PID_manual_output_on(self,port):
    if not self._ch_enabled[port-1]:
      logging.warn('Port %d is disabled.' % port)
      return np.nan
    for attempt in range(np.max(( 1, self._retries_on_set_error ))):
      try:
        self._clear_output_buffer(port)
        self._write('SNDT %s,"AMAN?"' % port)
        r = self._ask('GETN? %s,80' % port)
        logging.debug(__name__ + ' : getting port %s PID manual_output on %s' % (port, r))
        if (r[:2]!="#3"): 
          raise Exception('Response %s is not in the expected format' % r)
        
        nbytes = int(r[2:5])

        if (nbytes < 1):
          time.sleep((1+attempt)*0.1)
          continue
        else:
          bytes = r[5:5+nbytes].replace("\n","").replace("\r","")
          logging.debug(__name__ + ' : parsed output on response: %s' % bytes)
          if bytes == '0':
            return False
          else:
            return True
      except Exception as e:
        logging.warn('Attempt #%d to turn port %d manual_output on SIM failed: %s' % (1+attempt, port, str(e)))
        time.sleep( .5 )
        self._clear_mainframe_output_buffer()
        self._wait_until_input_read(port)
        self._clear_output_buffer(port)
        time.sleep( .5+attempt )
    return np.nan
  
  #######
  # manual_outputs
  #####
  
  def _set_PID_manual_output(self,port,val):
    if not self._ch_enabled[port-1]:
      logging.warn('Port %d is disabled.' % port)
      return
    for attempt in range(np.max(( 1, self._retries_on_set_error ))):
      try:
        self._clear_output_buffer(port)
        self._write('SNDT %s,"MOUT %s"' % (port,val))
        logging.debug(__name__ + ' : setting port %s PID manual_output  %s' % (port, val))
      except Exception as e:
        logging.warn('Attempt #%d to set port %d manual_output  SIM failed: %s' % (1+attempt, port, str(e)))
        time.sleep( .5 )
        self._clear_mainframe_output_buffer()
        self._wait_until_input_read(port)
        self._clear_output_buffer(port)
        time.sleep( .5+attempt )
    return
  
  def _get_PID_manual_output(self,port):
    if not self._ch_enabled[port-1]:
      logging.warn('Port %d is disabled.' % port)
      return np.nan
    for attempt in range(np.max(( 1, self._retries_on_set_error ))):
      try:
        self._clear_output_buffer(port)
        self._write('SNDT %s,"MOUT?"' % port)
        r = self._ask('GETN? %s,80' % port)
        logging.debug(__name__ + ' : getting port %s PID manual_output on %s' % (port, r))
        if (r[:2]!="#3"): 
          raise Exception('Response %s is not in the expected format' % r)
        
        nbytes = int(r[2:5])

        if (nbytes < 1):
          time.sleep((1+attempt)*0.1)
          continue
        else:
          bytes = r[5:5+nbytes].replace("\n","").replace("\r","")
          logging.debug(__name__ + ' : parsed output on response: %s' % bytes)
          return float(bytes)
      except Exception as e:
        logging.warn('Attempt #%d to turn port %d manual_output on SIM failed: %s' % (1+attempt, port, str(e)))
        time.sleep( .5 )
        self._clear_mainframe_output_buffer()
        self._wait_until_input_read(port)
        self._clear_output_buffer(port)
        time.sleep( .5+attempt )
    return np.nan
	
  #######
  # upper_limit
  #####
  
  def _set_PID_upper_limit(self,port,val):
    if not self._ch_enabled[port-1]:
      logging.warn('Port %d is disabled.' % port)
      return
    llimit = self._get_PID_lower_limit(port)
    if val < llimit:
      logging.warn('Lower limit larger than upper limit.')
      return
    for attempt in range(np.max(( 1, self._retries_on_set_error ))):
      try:
        self._clear_output_buffer(port)
        self._write('SNDT %s,"ULIM %s"' % (port,val))
        logging.debug(__name__ + ' : setting port %s PID manual_output  %s' % (port, val))
      except Exception as e:
        logging.warn('Attempt #%d to set port %d manual_output  SIM failed: %s' % (1+attempt, port, str(e)))
        time.sleep( .5 )
        self._clear_mainframe_output_buffer()
        self._wait_until_input_read(port)
        self._clear_output_buffer(port)
        time.sleep( .5+attempt )
    return
  
  def _get_PID_upper_limit(self,port):
    if not self._ch_enabled[port-1]:
      logging.warn('Port %d is disabled.' % port)
      return np.nan
    for attempt in range(np.max(( 1, self._retries_on_set_error ))):
      try:
        self._clear_output_buffer(port)
        self._write('SNDT %s,"ULIM?"' % port)
        r = self._ask('GETN? %s,80' % port)
        logging.debug(__name__ + ' : getting port %s PID upper_limit %s' % (port, r))
        if (r[:2]!="#3"): 
          raise Exception('Response %s is not in the expected format' % r)
        
        nbytes = int(r[2:5])

        if (nbytes < 1):
          time.sleep((1+attempt)*0.1)
          continue
        else:
          bytes = r[5:5+nbytes].replace("\n","").replace("\r","")
          logging.debug(__name__ + ' : parsed output on response: %s' % bytes)
          return float(bytes)
      except Exception as e:
        logging.warn('Attempt #%d to turn port %d upper limit on SIM failed: %s' % (1+attempt, port, str(e)))
        time.sleep( .5 )
        self._clear_mainframe_output_buffer()
        self._wait_until_input_read(port)
        self._clear_output_buffer(port)
        time.sleep( .5+attempt )
    return np.nan
	
  
  #######
  # lower limits
  #####
  
  def _set_PID_lower_limit(self,port,val):
    if not self._ch_enabled[port-1]:
      logging.warn('Port %d is disabled.' % port)
      return
    ulimit = self._get_PID_upper_limit(port)
    if ulimit < val:
      logging.warn('Lower limit larger than upper limit.')
      return
    for attempt in range(np.max(( 1, self._retries_on_set_error ))):
      try:
        self._clear_output_buffer(port)
        self._write('SNDT %s,"LLIM %s"' % (port,val))
        logging.debug(__name__ + ' : setting port %s PID lower limit  %s' % (port, val))
      except Exception as e:
        logging.warn('Attempt #%d to set port %d lower limit  SIM failed: %s' % (1+attempt, port, str(e)))
        time.sleep( .5 )
        self._clear_mainframe_output_buffer()
        self._wait_until_input_read(port)
        self._clear_output_buffer(port)
        time.sleep( .5+attempt )
    return
  
  def _get_PID_lower_limit(self,port):
    if not self._ch_enabled[port-1]:
      logging.warn('Port %d is disabled.' % port)
      return np.nan
    for attempt in range(np.max(( 1, self._retries_on_set_error ))):
      try:
        self._clear_output_buffer(port)
        self._write('SNDT %s,"LLIM?"' % port)
        r = self._ask('GETN? %s,80' % port)
        logging.debug(__name__ + ' : getting port %s PID manual_output on %s' % (port, r))
        if (r[:2]!="#3"): 
          raise Exception('Response %s is not in the expected format' % r)
        
        nbytes = int(r[2:5])

        if (nbytes < 1):
          time.sleep((1+attempt)*0.1)
          continue
        else:
          bytes = r[5:5+nbytes].replace("\n","").replace("\r","")
          logging.debug(__name__ + ' : parsed output on response: %s' % bytes)
          return float(bytes)
      except Exception as e:
        logging.warn('Attempt #%d to turn port %d manual_output on SIM failed: %s' % (1+attempt, port, str(e)))
        time.sleep( .5 )
        self._clear_mainframe_output_buffer()
        self._wait_until_input_read(port)
        self._clear_output_buffer(port)
        time.sleep( .5+attempt )
    return np.nan
	
  
  
  def _get_voltage(self, port):
    if not self._ch_enabled[port-1]:
      logging.warn('Port %d is disabled.' % port)
      return np.nan
    
    for attempt in range(3):
      try:
        self._clear_output_buffer(port)
        self._write('SNDT %s,"VOLT?"' % port)
        r = self._ask('GETN? %s,80' % port)
        logging.debug(__name__ + ' : getting port %s voltage: %s' % (port, r))
      
        if (r[:2]!="#3"): raise Exception('Response %s is not in the expected format' % r)
        
        nbytes = int(r[2:5])

        if (nbytes < 1):
          time.sleep((1+attempt)*0.1)
          continue
        else:
          bytes = r[5:5+nbytes].replace("\n","").replace("\r","")
          logging.debug(__name__ + ' : parsed voltage response: %s' % bytes)
          return float(bytes)
      except Exception as e:
        logging.warn('Attempt #%d to get port %d voltage from SIM failed: %s' % (1+attempt, port, str(e)))
        time.sleep( .5 )
        self._clear_mainframe_output_buffer()
        self._wait_until_input_read(port)
        self._clear_output_buffer(port)
        time.sleep( .5+attempt )
    
    msg = "WARN: Could not get voltage for port %u." % port
    logging.warn(msg)
    return np.nan

  def _set_on(self, port, val):
    if not self._ch_enabled[port-1]:
      logging.warn('Port %d is disabled.' % port)
      return

    logging.debug(__name__ + ' : setting port %s output to %s' % (port, val))
    
    for attempt in range(np.max(( 1, self._retries_on_set_error ))):
      self._write('SNDT %s,"EXON %s"' % (port, int(val)))
      
      if self._retries_on_set_error > 0: # verify that the correct state was set
        new_val = getattr(self, 'get_port%d_on' % port)()
        if bool(new_val) == bool(val):
          return # success
        logging.warn('Attempt #%d to set output = %s for port %d failed.' % (attempt, str(val), port))
        time.sleep(.5*(1+attempt))

    logging.warn('The desired output state %s for port %d was not set!' % (str(val), port))

  def _get_on(self, port):
    if not self._ch_enabled[port-1]:
      logging.warn('Port %d is disabled.' % port)
      return False

    for attempt in range(3):
      try:
        self._clear_output_buffer(port)
        self._write('SNDT %s,"EXON?"' % port)
        r = self._ask('GETN? %s,80' % port)
        logging.debug(__name__ + ' : getting port %s output state: %s' % (port, r))
      
        if (r[:2]!="#3"): raise Exception('Response %s is not in the expected format' % r)
      
        nbytes = int(r[2:5])

        if (nbytes < 1):
          time.sleep((1+attempt)*0.1)
          continue
        else:
          bytes = r[5:5+nbytes].replace("\n","").replace("\r","")
          logging.debug(__name__ + ' : parsed output on response: %s' % bytes)
          return bool(bytes)
      except Exception as e:
        logging.warn('Attempt #%d to get port %d "on" status from SIM failed: %s' % (1+attempt, port, str(e)))
        time.sleep( .5 )
        self._clear_mainframe_output_buffer()
        self._wait_until_input_read(port)
        self._clear_output_buffer(port)
        time.sleep( .5+attempt )

    msg = "Could not determine whether port %u is on." % port
    logging.warn(msg)
    return False

  def set_port_voltage(self, port, voltage):
    if not isinstance(port, int):
      raise Exception('port must be specified as an integer, not %s.' % str(port))
    if port < 1 or port > 8:
      raise Exception('port must be between 1 and 8, not %s.' % str(port))
    getattr(self, 'set_port%s_voltage' % str(port))(voltage)

  def get_port_voltage(self, port):
    if not isinstance(port, int):
      raise Exception('port must be specified as an integer, not %s.' % str(port))
    if port < 1 or port > 8:
      raise Exception('port must be between 1 and 8, not %s.' % str(port))
    return getattr(self, 'get_port%s_voltage' % str(port))()

  def set_port_on(self, port, val):
    if not isinstance(port, int):
      raise Exception('port must be specified as an integer, not %s.' % str(port))
    if port < 1 or port > 8:
      raise Exception('port must be between 1 and 8, not %s.' % str(port))
    getattr(self, 'set_port%s_on' % str(port))(val)

  def get_port_on(self, port):
    if not isinstance(port, int):
      raise Exception('port must be specified as an integer, not %s.' % str(port))
    if port < 1 or port > 8:
      raise Exception('port must be between 1 and 8, not %s.' % str(port))
    return getattr(self, 'get_port%s_on' % str(port))()
  
  def set_port_PID_propotional_on(self, port, onoff):
    if not isinstance(port, int):
      raise Exception('port must be specified as an integer, not %s.' % str(port))
    if port < 1 or port > 8:
      raise Exception('port must be between 1 and 8, not %s.' % str(port))
    getattr(self, 'set_port%s_PID_propotional_on' % str(port))(onoff)
  
  def get_port_PID_propotional_on(self, port):
    if not isinstance(port, int):
      raise Exception('port must be specified as an integer, not %s.' % str(port))
    if port < 1 or port > 8:
      raise Exception('port must be between 1 and 8, not %s.' % str(port))
    getattr(self, 'get_port%s_PID_propotional_on' % str(port))()
  
  def set_port_PID_propotional_gain(self, port, val):
    if not isinstance(port, int):
      raise Exception('port must be specified as an integer, not %s.' % str(port))
    if port < 1 or port > 8:
      raise Exception('port must be between 1 and 8, not %s.' % str(port))
    getattr(self, 'set_port%s_PID_propotional_gain' % str(port))(val)
  
  def get_port_PID_propotional_gain(self, port):
    if not isinstance(port, int):
      raise Exception('port must be specified as an integer, not %s.' % str(port))
    if port < 1 or port > 8:
      raise Exception('port must be between 1 and 8, not %s.' % str(port))
    getattr(self, 'get_port%s_PID_propotional_gain' % str(port))()
  
  def set_port_PID_integral_on(self, port, onoff):
    if not isinstance(port, int):
      raise Exception('port must be specified as an integer, not %s.' % str(port))
    if port < 1 or port > 8:
      raise Exception('port must be between 1 and 8, not %s.' % str(port))
    getattr(self, 'set_port%s_PID_integral_on' % str(port))(onoff)
  
  def get_port_PID_integral_on(self, port):
    if not isinstance(port, int):
      raise Exception('port must be specified as an integer, not %s.' % str(port))
    if port < 1 or port > 8:
      raise Exception('port must be between 1 and 8, not %s.' % str(port))
    getattr(self, 'get_port%s_PID_integral_on' % str(port))()
  
  def set_port_PID_integral_gain(self, port, val):
    if not isinstance(port, int):
      raise Exception('port must be specified as an integer, not %s.' % str(port))
    if port < 1 or port > 8:
      raise Exception('port must be between 1 and 8, not %s.' % str(port))
    getattr(self, 'set_port%s_PID_integral_gain' % str(port))(val)
  
  def get_port_PID_integral_gain(self, port):
    if not isinstance(port, int):
      raise Exception('port must be specified as an integer, not %s.' % str(port))
    if port < 1 or port > 8:
      raise Exception('port must be between 1 and 8, not %s.' % str(port))
    getattr(self, 'get_port%s_PID_integral_gain' % str(port))()
    
  def set_port_PID_derivative_on(self, port, onoff):
    if not isinstance(port, int):
      raise Exception('port must be specified as an integer, not %s.' % str(port))
    if port < 1 or port > 8:
      raise Exception('port must be between 1 and 8, not %s.' % str(port))
    getattr(self, 'set_port%s_PID_derivative_on' % str(port))(onoff)
  
  def get_port_PID_derivative_on(self, port):
    if not isinstance(port, int):
      raise Exception('port must be specified as an integer, not %s.' % str(port))
    if port < 1 or port > 8:
      raise Exception('port must be between 1 and 8, not %s.' % str(port))
    getattr(self, 'get_port%s_PID_derivative_on' % str(port))()
  
  def set_port_PID_derivative_gain(self, port, val):
    if not isinstance(port, int):
      raise Exception('port must be specified as an integer, not %s.' % str(port))
    if port < 1 or port > 8:
      raise Exception('port must be between 1 and 8, not %s.' % str(port))
    getattr(self, 'set_port%s_PID_derivative_gain' % str(port))(val)
  
  def get_port_PID_derivative_gain(self, port):
    if not isinstance(port, int):
      raise Exception('port must be specified as an integer, not %s.' % str(port))
    if port < 1 or port > 8:
      raise Exception('port must be between 1 and 8, not %s.' % str(port))
    getattr(self, 'get_port%s_PID_derivative_gain' % str(port))()
  
  def set_port_PID_offset_on(self, port, onoff):
    if not isinstance(port, int):
      raise Exception('port must be specified as an integer, not %s.' % str(port))
    if port < 1 or port > 8:
      raise Exception('port must be between 1 and 8, not %s.' % str(port))
    getattr(self, 'set_port%s_PID_offset_on' % str(port))(onoff)
  
  def get_port_PID_offset_on(self, port):
    if not isinstance(port, int):
      raise Exception('port must be specified as an integer, not %s.' % str(port))
    if port < 1 or port > 8:
      raise Exception('port must be between 1 and 8, not %s.' % str(port))
    getattr(self, 'get_port%s_PID_offset_on' % str(port))()
  
  def set_port_PID_offset_gain(self, port, val):
    if not isinstance(port, int):
      raise Exception('port must be specified as an integer, not %s.' % str(port))
    if port < 1 or port > 8:
      raise Exception('port must be between 1 and 8, not %s.' % str(port))
    getattr(self, 'set_port%s_PID_offset_gain' % str(port))(val)
  
  def get_port_PID_offset_gain(self, port):
    if not isinstance(port, int):
      raise Exception('port must be specified as an integer, not %s.' % str(port))
    if port < 1 or port > 8:
      raise Exception('port must be between 1 and 8, not %s.' % str(port))
    getattr(self, 'get_port%s_PID_offset_gain' % str(port))()
  
  def set_port_PID_manual_output_on(self, port, onoff):
    if not isinstance(port, int):
      raise Exception('port must be specified as an integer, not %s.' % str(port))
    if port < 1 or port > 8:
      raise Exception('port must be between 1 and 8, not %s.' % str(port))
    getattr(self, 'set_port%s_PID_manual_output_on' % str(port))(onoff)
  
  def get_port_PID_manual_output_on(self, port):
    if not isinstance(port, int):
      raise Exception('port must be specified as an integer, not %s.' % str(port))
    if port < 1 or port > 8:
      raise Exception('port must be between 1 and 8, not %s.' % str(port))
    getattr(self, 'get_port%s_PID_manual_output_on' % str(port))()
  
  def set_port_PID_manual_output(self, port, val):
    if not isinstance(port, int):
      raise Exception('port must be specified as an integer, not %s.' % str(port))
    if port < 1 or port > 8:
      raise Exception('port must be between 1 and 8, not %s.' % str(port))
    getattr(self, 'set_port%s_PID_manual_output' % str(port))(val)
  
  def get_port_PID_manual_output(self, port):
    if not isinstance(port, int):
      raise Exception('port must be specified as an integer, not %s.' % str(port))
    if port < 1 or port > 8:
      raise Exception('port must be between 1 and 8, not %s.' % str(port))
    getattr(self, 'get_port%s_PID_manual_output' % str(port))()
    
  def set_port_PID_upper_limit(self, port, onoff):
    if not isinstance(port, int):
      raise Exception('port must be specified as an integer, not %s.' % str(port))
    if port < 1 or port > 8:
      raise Exception('port must be between 1 and 8, not %s.' % str(port))
    getattr(self, 'set_port%s_PID_upper_limit' % str(port))(onoff)
  
  def get_port_PID_upper_limit(self, port):
    if not isinstance(port, int):
      raise Exception('port must be specified as an integer, not %s.' % str(port))
    if port < 1 or port > 8:
      raise Exception('port must be between 1 and 8, not %s.' % str(port))
    getattr(self, 'get_port%s_PID_upper_limit' % str(port))()
  
  def set_port_PID_lower_limit(self, port, val):
    if not isinstance(port, int):
      raise Exception('port must be specified as an integer, not %s.' % str(port))
    if port < 1 or port > 8:
      raise Exception('port must be between 1 and 8, not %s.' % str(port))
    getattr(self, 'set_port%s_PID_lower_limit' % str(port))(val)
  
  def get_port_PID_lower_limit(self, port):
    if not isinstance(port, int):
      raise Exception('port must be specified as an integer, not %s.' % str(port))
    if port < 1 or port > 8:
      raise Exception('port must be between 1 and 8, not %s.' % str(port))
    getattr(self, 'get_port%s_PID_lower_limit' % str(port))()
  
  def do_set_ramp_stepsize(self, stepsize):
    self._ramp_stepsize = stepsize
  def do_get_ramp_stepsize(self):
    return self._ramp_stepsize

  def do_set_ramp_delaytime(self, delay):
    if delay < self._min_time_between_commands:
      logging.warn("Effective The ramp delay time is limited to >= %g ms due to communication speed limitations." % (self._min_time_between_commands * 1e3))
    self._ramp_delaytime = delay
  def do_get_ramp_delaytime(self):
    return self._ramp_delaytime
    
  def do_set_retries_on_set_error(self, retries):
    '''
    This command sets the number of retries in case setting the voltage or output state fails.
    If set to zero, the final output voltage/state is not checked after a set operation!
    
    Input:
        channel (int) : the port
        state (bool) : on (True) or off (False)

    Output:
        None
    '''
    self._retries_on_set_error = retries
  def do_get_retries_on_set_error(self):
    '''
    This command sets the number of retries in case setting the voltage or output state fails.
    If set to zero, the final output voltage/state is not checked after a set operation!
    '''
    return self._retries_on_set_error

  def do_set_voltage(self, voltage, channel):
    self._set_voltage(channel, voltage)
  def do_get_voltage(self, channel):
    return self._get_voltage(channel)
  def do_set_PID_propotional_on(self, onoff, channel):
    self._set_PID_propotional_on(channel,onoff)
  def do_get_PID_propotional_on(self, channel):
    return self._get_PID_propotional_on(channel)
  def do_set_PID_propotional_gain(self, val, channel):
    self._set_PID_propotional_gain(channel,val)
  def do_get_PID_propotional_gain(self, channel):
    return self._get_PID_propotional_gain(channel)
  def do_set_PID_integral_on(self, onoff, channel):
    self._set_PID_integral_on(channel,onoff)
  def do_get_PID_integral_on(self, channel):
    return self._get_PID_integral_on(channel)
  def do_set_PID_integral_gain(self, val, channel):
    self._set_PID_integral_gain(channel,val)
  def do_get_PID_integral_gain(self, channel):
    return self._get_PID_integral_gain(channel)
  def do_set_PID_derivative_on(self, onoff, channel):
    self._set_PID_derivative_on(channel,onoff)
  def do_get_PID_derivative_on(self, channel):
    return self._get_PID_derivative_on(channel)
  def do_set_PID_derivative_gain(self, val, channel):
    self._set_PID_derivative_gain(channel,val)
  def do_get_PID_derivative_gain(self, channel):
    return self._get_PID_derivative_gain(channel)
  def do_set_PID_offset_on(self, onoff, channel):
    self._set_PID_offset_on(channel,onoff)
  def do_get_PID_offset_on(self, channel):
    return self._get_PID_offset_on(channel)
  def do_set_PID_offset_gain(self, val, channel):
    self._set_PID_offset_gain(channel,val)
  def do_get_PID_offset_gain(self, channel):
    return self._get_PID_offset_gain(channel)
  def do_set_PID_manual_output_on(self, onoff, channel):
    self._set_PID_manual_output_on(channel,onoff)
  def do_get_PID_manual_output_on(self, channel):
    return self._get_PID_manual_output_on(channel)
  def do_set_PID_manual_output(self, val, channel):
    self._set_PID_manual_output(channel,val)
  def do_get_PID_manual_output(self, channel):
    return self._get_PID_manual_output(channel)
  def do_set_PID_upper_limit(self, onoff, channel):
    self._set_PID_upper_limit(channel,onoff)
  def do_get_PID_upper_limit(self, channel):
    return self._get_PID_upper_limit(channel)
  def do_set_PID_lower_limit(self, val, channel):
    self._set_PID_lower_limit(channel,val)
  def do_get_PID_lower_limit(self, channel):
    return self._get_PID_lower_limit(channel)    
    
  
  def do_set_on(self, val, channel):
    '''
    This command sets the output state of the port.
    Input:
        channel (int) : the port
        state (bool) : on (True) or off (False)

    Output:
        None
    '''
    self._set_on(channel, val)

  def do_get_on(self, channel):
    '''
    This command gets the output state of the port.
    Input:
        channel (int) : the queried port

    Output:
        state (int) : on (1) or off (0)
    '''
    return self._get_on(channel)
      
  def do_set_enabled(self, state, channel):
    '''
    This command sets the enabled state of the port.
    Input:
        channel (int) : the port
        state (bool) : on (True) or off (False)

    Output:
        None
    '''
    logging.debug(__name__ + ' : Set port %d enabled state = %s.' % (channel, state))
    if state:
      self._ch_enabled[channel-1] = True
    else:
      self._ch_enabled[channel-1] = False

  def do_get_enabled(self, channel):
    '''
    This command gets the enabled state of the port.
    Input:
        channel (int) : the queried port

    Output:
        state (int) : on (1) or off (0)
    '''
    return self._ch_enabled[channel-1]

  def do_get_idn(self, channel):
    '''
    This command gets the *IDN? response string from the specified port.
    Input:
        channel (int) : the queried port

    Output:
        IDN (string)
    '''
    #self._clear_mainframe_output_buffer()
    time.sleep(.2)
    self._clear_output_buffer(channel)
    self._write('SNDT %s,"*IDN?"' % channel)
    r = self._ask('GETN? %s,80' % channel)
    
    if (r[:2]!="#3"): raise Exception('Response %s is not in the expected format' % r)
        
    nbytes = int(r[2:5])
    bytes = r[5:5+nbytes].replace("\n","").replace("\r","")

    logging.info('SIM900 port %d IDN: %s' % (channel, bytes))

    return bytes

  def do_get_battery_status(self, channel):
    '''
    This command gets the battery state of the specified port.
    Input:
        channel (int) : the queried port

    Output:
        battery state (string)
    '''
    for attempt in range(1):
      try:
        self._clear_output_buffer(channel)
        self._write('SNDT %s,"BATS?"' % channel)
        r = self._ask('GETN? %s,80' % channel)
        
        logging.debug(__name__ + ' : getting port %d battery state: %s' % (channel, r))
      
        if (r[:2]!="#3"): raise Exception('Response %s is not in the expected format' % r)
      
        nbytes = int(r[2:5])
        bytes = r[5:5+nbytes].replace("\n","").replace("\r","")
        
        bytes = bytes.split(',')
        if len(bytes) != 3: raise Exception('Response %s is not in the expected format' % r)
        
        #logging.debug('Status bytes: %s' % str(bytes))
        
        for i in range(2):
          if bytes[i] == '1': bytes[i] = "used"
          if bytes[i] == '2': bytes[i] = "charging"
          if bytes[i] == '3': bytes[i] = "standby"
        if bytes[2] == '0': bytes[2] = "no"
        if bytes[2] == '1': bytes[2] = "yes"
        # otherwise leave the status byte as is (should always be one of the above though)
        
        return "A=%s, B=%s, service_required=%s" % tuple(bytes)
        
      except Exception as e:
        logging.warn('Attempt #%d to get port %d battery state from SIM failed: %s' % (1+attempt, channel, str(e)))
        time.sleep( .5 )
        self._clear_mainframe_output_buffer()
        self._wait_until_input_read(channel)
        self._clear_output_buffer(channel)
        time.sleep( .5+attempt )
