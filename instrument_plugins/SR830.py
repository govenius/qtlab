# SR830.py, Stanford Research 830 DSP lock-in driver
# Katja Nowack, Stevan Nadj-Perge, Arjan Beukman, Reinier Heeres
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
import time
import re

class SR830(Instrument):
    '''
    This is the python driver for the Lock-In SR830 from Stanford Research Systems.

    Usage:
    Initialize with
    <name> = instruments.create('name', 'SR830', address='<GPIB address>',
        reset=<bool>)
    '''

    def __init__(self, name, address, reset=False):
        '''
        Initializes the SR830.

        Input:
            name (string)    : name of the instrument
            address (string) : GPIB address
            reset (bool)     : resets to default values, default=false

        Output:
            None
        '''
        logging.info(__name__ + ' : Initializing instrument')
        Instrument.__init__(self, name, tags=['physical'])
        self._address = address
#>>>>>>>>>>>>>>
        assert False, "pyvisa syntax has changed, tweak the line below according to the instructions in qtlab/instrument_plugins/README_PYVISA_API_CHANGES"
        #self._visainstrument = visa.instrument(self._address, timeout=5.)
#<<<<<<<<<<<<<<

        self.add_parameter('mode',
           flags=Instrument.FLAG_SET,
           type=types.BooleanType)
        self.add_parameter('frequency', type=types.FloatType,
            flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET,
            minval=1e-3, maxval=102e3,
            units='Hz', format='%.04e')
        self.add_parameter('phase', type=types.FloatType,
            flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET,
            minval=-360, maxval=729.99, units='deg')
        self.add_parameter('harmonic',type=types.IntType,
                           flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET,
                           minval=1, maxval=19999)
        self.add_parameter('amplitude', type=types.FloatType,
            flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET,
            minval=0.004, maxval=5.0,
            units='V', format='%.04e')
        self.add_parameter('X', flags=Instrument.FLAG_GET, units='V', type=types.FloatType)
        self.add_parameter('Y', flags=Instrument.FLAG_GET, units='V', type=types.FloatType)
        self.add_parameter('R', flags=Instrument.FLAG_GET, units='V', type=types.FloatType)
        self.add_parameter('P', flags=Instrument.FLAG_GET, units='deg', type=types.FloatType)
        self.add_parameter('tau', type=types.IntType,
            flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET,
            format_map={
                0 : "10mus",
                1 : "30mus",
                2 : "100mus",
                3 : "300mus",
                4 : "1ms",
                5 : "3ms",
                6 : "10ms",
                7 : "30ms",
                8 : "100ms",
                9 : "300ms",
                10 : "1s",
                11 : "3s",
                12 : "10s",
                13 : "30s",
                14 : "100s",
                15 : "300s",
                16 : "1ks",
                17 : "3ks",
                18 : "10ks",
                19 : "30ks"
            })
        self.add_parameter('tau_in_seconds', flags=Instrument.FLAG_GETSET, units='s', type=types.FloatType) # for convenience

        self.add_parameter('out', type=types.FloatType, channels=(1,2,3,4),
            flags=Instrument.FLAG_GETSET,
            minval=-10.5, maxval=10.5, units='V', format='%.3f')
        self.add_parameter('in', type=types.FloatType, channels=(1,2,3,4),
            flags=Instrument.FLAG_GET,
            minval=-10.5, maxval=10.5, units='V', format='%.3f')
        self.add_parameter('reserve', type=types.IntType,
                           flags=Instrument.FLAG_GETSET,
                           format_map={0:'High reserve', 1:'Normal', 2:'Low noise'})
        self.add_parameter('input_config', type=types.IntType,
                           flags=Instrument.FLAG_GETSET,
                           format_map={0:'A', 1:'A-B', 2:'CVC 1MOhm', 3:'CVC 100MOhm'})
        self.add_parameter('input_shield', type=types.BooleanType,
                           flags=Instrument.FLAG_GETSET,
                           format_map={False:'Float', True:'GND'})
        self.add_parameter('input_coupling', type=types.BooleanType,
                           flags=Instrument.FLAG_GETSET,
                           format_map={False:'AC', True:'DC'})
        self.add_parameter('notch_filter', type=types.IntType,
                           flags=Instrument.FLAG_GETSET,
                           format_map={0:'off', 1:'1xline', 2:'2xline', 3:'both'})
        self.add_parameter('ref_input', type=types.BooleanType,
                           flags=Instrument.FLAG_GETSET,
                           format_map={False:'external', True:'internal'})
        self.add_parameter('ext_trigger', type=types.IntType,
                           flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET,
                           format_map={0:'Sine', 1:'TTL rising edge', 2:'TTL falling edge'})
        self.add_parameter('sync_filter', type=types.BooleanType,
                           flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET,
                           format_map={False:'off', True:'on'})
        self.add_parameter('filter_slope', type=types.IntType,
                           flags=Instrument.FLAG_GETSET,
                           format_map={0:'6dB/oct.', 1:'12dB/oct.', 2:'18dB/oct.', 3:'24dB/oct.'})
        self.add_parameter('unlocked', type=types.BooleanType,
                           flags=Instrument.FLAG_GET,
                           format_map={False:'locked', True:'unlocked'})
        self.add_parameter('input_overload', type=types.BooleanType,
                           flags=Instrument.FLAG_GET,
                           format_map={False:'normal', True:'overload'})
        self.add_parameter('time_constant_overload', type=types.BooleanType,
                           flags=Instrument.FLAG_GET,
                           format_map={False:'normal', True:'overload'})
        self.add_parameter('output_overload', type=types.BooleanType,
                           flags=Instrument.FLAG_GET,
                           format_map={False:'normal', True:'overload'})

        self._sensitivities_symbolic = {
                0 : "2nV",
                1 : "5nV",
                2 : "10nV",
                3 : "20nV",
                4 : "50nV",
                5 : "100nV",
                6 : "200nV",
                7 : "500nV",
                8 : "1muV",
                9 : "2muV",
                10 : "5muV",
                11 : "10muV",
                12 : "20muV",
                13 : "50muV",
                14 : "100muV",
                15 : "200muV",
                16 : "500muV",
                17 : "1mV",
                18 : "2mV",
                19 : "5mV",
                20 : "10mV",
                21 : "20mV",
                22 : "50mV",
                23 : "100mV",
                24 : "200mV",
                25 : "500mV",
                26 : "1V"
            }
        self.add_parameter('sensitivity', type=types.IntType,
            flags=Instrument.FLAG_GETSET,
            format_map=self._sensitivities_symbolic)
            
        # convert to volts
        unit_conversion = {'n': 1e-9, 'mu': 1e-6, 'm': 1e-3, '': 1.}
        self._sensitivities = [ (x[0],
                                 (lambda m: int(m[0])*unit_conversion[m[1]])(re.match(r'\s*(\d+)\s*([nmu]*)V', x[1]).groups())
                                 ) for x in self._sensitivities_symbolic.iteritems() ]
        self._sensitivities = dict(self._sensitivities)

        self.add_function('reset')
        self.add_function('get_all')
        self.add_function('reset_averaging')

        self.clear_output_buffer()
        if reset:
            self.reset()
        else:
            self.get_all()

        self.direct_output()

    # Functions
    def reset(self):
        '''
        Resets the instrument to default values

        Input:
            None

        Output:
            None
        '''
        logging.info(__name__ + ' : Resetting instrument')
        self.__write('*RST')
        self.clear_output_buffer()
        self.get_all()

    def clear_output_buffer(self):
        ''' Make sure there are no old replies in the visa buffer. '''
        old_timeout = self._visainstrument.timeout
        self._visainstrument.timeout = 0.3
        while True:
          try: r = self._visainstrument.read()
          except: break
          if len(r) == 0: break
        self._visainstrument.timeout = old_timeout

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
        self.get_sensitivity()
        self.get_tau()
        self.get_tau_in_seconds()
        self.get_frequency()
        self.get_amplitude()
        self.get_phase()
        self.get_X()
        self.get_Y()
        self.get_R()
        self.get_P()
        self.get_ref_input()
        self.get_ext_trigger()
        self.get_sync_filter()
        self.get_harmonic()
        self.get_input_config()
        self.get_input_shield()
        self.get_input_coupling()
        self.get_notch_filter()
        self.get_reserve()
        self.get_filter_slope()
        self.get_unlocked()
        self.get_input_overload()
        self.get_time_constant_overload()
        self.get_output_overload()

    def __ask(self, msg):
      ''' Internal helper that retries a few times in case the first attempt to communicate with the instrument fails. '''
      max_attempts = 5
      for attempt in range(max_attempts):
        try:
          return self._visainstrument.ask(msg)
        except:
          logging.exception('Attempt %d/%d to ask %s from SR830 failed.' % (1+attempt, max_attempts, msg))
          qt.msleep(attempt**2 * 1.)
          self.clear_output_buffer()
          if attempt == max_attempts-1: raise # last attempt failed
      raise Exception('This line should be unreachable.')

    def __write(self, msg):
      ''' Internal helper that retries a few times in case the first attempt to communicate with the instrument fails. '''
      max_attempts = 5
      for attempt in range(max_attempts):
        try:
          return self._visainstrument.write(msg)
        except:
          logging.exception('Attempt %d/%d to write %s to SR830 failed.' % (1+attempt, max_attempts, msg))
          qt.msleep(attempt**2 * 1.)
          self.clear_output_buffer()
          if attempt == max_attempts-1: raise # last attempt failed
      raise Exception('This line should be unreachable.')
        
    def disable_front_panel(self):
        '''
        enables the front panel of the lock-in
        while being in remote control
        '''
        self.__write('OVRM 0')

    def enable_front_panel(self):
        '''
        enables the front panel of the lock-in
        while being in remote control
        '''
        self.__write('OVRM 1')

    def auto_phase(self):
        '''
        offsets the phase so that
        the Y component is zero
        '''
        self.__write('APHS')

    def direct_output(self):
        '''
        select GPIB as interface
        '''
        self.__write('OUTX 1')

    def read_output(self,output, ovl):
        '''
        Read out R,X,Y or phase (P) of the Lock-In

        Input:
            mode (int) :
            1 : "X",
            2 : "Y",
            3 : "R"
            4 : "P"

        '''
        parameters = {
        1 : "X",
        2 : "Y",
        3 : "R",
        4 : "P"
        }
        #self.direct_output()
        if parameters.__contains__(output):
            logging.debug(__name__ + ' : Reading parameter from instrument: %s ' %parameters.get(output))
            if ovl:
                self.get_input_overload()
                self.get_time_constant_overload()
                self.get_output_overload()
            readvalue = float(self.__ask('OUTP?%s' %output))
            return readvalue
        else:
            raise Exception('Invalid output requested: %s' % str(output))

    def reset_averaging(self, instruments=None):
        '''
        Resets the averaging by temporarily reducing the time constant.
        
        kwargs:
          instruments -- reset the averaging on the specified list of instruments
                         instead of 'self.'
        '''
    
        insts = [self] if instruments == None else instruments

        old_taus = [ i.get_tau() for i in insts ]
        if np.array(old_taus).min() < 3: raise Exception('reset_averaging() not supported for tau < 3')

        max_tau = np.array(old_taus).max()

        # "erase" memory
        #for i in insts: i.set_tau(0)
        #qt.msleep(0.001)
        
        # set an intermediate tau
        #for i,tau_old in zip(insts, old_taus): i.set_tau(tau_old - 3)
        #qt.msleep(5*self.tau_index_to_seconds(max_tau - 3))
        for i,tau_old in zip(insts, old_taus): i.set_tau(tau_old - 2)
        qt.msleep(5*self.tau_index_to_seconds(max_tau - 2))
        for i,tau_old in zip(insts, old_taus): i.set_tau(tau_old - 1)
        qt.msleep(2*self.tau_index_to_seconds(max_tau - 1))
        
        # restore the old tau
        for i,tau_old in zip(insts, old_taus): i.set_tau(tau_old)
        
    def wait_for_steady_value(self, derivative_sign_changes=2, soft_averages_when_sampling=.3, instruments=None):
        '''
        Block execution until the measured value settles.
        
        kwargs:
          derivate_sign_changes --- wait for the sign of the time derivative of R to
                                    change this many times
          soft_averages_when_sampling  -- how long to average the samples (in units of tau)
          instruments -- wait for the specified list of instruments
                         instead of 'self.'
        '''
        insts = [self] if instruments == None else instruments
        
        r0 = (self.get_XY(soft_averages=soft_averages_when_sampling, instruments=insts)**2).sum(axis=1)
        r1 = (self.get_XY(soft_averages=soft_averages_when_sampling, instruments=insts)**2).sum(axis=1)

        sign_changed = np.array([ 0 for i in insts ], dtype=np.int)
        while sign_changed.min() < derivative_sign_changes:
          # get new samples
          r2 = (self.get_XY(soft_averages=soft_averages_when_sampling, instruments=insts)**2).sum(axis=1)
          #logging.debug('sampled r = %g' % r2)
          
          # check if the sign of the derivative changed
          sign_changed += ((r2-r1) * (r1-r0) <= 0)
          r0 = r1
          r1 = r2
          
          logging.debug('sign_changed = %s' % str(sign_changed))
          logging.debug('min(sign_changed) = %d < derivative_sign_changes = %d --> %s' % (sign_changed.min(), derivative_sign_changes, str(sign_changed.min() < derivative_sign_changes) ))
        
    def get_XY(self, ovl=False, soft_averages=None, instruments=None, auto_sensitivity=False):
        '''
        Get the current (X,Y) tuple.
        
        kwargs:
          soft_averages --- None or >= 0.1, which causes software averaging of the measured XY.
                            Specified in units of the time constant tau.
                            Note that soft_average=1 is different from None, since
                            the former averages (a few samples) over one tau, while
                            the latter returns a value immediately.
          instruments   --- return a list of (X,Y) tuples from the specified
                            list of instruments instead of 'self.' This way you
                            can do soft_averages on multiple instruments in parallel.
          auto_sensitivity --- tune sensitivity automatically if it is completely wrong
        '''
        assert soft_averages == None or soft_averages > .1, 'soft_averages ~< 0 does not make sense.'
        
        insts = [self] if instruments == None else instruments
        taus = [ i.get_tau_in_seconds() for i in insts ]
        max_tau = np.array(taus).max()

        assert soft_averages == None or max_tau > 0.090, 'soft_averages on ~< 100ms timescales is not a good idea.'
        
        while True: # repeat until value is in range
        
          xy = np.zeros((len(insts), 2), dtype=np.float)
          
          if soft_averages != None:
            samples = np.max((2, int(np.round(3 * soft_averages))))
            dt = soft_averages*max_tau / samples
            for j in range(samples):
              t0 = time.time()
              xy[:,0] += np.array([ i.read_output(1, ovl) for i in insts ])
              xy[:,1] += np.array([ i.read_output(2, ovl) for i in insts ])
              qt.msleep(np.max((0., dt - (time.time() - t0))))
            xy /= samples
          else:
            xy[:,0] += np.array([ i.read_output(1, ovl) for i in insts ])
            xy[:,1] += np.array([ i.read_output(2, ovl) for i in insts ])

          if auto_sensitivity:
            out_of_range = False
            sensitivity = np.array([ i.get_sensitivity() for i in insts ])
            for i in range(len(insts)):
              abs_val = np.abs(complex(*(xy[i])))
              current_range = self._sensitivities[sensitivity[i]]
              if abs_val > 0.9*current_range:
                if sensitivity[i] == len(self._sensitivities)-1:
                  logging.warn('Cannot increase sensitivity anymore!')
                else:
                  insts[i].set_sensitivity(sensitivity[i] + 1)
                  out_of_range = True
              elif abs_val < current_range/11.:
                if sensitivity[i] == 0:
                  logging.warn('Cannot decrease sensitivity anymore!')
                else:
                  insts[i].set_sensitivity(sensitivity[i] - 1)
                  out_of_range = True

            if out_of_range: continue # remeasure
          
          if instruments == None:
            return xy[0,:]
          else:
            return xy
        
    def do_get_X(self, ovl=False):
        '''
        Read out X of the Lock In
        Check for overloads if ovl is True
        '''
        return self.read_output(1, ovl)

    def do_get_Y(self, ovl=False):
        '''
        Read out Y of the Lock In
        Check for overloads if ovl is True
        '''
        return self.read_output(2, ovl)

    def do_get_R(self, ovl=False):
        '''
        Read out R of the Lock In
        Check for overloads if ovl is True
        '''
        return self.read_output(3, ovl)

    def do_get_P(self, ovl=False):
        '''
        Read out P of the Lock In
        Check for overloads if ovl is True
        '''
        return self.read_output(4, ovl)

    def do_set_frequency(self, frequency):
        '''
        Set frequency of the local oscillator

        Input:
            frequency (float) : frequency in Hz

        Output:
            None
        '''
        logging.debug(__name__ + ' : setting frequency to %s Hz' % frequency)
        self.__write('FREQ %e' % frequency)


    def do_get_frequency(self):
        '''
        Get the frequency of the local oscillator

        Input:
            None

        Output:
            frequency (float) : frequency in Hz
        '''
        #self.direct_output()
        logging.debug(__name__ + ' : reading frequency from instrument')
        return float(self.__ask('FREQ?'))

    def do_get_amplitude(self):
        '''
        Get the frequency of the local oscillator

        Input:
            None

        Output:
            frequency (float) : frequency in Hz
        '''
        #self.direct_output()
        logging.debug(__name__ + ' : reading frequency from instrument')
        return float(self.__ask('SLVL?'))

    def do_set_mode(self,val):
        logging.debug(__name__ + ' : Setting Reference mode to external' )
        self.__write('FMOD %d' %val)


    def do_set_amplitude(self, amplitude):
        '''
        Set frequency of the local oscillator

        Input:
            frequency (float) : frequency in Hz

        Output:
            None
        '''
        logging.debug(__name__ + ' : setting amplitude to %s V' % amplitude)
        self.__write('SLVL %e' % amplitude)


    def do_set_tau(self,timeconstant):
        '''
        Set the index of time constant of the LockIn

        Input:
            time constant (integer) : integer from 0 to 19

        Output:
            None
        '''

        #self.direct_output()
        logging.debug(__name__ + ' : setting time constant on instrument to %s'%(timeconstant))
        self.__write('OFLT %s' % timeconstant)
        self.update_value('tau_in_seconds', self.tau_index_to_seconds(timeconstant))

    def do_get_tau(self):
        '''
        Get the index of time constant of the LockIn

        Input:
            None
        Output:
            time constant (integer) : integer from 0 to 19
        '''

        #self.direct_output()
        r = self.__ask('OFLT?')
        ind = int(r)
        secs = self.tau_index_to_seconds(ind)
        logging.debug(__name__ + ' : getting time constant on instrument: %s == %.1e s' % (r, secs))
        self.update_value('tau_in_seconds', secs)
        return ind

    def do_get_tau_in_seconds(self):
        '''
        Get the time constant of the LockIn in seconds

        Input:
            None
        Output:
            time constant (float) : time constant in seconds
        '''
        return self.tau_index_to_seconds(self.get_tau())

    def do_set_tau_in_seconds(self, val):
        '''
        Set the time constant of the LockIn in seconds

        Input:
            (float) : time constant in seconds. Allowed values are given in the doc for get_tau().
        Output:
            None
        '''

        ind = self.seconds_to_tau_index(val)
        logging.debug('setting tau to %s --> %.1e' % (ind, val))
        self.set_tau(ind)
    
    def tau_index_to_seconds(self, ind):
        return 10**(-5 + int(ind)/2) * (1.+2*(int(ind)%2))
    
    def seconds_to_tau_index(self, val):
        # check that a valid value was provided
        if ( val < 10e-6-1e-9 or val > 30e3 + 1):
          raise Exception('Invalid time constant %.3e. Out of range.' % val)

        power = int(np.round(np.log10(val)))
        prefactor = 10 ** (np.log10(val) % 1)
        
        if not ( np.abs(prefactor - 1.) < 1e-3 or np.abs(prefactor - 3.) < 1e-3 ):
          raise Exception('Invalid time constant %.3e. The prefactor must be 1 or 3.' % val)

        return (0 if np.abs(prefactor-1) < 0.1 else 1) + 2*(power+5)

    def do_set_sensitivity(self, sens):
        '''
        Set the sensitivity of the LockIn

        Input:
            sensitivity (integer) : integer from 0 to 26

        Output:
            None
        '''

        #self.direct_output()
        logging.debug(__name__ + ' : setting sensitivity on instrument to %s'%(sens))
        self.__write('SENS %d' % sens)

    def do_get_sensitivity(self):
        '''
        Get the sensitivity
            Output:
            sensitivity (integer) : integer from 0 to 26
        '''
        #self.direct_output()
        logging.debug(__name__ + ' : reading sensitivity from instrument')
        return float(self.__ask('SENS?'))

    def sensitivity_to_volts(self, index):
        '''
        converts the integer returned by get_sensitivity() to the corresponding voltage.
        '''
        return self._sensitivities[index]

    def do_get_phase(self):
        '''
        Get the reference phase shift

        Input:
            None

        Output:
            phase (float) : reference phase shit in degree
        '''
        #self.direct_output()
        logging.debug(__name__ + ' : reading frequency from instrument')
        return float(self.__ask('PHAS?'))


    def do_set_phase(self, phase):
        '''
        Set the reference phase shift

        Input:
            phase (float) : reference phase shit in degree

        Output:
            None
        '''
        logging.debug(__name__ + ' : setting the reference phase shift to %s degree' %phase)
        self.__write('PHAS %e' % phase)

    def set_aux(self, output, value):
        '''
        Set the voltage on the aux output
        Input:
            output - number 1-4 (defining which output you are addressing)
            value  - the voltage in Volts
        Output:
            None
        '''
        logging.debug(__name__ + ' : setting the output %(out)i to value %(val).3f' % {'out':output,'val': value})
        self.__write('AUXV %(out)i, %(val).3f' % {'out':output,'val':value})

    def read_aux(self, output):
        '''
        Query the voltage on the aux output
        Input:
            output - number 1-4 (defining which output you are addressing)
        Output:
            voltage on the output D/A converter
        '''
        logging.debug(__name__ + ' : reading the output %i' %output)
        return float(self.__ask('AUXV? %i' %output))

    def get_oaux(self, value):
        '''
        Query the voltage on the aux output
        Input:
            output - number 1-4 (defining which output you are adressing)
        Output:
            voltage on the input A/D converter
        '''
        logging.debug(__name__ + ' : reading the input %i' %value)
        return float(self.__ask('OAUX? %i' %value))

    def do_set_out(self, value, channel):
        '''
        Set output voltage, rounded to nearest mV.
        '''
        self.set_aux(channel, value)

    def do_get_out(self, channel):
        '''
        Read output voltage.
        '''
        return self.read_aux(channel)

    def do_get_in(self, channel):
        '''
        Read input voltage, resolution is 1/3 mV.
        '''
        return self.get_oaux(channel)

    def do_get_ref_input(self):
        '''
        Query reference input: internal (true,1) or external (false,0)
        '''
        return int(self.__ask('FMOD?'))==1

    def do_set_ref_input(self, value):
        '''
        Set reference input: internal (true,1) or external (false,0)
        '''
        if value:
            self.__write('FMOD 1')
        else:
            self.__write('FMOD 0')

    def do_get_ext_trigger(self):
        '''
        Query trigger source for external reference: sine (0), TTL rising edge (1), TTL falling edge (2)
        '''
        return int(self.__ask('RSLP?'))

    def do_set_ext_trigger(self, value):
        '''
        Set trigger source for external reference: sine (0), TTL rising edge (1), TTL falling edge (2)
        '''
        self.__write('RSLP '+ str(value))

    def do_get_sync_filter(self):
        '''
        Query sync filter. Note: only available below 200Hz
        '''
        return int(self.__ask('SYNC?'))==1

    def do_set_sync_filter(self, value):
        '''
        Set sync filter. Note: only available below 200Hz
        '''
        if value:
            self.__write('SYNC 1')
        else:
            self.__write('SYNC 0')

    def do_get_harmonic(self):
        '''
        Query detection harmonic in the range of 1..19999.
        Note: frequency*harmonic<102kHz
        '''
        return int(self.__ask('HARM?'))

    def do_set_harmonic(self, value):
        '''
        Set detection harmonic in the range of 1..19999.
        Note: frequency*harmonic<102kHz
        '''
        self.__write('HARM '+ str(value))

    def do_get_input_config(self):
        '''
        Query input configuration: A (0), A-B (1), CVC 1MOhm (2), CVC 100MOhm (3)
        '''
        return int(self.__ask('ISRC?'))

    def do_set_input_config(self, value):
        '''
        Set input configuration: A (0), A-B (1), CVC 1MOhm (2), CVC 100MOhm (3)
        '''
        self.__write('ISRC '+ str(value))

    def do_get_input_shield(self):
        '''
        Query input shield: float (false,0), gnd (true,1)
        '''
        return int(self.__ask('IGND?'))==1

    def do_set_input_shield(self, value):
        '''
        Set input shield: float (false,0), gnd (true,1)
        '''
        if value:
            self.__write('IGND 1')
        else:
            self.__write('IGND 0')

    def do_get_input_coupling(self):
        '''
        Query input coupling: AC (false,0), DC (true,1)
        '''
        return int(self.__ask('ICPL?'))==1

    def do_set_input_coupling(self, value):
        '''
        Set input coupling: AC (false,0), DC (true,1)
        '''
        if value:
            self.__write('ICPL 1')
        else:
            self.__write('ICPL 0')

    def do_get_notch_filter(self):
        '''
        Query notch filter: none (0), 1xline (1), 2xline(2), both (3)
        '''
        return int(self.__ask('ILIN?'))

    def do_set_notch_filter(self, value):
        '''
        Set notch filter: none (0), 1xline (1), 2xline(2), both (3)
        '''
        self.__write('ILIN ' + str(value))

    def do_get_reserve(self):
        '''
        Query reserve: High reserve (0), Normal (1), Low noise (2)
        '''
        return int(self.__ask('RMOD?'))

    def do_set_reserve(self, value):
        '''
        Set reserve: High reserve (0), Normal (1), Low noise (2)
        '''
        self.__write('RMOD ' + str(value))

    def do_get_filter_slope(self):
        '''
        Query filter slope: 6dB/oct. (0), 12dB/oct. (1), 18dB/oct. (2), 24dB/oct. (3)
        '''
        return int(self.__ask('OFSL?'))

    def do_set_filter_slope(self, value):
        '''
        Set filter slope: 6dB/oct. (0), 12dB/oct. (1), 18dB/oct. (2), 24dB/oct. (3)
        '''
        self.__write('OFSL ' + str(value))
    def do_get_unlocked(self, update=True):
        '''
        Query if PLL is locked.
        Note: the status bit will be cleared after readout!
        Set update to True for querying present unlock situation, False for querying past events
        '''
        if update:
           self.__ask('LIAS? 3')     #for realtime detection we clear the bit by reading it
           time.sleep(0.02)                        #and wait for a little while so that it can be set
        return int(self.__ask('LIAS? 3'))==1

    def do_get_input_overload(self, update=True):
        '''
        Query if input or amplifier is in overload.
        Note: the status bit will be cleared after readout!
        Set update to True for querying present overload, False for querying past events
        '''
        if update:
            self.__ask('LIAS? 0')     #for realtime detection we clear the bit by reading it
            time.sleep(0.02)                        #and wait for a little while so that it can be set again
        return int(self.__ask('LIAS? 0'))==1

    def do_get_time_constant_overload(self, update=True):
        '''
        Query if filter is in overload.
        Note: the status bit will be cleared after readout!
        Set update to True for querying present overload, False for querying past events
        '''
        if update:
            self.__ask('LIAS? 1')     #for realtime detection we clear the bit by reading it
            time.sleep(0.02)                        #and wait for a little while so that it can be set again
        return int(self.__ask('LIAS? 1'))==1

    def do_get_output_overload(self, update=True):
        '''
        Query if output (also main display) is in overload.
        Note: the status bit will be cleared after readout!
        Set update to True for querying present overload, False for querying past events
        '''
        if update:
            self.__ask('LIAS? 2')     #for realtime detection we clear the bit by reading it
            time.sleep(0.02)                        #and wait for a little while so that it can be set again
        return int(self.__ask('LIAS? 2'))==1
