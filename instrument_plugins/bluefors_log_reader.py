# bluefors_log_reader.py
# Joonas Govenius <joonas.goveius@aalto.fi>, 2014
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
import types
import logging
import numpy as np
from scipy import interpolate
import datetime
import pytz
from dateutil import tz
import os
import qt
import time
import itertools
import re

class bluefors_log_reader(Instrument):
    '''
    This is a driver for reading the Bluefors dillution fridge log files.

    Usage:
    Initialize with
    <name> = instruments.create('<name>', 'bluefors_log_reader', address='<path_to_log_files>', reset=<bool>)
    '''

    def __init__(self, name, address, reset=False, temperature_channels=(1,2,5,6)):
        '''
        Initializes the bluefors_log_reader.

        Input:
          name (string)    : name of the instrument
          address (string) : path to log files
          reset (bool)     : resets to default values, default=False
        '''
        logging.info(__name__ + ' : Initializing instrument bluefors_log_reader')
        Instrument.__init__(self, name, tags=['physical'])

        # Add some global constants
        self._address = address
        self._UNIX_EPOCH = datetime.datetime(1970, 1, 1, 0, 0, tzinfo = pytz.utc)
        
        self._heater_current_to_t6_calibration_ends = 0.006 # in amps
        self._heater_current_to_t6_polyfit_coefficients = np.array([-2.07985, 1.97048e3, -1.71080e6, 8.57267e8, - 2.25600e11, 2.95946e13, -1.52644e15]) # for current in A, gives log10(T/K)

        self._tchannels = temperature_channels
        self._rchannels = self._tchannels
        self._pchannels = (1,2,3,4,5,6)
        
        self.add_parameter('latest_t', channels=self._tchannels, format='%.3g',
            flags=Instrument.FLAG_GET, units='K', type=types.FloatType)
        self.add_parameter('latest_r', channels=self._rchannels, format='%.3g',
            flags=Instrument.FLAG_GET, units='Ohm', type=types.FloatType)
        self.add_parameter('latest_p', channels=self._pchannels, format='%.3g',
            flags=Instrument.FLAG_GET, units='mbar', type=types.FloatType)

        self.add_parameter('latest_flow',
            flags=Instrument.FLAG_GET, units='mmol/s', type=types.FloatType, format='%.3f')

        self.add_function('get_all')
        self.add_function('get_temperature')
        self.add_function('get_pressure')
        self.add_function('get_flow')

        # Add a number of parameters that are stored and named according to the same convention.
        self._params_in_common_format = [('turbo frequency', 'Hz'),
                                         ('turbo power', 'W'),
                                         ('turbo temperature_body', 'C'),
                                         ('turbo temperature_bearing', 'C'),
                                         ('turbo temperature_controller', 'C'),
                                         ('turbo error_code', ''),
                                         ('compressor oil_temperature', 'C'),
                                         ('compressor helium_temperature', 'C'),
                                         ('compressor water_in_temperature', 'C'),
                                         ('compressor water_out_temperature', 'C'),
                                         ('compressor pressure_low', 'psi (absolute)'),
                                         ('compressor pressure_high', 'psi (absolute)')]
        for param,units in self._params_in_common_format:
          param_wo_spaces = param.replace(' ','_')
          load_param = lambda t, ss=self, p=param: ss.__load_data(t, '%s %%s.log' % p)
          interp_param = ( lambda t=None, pp=param_wo_spaces, load_fn=load_param:
                           self.__interpolate_value_at_time(pp, load_fn, t) )
          interp_param.__doc__ = '''
          Gets %s at time t.

          Input:
            t -- datetime object or a pair of them.
                 If a single datetime, an interpolated value is returned.
                 If a pair, all recorded points between are returned.
          ''' % param

          setattr(self, 'get_%s' % param_wo_spaces, interp_param)
          self.add_function('get_%s' % param_wo_spaces)

          setattr(self, 'do_get_latest_%s' % param_wo_spaces, interp_param)
          self.add_parameter('latest_%s' % param_wo_spaces, format='%.3g', units=units,
                             flags=Instrument.FLAG_GET, type=types.FloatType)
        
        self.add_function('base_heater_current_to_t6')


        if (reset):
            self.reset()
        else:
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
        logging.info(__name__ + ' : get all')
        
        for ch in self._tchannels: getattr(self, 'get_latest_t%s' % ch)()
        for ch in self._rchannels: getattr(self, 'get_latest_r%s' % ch)()
        for ch in self._pchannels: getattr(self, 'get_latest_p%s' % ch)()

        self.get_latest_flow()

        for param,units in self._params_in_common_format:
          getattr(self, 'get_latest_%s' % param.replace(' ','_'))()
        
    def reset(self):
        '''
        Resets the instrument to default values

        Input:
            None

        Output:
            None
        '''
        logging.info(__name__ + ' : resetting instrument')
        pass
      
    def get_temperature(self, channel, t=None):
        '''
        Gets the temperature of channel at time t.

        Input:
            channel -- channel no.
            t -- datetime object or a pair of them.
                 If a single datetime, an interpolated value is returned.
                 If a pair, all recorded points between are returned.

        Output:
            temperature in K
        '''

        logging.debug(__name__ + ' : getting temperature for channel {0} at t = {1}'.format(channel, str(t)))
        
        return self.__interpolate_value_at_time(
          'T%d' % channel, lambda t: self.__load_data(t, 'CH%s T %%s.log' % channel), t)

    def get_resistance(self, channel, t=None):
        '''
        Gets the resistance of channel at time t.

        Input:
            channel -- channel no.
            t -- datetime object or a pair of them.
                 If a single datetime, an interpolated value is returned.
                 If a pair, all recorded points between are returned.

        Output:
            resistance in Ohm
        '''

        logging.debug(__name__ + ' : getting resistance for channel {0} at t = {1}'.format(channel, str(t)))

        return self.__interpolate_value_at_time(
          'R%d' % channel, lambda t: self.__load_data(t, 'CH%s R %%s.log' % channel), t)

    def get_boolean_channels(self, t=None):
        '''
        Gets the boolean channel values at time t.

        Input:
            t -- datetime object or a pair of them.
                 If a pair, all recorded points between are returned.

        Output:
            Dictionary of all boolean channels.
        '''

        logging.debug(__name__ + ' : getting boolean channels at t = {0}'.format(str(t)))
        
        n_boolean_channels = 29
        formats = ['i8'] # + 1 integer right after the timestamp
        for i in range(n_boolean_channels): formats.append('S20'); formats.append('i1')

        def load_boolean_channels_data(t):
          dd = self.__load_data(t, 'Channels %s.log',
                                valueformats=formats )

          if (not isinstance(dd, np.ndarray)) or len(dd) == 0: raise Exception('load_data returned %s.' % dd)

          # Convert to dict
          # Not sure what the first value after the timestamp is... Code running status code?
          # Drop it.
          dd = map(lambda r: [ r[0], dict(zip(r[2::2], r[3::2])) ],
                   dd)

          return np.array(dd)
	
        return self.__interpolate_value_at_time('boolean_channels', load_boolean_channels_data, t,
                                                interpolation_kind='previous',
                                                value_if_data_not_available=None)

    def get_pressure(self, channel, t=None):
        '''
        Gets the pressure of channel at time t.

        Input:
            channel -- channel no.
            t -- datetime object or a pair of them.
                 If a single datetime, an interpolated value is returned.
                 If a pair, all recorded points between are returned.

        Output:
            pressure of channel in mbar at time t. nan if sensor was off.
        '''

        logging.debug(__name__ + ' : getting pressure for channel {0} at t = {1}'.format(channel, str(t)))
        
        def load_pressure_data(t):
          dd = self.__load_data(t, 'Maxigauge %s.log',
                                valueformats=['f', 'i1'],
                                usecols=(0,1,2+6*(channel-1)+3,2+6*(channel-1)+2))

          if (not isinstance(dd, np.ndarray)) or len(dd) == 0: raise Exception('load_data returned %s.' % dd)

          # replace value (2nd col) if the sensor was off (3rd col == 0)
          dd[dd[:,2] == 0, 1] = np.nan
          return dd[:,:2]
            
	
        return self.__interpolate_value_at_time('P%d' % channel, load_pressure_data, t)

    def get_flow(self, t=None):
        '''
        Gets the flow at time t.

        Input:
            t -- datetime object or a pair of them.
                 If a single datetime, an interpolated value is returned.
                 If a pair, all recorded points between are returned.

        Output:
            flow in mmol/s
        '''

        logging.debug(__name__ + ' : getting flow at t = {0}'.format(str(t)))
        
        return self.__interpolate_value_at_time(
          'flow', lambda t: self.__load_data(t, 'Flowmeter %s.log'), t)

    def do_get_latest_t(self, channel):
        '''
        Input:
            None

        Output:
            latest channel temperature in Kelvin.
        '''
        return self.get_temperature(channel)

    def do_get_latest_r(self, channel):
        '''
        Input:
            None

        Output:
            latest channel resistance in Ohms.
        '''
        return self.get_resistance(channel)

    def do_get_latest_p(self, channel):
        '''
        Input:
            None

        Output:
            latest channel pressure in mbar. nan if sensor is off.
        '''
        return self.get_pressure(channel)
        
    def do_get_latest_flow(self):
        '''
        Input:
            None

        Output:
            latest flow meter reading in mmol/s.
        '''
        return self.get_flow()


    def base_heater_current_to_t6(self, current):
      try:
        t6 = np.zeros(len(current))
        past_calibration_range = current.max() > self._heater_current_to_t6_calibration_ends
        scalar_input = False
      except TypeError:
        t6 = np.zeros(1)
        past_calibration_range = current > self._heater_current_to_t6_calibration_ends
        scalar_input = True
      
      if past_calibration_range:
        logging.warn("t6 has not been calibrated for heater currents exceeding %.3e Amp." % self._heater_current_to_t6_calibration_ends)
      
      for deg, coeff in enumerate(self._heater_current_to_t6_polyfit_coefficients):
        t6 += coeff * np.power(current, deg)
      
      # convert from log10 to linear scale
      t6 = np.power(10., t6)
      
      if scalar_input: return t6[0]
      else:            return t6

    def get_peak_values(self, channel='P5', time_periods=None, minimum=False, plot=False):
      '''
      Get the maximum value for the specified parameter over the specified time periods.

      The "channel" can be, e.g., 'P1', 'P2', 'T1', 'T2', 'oil_temperature'...

      If minimum=True, return the minimum instead of maximum.

      There are a few special "channels" that do something more complex:
        * 'tank pressure'
        * 'static compressor pressure high'
        * 'static compressor pressure low'
        * 'pre-warmup p6'

      In combination with find_cooldown(all_between=...) this is a nice way of following
      changes in tank pressure, compressor oil temperature, or base temperature
      over months or years.

      # For example:
      all_cd = bflog.find_cooldown(all_between=['12-03-22', '15-06-25']) # slow
      bflog.get_peak_values('P5', all_cd, plot=True)

      # Store the cooldown times for later use
      import cPickle as pickle
      with open('all_cooldowns.pickled', 'w') as f: pickle.dump(all_cd, f)
      # To load them later you can use:
      #with open('all_cooldowns.pickled', 'r') as f: all_cd = pickle.load(f)
      '''

      get_vals = None

      if channel.lower() == 'tank pressure':
        # get the tank pressure as measured by P4 during the mixture pump-out phase.
        def get_vals(ends):
          # find the end of the last subinterval where scroll1 and V13 were both on
          booleans = self.get_boolean_channels(ends)
          times = np.array([ b[0] for b in booleans ])
          vals = np.array([ b[1]['scroll1'] & b[1]['v13']  for b in booleans ], dtype=np.bool)

          try:
            last_on_end = times[1:][vals[:-1]][-1]
            last_on_start = None
            for t, v in reversed(np.array([times, vals]).T):
              if t >= last_on_end: continue
              if v: last_on_start = t
              else: break

            subinterval_start = last_on_start + (last_on_end-last_on_start)/2
            return self.get_pressure(4, (subinterval_start, last_on_end))

          except:
            logging.warn('Could not find a subinterval in %s when both scroll1 and V13 are on. Perhaps the mixture was not pumped out normally?', ends)
            return np.array([ (ends[0], np.nan), (ends[-1], np.nan) ])

      elif channel.lower() == 'pre-warmup p6':
        # Get P6 just before starting the warmup,
        # i.e., before the turbo is turned off.

        def get_vals(ends):
          # find the end of the last subinterval where we were circulating normally (with the turbo on).
          booleans = self.get_boolean_channels(ends)
          times = np.array([ b_next[0] for b,b_next in zip(booleans[:-1], booleans[1:])
                             if b[1]['scroll1'] & b[1]['turbo1']
                                & b[1]['v1'] & b[1]['v10'] & b[1]['v4']
                                & (b[1]['v8'] | (b[1]['v7'] & b[1]['v9'])) ])

          if len(times) < 1:
            logging.warn('Could not find a subinterval in %s when the circulation was normal.', ends)
            return np.array([ (ends[0], np.nan), (ends[-1], np.nan) ])

          last_on = times.max()

          subinterval_start = last_on - datetime.timedelta(hours=2)
          subinterval_end = last_on - datetime.timedelta(minutes=10)
          return self.get_pressure(6, (subinterval_start, subinterval_end))

      elif channel.lower() in [ 'static compressor pressure high', 'static compressor pressure low' ]:
        # Get the helium pressure in the interval before the compressor is turned on.
        # Assumes that the compressor has been off for a while.
        def get_vals(ends):
          # find out when the compressor is first turned on
          p_high = self.get_compressor_pressure_high(ends)
          if np.isscalar(p_high) or len(p_high) < 1:
              logging.warn('no compressor pressure data for %s', ends[0])
              return np.array([ (ends[0], np.nan), (ends[-1], np.nan) ])
          times = p_high[:,0]
          p_high = p_high[:,1]

          p_low_pts = self.get_compressor_pressure_low(ends)
          def p_low(t): return p_low_pts[np.argmin(np.abs(p_low_pts[:,0] - t)), 1]

          threshold_pressure_diff = 10. # psi
          if np.abs(p_high[0] - p_low(times[0])) > threshold_pressure_diff:
              # it should be off at the first time point
              logging.warn('compressor seems to be on at %s', times[0])
              return np.array([ (ends[0], np.nan), (ends[-1], np.nan) ])

          last_off = None
          for t, v in np.array([times, p_high]).T:
            if np.abs(v - p_low(t)) > threshold_pressure_diff: break
            last_off = t
          last_off -= (last_off - ends[0])/10

          if last_off == None:
            logging.warn('Could not find a subinterval in %s when both scroll1 and V13 are on. Perhaps the mixture was not pumped out normally?', ends)
            return np.array([ (ends[0], np.nan), (ends[-1], np.nan) ])

          interval = (ends[0], last_off)
          if channel.endswith('high'):
            return self.get_compressor_pressure_high(interval)
          elif channel.endswith('low'):
            return self.get_compressor_pressure_low(interval)
          else:
            assert False

      elif channel.lower().startswith('oil temperature') or channel.lower() == 'toil':
          get_vals = self.get_compressor_oil_temperature
      elif channel.lower().startswith('water in temperature') or channel.lower() == 'twaterin':
          get_vals = self.get_compressor_water_in_temperature
      elif channel.lower().startswith('water out temperature') or channel.lower() == 'twaterout':
          get_vals = self.get_compressor_water_out_temperature
      elif channel.lower().startswith('p'):
          get_vals = lambda x, ch=int(channel[1:]): getattr(self, 'get_pressure')(ch, x)
      elif channel.lower().startswith('t'):
          get_vals = lambda x, ch=int(channel[1:]): getattr(self, 'get_temperature')(ch, x)

      assert get_vals != None, 'Unknown channel %s' % channel


      def peak_fn(time_and_value_pairs):
        try:
          max_index = (np.argmin if minimum else np.argmax)(time_and_value_pairs[:,1])
          return time_and_value_pairs[max_index]
        except:
          return [np.nan, np.nan]

      time_and_peak_pairs = [ peak_fn( get_vals(ends) ) for ends in time_periods ]
      time_and_peak_pairs = np.array(filter(lambda x: isinstance(x[0], datetime.datetime),
                                            time_and_peak_pairs))

      if plot:
        import plot
        plt_name = 'BlueFors - peak %s from %s to %s' % (channel, time_and_peak_pairs[0][0].strftime('%Y-%m-%d'), time_and_peak_pairs[-1][0].strftime('%Y-%m-%d'))
        p = plot.get_plot(plt_name).get_plot()
        p.clear()

        p.set_title(plt_name)
        p.set_xlabel('time (days)')
        p.set_ylabel(channel)
        #p.set_ylog(True)

        ref_time = datetime.datetime(time_and_peak_pairs[0][0].year,
                                     time_and_peak_pairs[0][0].month,
                                     time_and_peak_pairs[0][0].day,
                                     0, 0, tzinfo=tz.tzlocal())
        hours_since_beginning = np.array([ (t - ref_time).total_seconds() for t in time_and_peak_pairs[:,0] ])/3600.
        p.add_trace(hours_since_beginning/24., time_and_peak_pairs[:,1].astype(np.float),
                    points=True, lines=False, title=channel)
        p.update()
        p.run()

      return time_and_peak_pairs

    def find_cooldown(self, near=None, forward_search=False, all_between=None):
      '''
      Find the start and end time of a cooldown (returned as a pair of datetime objects).

      near --- datetime object to begin the search from. Default is current time.
               Alternatively, can be a string in the "YY-MM-DD" format.
      forward_search --- search forward/backward in time, if near is not within a cooldown.
      all_between -- find all cooldowns between the dates specified as a pair of datetime
                     objects, or a pair of strings in the "YY-MM-DD" format.
      '''

      if all_between != None:
        logging.warn('Finding the cooldowns is quite slow. You can follow the progress from the INFO level log messages. Consider caching the results with, e.g., "all_cd = bflog.find_cooldown(all_between=[\'15-05-01\', \'15-06-25\']); import cPickle as pickle" and then "with open(\'all_cooldowns.pickled\', \'w\') as f: pickle.dump(all_cd, f)". ')
        # Find the latest one
        all_cooldowns = [ self.find_cooldown(near=all_between[1]) ]

        # Find all the ones before
        until = self.__parse_datestr(all_between[0])
        while True:
          c = all_cooldowns[-1]
          logging.info('Found a cooldown from %s to %s' % (c[0].strftime('%Y-%m-%d'), c[1].strftime('%m-%d')))
          if c[0] < until: break
          try:
            all_cooldowns.append( self.find_cooldown(near=c[0]) )
          except:
            logging.exception('Could not find any more cooldowns.')
            break

        all_cooldowns.reverse()
        return all_cooldowns
        

      flow_threshold = 0.05
      p1_threshold = 900.
      def within_cooldown(t):
        try:
          p1 = self.get_pressure(1,t)
          if p1 < p1_threshold:
            return True
          elif np.isnan(p1):
            raise Exception # P1 sensor is off or no data exists.
          else:
            return False
        except:
          # fall back to checking flow
          try: return self.get_flow(t) > flow_threshold
          except: return False

      dt_rough = datetime.timedelta(0.2*(2*int(forward_search)-1))

      # convert input to a datetime object
      if near == None:
        t = datetime.datetime.now(tz.tzlocal()) - datetime.timedelta(0,120)
      else:
        parsed = self.__parse_datestr(near)
        if parsed != None:
          t = parsed
        else:
          raise Exception('%s is neither None, a datetime object, or a string in the "YY-MM-DD" format.' % str(near))

      # find a point within a cooldown
      for i in range(400):
        t += dt_rough
        if within_cooldown(t): break

      assert within_cooldown(t), 'No cooldown found. Stopping search at: %s' % t

      # find the start and end points
      tstart = t
      dt_rough = datetime.timedelta(0.5)

      while within_cooldown(tstart):
        tstart -= dt_rough

      tend = t
      now = datetime.datetime.now(tz.tzlocal())
      while within_cooldown(tend) and tend < now:
        tend += dt_rough

      # get the end time more exactly based on flow
      flow = self.get_flow((tstart, tend))
      nonzero_flow = np.where(flow[:,1] > flow_threshold)[0]
      if len(nonzero_flow) > 0: # may not be the case if still pre-cooling
        tend = flow[nonzero_flow[-1], 0]
        tflowstart = flow[nonzero_flow[0], 0]
      else:
        tflowstart = t

      # get the start time more exactly based on P1
      p1 = self.get_pressure(1, (tstart, tend))
      vc_pumped = np.where(p1[:,1] < p1_threshold)[0]
      if len(vc_pumped) > 0: # should always be the case, unless logging was off
        tstart = min( p1[vc_pumped[0], 0],
                      tflowstart )

      # add some time to the beginning and end
      tstart -= datetime.timedelta(0, 10*60)
      tend += datetime.timedelta(1, 0)

      return (tstart, tend)

    def plot(self, start=None, end=None, time_since_start_of_day=False,
             flow=False, temperatures=True, resistances=False, pressures=False,
             turbo=False, compressor=False, heatswitches=False, condensing_compressor=False,
             scrolls=False):
      '''
      Plot statistics for the time range (start, end), specified as datetime objects,
      or alternatively, as strings in the "YY-MM-DD" format.

      If end is None, the current time is used.
      If start is None, the start of the previous cooldown before end is used.

      time_since_start_of_day means that the time axis will be given in hours
      since the beginning of the first day in the included range
      (makes it easier to figure out the corresponding time of day).
      Otherwise, it will be in hours since the first point.

      Returns the end points of the plotted timerange.
      '''

      ends = [None, None]
      for i,t in enumerate([start, end]):
        if t == None:
          continue
        elif isinstance(t, datetime.datetime):
          ends[i] = t
        else:
          parsed = self.__parse_datestr(t)
          if parsed != None:
            ends[i] = parsed
            if i == 1: ends[i] += datetime.timedelta(0, 23*3600 + 59*60 + 59)
          else:
            raise Exception('%s is neither None, a datetime object, or a string in the "YY-MM-DD" format.' % str(t))

      if ends[1] == None: ends[1] = datetime.datetime.now(tz.tzlocal())
      if ends[0] == None: ends[0] = self.find_cooldown(near=ends[1])[0]

      logging.info('Plotting %s.', ends)

      import plot
      p = plot.get_plot('BlueFors stats').get_plot()
      p.clear()
      p.set_title('BlueFors stats from %s to %s' % (ends[0].strftime('%Y-%m-%d'), ends[1].strftime('%m-%d')))
      p.set_xlabel('time (h)')
      p.set_ylog(True)

      quantities_to_plot = []

      if heatswitches or turbo or condensing_compressor or scrolls or compressor:
        booleans = self.get_boolean_channels(ends)
        if booleans != None:
          def bool_channel_as_vector_of_tuples(ch_name, offset=0):
            times = np.array([ b[0] for b in booleans ])
            vals = np.array([ offset + b[1][ch_name] for b in booleans ])

            # duplicate the points so that we get horizontal and vertical lines in line plots
            if len(times) > 1:
              times = np.array([ times[:-1], times[1:] ]).T.reshape((-1))
              times = np.append(times, [ times[-1] ])
              vals = np.array([ vals[:], vals[:] ]).T.reshape((-1))[:-1]

            return np.array([times, vals]).T

      if heatswitches:
          quantities_to_plot.append( ('hs-still', bool_channel_as_vector_of_tuples('hs-still',0.1), 0, 5 ) )
          quantities_to_plot.append( ('hs-mc', bool_channel_as_vector_of_tuples('hs-mc',0.15), 1, 5 ) )

      if condensing_compressor:
          quantities_to_plot.append( ('cond. compressor', bool_channel_as_vector_of_tuples('compressor',0.2), 0, 5 ) )

      if scrolls:
          quantities_to_plot.append( ('scroll1', bool_channel_as_vector_of_tuples('scroll1',0.05), 0, 5 ) )
          quantities_to_plot.append( ('scroll2', bool_channel_as_vector_of_tuples('scroll2',0.125), 1, 5 ) )

      if flow:
        q = self.get_flow(ends)
        if isinstance(q, np.ndarray):
          quantities_to_plot.append( ('flow (mmol/s)', q, 0, 5 ) )

      if temperatures:
        for ch in self._tchannels:
          q = self.get_temperature(ch, ends)
          if isinstance(q, np.ndarray):
            quantities_to_plot.append( ('T%s (K)' % ch, q, ch, 7 ) )

      if resistances:
        for ch in self._rchannels:
          q = self.get_resistance(ch, ends)
          if isinstance(q, np.ndarray):
            quantities_to_plot.append( ('R%s ({/Symbol O})' % ch, q, ch, 8 ) )

      if pressures:
        for ch in self._pchannels:
          q = self.get_pressure(ch, ends)
          if isinstance(q, np.ndarray):
            quantities_to_plot.append( ('P%s (mbar)' % ch, q, ch, 6 ) )

      prefixes = []
      if turbo: prefixes.append('turbo ')
      if compressor: prefixes.append('compressor ')
      for prefix in prefixes:

        if prefix == 'compressor ':
          try: quantities_to_plot.append( ('compressor ctrl panel switch', bool_channel_as_vector_of_tuples('compressor',0.2), 2, 5 ) )
          except: logging.exception('Could not plot compressor control panel switch status.')

        if prefix == 'turbo ':
          try: quantities_to_plot.append( ('turbo ctrl panel switch', bool_channel_as_vector_of_tuples('turbo1',0.2), 2, 5 ) )
          except: logging.exception('Could not plot turbo control panel switch status.')

        for paramno, param_and_units in enumerate(self._params_in_common_format):
            param, units = param_and_units

            if param.startswith(prefix):
              q = getattr(self, 'get_%s' % param.replace(' ','_'))(ends)
              if isinstance(q, np.ndarray):
                quantities_to_plot.append( ('%s (%s)' % (param.replace('_',' '), units),
                  q, paramno, 9 if prefix.startswith('turbo') else 10 ) )

      for title,pts,color,pointtype in quantities_to_plot:
        ref_time = datetime.datetime(ends[0].year, ends[0].month, ends[0].day, 0, 0, tzinfo=tz.tzlocal()) if time_since_start_of_day else ends[0]
        if len(pts) == 0:
          logging.warn('No %s data for the specified time period.', title)
          continue
        hours_since_beginning = np.array([ (t - ref_time).total_seconds() for t in pts[:,0] ]) / 3600.

        # color=6 is bright yellow in the default gnuplot color scheme. Skip it.
        try: color += 1 if color>5 else 0
        except: pass # in case color is not an integer

        p.add_trace(hours_since_beginning, pts[:,1].astype(np.float),
                    points=True, lines=True,
                    color=color,
                    pointtype=pointtype,
                    title=title)

      p.update()
      p.run()

      return ends

    def __interpolate_value_at_time(self, value_name, load_data, at_time=None, interpolation_kind='linear', cache_life_time=10., value_if_data_not_available=np.nan):
        '''
        Returns the interpolated value at 'at_time' based on the data loaded by the load_data function.

        Input:
            load_data(t)  -- function that loads the data in the neighborhood of time t
                             as a sequence of pairs [timestamp_as_datetime, value0_as_float, value1_as_float, ...]
            at_time    -- time to interpolate to, given as a datetime object.
                          Alternatively, at_time can be a pair of datetime objects specifying
                          a time range for which all recorded points are returned.
            value_name -- the value being queried, e.g. T1, T2, ... P1, P2, ...
            cache_life_time -- specifies how long previously parsed data is used (in seconds) before reparsing
            value_if_data_not_available -- what to return if loading real data was unsuccessful
            
        Output:
            Interpolated value at 'at_time'. Latest value if at_time==None.
        '''
        
        if at_time==None:
            t = datetime.datetime.now(tz.tzlocal())
        else:
            t = at_time
        
        # Check if a cache file for the given date exists
        
        try:
          if t[1] == None: t[1] = datetime.datetime.now(tz.tzlocal())
          if (t[1] - t[0]).total_seconds() <= 0:
            logging.warn('%s is not a pair of increasing datetime objects.', t)
            return np.array([])
          range_given = True
          cache_file_name = "%d-%d-%d_%d-%d-%d_%s_bflog.npy" % (t[0].year, t[0].month, t[0].day,
                                                                t[1].year, t[1].month, t[1].day,
                                                                value_name)
        except:
          # Assume that t is a datetime object
          range_given = False
          cache_file_name = "%d-%d-%d_%s_bflog.npy" % (t.year, t.month, t.day, value_name)

        cache_file_path = os.path.join(qt.config.get('tempdir'), cache_file_name)
        data = None
        from_cache = False
        try:
          if at_time != None and time.time() - os.path.getmtime(cache_file_path) < cache_life_time:
            with open(cache_file_path, 'rb') as f:
              data = np.load(f)
              logging.debug('Loaded cached data from %s.' % (cache_file_path))
              from_cache = True
        except Exception as e:
          # cache file probably doesn't exist
          logging.debug('Failed to load a cached interpolating function from %s: %s' % (cache_file_path, str(e)))

        if not from_cache:
          # parse the data log files
          try:
            data = load_data(t)
            if (not isinstance(data, np.ndarray)) or len(data) == 0: raise Exception('load_data returned %s.' % data)
          except Exception as e:
            logging.exception('Could not load %s at %s. Returning %s.' % (value_name, str(t), value_if_data_not_available))
            return value_if_data_not_available
                                                    
          try:
            with open(cache_file_path, 'wb') as f:
              np.save(f, data)
              logging.debug('Cached data in %s.' % (cache_file_path))
          except Exception as e:
            logging.debug('Could not dump data in %s: %s' % (cache_file_path, str(e)))

        # return the latest data point if nothing was specified.
        if at_time==None:
          if (t - data[-1][0]).total_seconds() > 305:
            logging.warn('last %s point from %s ago.' % (value_name, str(t - data[-1][0])))
          columns_to_return = 1 if len(data[-1]) == 2 else slice(1,None)
          return data[-1][columns_to_return]

        # if a range was specified, return all points in it
        if range_given:
          timestamps = data[:,0]
          return data[np.logical_and(timestamps >= t[0], timestamps <= t[1])]
      
        # create the interpolating function
        val_times = [ (d[0] - self._UNIX_EPOCH).total_seconds() for d in data ]
        vals = data[:,1]

        if interpolation_kind == 'previous':
          def interpolating_fn(ttt):
            assert np.isscalar(ttt)
            return vals[val_times < ttt][-1]

        else:
          interpolating_fn = interpolate.interp1d(val_times, vals,
                                                  kind=interpolation_kind, bounds_error=True)

        # finally use the interpolation function
        try:
            val = interpolating_fn((t - self._UNIX_EPOCH).total_seconds())
        except Exception as e:
            logging.warn('Could not interpolate value %s for t=%s: %s' %(value_name, str(t), str(e)))
            raise e
        
        return val


    def __load_data(self, t, filename, valueformats=['f'], usecols=None):
      ''' Load data from the day specified by t (datetime object)
          as well as the preceding and following ones.
          Alternatively, t can be a pair of datetime objects, in which case
          it is interpreted as a date range to load.

          filename must be a string with "%s" in place of the date string,
            e.g., "Flowmeter %s.log".

          valueformats describes the formats of the values stored on each line,
          excluding the time stamp.

          usecols can be passed as an additional parameter to loadtxt
          in order to ignore some columns.
      '''

      all_data = None

      try:
        assert (t[1] - t[0]).total_seconds() > 0, 't is not a pair of increasing datetime objects.'
        dates = [ self.__time_to_datestr(t[0]) ]
        i = 0
        while self.__time_to_datestr(t[1]) != dates[-1]:
          i += 1
          dates.append(self.__time_to_datestr( t[0] + datetime.timedelta(i,0,0,0) ))
      except:
        # Assume that t is a datetime object
        dates = map(self.__time_to_datestr,
                    [t-datetime.timedelta(1,0,0,0), t, t+datetime.timedelta(1,0,0,0)])

      for datestr in dates:
        fname = os.path.join(self._address, datestr, filename % datestr)

        # Some newer versions of the BlueFors software store the pressures in a file called
        # "maxigauge..." rather than "Maxigauge...", so also try a lower cased version.
        if not os.path.exists(fname): fname = os.path.join(self._address, datestr, filename.lower() % datestr)

        try:
          data = np.loadtxt(fname,
                            dtype={
                              'names': tuple(itertools.chain(['date', 'time'], ['value%d' % i for i in range(len(valueformats)) ])),
                              'formats': tuple(itertools.chain(['S9', 'S8'], valueformats))
                            }, delimiter=',', usecols=usecols, ndmin=1)

          # convert the date & time strings to a datetime object
          data = np.array([ list(itertools.chain(
              [ datetime.datetime(int('20'+d[0].strip()[6:8]),
                                  int(d[0].strip()[3:5]),
                                  int(d[0].strip()[0:2]),
                                  int(d[1][0:2]), int(d[1][3:5]), int(d[1][6:8]),
                                  tzinfo=tz.tzlocal()) ],
              ( d[2+i] for i in range(len(valueformats)) )
            )) for d in data ])

          all_data = np.concatenate((all_data, data), axis=0) if isinstance(all_data, np.ndarray) else data

        except IOError as e:
          pass # file doesn't exist. this is fairly normal, especially if datestr is in the future

        except Exception as e:
          logging.exception('Failed to load data from %s.' % str(fname))

      if not isinstance(all_data, np.ndarray):
          logging.warn('No data loaded for t = %s. Last attempt was from %s.', str(t), fname)

      return all_data


    def __time_to_datestr(self, t):
      ''' Generate a string in the "YY-MM-DD" format from a date, i.e.,
          the same format as the folder naming for the BlueFors log files. '''
      return '{0}-{1:02d}-{2:02d}'.format(str(t.year)[-2:], t.month, t.day)

    def __parse_datestr(self, datestr):
      ''' Parse a date given in the "YY-MM-DD" format, i.e.,
          the same format as the folder naming for the BlueFors log files. '''
      if isinstance(datestr, datetime.datetime): return datestr # already a datetime object
      m = re.match(r'(\d\d)-(\d\d)-(\d\d)', datestr)
      if m == None: return None
      assert len(m.groups()) == 3

      return datetime.datetime(int('20'+m.group(1)), int(m.group(2)), int(m.group(3)), 0, 0, tzinfo=tz.tzlocal())
