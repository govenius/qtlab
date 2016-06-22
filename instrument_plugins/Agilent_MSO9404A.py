# Agilent_MSO9404A.py class, for commucation with an Agilent MSO9404A mixed signal oscilloscope.
# Russell Lake <russell.lake@aalto.fi>
# Joonas Govenius <joonas.govenius@aalto.fi>
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
from scipy import interpolate
import struct
import qt
import time

class Agilent_MSO9404A(Instrument):
    '''
    This is the driver for the Agilent MSO9404A scope.

    Usage:
    Initialize with
    <name> = instruments.create('<name>', 'Agilent_MSO9404A', address='<GBIP address>, reset=<bool>')

    Typical use (assuming <name> == scope):

    # Configure:
    scope.reset()
    scope.set_timebase_ref_clock_mode('EXT')
    scope.set_ch1_input_coupling('DC')
    scope.set_ch2_input_coupling('DC')
    scope.set_display(1,1)
    scope.set_display(1,2)
    scope.set_ch1_vertical_range(8.)
    scope.set_ch1_vertical_offset(.5)
    scope.set_ch2_vertical_range(8.)
    scope.set_ch2_vertical_offset(0.)
    scope.set_acquire_mode('RTIM')
    scope.set_acquire_average_mode('ON')
    scope.set_acquire_average_count(4)
    scope.stop()

    # Connect probe comp. to channel 1 for this test.
    scope.set_timebase_range(5e-3) # Total duration to capture
    scope.set_timebase_reference('LEFT')
    scope.set_timebase_position(0.)

    scope.set_acquire_points(4096)

    # Assuming you wish to trigger from channel 1:
    scope.setup_edge_trigger(trigger_source_channel=1, trigger_level=.5)

    ## Acquire the data
    scope_data = scope.acquire_and_return_data(channels=[1,2], resample=True)
    time_axis = scope_data[0]['time_axis']
    ch1 = scope_data[0]['wav']
    ch2 = scope_data[1]['wav']


    # Now you can e.g. plot ch1 vs time_axis.
    p = plot.get_plot('scope channels')
    p = p.get_plot()
    p.clear()

    p.add_trace(time_axis, ch1,
                lines=True,
                points=True,
                title='ch1')
    p.add_trace(time_axis, ch2,
                lines=True,
                points=True,
                title='ch2')

    p.set_xlabel('time (s)')
    p.set_ylabel('voltage (V)')

    p.update()
    #p.run() # This reopens the plot, if you closed it
    '''

    def __init__(self, name, address, reset=False):
        '''
        Initializes the Agilent MSO9404A, and communicates with the wrapper.

        Input:
          name (string)    : name of the instrument
          address (string) : GPIB address
          reset (bool)     : resets to default values, default=False
        '''
        logging.info(__name__ + ' : Initializing instrument Agilent_MSO9404A')
        Instrument.__init__(self, name, tags=['physical'])

        # Add some global constants
        self._address = address
        self._visainstrument = visa.ResourceManager().open_resource(self._address, timeout=30000) # timeout is in milliseconds
        self._visainstrument.read_termination = '\n'
        self._visainstrument.write_termination = '\n'

        # Tell the instrument not to repeat the command in the responses. We assume this everywhere.
        self._visainstrument.write(':SYST:HEAD 0')
        
        # # we assume most-significant-byte-first WORDs in __words_to_waveform()
        # self._visainstrument.write(':WAVEFORM:FORMAT WORD')
        # self._visainstrument.write(':WAV:BYT MSBF')

        # we assume ASCii in __ascii_to_waveform()
        self._visainstrument.write(':WAVEFORM:FORMAT ASC')

        # This is important for get_waveform() to actually return what we see on the screen
        self._visainstrument.write(':WAVEFORM:VIEW WIND')
        
        # Timebase parameters 
        self.add_parameter('timebase_range',
            flags=Instrument.FLAG_GETSET|Instrument.FLAG_GET_AFTER_SET, units='s', minval=5E-11, maxval=200, type=types.FloatType)

        self.add_parameter('timebase_scale',
            flags=Instrument.FLAG_GETSET|Instrument.FLAG_GET_AFTER_SET, units='s/div', minval=5E-12, maxval=20, type=types.FloatType)

        self.add_parameter('timebase_delay',
            flags=Instrument.FLAG_GETSET|Instrument.FLAG_GET_AFTER_SET, units='s', minval=0., maxval=10., type=types.FloatType)

        self.add_parameter('timebase_position',
            flags=Instrument.FLAG_GETSET|Instrument.FLAG_GET_AFTER_SET, units='s', minval=0., maxval=20., type=types.FloatType)

        self.add_parameter('timebase_reference',
            flags=Instrument.FLAG_GETSET|Instrument.FLAG_GET_AFTER_SET, type=types.StringType, 
                           format_map={"LEFT" : "left","CENT" : "center","RIGH" : "right"})
                           
        self.add_parameter('timebase_ref_clock_mode',
              flags=Instrument.FLAG_GETSET|Instrument.FLAG_GET_AFTER_SET,
              type=types.StringType,
              option_list=('INT', 'EXT')
            )

        # 'Channel' parameters 
        self.add_parameter('vertical_range',
            flags=Instrument.FLAG_GETSET|Instrument.FLAG_GET_AFTER_SET, channels=(1, 4), units='unit', minval=0.001, 
                           type=types.FloatType, channel_prefix='ch%d_')

        self.add_parameter('vertical_scale',
            flags=Instrument.FLAG_GETSET|Instrument.FLAG_GET_AFTER_SET, channels=(1, 4), units='unit', minval=0.001, maxval=1, 
                           type=types.FloatType, channel_prefix='ch%d_')

        self.add_parameter('vertical_offset',
            flags=Instrument.FLAG_GETSET|Instrument.FLAG_GET_AFTER_SET, channels=(1, 4), units='unit', minval=-10, maxval=10, 
                           type=types.FloatType, channel_prefix='ch%d_')

        self.add_parameter('vertical_label',
            flags=Instrument.FLAG_GETSET|Instrument.FLAG_GET_AFTER_SET, channels=(1, 4),
                           type=types.StringType, channel_prefix='ch%d_')

        self.add_parameter('vertical_units',
            flags=Instrument.FLAG_GETSET|Instrument.FLAG_GET_AFTER_SET, channels=(1, 4),
                           type=types.StringType, channel_prefix='ch%d_',
                           format_map={"VOLT" : "volt","AMPere" : "AMP","WATT" : "Watt", "UNKNown" : "unknown" })

        self.add_parameter('input_coupling',
            flags=Instrument.FLAG_GETSET|Instrument.FLAG_GET_AFTER_SET, channels=(1, 4),
                           type=types.StringType, channel_prefix='ch%d_',
                           format_map={"DC" : "DC coupling","DC50" : "DC 50 Ohm impedance",
                                     "AC" : "AC 1MOhm impedance", "LFR1" : "1 MOhm input impedance"})

        self.add_parameter('waveform_preamble',
            flags=Instrument.FLAG_GET,
                           type=types.DictionaryType)

        # triggering parameters
        self.add_parameter('trigger_mode',
            flags=Instrument.FLAG_GETSET|Instrument.FLAG_GET_AFTER_SET, type=types.StringType, 
                           format_map={"EDGE" : "edge",
                                       "GLIT" : "glitch",
                                       "PATT" : "pattern",
                                       "STAT" : "state",
                                       "DEL" : "delay",
                                       "TIM" : "timeout",
                                       "TV" : "television waveform",
                                       "COMM" : "serial pattern",
                                       "RUNT" : "runt",
                                       "SHOL" : "setup and hold",
                                       "TRAN" : "edge transition",
                                       "WIND" : "window",
                                       "PWID" : "pulse width",
                                       "ADV" : "advanced",
                                       "SBUS1" : "serial triggering sbus#",
                                       "SBUS2" : "serial triggering sbus#",
                                       "SBUS3" : "serial triggering sbus#",
                                       "SBUS4" : "serial triggering sbus#"
                                       })

        self.add_parameter('trigger_sweep_mode',
            flags=Instrument.FLAG_GETSET|Instrument.FLAG_GET_AFTER_SET, type=types.StringType, 
                           format_map={"AUTO" : "auto","TRIG" : "triggered","SING" : "single"})

        self.add_parameter('trigger_and_source',
            flags=Instrument.FLAG_GETSET|Instrument.FLAG_GET_AFTER_SET, type=types.StringType,channels=(1, 4),
            channel_prefix='ch%d_', format_map={"HIGH" : "high","LOW" : "low","DONT" : "don't care"})

        self.add_parameter('trigger_level',
            flags=Instrument.FLAG_GETSET|Instrument.FLAG_GET_AFTER_SET, channels=(1, 4), units='current unit', minval=-5., maxval=5., 
                           type=types.FloatType, channel_prefix='ch%d_')

##----------------------------- edge trigger parameters p.839 --------------------

        self.add_parameter('edge_trigger_coupling',
           flags=Instrument.FLAG_GETSET|Instrument.FLAG_GET_AFTER_SET,
                          type=types.StringType, format_map={"AC" : "ac","DC" : "dc","LFR" : "LF reject", "HFR" : "HF reject"})
        

        self.add_parameter('edge_trigger_slope',
            flags=Instrument.FLAG_GETSET|Instrument.FLAG_GET_AFTER_SET,
                           type=types.StringType, format_map={"POS" : "positive","NEG" : "negative","EITH" : "either"})


        self.add_parameter('edge_trigger_source',
            flags=Instrument.FLAG_GETSET|Instrument.FLAG_GET_AFTER_SET,
                           type=types.StringType, format_map={"CHAN1" : "channel 1",
                                                              "CHAN2" : "channel 2", 
                                                              "CHAN3" : "channel 3", 
                                                              "CHAN4" : "channel 4", 
                                                              "DIG#" : "# from  0 to 15",
                                                              "AUX" : "aux",
                                                              "LINE" : "line"})
##------Acqusition parameters -----------------------------------------------------------------------
        self.add_parameter('acquire_mode',
            flags=Instrument.FLAG_GETSET|Instrument.FLAG_GET_AFTER_SET,
                           type=types.StringType, format_map={"ETIM" : "Equivalent Time",
                                                              "RTIM" : "Real Time", 
                                                              "PDET" : "Real Time Peak Detect", 
                                                              "HRES" : "High Resolution Real Time", 
                                                              "SEGM" : "Segmented", 
                                                              "SEGP" : "Peak Detect Segmented Mode",
                                                              "SEGH" : "High Resolution Segmented Mode"})

        self.add_parameter('waveform_source',
            flags=Instrument.FLAG_GETSET|Instrument.FLAG_GET_AFTER_SET,
                           type=types.StringType, format_map={"": "no arguement",
                                                              "CHAN1" : "channel 1",
                                                              "CHAN2" : "channel 2", 
                                                              "CHAN3" : "channel 3", 
                                                              "CHAN4" : "channel 4", 
                                                              "DIFF1" : "differential between 1,3", 
                                                              "COMM1" : "common mode between 1,3", 
                                                              "DIFF2" : "differential between 2,4", 
                                                              "COMM2" : "common mode between 2,4",
                                                              "others.." : "see p. 1028"})                                                                                                                      

        self.add_parameter('acquire_average_count',
            flags=Instrument.FLAG_GETSET|Instrument.FLAG_GET_AFTER_SET, minval=2, maxval=65534, type=types.IntType)


        self.add_parameter('acquire_average_mode',
            flags=Instrument.FLAG_GETSET|Instrument.FLAG_GET_AFTER_SET, type=types.StringType)


        self.add_parameter('acquire_points',
            flags=Instrument.FLAG_GETSET|Instrument.FLAG_GET_AFTER_SET, type=types.IntType)

#####  ---------------------------------

      # Add functions
        self.add_function('reset')
        self.add_function ('get_all')
        self.add_function('clear_status')
        self.add_function('run')
        self.add_function('stop')
        self.add_function('single')        
        self.add_function('autoscale')
        self.add_function('autoscale_vertical')
        self.add_function('digitize')
        self.add_function('beep')
        
        
        if (reset):
            self.reset()
        else:
            self.get_all()



# Functions

    def run(self):
        '''
        Run continuously.

        Input:
            None

        Output:
            None
        '''
        logging.debug(__name__ + ' : run continuously.')
        self._visainstrument.write(':RUN')


    def beep(self):
        '''
        Beep tones.

        Input:
            None

        Output:
            None
        '''
        logging.info(__name__ + ' : Playing three beep tones.')
        self._visainstrument.write(':BEEP 440,200')
        self._visainstrument.write(':BEEP 392.00,200')
        self._visainstrument.write(':BEEP 783.99,200')



    def stop(self):
        '''
        Resets the instrument to default values

        Input:
            None

        Output:
            None
        '''
        logging.debug(__name__ + ' : Stop acquiring data')
        self._visainstrument.write(':STOP')


    def single(self):
        '''
        Triggers a single acquisition (without blocking execution).
        '''
        logging.debug(__name__ + ' : Make a single acquisition.')
        
        self._visainstrument.write(':SINGLE')


    def acquire_and_return_data(self, channels, resample=False):
        '''
        Resets averaging, then
        triggers enough acquisition until average count has been reached.
        Blocks execution until data is returned.

        Input:
            channels (sequence): channels for which data is returned
            resample:
                resample the data to the desired time interval/number of points
                (the scope often returns more points than you request)

        Output:
            None
        '''
        for attempt in range(3):

          if self.get_acquire_average_mode() == 'ON':
            self.reset_averaging()
            target_avg_count = self.get_acquire_average_count()
          else:
            target_avg_count = 1
          
          total_duration = self.get_timebase_range()
          
          self.run()
          start_time = time.time()

          avg_count = 0
          
          for wait_attempt in range(8):
            # get the data from the scope
            try:
              # make sure scope digitize has actually completed
              
              fraction_acquired = avg_count / float(target_avg_count)
              if avg_count < 10 and fraction_acquired < 0.1:
                sleeping = .5 + target_avg_count*total_duration + (wait_attempt) * max(target_avg_count*total_duration, 2.)
              else:
                time_spent = time.time() - start_time
                time_left = time_spent/fraction_acquired * (target_avg_count-avg_count)/float(target_avg_count)
                sleeping = 1.*wait_attempt + time_left
              logging.debug('%d/%d averages acquired. Expecting measurement to be done in %g seconds.' % (avg_count, target_avg_count, sleeping))
              qt.msleep( sleeping )

              avg_count = self.get_waveform_count()
              complete = (avg_count >= target_avg_count)
              if not complete: raise Exception( 'Acquisition taking longer (already %g s) than expected (%g s). %d/%d averages acquired.'
                                                % (time.time() - start_time,
                                                  total_duration*target_avg_count,
                                                  avg_count, target_avg_count))
              self.stop()
              break

            except Exception as e:
              if str(e).lower().strip() == 'human abort': raise
              if wait_attempt > 0:
                logging.warn('Sleeping more... Exception was: %s' % str(e)) 
              else:
                logging.debug('Sleeping more... Exception was: %s' % str(e)) 
              
          logging.debug('Data digitized. Now fetching it from the scope.')
          pres = []
          wavs = []
          try:
            for ch in channels:
              self.set_waveform_source('CHAN%u' % ch)
              pre, wav = self.get_waveform()
              pres.append(pre)
              wavs.append(wav)
              logging.debug('Got %d pts for channel %s' % (len(wavs[-1]), ch))
          except:
            logging.exception('Failed to get waveform from scope. Trying again...')
            continue

          if not complete:
            raise Exception('Failed to reach %d averages as requested. (Got up to %d.)' % (target_avg_count, avg_count))

          if not resample:
            return [ {'pre': p, "wav": w, 'time_axis': self.get_time_axis(p) }
                     for p,w in zip(pres, wavs) ]

          # often the scope returns a longer time segment than you ask for...
          # resample the returned data to match the specified timebase and number of points
          target_no_of_pts = self.get_acquire_points()
          new_time_axis = np.linspace(0, total_duration, target_no_of_pts)
          epsilon_t = total_duration / float(target_no_of_pts) / 10. # negligible amount of time (for == comparisons)

          assert (np.abs(self.get_timebase_position()) < epsilon_t
                  and self.get_timebase_reference().lower() == 'left'), 'resample only implemented for timebase ref = LEFT and position = 0.'
          
          for i in range(len(wavs)):
            scope_time_axis = self.get_time_axis(pres[i])
            returned_no_of_pts = len(wavs[i])
            
            if (returned_no_of_pts != target_no_of_pts
                or np.abs(scope_time_axis[0]) > epsilon_t
                or np.abs(scope_time_axis[-1] - total_pattern_duration) > epsilon_t):
              first_pt = np.argmin(np.abs(scope_time_axis))
              last_pt = np.argmin(np.abs(scope_time_axis - total_duration))
              
              # smooth before resampling (if down sampling)
              scope_dt = np.median(np.abs(scope_time_axis[1:] - scope_time_axis[:-1]))
              requested_dt = np.median(np.abs(new_time_axis[1:] - new_time_axis[:-1]))
              smooth_window = int(np.floor(requested_dt/scope_dt))
              if smooth_window > 2:
                smooth_wav = np.convolve(wavs[i], np.ones(smooth_window,dtype=np.float)/smooth_window, mode='same')
              else:
                smooth_wav = wavs[i]
              
              # resample
              wavs[i] = interpolate.interp1d(scope_time_axis, smooth_wav, kind='linear',
                                             bounds_error=False, fill_value=np.nan)(new_time_axis)
              invalid_pts = np.isnan(wavs[i]).sum()
              assert invalid_pts <= 0.05*target_no_of_pts, 'The digitized duration was much too short %d out of %d points invalid.' % (invalid_pts, target_no_of_pts)
              
              logging.debug("Resamples channel %s waveform from %d to %d pts." % (channels[i], returned_no_of_pts, len(wavs[i])))

          return [ {'pre': p, "wav": w, 'time_axis': new_time_axis } for p,w in zip(pres, wavs) ]
          
        raise Exception('Giving up...')

    def reset_averaging(self):
      '''
      Clears old averages.
      '''
      assert self.get_acquire_average_mode() == 'ON', 'Averaging is off!'

      avgs = self.get_acquire_average_count()
      self.set_acquire_average_mode('OFF')
      self.set_acquire_average_mode('ON')
      self.set_acquire_average_count(avgs)

    def reset(self):
        '''
        Resets the instrument to default values

        Input:
            None

        Output:
            None
        '''
        logging.info(__name__ + ' : resetting instrument')
        self._visainstrument.write('*RST')
        self.get_all()



    def operation_complete_query(self):
        '''
        The *OPC? query places an ASCII character "1" in the oscilloscope's output
        queue when all pending selected device operations have finished.

        Input:
            None

        Output:
            None
        '''
        logging.debug(__name__ + ' : asking operation complete status.')
        return self._visainstrument.ask('*OPC?')



    def trigger_event_register(self):
        '''
        
        The :TER? query reads the Trigger Event Register. A "1" is returned if a
        trigger has occurred. A "0" is returned if a trigger has not occurred. The
        autotrigger does not set this register. The register is set to a value of 1
        only when the waveform meets the trigger criteria.
        
        Input:
            None

        Output:
            None
        '''
        logging.debug(__name__ + ' : trigger event register query.')
        return self._visainstrument.ask('TER?')


    def clear(self):
        '''
        The *CLS command clears all status and error registers.

        Input:
            None

        Output:
            None
        '''
        logging.info(__name__ + ' : clearing status and event registers')
        self._visainstrument.write('*CLS')
        self.get_all()




    def autoscale(self):
        '''
        Autoscale all axis.

        Input:
            None

        Output:
            None
        '''
        logging.debug(__name__ + ' : autoscalubg all axis')
        self._visainstrument.write(':AUTOSCALE')


    def autoscale_vertical(self, channel):
        '''
        Autoscale the vertical axis for the given channel.

        Input:
            channel (string): CHAN1, CHAN2, CHAN3, CHAN4

        Output:
            None
        '''
        logging.info(__name__ + ' : autoscaling channel %s vertical axis' % channel)
        self._visainstrument.write(':AUTOSCALE:VERTICAL %s' % channel)



# a few acquisition functions 

    def get_waveform_as_words(self):
        '''
        Return the waveform data in binary.
        '''
        #self._visainstrument.write(':WAVEFORM:FORMAT WORD')
        assert self._visainstrument.ask(':WAVEFORM:FORMAT?').upper().strip() == 'WORD', 'Wrong data format'
        return self._visainstrument.ask(':WAVEFORM:DATA?')

    def get_waveform_as_ascii(self):
        '''
        Return the waveform data, assuming that the .
        '''
        #print 'get waveform as ascii...'
        #self._visainstrument.write(':WAVEFORM:FORMAT ASC')
        assert self._visainstrument.ask(':WAVEFORM:FORMAT?').upper().strip().startswith('ASC'), 'Wrong data format'

        values = self._visainstrument.ask(':WAVEFORM:DATA?').split(',')
        assert values[-1] == '', 'The last element should be empty.'
        return np.array(map(float, values[:-1]))

    def digitize(self, channel=None):
        '''
        Initialize the selected channels or functions and acquire according to the current settings.

        Input:
            selected channel (string): CHAN1, CHAN2, CHAN3, CHAN4, DIG#, COMM1/2, DIFF1/2, leave blank '', or see p.683.

        Output:
            None
        '''
        logging.debug(__name__ + ' : capturing data with Digitize on channel %s.' % (str(channel)))
        self._visainstrument.write(':DIGITIZE %s' % channel if channel != None else ':DIG')


    def set_display(self,val,channel):
        '''
        You can specify the display state of a channel.

        Input:
            val (int): Turn ON (1) or OFF (0) the display for the given channel
            channel (int): Specify the channel [1,2,3,4].
           
        Output:
            None
        '''
        logging.info(__name__ + ' : set display state of ch%u to %u.' % (channel,val)  )
        self._visainstrument.write(':CHAN%u:DISP %s' % (channel,val))



    def do_get_waveform_preamble(self):
        '''
        Get the waveform preamble.

        Input:
            None
            
        Output:
            The waveform preamble parsed into a dict. Note that all values are left as strings.
        '''
        params = self._visainstrument.ask(':WAV:PRE?').split(',')
        assert len(params) == 24, "There is no waveform. Number of params in preamble (%d) should be 24!" % len(params)

        # the order of the parameters is specified in the Programmer's Reference, p. 1020
        # params1 =     {'format' : params[0],
        #               'type' : params[1],
        #                   'points': params[2],
        #                   'count': params[3],
        #                   'x_increment' : params[4],
        #                   'x_origin' : params[5],
        #                   'x_reference' : params[6],
        #                   'y_increment' : params[7],
        #                   'y_origin' : params[8],
        #                   'y_reference' : params[9],
        #                   'coupling' : params[10],
        #                   'x_display_range' : params[11],
        #                   'x_display_origin' : params[12], 
        #                   'y_display_range' : params[13],
        #                   'y_display_origin' : params[14],
        #                   'date' : params[15],
        #                   'time' : params[16],
        #                   'frame_model_no' : params[17],
        #                   'aquisition_mode' : params[18],
        #                   'completion' : params[19],
        #                   'x_units' : params[20],
        #                   'y_units' : params[21],
        #                   'max_bandwidth_limit' : params[22],
        #                   'min_bandwidth_limit' : params[23]}


        # the order of the parameters is specified in the Programmer's Reference, p. 1020
        params = dict(zip(['format',
                          'type',
                          'points',
                          'count',
                          'x_increment',
                          'x_origin',
                          'x_reference',
                          'y_increment',
                          'y_origin',
                          'y_reference',
                          'coupling',
                          'x_display_range',
                          'x_display_origin', 
                          'y_display_range',
                          'y_display_origin',
                          'date',
                          'time',
                          'frame_model_no',
                          'aquisition_mode',
                          'completion',
                          'x_units',
                          'y_units',
                          'max_bandwidth_limit',
                          'min_bandwidth_limit' ], params))


        return params


    def get_waveform_count(self): 
        '''
        Returns the fewest number of hits in all of the time buckets
        for the currently selected waveform.
        
        For the AVERage waveform type, the count value is the fewest
        number of hits for all time buckets. This value may be less
        than or equal to the value specified with the
        :ACQuire:AVERage:COUNt command.

        Input:
            none
        Output:
            completion % before measuring (float): see p. 147 of the programming manual.
        '''
        outp = self._visainstrument.ask(':WAVEFORM:COUNT?')
        logging.debug(__name__ + ' : Getting waveform count (number of averages completed): %s' % outp)
        return(int(outp))




    def get_completion_criterion(self): 
        '''
        Get the completion criteria (p.147)

        Input:
            none
        Output:
            completion % before measuring (float): see p. 147 of the programming manual.
        '''
        logging.debug(__name__ + ' : Getting acquisition sampling mode.')
        outp = self._visainstrument.ask(':ACQUIRE:COMPLETE?')
        return(outp)


    def set_completion_criterion(self, val=90):
        '''
        Input:
            val (float) : completion % before measuring.  Default is 90.

        Output:
            True or false.
        '''
        logging.debug(__name__ + ' : acquisition sampling mode %s' % val)
        self._visainstrument.write(':ACQUIRE:COMPLETE %u' % val)
###



    def get_waveform(self):
        '''
        Acquires the waveform (y points) on the screen. Use get_time_axis(preamble) to get the time points.

        Input:
            none
        Output:
            preamble, waveform (1D vector)
        '''
        pre = self.get_waveform_preamble()
        dat = self.get_waveform_as_ascii()
        
        return pre, dat
        


    def get_time_axis(self, preamble=None, waveform=None):
        '''
        Return a list of the time values from the waveform preamble data.
        If no preamble is provided, a new one is acquired from the scope.
        '''
        pre = preamble if preamble != None else self.get_waveform_preamble()

        # generate the time points from the information in the preamble
        xorg = float(pre['x_origin'])
        xinc = float(pre['x_increment'])
        points = len(waveform) if waveform != None else int(pre['points'])

        return np.array([(xorg + i*xinc) for i in range(points)])


    def __words_to_waveform(self, bytes, preamble=None):
        '''
        Converts an array of bytes in WORD format (2 bytes per number, MSB first)
        queried from the MSO9404A into a list of real numbers.
        '''
        assert bytes[0] == "#", "The first character must be a hash!"
        offset = 2 + int(bytes[1])
        bytecount = int(bytes[2:offset])
        fmt = '>%uh' % (bytecount/2)
        data = bytes[offset:]
        
        if bytecount%2 != 0 or bytecount != len(data):
          msg = 'WARN: wrong byte count (%u)! fmt: %s  len(data) = %u' % (bytecount, fmt, len(data))
          raise Exception(msg)

        # convert the bytes into unsigned integers
        integer_data = np.array(struct.unpack(fmt, data))

        # convert the integers into floats in the plus (exclusive) / minus (inclusive) one range
        # float_data = integer_data / float(2**15)

        # convert the integers to voltages using parameters from the preamble
        pre = preamble if preamble != None else self.get_waveform_preamble()  
            
        yinc = np.float(pre['y_increment'])
        yorg = np.float(pre['y_origin'])
        voltages = np.array([(yorg + integer_data[i]*yinc) for i in range(0,len(integer_data))])
        
        # This magic value indicates that the scope has not
        # acquired enough data to assign a value for this point
        voltages[integer_data == 31232] = np.NaN


        # This magic value indicates that the scope has not
        # acquired enough data to assign a value for this point
        #float_data[integer_data == 31232] = np.NaN

        # if preamble != None:
        
        # Convert into physical units (scale, offset) if preamble was provided
        return voltages

    def restart_averaging(self):
      # reset the number of averages
      self.set_acquire_average_mode('OFF')
      self.set_acquire_average_mode('ON')


    def setup_edge_trigger(self,
                                 trigger_source_channel=4,
                                 trigger_level=0.2,
                                 trigger_on_positive_slope=True):
        #### fix edge trigger source issue!  #########

        '''
           Wait for a single edge trigger event on channel 2, then do a single aquisition.
           You can override the default trigger options using the parameters. If you set
           them to None, the current parameters are used.
           
         Input:
             trigger_level(float): Positive slope edge trigger level to set on channel 2.
         Output:
             none.
         '''
#       assert (trigger_source_channel in [1, 2, 3, 4]), 'Invalid trigger source channel!'
        self.set_trigger_sweep_mode('TRIG')
        #self.set_trigger_sweep_mode('SING')

        if trigger_level != None:
          getattr(self, 'set_ch%u_trigger_level' % trigger_source_channel)(trigger_level)

        self.set_trigger_mode('EDGE')

        if trigger_source_channel != None:
          self.set_edge_trigger_source('CHAN%u' % trigger_source_channel)
        
        self.set_edge_trigger_coupling('DC')

        if trigger_on_positive_slope != None:
          self.set_edge_trigger_slope('POS')
        else:
          self.set_edge_trigger_slope('NEG')
        
        #self.run()

## begin the parameter functions

    def do_get_timebase_ref_clock_mode(self): 
        '''
        Query whether averaging mode is ON or OFF.

        Input:
            none
        Output:
            status(string): ON or OFF.
        '''
        logging.debug(__name__ + ' Getting status of external reference clock: ')
        outp = self._visainstrument.ask(':TIMEBASE:REFCLOCK?')
        assert (outp in ['0', '1']), 'Did not understand "%s"' % outp
        return 'EXT' if outp=='1' else 'INT'
    
    def do_set_timebase_ref_clock_mode(self, val):
        '''

        Input:
            val : external reference clock EXT or INT.

        Output:
            True or false.
        '''
        logging.debug(__name__ + ' : Setting status of external reference clock:  %s' % val)
        self._visainstrument.write(':TIMEBASE:REFCLOCK %u' % int(val=='EXT'))

    
    def do_get_acquire_average_mode(self): 
        '''
        Query whether averaging mode is ON or OFF.

        Input:
            none
        Output:
            status(string): ON or OFF.
        '''
        logging.debug(__name__ + ' : Getting acquisition sampling mode.')
        outp = self._visainstrument.ask(':ACQUIRE:AVERAGE?')
        if outp=='1':
            return('ON')
        elif outp=='0':
            return('OFF')
        else:
            return(np.NaN)
        

    def do_set_acquire_average_mode(self, val):
        '''

        Input:
            val (string) : averaging mode ON(1)  or OFF(0).

        Output:
            True or false.
        '''
        logging.debug(__name__ + ' : acquisition sampling mode %s' % val)
        self._visainstrument.write(':ACQUIRE:AVERAGE %s' % val)



    def do_get_acquire_average_count(self): 
        '''
        Query the averaging count.

        Input:
            none
        Output:
            count(int): number of averages per time bucket (p.145)
        '''
        logging.debug(__name__ + ' : Getting acquisition sampling mode.')
        outp = self._visainstrument.ask(':ACQUIRE:AVERAGE:COUNT?')
        return(outp)


    def do_set_acquire_average_count(self, val):
        '''

        Input:
            val (int) : averaging count.

        Output:
            True or false.
        '''
        logging.debug(__name__ + ' : acquisition average count %u' % val)
        self._visainstrument.write(':ACQUIRE:AVERAGE:COUNT %u' % val)



    def do_get_acquire_points(self): 
        '''
        Query the averaging count.

        Input:
            none
        Output:
            count(int): number of averages per time bucket (p.145)
        '''
        logging.debug(__name__ + ' : Getting number of acquired points.')
        outp = self._visainstrument.ask(':ACQUIRE:POINTS?')
        return(int(outp))


    def do_set_acquire_points(self, val='AUTO'):
        '''

        Input:
            val (string) : number of points to acquire or 'AUTO'

        Output:
            True or false.
        '''
        logging.debug(__name__ + ' : number of acquired points set to %s' % val)
        self._visainstrument.write(':ACQUIRE:SRATE:AUTO 1')
        
        if val == 'AUTO':
          self._visainstrument.write(':ACQUIRE:POINTS:AUTO 1')
        else:
          self._visainstrument.write(':ACQUIRE:POINTS:AUTO 0')
          self._visainstrument.write(':ACQUIRE:POINTS %s' % val)

    def do_get_acquire_mode(self): 
        '''
        Set the acquire mode (p.151)

        Input:
            none
        Output:
            acquisition sampling mode(string): Current acquisition sampling mode. See context help.
        '''
        logging.debug(__name__ + ' : Getting acquisition sampling mode.')
        outp = self._visainstrument.ask(':ACQUIRE:MODE?')
        return(outp)


    def do_set_acquire_mode(self, val):
        '''

        Input:
            val (string) : Acquisition sampling mode.  Choose from string options in context help.

        Output:
            True or false.
        '''
        logging.debug(__name__ + ' : acquisition sampling mode %s' % val)
        self._visainstrument.write(':ACQUIRE:MODE %s' % val)



    def do_get_waveform_source(self): 
        '''
        This command queries which channel, function, waveform memory or histogram is used as the waveform source (p.1028).

        Input:
            none
        Output:
            waveform source (string).
        '''
        logging.debug(__name__ + ' : Getting the waveform source.')
        outp = self._visainstrument.ask(':WAVEFORM:SOURCE?')
        return(outp) 


    def do_set_waveform_source(self, val):
        '''

        Input:
            val (string) : The waveform source. See context help and p.1028 for more options.  

        Output:
            True or false.
        '''
        logging.debug(__name__ + ' : set wavform source for digitize to %s' % val)
        self._visainstrument.write(':WAVEFORM:SOURCE %s' % val)

####




    def do_get_timebase_position(self): 
        '''
        This command sets the time interval between the trigger event
        and the delay reference point.  
        Input: none 
        Output: Timebase position (s).
        '''
        logging.debug(__name__ + ' : getting the timebase range')
        p = self._visainstrument.ask(':TIMEBASE:POSITION?')
        try:
            return float(p)
        except ValueError:
            print "Could not convert {0} to float.".format(p)
            return float('nan')


    def do_set_timebase_position(self, val):
        '''

        Input:
            len (float) : Timebase position (s)

        Output:
            True or false.
        '''
        logging.debug(__name__ + ' : set timebase range to %f' % val)
        self._visainstrument.write(':TIMEBASE:POSITION %s' % val)



    def do_get_timebase_range(self): 
        '''
        This command sets full-scale horizontal time in seconds.  The range is 10 times the time-per-division value.
        Input:
        none
        Output:
        Full-scale horizontal range (s).
        '''
        logging.debug(__name__ + ' : getting the timebase range')
        p = self._visainstrument.ask(':TIMEBASE:RANGE?')
        try:
            return float(p)
        except ValueError:
            print "Could not convert {0} to float.".format(p)
            return float('nan')


    def do_set_timebase_range(self, val):
        '''

        Input:
            len (int) : full-scale horizontal range (s)

        Output:
            True or false.
        '''
        logging.debug(__name__ + ' : set timebase range to %f' % val)
        self._visainstrument.write(':TIMEBASE:RANGE %s' % val)






    def do_get_timebase_scale(self): 
        '''
        This command sets full-scale horizontal time in seconds.  The range is 10 times the time-per-division value.

        Input:
            none
        Output:
            Horizontal cale value displayed as time/div on the scope screen in units of div/s.
        '''
        logging.debug(__name__ + ' : getting the timebase range')
        p = self._visainstrument.ask(':TIMEBASE:SCALE?')
        try:
            return float(p)
        except ValueError:
            print "Could not convert {0} to float.".format(p)
            return float('nan')


    def do_set_timebase_scale(self, val):
        '''

        Input:
            val (float) : timescale in (s/div)

        Output:
            True or false.
        '''
        logging.debug(__name__ + ' : set timebase scale to %f' % val)
        self._visainstrument.write(':TIMEBASE:SCALE %s' % val)



    def do_get_timebase_delay(self): 
        '''
        This command sets full-scale horizontal time in seconds.  The range is 10 times the time-per-division value.

        Input:
            none
        Output:
            Real number for the time in seconds from the trigger event to the delay reference point.
        '''
        logging.debug(__name__ + ' : getting the timebase delay')
        p = self._visainstrument.ask(':TIMEBASE:WINDOW:DELAY?')
        try:
            return float(p)
        except ValueError:
            print "Could not convert {0} to float.".format(p)
            return float('nan')



    def do_set_timebase_delay(self, val):
        '''

        Input:
            val (float) : Real number for the time in seconds from the trigger event to the delay reference point.

        Output:
            True or false.
        '''
        logging.debug(__name__ + ' : set timebase scale to %f' % val)
        self._visainstrument.write(':TIMEBASE:WINDOW:DELAY %s' % val)


    def do_get_timebase_reference(self): 
        '''
        This command sets full-scale horizontal time in seconds.  The range is 10 times the time-per-division value.

        Input:
            none
        Output:
            The current delay reference position (left, center or right of the display).
        '''
        logging.debug(__name__ + ' : Getting the timebase reference position.')
        outp = self._visainstrument.ask(':TIMEBASE:REFERENCE?')
        if (outp=='LEFT'):
            return 'left'
        elif (outp=='CENT'):
            return 'center'
        elif (outp=='RIGH'):
            return 'right'
        else: 
            return 'an error occurred while reading status from instrument'


    def do_set_timebase_reference(self, val):
        '''

        Input:
            val (string) : delay reference to the LEFT, CENTer or RIGHt of the display.

        Output:
            True or false.
        '''
        logging.debug(__name__ + ' : set timebase scale to %s' % val)
        self._visainstrument.write(':TIMEBASE:REFERENCE %s' % val)



###  begin channel functions

    def do_get_vertical_range(self,channel):
        '''
        Input:
            none

        Output:
            Full scale vertical range in volts.
        '''
        p = self._visainstrument.ask(':CHAN%u:RANGE?' % channel)
        try:
            return float(p)
        except ValueError:
            print "Could not convert {0} to float.".format(p)
            return float('nan')


    def do_set_vertical_range(self,val, channel):
        '''
        Input:
            val (string) : full scale vertical axis (V)
            channel (int): channel to set

        Output:
            True or false.
        '''
        logging.debug(__name__ + ' : set full vertical range to %s' % val)
        self._visainstrument.write(':CHAN%u:RANG %s' % (channel,val))

    def do_get_vertical_offset(self,channel):
        '''
        Input:
            none

        Output:
            Vertical value that is represented at the center of the display for the selected channel.
        '''
        p = self._visainstrument.ask(':CHAN%u:OFFS?' % channel)
        try:
            return float(p)
        except ValueError:
            print "Could not convert {0} to float.".format(p)
            return float('nan')


    def do_set_vertical_offset(self, val, channel):
        '''
        Input:
            val (string) : Vertical value that is represented at the center of the display for the selected channel in the current units.
            channel (int): channel to set

        Output:
            True or false.
        '''
        logging.debug(__name__ + ' : set vertical offset to %s' % val)
        self._visainstrument.write(':CHAN%u:OFFS %s' % (channel,val))



    def do_get_vertical_scale(self,channel):
        '''
        Input:
            none

        Output (float):
            Units per division for vertical scale (units/div)
        '''
        p = self._visainstrument.ask(':CHAN%u:SCAL?' % channel)
        try:
            return float(p)
        except ValueError:
            print "Could not convert {0} to float.".format(p)
            return float('nan')


    def do_set_vertical_scale(self,val, channel):
        '''

        Input:
            val (string) : Units per division for vertical scale.
            channel (int): channel to set

        Output:
            True or false.
        '''
        logging.debug(__name__ + ' : set timebase scale to %s' % val)
        self._visainstrument.write(':CHAN%u:SCAL %s' % (channel,val))




    def do_get_vertical_scale(self,channel):
        '''
        Input:
            none

        Output (float):
            Units per division for vertical scale (units/div)
        '''
        p = self._visainstrument.ask(':CHAN%u:SCAL?' % channel)
        try:
            return float(p)
        except ValueError:
            print "Could not convert {0} to float.".format(p)
            return float('nan')


    def do_set_vertical_scale(self,val, channel):
        '''

        Input:
            val (string) : Units per division for vertical scale.
            channel (int): channel to set

        Output:
            True or false.
        '''
        logging.debug(__name__ + ' : set timebase scale to %s' % val)
        self._visainstrument.write(':CHAN%u:SCAL %s' % (channel,val))




    def do_get_vertical_units(self,channel):
        '''
        Input:
            none

        Output (float):
            Units per division for vertical scale (units/div)
        '''
        p = self._visainstrument.ask(':CHAN%u:UNIT?' % channel)
        try:
            return np.str(p)
        except ValueError:
            print "Could not convert {0} to float.".format(p)
            return float('nan')


    def do_set_vertical_units(self,val,channel):
        '''

        Input:
            val (string) : Units per division for vertical scale.
            channel (int): channel to set

        Output:
            True or false.
        '''
        logging.debug(__name__ + ' : set timebase scale to %s' % val)
        self._visainstrument.write(':CHAN%u:UNIT %s' % (channel,val))


    def do_get_vertical_label(self,channel):
        '''
        Input:
            none

        Output (float):
            Label for the selected channel.
        '''
        p = self._visainstrument.ask(':CHAN%u:LAB?' % channel)
        try:
            return np.str(p)
        except ValueError:
            print "Could not convert {0} to float.".format(p)
            return float('nan')


    def do_set_vertical_label(self,val,channel):
        '''

        Input:
            val (string) : channel label 
            channel (int): channel to set

        Output:
            True or false.
        '''
        logging.debug(__name__ + ' : set timebase scale to %s' % val)
        self._visainstrument.write(':CHAN%u:LAB "%s"' % (channel,val))


    def do_get_input_coupling(self,channel):
        '''
        Input:
            none

        Output (string):
            Coupling, impedance, and LF/HF reject.
        '''
        p = self._visainstrument.ask(':CHAN%u:INP?' % channel)
        try:
            return np.str(p)
        except ValueError:
            print "Could not convert {0} to string.".format(p)
            return float('nan')



    def do_set_input_coupling(self,val,channel):
        '''

        Input:
            val (string) : Input coupling mode: DC, DC50, AC, LFR1/LFR2, 
            channel (int): channel to set

        Output:
            True or false.
        '''
        logging.debug(__name__ + ' : set timebase scale to %s' % val)
        self._visainstrument.write(':CHAN%u:INP %s' % (channel,val))



####  'triggering' functions
    def do_get_trigger_mode(self):
        '''
        Input: None
        Output:
               Trigger mode (string)

        '''
        p = self._visainstrument.ask(':TRIGGER:MODE?')
        try:
            return np.str(p)
        except ValueError:
            print "Could not convert {0} to string.".format(p)
            return float('nan')


    def do_set_trigger_mode(self,val):
        '''
        Input:
            val (string) : Choose from list in the context help.
            channel (int): Channel to set.

        Output:
            True or false.
        '''
        logging.debug(__name__ + ' : set timebase scale to %s' % val)
        self._visainstrument.write(':TRIGGER:MODE %s' % val)


    def do_get_trigger_sweep_mode(self):
        '''
        Input: None
        Output:
               Oscilloscope sweep mode (string)
               AUTO, TRIGgered, SINGle

        '''
        p = self._visainstrument.ask(':TRIGGER:SWEEP?')
        try:
            return np.str(p)
        except ValueError:
            print "Could not convert {0} to string.".format(p)
            return float('nan')


    def do_set_trigger_sweep_mode(self,val):
        '''
        Input:
            val (string) : Choose from list in the context help 
            (AUTO, TRIGgered, SINGle).

        Output:
            True or false.
        '''
        logging.debug(__name__ + ' : set trigger sweep mode to %s' % val)
        self._visainstrument.write(':TRIGGER:SWEEP %s' % val)


    def do_get_trigger_and_source(self,channel):
        '''
        Input: Specified channel (1,4)
        Output:
               Trigger and source logic value (HIGH, LOW, DONTcare).

        '''
        p = self._visainstrument.ask(':TRIGGER:AND:SOURCE? CHAN%u' % channel)
        try:
            return np.str(p)
        except ValueError:
            print "Could not convert {0} to string.".format(p)
            return float('nan')


    def do_set_trigger_and_source(self,val,channel):
        '''
        Input:
            val (string) : Choose from list in the context help 
            (HIGH, LOW, DONTcare).

        Output:
            True or false.
        '''
        logging.debug(__name__ + ' : trigger and source settings for ch%u and logic value: %s' % (channel, val))
        self._visainstrument.write(':TRIGGER:AND:SOURCE CHAN%u,%s' % (channel,val))

    def do_get_trigger_level(self,channel):
        '''
        Input: Specified channel (1,4)
        Output:
               Trigger level in current units.

        '''
        p = self._visainstrument.ask(':TRIGGER:LEVEL? CHAN%u' % channel)
        try:
            return np.str(p)
        except ValueError:
            print "Could not convert {0} to string.".format(p)
            return float('nan')


    def do_set_trigger_level(self,val,channel):
        '''
        Input:
            val (string) : Trigger level for the specifed channel.

        Output:
            True or false.
        '''
        logging.debug(__name__ + ' : trigger and source settings for ch%u and logic value: %s' % (channel, val))
        self._visainstrument.write(':TRIGGER:LEVEL CHAN%u,%s' % (channel,val))

### begin edge trigger functions
    def do_get_edge_trigger_source(self):
        '''
        Input: none.
        Output:
               The current edge trigger source channel.

        '''
        p = self._visainstrument.ask(':TRIGGER:EDGE:SOURCE?')
        try:
            return np.str(p)
        except ValueError:
            print "Could not convert {0} to string.".format(p)
            return float('nan')


    def do_set_edge_trigger_source(self,channel):
        '''
        Input:
            val (string) : Edge trigger souce channel.  See context help.

        Output:
            True or false.
        '''
        logging.debug(__name__ + ' : edge trigger source set to %s' % channel)
        self._visainstrument.write(':TRIGGER:EDGE:SOURCE %s' % channel)



    def do_get_edge_trigger_slope(self):
        '''
        Input: None
        Output:
               Slope of the trigger source (string)
               Positive, Negative, Either

        '''
        p = self._visainstrument.ask(':TRIGGER:EDGE:SLOPE?')
        try:
            return np.str(p)
        except ValueError:
            print "Could not convert {0} to string.".format(p)
            return float('nan')


    def do_set_edge_trigger_slope(self,val):
        '''
        Input:
            val (string) : Choose from list in the context help 
            (POSitive, NEGative, EITHer).

        Output:
            True or false.
        '''
        logging.debug(__name__ + ' : set edge trigger slope %s' % val)
        self._visainstrument.write(':TRIGGER:EDGE:SLOPE %s' % val)



    def do_get_edge_trigger_coupling(self):
        '''
        Input: None
        Output:
               Currently selected coupling for the specified edge trigger source: AC, DC, LFReject or HFReject.              
        '''
        p = self._visainstrument.ask(':TRIGGER:EDGE:COUPLING?')
        try:
            return np.str(p)
        except ValueError:
            print "Could not convert {0} to string.".format(p)
            return float('nan')


    def do_set_edge_trigger_coupling(self,val):
        '''
        Input:
            val (string) : Choose from list in the context help: 
            (AC, DC, LFReject or HFReject)

        Output:
            True or false.
        '''
        logging.debug(__name__ + ' : set edge trigger coupling %s' % val)
        self._visainstrument.write(':TRIGGER:EDGE:COUPLING %s' % val)

###-----------------------------------






    def clear_status(self):
        '''
        Go from the idle to waiting-for-trigger state.

        Input:
            None

        Output:
            None
        '''
        logging.debug(__name__ + ' : INIT')
        self._visainstrument.write('*CLS')



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
        self.get_timebase_range()
        self.get_timebase_scale()
        self.get_timebase_delay()
        self.get_timebase_reference()
        self.get_timebase_ref_clock_mode()
        self.get_trigger_mode()
        self.get_trigger_sweep_mode()
        self.get_edge_trigger_source()
        self.get_edge_trigger_slope()
        self.get_edge_trigger_coupling()
        self.get_acquire_mode()
        self.get_waveform_source()
        self.get_acquire_average_mode()
        self.get_acquire_average_count()
        self.get_acquire_points()

        for i in range(1,5):
            self.get('ch%d_vertical_range' % i)
            self.get('ch%d_vertical_scale' % i)
            self.get('ch%d_vertical_offset' % i)
            self.get('ch%d_vertical_label' % i)
            self.get('ch%d_vertical_units' % i)
            self.get('ch%d_input_coupling' % i)
            self.get('ch%d_trigger_and_source' % i)
            self.get('ch%d_trigger_level' % i)
