# Driver for Rhode & Schwartz VNA ZVCE 1127 8600 50
# Russell Lake <russell.lake@aalto.fi>, 2013
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



#Things to do
#------------------
# 1 set/get paramters

# - number of points
# - frequency range
# - averaging (IF bandwidth, OR sensitivity?)

# 2 get Data
# get frequency data
# get y-axis (i.e. power) data

# 3 Triggering...
# start single scan
# start continuous scans



from instrument import Instrument
import visa
import types
import logging
import numpy as np
import struct



import qt


class RhodeSchwartz_FSEM30(Instrument):
    '''
    This is the driver for the Rohde & Schwarz FSL spectrum analyzer

    Usage:
    Initialize with
    <name> = qt.instruments.create('<name>', 'RhodeSchwartz_ZVCE_1127_8600_50',
        address='TCPIP::<IP-address>::INSTR', 
        reset=<bool>,)

    For GPIB the address is: 'GPIB::<gpib-address>'
    '''
        
    
    def __init__(self, name, address, reset=False):
        '''
        Initializes a R&S FSL, and communicates with the wrapper.

        Input:
            name (string)           : name of the instrument
            address (string)        : GPIB address
            reset (bool)            : resets to default values
        '''
        # Initialize wrapper functions
        logging.info('Initializing instrument Rohde & Schwarz FSL spectrum analyzer')
        Instrument.__init__(self, name, tags=['physical'])
        
        # Add some global constants
        self._address = address
        self._visainstrument = visa.ResourceManager().open_resource(self._address, timeout=10000)
        
        self._freq_unit = 1
        self._freq_unit_symbol = 'Hz'
        
        # self._channel_to_s = { 1: 'S11', 2: 'S21', 3: 'S12', 4: 'S22' }
        # self._s_to_channel = dict([(v,k) for k,v in self._channel_to_s.iteritems()])

        # Add parameters to wrapper

        self.add_parameter('measure_mode', type=types.IntType, format='%g',
                           flags=Instrument.FLAG_GETSET|Instrument.FLAG_GET_AFTER_SET,
                           units='Hz', minval=1, maxval=5)
        self.add_parameter('start_frequency', type=types.FloatType, format='%.6e',
                           flags=Instrument.FLAG_GETSET|Instrument.FLAG_GET_AFTER_SET,
                           units='Hz', minval=0, maxval=27e9)

        self.add_parameter('stop_frequency', type=types.FloatType, format='%.6e',
                           flags=Instrument.FLAG_GETSET|Instrument.FLAG_GET_AFTER_SET,
                           units='Hz', minval=0, maxval=27e9)
              
        self.add_parameter('center_frequency', type=types.FloatType, format='%.6e',
                           flags=Instrument.FLAG_GETSET|Instrument.FLAG_GET_AFTER_SET,
                           units='Hz', minval=0, maxval=27e9)                   

        # self.add_parameter('numpoints', type=types.IntType, format='%g',
                           # flags=Instrument.FLAG_GETSET|Instrument.FLAG_GET_AFTER_SET,
                           # units='', minval=0, maxval=501)
        
        self.add_parameter('span_frequency', type=types.FloatType, format='%.6e',
                           flags=Instrument.FLAG_GETSET|Instrument.FLAG_GET_AFTER_SET,
                           units='Hz', minval=0, maxval=27e9)      

        self.add_parameter('if_bandwidth', type=types.FloatType, format='%.0e',
                           flags=Instrument.FLAG_GETSET|Instrument.FLAG_GET_AFTER_SET,
                           units=self._freq_unit_symbol)
                           
        self.add_parameter('sweeptime', type=types.FloatType, format='%g',
                           flags=Instrument.FLAG_GET,
                           units='s')
                           
        self.add_parameter('averaging', type=types.BooleanType,
                           flags=Instrument.FLAG_GETSET|Instrument.FLAG_GET_AFTER_SET)                   
                           
        self.add_parameter('aver_count', type=types.IntType, format='%g',
                           flags=Instrument.FLAG_GETSET|Instrument.FLAG_GET_AFTER_SET,
                           minval=0, maxval=2000 )
                           
        self.add_parameter('sweeptime_auto', type=types.BooleanType,
                           flags=Instrument.FLAG_GETSET|Instrument.FLAG_GET_AFTER_SET)
                           
        self.add_parameter('source_power', type=types.FloatType,
                           flags=Instrument.FLAG_GETSET|Instrument.FLAG_GET_AFTER_SET,
                           units='dBm')
                           
        self.add_parameter('trigger_source', type=types.StringType,
                          flags=Instrument.FLAG_GETSET|Instrument.FLAG_GET_AFTER_SET,
                          format_map = {
                            "imm" : "immediate (continuous)",
                            "ext" : "external",
                            "line" : "line",
                            "tim" : "periodic timer",
                            "rtcl" : "real time clock",
                            "man" : "manual"
                          })

        self.add_parameter('res_bw', type=types.FloatType, format='%.6e',
                           flags=Instrument.FLAG_GETSET|Instrument.FLAG_GET_AFTER_SET,
                           units='Hz', minval=1, maxval=1e7)
                           
        self.add_parameter('res_bw_auto', type=types.BooleanType,
                           flags=Instrument.FLAG_GETSET|Instrument.FLAG_GET_AFTER_SET)

        self.add_parameter('vid_bw', type=types.FloatType, format='%.6e',
                           flags=Instrument.FLAG_GETSET|Instrument.FLAG_GET_AFTER_SET,
                           units='Hz', minval=1, maxval=1e7)
                           
        self.add_parameter('vid_bw_auto', type=types.BooleanType,
                           flags=Instrument.FLAG_GETSET|Instrument.FLAG_GET_AFTER_SET)                         
                           
        self.add_parameter('sweep_mode', type=types.StringType,
                          flags=Instrument.FLAG_GETSET|Instrument.FLAG_GET_AFTER_SET,
                          format_map = {
                           "single" : "single",
                           "cont" : "continuous"
                          })
                                 
        self.add_parameter('external_reference', type=types.BooleanType,
                          flags=Instrument.FLAG_GETSET|Instrument.FLAG_GET_AFTER_SET)

        self.add_parameter('external_reference_frequency', type=types.IntType,
                          flags=Instrument.FLAG_GETSET|Instrument.FLAG_GET_AFTER_SET,
                          units='MHz')
                          
        self.add_parameter('source_status', type=types.BooleanType,
                          flags=Instrument.FLAG_GETSET|Instrument.FLAG_GET_AFTER_SET)

                          
        
        # Connect to measurement flow to detect start and stop of measurement
        qt.flow.connect('measurement-start', self._measurement_start_cb)
        
        qt.flow.connect('measurement-end', self._measurement_end_cb)
        
        self.add_function('reset')
        self.add_function('send_trigger')
        self.add_function('get_all')
       
        if reset:
            self.reset()
        else:
#            self.set_default_channel_config()
            self.set_default_window_config()
            self.get_all()
        

# --------------------------------------
#           functions
# --------------------------------------

#Our code...
#Things to do
#------------------
# 1 set/get paramters

# - number of points
# - frequency range
# - averaging (IF bandwidth, OR sensitivity?)

# 2 get Data
# get frequency data
# get y-axis (i.e. power) data

# 3 Triggering...
# start single scan
# start continuous scans

    
    def reset(self):
      self._visainstrument.write('*RST') #reset to default settings
#      self.set_default_channel_config()
#      self.set_default_window_config()
#      self.set_sweep_mode('single')
#      self.get_all()


    def send_trigger(self):
      s = self.get_trigger_source()
      if s == 'imm':
        self._visainstrument.write('INIT:IMM')
      elif s == 'man':
        self._visainstrument.write('*TRG')
      else:
        raise Exception('Not sure how to trigger manually when trigger source is set to "%s"' % s)

    def set_default_window_config(self):
      self._visainstrument.write('DISPlay:FORMat QQSPlit')
      for chan in range(1,5):
          self._visainstrument.write('DISP:WIND%u:DIAG CDB' % chan)
          self._visainstrument.write('CALC%u:FORMat MAGNitude' % chan)
      self.autoscale_once()

    def autoscale_once(self):
      for chan in range(1,5):
          self._visainstrument.write('DISPlay:WIND%u:TRAC:Y:SCALe:AUTO ONCE' % chan)

    # def set_default_channel_config(self):
    #   for chan in range(1,5):
    #     self._visainstrument.write('SENSe%d:FUNCtion:ON XFR:POW:%s' % (chan, self._channel_to_s[chan]))
    #     self._visainstrument.write('SENSe%d:FUNCtion:ON XPOW:POW:%s' % (chan, self._channel_to_s[chan]))
    #     self._visainstrument.write('SENSe%d:FUNCtion:ON XTIM:POW:%s' % (chan, self._channel_to_s[chan]))      

    def get_all(self):
        
        self.get_measure_mode()
        self.get_sweep_mode()
        self.get_trigger_source()
        self.get_start_frequency()
        self.get_stop_frequency()
        self.get_if_bandwidth()
        self.get_external_reference()
        self.get_external_reference_frequency()
        self.get_source_power()
        self.get_sweeptime()
        self.get_sweeptime_auto()
        self.get_res_bw()
        self.get_res_bw_auto()
        self.get_vid_bw()
        self.get_vid_bw_auto()
        self.get_center_frequency()
        self.get_aver_count()
        self.get_averaging()
        self.get_source_status()
        
        
        
        
    # def get_data(self, s_parameter):
        # '''
        # Get unformatted measured data from the current channel.

        # s_parameter --- must be on of ['S11', 'S21', 'S12', 'S22']
        # ''' 
        # s = s_parameter.upper().strip()
        # assert s in ['S11', 'S21', 'S12', 'S22'], 'Invalid S-parameter: %s' % s_parameter

        # logging.debug(__name__ + ' : Get trace data.')
        
        # # assert that the channels are configured for the correct s-parameter measurements 
        # r = self._visainstrument.ask('SENSe%u:FUNCtion?' % (self._s_to_channel[s]))
        # assert r.strip().strip("'").upper().endswith(s), 'Channel configuration has been changed! (%s)' % r

        # self._visainstrument.write('FORM REAL,32')

        # raw = self._visainstrument.ask('TRAC? CH%uDATA' % float(self._s_to_channel[s_parameter]))
        # return np.array([ r + 1j*i for r,i in self.__real32_byte_array_to_ndarray(raw).reshape((-1,2)) ])
    
    
      
    
    
    
    def get_data(self,chan):
      self._visainstrument.write('FORM REAL,32')
      return self._visainstrument.query_binary_values('TRAC? TRACE%s' % chan,
                                                datatype='f',
                                                is_big_endian=False,
                                                container=np.array,
                                                header_fmt='ieee')
      



    def get_frequency_data(self,num):
    
      mini=self.get_start_frequency()
      maxi=self.get_stop_frequency()
      points=len(self.get_data(num))
       
      return np.linspace(mini,maxi,points)

     
    def get_transm(self,num):
        '''
        Gives the transmission spectrum
        Use self.set_source_status(1) before!!
        '''
        self._visainstrument.write('TRAC%s' %num)
        self._visainstrument.write('CORR:METH:TRAN')
        
        return self.get_data(num)
        
#       return eval('[' + self._visainstrument.ask('TRAC:STIM? CH1DATA') + ']')

        

    def do_get_measure_mode(self):
        logging.debug('Reading measuring mode')
        return float(self._visainstrument.ask('INST:NSEL?'))
     
    def do_set_measure_mode(self, val): #in Hz
       
       '''
      Mode 1 :  Analyzer
      Mode 2 :  Vector analyzer ADEMode
      Mode 3 :  Vector analyzer DDEMode
      '''
       logging.debug('Setting instrument for %s . mode' % val)
       return self._visainstrument.write('INST:NSEL %E' % val)
    
    def do_get_start_frequency(self): #in Hz
        '''
        Start of sweep (Hz)
        '''
        logging.debug('Reading start frequency')
        return float(self._visainstrument.ask('SENS:FREQ:STAR?'))

    def do_set_start_frequency(self, start): #in Hz
        logging.debug('Setting start freq to %s' % start)
        return self._visainstrument.write('SENS:FREQ:STAR %E' % start)

    def do_get_center_frequency(self): #in Hz
    #     '''
    #     Start of sweep (Hz)
    #     '''
         logging.debug('Reading start frequency')
         return float(self._visainstrument.ask('SENS:FREQ:CENT?'))

    def do_set_center_frequency(self, start): #in Hz
         logging.debug('Setting start freq to %s' % start)
         return self._visainstrument.write('SENS:FREQ:CENT %E' % start)

    def do_get_span_frequency(self): #in Hz
        '''
        get Span
        '''
        logging.debug('Reading span frequency')
        return float(self._visainstrument.ask('SENS:FREQ:SPAN?'))
        
    def do_set_span_frequency(self,span):
        '''
        Set span
        '''
        logging.debug('Reading span frequency')
        return self._visainstrument.write('SENS:FREQ:SPAN %E' % span)

        
        
        
    def do_get_stop_frequency(self): #in Hz
        '''
        End of sweep (Hz)
        '''
        logging.debug('Reading stop frequency')
        return float(self._visainstrument.ask('SENS:FREQ:STOP?'))

    def do_set_stop_frequency(self, stop): #in Hz
        logging.debug('Setting stop freq to %s' % stop)
        return self._visainstrument.write('SENS:FREQ:STOP %E' % stop)

    def do_set_res_bw(self, numb):
        logging.debug('Setting resolution bandwidth to %s' % numb)
        return self._visainstrument.write('SENSe:BAND:RES %E' % numb)
        
    def do_get_res_bw(self):
        logging.debug('Reading resolution bandwidth')
        return float(self._visainstrument.ask('SENSe:BAND:RES?'))
        
    def do_get_res_bw_auto(self):
        logging.debug('reading auto resolution bandwidth')
        r = self._visainstrument.ask('SENS:BAND:AUTO?').lower().strip()
        return r.startswith('1') or r.startswith('on')
    
    def do_set_res_bw_auto(self, val): #in seconds
        logging.debug('Setting auto resolution bandwidth to %s' % val)
        return self._visainstrument.write('SENS:BAND:AUTO %s' % ('ON' if val else 'OFF'))
        
    def do_set_vid_bw(self, numb):
        logging.debug('Setting video bandwidth to %s' % numb)
        return self._visainstrument.write('SENSe:BAND:VID %E' % numb)
        
    def do_get_vid_bw(self):
        logging.debug('Reading video bandwidth')
        return float(self._visainstrument.ask('SENSe:BAND:VID?'))
        
    def do_get_vid_bw_auto(self):
        logging.debug('reading auto video bandwidth')
        r = self._visainstrument.ask('SENS:BAND:VID:AUTO?').lower().strip()
        return r.startswith('1') or r.startswith('on')
    
    def do_set_vid_bw_auto(self, val): #in seconds
        logging.debug('Setting auto Video bandwidth to %s' % val)
        return self._visainstrument.write('SENS:BAND:VID:AUTO %s' % ('ON' if val else 'OFF'))
        
       
    def do_set_averaging(self,val):
        logging.debug('Set averaging to %s' % val)
        return self._visainstrument.write('SENS:AVER %s' % ('ON' if val else 'OFF'))
        
    def do_get_averaging(self):
        logging.debug('Get averaging')
        r = self._visainstrument.ask('SENS:AVER?').lower().strip()
        return r.startswith('1') or r.startswith('on')     
        
    def do_set_aver_count(self,count):
        '''
        Number of averages per sweep.
        '''
        logging.debug('Setting sweep count to %s' % count)
        return self._visainstrument.write('SENS:AVER:COUN %s' % count)
        
    def do_get_aver_count(self):
        logging.debug('Get sweep count')
        return float(self._visainstrument.ask('SENS:AVER:COUN?'))   
     
       
    def do_get_source_status(self): 
        logging.debug('Get status of Output')
        return float(self._visainstrument.ask('OUTP:STAT?'))

    def do_set_source_status(self, val): 
        logging.debug('Setting source status %s' % val)
        return self._visainstrument.write('OUTP:STAT %s' % ('ON' if val else 'OFF'))    
        
       
    #  def do_get_numpoints(self):
    # # #     '''
    # # #     Number of points in frequency
    # ##     '''
    #     logging.debug('Reading sweep points')
    #     return int(self._visainstrument.ask('SENS1:SWE:POIN?'))
    
    # def do_set_numpoints(self,numpoints):
        # print('I am running')
        # logging.debug('Setting sweep points to %f' % numpoints)
        # self._visainstrument.write('SENSE1:SWE:COUN %f' % numpoints)
        #        self.get_sweeptime()

    # def get_flist(self):
    #     self._visainstrument.write('INIT;*WAI')
    #     return eval('[' + self._visainstrument.ask('TRAC:STIM? CH1DATA') + ']')
 
    def do_get_if_bandwidth(self): #in Hz
        logging.debug('Reading resolution bandwidth')
        r = self._visainstrument.ask('BAND?')
        if r.strip().lower().startswith('max'): r = 26000
        return float(r)/self._freq_unit

    def do_set_if_bandwidth(self,if_bandwidth): #in Hz
        '''
        Note that video BW is automatically kept at 3x reolution BW
        It can be change manually on the FSL or using 'BAND:VID %sHz'
        '''
        logging.debug('Setting Resolution BW to %s' % if_bandwidth)
        if np.abs(if_bandwidth - 26e3) > 1:
          self._visainstrument.write('BAND %sHz' % if_bandwidth)
        else:
          self._visainstrument.write('BAND MAX')
        self.get_sweeptime()

    def do_get_sweeptime(self):
        logging.debug('reading sweeptime')
        return float(self._visainstrument.ask('SWE:TIME?'))

    def do_get_sweeptime_auto(self):
        logging.debug('reading sweeptime')
        r = self._visainstrument.ask('SWE:TIME:AUTO?').lower().strip()
        return r.startswith('1') or r.startswith('on')

    def do_set_sweeptime_auto(self, val): #in seconds
        logging.debug('Setting sweeptime auto to %s' % val)
        return self._visainstrument.write('SWE:TIME:AUTO %s' % ('ON' if val else 'OFF'))
    
    def do_get_source_power(self):
        logging.debug('Reading Source power')
        return float(self._visainstrument.ask('SOUR:POW?'))
        

    #def do_set_center_frequency(self, stop): #in Hz
    #    logging.debug('Setting center freq to %s' % stop)
    #    return self._visainstrument.write('FREQ:CENT %s Hz' % stop)

    #def do_set_span_frequency(self, stop): #in Hz
    #    logging.debug('Setting center freq to %s' % stop)
    #    return self._visainstrument.write('FREQ:SPAN %s Hz' % stop)


    def do_set_source_power(self, source_power): #in dBm
        '''
        Can be set to 0,-10,-20,-30 dBm. on 18GHz FSL
        For 3GHz FSL 1 dBm increments between 0 and -20dBm
        Default is -20dBm
        
        Note: calibration should be done at instrument.
        Details such as power offset can also be adjusted at instrument (op manual p. 294)
        '''
        logging.debug('Setting tracking generator power to %s' % source_power)
    #    if self.get_tracking()==False:
     #       print 'Source off since not in tracking mode. Will be at %sdBm.' % source_power
        self._visainstrument.write('SOUR:POW %s dBm' % source_power)
    
    def do_get_sweep_mode(self):
        logging.debug('Getting sweep mode.')
        return 'cont' if bool(int(self._visainstrument.ask('INIT:CONT?'))) else 'single'

    def do_set_sweep_mode(self, val):
        logging.debug('Setting sweep mode: %s' % val)
        self._visainstrument.write('INIT:CONT %d' % (int(val=='cont')))    

    def do_get_external_reference(self):
        logging.debug('Getting ext ref mode.')
        return self._visainstrument.ask('SENS:ROSC:SOUR?').lower().startswith('ext')

    def do_set_external_reference(self, val):
        logging.debug('Setting ext ref: %s' % val)
        self._visainstrument.write('SENS:ROSC:SOUR %s' % (val))

    def do_get_external_reference_frequency(self):
        logging.debug('Getting ext ref freq.')
        if not self.get_external_reference(): return -1
        return self._visainstrument.ask('SENS:ROSC:EXT:FREQ?')

    def do_set_external_reference_frequency(self, val):
        logging.debug('Setting ext ref freq: %s' % val)
        self._visainstrument.write('SENS:ROSC:EXT:FREQ %dMHZ' % (val))

    def do_get_trigger_source(self):
        logging.debug('Getting trigger source.')
        r = self._visainstrument.ask('TRIGGER:SEQUENCE:SOURCE?').lower().strip()
        if r.startswith('lin') or r.startswith('rtc'):
          r = r[:4]
        else:
          r = r[:3]
        return r
    
    def do_set_trigger_source(self, val):
        logging.debug('Setting trigger source: %s' % val)
        self._visainstrument.write('TRIGGER:SEQUENCE:SOURCE %s' % (val))
        
    def do_get_trigger_level(self):
        logging.debug('Getting trigger level')
        return self._visainstrument.ask('TRIG:LEV?')
    
    def do_set_trigger_level(self,level):
        logging.debug('Setting trigger level to %s' % level)
        self._visainstrument.write('TRIG:LEV %s' % (level))
        

# --------------------------------------
#           Internal Routines
# --------------------------------------
#####
    def __real32_byte_array_to_ndarray(self, bytes):
        '''
        Converts an array of bytes (a string) queried from the ZVCE
        in the binary 'REAL,32' format into a list of complex scattering parameters.
        '''
        assert bytes[0] == "#", "The first character must be a hash! (See VNA online help for the data format.)"
        offset = 2 + int(bytes[1])
        bytecount = int(bytes[2:offset])
        
        fmt = '<%uf' % (bytecount / 4)
        data = bytes[offset:]
        
        if bytecount > len(data):
          raise Exception('wrong byte count (%u)! fmt: %s  len(data) = %u' % (bytecount, fmt, len(data)))
        data = data[:bytecount] # Sometimes there is additional crap at the end...

        return np.array(struct.unpack(fmt, data))

    def _measurement_start_cb(self, sender):
        '''
        Things to do at starting of measurement
        '''
        #self.set_trace_continuous(False) #switch to single trace mode
        #self.get_all()

    def _measurement_end_cb(self, sender):
        '''
        Things to do after the measurement
        '''
        #self.set_trace_continuous(True) #turn continuous back on
    
