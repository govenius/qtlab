# Driver for Rhode & Schwartz VNA ZNB40
# Russell Lake <russell.lake@aalto.fi>, 2014
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
import struct


import qt


class RhodeSchwartz_ZNB40(Instrument):
    '''
    This is the driver for the Rohde & Schwarz FSL spectrum analyzer.

    Note that, for simplicity, the set functions set the specified
    parameter for all channels.
    The get functions return the value for the currently active channel.

    Usage:
    Initialize with
    <name> = qt.instruments.create('<name>', 'RhodeSchwartz_ZNB40',
        address='TCPIP::<IP-address>::INSTR',
        reset=<bool>,)

    For GPIB the address is: 'GPIB<interface_nunmber>::<gpib-address>'
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
        self._default_timeout = 120000. # ms
        self._visainstrument = visa.ResourceManager().open_resource(self._address,
                                                                    timeout=self._default_timeout)
        self._freq_unit = 1
        self._freq_unit_symbol = 'Hz'
        

        # Add parameters to wrapper

        self.add_parameter('start_frequency', type=types.FloatType, format='%.6e',
                           flags=Instrument.FLAG_GETSET|Instrument.FLAG_GET_AFTER_SET,
                           units=self._freq_unit_symbol, minval=20e3/self._freq_unit, maxval=40e9/self._freq_unit)
        self.add_parameter('stop_frequency', type=types.FloatType, format='%.6e',
                           flags=Instrument.FLAG_GETSET|Instrument.FLAG_GET_AFTER_SET,
                           units=self._freq_unit_symbol, minval=20e3/self._freq_unit, maxval=40e9/self._freq_unit)

        self.add_parameter('center_frequency', type=types.FloatType, format='%.06e',
                          flags=Instrument.FLAG_GETSET|Instrument.FLAG_GET_AFTER_SET,
                           units=self._freq_unit_symbol, minval=0.020, maxval=40e9)

        self.add_parameter('span', type=types.FloatType, format='%.06e',
                          flags=Instrument.FLAG_GETSET|Instrument.FLAG_GET_AFTER_SET,
                          units=self._freq_unit_symbol, minval=0.020)

        self.add_parameter('numpoints', type=types.IntType, format='%g',
                           flags=Instrument.FLAG_GETSET|Instrument.FLAG_GET_AFTER_SET,
                           units='', minval=1)
        self.add_parameter('average_mode', type=types.StringType,
                           flags=Instrument.FLAG_GETSET|Instrument.FLAG_GET_AFTER_SET,
                           format_map={'AUTO': 'automatic',
                                       'FLAT': 'cumulative average of magnitude and phase',
                                       'RED': 'cumulative average of quadratures',
                                       'MOV': 'simple average of quadratures'})
        self.add_parameter('averages', type=types.IntType,
                           flags=Instrument.FLAG_GETSET|Instrument.FLAG_GET_AFTER_SET,
                           minval=1, maxval=1000)
        self.add_parameter('if_bandwidth', type=types.FloatType, format='%.0e',
                           minval=1., maxval=1e6,
                           flags=Instrument.FLAG_GETSET|Instrument.FLAG_GET_AFTER_SET,
                           units=self._freq_unit_symbol)
        self.add_parameter('if_selectivity', type=types.StringType,
                           flags=Instrument.FLAG_GETSET|Instrument.FLAG_GET_AFTER_SET,
                           format_map={'NORM': 'normal',
                                       'MED': 'medium',
                                       'HIGH': 'high'})
        self.add_parameter('sweeptime', type=types.FloatType, format='%g',
                           flags=Instrument.FLAG_GET,
                           units='s')
        self.add_parameter('sweeptime_auto', type=types.BooleanType,
                           flags=Instrument.FLAG_GETSET|Instrument.FLAG_GET_AFTER_SET)

        self.add_parameter('source_power', type=types.FloatType,
                           flags=Instrument.FLAG_GETSET|Instrument.FLAG_GET_AFTER_SET,
                           units='dBm',minval=-30., maxval=10.)

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

        self.add_function('reset')
        self.add_function('send_trigger')
        self.add_function('get_all')

        # for backwards compatibility with the old parameter name ("span_frequency")
        self.add_function('get_span_frequency')
        self.add_function('set_span_frequency')
        
        if reset:
            self.reset()
        else:
#            self.set_default_window_config()
#            self.set_default_channel_config()
            self.get_all()
        

# --------------------------------------
#           functions
# --------------------------------------

    # for backwards compatibility with the old parameter name "span_frequency"
    def get_span_frequency(self): return self.get_span()
    def set_span_frequency(self, s): self.set_span(s)

    def reset(self):
      self._visainstrument.write('*RST') #reset to default settings
#      self.set_default_channel_config()
#      self.set_default_window_config()
      self.set_sweep_mode('single')
      self.get_all()

    def clear_status_reg(self):
      self._visainstrument.write('*CLS')

    def send_trigger(self):
      s = self.get_trigger_source()
      if s == 'imm':
        self._visainstrument.write('INIT:IMM:ALL')
      elif s == 'man':
        self._visainstrument.write('*TRG')
      else:
        raise Exception('Not sure how to trigger manually when trigger source is set to "%s"' % s)

    def trigger_n_times(self, n, block_until_done=False):
        ''' Trigger exactly n sweeps. Waits until done before returning iff block_until_done==True. '''
        self._visainstrument.write('SENSE:SWEEP:COUNT:ALL %u' % n)
        cmd = 'INIT:IMM:ALL%s' % ('; *OPC?' if block_until_done else '')
        min_wait_time = n*self.get_sweeptime()
        if block_until_done and min_wait_time < 10.:
          r = self._visainstrument.ask(cmd)
          assert r.strip() == '1', r
        else:
          self._visainstrument.write(cmd)
          if block_until_done:
            qt.msleep(min_wait_time - 5.)
            r = self._visainstrument.read()
            assert r.strip() == '1', r

    def clear_averages(self):
        ''' Restart averaging (on all channels). '''
        ch_numbers, ch_names = self.ch_catalog()
        for chan in ch_numbers:
          self._visainstrument.write('SENSE%s:AVER:CLE' % (chan))
    
    def autoscale_once(self):
        ch_numbers, ch_names = self.ch_catalog()
        for chan in ch_numbers:
           self._visainstrument.write('DISPlay:WINDow%u:TRAC%u:Y:SCALe:AUTO ONCE' % (chan,chan))

    def set_S21_only_channel_config(self):
        self._visainstrument.write('*RST')
        self._visainstrument.write('SYSTEM:DISPLAY:UPDATE ON')
        self.set_sweep_mode('single')
        self.autoscale_once()
        self.get_all()

    def set_default_channel_config(self):
        default_channel_to_s = { 1: 'S11', 2: 'S21', 3: 'S12', 4: 'S22' }
        self._visainstrument.write('*RST')
        self.set_sweep_mode('single')
        for chan in range(1,5):
          self._visainstrument.write(':CALCULATE%d:PARAMETER:SDEFINE "Trc%u", "%s"' % (chan,chan, default_channel_to_s[chan]))
          self._visainstrument.write(':DISPLAY:WINDOW%u:STATE ON' % chan)
          self._visainstrument.write(':DISPLAY:WINDOW%u:TRACE%u:FEED "Trc%u"' % (chan, chan, chan))
        self._visainstrument.write('SYSTEM:DISPLAY:UPDATE ON')
        self.get_all()

    def get_all(self):
        self.get_sweep_mode()
        self.get_trigger_source()
        self.get_start_frequency()
        self.get_stop_frequency()
        self.get_numpoints()
        self.get_average_mode()
        self.get_averages()
        self.get_if_bandwidth()
        self.get_if_selectivity()
        self.get_external_reference()
        self.get_external_reference_frequency()
        self.get_source_power()
        self.get_sweeptime()
        self.get_sweeptime_auto()
        self.get_center_frequency()
        self.get_span()
        
    def get_data(self, s_parameter):
        '''
        Get the measured S parameter.
        s_parameter --- must be one of ['S11', 'S21', 'S12', 'S22']
        '''
        s = s_parameter.upper().strip()
        assert s in ['S11', 'S21', 'S12', 'S22'], 'Invalid S-parameter: %s' % s_parameter
        logging.debug(__name__ + ' : Get %s data.' % s)
        
        # check that the requested S parameter is being measured on some channel
        try:
            s2chan = self.s_to_channel_dict()[s_parameter]
        except KeyError:
            logging.warn('%s is not currently being measured.', s_parameter)
            raise

        # Verify the function (again <-- seems unnecessary but doesn't hurt either)
        r = self._visainstrument.ask('SENSe%u:FUNCtion?' % s2chan)
        assert r.strip().strip("'").upper().endswith(s), 'Channel configuration has been changed! (%s)' % r

        self._visainstrument.write('FORM REAL,32')
        raw = self._visainstrument.query_binary_values('TRAC? CH%uDATA' % s2chan,
                                                        datatype='f',
                                                        is_big_endian=False,
                                                        container=np.array,
                                                        header_fmt='ieee')

        return np.array([ r + 1j*i for r,i in raw.reshape((-1,2)) ])

    def channel_to_s_dict(self):
        channel_numbers, channel_names = self.ch_catalog()
        meas = []
        for chan in channel_numbers:
            sparam = self._visainstrument.ask('SENSE%u:FUNCTION?' % chan).strip().strip("'").upper().split(":")[2]
            meas.append(sparam)
        return dict(zip(channel_numbers, meas))

    def s_to_channel_dict(self):
        return dict([(v,k) for k,v in self.channel_to_s_dict().iteritems()]) 
        
    def start_single_sweep(self):
        '''
        Same as restart sweep in manual operation.
        '''
        logging.debug(__name__ + 'start a single sweep')

    def get_function(self,chan):
        r = self._visainstrument.ask('SENSe%u:FUNCtion?' % chan)        
        return r

    def get_frequency_data(self, channel_number=None):
        '''
        Get the current x-axis frequency values.

        If channel_number == None, use the first channel in the channel catalog.
        '''
        logging.debug(__name__ + 'Get the current x-axis values.')

        ch = self.ch_catalog()[0][0] if channel_number == None else channel_number
        assert int(ch) == ch, 'channel_number must be an integer'

        self._visainstrument.write('FORM REAL,32')
        return self._visainstrument.query_binary_values('TRAC:STIM? CH%dDATA' % ch,
                                                        datatype='f',
                                                        is_big_endian=False,
                                                        container=np.array,
                                                        header_fmt='ieee')

#       return eval('[' + self._visainstrument.ask('TRAC:STIM? CH1DATA') + ']')    

    def do_get_start_frequency(self):
        '''
        Start of sweep (Hz)
        '''
        logging.debug('Reading start frequency')
        return float(self._visainstrument.ask('SENS:FREQ:STAR?'))/self._freq_unit

    def do_set_start_frequency(self, start): #in Hz
        logging.debug('Setting start freq to %s' % start)
        ch_numbers, ch_names = self.ch_catalog()
        for chan in ch_numbers:
          self._visainstrument.write('SENSE%s:FREQ:STAR %E' % (chan, start*self._freq_unit))
        self.get_center_frequency()
        self.get_span()


    def do_get_stop_frequency(self):
        '''
        End of sweep (Hz)
        '''
        logging.debug('Reading stop frequency')
        return float(self._visainstrument.ask('SENS:FREQ:STOP?'))/self._freq_unit

    def do_set_stop_frequency(self, stop): #in Hz
        logging.debug('Setting stop freq to %s' % stop)
        ch_numbers, ch_names = self.ch_catalog()
        for chan in ch_numbers:
          self._visainstrument.write('SENSE%s:FREQ:STOP %E' % (chan, stop*self._freq_unit))
        self.get_center_frequency()
        self.get_span()


    def do_get_center_frequency(self):
        '''
        End of sweep (Hz)
        '''
        logging.debug('Reading the center frequency')
        return float(self._visainstrument.ask('SENS:FREQ:CENT?'))/self._freq_unit

    def do_set_center_frequency(self, s): #in Hz
        logging.debug('Setting center freq to %s' % s)
        ch_numbers, ch_names = self.ch_catalog()
        for chan in ch_numbers:
            self._visainstrument.write('SENSE%s:FREQ:CENT %s Hz' % (chan,s))
        self.get_start_frequency()
        self.get_stop_frequency()

    def do_get_span(self):
        '''
        End of sweep (Hz)
        '''
        logging.debug('Reading the span')
        return float(self._visainstrument.ask('SENS:FREQ:SPAN?'))/self._freq_unit

    def do_set_span(self, s):
        logging.debug('Setting span to %s' % s)
        ch_numbers, ch_names = self.ch_catalog()
        for chan in ch_numbers:
            self._visainstrument.write('SENSE%s:FREQ:SPAN %s Hz' % (chan,s))
        self.get_start_frequency()
        self.get_stop_frequency()


    def do_get_numpoints(self):
        '''
        Number of points in frequency
        '''
        logging.debug('Reading sweep points')
        return int(self._visainstrument.ask('SENS1:SWE:POIN?'))

    def do_set_numpoints(self,numpoints):
        logging.debug('Setting sweep points to %f' % numpoints)
        ch_numbers, ch_names = self.ch_catalog()
        for chan in ch_numbers:
            self._visainstrument.write('SENSE%s:SWE:POIN %f' % (chan, numpoints))
        return self._visainstrument.ask('SWE:POIN?')

    def do_get_if_bandwidth(self):
        logging.debug('Reading resolution bandwidth')
        r = self._visainstrument.ask('BAND?')
        if r.strip().lower().startswith('max'): r = 1e6
        return float(r)/self._freq_unit

    def do_get_if_selectivity(self):
        logging.debug('Reading IF filter selectivity')
        r = self._visainstrument.ask('BAND:SEL?')
        return r.strip().upper()
    
    def ch_catalog(self):
        logging.debug('return numbers and names of all channels')
        catalog = self._visainstrument.ask('CONFIGURE:CHANNEL:CATALOG?').strip().strip("'").upper().split(",")
        ch_names = catalog[1::2]
        ch_numbers = map(int, catalog[0::2])
        return ch_numbers, ch_names

    def trace_catalog(self):
        logging.debug('return numbers and names of all channels')
        return self._visainstrument.ask('CONFIGURE:TRACE:CATALOG?').strip().strip("'").upper().split(",")

    def window_catalog(self,wnd,wndtr):
        logging.debug('return numbers and names of all channels')
        return self._visainstrument.ask('DISPLAY:WINDOW%u:TRACE%u:CATALOG?' % (wnd,wndtr))

    def list_traces_in_chan(self,chan):
        logging.debug('return numbers and names of all channels')
        return self._visainstrument.ask('CONFIGURE%u:TRACE:CATALOG?' % chan)

    def paramter_select_query(self,chan):
        logging.debug('return numbers and names of all channels')
        return self._visainstrument.ask(':CALCULATE%u:PARAMETER:SELECT?' % chan)


    def do_set_if_bandwidth(self,if_bandwidth):
        logging.debug('Setting Resolution BW to %s' % if_bandwidth)
        ch_numbers, ch_names = self.ch_catalog()
        for chan in ch_numbers:
          self._visainstrument.write('SENSE%s:BWIDTH:RESOLUTION %s' % (chan,if_bandwidth))
        self.get_sweeptime()

    def do_set_if_selectivity(self, if_sel):
        logging.debug('Setting IF filter selectivity to %s' % if_sel)
        ch_numbers, ch_names = self.ch_catalog()
        for chan in ch_numbers:
            self._visainstrument.write('SENSE%s:BWIDTH:SEL %s' % (chan, if_sel))
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
        self._visainstrument.write('SWE:TIME:AUTO %s' % ('ON' if val else 'OFF'))

    def do_get_source_power(self):
        logging.debug('Reading Source power')
        return float(self._visainstrument.ask('SOUR:POW?'))

    def do_get_average_mode(self):
        logging.debug(__name__ + ' : get averaging mode')
        m = self._visainstrument.ask('AVER:MODE?').strip().upper()
        if m.startswith('AUTO'): m = 'AUTO'
        elif m.startswith('FLAT'): m = 'FLAT'
        elif m.startswith('RED'): m = 'RED'
        elif m.startswith('MOV'): m = 'MOV'
        else: raise Exception('unknown averaging mode: %s' % m)
        return m

    def do_set_average_mode(self, m):
        logging.debug(__name__ + ' : set averaging mode to %s' % m)
        if m.startswith('AUTO'): m = 'AUTO'
        elif m.startswith('FLAT'): m = 'FLAT'
        elif m.startswith('RED'): m = 'RED'
        elif m.startswith('MOV'): m = 'MOV'
        ch_numbers, ch_names = self.ch_catalog()
        for chan in ch_numbers:
          self._visainstrument.write('SENSE%s:AVER:MODE %s' % (chan, m))

    def do_get_averages(self):
        '''
        Number of sweeps to average.
        '''
        logging.debug('Reading number of averages')
        return int(self._visainstrument.ask('AVER:COUN?'))

    def do_set_averages(self, averages):
        logging.debug('Setting number of averages to %s' % averages)
        ch_numbers, ch_names = self.ch_catalog()
        for chan in ch_numbers:
          self._visainstrument.write('SENSE%s:AVER:COUN %s' % (chan, averages))
          self._visainstrument.write('SENSE%s:AVER:STAT %s' % (chan, (1 if averages > 1 else 0)))

    def do_set_source_power(self, source_power):
        '''
        Set output power in dBm.
        '''
        logging.debug('Setting generator power to %s' % source_power)
        ch_numbers, ch_names = self.ch_catalog()
        for chan in ch_numbers:
            self._visainstrument.write('SOUR%s:POW %s dBm' % (chan, source_power))
    
    def do_get_sweep_mode(self):
        logging.debug('Getting sweep mode.')
        return 'cont' if self._visainstrument.ask('INIT:CONT?').strip().lower() in ['1'] else 'single'

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
