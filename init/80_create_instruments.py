#pwrsens1 = qt.instruments.create('pwrsens1', 'Agilent_N1913A', address='TCPIP0::10.0.100.101::inst0', reset=False)
#mwsrc1 = qt.instruments.create('mwsrc1', 'Agilent_E8257D', address='TCPIP0::10.0.100.103::inst0', reset=False)
#mwsrc2 = qt.instruments.create('mwsrc2', 'Gigatronics_2550B', address='10.0.100.102:2550', reset=False)
#multimeter1 = qt.instruments.create('multimeter1', 'Agilent_34410A', address='TCPIP0::10.0.100.100::inst0', reset=False)
#multimeter2 = qt.instruments.create('multimeter2', 'Agilent_34410A', address='TCPIP0::10.0.100.104::inst0', reset=False)
#multimeter3 = qt.instruments.create('multimeter3', 'Agilent_34410A', address='TCPIP0::10.0.100.105::inst0', reset=False)
#multimeter6 = qt.instruments.create('multimeter6', 'Agilent_34410A', address='TCPIP0::10.0.100.111::inst0', reset=False)
#smu1 = qt.instruments.create('smu1', 'Keithley_6430', address='GPIB0::11', reset=False)
#lockin1 = qt.instruments.create('lockin1', 'SR830', address='GPIB0::27', reset=False)
#lockin2 = qt.instruments.create('lockin2', 'SR830', address='GPIB0::8', reset=False)
#SIM900 = qt.instruments.create('SIM900', 'SIM900', address='GPIB0::2', reset=False)
#femto1 = qt.instruments.create('femto1', 'Femto_Luci10', address=0, reset=False, device_type='DLPVA-100-B-D')

#import bluefors_log_writer
#logger = lambda quantity, channel, value: bluefors_log_writer.write(quantity, channel, value, address='K:')
#lakeshore1 = qt.instruments.create('lakeshore1', 'Lakeshore_370', address='ASRL9', reset=False, logger=logger)
#bflog = qt.instruments.create('bflog', 'bluefors_log_reader', address=r'K:', reset=False) # Still useful to get the flow, even if Lakeshore is read separately

#cryomech1 = qt.instruments.create('cryomech1', 'Cryomech_CP2800', address='COM1', reset=False, logger=logger, autoupdate_interval=60)
#turbo1 = qt.instruments.create('turbo1', 'Agilent_V750', address='COM3', reset=False, logger=logger, autoupdate_interval=60)

#base_heater_port = 5
#base_heater_series_resistance = 4.0257e3 # measured with multimeter
#heater_ctrl = qt.instruments.create('heater_ctrl', 'heater_controller',
#                                    set_heater_current=(lambda current, sim=SIM900: sim.set_port_voltage(base_heater_port, current*base_heater_series_resistance)),
#                                    get_heater_current=(lambda sim=SIM900: sim.get_port_voltage(base_heater_port)) )

#heater_ctrl = qt.instruments.create('heater_ctrl', 'heater_controller',
#                                    set_heater_current=(lambda current, lakeshore=lakeshore1: lakeshore.set_temperature_control_setpoint(current)),
#                                    get_heater_current=(lambda lakeshore=lakeshore1: lakeshore.get_temperature_control_setpoint()) )


#ips = qt.instruments.create('ips', 'Oxford_Mercury_IPS', address='10.0.100.107:7020')
#afg1 = qt.instruments.create('afg1', 'Tektronix_AFG3252', address='TCPIP0::10.0.100.108::inst0', reset=False)
#scope1 = qt.instruments.create('scope1', 'Agilent_MSO9404A', address='TCPIP0::10.0.100.109::inst0', reset=False)
#vna1 = qt.instruments.create('vna1', 'Agilent_N9928A', address='TCPIP0::10.0.100.110::inst0', reset=False)

###########
#
# Examples:
#
##########

example1 = qt.instruments.create('example1', 'example', address='GPIB::1', reset=True)
#dsgen = qt.instruments.create('dsgen', 'dummy_signal_generator')
#pos = qt.instruments.create('pos', 'dummy_positioner')
#combined = qt.instruments.create('combined', 'virtual_composite')
#combined.add_variable_scaled('magnet', example1, 'chA_output', 0.02, -0.13, units='mT')
#combined.add_variable_combined('waveoffset', [{
#    'instrument': dmm1,
#    'parameter': 'ch2_output',
#    'scale': 1,
#    'offset': 0}, {
#    'instrument': dsgen,
#    'parameter': 'wave',
#    'scale': 0.5,
#    'offset': 0
#    }], format='%.04f')
