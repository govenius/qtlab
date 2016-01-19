#!/usr/bin/python


################################################################################
# Initialization stuff
################################################################################
import sys
import os
import numpy as np
import itertools
import time
import logging

## Override some path variables
## This allows the script to be run "standalone" by simply executing it
## from the shell (i.e. not from within qtlab).

qt_path = '../../qtlab' # Set this if you want to run the script from bash
#qt_path = None # None is fine if you're running the script from the qtlab shell


if qt_path != None and os.path.join(qt_path, 'source') not in sys.path: sys.path.insert(0, os.path.join(qt_path, 'source'))

try:
  import qt
  import plot
except Exception as e:
  logging.exception('Is "%s" the correct path to the main qtlab directory?' % qt_path)
  raise e

if qt_path != None:
  qt.config.set('execdir', qt_path)
  qt.config.set('tempdir', os.path.join(qt_path, 'tmp'))
  qt.config.set('gnuplot_terminal', 'wxt enhanced size 1200,650')
  reload(sys.modules['instruments'])

import analysis.dataview as dw

## Override log level
log_level = logging.INFO
logging.getLogger().level = log_level
#for h in logging.getLogger().handlers: # force stderr log level
#  if isinstance(h, logging.StreamHandler): h.level = log_level

## print current config
#logging.info('qtlab.cfg:\n' + '\n'.join([ str(i) for i in qt.config.get_all().items() ]))


################################################################################
# Load data
################################################################################


# Make up some fake data, typically you would initialize
# the Data objects from files instead using the following syntax:
# dd = qt.Data(r'D:/data/20141024/measured_data_dir') # If you want to load real data
data_objects = []
t = np.linspace(-10.,10.,11)

for i in range(2): # how many qt.Data objects to create
  dd = qt.Data(name='fake_data_%d' % i, inmem=True, infile=False)
  dd.add_coordinate('t', units='s')
  dd.add_value('y', units='V')
  #dd.create_tempfile()

  for j in range(2): # how many y vs t "sweeps" to insert in each data file
    time.sleep(.1)
    # Record some values that are the same for every point within a sweep
    dd.add_comment("sweep_start_time = %.10e s" % (time.time()))
    dd.add_comment("heater_current = %.10e A" % (i*1e-3 + j*1e-4))
    # Now add the y vs t data itself:
    y = (t+5*i+.5*j)**2 + np.random.randn(len(t))
    dd.add_data_point(t, y)

  #dd.rewrite_tempfile()
  logging.info((dd.get_name(), dd.get_filename().strip('.dat')))
  data_objects.append(dd)

# Now create a single DataView object out of the files.
# This essentially concatenates the columns and comments from
# each data object.
d = dw.DataView(data_objects)

# You can now access the concatenated columns as:
logging.info("d['t'][::4] = " + str(d['t'][::4]))

##############################################################################
# Define virtual dimensions 
##############################################################################

# you can parse comments in the data files
d.add_virtual_dimension('sweep_start_time', comment_regex='(?i)sweep_start_time\s*=\s*([e\d\.\+\-]*)')

# you can also specify an explicit data type
d.add_virtual_dimension('heater_current', comment_regex=('(?i)heater_current\s*=\s*([e\d\.\+\-]*)', np.float))

# or define an arbitrary function that takes the DataView object as an argument
d.add_virtual_dimension('absolute_time', lambda d: (d['sweep_start_time'] + d['t']),
                        cache_fn_values=True) # whether the function is evaluated immediately or only on demand

# or parse values from the instrument settings files.
# (But the files don't exist for these fake data sets.)
#d.add_virtual_dimension('lockin_amp', from_set=('lockin1', 'amplitude'))
#d.add_virtual_dimension('lockin_tau', from_set=('lockin1', 'tau', np.int))

# You can now access the virtual dimensions just like the real data columns:
logging.info("d['heater_current'][::11] = " + str(d['heater_current'][::11]))

logging.info("All dimensions: " + str(d.get_dimensions()))

##############################################################################
# Hide some rows
##############################################################################

# Let's say a battery ran out when heater_current was 1 mA so we want to
# completely ignore that sweep
d.mask_rows(np.abs(d['heater_current'] - 1e-3) < 1e-9)

##############################################################################
# Divide the data into y vs t sweeps
##############################################################################

# Can be done by using a value that is constant for the whole sweep
sweeps = d.divide_into_sweeps('sweep_start_time')
logging.info('sweeps based on a constant value: ' + str(sweeps))

# Or based on the direction of change of a coordinate
sweeps = d.divide_into_sweeps('t')
logging.info('sweeps based on the swept coordinate: ' + str(sweeps))


##############################################################################
# Plot each sweep separately
##############################################################################

p = plot.get_plot(name='dataview example').get_plot()
p.set_xlabel('t')
p.set_ylabel('y')

for sweep_start, sweep_end in sweeps:
  # Make a "shallow copy", i.e. only the mask of dd is independent of d.
  # This is because the indices in "sweeps" are relative
  # to the _unmasked_ rows of d so we don't want to change its mask
  # in each iteration.
  dd = d.copy()
  dd.mask_rows(slice(sweep_start, sweep_end), unmask_instead=True) # hide the other rows

  assert len(np.unique(dd['heater_current'])) == 1
  heater_cur = dd['heater_current'][0]

  p.add_trace(dd['t'], dd['y'],
              title='%g (mA)' % (heater_cur*1e3))

p.update()

print "\nPress enter to end script."
sys.stdin.read(1)
