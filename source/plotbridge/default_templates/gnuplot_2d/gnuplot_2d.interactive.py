#!/usr/bin/python

import os
import sys
import time
import logging

import subprocess



replot_poll_period = 1
plot_script_extension = '.gnuplot'

plot_script = None
for f in os.listdir('.'):
  if f.endswith(plot_script_extension):
    plot_script = f
    break
assert plot_script != None, 'No file ending with "%s" found in current directory.' % plot_script_extension

# Check if the plot script is already being plotted
# by another instance of the script
lock_file = plot_script + '.lock'

def refresh_lock():
  with open(lock_file, 'w') as lock:
    lock.write( "%.3f" % (time.time()) )

def exit_if_locked():
  try:
    with open(lock_file, 'r') as f:
      # the lockfile contains a timestamp
      if float(f.read()) > time.time() - max(3, 2*replot_poll_period):
        logging.warn("It seems that the file (%s) is already being plotted. Exiting...",
                     plot_script)
        exit()
  except IOError:
    return # lock doesn't exist


# Output the initial plot script
with open(plot_script, 'r') as f:
  print '---start of %s---' % plot_script
  print f.read()
  print '---end of %s---\n' % plot_script

exit_if_locked() # technically, this and the lock file creation should be done atomically...
try:
  refresh_lock()
  
  # Initial plot
  plot_last_changed_time = os.path.getmtime(plot_script)
  eps_last_changed_time = 0
  gp = subprocess.Popen(['gnuplot', plot_script, '-'],
                        stdin=subprocess.PIPE, stdout=sys.stdout, stderr=sys.stderr)

  # Watch directory for changes and replot when necessary.
  # Use simple polling of st_mtime since it works on Linux and Windows
  # and the polling period is reasonably slow (~seconds).

  print "Replotting every %g seconds (if plot script modification time changes)..." % replot_poll_period
  print "Hit <ctrl> + C to exit."

  while gp.poll() == None: # keep polling as long as gnuplot is alive

    time.sleep(replot_poll_period)
    refresh_lock()

    # Update plot if the plot script was modified
    plot_changed_time = os.path.getmtime(plot_script)
    if plot_changed_time != plot_last_changed_time:
      #logging.debug('Plot changed. Reloading plot script.')
      gp.stdin.write('load "%s"\n' % plot_script)
      plot_last_changed_time = plot_changed_time

    # convert EPS to PDF
    if os.path.isfile('output.eps'):
      try:
        eps_changed_time = os.path.getmtime('output.eps')
        if eps_changed_time != eps_last_changed_time:
          eps_last_changed_time = eps_changed_time
          with open(os.devnull, 'wb') as DEVNULL:
            subprocess.call(['ps2pdf', '-dEPSCrop', 'output.eps'], stdin=None, stdout=DEVNULL, stderr=DEVNULL)
      except:
        pass # can fail because ps2pdf is not installed or because EPS is not fully generated yet

finally:
  try: os.remove(lock_file)
  except: pass
