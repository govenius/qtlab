# plotbridge.py, template based plot generation
# Joonas Govenius <joonas.govenius@aalto.fi>, 2014
#
# Parts from original (2014) qtlab plotting code by:
# Reinier Heeres <reinier@heeres.eu>
# Pieter de Groot <pieterdegroot@gmail.com>
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

import gobject
import logging
import os
import stat
import shutil
import time
import types
import numpy as np
import uuid
from collections import OrderedDict
import itertools
import subprocess
import re



class Plot():
  '''
  Container for a set of traces ("a plot") and a
  template based system for generating a script that plots them.

  Takes in "global" (i.e. per-plot) properties
  together with a set of traces (data + per-trace-options).

  Template based means that the the output is a text file that can
  be executed using an external plot engine (e.g. gnuplot or Matlab).
  For example, the output for a plot of two traces called "iv" (using the "gnuplot_2d"
  template) consists of a directory 'iv' that contains:
    * gnuplot_2d.cfg   (a text config file specifying which interpreter to use etc.)
    * gnuplot_2d.preprocess   (an optional script that preprocesses the trace data and/or plot script)
    * gnuplot_2d.interactive.py  (an optional script that plots iv.gnuplot in an interactive mode,
                                  if different from running iv.gnuplot directly.)
    * iv.gnuplot              (the main plot script passed to the plot engine [gnuplot]) 
    * trace_UUID1.npy         (binary data for trace 1)
    * trace_UUID2.npy         (binary data for trace 2)
  where 'trace_...' contain the traces as binary data (referenced in .gnuplot).


  You can easily add templates by modifying the default ones.


  Some principles that should help keep the system simple and
  maintainable in the long term (as maintainers change):

  * We provide a reasonable set of properties (xlabel, title, etc.)
    for visualizing data, but fine tuning (e.g., tick marker sizes)
    for "publication quality" plots is done directly in a custom template file.
    It is a good idea to leave commented out examples of such fine tuning
    in the default templates.

  * Options should be explicit, rather than *args or *kwargs, when possible.
    We will also raise loud warnings (or exceptions) if we are not sure
    what the user wants.
    In non-"mission critical" tasks, silently guessing is rarely a good idea
    as it makes it hard to find out what's wrong (if the guess is wrong).

  * Trace data is stored in binary "trace_UUID.npy" files only, i.e.,
    no in-memory arrays should be stored after add_trace() has returned.
    This prevents errors originating from discrepancies in duplicate data
    and reduces the risk of memory leaks.

  * The binary "trace_UUID.npy" files are the only method of transmitting
    the traces to plot engines.
    No ASCII files or pipes in Python. (If your plot engine can't read binary,
    write a separate .preprocess script that does the conversion before calling
    the actual plot engine (i.e. in your template's .preprocess file).)

  * Only the trace data (not trace options) can be updated after add_trace() has been called,
    just to keep things simple.
    Updating the data is done by update_trace() and it simply updates the corresponding
    binary "trace_UUID.npy" file.

  * This Plot class does not directly rely on qtlab objects, but is rather a
    self-contained plotting framework using standard Python/numpy functionality and Jinja2.
    Use qt.plot() for automatically creating and updating plots/traces from qtlab.Data objects
    (as opposed to Numpy arrays).
  '''

  _already_warned_about_subprocess_version = False

  def __init__(self, name=None, template='gnuplot_2d', output_dir=None, overwrite=False):
    '''
    Create a plot object.

    name (string)       --- default will be 'plot<n>'
    template (string)   --- the template (e.g. 'gnuplot_2d') for generating the output,
                            see the default_templates subdir for available built-in alternatives
                            or provide a full path to your own custom template.
    output_dir (string) --- where output subdir should be created,
                            default is current working dir.
    overwrite  (string) --- If False, append a number to output_dir if it already exists.
    '''

    parent_dir = (output_dir if output_dir != None else '.')
    assert os.path.isdir(parent_dir), '%s is not a directory.' % parent_dir

    self._set_name(name, parent_dir, not overwrite)

    self._set_template(template)

    # per-plot-properties
    self._global_opts = {}
    self.reset_options()

    # dict of per-trace-properties, keys are trace_ids (i.e. randmon UUIDs)
    self._traces = OrderedDict()

    # create the output directory and copyt the helper files there
    self.update()

  ##############
  #
  # Plot wide operations
  #
  ##############  

  def _set_name(self, name, parent_dir, auto_increment):
    '''Internal: Set the plot name. (call only from __init)'''
    self._name = name if name != None else 'plot'
    path_friendly_name = self.get_name(path_friendly=True)
    assert len(path_friendly_name) > 0, 'invalid name %s' % name
    d = os.path.join(parent_dir, path_friendly_name)

    if auto_increment:
      # Append a number to name if the dir already exists
      i = 2
      while os.path.isdir(d):
        d = os.path.join(parent_dir, '%s_%d' % (path_friendly_name, i))
        self._name = '%s_%d' % (name, i)
        i += 1

    if os.path.exists(d):

      self._output_dir = d
      logging.warn('Removing contents of old "%s"', d)
      try:
        for f in os.listdir(d):
          if f.endswith('.lock'): continue # don't remove lock files
          ppp = os.path.join(d,f)
          try:
            if   os.path.isfile(ppp) or os.path.islink(ppp): os.unlink(ppp)
            elif os.path.isdir(ppp): logging.warn('Ignoring directory %s', ppp) #shutil.rmtree(ppp)  # Normally there are no subdirs, so stay on the safe side and don't delete them...
          except:
            pass # files may be locked by other processes (especially on Windows), keep removing others
      except:
        logging.exception('Could not remove old %s', d)

    else:

      try:
        # Don't create parent dirs automatically (os.makedirs), because it's likely
        # that there's a typo in the path if parent_dir does not exist.
        os.mkdir(d)
        self._output_dir = d
      except:
        logging.exception('Could not create %s. Are you sure the parent exists and is writable by you?', d)
        raise

  def get_name(self, path_friendly=False):
    '''Get plot name.'''
    if not path_friendly:
      return self._name
    else:
      # leave out non-alphanumeric characters
      return "".join(x if x.isalnum() else '_' for x in self._name)

  def get_output_dir(self):
    '''Get output dir name.'''
    return os.path.abspath(self._output_dir)

  def _set_template(self, template):
    '''
    Internal: Set the template name. (call only from __init)
    '''
    template_name = os.path.split(template.strip('/\\'))[1]
    abspath = os.path.abspath(os.path.join(template, template_name + '.template'))
    try: # see if a full path was already given
      with open(abspath, 'r') as f:
        f.read(1) # make sure we can read from the file
      self._template = abspath
      return
    except:
      default_template_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'default_templates') # this should give us something that ends .../qtlab/source/plot_engines/default_templates
      try: # look in the default templates dir
        absdefpath = os.path.join(default_template_path, template_name, template_name + '.template')
        with open(absdefpath, 'r') as f:
          f.read(1) # make sure we can read from the file
        self._template = absdefpath
        return
      except:
        logging.exception('Could not find a template in %s or %s.', abspath, absdefpath)
        raise

  def get_template(self):
    '''
    template (string)   --- returns the full path to the .template file used for this plot
                            as a triple (directory, filname, extension).
    '''
    template_dir, template_file = os.path.split(self._template)
    ext_start = template_file.rindex('.')
    return template_dir, template_file[:ext_start], template_file[ext_start+1:]

  def clear(self, update=True):
    '''Remove all traces.'''
    logging.info('Clearing plot %s...', self._name)
    for trace_id in self._traces.keys(): self.remove_trace(trace_id, update=False)
    if update: self.update()

  def update(self):
    '''Regenerate the outputs based on the specified template.'''
    from jinja2 import Environment, FileSystemLoader

    template_dir, template_name, template_ext = self.get_template()
    template_file = template_name + '.' + template_ext
    try:
      env = Environment(loader=FileSystemLoader(template_dir),
                trim_blocks=True,
                keep_trailing_newline=True) # This option requires at least version 2.7 of jinja2

      env.filters['isint'] = lambda x: isinstance(x, int)
      env.filters['ifnone'] = lambda x, default='': default if x == None else x
      env.filters['allnone'] = lambda x: min(map(lambda y: y == None, x)) # check if all entries of iterable are None

      template = env.get_template(template_file)
    except:
      logging.exception('Could not load a template from %s. Refer to Jinja2 documentation for valid syntax.', self.get_template())
      raise

    cfg = self._read_config()

    trace_opts = []
    for trace_id in self._traces.keys():
      d = dict(self._traces[trace_id])
      d['npyfile'] = 'trace_%s.npy' % trace_id
      trace_opts.append(d)

    # Generate the plot script
    # Write to a temp file (.new) first and rename it once its complete.
    # This way the update is atomic from the point of view of
    # external programs that monitor changes to the file.
    out_dir = self.get_output_dir()
    plot_script = os.path.join(out_dir, self.get_name(path_friendly=True) + cfg['extension'])
    with open(plot_script + '.new', 'w') as f:
      f.write( template.render(global_opts=self._global_opts, traces=trace_opts) )
    shutil.move(plot_script + '.new', plot_script)

    if os.name != 'nt' and ('executable' in cfg.keys()) and bool(cfg['executable']):
      # Make the plot script executable
      st = os.stat(plot_script)
      os.chmod(plot_script, st.st_mode | stat.S_IXUSR | stat.S_IXGRP)

    # Copy helper files associated with template (i.e. all other files that start with the template name.)
    for helper_file in os.listdir(template_dir):
      if (helper_file.endswith('~')
        or helper_file.startswith('.')
        or helper_file.startswith('#')
        or helper_file.endswith('.template')
        or helper_file.endswith('.cfg')): continue
      shutil.copy(os.path.join(template_dir, helper_file),
                  os.path.join(out_dir, helper_file))

    # Run the .preprocess script if it exists.
    preprocess_script = os.path.abspath( os.path.join(out_dir, template_name + '.preprocess') )
    if os.path.exists(preprocess_script):
      interpreter = cfg['preprocess-interpreter'] if 'preprocess-interpreter' in cfg.keys() else ''
      try:

        try:
          # This back-ported version includes the "timeout" parameter
          from subprocess32 import call
        except:
          # Otherwise, use the standard module, which only includes "timeout" for Python >= 3.3
          from subprocess import call

        preprocess_timeout = max(1, int(cfg['preprocess-timeout'])) if 'preprocess-timeout' in cfg.keys() else 15.
        if len(interpreter) > 0: call_args = [ interpreter, preprocess_script ]
        else:                    call_args = [ preprocess_script ]
        logging.info('Executing %s.' % (preprocess_script))
        with open(os.devnull, 'wb') as DEVNULL:
          try:
            call(call_args, cwd=out_dir, stdin=None, stdout=DEVNULL, stderr=DEVNULL,
                 timeout=preprocess_timeout)
          except:
            if not Plot._already_warned_about_subprocess_version:
              logging.warn('Your version of subprocess.call() does not support the timeout parameter so the preprocess script can block execution indefinitely (if it hangs). You are encouraged to install the back-ported subprocess32 module ("pip install subprocess32").')
              Plot._already_warned_about_subprocess_version = True
            subprocess.call(call_args, cwd=out_dir, stdin=None, stdout=DEVNULL, stderr=DEVNULL)

      except:
        logging.exception('Failed to execute %s', preprocess_script)



  def run(self, interactive=True):
    '''
    Execute the generated plot script.

    If interactive, execute the .interactive script instead.
    '''

    cfg = self._read_config()
    template_dir, template_name, template_ext = self.get_template()

    if interactive:
      plot_script = template_name + '.interactive' + (('.'+cfg['interactive-extension']) if 'interactive-extension' in cfg.keys() else '')
      interpreter = cfg['interactive-interpreter'] if 'interactive-interpreter' in cfg.keys() else ''
    else:
      plot_script = self.get_name(path_friendly=True) + cfg['extension']
      interpreter = cfg['interpreter'] if 'interpreter' in cfg.keys() else ''

    out_dir = os.path.abspath(self.get_output_dir())

    p_args = ( [ interpreter, os.path.join(out_dir, plot_script) ]
               if len(interpreter) > 0 else
               [ os.path.join(out_dir, plot_script) ] )

    # from subprocess import DEVNULL # Python 3 only
    log_file = os.path.join(out_dir, '%s%s.out' % (self.get_name(path_friendly=True), '_interactive' if interactive else ''))
    logging.info('Executing %s. Log messages in %s.', plot_script, log_file)

    log_file = open(log_file, 'a')
    print p_args
    subprocess.Popen(p_args, cwd=out_dir,
                     stdin=None, stdout=log_file, stderr=log_file)


  ##############
  #
  # Global plot properties
  #
  ##############

  def reset_options(self):
    ''' Set global plot properties back to defaults. '''
    self.set_export_png()
    self.set_export_eps()
    self.set_show_on_screen()
    self.set_width()
    self.set_height()
    self.set_fontsize()
    self.set_title(self.get_name())
    self.set_legend()
    self.set_xlabel()
    self.set_x2label()
    self.set_ylabel()
    self.set_y2label()
    self.set_zlabel()
    self.set_cblabel()
    self.set_xlog()
    self.set_x2log()
    self.set_ylog()
    self.set_y2log()
    self.set_zlog()
    self.set_xticks()
    self.set_x2ticks()
    self.set_yticks()
    self.set_y2ticks()
    self.set_zticks()
    self.set_xrange()
    self.set_x2range()
    self.set_yrange()
    self.set_y2range()
    self.set_zrange()
    self.set_grid()


  def set_export_png(self, val=True):
    ''' Export a PNG image of the plot.
        Some templates may export to another rasterized format or not support this at all. '''
    assert isinstance(val, bool), 'export_png must be set to True or False.'
    self._global_opts['export_png'] = val

  def set_export_eps(self, val=True):
    ''' Export an EPS image of the plot.
        Some templates may export to another vectorized format or not support this at all. '''
    assert isinstance(val, bool), 'export_eps must be set to True or False.'
    self._global_opts['export_eps'] = val

  def set_show_on_screen(self, val=True):
    ''' Shown the output on screen (as opposed to only exporting to PNG/EPS).

        Calling this does not automatically execute the plot script.
        (Do it using .run(interactive=True) or directly from a shell.) '''
    assert isinstance(val, bool), 'show_on_screen must be set to True or False.'
    self._global_opts['show_on_screen'] = val

  def set_width(self, val=800):
    ''' Set plot width in pixels.'''
    assert int(val) > 10, 'width must be given in pixels'
    self._global_opts['width'] = int(val)

  def set_height(self, val=600):
    ''' Set plot height in pixels.'''
    assert int(val) > 10, 'height must be given in pixels'
    self._global_opts['height'] = int(val)

  def set_fontsize(self, val=20):
    ''' Set the "base" font size for axis labels etc.
      Templates usually scale this to a smaller value for ticks etc. '''
    assert int(val) > 2, 'font size must be given in integer number of points.'
    self._global_opts['basefontsize'] = int(val)

  def set_title(self, val=None):
    ''' Set the title of the plot window. '''
    assert val == None or isinstance(val, basestring), 'title must be a string'
    self._global_opts['title'] = val

  def set_legend(self, val=True):
    ''' Enable/disable legend.'''
    assert isinstance(val, bool), 'legend must be set to True or False.'
    self._global_opts['legend'] = val

  def set_xlabel(self, val=None):
    '''Set label for left x axis.'''
    assert val == None or isinstance(val, basestring), 'xlabel must be a string'
    self._global_opts['xlabel'] = val

  def set_x2label(self, val=None):
    '''Set label for right x axis.'''
    assert val == None or isinstance(val, basestring), 'x2label must be a string'
    self._global_opts['x2label'] = val

  def set_ylabel(self, val=None):
    '''Set label for bottom y axis.'''
    assert val == None or isinstance(val, basestring), 'ylabel must be a string'
    self._global_opts['ylabel'] = val

  def set_y2label(self, val=None):
    '''Set label for top y axis.'''
    assert val == None or isinstance(val, basestring), 'y2label must be a string'
    self._global_opts['y2label'] = val

  def set_zlabel(self, val=None):
    '''Set label for z/color axis.'''
    assert val == None or isinstance(val, basestring), 'zlabel must be a string'
    self._global_opts['zlabel'] = val

  def set_cblabel(self, val=None):
    '''Set label for z/color axis.'''
    assert val == None or isinstance(val, basestring), 'cblabel must be a string'
    self._global_opts['cblabel'] = val

  def set_xlog(self, val=False):
    '''Set log scale on left x axis.'''
    assert isinstance(val, bool), 'xlog must be set to True or False.'
    self._global_opts['xlog'] = val

  def set_x2log(self, val=False):
    '''Set log scale on right x axis.'''
    assert isinstance(val, bool), 'x2log must be set to True or False.'
    self._global_opts['x2log'] = val

  def set_ylog(self, val=False):
    '''Set log scale on bottom y axis.'''
    assert isinstance(val, bool), 'ylog must be set to True or False.'
    self._global_opts['ylog'] = val

  def set_y2log(self, val=False):
    '''Set log scale on top y axis.'''
    assert isinstance(val, bool), 'y2log must be set to True or False.'
    self._global_opts['y2log'] = val

  def set_zlog(self, val=False):
    '''Set log scale on z/colorbar.'''
    assert isinstance(val, bool), 'zlog must be set to True or False.'
    self._global_opts['zlog'] = val

  def set_xticks(self, val=True, options=None):
    '''Enable/disable ticks on left x axis.'''
    assert isinstance(val, bool), 'xticks must be set to True or False.'
    self._global_opts['xticks'] = val

  def set_x2ticks(self, val=True, options=None):
    '''Enable/disable ticks on right x axis, or set to "mirror" for mirroring x1 ticks.'''
    assert isinstance(val, bool) or val=='mirror', 'xt2icks must be set to True, False, or "mirror".'
    self._global_opts['x2ticks'] = val

  def set_yticks(self, val=True, options=None):
    '''Enable/disable ticks on bottom y axis.'''
    assert isinstance(val, bool), 'yticks must be set to True or False.'
    self._global_opts['yticks'] = val

  def set_y2ticks(self, val=True, options=None):
    '''Enable/disable ticks on top y axis, or set to "mirror" for mirroring y1 ticks.'''
    assert isinstance(val, bool) or val=='mirror', 'y2ticks must be set to True, False, or "mirror".'
    self._global_opts['y2ticks'] = val

  def set_zticks(self, val=True, options=None):
    '''Enable/disable ticks on z axis.'''
    assert isinstance(val, bool), 'zticks must be set to True or False.'
    self._global_opts['zticks'] = val

  def set_xrange(self, minval=None, maxval=None):
    '''Set left x axis range, None means auto.'''
    assert minval == None or np.isreal(minval), 'minval must be a real number.'
    assert maxval == None or np.isreal(maxval), 'maxval must be a real number.'
    self._global_opts['xrange'] = (minval, maxval)

  def set_x2range(self, minval=None, maxval=None):
    '''Set right x axis range, None means auto.'''
    assert minval == None or np.isreal(minval), 'minval must be a real number.'
    assert maxval == None or np.isreal(maxval), 'maxval must be a real number.'
    self._global_opts['x2range'] = (minval, maxval)

  def set_yrange(self, minval=None, maxval=None):
    '''Set bottom y axis range, None means auto.'''
    assert minval == None or np.isreal(minval), 'minval must be a real number.'
    assert maxval == None or np.isreal(maxval), 'maxval must be a real number.'
    self._global_opts['yrange'] = (minval, maxval)

  def set_y2range(self, minval=None, maxval=None):
    '''Set top y axis range, None means auto.'''
    assert minval == None or np.isreal(minval), 'minval must be a real number.'
    assert maxval == None or np.isreal(maxval), 'maxval must be a real number.'
    self._global_opts['y2range'] = (minval, maxval)

  def set_zrange(self, minval=None, maxval=None):
    '''Set z axis range, None means auto.'''
    assert minval == None or np.isreal(minval), 'minval must be a real number.'
    assert maxval == None or np.isreal(maxval), 'maxval must be a real number.'
    self._global_opts['zrange'] = (minval, maxval)

  def set_grid(self, val=True):
    '''Show grid lines.'''
    assert isinstance(val, bool), 'grid must be set to True or False.'
    self._global_opts['grid'] = val


  ##############
  #
  # Adding/updating/removing traces
  #
  ##############

  def add_trace(self, x, y=None, yerr=None,
        slowcoordinate=None,
        points=True, lines=False,
        x_plot_units=1, y_plot_units=1, # divide data by these factors before plotting
        skip=1, # plot only every skip'th point
        crop=0, # crop this many points at the ends
        title=None,
        pointtype=None, pointsize=None,
        linetype=None, linewidth=None,
        color=None,
        right=None,
        update=False):
    '''
    Add a 1D trace to the plot (or rather a 2D parametric curve).

    x,y    --- the x and y coordinates as two separate vectors of length N or as a Nx2 or 2xN matrix
    yerr   --- error bars for the ycoordinate
    slowcoordinate --- used as a second coordinate in some templates (e.g. 2D color maps)
    points --- draw points
    lines  --- connect points with lines
    x_plot_units  --- divide x data by this factor before plotting
    y_plot_units  --- divide y data by this factor before plotting
    skip   --- plot only every skip'th point
    crop   --- leave out crop points at each end
    title  --- label for this trace
    point/linetype --- integer indicating the point/line style (see gnuplot pointtypes)
    point/linesize --- size
    color  --- point/line color
    right  --- use the second y-axis (left == first y-axis)
    update --- whether plot script should be regenerated
    '''
    assert crop == None or isinstance(crop, int), 'crop must be an int'
    assert skip == None or isinstance(skip, int), 'skip must be an int'

    if slowcoordinate == None and title != None and len(title) > 0:
      # Attempt parsing the slow coordinate value for title, if none was specified.
      m = re.search(r'([e\d\.\+\-]+)', title)
      if m and len(m.groups()) == 1: # don't try to guess if multiple matches
        try:
          slowcoordinate = float(m.group(1))
        except:
          pass

    # Generate a random unique id for the trace
    trace_id = uuid.uuid4().hex

    self._traces[trace_id] = {
      'recordformat':[], # this is updated by update_trace() (called below)
      'yerrorcol':None, # this is updated by update_trace() (called below)
      'slowcoordinate':slowcoordinate,
      'points':points,
      'lines':lines,
      'x_plot_units':x_plot_units,
      'y_plot_units':y_plot_units,
      'skip':skip,
      'crop':crop,
      'title':title,
      'pointtype':pointtype,
      'pointsize':pointsize,
      'linetype':linetype,
      'linewidth':linewidth,
      'color':color,
      'right':right
      }

    # Generate the binary data file
    self.update_trace(trace_id, x, y, yerr, update=update)

    return trace_id

  def update_trace(self, trace_id, x, y=None, yerr=None, update=True):
    '''
    Update data points of the specified trace.

    update --- whether plot script should be regenerated

    (Trace options cannot be updated after initial add_trace().)
    '''

    # Convert the data to a two or three (if yerr!=none) column format
    x_plot_units = self._traces[trace_id]['x_plot_units']
    y_plot_units = self._traces[trace_id]['y_plot_units']
    dd, yerr_column = self._convert_trace_input_to_list_of_tuples(x, y, yerr,
                             x_plot_units, y_plot_units)

    # Drop specified points
    crop = self._traces[trace_id]['crop']
    skip = self._traces[trace_id]['skip']
    if crop > 0 and len(data) > crop: dd = dd[crop:-crop]
    if skip > 1: dd = dd[::skip]

    trace_npy = os.path.join(self.get_output_dir(), 'trace_%s.npy' % trace_id)

    if len(dd) < 1:
      logging.warn('No points in the added/updated trace.')
      try: os.remove(trace_npy)
      except: pass # normal if there was no previous version
    else:
      # Convert all inputs to float for simplicity.
      # Should not be a big problem for plotting purposes...
      dd = dd.astype(np.float)

      # Use the default numpy binary format for output
      # Write to a temp file (.new) first and rename it once its complete.
      # This way the update is atomic from the point of view of
      # external programs that monitor changes to the file.
      dd.astype(np.float).tofile(trace_npy + '.new')
      shutil.move(trace_npy + '.new', trace_npy)

      # Update the 'recordformat' field for the trace.
      _DATA_TYPES = {
        np.dtype('int8'): 'int8',
        np.dtype('int16'): 'int16',
        np.dtype('int32'): 'int32',
        np.dtype('int64'): 'int64',
        np.dtype('uint8'): 'uint8',
        np.dtype('uint16'): 'uint16',
        np.dtype('uint32'): 'uint32',
        np.dtype('uint64'): 'uint64',
        np.dtype('float32'): 'float32',
        np.dtype('float64'): 'float64',
        }
      # All columns currently have the same format...
      self._traces[trace_id]['recordformat'] = [ _DATA_TYPES[dd.dtype] for i in range(len(dd[0])) ]
      self._traces[trace_id]['yerrorcol'] = yerr_column

    if update: self.update()

  def remove_trace(self, trace_id, update=True):
    '''
    Remove the specified trace.
    '''
    if trace_id not in self._traces.keys():
      logging.warn('No trace %s in plot %d.', trace_id, self.get_name())
      return

    # delete the binary data file
    bin_path = os.path.join(self.get_output_dir(), 'trace_%s.npy' % trace_id)
    try:
      os.remove(bin_path)
    except:
      # this can happen if the added trace was empty
      logging.debug('Could not remove %s.', bin_path)

    del self._traces[trace_id]

    if update: self.update()

  def get_ntraces(self):
    ''' Number of traces. '''
    return len(self._traces)

  ##############
  #
  # Internal helpers
  #
  ##############

  def _read_config(self):
    template_dir, template_name, template_ext = self.get_template()
    import ConfigParser
    cfg = ConfigParser.SafeConfigParser()
    cfg_path = os.path.join(template_dir, template_name + '.cfg')
    try:
      cfg.read(cfg_path)
      return dict(itertools.chain(
          cfg.items('general'),
          cfg.items('windows') if os.name == 'nt' else cfg.items('unix')
          ))
    except:
      logging.exception('Could not read the template config file %s.', cfg_path)
      raise


  def _convert_trace_input_to_list_of_tuples(self, x, y=None, yerr=None,
                         x_plot_units=1, y_plot_units=1):
    '''
    Interpret input vectors/matrices and convert them to
    the standard internal [ (x0,y0), (x1,y1), ...] format.

    (This is a conscious violation of the "no guessing"
     principle mentioned in the class doc string.)
    '''

    complex_dtypes = [np.complex, np.complex64, np.complex128, np.complexfloating, np.complex_]
    try: complex_dtypes.append(np.complex256) # 64-bit systems only? (Or maybe a numpy version thing.)
    except AttributeError: pass

    if len(x.shape) == 1:

      if x.dtype in complex_dtypes:
        assert y == None, 'x is complex so y must be None. Points described by a single complex vector will be plotted in the complex plane.'
        dd = np.array(( x.real / (1. if x_plot_units==None else x_plot_units), x.imag / (1. if y_plot_units==None else y_plot_units) )).T
      else:
        assert y != None, 'y must be given if x is a real vector.'
        assert y.dtype not in complex_dtypes, 'y must be real if x is a real vector.'
        assert x.shape == y.shape, "x and y vector lengths don't match."
        dd = np.array(( x / (1. if x_plot_units==None else x_plot_units), y / (1. if y_plot_units==None else y_plot_units) )).T

    elif len(x.shape) == 2:
      assert y == None, "If x is 2D matrix, y cannot be specified."
      assert x.dtype not in complex_dtypes, 'x must be real if x is a 2D matrix.'
      assert x.shape[0] == 2 or x.shape[1] == 2, "Input 2D matrix size is not 2xn or nx2. Don't know what you mean."
      xx = x if x.shape[0] >= x.shape[1] else x.T
      dd = np.array(( xx[:,0] / (1. if x_plot_units==None else x_plot_units), xx[:,1] / (1. if y_plot_units==None else y_plot_units) )).T

    else:
      raise Exception("Input matrix is more than 2D. Don't know how to interpret it.")

    if yerr != None:
      yerr1d = np.zeros(len(dd), dtype=np.float) + np.nan
      try: # whether a single scalar
        yerr1d[:] = float(yerr)
      except: # or a 1D vector
        assert len(yerr) == len(dd), 'yerr must be a positive scalar or a vector of same length as x'
        assert len(yerr.shape) == 1, 'yerr must be a 1D (numpy) vector'
        assert (yerr < 0).max() == 0, 'components of yerr must be non-negative'
        yerr1d[:] = yerr
      yerr1d /= (1. if y_plot_units==None else y_plot_units)
      dd = np.hstack(( dd, yerr1d.reshape((-1,1)) ))
      yerr_column = 2
    else:
      yerr_column = None

    return dd, yerr_column


##########################
#
# Test code
#
###########################

if __name__ == "__main__":
  logging.basicConfig(level=logging.DEBUG,
                      format="%(asctime)s %(filename)s:%(lineno)s %(message)s",
                      datefmt="%H:%M:%S")

  p = Plot(name='test_plot', output_dir='.')
  p.set_title('Fake IV')
  p.set_xlabel('current (nA)')
  p.set_ylabel('voltage (uV)')

  current_units = 1e-9
  voltage_units = 1e-6

  current = np.linspace(-5,5,41) * current_units
  p.set_xrange(-5.5,None) # None means autorange

  for i in range(2):
    # linear IV
    voltage = 3e3 * current

    # add noise
    errorbars = 0.1*np.abs(voltage)
    voltage += errorbars*np.random.randn(len(voltage))

    p.add_trace(current, voltage,
                yerr=errorbars,
                title='voltage %d' % i,
                x_plot_units=current_units,
                y_plot_units=voltage_units,
                lines = True, points = True,
                update=False)

  p.update()
  p.run(interactive=True)
