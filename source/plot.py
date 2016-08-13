# plot.py, abstract plotting classes
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
import time
import types
import numpy

from lib.config import get_config
config = get_config()

from data import Data
from lib import namedlist
from lib.misc import get_dict_keys
from lib.network.object_sharer import SharedGObject, cache_result
from plotbridge.plot import Plot as plt


def get_plot(name=None,
             mintime=1,
             autoupdate=True,
             template='gnuplot_2d',
             output_dir='.',
             run=True,
             replace_if_exists=False):
    '''
    Get a reference to an existing plot or create a new one.

    See plot.Plot? for description of arguments. Additionally:
        
        replace_if_exists: create a new plot, even if an old one by the same name exists.
    '''

    graph = Plot._plot_list[name]
    if replace_if_exists and graph != None:
      Plot._plot_list.remove(name)
      graph = None

    if graph is None:
        graph = Plot(name,
                     mintime,
                     autoupdate,
                     template,
                     output_dir,
                     run)

    return graph

def get_plots():
    '''
    Return all plots as a named list.
    '''

    return Plot._plot_list

def replot_all():
    '''
    Update all plots in the plot-list.
    '''
    plots = Plot._plot_list
    for p in plots:
        plots[p]._pltbr.update()


class _PlotList(namedlist.NamedList):
    def __init__(self):
        namedlist.NamedList.__init__(self, base_name='plot')

    def add(self, name, item):
        '''Add an item to the list.'''
        if name in self._list:
            self.remove(name)
        self._list[name] = item
        self._last_item = item
        self.emit('item-added', name)


class Plot(SharedGObject):
    '''
    Class for plotting Data objects.

    This is a thin wrapper around plot_engines/plotbridge.py
    and mainly adds (some degree of) backward compatibility
    and the possibility to have your plot automatically
    update as new data is added to the Data object.

    Additionally, this class keeps a list of plots, allowing
    you to recover a reference to an old plot by its name
    using plot.get_plot('name of plot').

    To customize the plot, use the plot.get_plot('name of plot').get_plot()
    function, which returns the underlying plotbridge object.

    If you are not plotting Data objects (i.e. just numpy arrays),
    you should use plotbridge directly.
    '''

    _plot_list = _PlotList()

    def __init__(self,
                 name=None,
                 mintime=1,
                 autoupdate=True,
                 template='gnuplot_2d',
                 output_dir='.',
                 run=True):
        '''
        Create a plot wrapper for plotting qt.Data objects.

        args input:
            data objects (Data)

        kwargs input:
            name (string)           --- default 'plot<n>'
            mintime (int, seconds)  --- min time between autoupdates, default 1
            autoupdate (bool)       --- update the plot when data points added, default True
            template (string)       --- default 'gnuplot_2d'
            output_dir (string)     --- directory for storing the plot. default '.'
            run (bool)              --- whether the plot script should be immediately ran (opened)
                                        (But a window may not pop up until data is added.)
         '''

        self._name = name if name else ''
        self._name = Plot._plot_list.new_item_name(self, self._name)

        SharedGObject.__init__(self, 'plot_%s' % self._name, replace=True)


        self._data = []

        self._mintime = mintime
        self._autoupdate = autoupdate

        self._last_update = 0
        self._update_hid = None

        self._pltbr = plt(name=self._name, template=template,
                                          output_dir=output_dir, overwrite=True)

        Plot._plot_list.add(self._name, self)

        if run:
          self._pltbr.run(interactive=True)

    def get_plot(self):
      ''' Returns the underlying plotbridge object. Useful for customizing plot options. '''
      return self._pltbr

    def get_name(self):
        '''Get plot name.'''
        return self._name

    def add_data(self, data, coorddim=0, valdim=1, title=None, update=True):
        '''
        Add a Data object to the plot.

        Input:
            data (Data):
                qt.Data object
            coorddim (int or tuple of ints):
                Which coordinate column(s) to use (default 0). Use, e.g., [0,1] for 3D plots.
            valdim (int or tuple of ints):
                Which value column to use.
            title (string): name of the trace
            update: whether to the plot should be updated immediately
        '''

        data_entry = {}
        data_entry['data'] = data
        data_entry['new-data-point-hid'] = \
                data.connect('new-data-point', self._new_data_point_cb)
        data_entry['new-data-block-hid'] = \
                data.connect('new-data-block', self._new_data_block_cb)

        coorddims = [ coorddim ] if isinstance(coorddim, int) else coorddim # convert a single int to a list
        data_entry['coorddims'] = coorddims

        data_entry['valdim'] = valdim

        data_entry['title'] = title if title else data.get_title(coorddims, valdim)

        self._data.append(data_entry)

        if update: self.update()

    def set_default_labels(self):
        '''
        Set default labels for the plot based on the names of the columns in the qt.Data object(s).
        '''

        x = ''
        y = ''
        z = ''
        for datadict in self._data:
            data = datadict['data']

            if x == '' and len(datadict['coorddims']) > 0:
                x = data.format_label(datadict['coorddims'][0])

            if y == '' and len(datadict['coorddims']) == 1:
                y = data.format_label(datadict['valdim'])
            elif y == '':
                y = data.format_label(datadict['coorddims'][1])

            if z == '' and len(datadict['coorddims']) > 1:
                z = data.format_label(datadict['valdim'])

        if x == '':
            x = 'X'
        self._pltbr.set_xlabel(x)
        if y == '':
            y = 'Y'
        self._pltbr.set_ylabel(y)
        if z == '':
            z = 'Z'
        self._pltbr.set_zlabel(z)
        self._pltbr.set_cblabel(z)

    def set_mintime(self, t):
        self._mintime = t

    def get_mintime(self):
        return self._mintime

    def clear(self):
        self._date = []
        self._pltbr.clear(update=True)

    def reset(self):
        self._pltbr.reset_options()

    def update(self, force=True, **kwargs):
        '''
        Update the plot.

        Input:
            force (bool): if True force an update, else check whether we
                would like to autoupdate and whether the last update is longer
                than 'mintime' ago.
        '''
        dt = time.time() - self._last_update

        if not force and self._autoupdate is not None and not self._autoupdate:
            return

        if self._update_hid is not None:
            if force:
                gobject.source_remove(self._update_hid)
                self._update_hid = None
            else:
                return

        cfgau = config.get('live-plot', True)
        if force or (cfgau and dt > self._mintime):
            self._last_update = time.time()
            self._do_update(**kwargs)

        # Auto-update later
        elif cfgau:
            self._queue_update(force=force, **kwargs)

    def _do_update(self):
      if len(self._data) == 0:
        self._pltbr.update() # not plotting data objects
      else:
        # Plot data from the specified data objects 
        self._pltbr.clear()
        for d in self._data:
          if d['data'].get_npoints() == 0: continue
          if len(d['coorddims']) > 1: logging.warn('Multidimensional plots from Data files not implemented.')
          self._pltbr.add_trace(d['data'].get_data()[:,d['coorddims'][0]],
                                d['data'].get_data()[:,d['valdim']],
                                title=d['title'])
        self._pltbr.update()

    def _queue_update(self, force=False, **kwargs):
        if self._update_hid is not None:
            return
        self._update_hid = gobject.timeout_add(int(self._mintime * 1000),
                self._delayed_update, force, **kwargs)

    def _delayed_update(self, force=True, **kwargs):
        self._update_hid = None
        self.update(force=force, **kwargs)
        return False

    def _new_data_point_cb(self, sender):
        try:
            self.update(force=False)
        except Exception, e:
            logging.warning('Failed to update plot %s: %s', self._name, str(e))

    def _new_data_block_cb(self, sender):
        self.update(force=False)

    def save_png(self, *args, **kwargs):
      ''' Deprecated. Use plotbridge.Plot.set_export_png(True) or plotbridge.Plot.set_export_eps(True) followed by run(). '''
      logging.warn('This function is deprecated. Use plotbridge.Plot.set_export_png(True) or plotbridge.Plot.set_export_eps(True) followed by run(). Arguments ignored: %s, %s', args, kwargs)

    def save_eps(self, *args, **kwargs):
      ''' Deprecated. Use plotbridge.Plot.set_export_png(True) or plotbridge.Plot.set_export_eps(True) followed by run(). '''
      logging.warn('This function is deprecated. Use plotbridge.Plot.set_export_png(True) or plotbridge.Plot.set_export_eps(True) followed by run(). Arguments ignored: %s, %s', args, kwargs)

    def set_labels(self, x='', y='', z='', update=True):
      ''' Deprecated. '''
      logging.warn('This function is deprecated. Use set_default_labels() if you want to set the labels to the names specified in the qt.Data object(s). Otherwise, use get_plot().set_xlabel etc.')



################ Deprecated. ##################

def plot(*args, **kwargs):
  ''' Deprecated. '''
  logging.warn("This function is deprecated and does nothing. Use plot.get_plot('name of plot') instead.")

def waterfall(*args, **kwargs):
    ''' Deprecated. '''
    logging.warn("This function is deprecated and does nothing. Use plot.get_plot('name of plot') and an appropriate template instead.")

def plot3(*args, **kwargs):
  ''' Deprecated. '''
  logging.warn("This function is deprecated and does nothing. Use plot.get_plot('name of plot') instead.")
