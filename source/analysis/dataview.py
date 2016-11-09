# dataview.py, class for post-processing measurement data
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

import gobject
import os
import os.path
import time
import numpy as np
import types
import re
import logging
import copy
import shutil
import itertools
from collections import OrderedDict

from gettext import gettext as _L

from lib import namedlist, temp
from lib.misc import dict_to_ordered_tuples, get_arg_type
from lib.config import get_config
config = get_config()
in_qtlab = config.get('qtlab', False)
from lib.network.object_sharer import SharedGObject, cache_result

if in_qtlab:
    import qt

class DataView():
    '''
    Class for post-processing measurement data. Main features are:
      * Concatenating multiple qt.Data objects
      * Creating "virtual" columns by parsing comments or .set files
        or by applying arbitrary functions to the data
      * Dividing the rows into sweeps.

    See qtlab/examples/analysis_with_dataview.py for example use.
    '''

    def __init__(self, data, deep_copy=False, source_column_name='data_source', fill_value=None, **kwargs):
        '''
        Create a new view of an existing data object for post-processing.
        The original data object will not be modified.

        args:
          data -- qt.Data object(s)
        
        kwargs input:
          deep_copy          -- specifies whether the underlying data is copied or 
                                only referenced (more error prone, but memory efficient)
          source_column_name -- specifies the name of the (virtual) column that tells which
                                data object the row originates from. Specify None, if
                                you don't want this column to be added.
          fill_value         -- fill value for columns that do not exist in all data objects.
                                Default is None, in which case the column is omitted entirely.
        '''

        self._virtual_dims = {}

        if isinstance(data, DataView): # clone
          # these private variables should be immutable so no need to deep copy
          self._dimensions = data._dimensions
          self._dimension_indices = data._dimension_indices
          self._source_col = data._source_col
          self._comments = data._comments
          self._settings = data._settings
          
          if deep_copy:
            self._data = data._data.copy()
          else:
            self._data = data._data

          # Always deep copy the mask
          self._mask = data._mask.copy()

          for name, fn in data._virtual_dims.items():
              self._virtual_dims[name] = fn

          return

        try: # see if a single Data object
          self._dimensions = data.get_dimension_names()
          unmasked = data.get_data().copy() if deep_copy else data.get_data()
          
          if source_column_name != None:
            n = data.get_name()
            self._source_col = [n for i in range(data.get_npoints())]
          else:
            self._source_col = None

          self._comments = data.get_comment(include_row_numbers=True)

          try:
            self._settings = [ (0, self._parse_settings(data)) ]
          except:
            if len(data.get_filepath()) > 0:
              # don't warn the user if the data object has no files associated with it
              logging.exception("Could not parse the instrument settings file. Doesn't matter if you were not planning to add virtual columns based on values in the .set files.")
            self._settings = None

        except MemoryError as e:
          raise

        except Exception as e: # probably a sequence of Data objects then
          self._dimensions = set(itertools.chain( *(dd.get_dimension_names() for dd in data) ))
          
          unmasked = {}
          for dim in self._dimensions:
            unmasked[dim] = []
            for dat in data:
              if len(dat.get_dimension_names()) == 0:
                logging.warn("Data object '%s' seems to contain zero columns. Skipping it..." % (str(dat)))
                break

              n_rows = len(dat[:,0])
              if n_rows == 0:
                logging.warn("Data object '%s' seems to contain zero rows. Skipping it..." % (str(dat)))
                break

              try:
                unmasked[dim].append(dat[dim])
              except:
                msg = "Dimension '%s' does not exist in Data object '%s'. " % (dim, str(dat))
                if fill_value == None:
                  # ignore dimensions that don't exist in all data objects
                  del unmasked[dim]
                  msg += ' Omitting the dimension.'
                  logging.warn(msg)
                  break
                else:
                  unmasked[dim].append(fill_value + np.zeros(n_rows, dtype=type(fill_value)))
                  msg += ' Using fill_value = %s (for %d rows)' % (str(fill_value), len(unmasked[dim][-1]))
                  logging.warn(msg)

            # concatenate rows from all files
            if dim in unmasked.keys():
              unmasked[dim] = np.concatenate(unmasked[dim])
          
          # add a column that specifies the source data file
          lens = [ dat.get_npoints() for dat in data ]
          if source_column_name != None:
            names = [ '%s_(%s)' % (dat.get_name(), dat.get_filename().strip('.dat')) for dat in data ]
            self._source_col = [ [n for jj in range(l)] for n,l in zip(names,lens) ]
            #self._source_col = [ jj for jj in itertools.chain.from_iterable(self._source_col) ] # flatten
            self._source_col = list(itertools.chain.from_iterable(self._source_col)) # flatten
          else:
            self._source_col = None
          
          # keep only dimensions that could be parsed from all files
          self._dimensions = unmasked.keys()
          unmasked = np.array([unmasked[k] for k in self._dimensions]).T

          # concatenate comments, adjusting row numbers from Data object rows to the corresponding dataview rows
          lens = np.array(lens)
          self._comments = [ dat.get_comment(include_row_numbers=True) for dat in data ]
          all_comments = []
          for jj,comments in enumerate(self._comments):
              all_comments.append([ (rowno + lens[:jj].sum(), commentstr) for rowno,commentstr in comments ])
          self._comments = list(itertools.chain.from_iterable(all_comments)) # flatten by one level

          # Parse all settings (.set) files and store them in a dict where the key indicates
          # the starting row of the .set
          try:
            self._settings = []
            all_settings = [ self._parse_settings(dat) for dat in data ]
            for jj,settings in enumerate(all_settings):
              self._settings.append( (lens[:jj].sum(), settings) )
          except:
            if not all( len(dat.get_filepath()) == 0 for dat in data ):
              # don't warn the user if the data objects have no files associated with them
              logging.exception("Could not parse the instrument settings file for one or more qt.Data objects. Doesn't matter if you were not planning to add virtual columns based on values in the .set files.")
            self._settings = None

        self._data = unmasked
        self._mask = np.zeros(len(unmasked), dtype=np.bool)
        self._mask_stack = []

        self._dimension_indices = dict([(n,i) for i,n in enumerate(self._dimensions)])
        self.set_mask(False)

        if source_column_name != None:
          self.add_virtual_dimension(source_column_name, arr=np.array(self._source_col))

    def __getitem__(self, index):
        '''
        Access the data.

        index may be a slice or a string, in which case it is interpreted
        as a dimension name.
        '''
        if isinstance(index, basestring):
            return self.get_column(index)
        else:
            return self.get_data()[index]

    def copy(self, copy_data=False):
        '''
        Make a copy of the view. The returned copy will always have an independent mask.
        
        copy_data -- whether the underlying data is also deep copied.
        '''
        return DataView(self, deep_copy=copy_data)

    def get_data_source(self):
        '''
        Returns a list of strings that tell which Data object each of the unmasked rows originated from.
        '''
        return [ i for i in itertools.compress(self._source_col, ~(self._mask)) ]

    def clear_mask(self):
        '''
        Unmask all data (i.e. make all data in the initially
        provided Data object visible again).
        '''
        self._mask[:] = False
        self._mask_stack = []

    def get_mask(self):
        '''
        Get a vector of booleans indicating which rows are masked.
        '''
        return self._mask.copy()

    def get_dimensions(self):
        '''
        Returns a list of all dimensions, both real and virtual.
        '''
        return list(itertools.chain(self._dimension_indices.keys(), self._virtual_dims.keys()))

    def get_comments(self, include_row_numbers=True):
        '''
        Return the comments parsed from the data files.
        '''
        return self._comments if include_row_numbers else [ commentstr for rowno,commentstr in self._comments ]

    def get_continuous_ranges(self, masked_ranges=False):
        '''
        Returns a list of (start,stop) tuples that indicate continuous ranges of (un)masked data.
        '''
        m = self.get_mask() * (-1 if masked_ranges else 1)
        
        dm = m[1:] - m[:-1]
        starts = 1+np.where(dm < 0)[0]
        stops = 1+np.where(dm > 0)[0]

        if not m[0]:
            starts = np.concatenate(( [0], starts ))
        if not m[-1]:
            stops = np.concatenate(( stops, [len(m)] ))

        return zip(starts, stops)

    def set_mask(self, mask):
        '''
        Set an arbitrary mask for the data. Should be a vector of booleans of
        the same length as the number of data points.
        Alternatively, simply True/False masks/unmasks all data.

        See also mask_rows().
        '''
        try:
          if mask:
            self._mask[:] = True
          else:
            self._mask[:] = False
        except:
          m = np.zeros(len(self._mask), dtype=np.bool)
          m[mask] = True
          self._mask = m

    def mask_rows(self, row_mask, unmask_instead = False):
        '''
        Mask rows in the data. row_mask can be a slice or a boolean vector with
        length equal to the number of previously unmasked rows.

        The old mask is determined from the mask of the first column.

        Example:
          d = DataView(...)
          # ignore points where source current exceeds 1 uA.
          d.mask_rows(np.abs(d['I_source']) > 1e-6)
        '''
        old_mask = self._mask
        n = (~old_mask).astype(np.int).sum() # no. of previously unmasked entries
        #logging.debug("previously unmasked rows = %d" % n)

        # new mask for the previously unmasked rows
        new_mask = np.empty(n, dtype=np.bool); new_mask[:] = unmask_instead
        new_mask[row_mask] = (not unmask_instead)
        #logging.debug("new_mask.sum() = %d" % new_mask.sum())

        # combine the old and new masks
        full_mask = old_mask.copy()
        full_mask[~old_mask] = new_mask

        logging.debug("# of masked/unmasked rows = %d/%d" % (full_mask.astype(np.int).sum(), (~full_mask).astype(np.int).sum()))
        self.set_mask(full_mask)

    def push_mask(self, mask, unmask_instead = False):
        '''
        Same as mask_rows(), but also pushes the mask to a 'mask stack'.
        Handy for temporary masks e.g. inside loops.
        See also pop_mask().
        '''
        self._mask_stack.append(self.get_mask())
        self.mask_rows(mask, unmask_instead = unmask_instead)

    def pop_mask(self):
        '''
        Pop the topmost mask from the mask stack,
        set previous mask in the stack as current one
        and return the popped mask.
        Raises an exception if trying to pop an empty stack.
        '''
        try:
          previous_mask = self._mask_stack.pop()
        except IndexError as e:
          raise Exception("Trying to pop empty mask stack: %s" % e)

        self.set_mask(previous_mask)
        return previous_mask

    def remove_masked_rows_permanently(self):
        '''
        Removes the currently masked rows permanently.

        This is typically unnecessary, but may be useful
        before adding (cached) virtual columns to
        huge data sets where most rows are masked (because
        the cached virtual columns are computed for
        masked rows as well.)
        '''
        # Removing the real data rows themselves is easy.
        self._data = self._data[~(self._mask),:]
        
        # but we have to also adjust the comment & settings line numbers
        s = np.cumsum(self._mask.astype(np.int))
        def n_masked_before_line(lineno): return s[max(0, min(len(s)-1, lineno-1))]
        self._comments = [ (max(0,lineno-n_masked_before_line(lineno)), comment) for lineno,comment in self._comments ]
        self._settings = [ (max(0,lineno-n_masked_before_line(lineno)), setting) for lineno,setting in self._settings ]

        # as well as remove the masked rows from cached virtual columns.
        # However, _virtual_dims is assumed to be immutable in copy() so
        # we must copy it here!
        old_dims = self._virtual_dims
        self._virtual_dims = {}
        for name, dim in old_dims.iteritems():
          cached_arr = dim['cached_array']
          if isinstance(cached_arr, np.ndarray):
            cached_arr = cached_arr[~(self._mask)]
          elif cached_arr != None:
            cached_arr = [ val for i,val in enumerate(cached_arr) if not self._mask[i] ]
          self._virtual_dims[name] = { 'fn': dim['fn'], 'cached_array': cached_arr }

        # finally remove the obsolete mask(s)
        self._mask = np.zeros(len(self._data), dtype=np.bool)
        self._mask_stack = []

    def get_single_valued_parameter(self, param):
        ''' If all values in the (virtual) dimension "param" are the same, return that value. '''
        assert len(np.unique(self[param])) == 1 or (all(np.isnan(self[param])) and len(self[param]) > 0), \
            '%s is not single valued for the current unmasked rows: %s' % (param, np.unique(self[param]))
        return self[param][0]

    def get_all_single_valued_parameters(self):
        params = OrderedDict()
        for p in self.get_dimensions():
          try: params[p] = self.get_single_valued_parameter(p)
          except: pass
        return params

    def divide_into_sweeps(self, sweep_dimension, use_sweep_direction = None):
        '''
        Divide the rows into "sweeps" based on a changing value of column 'sweep_dimension'
        or based on changing direction of 'sweep_dimension'. If use_sweep_direction is None,
        the method tries to figure out which one is more reasonable.

        Sequences of four or more points with a constant value of 'sweep_dimension' are also
        considered a sweep.
        
        Returns a sequence of tuples indicating the start and end of each sweep.

        Note that the indices are relative to the currently _unmasked_ rows only.
        '''
        sdim = self[sweep_dimension]
        dx = np.sign(sdim[1:] - sdim[:-1])

        if use_sweep_direction == None:
          use_sweep_direction = ( np.abs(dx).astype(np.int).sum() > len(dx)/4. )

        if use_sweep_direction:
          logging.info("Assuming '%s' is swept." % sweep_dimension)
        else:
          logging.info("Assuming '%s' stays constant within a sweep." % sweep_dimension)

        if use_sweep_direction:
          for i in range(1,len(dx)):
              if i+1 < len(dx) and dx[i] == 0: dx[i]=dx[i+1] # this is necessary to detect changes in direction, when the end point is repeated
          change_in_sign = (2 + np.array(np.where(dx[1:] * dx[:-1] < 0),dtype=np.int).reshape((-1))).tolist()

          # the direction changing twice in a row means that sweeps are being done repeatedly
          # in the same direction.
          for i in range(len(change_in_sign)-1, 0, -1):
            if change_in_sign[i]-change_in_sign[i-1] == 1: del change_in_sign[i]

          if len(change_in_sign) == 0: return np.array([[0, len(sdim)]])

          start_indices = np.concatenate(([0], change_in_sign))
          stop_indices  = np.concatenate((change_in_sign, [len(sdim)]))

          sweeps = np.concatenate((start_indices, stop_indices)).reshape((2,-1)).T
        else:
          change_in_sdim = 1 + np.array(np.where(dx != 0)).reshape((-1))
          if len(change_in_sdim) == 0: return np.array([[0, len(sdim)]])

          start_indices = np.concatenate(([0], change_in_sdim))
          stop_indices  = np.concatenate((change_in_sdim, [len(sdim)]))
        
          sweeps = np.concatenate((start_indices, stop_indices)).reshape((2,-1)).T

        return sweeps

    def mask_sweeps(self, sweep_dimension, sl, unmask_instead=False):
        '''
        Mask entire sweeps (see divide_into_sweeps()).

        sl can be a single integer or any slice object compatible with a 1D numpy.ndarray (list of sweeps).

        unmask_instead -- unmask the specified sweeps instead, mask everything else
        '''
        sweeps = self.divide_into_sweeps(sweep_dimension)
        row_mask = np.zeros(len(self[sweep_dimension]), dtype=np.bool)
        for start,stop in ([sweeps[sl]] if isinstance(sl, int) else sweeps[sl]):
            logging.debug("%smasking start: %d, stop %d" % ('un' if unmask_instead else '',start, stop))
            row_mask[start:stop] = True
        self.mask_rows(~row_mask if unmask_instead else row_mask)


    def unmask_sweeps(self, sweep_dimension, sl):
        '''
        Mask all rows except the specified sweeps (see divide_into_sweeps()).

        sl can be a single integer or any slice object compatible with a 1D numpy.ndarray (list of sweeps).
        '''
        self.mask_sweeps(sweep_dimension, sl, unmask_instead=True)


    def get_data(self, deep_copy=False):
        '''
        Get the non-masked data as a 2D ndarray.

        kwargs:
          deep_copy -- copy the returned data so that it is safe to modify it.
        '''
        d = self._data[~(self._mask)]
        if deep_copy: d = d.copy()
        return d

    def get_column(self, name, deep_copy=False):
        '''
        Get the non-masked entries of dimension 'name' as a 1D ndarray.
        name is the dimension name.

        kwargs:
          deep_copy -- copy the returned data so that it is safe to modify it.
        '''
        if name in self._virtual_dims.keys():
            d = self._virtual_dims[name]['cached_array']
            if d is None: d = self._virtual_dims[name]['fn'](self)
            if len(d) == len(self._mask): # The function may return masked or unmasked data...
              # The function returned unmasked data so apply the mask
              try:
                d = d[~(self._mask)] # This works for ndarrays
              except:
                # workaround to mask native python arrays
                d = [ x for i,x in enumerate(d) if not self._mask[i] ]
            return d
        else:
            d = self._data[~(self._mask),self._dimension_indices[name]]
        
        if deep_copy: d = d.copy()
        return d

    non_numpy_array_warning_given = []
    def add_virtual_dimension(self, name, fn=None, arr=None, comment_regex=None, from_set=None, cache_fn_values=True, return_result=False):
        '''
        Makes a computed vector accessible as self[name].
        The computed vector depends on whether fn, arr or comment_regex is specified.

        It is advisable that the computed vector is of the same length as
        the real data columns.
        
        kwargs:
          fn            -- the function applied to the DataView object, i.e self[name] returns fn(self)
          arr           -- specify the column directly as an array, i.e. self[name] returns arr
          comment_regex -- for each row, take the value from the last match in a comment, otherwise np.NaN.
                           Can be just a regex or a (regex, dtype) tuple.
          from_set      -- for each row, take the value from the corresponding .set file. Specify as
                           a tuple ("instrument_name", "parameter_name", dtype=np.float).

          cache_fn_values -- evaluate fn(self) immediately for the entire (unmasked) array and cache the result
          return_result   -- return the result directly as an (nd)array instead of adding it as a virtual dimension
        '''
        logging.debug('adding virtual dimension "%s"' % name)

        assert (fn != None) + (arr is not None) + (comment_regex != None) + (from_set != None) == 1, 'You must specify exactly one of "fn", "arr", or "comment_regex".'

        if arr is not None:
          assert len(arr) == len(self._mask), '"arr" must be a vector of the same length as the real data columns. If you want to do something fancier, specify your own fn.'

        if from_set != None:
            assert len(from_set) in [2, 3], 'from_set must be a tuple or triple.'
            assert self._settings != None, '.set files were not successfully parsed during dataview initialization.'

        if comment_regex != None or from_set != None:
            # construct the column by parsing the comments or .sets
            use_set = (from_set != None) # shorthand for convenience

            # pre-allocate an array
            if use_set:
              dtype = np.float if len(from_set)<3 else from_set[2]
            else:
              if isinstance(comment_regex, basestring):
                regex = comment_regex
                dtype = np.float
              else:
                regex = comment_regex[0]
                dtype = comment_regex[1]
            try:
              if issubclass(dtype, basestring):
                raise Exception('Do not store strings in numpy arrays (because it "works" but the behavior is unintuitive, i.e. only the first character is stored if you just specify dtype=str).')
              vals = np.zeros(len(self._mask), dtype=dtype)
              if dtype == np.float: vals += np.nan # initialize to NaN instead of zeros
              prev_val = np.nan
            except:
              if not name in self.non_numpy_array_warning_given:
                logging.warn("%s does not seem to be a numpy data type. The virtual column '%s' will be a native python array instead, which may be very slow." % (str(dtype), name))
                self.non_numpy_array_warning_given.append(name)
              vals = [None for jjj in range(len(self._mask))]
              prev_val = None

            prev_match_on_row = 0

            #logging.debug(self._comments)

            for rowno,commentstr in (self._settings if use_set else self._comments):
                if use_set:
                  # simply use the value from the .set file
                  assert from_set[0] in commentstr.keys(), 'Instrument "%s" not found in all .set files.' % from_set[0]
                  assert from_set[1] in commentstr[from_set[0]].keys(), 'Attribute "%s:%s" not found in all .set files.' % (from_set[0], from_set[1])
                  new_val = commentstr[from_set[0]][from_set[1]]
                else:
                  # see if the comment matches the specified regex
                  m = re.search(regex, commentstr)
                  if m == None: continue
                  #logging.debug('Match on row %d: "%s"' % (rowno, commentstr))

                  if len(m.groups()) != 1:
                    logging.warn('Did not get a unique match (%s) in comment (%d): %s'
                                 % (str(groups), rowno, commentstr))
                  new_val = m.group(1)

                try:
                  new_val = dtype(new_val)
                except:
                  logging.exception('Could not convert the parsed value (%s) to the specifed data type (%s).'
                                    % (new_val, dtype))
                  raise

                if isinstance(vals, np.ndarray): vals[prev_match_on_row:rowno] = prev_val
                else: vals[prev_match_on_row:rowno] = ( prev_val for jjj in range(rowno-prev_match_on_row) )
                logging.debug('Setting value for rows %d:%d = %s' % (prev_match_on_row, rowno, prev_val))

                prev_match_on_row = rowno
                prev_val = new_val

            logging.debug('Setting value for (remaining) rows %d: = %s' % (prev_match_on_row, prev_val))
            if isinstance(vals, np.ndarray): vals[prev_match_on_row:] = prev_val
            else: vals[prev_match_on_row:] = ( prev_val for jjj in range(len(vals)-prev_match_on_row) )
            

            return self.add_virtual_dimension(name, arr=vals, return_result=return_result)

        if cache_fn_values and arr is None:
            old_mask = self.get_mask().copy() # backup the mask
            self.clear_mask()
            vals = fn(self)
            self.mask_rows(old_mask) # restore the mask

            return self.add_virtual_dimension(name, arr=vals, cache_fn_values=False, return_result=return_result)

        if return_result:
          return arr
        else:
          self._virtual_dims[name] = {'fn': fn, 'cached_array': arr}

    def remove_virtual_dimension(self, name):
        if name in self._virtual_dims.keys():
            del self._virtual_dims[name]
        else:
            logging.warn('Virtual dimension "%s" does not exist.' % name)

    def remove_virtual_dimensions(self):
        self._virtual_dims = {}

    def _parse_settings(self, data):
      '''
      Parse a settings file (.set) into a dict (instruments) of dicts (settings).

      data must be a qt.Data object.
      '''
      parsed_settings = {}

      try:
        set_path = data.get_settings_filepath()
      except Exception as e:
        logging.exception("Could not determine .set path from '%s'." % str(data))
        raise

      try:
        with open(set_path, 'r') as f:
          settings_string = f.read()
      except Exception as e:
        #logging.exception("Could not load .set file from '%s'." % set_path)
        raise

      try:
        header = re.match(r'(.+?)^Instrument: ', settings_string, flags=(re.MULTILINE | re.DOTALL)).group(1)
        off = len(header)
        parsed_settings['header'] = {'comments' : header}
      except Exception as e:
        logging.exception("Could not parse header in:\n\n%s" % settings_string)
        raise

      try:
        m = re.split(r'^Instrument: (.+?)$', settings_string[off:], flags=(re.MULTILINE | re.DOTALL))
        instrument_names = [ x.replace('\r','').replace('\n',' ').strip() for x in m[1::2] ]
        instrument_attr_strings = m[2::2]
        #logging.debug(instrument_names)
      except Exception as e:
        logging.exception("Could not parse instrument names in:\n\n%s" % settings_string)
        raise

      for instr_name, attr_string in zip(instrument_names, instrument_attr_strings):
        try:
          m = re.split(r'^\s+(.+?): (.+?)$', attr_string, flags=(re.MULTILINE | re.DOTALL))
          attr_names = [ x.replace('\r','').replace('\n',' ').strip() for x in m[1::3] ]
          attr_vals = m[2::3]
          parsed_settings[instr_name] = dict(zip(attr_names, attr_vals))
          #logging.debug("\n")
          #logging.debug(instr_name)
          #logging.debug(attr_names)
        except Exception as e:
          logging.exception("Could not parse atributes for '%s' from:\n\n%s" % (instr_name, attr_string))
          raise

      return parsed_settings

