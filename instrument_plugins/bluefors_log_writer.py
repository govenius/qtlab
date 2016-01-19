# bluefors_log_writer.py class, for writing temperature log files in the Bluefors format
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

import datetime
import pytz
from dateutil import tz
import os
import logging

def write(quantity, channel, value, address='D:/bluefors_logs'):

  def __time_to_datestr(t): return '{0}-{1:02d}-{2:02d}'.format(str(t.year)[-2:], t.month, t.day)

  # construct the file name
  tt = datetime.datetime.now(tz.tzlocal())
  all_data = None
  datestr = __time_to_datestr(tt)
  
  target_dir = os.path.join(address, datestr)
  try:
    if not os.path.exists(target_dir): os.makedirs(target_dir)
  except:
    logging.exception('Failed to write %s%s value %g because "%s" could not be accessed/created.' % (quantity, channel, value, target_dir))
    raise
  
  if quantity in ['kelvin', 'resistance']:
    fname = os.path.join(target_dir, 'CH{0} {1} {2}.log'.format(channel, {'kelvin': 'T',
                                                                          'resistance': 'R'
                                                                          }[quantity], datestr))
  else:
    fname = os.path.join(target_dir, '{0} {1} {2}.log'.format(channel, quantity, datestr))

  
  try:
    with open(fname,'a') as f:
      f.write(tt.strftime('%d-%m-%y,%H:%M:%S,') + ("%.6E\n" % value))
  except Exception as e:
    logging.exception('Failed to write %s%s value %g to log file "%s".' % (quantity, channel, value, fname))
    raise
