# instrument.py, class that mimics visainstrument but communicates over raw TCP
# Joonas govenius <joonas.govenius@aalto.fi>, 2012
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

import threading
import struct
import re
import socket
import time
import logging
import numpy as np


class TCPInstrument():
    """
    Base class for instruments that communicate over raw TCP. Mimics visainstrument.

    Usage:
    <TCPInstrumentInstance>.write(<command>)
    <TCPInstrumentInstance>.ask(<command>)

    """

    def __init__(self, address, tcp_inactive_period = 1., tcp_min_time_between_connections = 0.1):
        '''
            Address must be given as <IPv4-address>:<TCP-port> in base 10 notation, e.g. "10.0.0.1:2550".
            
            tcp_inactive_period determines how long the connection is allowed to remain inactive
            before being closed. Set to < 0 in order to close the connection after each transaction.
        '''
        self.__tcp_lock = threading.Semaphore()
        self.__tcp_close_thread = None
        self.__tcp_connected = False
        self.__tcp_last_used = 0.
        self.__tcp_last_closed = 0.
        
        # period in seconds after which TCP connection is considered inactive and will be closed
        self.__TCP_INACTIVE_PERIOD = np.max(tcp_inactive_period, 0.)
        
        # minimum time period between closing a TCP connection and opening a new one
        self.__MIN_TIME_BETWEEN_CONNECTIONS = tcp_min_time_between_connections
        
        # address must be given as <IPv4-address>:<TCP-port>, e.g. "10.0.0.1:2550".
        self._address = address
        self._tcpdst = re.match(r"^(\d+\.\d+\.\d+\.\d+):(\d+)", address).groups() # parse into IP & port
        if len(self._tcpdst) != 2: raise Exception("Could not parse {0} into an IPv4 address and a port. Should be in format 192.168.1.1:2550.".format(address))

    def ask(self, querystr, end_of_message='\n', wait_time=None):
        self.__tcp_lock.acquire()

        try:
            self.__connect()
            if querystr != None:  self._socket.sendall(querystr)
            if wait_time != None: time.sleep(wait_time)

            reply = self._socket.recv(512)
            while (end_of_message != None) and (end_of_message not in reply):
                reply = reply + self._socket.recv(512)

            if querystr != None:
              logging.debug(__name__ + ' : ' + self._address + ' says ' + reply.replace("\n", " ").replace("\r", " ") + ' in response to ' + querystr.replace("\n", " ").replace("\r", " "))
            else:
              logging.debug(__name__ + ' : ' + self._address + ' says ' + reply.replace("\n", " ").replace("\r", " "))

        except Exception:
            self.__tcp_lock.release()
            raise

        # close connection immediately unless used many times within a short
        # time interval
        t = time.time()
        if self.__tcp_connected:
            if (t - self.__tcp_last_used) >= self.__TCP_INACTIVE_PERIOD:
                # close immediately
                self.close_connection_gracefully()
            elif self.__tcp_close_thread == None:
                # delay closing
                self.__tcp_close_thread = threading.Thread(target=self.__close_inactive_connection, name="gigatronics_delayed_close")
                self.__tcp_close_thread.daemon = True  # a daemon thread doesn't prevent program from exiting
                self.__tcp_close_thread.start()
        
        self.__tcp_last_used = t
        self.__tcp_lock.release()

        return reply

    def read(self):
        return self.ask(None)

    def write(self, cmd):
        self.__tcp_lock.acquire()

        try:
            self.__connect()
            logging.debug(__name__ + ' : telling ' + self._address + ' to ' + cmd)
            self._socket.sendall(cmd)
        except Exception:
            self.__tcp_lock.release()
            raise

        # close connection immediately unless used many times within a short
        # time interval
        t = time.time()
        if self.__tcp_connected:
            if (t - self.__tcp_last_used) >= self.__TCP_INACTIVE_PERIOD:
                # close immediately
                self.close_connection_gracefully()
            elif self.__tcp_close_thread == None:
                # delay closing
                self.__tcp_close_thread = threading.Thread(target=self.__close_inactive_connection, name="gigatronics_delayed_close")
                self.__tcp_close_thread.daemon = True  # a daemon thread doesn't prevent program from exiting
                self.__tcp_close_thread.start()
        
        self.__tcp_last_used = t
        self.__tcp_lock.release()

        return

    def close_connection_gracefully(self):
        '''
        Closes the TCP connection gracefully

        Input:
            None

        Output:
            None
        '''
        if self.__tcp_connected:
            logging.debug(__name__ + ' : closing TCP socket')
            try:
                self._socket.shutdown(socket.SHUT_WR)
                self._socket.settimeout(1.) # timeout in seconds
                self._socket.recv(512)
            except Exception:
                logging.debug(__name__ + ' : failed to gracefully shutdown TCP socket to ' + self._address)
            #time.sleep(0.5)
            self._socket.close()
            self.__tcp_connected = False
            self.__tcp_last_closed = time.time()

    def __connect(self):
        if not (self.__tcp_connected):
            logging.debug(__name__ + ' : opening TCP socket to ' + self._address)

            time_to_wait = self.__MIN_TIME_BETWEEN_CONNECTIONS - (time.time() - self.__tcp_last_closed)
            if time_to_wait > 0.: time.sleep(time_to_wait)

            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            
            # this should cause RST to be sent when socket.close() is called
            l_onoff = 1
            l_linger = 1
            self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack('ii', l_onoff, l_linger))
            
            self._socket.settimeout(3.) # timeout in seconds
            self._socket.connect((self._tcpdst[0], int(self._tcpdst[1])))
            self.__tcp_connected = True

    def __close_inactive_connection(self):
        t0 = time.time()
        while self.__tcp_connected:
            time.sleep(self.__TCP_INACTIVE_PERIOD + .2)
            t1 = time.time()
            self.__tcp_lock.acquire()
            t2 = time.time()
            #logging.debug(__name__ + ' : tcp_connected == ' + str(self.__tcp_connected))
            if self.__tcp_connected and (t2 - self.__tcp_last_used) >= self.__TCP_INACTIVE_PERIOD:
                self.close_connection_gracefully()
                self.__tcp_close_thread = None
            t3 = time.time()
            self.__tcp_lock.release()
            t4 = time.time()
        logging.debug(__name__ + ' : dt1 = {0:0.3f}, dt2 = {0:0.3f}, dt3 = {0:0.3f}, dt4 = {0:0.3f}'.format(t1-t0, t2-t1, t3-t2, t4-t3))
