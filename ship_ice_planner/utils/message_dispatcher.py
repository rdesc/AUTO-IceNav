import logging
import queue
from copy import deepcopy
from multiprocessing import Pipe, Queue
from typing import List

import numpy as np

from ship_ice_planner.geometry.utils import get_global_obs_coords


def get_communication_interface(**kwargs):
    if kwargs.get('socket_comm', False):
        from ship_ice_planner.NRC_wrapper.socket_communication import PlannerNRCSocketServer, _HOST, _PORT
        return PlannerNRCSocketServer(_HOST, _PORT, **kwargs)
    else:
        return MessageDispatcher(**kwargs)


class MessageDispatcher:

    def __init__(self, **kwargs):
        """
        Class which stores the relevant multiprocessing objects in order to send/receive messages
        """
        self.pipe: Pipe = kwargs.get('pipe', None)
        self.queue: Queue = kwargs.get('queue', None)
        # class for mapping between planner coordinate frame and world coordinate frame
        # assumes it has a constructor and a method __call__
        self.transform = kwargs.get('transform', None)

        self._goal = None        # goal information
        self._ship_state = None  # ship state information, e.g. (x, y, psi)
        self._obstacles = None   # obstacle vertices in world coordinates
        self._masses = None      # the mass of obstacles
        self._metadata = {}      # metadata such as time stamp

        self.raw_message = None
        self.processed_message = None

        self.shutdown = False
        self.logger = logging.getLogger(__name__ + '.MessageDispatcher')

    def start(self):
        self.logger.info('Starting Message Dispatcher...')
        if self.transform:
            self.initialize_transform()
        if not (self.pipe or self.queue):
            raise ConnectionError('No connection established!!')

    @property
    def goal(self) -> np.ndarray:
        return self._goal

    @goal.setter
    def goal(self, value):
        if value is not None:
            self.logger.info('Received new goal: {}'.format(value))
            value = np.array(value)  # also makes a copy
            if self.transform:
                self._goal = self.transform(value, coord='planner')
                self.logger.info('Processed new goal: {}'.format(self.goal))
            else:
                self._goal = value

    @property
    def ship_state(self) -> np.ndarray:
        return self._ship_state

    @ship_state.setter
    def ship_state(self, value):
        if value is not None:
            self.logger.info('Received new ship state: {}'.format(value))
            value = np.array(value)
            if self.transform:
                self._ship_state = self.transform(value, coord='planner')
                self.logger.info('Processed new ship state: {}'.format(self.ship_state))
            else:
                self._ship_state = value

    @property
    def obstacles(self) -> List[np.ndarray]:
        return self._obstacles

    @obstacles.setter
    def obstacles(self, value):
        if value is not None:
            self.logger.info('Received new obstacles: total={}'.format(len(value)))
            if self.transform:
                obstacles = [self.transform(ob, coord='planner') for ob in value]
            else:
                obstacles = value
            # try separating obstacles
            self._obstacles = obstacles  # [poly for ob in obstacles for poly in separate_polygons(ob)]
            self.logger.info('Processed new obstacles: total={}'.format(len(self.obstacles)))

    @property
    def masses(self) -> List[float]:
        return self._masses  # should be same length as obstacles

    @masses.setter
    def masses(self, value):
        if value is not None:
            self.logger.info('Received new obstacle mass list: total={}'.format(len(value)))
            self._masses = value

    @property
    def metadata(self) -> dict:
        return self._metadata

    @metadata.setter
    def metadata(self, value):
        if value is not None:
            self.logger.info('Received new metadata: {}'.format(value))
            self._metadata = value

    def initialize_transform(self):
        self.logger.info('Initializing transform for planner and world coordinates')
        self.receive_message()
        self.transform = self.transform(self.raw_message)  # call the constructor

    def receive_message(self, timeout=1.0):
        if not self.queue:
            return
        while not self.shutdown:
            try:
                self.raw_message = self.queue.get(block=True, timeout=timeout)  # blocking call
                self.logger.info('\033[96mReceived new message!\033[0m')

                if type(self.raw_message) is not dict:
                    raise ValueError('Received terminating signal!')

                self.goal = self.raw_message.get('goal', None)
                self.ship_state = self.raw_message.get('ship_state', None)
                self.obstacles = self.raw_message.get('obstacles', None)
                self.masses = self.raw_message.get('masses', None)
                self.metadata = self.raw_message.get('metadata', None)

                # hacky... but reduces computation on main process e.g. sim
                if self.obstacles is not None and type(self.obstacles) is tuple:
                    self.obstacles = get_global_obs_coords(*self.obstacles)

                self.processed_message = {
                    'goal': np.copy(self.goal),
                    'ship_state': np.copy(self.ship_state),
                    'obstacles': deepcopy(self.obstacles),
                    'masses': np.copy(self.masses),
                    'metadata': self.metadata
                }
                break

            except queue.Empty:
                # nothing in queue so skip
                self.logger.warning('\033[93mNo new information received. Trying again...\033[0m')
                pass

            except ValueError as err:
                self.logger.warning('\033[93mQueue closed: {}\033[0m'.format(err))
                self.shutdown = True
                self.queue.close()

    def send_message(self, path: np.ndarray):
        """
        Sends path to controller [(x1, y1, psi_1),...,(xn, yn, psi_n)]
        """
        if self.pipe:
            if len(path) == 0:
                self.logger.warning('Planned an empty path, skipping...')
                return
            if self.transform:
                path = self.transform(path, coord='world')
            self.logger.info('\033[92mSending path of size {}...\033[0m'.format(path.shape))
            try:
                self.pipe.send(path)  # blocking call
            except BrokenPipeError:
                self.logger.warning('\033[93mPipe is broken!\033[0m')
                self.shutdown = True

    def close(self):
        if self.pipe:
            self.pipe.close()
            self.queue.close()
        self.logger.info('\033[93mConnection closed!\033[0m')
