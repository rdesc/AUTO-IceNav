import logging
import queue
from multiprocessing import Pipe, Queue

import numpy as np


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
        self._ship_state = None  # ship state information, e.g. (x, y, theta)
        self._obstacles = None   # ice field information
        self._metadata = {}      # metadata such as time stamp

        self.raw_message = None
        self.processed_message = None

        self.shutdown = False
        self.logger = logging.getLogger(__name__ + '.MessageDispatcher')

        if self.transform:
            self.initialize_transform()

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
    def obstacles(self) -> np.ndarray:
        return self._obstacles

    @obstacles.setter
    def obstacles(self, value):
        if value is not None:
            self.logger.info('Received new obstacles: total={}'.format(len(value)))
            value = np.array(value)
            if self.transform:
                self._obstacles = [self.transform(ob, coord='planner') for ob in value]
                self.logger.info('Processed new obstacles: total={}'.format(len(self.obstacles)))
            else:
                self._obstacles = value

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
                self.logger.info('Received new message!')

                if type(self.raw_message) is not dict:
                    raise ValueError

                self.goal = self.raw_message.get('goal', None)
                self.ship_state = self.raw_message.get('ship_state', None)
                self.obstacles = self.raw_message.get('obstacles', None)
                self.metadata = self.raw_message.get('metadata', None)

                self.processed_message = {
                    'goal': np.copy(self.goal),
                    'ship_state': np.copy(self.ship_state),
                    'obstacles': np.copy(self.obstacles),
                    'metadata': self.metadata
                }
                break

            except queue.Empty:
                # nothing in queue so skip
                self.logger.warning('No new information received. Trying again...')
                pass

            except ValueError as err:
                self.logger.warning('Queue closed: {}'.format(err))
                self.shutdown = True
                self.queue.close()

    def send_message(self, path: np.ndarray):
        """
        Sends path to controller [(x1, y1, theta_1),...,(xn, yn, theta_n)]
        """
        if self.pipe:
            if len(path) == 0:
                self.logger.warning('Planned an empty path, skipping...')
                return
            self.logger.info('Sending path...')
            if self.transform:
                path = self.transform(path, coord='world')
            self.pipe.send(path)  # blocking call
