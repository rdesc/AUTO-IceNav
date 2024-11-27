import socket
import logging
from copy import deepcopy
from time import sleep
from typing import Callable, List

import numpy as np

from ship_ice_planner.NRC_wrapper.nrc_config import *
from ship_ice_planner.NRC_wrapper.comm import MessageConnection
from ship_ice_planner.NRC_wrapper.planner_message_pb2 import (
    PlannerInputMessage, PlannerResponseMessage, Point2DYawMessage, Point2DMessage, IcePieceMessage
)

_PLANNER_INPUT_MSG_ID = 30
_PLANNER_RESPONSE_MSG_ID = 31
_HOST = '192.168.58.250'
_PORT = 30002

DEG_TO_RAD = np.pi / 180
RAD_TO_DEG = 1 / DEG_TO_RAD


class PlannerNRCSocketServer:
    """
    socket server
    good illustration https://stackoverflow.com/a/27017691
    """

    def __init__(self, host: str, port: int, **kwargs):
        self.host = host
        self.port = port
        self.server_socket = None
        self._connection = None

        self._goal = None        # goal information
        self._ship_state = None  # ship state information, e.g. (x, y, theta)
        self._obstacles = None   # ice field information
        self._metadata = {}      # metadata such as time stamp

        self.raw_message = None
        self.processed_message = None
        self.sent_message = None  # TODO: save sent message

        self.shutdown = False

        self.transform_world_to_planner: Callable = None
        self.transform_planner_to_world: Callable = None

        # option for hard coded goal
        self._fixed_goal = None

        self.logger = logging.getLogger(__name__ + '.ProtobufSocketServer')

    def start(self):
        self.logger.info('Starting Socket Server...')
        self._start()
        self._accept_client()
        self._initialize_transform()

    @property
    def goal(self) -> np.ndarray:
        return self._goal

    @goal.setter
    def goal(self, message):
        # if message is not None:
        #     self.logger.info('Received new goal: {}'.format(message))
        #     self._goal = self.transform_world_to_planner(np.asarray([message.x, message.y]), 2)
        #     self.logger.info('Processed new goal: {}'.format(self._goal))
        self.logger.info('Received new goal: {}'.format(message))
        self._goal = self.transform_world_to_planner(np.asarray(message), 2)
        self.logger.info('Processed new goal: {}'.format(self._goal))

    @property
    def ship_state(self) -> np.ndarray:
        return self._ship_state

    @ship_state.setter
    def ship_state(self, message):
        if message is not None:
            self.logger.info('Received new ship state: {}'.format(message))
            self._ship_state = self.transform_world_to_planner(
                np.asarray([message.x, message.y, (message.yaw * DEG_TO_RAD) % (2 * np.pi)]), 3
            )
            self._ship_state[2] = self.adjust_to_0_2pi(self._ship_state[2])
            self.logger.info('Processed new ship state: {}'.format(self._ship_state))

    @property
    def obstacles(self) -> List[np.ndarray]:
        return self._obstacles

    @obstacles.setter
    def obstacles(self, message):
        if message is not None:
            self.logger.info('Received new obstacles: total={}'.format(len(message)))
            self._obstacles = [
                # message parser on Kevin's side has a bug, so we need to swap x and y
                self.transform_world_to_planner(np.asarray([[pt.y, pt.x] for pt in ob.perimeter]), 2) for ob in message
            ]
            self.logger.info('Processed new obstacles: total={}'.format(len(self._obstacles)))

    @property
    def metadata(self) -> dict:
        return self._metadata

    @metadata.setter
    def metadata(self, message):
        if message is not None:
            self._metadata = {
                'run_id': message.run_id,  # HACK but this is now used to determine when a trial has started and ended
                'timestamp': (message.timestamp_field.seconds, message.timestamp_field.nanos),
                'velocity': message.speed,
                'setpoint': self.transform_world_to_planner(np.asarray([message.goal.x, message.goal.y]), 2)
            }
            self.logger.info('Received new metadata: {}'.format(self._metadata))

    def _start(self):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen()

    def _accept_client(self):
        client_socket, client_address = self.server_socket.accept()
        self._connection = MessageConnection.from_existing_socket(client_socket, client_address)

    def _parse_message(self, body):
        self.raw_message = body
        message = PlannerInputMessage()
        message.ParseFromString(body)

        self.goal = self._fixed_goal  # message.goal         # metres x, y
        self.ship_state = message.start  # metres x, y, radians yaw
        self.obstacles = message.ice_pieces
        self.metadata = message

        self.processed_message = {
            'goal': np.copy(self.goal),
            'ship_state': np.copy(self.ship_state),
            'obstacles': deepcopy(self.obstacles),
            'metadata': self.metadata,
        }

    def _initialize_transform(self):
        self.logger.info('Initializing transforms for planner and NRC coordinate systems...')

        while True:
            if self._connection.data_available():
                msg_id, body = self._connection.receive_raw_message()
                break
            else:
                sleep(0.5)

        if msg_id == _PLANNER_INPUT_MSG_ID:
            self.logger.info('Received new message!')
            message = PlannerInputMessage()
            message.ParseFromString(body)

            if message.start.x < MID_TANK[0]:
                self._fixed_goal = GOAL_UP
                self.transform_world_to_planner = transform_world_to_planner1_fn
                self.transform_planner_to_world = transform_planner_to_world1_fn
            else:
                self._fixed_goal = GOAL_DOWN
                self.transform_world_to_planner = transform_world_to_planner2_fn
                self.transform_planner_to_world = transform_planner_to_world2_fn

        else:
            self.logger.warning('Received message with unknown message type: {}'.format(self.raw_message.message_type))

    def send_message(self, path: np.ndarray):
        if len(path) == 0:
            self.logger.warning('Planned an empty path, skipping...')
            return
        path = self.transform_planner_to_world(path, 3)
        path_message = PlannerResponseMessage(path=[
            Point2DYawMessage(x=x, y=y, yaw=yaw * RAD_TO_DEG) for x, y, yaw in path
        ])
        path_message.timestamp_field.GetCurrentTime()

        self.logger.info('\033[92mSending path of size {}...\033[0m'.format(path.shape))
        self._connection.send_raw_message(_PLANNER_RESPONSE_MSG_ID,
                                          path_message.SerializeToString())  # should have also sent a planning iteration index

    def receive_message(self):
        while True:
            if self._connection.data_available():
                # first in first out order
                while self._connection.data_available():
                    msg_id, body = self._connection.receive_raw_message()
                break
            else:
                self.logger.warning('\033[93mNo new information received. Trying again...\033[0m')
                sleep(0.1)

        if msg_id == _PLANNER_INPUT_MSG_ID:
            self.logger.info('\033[96mReceived new message!\033[0m')
            self._parse_message(body)
        else:
            self.logger.warning('Received message with unknown message type: {}'.format(self.raw_message.message_type))

    def close(self):
        if self._connection:
            self._connection.close()
        if self.server_socket:
            # self.server_socket.shutdown(socket.SHUT_RDWR)
            self.server_socket.close()
        self.logger.info('\033[93mConnection closed!\033[0m')

    @staticmethod
    def adjust_to_0_2pi(angle_radians):
        angle_radians_0_to_2pi = angle_radians % (2 * np.pi)  # Calculate angle in range 0 to 2pi
        if angle_radians_0_to_2pi < 0:
            angle_radians_0_to_2pi += 2 * np.pi  # Ensure positive value within range
        return angle_radians_0_to_2pi

    @staticmethod
    def parse_raw_message(body):
        message = PlannerInputMessage()
        message.ParseFromString(body)
        return message

    @staticmethod
    def parse_obstacles(message, transform_world_to_planner: Callable = None):
        if transform_world_to_planner is not None:
            return [
                transform_world_to_planner(np.asarray([[pt.y, pt.x] for pt in ob.perimeter]), 2) for ob in
                message.ice_pieces
            ]
        else:
            return [
                np.asarray([[pt.y, pt.x] for pt in ob.perimeter]) for ob in message.ice_pieces
            ]

    @staticmethod
    def parse_goal(message, transform_world_to_planner: Callable = None):
        if message.start.x < MID_TANK[0]:
            return GOAL_UP
        else:
            return GOAL_DOWN
        # if transform_world_to_planner is not None:
        #     return transform_world_to_planner(np.asarray(message.goal), 2)
        # else:
        #     return np.asarray(message.goal)

    @staticmethod
    def parse_ship_state(message, transform_world_to_planner: Callable = None):
        if transform_world_to_planner is not None:
            ship_state = transform_world_to_planner(
                np.asarray([message.start.x, message.start.y, (message.start.yaw * DEG_TO_RAD) % (2 * np.pi)]), 3
            )
            ship_state[2] = PlannerNRCSocketServer.adjust_to_0_2pi(ship_state[2])
            return ship_state
        else:
            return np.asarray([message.start.x, message.start.y, (message.start.yaw * DEG_TO_RAD) % (2 * np.pi)])

    @staticmethod
    def get_transform_planner_to_world(message):
        if message.start.x < MID_TANK[0]:
            return transform_matrix_planner_to_world1
        else:
            return transform_matrix_planner_to_world2

    @staticmethod
    def get_transform_world_to_planner(message):
        if message.start.x < MID_TANK[0]:
            return transform_matrix_world_to_planner1
        else:
            return transform_matrix_world_to_planner2


def demo_client():
    sleep(2)

    # client which connects to the server
    client = MessageConnection((_HOST, _PORT))
    client.connect()

    # create a message
    message = PlannerInputMessage()
    message.goal.x = 1
    message.goal.y = 2
    message.start.x = 3
    message.start.y = 4
    message.start.yaw = np.pi * RAD_TO_DEG
    message.timestamp_field.GetCurrentTime()
    message.speed = 0.5
    message.ice_pieces.extend([
        IcePieceMessage(perimeter=[
            Point2DMessage(x=1, y=2),
            Point2DMessage(x=3, y=4),
            Point2DMessage(x=5, y=6),
        ]),
    ])
    print(message)

    count = 0
    try:
        while True:
            print('Client: Sending message...')
            message.timestamp_field.GetCurrentTime()
            client.send_raw_message(_PLANNER_INPUT_MSG_ID, message.SerializeToString())
            if client.data_available():
                msg_id, body = client.receive_raw_message()
                recv_message = PlannerResponseMessage()
                recv_message.ParseFromString(body)
                count += 1
                print('Client: Message received! count=', count, 'message timestamp', recv_message.timestamp_field)
            else:
                print('No new information received. Trying again...')
            sleep(1)

    finally:
        print('Client: closing client connection.')
        client.close()


if __name__ == "__main__":
    # sever client demo
    from multiprocessing import Process
    from ship_ice_planner.utils.utils import setup_logger

    setup_logger(name=__name__)

    client = Process(target=demo_client)
    client.start()

    server = PlannerNRCSocketServer(host='localhost', port=_PORT)
    server.start()

    try:
        while True:
            server.receive_message()
            server.send_message(np.ones((100, 3)))
            sleep(1)

    finally:
        server.close()
        client.join()
        client.terminate()
