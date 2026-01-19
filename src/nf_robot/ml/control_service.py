import asyncio
import grpc
import numpy as np

from .robot_control_service_pb2 import (
    GetObservationRequest, GetObservationResponse,
    TakeActionRequest, TakeActionResponse,
    GetGamepadActionRequest,
    GetEpisodeControlRequest, GetEpisodeControlResponse,
    Point3D
)
from .robot_control_service_pb2_grpc import RobotControlServiceServicer, add_RobotControlServiceServicer_to_server

# prefer whichever camera seems to have the best lighting, but stick to one.
PREFERRED_ANCHOR_0 = 3
PREFERRED_ANCHOR_1 = 2

# positional argument constructor for Point3D proto message
def Point3Dp(x, y, z):
    return Point3D(x=x, y=y, z=z)

class RobotControlService(RobotControlServiceServicer):
    """
    Server for accepting a connection from lerobot to control stringman
    Meant to be run by AsyncObserver
    """

    def __init__(self, app_state_manager):
        self.ob = app_state_manager # the instance of AsyncObserver
        print("RobotControlService initialized.")

    async def GetObservation(self, request: GetObservationRequest, context) -> GetObservationResponse:
        winch = self.ob.datastore.winch_line_record.getLast()
        finger = self.ob.datastore.finger.getLast()
        imu = self.ob.datastore.imu_rotvec.getLast()[1:]
        laser = self.ob.datastore.range_record.getLast()[1:]

        # ob is the instance of AsyncObserver (observer.py)
        # pe is the instance of Positioner2 (position_estimator.py)
        # gant_pos and gant_vel attributes are the output of a kalman filter continuously updated from observations.
        gant_pos = self.ob.pe.gant_pos
        gant_vel = self.ob.pe.gant_vel

        response = GetObservationResponse(
            gantry_pos=Point3Dp(*gant_pos),
            gantry_vel=Point3Dp(*gant_vel),
            winch_line_speed=float(winch[2]), # index 2 = speed
            finger_angle=float(finger[1]), # index 1 = angle
            gripper_imu_rot=Point3Dp(*imu),
            laser_rangefinder=float(laser),
            finger_pad_voltage=float(finger[2]), # index 2 = voltage
            gripper_camera=self.ob.get_last_frame('g'),
            anchor_camera_0=self.ob.get_last_frame(PREFERRED_ANCHOR_0),
            anchor_camera_1=self.ob.get_last_frame(PREFERRED_ANCHOR_1),
        )

        return response

    async def TakeAction(self, request: TakeActionRequest, context) -> TakeActionResponse:
        gantry_vel = np.array([request.gantry_vel.x, request.gantry_vel.y, request.gantry_vel.z])
        winch = request.winch_line_speed
        finger = request.finger_angle
        print(f'TakeAction received on grpc channel gantry_goal_pos={gantry_vel} winch={winch} finger={finger}')

        # Send both commands concurrently before waiting for both.
        # If AsyncObserver clipped these values, the results will be what they were clipped to
        (winch, finger), commanded_vel = await asyncio.gather(
            self.ob.send_winch_and_finger(0, finger),
            self.ob.move_direction_speed(gantry_vel)
        )

        winch=0

        return TakeActionResponse(
            gantry_vel = Point3Dp(*commanded_vel),
            winch_line_speed = float(winch),
            finger_angle = float(finger),
        )

    async def GetGamepadAction(self, request: GetGamepadActionRequest, context) -> TakeActionResponse:
        """
        get the last action that was caused directly by the gamepad.
        This should be post filtering and bounds checking.
        """
        commanded_vel, winch, finger = self.ob.last_gp_action

        return TakeActionResponse(
            gantry_vel = Point3Dp(*commanded_vel),
            winch_line_speed = float(winch),
            finger_angle = float(finger),
        )

    async def GetEpisodeControl(self, request: GetEpisodeControlRequest, context) -> GetEpisodeControlResponse:
        return GetEpisodeControlResponse(events=self.ob.get_episode_control_events())

async def start_robot_control_server(app_state_manager, port='[::]:50051'):
    server = grpc.aio.server()
    add_RobotControlServiceServicer_to_server(RobotControlService(app_state_manager), server)
    server.add_insecure_port(port)
    print(f"gRPC server listening on {port}")
    await server.start()
    return server # just save this and call stop on it when you want to terminate it