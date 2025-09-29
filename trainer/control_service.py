import asyncio
import grpc
import numpy as np

from .robot_control_service_pb2 import (
    GetObservationRequest, GetObservationResponse,
    TakeActionRequest, TakeActionResponse,
    GetWandInfoRequest, GetWandInfoResponse,
    Point3D, NpyImage
)
from .robot_control_service_pb2_grpc import RobotControlServiceServicer, add_RobotControlServiceServicer_to_server

class RobotControlService(RobotControlServiceServicer):
    """
    Server for accepting a connection from lerobot to control stringman
    Meant to be run by AsyncObserver
    """

    def __init__(self, app_state_manager):
        """
        Initialize the service with a reference to your application's state manager
        or any other object that provides access to your application's core logic.
        """
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
        gant_vel = self.ob.pe.gant_vel

        response = GetObservationResponse(
            gantry_vel=Point3D(*gant_vel),
            winch_line_speed=winch[2], # index 2 = speed
            finger_angle=finger[1], # index 1 = angle
            gripper_imu_rot=Point3D(*imu),
            laser_rangefinder=laser,
            finger_pad_voltage=finger[2], # index 2 = voltage
        )

        g_image = self.ob.get_last_frame('g')
        if g_image is not None:
            response.gripper_cam = NpyImage(
                data=g_image.tobytes(),
                shape=list(g_image.shape),
                dtype=str(g_image.dtype)
            )

        # prefer whichever camera seems to have the best lighting, but stick to one.
        preferred_anchor = 3
        a_image = self.ob.get_last_frame(preferred_anchor)
        if a_image is not None:
            response.anchor_cam = NpyImage(
                data=a_image.tobytes(),
                shape=list(a_image.shape),
                dtype=str(a_image.dtype)
            )

        return response

    async def TakeAction(self, request: TakeActionRequest, context) -> TakeActionResponse:
        gantry_vel = np.array([request.gantry_vel.x, request.gantry_vel.y, request.gantry_vel.z])
        winch = request.winch_line_speed
        finger = request.finger_angle
        print(f'TakeAction received on grpc channel gantry_goal_pos={gantry_vel} winch={winch} finger={finger}')

        # If AsyncObserver clipped these values to legal limits, return what they were clipped to
        winch, finger = await self.ob.send_winch_and_finger(winch, finger)
        commanded_vel = await self.ob.move_direction_speed(gantry_vel)

        return TakeActionResponse(
            gantry_vel = Point3D(*commanded_vel),
            winch_line_speed = winch,
            finger_angle = finger,
        )

    async def GetWandInfo(self, request: GetWandInfoRequest, context) -> GetWandInfoResponse:
        # wand_vel is the output of a kalman filter continuously estimating position and velocity from apriltag observations.
        wand_vel = self.ob.pe.wand_vel
        response = GetWandInfoResponse(wand_vel=Point3D(*wand_vel))
        return response

async def start_robot_control_server(app_state_manager, port='[::]:50051'):
    server = grpc.aio.server()
    add_RobotControlServiceServicer_to_server(RobotControlService(app_state_manager), server)
    server.add_insecure_port(port)
    print(f"gRPC server listening on {port}")
    await server.start()
    return server # just save this and call stop on it when you want to terminate it