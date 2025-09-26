import asyncio
import grpc
import numpy as np

from .robot_control_service_pb2 import (
    GetObservationRequest, GetObservationResponse,
    TakeActionRequest, TakeActionResponse,
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

        winch = self.ob.datastore.winch_line_record.getLast()[1]
        finger = self.ob.datastore.finger.getLast()
        imu = self.ob.datastore.imu_rotvec.getLast()[1:]
        laser = self.ob.datastore.range_record.getLast()[1:]

        gantpos = self.ob.pe.gant_pos

        response = GetObservationResponse(
            gantry_pos=Point3D(*gantpos),
            winch_length=winch,
            finger_angle=finger[1],
            gripper_imu_rot=Point3D(*imu),
            laser_rangefinder=laser,
            finger_pad_voltage=finger[2],
        )

        image = np.zeros((1920,1080,3), dytpe='uint8')
        response.gripper_cam = NpyImage(
            data=image.tobytes(),
            shape=list(image.shape),
            dtype=str(image.dtype)
        )

        return response

    async def TakeAction(self, request: TakeActionRequest, context) -> TakeActionResponse:
        gantry_goal_pos = np.array([request.gantry_pos.x, request.gantry_pos.y, request.gantry_pos.z])
        winch = request.winch_length
        finger = request.finger_angle
        print(f'gantry_goal_pos={gantry_goal_pos} winch={winch} finger={finger}')

        return TakeActionResponse(success=True)

async def start_robot_control_server(app_state_manager, port='[::]:50051'):
    server = grpc.aio.server()
    add_RobotControlServiceServicer_to_server(RobotControlService(app_state_manager), server)
    server.add_insecure_port(port)
    print(f"gRPC server listening on {port}")
    await server.start()
    return server # just save this and call stop on it when you want to terminate it