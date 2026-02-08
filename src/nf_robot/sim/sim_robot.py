import asyncio
import argparse
import time
import math
import random
import logging
import websockets
import numpy as np
from dataclasses import dataclass
from typing import List, Optional

from nf_robot.generated.nf import telemetry, common, control

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("SimRobot")

# Constants
ROOM_SIZE_X = 5.0
ROOM_SIZE_Y = 5.0
ANCHOR_HEIGHT = 2.5
GRIPPER_OFFSET_Z = 0.53
UPDATE_RATE_HZ = 30
DT = 1.0 / UPDATE_RATE_HZ
ROBOT_ID = 'simulated_robot_1'
INACTIVITY_TIMEOUT_SEC = 60.0

# Define bounds as numpy arrays for vector clamping
MIN_BOUNDS = np.array([-ROOM_SIZE_X / 2.0, -ROOM_SIZE_Y / 2.0, 0.0])
MAX_BOUNDS = np.array([ROOM_SIZE_X / 2.0, ROOM_SIZE_Y / 2.0, ANCHOR_HEIGHT])

def euler_to_rodrigues(roll_deg, pitch_deg, yaw_deg):
    """
    Convert Euler angles (degrees) to Rodrigues rotation vector.
    Rodrigues vector r: direction is axis of rotation, magnitude is angle in radians.
    """
    # Convert to radians
    roll = np.radians(roll_deg)
    pitch = np.radians(pitch_deg)
    yaw = np.radians(yaw_deg)

    # Quaternion components
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    # Convert Quaternion to Axis-Angle (Rodrigues)
    sin_half_theta_sq = x*x + y*y + z*z

    if sin_half_theta_sq < 1e-7:
        return np.array([0.0, 0.0, 0.0])

    sin_half_theta = np.sqrt(sin_half_theta_sq)
    theta = 2.0 * np.arctan2(sin_half_theta, w)

    scale = theta / sin_half_theta
    return np.array([x * scale, y * scale, z * scale])

@dataclass
class RobotState:
    # Gantry kinematics (Origin is center of room, z=0 is floor)
    pos: np.ndarray # [x, y, z]
    vel: np.ndarray # [x, y, z]
    target_vel: np.ndarray # [x, y, z]
    
    # Gripper state
    wrist_angle: float = 0.0
    finger_angle: float = 0.0 # -90 to 90
    
    last_update: float = 0.0
    last_control_time: float = 0.0
    is_sleeping: bool = False

    def __init__(self):
        self.pos = np.array([0.0, 0.0, 1.0])
        self.vel = np.array([0.0, 0.0, 0.0])
        self.target_vel = np.array([0.0, 0.0, 0.0])
        self.last_update = time.time()
        self.last_control_time = time.time()

class SimulatedRobot:
    def __init__(self, websocket):
        self.websocket = websocket
        self.state = RobotState()
        self.tasks: List[asyncio.Task] = []
        self._running = False

    def _get_anchor_poses(self):
        """
        Define 4 anchors in corners of a 5m square room, 2.5m high.
        Origin is center of room (0,0).
        Rotated along Z to look outwards (previous inward angle + 180).
        """
        poses = []
        
        # Corner definitions: (x, y) -> facing angle (degrees)
        # Forward is +Y (0 deg).
        # Previous angles: -45, 45, 135, 225.
        # New angles: +180 to each.
        
        corners = [
            (-2.5, -2.5, -45 + 180), # Anchor 0: Bottom-Left
            (2.5, -2.5, 45 + 180),   # Anchor 1: Bottom-Right
            (2.5, 2.5, 135 + 180),   # Anchor 2: Top-Right
            (-2.5, 2.5, 225 + 180)   # Anchor 3: Top-Left
        ]

        for x, y, deg in corners:
            # Assuming Z-up coordinate system where rotation is around Z
            rotation = euler_to_rodrigues(0, 0, deg)
            
            pos = common.Vec3(x=x, y=y, z=ANCHOR_HEIGHT)
            poses.append(common.Pose(
                position=pos, 
                rotation=common.Vec3(x=rotation[0], y=rotation[1], z=rotation[2])
            ))
            
        return telemetry.AnchorPoses(poses=poses)

    async def _simulate_component_connection(self, is_gripper, anchor_num=0):
        """
        Simulates the connection lifecycle of a component:
        Connecting (2s) -> Connected (Websocket) -> Connected (Video).
        """
        # Phase 1: Connecting (2 seconds)
        status_msg = telemetry.ComponentConnStatus(
            is_gripper=is_gripper,
            anchor_num=anchor_num if not is_gripper else 0,
            websocket_status=telemetry.ConnStatus.CONNECTING,
            video_status=telemetry.ConnStatus.NOT_DETECTED,
            ip_address="192.168.1.10" + str(anchor_num if not is_gripper else 9),
            gripper_model=telemetry.GripperModel.PILOT if is_gripper else None
        )
        
        update = telemetry.TelemetryBatchUpdate(
            robot_id=ROBOT_ID,
            updates=[telemetry.TelemetryItem(
                component_conn_status=status_msg,
                retain_key=f"conn_status_{'gripper' if is_gripper else f'anchor_{anchor_num}'}"
            )]
        )
        
        try:
            await self.websocket.send(bytes(update))
        except Exception as e:
            logger.error(f"Failed to send connecting status: {e}")
            return

        await asyncio.sleep(3.0)

        # Phase 2: Websocket Connected
        status_msg.websocket_status = telemetry.ConnStatus.CONNECTED
        update.updates[0].component_conn_status = status_msg
        
        try:
            await self.websocket.send(bytes(update))
        except Exception as e:
            logger.error(f"Failed to send connected status: {e}")
            return
            
        await asyncio.sleep(3.0) # visible delay before video comes online

        # Phase 3: Video Connected
        status_msg.video_status = telemetry.ConnStatus.CONNECTED
        update.updates[0].component_conn_status = status_msg
        
        try:
            await self.websocket.send(bytes(update))
        except Exception as e:
            logger.error(f"Failed to send video status: {e}")

    async def _physics_loop(self):
        """
        Main loop producing 30fps telemetry updates.
        """
        self.state.last_update = time.time()
        
        while self._running:
            now = time.time()
            
            # Check for inactivity sleep
            if now - self.state.last_control_time > INACTIVITY_TIMEOUT_SEC:
                if not self.state.is_sleeping:
                    logger.info("Entering sleep mode due to inactivity")
                    self.state.is_sleeping = True
                await asyncio.sleep(0.5)
                continue
            
            # If we just woke up, reset the delta time to avoid a physics jump
            if self.state.is_sleeping:
                logger.info("Waking up from sleep")
                self.state.is_sleeping = False
                self.state.last_update = now

            dt_actual = now - self.state.last_update
            self.state.last_update = now

            # Update Gantry Position
            # Simple Euler integration with vector operations
            self.state.vel += (self.state.target_vel - self.state.vel) * 0.1 # Simple smoothing
            self.state.pos += self.state.vel * dt_actual

            # Clamp to room boundaries
            self.state.pos = np.clip(self.state.pos, MIN_BOUNDS, MAX_BOUNDS)
            
            # Position Estimate
            # Gripper is 53cm below gantry in arpeggio configuration
            gripper_pos_arr = self.state.pos - np.array([0, 0, GRIPPER_OFFSET_Z])
            gripper_rot = euler_to_rodrigues(0, 0, self.state.wrist_angle)
            
            pos_est = telemetry.PositionEstimate(
                gantry_position=common.Vec3(x=self.state.pos[0], y=self.state.pos[1], z=self.state.pos[2]),
                gantry_velocity=common.Vec3(x=self.state.vel[0], y=self.state.vel[1], z=self.state.vel[2]),
                gripper_pose=common.Pose(
                    position=common.Vec3(x=gripper_pos_arr[0], y=gripper_pos_arr[1], z=gripper_pos_arr[2]),
                    rotation=common.Vec3(x=gripper_rot[0], y=gripper_rot[1], z=gripper_rot[2])
                ),
                data_ts=now,
                slack=[False, False, False, False]
            )

            # Position Factors
            # Add noise to "real" position for visual
            noise_level = 0.02
            noise = np.random.uniform(-noise_level, noise_level, 3)
            vis_pos_arr = self.state.pos + noise
            
            pos_factors = telemetry.PositionFactors(
                visual_pos=common.Vec3(x=vis_pos_arr[0], y=vis_pos_arr[1], z=vis_pos_arr[2]),
                visual_vel=common.Vec3(x=self.state.vel[0], y=self.state.vel[1], z=self.state.vel[2]),
                hanging_pos=common.Vec3(x=self.state.pos[0], y=self.state.pos[1], z=self.state.pos[2]), # Ideal hanging
                hanging_vel=common.Vec3(x=self.state.vel[0], y=self.state.vel[1], z=self.state.vel[2])
            )

            # Gripper Sensors
            # Calculate real range to floor (z=0)
            range_to_floor = max(0.0, (self.state.pos[2] - GRIPPER_OFFSET_Z))
            
            # Simulate pressure only if close to floor and gripper is closed
            simulated_pressure = 0.0
            if self.state.finger_angle > 45 and range_to_floor < 0.1:
                simulated_pressure = random.uniform(0.5, 1.5)

            grip_sensors = telemetry.GripperSensors(
                range=range_to_floor + random.uniform(-0.005, 0.005), # Range from palm to floor
                angle=self.state.finger_angle,
                pressure=simulated_pressure, 
                wrist=self.state.wrist_angle
            )

            # Send backCommanded Velocity
            # The velocity commanded after any clamping or alteration
            cmd_vel = telemetry.CommandedVelocity(
                velocity=common.Vec3(x=self.state.target_vel[0], y=self.state.target_vel[1], z=self.state.target_vel[2])
            )

            # Construct Batch Update
            batch = telemetry.TelemetryBatchUpdate(
                robot_id=ROBOT_ID,
                updates=[
                    telemetry.TelemetryItem(pos_estimate=pos_est, retain_key="pos_estimate"),
                    telemetry.TelemetryItem(pos_factors_debug=pos_factors),
                    telemetry.TelemetryItem(grip_sensors=grip_sensors, retain_key="grip_sensors"),
                    telemetry.TelemetryItem(last_commanded_vel=cmd_vel, retain_key="cmd_vel")
                ]
            )

            try:
                await self.websocket.send(bytes(batch))
            except Exception as e:
                logger.error(f"Send failed in physics loop: {e}")
                # If sending fails, we assume connection might be broken or closing
                break

            await asyncio.sleep(DT)

    async def _receive_loop(self):
        """
        Listens for ControlBatchUpdate messages and updates state.
        """
        async for message in self.websocket:
            if not self._running:
                break

            # Reset timeout timer on any message
            self.state.last_control_time = time.time()
            
            # Letting exceptions propagate if parsing fails
            try:
                batch = control.ControlBatchUpdate().parse(message)
                
                for item in batch.updates:
                    # betterproto2 exposes oneofs as attributes. Only one will be non-none
                    
                    if item.command:
                        cmd = item.command
                        logger.info(f"Received CommonCommand: {cmd.name}")
                        if cmd.name == control.Command.STOP_ALL:
                            logger.info("STOPPING ALL MOTION")
                            self.state.target_vel = np.array([0.0, 0.0, 0.0])
                            self.state.vel = np.array([0.0, 0.0, 0.0])

                    elif item.move:
                        move = item.move
                        logger.info(f"Received Move: {move}")
                        
                        # Update velocity if either direction or speed is provided.
                        
                        if move.speed is not None and move.speed == 0.0:
                            # Explicit stop command
                            self.state.target_vel = np.array([0.0, 0.0, 0.0])
                            
                        elif move.direction is not None:
                            dir_vec = np.array([move.direction.x, move.direction.y, move.direction.z])
                            mag = np.linalg.norm(dir_vec)
                            
                            if mag > 0:
                                if move.speed is not None:
                                    # Speed provided: normalize direction and scale
                                    self.state.target_vel = (dir_vec / mag) * move.speed
                                else:
                                    # Speed not provided: direction is velocity
                                    self.state.target_vel = dir_vec
                            else:
                                # Direction is (0,0,0) -> Stop
                                pass 

                        # Update Finger
                        if move.finger is not None:
                            self.state.finger_angle = move.finger
                            
                        # Update Wrist
                        if move.wrist is not None:
                            self.state.wrist_angle = move.wrist

                    else:
                        # Log other commands but do nothing
                        logger.info(f"Ignored control command item: {item}")
            except Exception as e:
                logger.error(f"Error handling message: {e}")
                import traceback
                traceback.print_exc()

    async def start_robot(self):
        """
        Initializes the robot simulation, starts background tasks, and waits until
        shutdown is called or the connection closes.
        """
        self._running = True
        logger.info("Starting SimulatedRobot...")

        # 1. Send Anchor Poses immediately
        anchor_poses_msg = self._get_anchor_poses()
        init_update = telemetry.TelemetryBatchUpdate(
            robot_id=ROBOT_ID,
            updates=[telemetry.TelemetryItem(new_anchor_poses=anchor_poses_msg, retain_key="anchor_poses")]
        )
        try:
            await self.websocket.send(bytes(init_update))
            logger.info("Sent initial AnchorPoses.")
        except Exception as e:
            logger.error(f"Failed to send initial poses: {e}")
            return

        # 2. Start Connection Simulation Tasks (Background)
        # 4 Anchors
        for i in range(4):
            # Stagger startup slightly
            await asyncio.sleep(1.0)
            self.tasks.append(asyncio.create_task(self._simulate_component_connection(is_gripper=False, anchor_num=i)))
        # Gripper
        self.tasks.append(asyncio.create_task(self._simulate_component_connection(is_gripper=True)))

        # 3. Start Loops
        self.tasks.append(asyncio.create_task(self._physics_loop()))
        self.tasks.append(asyncio.create_task(self._receive_loop()))

        # 4. Wait for loops
        try:
            await asyncio.gather(*self.tasks)
        except asyncio.CancelledError:
            logger.info("Robot tasks cancelled.")
        except Exception as e:
            logger.error(f"Unexpected error in robot tasks: {e}")
        finally:
            self._running = False

    async def shutdown(self):
        """
        Gracefully shuts down the robot tasks.
        """
        logger.info("Shutting down SimulatedRobot...")
        self._running = False
        for task in self.tasks:
            task.cancel()
        
        # Allow tasks to clean up
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)

async def main():
    parser = argparse.ArgumentParser(description="Simulated Cable Robot")
    parser.add_argument("server_address", help="WebSocket server address (e.g., ws://localhost:8080)")
    args = parser.parse_args()

    uri = f'{args.server_address}/telemetry/{ROBOT_ID}'
    logger.info(f"Connecting to {uri}...")

    robot: Optional[SimulatedRobot] = None

    try:
        async with websockets.connect(uri) as websocket:
            logger.info("Connected to server.")
            
            robot = SimulatedRobot(websocket)
            await robot.start_robot()
            
    except websockets.exceptions.ConnectionClosed:
        logger.info("Connection closed by server.")
    except Exception as e:
        logger.error(f"Connection error: {e}")
    finally:
        if robot:
            await robot.shutdown()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Simulation stopped by user (KeyboardInterrupt).")