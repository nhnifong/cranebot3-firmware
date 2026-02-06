import asyncio
import argparse
import time
import math
import random
import logging
import websockets
from dataclasses import dataclass

# Importing generated protobufs
# Ensure these are in your python path
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
HALF_ROOM_X = ROOM_SIZE_X / 2.0
HALF_ROOM_Y = ROOM_SIZE_Y / 2.0
ANCHOR_HEIGHT = 2.5
GRIPPER_OFFSET_Z = 0.53
UPDATE_RATE_HZ = 30
DT = 1.0 / UPDATE_RATE_HZ
ROBOT_ID = 'simulated_robot_1'

@dataclass
class RobotState:
    # Gantry kinematics (Origin is now center of room, z=0 is floor)
    pos_x: float = 0.0
    pos_y: float = 0.0
    pos_z: float = 1.0
    vel_x: float = 0.0
    vel_y: float = 0.0
    vel_z: float = 0.0
    
    # Gripper state
    wrist_angle: float = 0.0
    finger_angle: float = 0.0 # -90 to 90
    
    # Simulation internals
    target_vel_x: float = 0.0
    target_vel_y: float = 0.0
    target_vel_z: float = 0.0
    
    last_update: float = 0.0

def euler_to_rodrigues(roll, pitch, yaw):
    """
    Convert Euler angles (radians) to Rodrigues rotation vector (Vec3).
    Rodrigues vector r: direction is axis of rotation, magnitude is angle in radians.
    """
    # 1. Convert Euler to Quaternion (w, x, y, z)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    # 2. Convert Quaternion to Axis-Angle (Rodrigues)
    # Angle theta = 2 * acos(w)
    # Axis v = (x, y, z) / sin(theta/2)
    # Rodrigues r = theta * v
    
    # Calculate magnitude of the vector part (sin(theta/2))
    sin_half_theta_sq = x*x + y*y + z*z
    
    # Avoid division by zero for very small rotations
    if sin_half_theta_sq < 1e-7:
        return common.Vec3(x=0.0, y=0.0, z=0.0)
    
    sin_half_theta = math.sqrt(sin_half_theta_sq)
    
    # Use atan2 for stable angle calculation
    theta = 2.0 * math.atan2(sin_half_theta, w)
    
    scale = theta / sin_half_theta
    
    return common.Vec3(x=x*scale, y=y*scale, z=z*scale)

def get_anchor_poses():
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
        # Convert degrees to radians for yaw
        yaw_rad = math.radians(deg)
        
        # Assuming Z-up coordinate system where rotation is around Z
        rotation = euler_to_rodrigues(0, 0, yaw_rad)
        
        pos = common.Vec3(x=x, y=y, z=ANCHOR_HEIGHT)
        poses.append(common.Pose(position=pos, rotation=rotation))
        
    return telemetry.AnchorPoses(poses=poses)

async def simulate_component_connection(websocket, is_gripper, anchor_num=0):
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
        await websocket.send(bytes(update))
    except Exception as e:
        logger.error(f"Failed to send connecting status: {e}")
        return

    await asyncio.sleep(3.0)

    # Phase 2: Websocket Connected
    status_msg.websocket_status = telemetry.ConnStatus.CONNECTED
    update.updates[0].component_conn_status = status_msg
    
    try:
        await websocket.send(bytes(update))
    except Exception as e:
        logger.error(f"Failed to send connected status: {e}")
        return
        
    await asyncio.sleep(3.0) # visible delay before video comes online

    # Phase 3: Video Connected
    status_msg.video_status = telemetry.ConnStatus.CONNECTED
    update.updates[0].component_conn_status = status_msg
    
    try:
        await websocket.send(bytes(update))
    except Exception as e:
        logger.error(f"Failed to send video status: {e}")

async def physics_loop(websocket, state: RobotState):
    """
    Main loop producing 30fps telemetry updates.
    """
    state.last_update = time.time()
    
    while True:
        now = time.time()
        dt_actual = now - state.last_update
        state.last_update = now

        # Update Gantry Position
        # Simple Euler integration
        state.vel_x += (state.target_vel_x - state.vel_x) * 0.1 # Simple smoothing
        state.vel_y += (state.target_vel_y - state.vel_y) * 0.1
        state.vel_z += (state.target_vel_z - state.vel_z) * 0.1

        state.pos_x += state.vel_x * dt_actual
        state.pos_y += state.vel_y * dt_actual
        state.pos_z += state.vel_z * dt_actual

        # Clamp to room boundaries (Origin at center, so +/- half size)
        state.pos_x = max(-HALF_ROOM_X, min(HALF_ROOM_X, state.pos_x))
        state.pos_y = max(-HALF_ROOM_Y, min(HALF_ROOM_Y, state.pos_y))
        state.pos_z = max(0.0, min(ANCHOR_HEIGHT, state.pos_z))

        # --- Prepare Telemetry ---
        
        # 1. Position Estimate
        # Gripper is 53cm below gantry
        gripper_pos = common.Vec3(x=state.pos_x, y=state.pos_y, z=state.pos_z - GRIPPER_OFFSET_Z)
        
        # Random rotation for gripper to simulate "guessing" if we don't have commanded data
        # Gripper rotated -90 degrees around Y (Pitch = -90)
        gripper_rot = euler_to_rodrigues(0, 0, math.radians(state.wrist_angle))
        
        pos_est = telemetry.PositionEstimate(
            gantry_position=common.Vec3(x=state.pos_x, y=state.pos_y, z=state.pos_z),
            gantry_velocity=common.Vec3(x=state.vel_x, y=state.vel_y, z=state.vel_z),
            gripper_pose=common.Pose(position=gripper_pos, rotation=gripper_rot),
            data_ts=now,
            slack=[False, False, False, False]
        )

        # 2. Position Factors
        # Add noise to "real" position for visual
        noise_level = 0.02
        vis_pos = common.Vec3(
            x=state.pos_x + random.uniform(-noise_level, noise_level),
            y=state.pos_y + random.uniform(-noise_level, noise_level),
            z=state.pos_z + random.uniform(-noise_level, noise_level)
        )
        pos_factors = telemetry.PositionFactors(
            visual_pos=vis_pos,
            visual_vel=common.Vec3(x=state.vel_x, y=state.vel_y, z=state.vel_z),
            hanging_pos=common.Vec3(x=state.pos_x, y=state.pos_y, z=state.pos_z), # Ideal hanging
            hanging_vel=common.Vec3(x=state.vel_x, y=state.vel_y, z=state.vel_z)
        )

        # 3. Gripper Sensors
        # Calculate real range to floor (z=0)
        range_to_floor = max(0.0, (state.pos_z - GRIPPER_OFFSET_Z))
        
        # Simulate pressure only if close to floor and gripper is closed
        simulated_pressure = 0.0
        if state.finger_angle > 45 and range_to_floor < 0.1:
            simulated_pressure = random.uniform(0.5, 1.5)

        grip_sensors = telemetry.GripperSensors(
            range=range_to_floor + random.uniform(-0.005, 0.005), # Range from palm to floor
            angle=state.finger_angle,
            pressure=simulated_pressure, 
            wrist=state.wrist_angle
        )

        # 4. Commanded Velocity
        # The velocity we are actually trying to achieve (target_vel)
        cmd_vel = telemetry.CommandedVelocity(
            velocity=common.Vec3(x=state.target_vel_x, y=state.target_vel_y, z=state.target_vel_z)
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
            await websocket.send(bytes(batch))
        except Exception as e:
            logger.error(f"Send failed in physics loop: {e}")
            break

        await asyncio.sleep(DT)

async def receive_loop(websocket, state: RobotState):
    """
    Listens for ControlBatchUpdate messages and updates state.
    """
    async for message in websocket:
        # Letting exceptions propagate if parsing fails
        try:
            batch = control.ControlBatchUpdate().parse(message)
            
            for item in batch.updates:
                # betterproto2 exposes oneofs as attributes.
                # We check the ones we care about.
                
                if item.command:
                    cmd = item.command
                    logger.info(f"Received CommonCommand: {cmd.name}")
                    if cmd.name == control.Command.STOP_ALL:
                        logger.info("STOPPING ALL MOTION")
                        state.target_vel_x = 0
                        state.target_vel_y = 0
                        state.target_vel_z = 0
                        state.vel_x = 0
                        state.vel_y = 0
                        state.vel_z = 0

                elif item.move:
                    move = item.move
                    logger.info(f"Received Move: {move}")
                    
                    # --- Velocity Update Logic ---
                    # We update velocity if either direction or speed is provided.
                    # This handles:
                    # 1. CombinedMove(direction=Vec3(), speed=0.0) -> Stop
                    # 2. CombinedMove(speed=0.0) -> Stop
                    # 3. CombinedMove(direction=Vec3(1,0,0), speed=1.0) -> Move
                    # 4. CombinedMove(direction=Vec3(1,0,0)) -> Move (direction is velocity)
                    
                    should_update_vel = False
                    new_vel_x, new_vel_y, new_vel_z = 0.0, 0.0, 0.0

                    # Check for explicit stop command (speed=0)
                    if move.speed is not None and move.speed == 0.0:
                        should_update_vel = True
                        # Velocities remain 0.0
                        
                    elif move.direction is not None:
                        should_update_vel = True
                        mag = math.sqrt(move.direction.x**2 + move.direction.y**2 + move.direction.z**2)
                        
                        if mag > 0:
                            if move.speed is not None:
                                # Speed provided: normalize direction and scale
                                scale = move.speed / mag
                                new_vel_x = move.direction.x * scale
                                new_vel_y = move.direction.y * scale
                                new_vel_z = move.direction.z * scale
                            else:
                                # Speed not provided: direction is velocity
                                new_vel_x = move.direction.x
                                new_vel_y = move.direction.y
                                new_vel_z = move.direction.z
                        else:
                            # Direction is (0,0,0) -> Stop
                            # Velocities remain 0.0
                            pass

                    if should_update_vel:
                        state.target_vel_x = new_vel_x
                        state.target_vel_y = new_vel_y
                        state.target_vel_z = new_vel_z
                    
                    # Update Finger
                    if move.finger is not None:
                        state.finger_angle = move.finger
                        
                    # Update Wrist
                    if move.wrist is not None:
                        state.wrist_angle = move.wrist

                else:
                    # Log other commands but do nothing
                    logger.info(f"Ignored control command item: {item}")
        except Exception as e:
            logger.error(f"Error handling message: {e}")
            import traceback
            traceback.print_exc()

async def main():
    parser = argparse.ArgumentParser(description="Simulated Cable Robot")
    parser.add_argument("server_address", help="WebSocket server address (e.g., ws://localhost:8080)")
    args = parser.parse_args()

    uri = f'{args.server_address}/telemetry/{ROBOT_ID}'
    logger.info(f"Connecting to {uri}...")

    async with websockets.connect(uri) as websocket:
        logger.info("Connected to server.")
        
        state = RobotState()
        
        # 1. Send Anchor Poses immediately
        anchor_poses_msg = get_anchor_poses()
        init_update = telemetry.TelemetryBatchUpdate(
            robot_id=ROBOT_ID,
            updates=[telemetry.TelemetryItem(new_anchor_poses=anchor_poses_msg, retain_key="anchor_poses")]
        )
        await websocket.send(bytes(init_update))
        logger.info("Sent initial AnchorPoses.")

        # 2. Start Connection Simulation Tasks (Background)
        conn_tasks = []
        # 4 Anchors
        for i in range(4):
            await asyncio.sleep(1.0)
            conn_tasks.append(asyncio.create_task(simulate_component_connection(websocket, is_gripper=False, anchor_num=i)))
        # 1 Gripper
        conn_tasks.append(asyncio.create_task(simulate_component_connection(websocket, is_gripper=True)))

        # 3. Start Loops
        physics_task = asyncio.create_task(physics_loop(websocket, state))
        receive_task = asyncio.create_task(receive_loop(websocket, state))

        # Wait for loops (they generally run forever until connection drop)
        await asyncio.gather(physics_task, receive_task, *conn_tasks)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Simulation stopped by user.")