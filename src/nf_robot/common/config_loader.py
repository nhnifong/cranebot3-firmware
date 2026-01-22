import numpy as np
from pathlib import Path
import uuid
from nf_robot.generated.nf import common, config as nf_config

DEFAULT_CONFIG_PATH = Path(__file__).parent / 'configuration.json'

def create_default_config() -> nf_config.StringmanPilotConfig:
    """
    Creates a protobuf configuration object populated with reasonable defaults.
    """
    config = nf_config.StringmanPilotConfig()
    # provision a random ID
    # once the robot tells the backend what this ID is, it has to stick to it, or the owner may see it disappear from their dashboard
    config.robot_id = str(uuid.uuid4())
    config.has_been_calibrated = False
    config.connect_cloud_telemetry = False

    # Anchors
    # Defaults based on a square room setup, pointing towards center.
    anchor_defs = [
        # (num, position_xyz, rotation_rvec_xyz)
        (0, (3.0, 3.0, 2.0),  (0.0, 0.0, -np.pi/4)),    # -45 deg
        (1, (3.0, -3.0, 2.0), (0.0, 0.0, -3*np.pi/4)),  # -135 deg
        (2, (-3.0, 3.0, 2.0), (0.0, 0.0, np.pi/4)),     # 45 deg
        (3, (-3.0, -3.0, 2.0),(0.0, 0.0, 3*np.pi/4)),   # 135 deg
    ]

    for num, pos, rot in anchor_defs:
        anchor = nf_config.Anchor()
        anchor.num = num
        # leaving service_name None is a indicator that this anchor config is a placeholder
        # and no such service has been disovered yet and assigned this anchor number
        
        # Construct Pose using common.Vec3 for rvec (rotation) and tvec (translation)
        anchor.pose = common.Pose(
            rotation=common.Vec3(x=rot[0], y=rot[1], z=rot[2]),
            position=common.Vec3(x=pos[0], y=pos[1], z=pos[2]),
        )
        config.anchors.append(anchor)
    
    # Camera Calibration 
    config.camera_cal = nf_config.CameraCalibration()
    config.camera_cal.resolution = nf_config.Resolution(
        width=1920, 
        height=1080
    )

    # Default Intrinsic Matrix.
    # This is calibrated for the standard FOV Raspberry Pi Camera module 3
    # with the autofocus set to a fixed lens position of 0.1
    intrinsic_np = np.array([
        [1424.,    0., 960.],
        [   0., 1424., 540.],
        [   0.,    0.,   1.]
    ])
    config.camera_cal.intrinsic_matrix = intrinsic_np.flatten().tolist()

    # Default Distortion Coefficients
    distortion_np = np.array([ 0.0115842, 0.18723804, -0.00126164, 0.00058383, -0.38807272])
    config.camera_cal.distortion_coeff = distortion_np.flatten().tolist()

    # Gripper
    config.gripper = nf_config.Gripper()
    config.gripper.frame_room_spin = (50.0 / 180.0) * np.pi
    
    # Preferred Cameras
    config.preferred_cameras = [0, 3]
    
    # Miscelleneous anchor vars
    config.max_accel = 0.3
    config.rec_mod = 1
    config.running_ws_delay = 0.03

    return config

def save_config(config: nf_config.StringmanPilotConfig, path: Path=DEFAULT_CONFIG_PATH):
    """
    Writes the proto to a JSON file.
    """
    if path is None:
        return
    with open(path, 'w') as f:
        f.write(config.to_json(indent=2))

def load_config(path: Path=DEFAULT_CONFIG_PATH) -> nf_config.StringmanPilotConfig:
    """
    Loads the proto from a JSON file.
    """
    try:
        if path is None:
            raise FileNotFoundError # observer unit test path
        with open(path, 'r') as f:
            print(f'Loaded config from {path}')
            return nf_config.StringmanPilotConfig().from_json(f.read())
            
    except FileNotFoundError:
        print(f"No config found at {path}, creating default.")
        config = create_default_config()
        print(f"New robot id chosen {config.robot_id}.")
        save_config(config, path)
        return config

def config_has_any_address(config: nf_config.StringmanPilotConfig):
    """Return true if this config has the address of at least one component"""
    return any([c.address is not None for c in [config.gripper, *config.anchors]])

if __name__ == "__main__":
    cfg = load_config(DEFAULT_CONFIG_PATH)
    print(f"Loaded config for robot: {cfg.robot_id}")
    print(f"Gripper Spin: {cfg.gripper.frame_room_spin}")