import numpy as np
from pathlib import Path
import uuid
from nf_robot.generated.nf import common, config as nf_config

DEFAULT_CONFIG_PATH = Path(__file__).parent / 'configuration.json'

# Anchors
# Defaults based on a square room setup, pointing towards center.
anchor_defs = [
    # (num, position_xyz, rotation_rvec_xyz)
    (0, (3.0, 3.0, 2.0),  (0.0, 0.0, -np.pi/4)),    # -45 deg
    (1, (3.0, -3.0, 2.0), (0.0, 0.0, -3*np.pi/4)),  # -135 deg
    (2, (-3.0, 3.0, 2.0), (0.0, 0.0, np.pi/4)),     # 45 deg
    (3, (-3.0, -3.0, 2.0),(0.0, 0.0, 3*np.pi/4)),   # 135 deg
]

def default_arp_anchors():
    anch_list = []
    for i in (0,1):
        anchor = nf_config.Anchor()
        anchor.num = i
        # leaving service_name None is a indicator that this anchor config is a placeholder
        # and no such service has been disovered yet and assigned this anchor number
        pos = anchor_defs[i*2][1]
        rot = anchor_defs[i*2][2]
        eye = anchor_defs[i*2+1][1]
        anchor.pose = common.Pose(
            rotation=common.Vec3(x=rot[0], y=rot[1], z=rot[2]),
            position=common.Vec3(x=pos[0], y=pos[1], z=pos[2]),
        )
        anchor.indirect_line = nf_config.IndirectLine(
            eyelet_pos=common.Vec3(x=eye[0], y=eye[1], z=eye[2]),
            cam_tilt=26,
        )
        anch_list.append(anchor)  
    return anch_list

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

    # Camera Calibration Standard
    config.camera_cal = nf_config.CameraCalibration()
    config.camera_cal.resolution = nf_config.Resolution(
        width=1920, 
        height=1080
    )

    # Default Intrinsic Matrix.
    # calcluated for the standard FOV Raspberry Pi Camera module 3
    # with autofocus set to fixed lens position 0.1
    # Derived by anchoring a known room height against solvePnP output
    # a chessboard calibration has been tried, but results were too far off center due to
    # the difficulty of positioning a large enough chessboard in the room.
    intrinsic_np = np.array([
        [1424.,    0., 960.],
        [   0., 1424., 540.],
        [   0.,    0.,   1.]
    ])
    config.camera_cal.intrinsic_matrix = intrinsic_np.flatten().tolist()

    # Default Distortion Coefficients
    distortion_np = np.array([ 0.0115842, 0.18723804, -0.00126164, 0.00058383, -0.38807272])
    config.camera_cal.distortion_coeff = distortion_np.flatten().tolist()

    # Camera Calibration Wide
    # Chessboard calibration of the 684x384 full-sensor-FOV wide camera stream (the whole
    # 16:9 field of view, no longer a center crop). Copied from conf_playroom_special.json.
    config.camera_cal_wide = nf_config.CameraCalibration()
    config.camera_cal_wide.resolution = nf_config.Resolution(width=684, height=384)
    intrinsic_np = np.array([
        [439.31834658631243,   0.,                342.],
        [  0.,                461.5621083718772,  192.],
        [  0.,                  0.,                 1.]
    ])
    config.camera_cal_wide.intrinsic_matrix = intrinsic_np.flatten().tolist()
    distortion_np = np.array([-0.026228587204545444, -0.012309725227594465, -0.00033204923591180567, 0.0015432535264626682, 0.10759316594344916])
    config.camera_cal_wide.distortion_coeff = distortion_np.flatten().tolist()

    # Gripper
    config.gripper = nf_config.Gripper()
    config.gripper.frame_room_spin = (50.0 / 180.0) * np.pi
    
    # Preferred Cameras
    config.preferred_cameras = [0, 1]
    
    # Miscelleneous anchor vars
    config.max_accel = 0.3
    config.rec_mod = 1
    config.running_ws_delay = 0.03

    # Swing cancellation
    config.swing_latency = 0.18 # seconds

    # tension safety
    config.max_safe_tension = 18 # newtons.

    # last known gantry position, lerobot history and route source/destination
    config.last_gantry_pos = common.Vec3(x=0, y=0, z=0)
    config.last_lerobot_policy = ""
    config.last_lerobot_dataset_repo_id = ""
    config.last_lerobot_prompt = ""
    config.last_route_source = common.RoutePoint.ALL_TARGETS
    config.last_route_destination = common.RoutePoint.HAMPER

    # pick and place tunables
    config.pick_and_place = nf_config.PickAndPlaceConstants(
        gantry_height_over_target=common.Vec3(x=0, y=0, z=0.9),
        gantry_height_over_dropoff=common.Vec3(x=0, y=0, z=1.1),
        relaxed_open=-7, # Open enough to drop and that fingers cannot be seen in frame
        delay_after_drop=0.6, # long enough that the payload is not visible anymore in the hand
        loop_delay=0.4,
        end_loop_timeout=10,
    )

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
            c = nf_config.StringmanPilotConfig().from_json(f.read())
            if c.camera_cal is None:
                c.camera_cal = create_default_config().camera_cal
            # This version requires the new full-FOV (684x384) wide camera calibration. Older
            # configs hold the 384x384 center-crop intrinsics, so always override whatever wide
            # cal was saved with the current default rather than only filling it in when missing.
            c.camera_cal_wide = create_default_config().camera_cal_wide
            if c.park_data is None:
                c.park_data = nf_config.ParkData()

            # any existing config which had anchors must have had pilot anchors
            if c.anchor_type is None and len(c.anchors) > 0:
                c.anchor_type = common.AnchorType.PILOT

            # Set camera tilt on configs that existed before the field was added
            if c.anchor_type == common.AnchorType.ARPEGGIO:
                for anchor in c.anchors:
                    if anchor.indirect_line.cam_tilt is None:
                        anchor.indirect_line.cam_tilt = 26.0

            if c.max_safe_tension == 0:
                c.max_safe_tension = 16

            # Backfill fields added after this config was first saved.
            default = create_default_config()
            if c.last_gantry_pos is None:
                c.last_gantry_pos = default.last_gantry_pos
            if c.last_route_source == common.RoutePoint.NA:
                c.last_route_source = default.last_route_source
            if c.last_route_destination == common.RoutePoint.NA:
                c.last_route_destination = default.last_route_destination
            if c.pick_and_place is None:
                c.pick_and_place = default.pick_and_place

            return c

            
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