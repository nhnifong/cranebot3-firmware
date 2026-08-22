import asyncio
import numpy as np
from collections import deque
import threading

from nf_robot.host.component_client import ComponentClient
from nf_robot.common.pose_functions import compose_poses
import nf_robot.common.definitions as model_constants
from nf_robot.generated.nf import telemetry, common
from nf_robot.common.cv_common import *
from nf_robot.common.pose_functions  import *
from nf_robot.common.util import *

# looking for cranebot-anchor-arpeggio-service

"""Host-side client for an "Arpeggio" anchor, the 2nd revision of the Stringman anchor.

Each anchor drives two lines and reports both: the direct line from anchor to marker box,
and the indirect line, which runs through a ceramic eyelet on an adjacent wall. The
reference_length handed to a spool controller decides what length it reports back, and for
the indirect line that is eyelet-to-marker-box, so calibration has to locate the eyelet.

The direct drive BLDC motors report continuous torque, not the pilot's binary tight/slack.
"""

class ArpeggioAnchorClient(ComponentClient):
    def __init__(self, address, port, anchor_num, datastore, ob, pool, stat, telemetry_env):
        super().__init__(address, port, datastore, ob, pool, stat, telemetry_env)
        self.anchor_num = anchor_num
        self.conn_status = telemetry.ComponentConnStatus(
            is_gripper=False,
            anchor_num=self.anchor_num,
            websocket_status=telemetry.ConnStatus.NOT_DETECTED,
            video_status=telemetry.ConnStatus.NOT_DETECTED,
            gripper_model=telemetry.GripperModel.ARPEGGIO,
        )
        self.anchor_pose = np.zeros((2, 3))
        self.camera_pose = np.zeros((2, 3))
        self.eye_pos = np.zeros(3)
        self.raw_gant_poses = deque(maxlen=24)
        self.gantry_pos_sightings = deque(maxlen=100)
        self.gantry_pos_sightings_lock = threading.RLock()

        self.updatePoseAndEye(
            poseProtoToTuple(self.config.anchors[anchor_num].pose),
            tonp(self.config.anchors[anchor_num].indirect_line.eyelet_pos),
        )

    async def send_config(self):
        anchor_config_vars = {}
        # TODO
        if len(anchor_config_vars) > 0:
            await self.websocket.send(json.dumps({'set_config_vars': anchor_config_vars}))

    def updatePoseAndEye(self, pose=None, eye=None):
        """Set the anchor's room pose and eyelet position, and rebuild camera_pose from them."""
        if pose is not None:
            self.anchor_pose = pose
        if eye is not None:
            self.eye_pos = eye
        # the model has the camera tilted 22 degrees; config records what it actually is
        extratilt = 22 - self.config.anchors[self.anchor_num].indirect_line.cam_tilt
        self.camera_pose = np.array(compose_poses([
            self.anchor_pose,
            model_constants.arp_anchor_camera,
            (np.array([extratilt/180*np.pi, 0, 0], dtype=float), np.zeros(3, dtype=float)),
        ]))

    async def handle_update_from_ws(self, update):
        if 'spool0' in update: # high spool (direct line)
            self.storeSpoolData(0, update['spool0'])
        if 'spool1' in update: # low spool (indirect line)
            self.storeSpoolData(1, update['spool1'])

        if len(self.gantry_pos_sightings) > 0:
            with self.gantry_pos_sightings_lock:
                self.ob.send_ui(gantry_sightings=telemetry.GantrySightings(
                    sightings=[common.Vec3(*position) for position in self.gantry_pos_sightings]
                ))
                self.gantry_pos_sightings.clear()

    def storeSpoolData(self, spool_no, data):
        """File one spool's [(time, line_length, line_speed, torque), ...] records."""
        line_number = self.anchor_num * 2 + spool_no
        self.datastore.anchor_line_record[line_number].insertList(np.array(data))
        self.datastore.anchor_line_record_event.set()

    def handle_detections(self, detections, timestamp):
        """File one frame's apriltag detections, called back from the detector pool.

        Every pose arrives relative to this anchor's camera and is composed through
        camera_pose into the room frame before anything else sees it.
        """
        self.stat.pending_frames_in_pool -= 1
        self.stat.detection_count += len(detections)

        for detection in detections:
            name = detection['n']
            self.last_known_centers[name] = detection['center']
            self.last_known_half_extents[name] = detection.get('half_extent')

            if name in CAL_MARKERS:
                # kept raw, for calibration to analyse later
                self.origin_poses[detection['n']].append(detection['p'])

            if name == 'gantry':
                pose = np.array(compose_poses([
                    self.camera_pose, # config dependent
                    detection['p'], # the pose obtained just now
                    self.ob.gantry_april_inv, # which marker the gantry has, per pole type
                ]))
                position = pose[1]
                self.datastore.gantry_pos.insert(np.concatenate([[timestamp], [self.anchor_num], position]))
                self.datastore.gantry_pos_event.set()

                self.last_gantry_frame_coords = detection['p'][1]
                with self.gantry_pos_sightings_lock:
                    self.gantry_pos_sightings.append(position)

                if self.save_raw:
                    # capture time included so a consumer can select a window of stillness
                    self.raw_gant_poses.append((timestamp, detection['p']))

            if name in OTHER_MARKERS:
                offset = model_constants.basket_offset_inv if name.endswith('back') else model_constants.basket_offset
                pose = np.array(compose_poses([
                    self.camera_pose,
                    detection['p'],
                    offset, # the named location is out in front of the tag
                ]))
                position = pose.reshape(6)[3:]
                self.ob.update_avg_named_pos(detection['n'], position)


    def process_frame(self, frame_to_encode):
        return frame_to_encode