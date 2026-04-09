import asyncio
import numpy as np
from collections import deque
import threading

from nf_robot.host.anchor_client import ComponentClient
from nf_robot.common.pose_functions import compose_poses
import nf_robot.common.definitions as model_constants
from nf_robot.generated.nf import telemetry, common
from nf_robot.common.cv_common import *
from nf_robot.common.pose_functions  import *
from nf_robot.common.util import *

# looking for cranebot-anchor-arpeggio-service

"""
"Arpeggio" is the codename of the 2nd revision of the Stringman

The new anchors differ drastically from the pilot version. Pairs of lines are combined into units,
So each server will report the length of two different lines. One line spans from the anchor to the marker box directly.
the other passes through a ceramic eyelet on an adjacent wall, referred to as the "indirect line"
What we pass as the reference_length to each spool controller determines the length we get back.
for the indirect line, we pass the length between the ceramic eyelet and the marker box.

Determining the real position of the ceramic eyelet is done during calibration.

These anchors also use a direct drive BLDC motor with built in FOC controller that continuously reports torque rather than
a binary tight/slack value.

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
            gripper_model=nf.telemetry.GripperModel.ARPEGGIO,
        )
        self.anchor_pose = np.zeros((2, 3))
        self.camera_pose = np.zeros((2, 3))
        self.eye_pos = np.zeros(3)
        # TODO inform web frontend of extra tilt.
        self.extratilt = 22 - self.config.anchors[anchor_num].indirect_line.cam_tilt
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

    def updatePoseAndEye(self, pose, eye):
        self.anchor_pose = pose
        self.eye_pos = eye
        self.camera_pose = np.array(compose_poses([
            self.anchor_pose,
            model_constants.arp_anchor_camera,
            (np.array([0,0,self.extratilt/180*np.pi], dtype=float), np.zeros(3, dtype=float)),
        ]))

    async def handle_update_from_ws(self, update):
        if 'spool0' in update:
            self.storeSpoolData(0, update['spool0'])
        if 'spool1' in update:
            self.storeSpoolData(1, update['spool1'])

        if len(self.gantry_pos_sightings) > 0:
            with self.gantry_pos_sightings_lock:
                self.ob.send_ui(gantry_sightings=telemetry.GantrySightings(
                    sightings=[common.Vec3(*position) for position in self.gantry_pos_sightings]
                ))
                self.gantry_pos_sightings.clear()

    def storeSpoolData(self, spool_no, data):
        # data= [(time, line_length, line_speed, torque), ...]
        line_number = self.anchor_num * 2 + spool_no
        self.datastore.anchor_line_record[line_number].insertList(np.array(data))
        self.datastore.anchor_line_record_event.set()

    def handle_detections(self, detections, timestamp):
        """
        handle a list of apriltag detections from the pool
        """
        self.stat.pending_frames_in_pool -= 1
        self.stat.detection_count += len(detections)

        for detection in detections:
            name = detection['n']
            self.last_known_centers[name] = detection['center']

            if name in CAL_MARKERS:
                # save all the detections of the origin for later analysis
                self.origin_poses[detection['n']].append(detection['p'])
                # if detection['n'] == "origin":
                #     print(detection)

            if name == 'gantry':
                # rotate and translate to where that object's origin would be
                # given the position and rotation of the camera that made this observation (relative to the origin)
                # store the time and that position in the appropriate measurement array in observer.
                # you have the pose of gantry_front relative to a particular anchor camera
                # convert it to a pose relative to the origin
                pose = np.array(compose_poses([
                    self.anchor_pose, # obtained from calibration
                    model_constants.arp_anchor_camera, # constant
                    detection['p'], # the pose obtained just now
                    gantry_april_inv, # constant
                ]))
                position = pose.reshape(6)[3:]
                self.datastore.gantry_pos.insert(np.concatenate([[timestamp], [self.anchor_num], position])) # take only the position
                # print(f'Inserted gantry pose ts={timestamp}, pose={pose}')
                self.datastore.gantry_pos_event.set()

                self.last_gantry_frame_coords = detection['p'][1] # second item in pose tuple is position
                with self.gantry_pos_sightings_lock:
                    self.gantry_pos_sightings.append(position)

                if self.save_raw:
                    self.raw_gant_poses.append(detection['p'])

            if name in OTHER_MARKERS:
                offset = model_constants.basket_offset_inv if name.endswith('back') else model_constants.basket_offset
                pose = np.array(compose_poses([
                    self.anchor_pose,
                    model_constants.arp_anchor_camera, # constant
                    detection['p'], # the pose obtained just now
                    offset, # the named location is out in front of the tag.
                ]))
                position = pose.reshape(6)[3:]
                # save the position of this object for use in various planning tasks.
                self.ob.update_avg_named_pos(detection['n'], position)