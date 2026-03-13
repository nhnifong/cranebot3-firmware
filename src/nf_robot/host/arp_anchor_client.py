import asyncio
import numpy as np

from nf_robot.host.anchor_client import ComponentClient
from nf_robot.common.pose_functions import compose_poses
import nf_robot.common.definitions as model_constants
from nf_robot.generated.nf import telemetry, common

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
    def __init__(self, address, port, datastore, ob, pool, stat, pe, local_telemetry):
        super().__init__(address, port, datastore, ob, pool, stat, local_telemetry)
        self.conn_status = telemetry.ComponentConnStatus(
            is_gripper=False,
            websocket_status=telemetry.ConnStatus.NOT_DETECTED,
            video_status=telemetry.ConnStatus.NOT_DETECTED,
        )
        self.anchor_num = None

        self.pe = pe

    def updatePose(self, pose):
        self.anchor_pose = pose
        self.camera_pose = np.array(compose_poses([
            self.anchor_pose,
            model_constants.double_anchor_camera,
            (np.array([0,0,self.extratilt/180*np.pi], dtype=float), np.zeros(3, dtype=float)),
        ]))

    def updateEyeletPos(self, position):
    	self.eyepos = position

    async def handle_update_from_ws(self, update):
        if 'spool1' in update:
            self.processSpoolData(update['spool1'])
        if 'spool2' in update:
            self.processSpoolData(update['spool2'])

    def processSpoolData(self, data):
    	pass

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
                    model_constants.double_anchor_camera, # constant
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
                    model_constants.double_anchor_camera, # constant
                    detection['p'], # the pose obtained just now
                    offset, # the named location is out in front of the tag.
                ]))
                position = pose.reshape(6)[3:]
                # save the position of this object for use in various planning tasks.
                self.ob.update_avg_named_pos(detection['n'], position)