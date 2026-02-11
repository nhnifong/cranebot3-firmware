"""
Unit tests for common pose functions
"""
import unittest
import numpy as np
import cv2
import copy

import nf_robot.common.definitions as model_constants
from nf_robot.common.cv_common import *
from nf_robot.common.pose_functions import compose_poses, create_lookat_pose
from nf_robot.common.config_loader import load_config
from nf_robot.common.util import *


class TestProjectionAndDetection(unittest.TestCase):

    def setUp(self):
        self.config = load_config('tests/configuration.json')

        # override intrinsics with something simple (Simple pinhole, no distortion)
        W, H = 1920, 1080
        cx, cy = W / 2.0, H / 2.0 # Principal point is exactly center
        self.simple_cal = copy.deepcopy(self.config.camera_cal)
        K = np.array([
            [1000, 0, cx],
            [0, 1000, cy],
            [0, 0, 1]
        ], dtype=float)
        D = np.zeros(5)

        self.simple_cal.intrinsic_matrix = K.flatten().tolist()
        self.simple_cal.distortion_coeff = D.flatten().tolist()

    def tearDown(self):
        pass

    def test_project_pixels_to_floor(self):
        anchor_pose = poseProtoToTuple(self.config.anchors[0].pose)
        anchor_camera_pose = np.array(compose_poses([
            anchor_pose,
            model_constants.anchor_camera,
        ]))

        pixels = np.array([
            (0.5, 0.4),
            (0.5, 0.5),
            (0.5, 0.6),
        ])
        floor_pos = project_pixels_to_floor(pixels, anchor_camera_pose, self.config.camera_cal)
        print(floor_pos)
        assert(floor_pos is not None)
        self.assertEqual((3, 2), floor_pos.shape)

    def test_project_center_pixel_from_corner(self):
        """
        Scenario: Camera is up in a corner (2m, 2m, 2m).
        Target: It is looking directly at the World Origin (0,0,0).
        Test: The CENTER pixel of the image MUST map to (0,0) on the floor.
        """
        
        # override intrinsics with something simple (Simple pinhole, no distortion)
        cam_position = [2.0, 2.0, 2.0] # 2 meters up/out in corner
        target_point = [0.0, 0.0, 0.0] # World Origin
        
        pose = create_lookat_pose(cam_position, target_point)
        center_pixel = (0.5, 0.5) 
        
        floor_pts = project_pixels_to_floor(np.array([center_pixel]), pose, self.simple_cal)
        
        print(f"\nTest Configuration:")
        print(f"  Camera Pos: {cam_position}")
        print(f"  Looking At: {target_point}")
        print(f"  Pixel Input: {center_pixel}")
        print(f"  Projected Floor Point: {floor_pts[0]}")
        
        self.assertIsNotNone(floor_pts, "Projection returned None (ray missed floor?)")
        
        # We expect 0.0, 0.0. Allow small float tolerance.
        np.testing.assert_allclose(
            floor_pts, 
            [target_point[:2]], # (0,0)
            atol=0.01,        # 1cm tolerance
            err_msg="The center pixel did not project to the world origin!"
        )

    def test_project_floor_to_pixels(self):
        cam_position = [2.0, 2.0, 2.0] # 2 meters up/out in corner
        target_point = [0.0, 0.0, 0.0] # World Origin
        
        pose = create_lookat_pose(cam_position, target_point)
        pixels = np.array([
            (0.5, 0.4),
            (0.5, 0.5),
            (0.5, 0.6),
        ])
        
        floor_pts = project_pixels_to_floor(pixels, pose, self.simple_cal)

        # round trip
        out_pixels = project_floor_to_pixels(floor_pts, pose, self.simple_cal)
        np.testing.assert_array_almost_equal(out_pixels, pixels)

    def test_detect_origin_card_in_image(self):
        frame = cv2.imread('tests/origin_card_on_floor.jpg')
        result = locate_markers(frame, self.config.camera_cal)
        seen = set()
        for detection in result:
            seen.add(detection['n'])
        self.assertTrue('origin' in seen)

    def test_detect_origin_card_in_image_gripper_wide(self):
        frame = cv2.imread('tests/gripper_cam.jpg')
        result = locate_markers_gripper(frame, self.config.camera_cal_wide)
        seen = set()
        for detection in result:
            seen.add(detection['n'])
        self.assertTrue('origin' in seen)