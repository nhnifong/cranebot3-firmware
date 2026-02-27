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

class TestWallNormal(unittest.TestCase):
    def setUp(self):
        # A simple 10x10 square room
        self.square_anchors = [
            np.array([0, 0, 3]),
            np.array([10, 0, 3]),
            np.array([10, 10, 3]),
            np.array([0, 10, 3])
        ]

    def test_center_closest_to_bottom_wall(self):
        # Point is at (5, 1), closest to the wall between (0,0) and (10,0)
        p = np.array([5, 1, 1])
        normal = get_inward_wall_normal(p, self.square_anchors)
        # Normal should point straight "up" (+Y direction)
        np.testing.assert_array_almost_equal(normal, [0, 1])

    def test_closest_to_right_wall(self):
        # Point is at (9, 5), closest to the wall between (10,0) and (10,10)
        p = np.array([9, 5, 1])
        normal = get_inward_wall_normal(p, self.square_anchors)
        # Normal should point "left" (-X direction)
        np.testing.assert_array_almost_equal(normal, [-1, 0])

    def test_non_rectangular_quad(self):
        # A diamond/skewed shape
        skewed_anchors = [
            np.array([0, 0, 0]),
            np.array([10, 2, 0]),
            np.array([8, 10, 0]),
            np.array([-2, 8, 0])
        ]
        p = np.array([9, 6, 0]) # Near the right-slanted wall
        normal = get_inward_wall_normal(p, skewed_anchors)
        
        # Ensure it's a unit vector
        self.assertAlmostEqual(np.linalg.norm(normal), 1.0)
        # Ensure it's not strictly axial
        self.assertTrue(normal[0] != 0 and normal[1] != 0)

    def test_outside_right_wall(self):
        # Point is at (15, 5), outside to the right
        p = np.array([15, 5, 0])
        normal = get_inward_wall_normal(p, self.square_anchors)
        # Should point "left" (-X)
        np.testing.assert_array_almost_equal(normal, [-1, 0])