import sys
import os
# This will let us import files and modules located in the parent directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import numpy as np
from unittest.mock import patch, MagicMock

import calibration

class TestCalibration(unittest.TestCase):
    def setUp(self):
        # Create a dummy dataset for testing
        # 4 anchors, 2 markers ('origin' and 'tag1')
        self.averages = {
            'origin': [
                (np.zeros(3), np.array([0.0, 0.0, 0.0])), # Seen by Anchor 0
                None,
                None,
                None
            ],
            'tag1': [
                (np.zeros(3), np.array([1.0, 0.0, 0.0])), # Seen by Anchor 0
                (np.zeros(3), np.array([2.0, 0.0, 0.0])), # Seen by Anchor 1
                None,
                None
            ]
        }
        
        # Helper to create a flat x vector representing 4 anchors
        # Shape: (4, 2, 3) -> flattened
        # Each anchor is ((rvec), (tvec))
        self.anchors_zero = np.zeros((4, 2, 3))
        self.x_zero = self.anchors_zero.flatten()

    @patch('calibration.compose_poses')
    @patch('calibration.model_constants')
    def test_multi_card_residuals_origin_constraint(self, mock_constants, mock_compose):
        """
        Test that origin markers produce weighted residuals based on distance from [0,0,0].
        """
        # Setup Mocks
        mock_constants.anchor_camera = np.zeros((2, 3))
        
        # Define compose_poses behavior: return the translation part of the marker pose directly
        # to simulate the anchor being at identity.
        # compose_poses signature is list of poses -> returns pose (r, t)
        def side_effect_compose(poses):
            # poses[0] is anchor, poses[1] is const, poses[2] is marker
            # Return marker pose to simulate identity anchor
            return poses[2] 
        
        mock_compose.side_effect = side_effect_compose

        # Execute
        residuals = calibration.multi_card_residuals(self.x_zero, {'origin': self.averages['origin']})

        # Analysis
        # The origin marker is at [0,0,0]. Residual should be 0.
        # But wait, logic is (pos - 0) * 5.0. 
        # If pos is 0, residual is 0.
        
        # Let's change the input to have an error
        # Origin marker seen at [1, 2, 3]
        bad_origin_data = {
            'origin': [(np.zeros(3), np.array([1.0, 2.0, 3.0])), None, None, None]
        }
        
        residuals = calibration.multi_card_residuals(self.x_zero, bad_origin_data)
        
        # Expected: ([1, 2, 3] - [0, 0, 0]) * 5.0
        expected_residuals = np.array([5.0, 10.0, 15.0])
        
        # Note: residuals array will also contain Z-constraints for the anchors (0 deviation)
        # Slicing the first 3 elements which correspond to the marker
        np.testing.assert_array_almost_equal(residuals[:3], expected_residuals)

    @patch('calibration.compose_poses')
    @patch('calibration.model_constants')
    def test_multi_card_residuals_consistency_constraint(self, mock_constants, mock_compose):
        """
        Test that shared markers minimize distance to their centroid.
        """
        mock_constants.anchor_camera = np.zeros((2, 3))
        
        # Scenario: 
        # Anchor 0 sees tag1 at [10, 0, 0]
        # Anchor 1 sees tag1 at [12, 0, 0]
        # Centroid is [11, 0, 0]
        # Errors: [10-11, 0, 0] and [12-11, 0, 0] -> [-1, 0, 0] and [1, 0, 0]
        
        data = {
            'tag1': [
                (np.zeros(3), np.array([10.0, 0.0, 0.0])), 
                (np.zeros(3), np.array([12.0, 0.0, 0.0])),
                None, 
                None
            ]
        }

        def side_effect_compose(poses):
            return poses[2] # Pass through marker pose
        mock_compose.side_effect = side_effect_compose

        residuals = calibration.multi_card_residuals(self.x_zero, data)

        # There are 2 sightings * 3 coords = 6 residuals for the marker
        marker_residuals = residuals[:6]
        
        expected = np.array([-1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(marker_residuals, expected)

    @patch('calibration.optimize.least_squares')
    @patch('calibration.invert_pose')
    @patch('calibration.compose_poses')
    @patch('calibration.model_constants')
    def test_optimize_anchor_poses_success(self, mock_constants, mock_compose, mock_invert, mock_least_squares):
        """
        Test the optimization driver function calls the solver and returns reshaped poses.
        """
        # Setup mocks
        mock_constants.anchor_camera = np.zeros((2, 3))
        mock_compose.return_value = "composed_pose"
        mock_invert.return_value = np.zeros((2, 3)) # Initial guess
        
        # Mock successful solver result
        mock_result = MagicMock()
        mock_result.success = True
        # Result x should be flattened 4 anchors * 6 params
        mock_result.x = np.ones(24) 
        mock_least_squares.return_value = mock_result

        # Execute
        result_poses = calibration.optimize_anchor_poses(self.averages)

        # Verify initial guess construction
        # Should be called 4 times (once per anchor)
        self.assertEqual(mock_invert.call_count, 4)
        
        # Verify solver call
        mock_least_squares.assert_called_once()
        args, kwargs = mock_least_squares.call_args
        self.assertTrue(kwargs['method'] == 'lm')
        
        # Verify return shape (4, 2, 3)
        self.assertEqual(result_poses.shape, (4, 2, 3))
        self.assertTrue(np.all(result_poses == 1)) # Based on our mock result.x

    @patch('calibration.optimize.least_squares')
    @patch('calibration.invert_pose')
    @patch('calibration.compose_poses')
    @patch('calibration.model_constants')
    def test_optimize_anchor_poses_failure(self, mock_constants, mock_compose, mock_invert, mock_least_squares):
        """
        Test that optimization failure returns None.
        """
        mock_invert.return_value = np.zeros((2, 3))
        
        # Mock failed solver result
        mock_result = MagicMock()
        mock_result.success = False
        mock_result.status = -1
        mock_result.message = "Failed to converge"
        mock_least_squares.return_value = mock_result

        # Execute
        result = calibration.optimize_anchor_poses(self.averages)

        # Verify
        self.assertIsNone(result)