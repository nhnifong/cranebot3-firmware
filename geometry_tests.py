import unittest
import numpy as np
from math import pi
from cv_common import compose_poses, invert_pose, average_pose

def p(l): # make a numpy array of floats out of the given list. for brevity
    return np.array(l, dtype=float)

class TestPoseFunctions(unittest.TestCase):
    def test_invert_pose_position_simple(self):
        # no rotation, move diagonally
        pose = (
            p([0,0,0]),
            p([1,1,1]),
        )
        expected = ( # no rotation, move back by the same amount
            p([0,0,0]),
            p([-1,-1,-1]),
        )
        result = invert_pose(pose)
        np.testing.assert_array_almost_equal(result[0], expected[0]) # rotation
        np.testing.assert_array_almost_equal(result[1], expected[1]) # position

    def test_invert_rotation_simple(self):
        # rotate 1 radian around the x axis, no translation
        pose = (
            p([1,0,0]),
            p([0,0,0]),
        )
        expected = (
            p([-1,0,0]),
            p([0,0,0]),
        )
        result = invert_pose(pose)
        np.testing.assert_array_almost_equal(result[0], expected[0]) # rotation
        np.testing.assert_array_almost_equal(result[1], expected[1]) # position

    def test_invert_turn_pi(self):
        # rotate pi radians around the z axis, move to +1x
        pose = (
            p([0,0,pi]),
            p([1,0,0]),
        )
        expected = (
            p([0,0,pi]), # not negative because if we rotate pi or more degress, i'll spin back the shorter way
            p([1,0,0]), # the starting position is to my +x
        )
        result = invert_pose(pose)
        np.testing.assert_array_almost_equal(result[0], expected[0]) # rotation
        np.testing.assert_array_almost_equal(result[1], expected[1]) # position

    def test_invert_turn_right_angle(self):
        # rotate half pi radians around the z axis, move to +1x
        pose = (
            p([0,0,0.5*pi]),
            p([1,0,0]),
        )
        expected = (
            p([0,0,-0.5*pi]),
            p([0,1,0]), # the starting position is now in my y+ direction
        )
        result = invert_pose(pose)
        np.testing.assert_array_almost_equal(result[0], expected[0]) # rotation
        np.testing.assert_array_almost_equal(result[1], expected[1]) # position

    def test_average_rotation(self):
        # one pose looks left 0.3 radians, one looks right 0.1 radians.
        poses = [
            (p([0,0,-0.3*pi]), p([0,0,0])),
            (p([0,0,0.1*pi]), p([0,0,0])),
        ]
        expected = (
            p([0,0,-0.1*pi]), # on average, they look 0.1 radians left
            p([0,0,0]),
        )
        result = average_pose(poses)
        np.testing.assert_array_almost_equal(result[0], expected[0]) # rotation
        np.testing.assert_array_almost_equal(result[1], expected[1]) # position

    def test_average_position(self):
        poses = [
            (p([0,0,0]), p([0,0,1])), # Move +z 1
            (p([0,0,0]), p([0,1,0])), # Move +y 1
        ]
        expected = (
            p([0,0,0]),
            p([0,0.5,0.5]), # on average, we are at the midpoint of the two locations
        )
        result = average_pose(poses)
        np.testing.assert_array_almost_equal(result[0], expected[0]) # rotation
        np.testing.assert_array_almost_equal(result[1], expected[1]) # position

    def test_average_combine(self):
        poses = [
            (p([0.5*pi,0,0.5*pi]), p([1,1,1])),
            (p([0.3*pi,0,0.3*pi]), p([-2,-2,-2])),
        ]
        expected = (
            p([0.4*pi,0,0.4*pi]),
            p([-0.5,-0.5,-0.5]),
        )
        result = average_pose(poses)
        np.testing.assert_array_almost_equal(result[0], expected[0]) # rotation
        np.testing.assert_array_almost_equal(result[1], expected[1]) # position

    def test_compose_simple(self):
        poses = [ # combine move with a rotation
            (p([0,0,0]), p([1,1,1])),
            (p([0.1,0.1,0.1]), p([0,0,0])),
        ]
        expected = (
            p([0.1,0.1,0.1]),
            p([1,1,1]),
        )
        result = compose_poses(poses)
        np.testing.assert_array_almost_equal(result[0], expected[0]) # rotation
        np.testing.assert_array_almost_equal(result[1], expected[1]) # position

    def test_compose_simple2(self):
        poses = [ # rotate first, then translate
            (p([0,0,0.5*pi]), p([0,0,0])),
            (p([0,0,0]), p([1,0,0])),
        ]
        expected = (
            p([0,0,0.5*pi]),
            p([0,1,0]), # we should be one unit in the +y direction
        )
        result = compose_poses(poses)
        np.testing.assert_array_almost_equal(result[0], expected[0]) # rotation
        np.testing.assert_array_almost_equal(result[1], expected[1]) # position

    def test_compose_turtle(self):
        poses = [ # move one unit along x, turn left, do this 4 times, you should be back where you started
            (p([0,0,0.5*pi]), p([1,0,0])),
            (p([0,0,0.5*pi]), p([1,0,0])),
            (p([0,0,0.5*pi]), p([1,0,0])),
            (p([0,0,0.5*pi]), p([1,0,0])),
        ]
        expected = (
            p([0,0,0]),
            p([0,0,0]),
        )
        result = compose_poses(poses)
        np.testing.assert_array_almost_equal(result[0], expected[0]) # rotation
        np.testing.assert_array_almost_equal(result[1], expected[1]) # position



if __name__ == '__main__':
    unittest.main()