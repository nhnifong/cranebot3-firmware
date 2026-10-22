import numpy as np

from nf_robot.host.visual_servo import limit_wander


def test_inside_the_limit_the_command_is_untouched():
    lateral, beyond = limit_wander(np.array([0.1, 0.0]), np.array([0.05, 0.0]),
                                   [0.02, 0.0], [0.0, 0.0], limit_m=0.15, tolerance_m=0.02)
    np.testing.assert_allclose(lateral, [0.1, 0.0])
    assert not beyond


def test_at_the_edge_only_the_outward_part_is_dropped():
    lateral, beyond = limit_wander(np.array([0.1, 0.1]), np.array([0.0, 0.05]),
                                   [1.16, 2.0], [1.0, 2.0], limit_m=0.15, tolerance_m=0.02)
    np.testing.assert_allclose(lateral, [0.0, 0.1])
    assert not beyond


def test_at_the_edge_steering_back_inward_is_allowed():
    lateral, _ = limit_wander(np.array([-0.1, 0.0]), np.array([-0.1, 0.0]),
                              [0.15, 0.0], [0.0, 0.0], limit_m=0.15, tolerance_m=0.02)
    np.testing.assert_allclose(lateral, [-0.1, 0.0])


def test_a_target_outside_the_circle_is_flagged():
    _, beyond = limit_wander(np.array([0.1, 0.0]), np.array([0.3, 0.0]),
                             [0.0, 0.0], [0.0, 0.0], limit_m=0.15, tolerance_m=0.02)
    assert beyond
