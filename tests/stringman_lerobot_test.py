"""
Unit tests for StringmanLeRobot and related utilities in stringman_lerobot.py.

Tests cover:
- Module-level constants (_CAMERA_MODES, _FEED_NAMES)
- StringmanConfig defaults
- StringmanLeRobot initialization and per-mode camera setup
- Observation and action feature dicts
- get_observation() return shape and keys
- _handle_video_ready() feed filtering and thread start
- disconnect() stop-event signaling
- Video frame resize targeting per-feed resolution
- record_until_disconnected / eval_until_disconnected camera_mode forwarding
"""

import threading
import unittest
from unittest.mock import MagicMock, Mock, call, patch

import numpy as np

from nf_robot.generated.nf import common, telemetry
from nf_robot.ml.stringman_lerobot import (
    CHECKPOINT_EVERY,
    DEFAULT_ACTION_SPACE,
    FPS,
    _ACTION_SPACES,
    _CAMERA_MODES,
    _FEED_NAMES,
    StringmanConfig,
    StringmanLeRobot,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_events():
    return {
        "episode_abandon": False,
        "end_recording": False,
        "start": False,
        "stop": False,
    }


def _make_robot(camera_mode="all"):
    """Instantiate StringmanLeRobot without connecting."""
    cfg = StringmanConfig(uri="ws://localhost:4245", camera_mode=camera_mode)
    return StringmanLeRobot(cfg, _make_events())


def _make_robot_ready(camera_mode="all"):
    """Robot ready for get_observation() without a live telemetry feed.

    last_observed_vel is initialized as np.zeros(3) in __init__ and get_observation()
    now indexes it with [0]/[1]/[2], so no extra setup is needed.
    """
    return _make_robot(camera_mode)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestConstants(unittest.TestCase):

    def test_feed_names_cover_all_feeds_used(self):
        all_feeds = set()
        for feeds in _CAMERA_MODES.values():
            all_feeds.update(feeds.keys())
        for feed in all_feeds:
            self.assertIn(feed, _FEED_NAMES, f"Feed {feed} used in _CAMERA_MODES but missing from _FEED_NAMES")

    def test_feed_names_unique(self):
        names = list(_FEED_NAMES.values())
        self.assertEqual(len(names), len(set(names)), "Feed names must be unique")

    def test_gripper_modes_only_contain_feed_0(self):
        self.assertEqual(set(_CAMERA_MODES["gripper_224"].keys()), {0})
        self.assertEqual(set(_CAMERA_MODES["gripper_384"].keys()), {0})

    def test_gripper_floor_modes_contain_feeds_0_and_3(self):
        self.assertEqual(set(_CAMERA_MODES["gripper_floor_224"].keys()), {0, 3})
        self.assertEqual(set(_CAMERA_MODES["gripper_floor_384"].keys()), {0, 3})

    def test_all_mode_contains_four_feeds(self):
        self.assertEqual(set(_CAMERA_MODES["all"].keys()), {0, 1, 2, 3})

    def test_resolutions_are_positive(self):
        for mode, feeds in _CAMERA_MODES.items():
            for feed, (w, h) in feeds.items():
                self.assertGreater(w, 0, f"{mode}/{feed} width must be positive")
                self.assertGreater(h, 0, f"{mode}/{feed} height must be positive")

    def test_224_modes_use_224(self):
        for mode_name in ["gripper_224", "gripper_floor_224"]:
            for feed, (w, h) in _CAMERA_MODES[mode_name].items():
                self.assertEqual(w, 224, f"{mode_name} feed {feed} width")
                self.assertEqual(h, 224, f"{mode_name} feed {feed} height")

    def test_384_modes_use_384_for_gripper(self):
        # "all" intentionally uses a non-square (684x384) gripper feed; "all_square"
        # keeps the square gripper. See camera_mode "new video size format".
        gripper_feed = 0
        for mode_name in ["gripper_384", "gripper_floor_384", "gripper_anchors_384", "all_square"]:
            if gripper_feed in _CAMERA_MODES[mode_name]:
                w, h = _CAMERA_MODES[mode_name][gripper_feed]
                self.assertEqual(w, 384)
                self.assertEqual(h, 384)

    def test_all_mode_gripper_is_684x384(self):
        w, h = _CAMERA_MODES["all"][0]
        self.assertEqual(w, 684)
        self.assertEqual(h, 384)

    def test_all_mode_anchor_cameras_are_960x544(self):
        for feed in [1, 2]:
            w, h = _CAMERA_MODES["all"][feed]
            self.assertEqual(w, 960)
            self.assertEqual(h, 544)

    def test_all_mode_floor_camera_is_512x512(self):
        w, h = _CAMERA_MODES["all"][3]
        self.assertEqual(w, 512)
        self.assertEqual(h, 512)


# ---------------------------------------------------------------------------
# StringmanConfig
# ---------------------------------------------------------------------------

class TestStringmanConfig(unittest.TestCase):

    def test_defaults(self):
        cfg = StringmanConfig(uri="ws://host:4245")
        self.assertEqual(cfg.uri, "ws://host:4245")
        self.assertIsNone(cfg.remote_stream_token)
        self.assertEqual(cfg.camera_mode, "all")

    def test_custom_values(self):
        cfg = StringmanConfig(uri="ws://host:4245", remote_stream_token="tok", camera_mode="gripper_224")
        self.assertEqual(cfg.remote_stream_token, "tok")
        self.assertEqual(cfg.camera_mode, "gripper_224")


# ---------------------------------------------------------------------------
# StringmanLeRobot initialization
# ---------------------------------------------------------------------------

class TestStringmanLeRobotInit(unittest.TestCase):

    def test_invalid_camera_mode_raises(self):
        cfg = StringmanConfig(uri="ws://localhost:4245", camera_mode="nonexistent")
        with self.assertRaises(ValueError):
            StringmanLeRobot(cfg, _make_events())

    def test_camera_specs_match_mode(self):
        for mode, expected_feeds in _CAMERA_MODES.items():
            robot = _make_robot(mode)
            self.assertEqual(robot.camera_specs, expected_feeds, f"mode={mode}")

    def test_locks_created_for_all_feeds(self):
        for mode, expected_feeds in _CAMERA_MODES.items():
            robot = _make_robot(mode)
            self.assertEqual(set(robot.camera_locks.keys()), set(expected_feeds.keys()))
            for lock in robot.camera_locks.values():
                self.assertIsInstance(lock, type(threading.Lock()))

    def test_stop_events_created_for_all_feeds(self):
        for mode, expected_feeds in _CAMERA_MODES.items():
            robot = _make_robot(mode)
            self.assertEqual(set(robot.stop_video_events.keys()), set(expected_feeds.keys()))
            for ev in robot.stop_video_events.values():
                self.assertIsInstance(ev, threading.Event)
                self.assertFalse(ev.is_set())

    def test_last_images_shape_matches_resolution(self):
        for mode, feed_specs in _CAMERA_MODES.items():
            robot = _make_robot(mode)
            for feed, (w, h) in feed_specs.items():
                img = robot.last_images[feed]
                self.assertEqual(img.shape, (h, w, 3), f"mode={mode} feed={feed}")
                self.assertEqual(img.dtype, np.uint8)

    def test_last_images_initialized_to_zeros(self):
        robot = _make_robot("gripper_384")
        self.assertTrue(np.all(robot.last_images[0] == 0))

    def test_video_threads_starts_empty(self):
        robot = _make_robot("all")
        self.assertEqual(robot.video_threads, {})

    def test_is_not_connected_after_init(self):
        robot = _make_robot()
        self.assertFalse(robot.is_connected)


# ---------------------------------------------------------------------------
# Feature dicts
# ---------------------------------------------------------------------------

class TestFeatures(unittest.TestCase):

    def test_cameras_ft_gripper_only_modes(self):
        for mode in ["gripper_224", "gripper_384"]:
            robot = _make_robot(mode)
            self.assertEqual(set(robot._cameras_ft.keys()), {"gripper_camera"})

    def test_cameras_ft_gripper_floor_modes(self):
        for mode in ["gripper_floor_224", "gripper_floor_384"]:
            robot = _make_robot(mode)
            self.assertEqual(set(robot._cameras_ft.keys()), {"gripper_camera", "overhead_camera"})

    def test_cameras_ft_all_mode(self):
        robot = _make_robot("all")
        self.assertEqual(
            set(robot._cameras_ft.keys()),
            {"gripper_camera", "overhead_camera", "anchor_camera_0", "anchor_camera_1"},
        )

    def test_cameras_ft_shape_tuples_are_hwc(self):
        robot = _make_robot("all")
        ft = robot._cameras_ft
        # gripper: 684x384 (non-square in "all" mode; hwc -> (384, 684, 3))
        self.assertEqual(ft["gripper_camera"], (384, 684, 3))
        # floor: 512x512
        self.assertEqual(ft["overhead_camera"], (512, 512, 3))
        # anchors: height=544, width=960
        self.assertEqual(ft["anchor_camera_0"], (544, 960, 3))
        self.assertEqual(ft["anchor_camera_1"], (544, 960, 3))

    def test_cameras_ft_224_mode(self):
        robot = _make_robot("gripper_224")
        self.assertEqual(robot._cameras_ft["gripper_camera"], (224, 224, 3))

    def test_observation_features_contains_camera_keys(self):
        robot = _make_robot("gripper_floor_384")
        obs_ft = robot.observation_features
        self.assertIn("gripper_camera", obs_ft)
        self.assertIn("overhead_camera", obs_ft)

    def test_observation_features_contains_motor_keys(self):
        robot = _make_robot()
        obs_ft = robot.observation_features
        for key in ["vel_x", "vel_y", "vel_z", "wrist_speed", "finger_speed"]:
            self.assertIn(key, obs_ft)

    def test_observation_features_contains_state_keys(self):
        robot = _make_robot()
        obs_ft = robot.observation_features
        expected = [
            "gripper_pos_x", "gripper_pos_y", "gripper_pos_z",
            "gripper_rot_0", "gripper_rot_1", "gripper_rot_2",
            "gripper_rot_3", "gripper_rot_4", "gripper_rot_5",
            "finger_angle", "laser_rangefinder", "finger_pressure",
            "wrist_angle", "target_force",
            "tension_0", "tension_1", "tension_2", "tension_3",
            "gantry_position_x", "gantry_position_y", "gantry_position_z",
            "visual_pos_x", "visual_pos_y", "visual_pos_z",
            "hang_pos_x", "hang_pos_y", "hang_pos_z",
            "swing_cancellation_on",
        ]
        for key in expected:
            self.assertIn(key, obs_ft, f"Missing: {key}")

    def test_observation_features_contains_bearing_distance_keys(self):
        robot = _make_robot()
        obs_ft = robot.observation_features
        for name in ["hamper", "toybox", "trashcan", "gamepad", "parking_location"]:
            self.assertIn(f"{name}_bearing", obs_ft)
            self.assertIn(f"{name}_distance", obs_ft)

    def test_action_features_match_default_action_space(self):
        robot = _make_robot()
        expected = set(_ACTION_SPACES[DEFAULT_ACTION_SPACE])
        self.assertEqual(set(robot.action_features.keys()), expected)

    def test_no_camera_keys_in_action_features(self):
        robot = _make_robot("all")
        for key in robot.action_features:
            self.assertNotIn("camera", key)


# ---------------------------------------------------------------------------
# get_observation
# ---------------------------------------------------------------------------

class TestGetObservation(unittest.TestCase):

    def _check_obs_keys(self, robot, obs):
        for key in robot.observation_features:
            self.assertIn(key, obs, f"Missing observation key: {key}")

    def test_gripper_only_mode_observation_keys(self):
        robot = _make_robot_ready("gripper_384")
        obs = robot.get_observation()
        self._check_obs_keys(robot, obs)
        self.assertIn("gripper_camera", obs)
        self.assertNotIn("overhead_camera", obs)
        self.assertNotIn("anchor_camera_0", obs)

    def test_gripper_floor_mode_observation_keys(self):
        robot = _make_robot_ready("gripper_floor_384")
        obs = robot.get_observation()
        self._check_obs_keys(robot, obs)
        self.assertIn("gripper_camera", obs)
        self.assertIn("overhead_camera", obs)
        self.assertNotIn("anchor_camera_0", obs)

    def test_all_mode_observation_keys(self):
        robot = _make_robot_ready("all")
        obs = robot.get_observation()
        self._check_obs_keys(robot, obs)
        self.assertIn("gripper_camera", obs)
        self.assertIn("overhead_camera", obs)
        self.assertIn("anchor_camera_0", obs)
        self.assertIn("anchor_camera_1", obs)

    def test_image_shapes_match_mode(self):
        for mode in _CAMERA_MODES:
            robot = _make_robot_ready(mode)
            obs = robot.get_observation()
            for feed, (w, h) in _CAMERA_MODES[mode].items():
                key = _FEED_NAMES[feed]
                self.assertEqual(obs[key].shape, (h, w, 3), f"mode={mode} key={key}")

    def test_observation_returns_copies_not_references(self):
        robot = _make_robot_ready("gripper_384")
        obs1 = robot.get_observation()
        obs1["gripper_camera"][:] = 255
        obs2 = robot.get_observation()
        self.assertTrue(np.all(obs2["gripper_camera"] == 0))

    def test_numeric_state_defaults_are_float(self):
        robot = _make_robot_ready("gripper_384")
        obs = robot.get_observation()
        for key in ["vel_x", "vel_y", "vel_z", "gripper_pos_x", "finger_angle",
                    "tension_0", "gantry_position_x", "hang_pos_z"]:
            self.assertIsInstance(obs[key], float, f"{key} should be float")

    def test_updated_state_reflected_in_observation(self):
        robot = _make_robot_ready("gripper_384")
        robot.last_finger_angle = 1.23
        obs = robot.get_observation()
        self.assertAlmostEqual(obs["finger_angle"], 1.23)


# ---------------------------------------------------------------------------
# _handle_video_ready
# ---------------------------------------------------------------------------

class TestHandleVideoReady(unittest.TestCase):

    def _video_ready_item(self, feed_num, local_uri="rtsp://localhost:8554/stream"):
        item = Mock()
        item.feed_number = feed_num
        item.local_uri = local_uri
        item.stream_path = None
        return item

    def test_unknown_feed_is_ignored(self):
        robot = _make_robot("gripper_384")  # only feed 0
        item = self._video_ready_item(feed_num=99)
        # Should return early without raising or starting threads
        robot._handle_video_ready(item)
        self.assertEqual(robot.video_threads, {})

    def test_floor_feed_ignored_in_gripper_only_mode(self):
        robot = _make_robot("gripper_384")
        item = self._video_ready_item(feed_num=3)
        robot._handle_video_ready(item)
        self.assertEqual(robot.video_threads, {})

    def test_known_feed_starts_thread(self):
        robot = _make_robot("gripper_384")
        item = self._video_ready_item(feed_num=0)
        with patch.object(robot, "_video_stream_loop"):
            robot._handle_video_ready(item)
        self.assertIn(0, robot.video_threads)
        self.assertIsInstance(robot.video_threads[0], threading.Thread)

    def test_already_alive_thread_not_replaced(self):
        robot = _make_robot("gripper_384")
        # Simulate an already-running thread
        alive_thread = Mock(spec=threading.Thread)
        alive_thread.is_alive.return_value = True
        robot.video_threads[0] = alive_thread

        item = self._video_ready_item(feed_num=0)
        robot._handle_video_ready(item)
        # The original mock thread should still be in place
        self.assertIs(robot.video_threads[0], alive_thread)

    def test_all_mode_accepts_all_four_feeds(self):
        robot = _make_robot("all")
        with patch.object(robot, "_video_stream_loop"):
            for feed in [0, 1, 2, 3]:
                item = self._video_ready_item(feed_num=feed)
                robot._handle_video_ready(item)
        self.assertEqual(set(robot.video_threads.keys()), {0, 1, 2, 3})


# ---------------------------------------------------------------------------
# disconnect
# ---------------------------------------------------------------------------

class TestDisconnect(unittest.TestCase):

    def test_disconnect_sets_all_stop_events(self):
        for mode in _CAMERA_MODES:
            robot = _make_robot(mode)
            # Patch websocket to avoid AttributeError on .close()
            robot.websocket = Mock()
            robot.disconnect()
            for feed, ev in robot.stop_video_events.items():
                self.assertTrue(ev.is_set(), f"mode={mode} feed={feed} stop event not set")

    def test_disconnect_clears_websocket(self):
        robot = _make_robot("gripper_384")
        robot.websocket = Mock()
        robot.disconnect()
        self.assertIsNone(robot.websocket)

    def test_disconnect_when_no_websocket(self):
        robot = _make_robot("gripper_384")
        # Should not raise even if websocket is None
        robot.disconnect()
        self.assertIsNone(robot.websocket)


# ---------------------------------------------------------------------------
# Video stream resize
# ---------------------------------------------------------------------------

class TestVideoStreamResize(unittest.TestCase):

    def _run_stream_until_frame_written(self, robot, feed_num, frame_shape):
        """Drive _video_stream_loop with one synthetic frame and capture the resize call.

        Production code: `for av_frame in container.decode(stream)`
        The iterator yields exactly one frame then exhausts, so the loop exits
        without needing the stop event.
        """
        mock_frame = Mock()
        mock_frame.to_ndarray.return_value = np.zeros(frame_shape, dtype=np.uint8)

        mock_stream = Mock()
        mock_stream.type = "video"

        mock_container = Mock()
        mock_container.streams.__iter__ = Mock(return_value=iter([mock_stream]))
        mock_container.decode.return_value = iter([mock_frame])

        with patch("av.open", return_value=mock_container), \
             patch("cv2.resize", wraps=lambda f, size: np.zeros((size[1], size[0], 3), dtype=np.uint8)) as mock_resize:
            robot._video_stream_loop("rtsp://localhost/stream", True, feed_num)
            return mock_resize

    def test_gripper_feed_resized_to_correct_resolution(self):
        for mode in ["gripper_384", "gripper_floor_384", "all"]:
            robot = _make_robot(mode)
            expected_w, expected_h = _CAMERA_MODES[mode][0]
            # Provide a frame with the wrong size to force a resize
            wrong_shape = (100, 100, 3)
            mock_resize = self._run_stream_until_frame_written(robot, 0, wrong_shape)
            mock_resize.assert_called_once_with(unittest.mock.ANY, (expected_w, expected_h))

    def test_no_resize_when_frame_already_correct_size(self):
        robot = _make_robot("gripper_384")
        w, h = _CAMERA_MODES["gripper_384"][0]  # 384, 384
        correct_shape = (h, w, 3)
        mock_resize = self._run_stream_until_frame_written(robot, 0, correct_shape)
        mock_resize.assert_not_called()

    def test_anchor_cam_resized_to_960x544(self):
        robot = _make_robot("all")
        feed = 1
        expected_w, expected_h = 960, 544
        wrong_shape = (480, 640, 3)
        mock_resize = self._run_stream_until_frame_written(robot, feed, wrong_shape)
        mock_resize.assert_called_once_with(unittest.mock.ANY, (expected_w, expected_h))


# ---------------------------------------------------------------------------
# camera_mode forwarding through record/eval entry points
# ---------------------------------------------------------------------------

class TestCameraModeForwarding(unittest.TestCase):
    """Verify that record_until_disconnected and eval_until_disconnected pass
    camera_mode into StringmanConfig."""

    # Patch out HF auth and the repo-existence check so the test reaches
    # StringmanConfig deterministically. Without this the test depends on a
    # cached HF token being present (true on a dev machine, false on CI), so
    # ensure_hf_auth() raises before StringmanConfig is ever constructed.
    @patch("nf_robot.ml.stringman_lerobot.repo_exists", return_value=False)
    @patch("nf_robot.ml.stringman_lerobot.ensure_hf_auth")
    @patch("nf_robot.ml.stringman_lerobot.StringmanLeRobot")
    @patch("nf_robot.ml.stringman_lerobot.StringmanConfig")
    def test_record_forwards_camera_mode(self, MockConfig, MockRobot, _mock_auth, _mock_repo_exists):
        from nf_robot.ml.stringman_lerobot import record_until_disconnected

        robot_instance = MockRobot.return_value
        robot_instance.is_connected = False
        robot_instance.observation_features = {}
        robot_instance.action_features = {}

        try:
            record_until_disconnected(
                "ws://localhost:4245",
                "naavox/test",
                "robot_1",
                upload=False,
                camera_mode="gripper_224",
            )
        except Exception:
            pass  # We only care that StringmanConfig was called with the right camera_mode

        _, kwargs = MockConfig.call_args
        self.assertEqual(kwargs.get("camera_mode"), "gripper_224")

    @patch("nf_robot.ml.stringman_lerobot.StringmanLeRobot")
    @patch("nf_robot.ml.stringman_lerobot.StringmanConfig")
    def test_eval_forwards_camera_mode(self, MockConfig, MockRobot):
        from nf_robot.ml.stringman_lerobot import eval_until_disconnected

        robot_instance = MockRobot.return_value
        robot_instance.is_connected = False

        # Pre-import lerobot modules so patch.object replaces attributes on
        # already-loaded modules rather than triggering a fresh import while
        # another patch is active (which breaks @dataclass inheritance).
        import lerobot.configs.policies as _lcp
        import lerobot.policies.factory as _lpf
        import huggingface_hub as _hfh

        mock_policy = Mock()
        mock_policy.eval = Mock()

        with patch.object(_hfh, "hf_hub_download", return_value="/tmp/fake.json"), \
             patch("builtins.open", unittest.mock.mock_open(read_data='{"dataset": {"repo_id": "naavox/ds"}}')), \
             patch("nf_robot.ml.stringman_lerobot.LeRobotDataset"), \
             patch("nf_robot.ml.stringman_lerobot.action_space_from_features", return_value="gripper_vel"), \
             patch.object(_lcp, "PreTrainedConfig"), \
             patch.object(_lpf, "make_pre_post_processors", return_value=(Mock(), Mock())), \
             patch.object(_lpf, "make_policy", return_value=mock_policy):
            try:
                eval_until_disconnected(
                    "ws://localhost:4245",
                    "naavox/policy",
                    "robot_1",
                    camera_mode="gripper_floor_224",
                )
            except Exception:
                pass

        _, kwargs = MockConfig.call_args
        self.assertEqual(kwargs.get("camera_mode"), "gripper_floor_224")


if __name__ == "__main__":
    unittest.main()
