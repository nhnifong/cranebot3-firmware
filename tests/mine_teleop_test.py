"""Tests for what the teleop miner does with a target it cannot see.

The labels come from projecting one room point back through the approach, and nothing in
that arithmetic knows whether the point landed in the picture. Frames where it did not are
the ones these are about: the honest label there is "no target in view", not a position the
image gives no evidence for.
"""

import math
import unittest

import numpy as np

from nf_robot.ml.visual_servoing.mine_teleop import (
    CANVAS_SCALE, FALSE_GRAB_PREFIX, MODE_FALSE_GRABS, MODE_GRASPS, MODE_NEGATIVES,
    NEGATIVE_PREFIX, OFF_SCREEN_MARGIN, ReservoirSampler, ShardWriter, close_run,
    final_grasp, gripping_segments, held_force, holding_labels, in_view, mine_episode,
    mine_false_grab_episode, shard_prefix)
from nf_robot.ml.visual_servoing.geometry import CAMERA_POS_BODY
from nf_robot.ml.visual_servoing.uv_methods import (
    DEFAULT_UV_METHOD, anchor_is_usable, DELTA_METHODS, UV_METHODS, grasp_point_room,
    gripper_camera_calibration, jaw_uv, project, project_camera, range_to_floor,
    target_track, unproject)


def rows_for(offsets, fps=30.0, grasp_at=60, length=90):
    """A descending approach that grasps at `grasp_at` and lifts afterwards.

    `offsets` gives the horizontal distance from the target at each frame, which is what
    decides whether the projected label lands in the frame.
    """
    rows = []
    for i in range(length):
        held = i >= grasp_at
        # down to the object, then up again carrying it - the rise after the grasp is
        # what find_grasp's caller uses to tell a real pick from closing on nothing
        height = (0.6 - 0.005 * i if not held
                  else 0.6 - 0.005 * grasp_at + 0.01 * (i - grasp_at))
        rows.append({
            "frame_index": i,
            "timestamp": i / fps,
            "gripper_pos": np.array([offsets[i], 0.0, height]),
            "spin": 0.0,
            # >= PRESSURE_THRESHOLD is a grasp; the recorded value rises with grip
            "pressure": 1.0 if held else 0.0,
            "wrist_angle": 540.0,
            "finger_angle": 0.0,
            "laser_rangefinder": max(height - 0.2, 0.05),
            "target_force": 0.0,
            # the close that makes the grasp
            "finger_speed": 90.0 if grasp_at - 10 <= i < grasp_at else 0.0,
        })
    # Exactly the velocity the track was built from, so a dead-reckoning method integrating
    # it has to land back on the positions - which is what makes a disagreement in these
    # tests a bug in the integration rather than in the fixture. spin is zero throughout,
    # so the gripper and room frames coincide and both velocity fields hold the same thing.
    for i, row in enumerate(rows):
        j = min(i + 1, len(rows) - 1)
        dt = rows[j]["timestamp"] - row["timestamp"]
        step = (rows[j]["gripper_pos"] - row["gripper_pos"]) / dt if dt else np.zeros(3)
        row["vel_cmd"] = step
        row["vel_obs"] = step.copy()
    return rows


# intrinsics as fractions of the frame, the way gripper_camera_calibration returns them
CALIBRATION = ((439.3 / 684.0, 461.6 / 384.0), (0.5, 0.308))

FRAME_W, FRAME_H = 684, 384


def synthetic_frames(rows, grasp, calibration=CALIBRATION, seed=7):
    """Frames in which the target's place is known, for testing a method that reads pixels.

    One noise texture slid so that the same feature lands where `room-delta` says the
    target is in each frame. Noise because it is what a tracker locks onto best, and a
    tracker that cannot follow this could not follow carpet; a plain translation because
    the question here is whether the track walks the right way from the right anchor, and
    a real approach's scale change would only blur that with the tracker's own accuracy.
    """
    import cv2

    from nf_robot.ml.visual_servoing.uv_methods import target_track

    rng = np.random.default_rng(seed)
    pad = 400
    texture = rng.integers(0, 255, (FRAME_H + 2 * pad, FRAME_W + 2 * pad, 3), dtype=np.uint8)
    texture = cv2.GaussianBlur(texture, (5, 5), 0)  # LK wants gradients, not white noise

    truth = target_track(rows, grasp, calibration, "room-delta")
    anchor = truth[grasp]
    frames = {}
    for row, point in zip(rows, truth):
        if point is None:
            continue
        dx = int(round((point[0] - anchor[0]) * FRAME_W))
        dy = int(round((point[1] - anchor[1]) * FRAME_H))
        x, y = pad - dx, pad - dy
        if not (0 <= x <= 2 * pad and 0 <= y <= 2 * pad):
            continue
        frames[row["frame_index"]] = texture[y:y + FRAME_H, x:x + FRAME_W].copy()
    return lambda i: frames[i]


class TestInView(unittest.TestCase):

    def test_inside_the_frame_is_in_view(self):
        self.assertTrue(in_view(0.5, 0.5))
        self.assertTrue(in_view(0.0, 1.0))

    def test_a_little_past_the_edge_is_still_worth_predicting(self):
        """The case the oversized canvas exists for: the object is in shot, the spot to
        grab it by has slipped past the edge."""
        self.assertTrue(in_view(1.0 + OFF_SCREEN_MARGIN / 2, 0.5))
        self.assertTrue(in_view(0.5, -OFF_SCREEN_MARGIN / 2))

    def test_far_outside_is_not(self):
        self.assertFalse(in_view(1.0 + OFF_SCREEN_MARGIN * 2, 0.5))
        self.assertFalse(in_view(0.5, -OFF_SCREEN_MARGIN * 2))

    def test_the_margin_is_inside_the_canvas(self):
        """Rows past the margin are still kept as frames; rows past the canvas are not
        mined at all, so the margin has to be the tighter of the two."""
        self.assertLess(OFF_SCREEN_MARGIN, (CANVAS_SCALE - 1.0) / 2.0)


def grip_rows(length=300, close=(40, 60), contact=50, settled=70, lift=100, release=250,
              level=0.2, creep=0.0, ramp_until=None, fps=30.0):
    """One grip, phase by phase: a close command over `close`, contact at `contact`, the
    fingers creeping on until `settled`, the operator's up command from `lift`, and the
    jaws opened at `release`.

    Pressure ramps from contact to `level` by `ramp_until` (default `settled`), then
    drifts by `creep` per second. target_force comes on a few frames after contact, the
    way the force controller lags the first touch.
    """
    ramp_until = settled if ramp_until is None else ramp_until
    rows = []
    z = 0.3
    for i in range(length):
        if contact <= i < release:
            if i < ramp_until:
                pressure = level * (0.2 + 0.8 * (i - contact) / (ramp_until - contact))
            else:
                pressure = level + creep * (i - ramp_until) / fps
        else:
            pressure = 0.0
        climbing = lift <= i < release
        if climbing:
            z += 0.01
        angle = (0.0 if i < close[0]
                 else 60.0 * (min(i, settled) - close[0]) / (settled - close[0]))
        rows.append({
            "frame_index": i,
            "timestamp": i / fps,
            "gripper_pos": np.array([0.0, 0.0, z]),
            "spin": 0.0,
            "pressure": pressure,
            "wrist_angle": 540.0,
            "finger_angle": angle,
            "laser_rangefinder": 0.11,
            "target_force": level if contact + 5 <= i < release else 0.0,
            "finger_speed": (90.0 if close[0] <= i < close[1]
                             else -90.0 if release <= i < release + 10 else 0.0),
            "vel_cmd": np.array([0.0, 0.0, 0.3 if climbing else 0.0]),
            "vel_obs": np.zeros(3),
        })
    return rows


class TestGrippingSegments(unittest.TestCase):

    def test_a_light_grip_in_force_mode_is_not_a_release(self):
        """Paper holds below the pressure floor; force mode is what says it is still held."""
        rows = grip_rows(level=0.01)
        segments = gripping_segments(rows, 30.0)
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0][1], 250)

    def test_lowering_the_force_mid_carry_is_not_a_release(self):
        """A negative finger command in force mode only asks for less force."""
        rows = grip_rows()
        for r in rows[150:160]:
            r["finger_speed"] = -90.0
        self.assertEqual(gripping_segments(rows, 30.0), [(50, 250)])

    def test_the_release_is_force_mode_off_and_pressure_gone(self):
        rows = grip_rows()
        for r in rows[250:260]:
            r["pressure"] = 0.1  # still squeezed while the jaws open
        self.assertEqual(gripping_segments(rows, 30.0), [(50, 260)])


class TestFinalGrasp(unittest.TestCase):

    def test_the_lift_is_the_up_command(self):
        """Closing on the floor levers the gripper up a few cm; that is not the lift."""
        rows = grip_rows()
        for i, r in enumerate(rows[55:70]):
            r["gripper_pos"] = r["gripper_pos"] + [0.0, 0.0, 0.002 * i]
        self.assertEqual(final_grasp(rows, 30.0, 0.10), (50, 250, 100))

    def test_an_earlier_grip_that_never_rose_is_passed_over(self):
        rows = grip_rows(length=500, close=(240, 260), contact=250, settled=270, lift=300,
                         release=450)
        for r in rows[50:150]:
            r["pressure"] = 0.2
        self.assertEqual(final_grasp(rows, 30.0, 0.10), (250, 450, 300))

    def test_the_last_lifted_grip_is_the_one_picked(self):
        rows = grip_rows(length=500)
        later = grip_rows(length=500, close=(340, 360), contact=350, settled=370, lift=380,
                          release=450)
        rows[300:] = later[300:]
        for r in rows[300:]:
            r["gripper_pos"] = r["gripper_pos"] + [0.0, 0.0, 1.5]
        self.assertEqual(final_grasp(rows, 30.0, 0.10), (350, 450, 380))

    def test_a_grip_that_never_rose(self):
        rows = grip_rows()
        for r in rows:
            r["gripper_pos"] = np.zeros(3)
        self.assertEqual(final_grasp(rows, 30.0, 0.10), "no_rise")

    def test_no_grip_at_all(self):
        rows = grip_rows()
        for r in rows:
            r["pressure"] = 0.0
            r["target_force"] = 0.0
        self.assertEqual(final_grasp(rows, 30.0, 0.10), "no_grasp")


class TestCloseRun(unittest.TestCase):

    def test_it_runs_from_the_command_to_the_fingers_settling(self):
        """The force controller keeps the fingers moving after the command ends."""
        onset, end = close_run(grip_rows(), 50, 100, 30.0)
        self.assertEqual(onset, 40)
        self.assertGreaterEqual(end, 70)
        self.assertLess(end, 100)

    def test_a_lift_before_the_fingers_settle_ends_it(self):
        self.assertEqual(close_run(grip_rows(lift=65), 50, 65, 30.0), (40, 65))

    def test_a_contact_with_no_close_leading_into_it_has_none(self):
        rows = grip_rows()
        for r in rows:
            r["finger_speed"] = 0.0
        self.assertIsNone(close_run(rows, 50, 100, 30.0))


class TestHoldingLabels(unittest.TestCase):
    """`holding` asks whether it is safe to lift now, which the operator answered by
    lifting."""

    def labels(self, **kwargs):
        rows = grip_rows(**kwargs)
        contact, release, lift = final_grasp(rows, 30.0, 0.10)
        _, settled = close_run(rows, contact, lift, 30.0)
        return holding_labels(rows, 30.0, contact, settled, lift, release)

    def test_nothing_in_the_hand_is_a_negative(self):
        labels = self.labels()
        self.assertTrue(all(h == 0 for h in labels[:50]))
        self.assertTrue(all(h == 0 for h in labels[250:]))

    def test_the_close_is_masked(self):
        self.assertTrue(all(h is None for h in self.labels()[50:70]))

    def test_from_the_lift_to_the_release_is_positive(self):
        self.assertTrue(all(h == 1 for h in self.labels()[100:250]))

    def test_a_steady_plateau_before_the_lift_is_positive(self):
        """As safe as the moment the operator chose; they were just slower to act."""
        self.assertTrue(all(h == 1 for h in self.labels()[90:100]))

    def test_pressure_still_climbing_before_the_lift_is_masked(self):
        labels = self.labels(ramp_until=90)
        self.assertTrue(all(h is None for h in labels[70:90]))
        self.assertTrue(all(h == 1 for h in labels[100:250]))

    def test_a_creep_the_operator_lifted_through_counts_as_steady(self):
        """Soft objects never stop creeping; the operator lifting is what says that rate
        is fine."""
        labels = self.labels(creep=0.03)
        self.assertTrue(all(h == 1 for h in labels[90:100]))


class TestHeldForce(unittest.TestCase):
    """The force label is what deployment closes to: the commanded force of the carry."""

    def test_a_spike_at_the_lift_does_not_move_it(self):
        rows = grip_rows()
        for r in rows[95:110]:
            r["pressure"] = 0.6
        self.assertAlmostEqual(held_force(rows, 100, 250), 0.2)

    def test_the_opening_ramp_is_left_out(self):
        rows = grip_rows()
        for i, r in enumerate(rows[230:250]):
            r["finger_speed"] = -90.0
            r["target_force"] = 0.2 * (1 - i / 20)
        self.assertAlmostEqual(held_force(rows, 100, 250), 0.2)

    def test_a_grip_never_in_force_mode_falls_back_to_the_felt_pressure(self):
        rows = grip_rows(level=0.05)
        for r in rows:
            r["target_force"] = 0.0
        self.assertAlmostEqual(held_force(rows, 100, 250), 0.05)


class TestMineEpisode(unittest.TestCase):

    def _mine(self, offsets):
        return mine_episode(rows_for(offsets), 30.0, CALIBRATION,
                            approach_seconds=2.0, carry_seconds=0.5, rise_m=0.05)

    def test_a_close_approach_keeps_its_position_labels(self):
        result, dropped, blind = self._mine([0.02] * 90)
        self.assertIsNotNone(result)
        labeled = [r for r in result if r["target_uv"] is not None]
        self.assertGreater(len(labeled), 0)
        self.assertEqual(blind, 0)
        self.assertTrue(all(r["target_present"] == 1 for r in labeled))

    def test_a_target_outside_the_frame_loses_its_position_labels(self):
        """The frame is kept - it is a real picture with the object out of shot - and
        every position label is dropped rather than pointed somewhere invented."""
        # drifting in from one side, so the label crosses the margin partway through
        offsets = list(np.linspace(0.45, 0.02, 60)) + [0.02] * 30
        result, dropped, blind = self._mine(offsets)
        self.assertGreater(blind, 0)
        blind_rows = [r for r in result if r["seconds_to_grasp"] > 0
                      and r["target_uv"] is None]
        self.assertEqual(len(blind_rows), blind)
        for row in blind_rows:
            self.assertIsNone(row["target_uv"])
            self.assertIsNone(row["target_range_m"])
            self.assertIsNone(row["grasp_axis_rad"])

    def test_blind_rows_still_carry_the_labels_that_do_not_need_the_target(self):
        """Finger and holding are about the gripper, not about where the object is."""
        result, _, blind = self._mine(list(np.linspace(0.45, 0.02, 60)) + [0.02] * 30)
        blind_rows = [r for r in result if r["seconds_to_grasp"] > 0 and r["target_uv"] is None]
        self.assertGreater(len(blind_rows), 0)
        for row in blind_rows:
            self.assertIsNotNone(row["finger"])
            self.assertIsNotNone(row["holding"])

    def test_auto_approach_stops_where_the_target_leaves_the_frame(self):
        """Back to the last frame it was in view, however long the hover, and no further."""
        offsets = [0.45] * 20 + list(np.linspace(0.45, 0.02, 30)) + [0.02] * 80
        result, _, blind = mine_episode(rows_for(offsets, grasp_at=100, length=130), 30.0,
                                        CALIBRATION, approach_seconds=None, carry_seconds=0.5,
                                        rise_m=0.05)
        before = [r for r in result if r["seconds_to_grasp"] > 0]
        self.assertEqual(blind, 0)
        # the 50 frames of hover, plus some of the drift in
        self.assertGreater(len(before), 50)
        self.assertLess(len(before), 80)
        self.assertTrue(all(r["target_uv"] is not None for r in before))

    def test_a_blind_row_does_not_claim_the_picture_is_empty(self):
        """Only that this object is out of shot. Something else graspable may well be in
        view, so present is masked rather than set to zero."""
        result, _, _ = self._mine(list(np.linspace(0.45, 0.02, 60)) + [0.02] * 30)
        blind_rows = [r for r in result if r["seconds_to_grasp"] > 0 and r["target_uv"] is None]
        self.assertGreater(len(blind_rows), 0)
        for row in blind_rows:
            self.assertIsNone(row["target_present"])

    def test_carry_frames_are_unchanged(self):
        """After the grasp the object rides in the jaws; those rows were already unlabeled
        for position and are not what the new masking is about.

        What they do keep is the labels that describe the gripper rather than where the
        object is. `holding` is one of those but has a window of its own, which
        TestHoldingWindow covers."""
        result, _, _ = self._mine([0.02] * 90)
        carried = [r for r in result if r["seconds_to_grasp"] < 0]
        self.assertGreater(len(carried), 0)
        for row in carried:
            self.assertIsNone(row["target_uv"])
            self.assertIsNone(row["target_present"])
            self.assertIsNotNone(row["finger"])

    def test_the_close_label_marks_the_closing_movement(self):
        """0 before the close that made the grasp, 1 through it, masked after."""
        result, _, _ = self._mine([0.02] * 90)
        by_frame = {r["frame_index"]: r["close_now"] for r in result}
        self.assertEqual(by_frame[49], 0)
        self.assertTrue(all(by_frame[i] == 1 for i in range(50, 60)))
        self.assertTrue(all(v is None for i, v in by_frame.items() if i >= 60))

    def test_everything_before_the_grasp_is_a_negative(self):
        """An object in view, in reach, and not in the hand is the case the head keeps
        getting wrong, so those frames are taught rather than masked."""
        result, _, _ = self._mine([0.02] * 90)
        approach = [r for r in result if r["seconds_to_grasp"] > 0]
        self.assertGreater(len(approach), 0)
        self.assertTrue(all(r["holding"] == 0 for r in approach))

    def test_an_episode_with_no_grasp_is_skipped(self):
        rows = rows_for([0.02] * 90)
        for row in rows:
            row["pressure"] = 0.0
        result, reason, blind = mine_episode(rows, 30.0, CALIBRATION, 2.0, 0.5, 0.05)
        self.assertIsNone(result)
        self.assertEqual(reason, "no_grasp")
        self.assertEqual(blind, 0)


class TestFalseGrabs(unittest.TestCase):
    """A recording where closing the jaws would catch nothing.

    Only two heads can be labeled from that: the close should not begin, and nothing is
    held. Everything else has to stay masked, and target_present is the one that matters -
    a false grab happens next to graspable things, so claiming the picture is empty would
    teach the target head the opposite of what the frame shows.
    """

    def setUp(self):
        # pressure high throughout: jaws closing on each other read exactly like jaws
        # closing on an object, and in this recording it is always the former
        self.rows = rows_for([0.0] * 90, grasp_at=0)
        self.out, _, _ = mine_false_grab_episode(self.rows, stride=1)

    def test_the_close_head_is_told_not_to_close(self):
        self.assertTrue(all(r["close_now"] == 0 for r in self.out))

    def test_the_holding_head_is_told_nothing_is_held(self):
        self.assertTrue(all(r["holding"] == 0 for r in self.out))

    def test_pressure_does_not_soften_the_holding_label(self):
        """mine_negative_episode masks holding when pressure is up, because something may
        have been picked up mid-recording. Here that reading is the fingers meeting, and
        those frames are the whole point of the mode."""
        self.assertTrue(all(r["pressure"] > 0 for r in self.rows[1:]))
        self.assertTrue(all(r["holding"] == 0 for r in self.out))

    def test_it_does_not_claim_the_picture_is_empty(self):
        self.assertTrue(all(r["target_present"] is None for r in self.out))

    def test_every_other_head_is_masked(self):
        for key in ("target_uv", "target_range_m", "grasp_axis_rad", "finger",
                    "grasp_pressure", "seconds_to_grasp"):
            with self.subTest(key=key):
                self.assertTrue(all(r[key] is None for r in self.out))

    def test_the_state_vector_still_reaches_the_row(self):
        """Masked labels, but the frame is still a real observation and the state inputs
        are what the model reads alongside it."""
        self.assertEqual(self.out[0]["state"]["laser_rangefinder"],
                         round(self.rows[0]["laser_rangefinder"], 4))

    def test_the_stride_thins_the_recording(self):
        thinned, _, _ = mine_false_grab_episode(self.rows, stride=5)
        self.assertEqual(len(thinned), 18)
        self.assertEqual([r["frame_index"] for r in thinned[:3]], [0, 5, 10])


class TestShardPrefix(unittest.TestCase):

    def test_the_miner_owns_only_its_own_shards(self):
        """Both producers write into one split, so a rerun of either must leave the
        other's files alone - emptying the directory used to delete the synthetic half
        without saying so."""
        self.assertEqual(ShardWriter.DEFAULT_PREFIX, "shard")
        self.assertNotEqual(ShardWriter.DEFAULT_PREFIX, "synth")

    def test_every_mode_writes_under_its_own_prefix(self):
        """One pool holds all three, so a rerun of any mode must replace only its own."""
        prefixes = [shard_prefix(m) for m in (MODE_GRASPS, MODE_NEGATIVES, MODE_FALSE_GRABS)]
        self.assertEqual(prefixes, [ShardWriter.DEFAULT_PREFIX, NEGATIVE_PREFIX,
                                    FALSE_GRAB_PREFIX])
        self.assertEqual(len(set(prefixes)), 3)


if __name__ == "__main__":
    unittest.main()


class TestReservoirSampler(unittest.TestCase):
    """The preview-only sink. It stands in for the shard writer, so what it keeps has to
    be a fair picture of what a real run would have written."""

    def test_it_keeps_everything_while_it_is_not_full(self):
        sampler = ReservoirSampler(10, seed=0)
        for i in range(4):
            sampler.add({"i": i})
        self.assertEqual([r["i"] for r in sampler.rows], [0, 1, 2, 3])
        self.assertEqual(sampler.total, 4)

    def test_it_counts_the_whole_stream_it_did_not_keep(self):
        """The count is what says how much a full run would write, and the preview says
        so: a sample of 20 out of 200 frames means something else than 20 out of 20."""
        sampler = ReservoirSampler(5, seed=0)
        for i in range(200):
            sampler.add({"i": i})
        self.assertEqual(sampler.total, 200)
        self.assertEqual(len(sampler.rows), 5)

    def test_the_same_seed_over_the_same_stream_picks_the_same_rows(self):
        """Two preview runs over unchanged sources have to land on the same frames, or a
        label change cannot be told apart from a different draw."""
        def run():
            sampler = ReservoirSampler(8, seed=3)
            for i in range(500):
                sampler.add({"i": i})
            return [r["i"] for r in sampler.rows]
        self.assertEqual(run(), run())

    def test_it_draws_from_the_whole_stream_not_just_the_head(self):
        """Keeping the first N would preview only the earliest episodes, which is exactly
        where an approach looks most alike."""
        late = 0
        for seed in range(20):
            sampler = ReservoirSampler(10, seed=seed)
            for i in range(1000):
                sampler.add({"i": i})
            late += sum(1 for r in sampler.rows if r["i"] >= 500)
        self.assertGreater(late, 50)


class TestUvMethods(unittest.TestCase):
    """The choice of how a frame's uv is decided. Each method is wrong in its own way, so
    what is tested here is the contract they share and the anchor each one commits to."""

    def setUp(self):
        self.rows = rows_for([0.0] * 90)
        self.grasp = 60
        self.frames = synthetic_frames(self.rows, self.grasp)
        self.jaw = jaw_uv(self.rows[self.grasp]["laser_rangefinder"], CALIBRATION)

    def track(self, method):
        return target_track(self.rows, self.grasp, CALIBRATION, method,
                            frames=self.frames)

    def test_every_method_answers_for_every_frame(self):
        """label_video renders the whole episode, not the mined window, so a track short
        of the episode would silently misalign the mark with the frame it is drawn on."""
        for method in UV_METHODS:
            with self.subTest(method=method):
                self.assertEqual(len(self.track(method)), len(self.rows))

    def test_dead_reckoning_anchors_on_the_jaws(self):
        """The whole method hangs off this frame: at the grasp the object is between the
        fingers, and that is the one place in an episode the answer is known."""
        for method in ("dead-reckon", "dead-reckon-observed", "optical-flow"):
            with self.subTest(method=method):
                u, v, _ = self.track(method)[self.grasp]
                self.assertAlmostEqual(u, self.jaw[0], places=5)
                self.assertAlmostEqual(v, self.jaw[1], places=5)

    def test_every_method_starts_from_the_same_place(self):
        """All of them anchor on the jaws now that the mount says where those are, so the
        grasp frame is one answer arrived at three ways and they have to agree."""
        answers = [self.track(m)[self.grasp] for m in UV_METHODS]
        for got in answers[1:]:
            self.assertAlmostEqual(got[0], answers[0][0], places=4)
            self.assertAlmostEqual(got[1], answers[0][1], places=4)

    def test_the_projecting_methods_range_to_the_jaws_not_straight_down(self):
        """The rangefinder reads the drop below the *lens* and the target hangs below the
        *jaws*, 2.7cm behind it, so the distance along the ray is the hypotenuse of those
        two and is always a little longer than the laser. Asserting the laser reading here
        is what a version of this that had forgotten the offset would do.
        """
        row = self.rows[self.grasp]
        # the laser reads the lens's own drop to the floor, and the jaw point sits 2.7cm
        # back along the nose axis from directly under it
        expected = float(np.hypot(row["laser_rangefinder"], CAMERA_POS_BODY[1]))
        self.assertGreater(expected, row["laser_rangefinder"])
        for method in DELTA_METHODS:
            with self.subTest(method=method):
                self.assertAlmostEqual(self.track(method)[self.grasp][2], expected, places=5)

    def test_optical_flow_ranges_off_the_floor_plane(self):
        """It has no 3D point to measure, only a bearing, so the range is where that
        bearing meets the floor the rangefinder found: L / cos of the angle off straight
        down. On this camera the jaws are nearly straight down, so at the grasp the two
        answers agree to well under a percent - which is the check that the cosine is the
        right way up, since dividing by it where multiplying was meant looks identical
        until the target is far off axis.
        """
        real = gripper_camera_calibration()
        row = self.rows[self.grasp]
        # the jaw ray is a couple of degrees off straight down, so the floor is a shade
        # further along it than the beam's own reading
        self.assertAlmostEqual(range_to_floor(*jaw_uv(row["laser_rangefinder"], real), row, real),
                               row["laser_rangefinder"], delta=0.004)
        # off to the side, the floor is further along the ray than it is straight down
        off_axis = (0.05, jaw_uv(row["laser_rangefinder"], real)[1])
        self.assertGreater(range_to_floor(*off_axis, row, real), row["laser_rangefinder"])

    def test_optical_flow_follows_the_target_it_was_given(self):
        """Against frames built so the target's place in each of them is known, the track
        has to be that place. Anchored at the jaws, so what is compared is the motion -
        which is all the method claims to recover."""
        track = self.track("optical-flow")
        truth = self.track("room-delta")
        for i in range(self.grasp - 25, self.grasp + 1, 5):
            with self.subTest(frame=i):
                self.assertIsNotNone(track[i], "flow lost a target that never left frame")
                self.assertAlmostEqual(track[i][0] - track[self.grasp][0],
                                       truth[i][0] - truth[self.grasp][0], places=2)
                self.assertAlmostEqual(track[i][1] - track[self.grasp][1],
                                       truth[i][1] - truth[self.grasp][1], places=2)

    def test_optical_flow_stops_rather_than_guessing(self):
        """A lost track snaps onto whatever else is nearby rather than drifting off
        slowly, so extrapolating past the loss would put a confident label on the wrong
        thing. Blank frames are the bluntest way to lose it."""
        blank = np.zeros((384, 684, 3), np.uint8)
        track = target_track(self.rows, self.grasp, CALIBRATION, "optical-flow",
                             frames=lambda i: blank)
        self.assertIsNotNone(track[self.grasp])
        self.assertTrue(all(t is None for i, t in enumerate(track) if i != self.grasp))

    def test_optical_flow_says_so_when_it_is_given_no_frames(self):
        """It is the one method that reads pixels, and a caller that cannot offer them has
        to be told which flag asked for them rather than handed a track of None."""
        with self.assertRaises(SystemExit) as caught:
            target_track(self.rows, self.grasp, CALIBRATION, "optical-flow")
        self.assertIn("optical-flow", str(caught.exception))

    def test_exact_velocity_reproduces_the_track_it_was_built_from(self):
        """With the recorded velocity equal to the motion that happened, the integrator
        has no error to make, so the two methods can differ only by their anchors - the
        same vector in every frame. A gap that drifts is an integration bug, and it is the
        one failure this fixture can tell apart from the real disagreement on a robot."""
        deltas = {m: DELTA_METHODS[m](self.rows, self.grasp, CALIBRATION, None)
                  for m in ("room-delta", "dead-reckon")}
        gaps = [deltas["dead-reckon"][i] - deltas["room-delta"][i]
                for i in range(len(self.rows))]
        for i, gap in enumerate(gaps):
            with self.subTest(frame=i):
                np.testing.assert_allclose(gap, gaps[self.grasp], atol=1e-9)

    def test_dead_reckoning_walks_the_object_out_of_frame_as_the_gripper_leaves(self):
        """Integrating away from the grasp has to move the mark the way the camera moved.
        Rising off the object puts it further away and nearer the middle, not nowhere."""
        track = self.track("dead-reckon")
        self.assertIsNotNone(track[0])
        self.assertGreater(track[0][2], track[self.grasp][2])

    def test_a_recording_without_velocity_says_which_flag_needed_it(self):
        """These fields are optional in read_columns, so the failure lands here rather than
        at read time, and a run that cannot use a method has to say so by name."""
        for row in self.rows:
            row["vel_cmd"] = None
        with self.assertRaises(SystemExit) as caught:
            self.track("dead-reckon")
        self.assertIn("vel_cmd", str(caught.exception))
        self.assertIn(DEFAULT_UV_METHOD, str(caught.exception))

    def test_an_unknown_method_names_the_ones_there_are(self):
        with self.assertRaises(SystemExit) as caught:
            self.track("wishful-thinking")
        self.assertIn("room-delta", str(caught.exception))

    def test_unproject_inverts_project(self):
        """The dead-reckoning anchor is an unprojection, so a sign error in it would put
        the target at a mirrored bearing and still produce a plausible looking track.
        Off-frame coordinates included: that is where the anchor is allowed to land."""
        for uv in (self.jaw, (0.1, 0.2), (1.2, -0.1)):
            with self.subTest(uv=uv):
                u, v, distance = project_camera(unproject(*uv, 0.35, CALIBRATION),
                                                CALIBRATION)
                self.assertAlmostEqual(u, uv[0], places=6)
                self.assertAlmostEqual(v, uv[1], places=6)
                self.assertAlmostEqual(distance, 0.35, places=6)


class TestUnusableRange(unittest.TestCase):
    """A grasp frame whose rangefinder read nothing. Every method hangs the target off that
    reading, so there is no target and the episode is not minable by any of them."""

    def rows_with_grasp_laser(self, laser):
        rows = rows_for([0.0] * 90)
        for row in rows:
            row["laser_rangefinder"] = laser
        return rows

    def test_a_gripper_resting_on_the_floor_has_no_anchor(self):
        """What a close commanded with the fingers already down reports. The jaw point is
        then at or behind the lens, where a projection still returns plausible numbers."""
        rows = self.rows_with_grasp_laser(0.004)
        self.assertFalse(anchor_is_usable(rows[60], gripper_camera_calibration()))

    def test_an_ordinary_grasp_range_does(self):
        rows = self.rows_with_grasp_laser(0.11)
        self.assertTrue(anchor_is_usable(rows[60], gripper_camera_calibration()))

    def test_the_miner_skips_it_by_name(self):
        """Named rather than left to drop frame by frame: a whole episode vanishing one
        frame at a time reports as off-canvas, which says nothing about the cause."""
        rows = self.rows_with_grasp_laser(0.004)
        result, reason, _ = mine_episode(rows, 30.0, gripper_camera_calibration(),
                                         2.0, 1.0, 0.05)
        self.assertIsNone(result)
        self.assertEqual(reason, "no_range")
