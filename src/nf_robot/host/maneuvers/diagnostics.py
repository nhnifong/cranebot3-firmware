"""Diagnostics: measurements of how well the robot flies, for tuning it."""

import asyncio
import logging
import time

import numpy as np
from scipy.spatial.transform import Rotation

import nf_robot.common.definitions as model_constants
from nf_robot.generated.nf import common, control
from nf_robot.host.maneuver import (ROUTE_POINT_TAG_NAMES, Maneuver, command,
                                    prefer_swing_cancellation, verb)

logger = logging.getLogger(__name__)


class Diagnostics(Maneuver):
    name = 'diagnostics'
    title = 'Diagnostics'

    @command(control.Command.HORIZONTAL_CHECK, motion=True)
    @verb('linear', motion=True)
    @prefer_swing_cancellation
    async def linear_height_check_task(self):
        """
        Measure the average deviation from an ideal constant height, as reported by the
        laser rangefinder, while traversing the floor along the currently selected route.
        Triggered by the debug command "linear". This is a motion task.

        Every room is different and only the operator can pick a path across the floor with
        no obstructions, so the traverse runs between the route source and destination,
        both at 1.5m altitude. The gantry flies directly to
        the source, pauses for 2 seconds, then traverses to the
        destination. Through an ideal move the laser should read (1.5 - pole - laser_offset)
        the whole way. Aborts if the laser altitude drops below 0.2m or if the gantry comes
        within 0.4m of the ceiling (the z position of anchor 0).
        """
        TEST_ALTITUDE_M = 1.5
        MIN_LASER_ALTITUDE_M = 0.2
        CEILING_MARGIN_M = 0.4
        SAMPLE_INTERVAL_S = 0.02
        ob = self.ob
        ideal_laser_range = TEST_ALTITUDE_M - ob.pole_offset()[2] - model_constants.laser_offset

        # ceiling height for the proximity abort
        ceiling_z = ob.anchor_points()[0][2]

        # Resolve the route endpoints to floor positions chosen by the operator.
        def route_point_floor_pos(route_point, label):
            if route_point in ROUTE_POINT_TAG_NAMES:
                position = ob.route_point_position(route_point)
                if position is None:
                    logger.warning(f'Linear height check: no saved position for {label} tag '
                                   f'"{ROUTE_POINT_TAG_NAMES[route_point]}"')
                return position
            if route_point == common.RoutePoint.ORIGIN:
                return np.zeros(3)
            logger.warning(f'Linear height check needs the {label} to be a tag or the origin, not {route_point}')
            return None

        src, dst = ob.route()
        src_pos = route_point_floor_pos(src, 'route source')
        dst_pos = route_point_floor_pos(dst, 'route destination')
        if src_pos is None or dst_pos is None:
            return
        point_a = np.array([src_pos[0], src_pos[1], TEST_ALTITUDE_M])
        point_b = np.array([dst_pos[0], dst_pos[1], TEST_ALTITUDE_M])

        # Fly directly to the route source with auto altitude, then pause before the test.
        await ob.seek_goal(point_a, auto_altitude=True)
        await asyncio.sleep(2.0)

        # Traverse to the route destination, sampling the laser the whole way.
        # disable altitude cruise during test
        deviations = []
        aborted = None
        move_task = asyncio.create_task(ob.seek_goal(point_b, auto_altitude=False))
        try:
            while not move_task.done():
                await asyncio.sleep(SAMPLE_INTERVAL_S)
                laser_range = ob.laser_range()
                if laser_range is None:
                    continue
                gant_z = ob.gantry_position()[2]
                if laser_range < MIN_LASER_ALTITUDE_M:
                    aborted = f'laser altitude {laser_range:.3f}m dropped below {MIN_LASER_ALTITUDE_M}m'
                    break
                if ceiling_z - gant_z < CEILING_MARGIN_M:
                    aborted = (f'gantry came within {CEILING_MARGIN_M}m of the ceiling '
                               f'(gantry z={gant_z:.3f}m, ceiling z={ceiling_z:.3f}m)')
                    break
                deviations.append(laser_range - ideal_laser_range)
        finally:
            move_task.cancel()
            try:
                await move_task
            except asyncio.CancelledError:
                pass
            ob.slow_stop_all_spools()

        if aborted is not None:
            logger.warning(f'Linear height check aborted: {aborted}')
            return

        if not deviations:
            logger.warning('Linear height check collected no laser samples')
            return

        deviations_cm = np.array(deviations) * 100
        result_message = (
            f'Linear height check complete over {len(deviations_cm)} samples. '
            f'Ideal laser range {ideal_laser_range * 100:.1f}cm. '
            f'Mean deviation {deviations_cm.mean():+.2f}cm, '
            f'mean abs deviation {np.abs(deviations_cm).mean():.2f}cm, '
            f'RMS {np.sqrt((deviations_cm ** 2).mean()):.2f}cm, '
            f'min {deviations_cm.min():+.2f}cm, max {deviations_cm.max():+.2f}cm')
        logger.info(result_message)
        self.notify(f'RMS {np.sqrt((deviations_cm ** 2).mean()):.2f}cm')

    @verb('goalseek', motion=True)
    @prefer_swing_cancellation
    async def goalseek_diagnostic_task(self):
        """
        Measure how accurately seek_goal parks the gripper over a route-point tag.
        Triggered by the debug command "goalseek". This is a motion task.

        Cycles through the four floor tags ("gamepad", "trash", "hamper", "toys"),
        goal-seeking over each one in turn until every tag has been visited
        VISITS_PER_TAG times.

        Once parked over a tag, read where it appears in the gripper camera and work out
        where the gantry actually is relative to the tag, in room axes. Comparing that
        against the commanded offset gives the deviation; the RMS across all trials is
        reported in cm.
        """
        TAG_CYCLE = ['gamepad', 'trash', 'hamper', 'toys']
        VISITS_PER_TAG = 3
        SETTLE_S = 2.0           # let the gripper swing settle before measuring
        MEASURE_WINDOW_S = 2.0   # average tag readings over this much of a window
        MEASURE_TIMEOUT_S = 5.0  # give up on a trial if the tag isn't seen in this long

        # (m) where the gripper camera lens is asked to sit, straight above the tag
        LENS_HEIGHT_OVER_TAG = 0.6
        ob = self.ob
        # The gantry is where the lines meet. The pole runs from there down to the gripper
        # origin, and the lens sits a little above that origin.
        lens_in_body = Rotation.from_euler('x', 90, degrees=True).apply(model_constants.gripper_camera[1])
        gantry_over_lens = ob.pole_offset()[2] - lens_in_body[2]
        IDEAL_GANTRY_OVER_TAG = np.array([0.0, 0.0, LENS_HEIGHT_OVER_TAG + gantry_over_lens])
        # These tags mark drop points, so each one's saved position is basket_offset out
        # along the tag's normal from the tag itself. Lying on the floor, that is straight up.
        SAVED_POSITION_OVER_TAG = np.array([0.0, 0.0, model_constants.basket_offset[1][2]])

        async def measure_gantry_over_tag(tag_name):
            """Average room-frame (gantry position - tag position) over a short window, or
            None if the tag is never seen.

            Raw sightings are in the camera's own tilted optical frame, which is no use for
            an altitude comparison: the camera sits below the gantry by the pole, is offset
            toward the nose, and looks 9.06 degrees back from straight down, on top of
            whatever heading and swing the gripper has at that instant.
            gantry_minus_card unwinds all of that, using the gripper's orientation
            at each sample's capture time, so these averages are in room axes.
            """
            start = time.time()
            deadline = start + MEASURE_TIMEOUT_S
            while time.time() < deadline:
                samples = ob.route_tag_samples(tag_name, since=start)
                if samples and samples[-1][0] - samples[0][0] >= MEASURE_WINDOW_S:
                    break
                await asyncio.sleep(0.1)

            samples = ob.route_tag_samples(tag_name, since=start)
            if not samples:
                return None
            return np.mean([ob.gantry_minus_card(pose, timestamp=ts) for ts, pose in samples], axis=0)

        # the order of visits: each tag VISITS_PER_TAG times, cycling through the list
        visit_order = TAG_CYCLE * VISITS_PER_TAG
        num_trials = len(visit_order)

        deviations = []
        for trial, tag_name in enumerate(visit_order):
            saved_pos = ob.named_position(tag_name)
            if saved_pos is None:
                logger.warning(f'Goalseek trial {trial + 1}: no saved position for tag "{tag_name}", skipping')
                continue

            logger.info(f'Goalseek trial {trial + 1}/{num_trials}: seeking to tag "{tag_name}"')

            tag_pos = saved_pos - SAVED_POSITION_OVER_TAG
            await ob.seek_goal(tag_pos + IDEAL_GANTRY_OVER_TAG, auto_altitude=True)
            await asyncio.sleep(SETTLE_S)

            observed = await measure_gantry_over_tag(tag_name)
            if observed is None:
                logger.warning(f'Goalseek trial {trial + 1}: tag "{tag_name}" not seen in gripper camera, skipping')
                continue
            deviation = observed - IDEAL_GANTRY_OVER_TAG
            logger.info(f'Goalseek trial {trial + 1}: "{tag_name}" deviation {np.round(deviation * 100, 1)}cm '
                        f'(magnitude {np.linalg.norm(deviation) * 100:.2f}cm)\n'
                        f'gantry measured {np.round(observed, 3)}m from the tag')
            deviations.append(deviation)

        if not deviations:
            logger.warning('Goalseek diagnostic collected no measurements')
            return

        deviations = np.array(deviations)
        magnitudes_cm = np.linalg.norm(deviations, axis=1) * 100
        rms_cm = np.sqrt((magnitudes_cm ** 2).mean())
        per_axis_rms_cm = np.sqrt((deviations ** 2).mean(axis=0)) * 100
        logger.info(
            f'Goalseek diagnostic complete over {len(deviations)} trials. '
            f'RMS deviation {rms_cm:.2f}cm '
            f'(per-axis x={per_axis_rms_cm[0]:.2f}cm y={per_axis_rms_cm[1]:.2f}cm z={per_axis_rms_cm[2]:.2f}cm)')
