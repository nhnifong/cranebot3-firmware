"""Diagnostics: measurements of how well the robot flies, for tuning it."""

import asyncio
import json
import logging
import math
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

    @verb('stretchtest', motion=True)
    async def stretch_test_task(self, step_cm='10', top_margin_cm='20'):
        """
        Step the gantry up and down at its current xy, and at each stop record what the line
        lengths, the anchor cameras, the rangefinder and the line tensions say about where it is.
        Triggered by the debug command "stretchtest [step_cm] [top_margin_cm]". This is a
        motion task.

        After a vertical move the hang position (from line lengths) settles further up or down
        than the visual position. Each line's reported length minus its length modeled from the
        visual position, plotted against that line's tension, tells apart line stretch (a
        straight line in tension), a spool scale error (follows the change in length, not the
        tension) and a camera bias (the rangefinder agrees with the hang position instead).

        The sweep climbs from the start height in steps of step_cm until it is top_margin_cm
        below the lowest pull point, or a line passes TENSION_STOP_FRACTION of the safe tension.
        It then descends until the rangefinder would read less than MIN_LASER_RANGE_M, and
        climbs back to the start height, so most heights are visited moving up and moving down.

        Writes one JSON line per stop to stretch_test/<time>.jsonl under the output root: a
        header line first, then one line per stop with the raw samples, then an end line.
        """
        MIN_LASER_RANGE_M = 0.3
        # gantry height floor when the rangefinder has no reading, above the pole length
        NO_LASER_FLOOR_CLEARANCE_M = 0.5
        TENSION_STOP_FRACTION = 0.8
        SETTLE_TIMEOUT_S = 10.0
        # still means this many sightings from one anchor, all within this of their median
        SETTLE_MIN_SIGHTINGS = 6
        SETTLE_MAX_SPREAD_M = 0.04
        SAMPLE_S = 3.0
        SAMPLE_INTERVAL_S = 0.1
        ob = self.ob

        step_m = float(step_cm) / 100
        top_margin_m = float(top_margin_cm) / 100
        start = ob.gantry_position()
        top_z = float(np.min(ob.anchor_points()[:, 2])) - top_margin_m
        if start[2] > top_z:
            self.notify(f'Stretch test: start height {start[2]:.2f}m is already above the top '
                        f'of the sweep ({top_z:.2f}m). Move the gantry lower and try again.')
            return
        if ob.is_holding():
            logger.warning('Stretch test started while holding something; the tensions will include it.')

        # Only for the progress bar. The real bottom is decided by the rangefinder at each stop.
        laser0 = ob.laser_range()
        if laser0 is not None:
            bottom_estimate = start[2] - (laser0 - MIN_LASER_RANGE_M)
        else:
            bottom_estimate = ob.pole_offset()[2] + NO_LASER_FLOOR_CLEARANCE_M
        n_up = math.floor((top_z - start[2]) / step_m)
        n_down = max(0, math.floor((start[2] + n_up * step_m - bottom_estimate) / step_m))
        planned_stops = 1 + n_up + n_down + max(0, n_down - n_up)

        path = self.output_dir('stretch_test') / f'{time.strftime("%Y%m%d-%H%M%S")}.jsonl'
        logger.info(f'Stretch test writing to {path}, about {planned_stops} stops')
        stops = 0
        end_reason = 'cancelled'

        def write(f, record):
            f.write(json.dumps(record) + '\n')
            f.flush()

        async def settle():
            """Wait for a window of camera sightings in which the gantry held still. Returns the
            time the window started, or None if none turned up within SETTLE_TIMEOUT_S.

            ob.await_still_gantry_window reads sightings that are only kept during calibration,
            so this judges the same thing from the sighting buffer that is always filled."""
            deadline = time.time() + SETTLE_TIMEOUT_S
            window_start = time.time()
            while time.time() < deadline:
                await asyncio.sleep(0.2)
                sightings = ob.gantry_sightings(window_start)
                full = [sightings[sightings[:, 1] == a, 2:] for a in np.unique(sightings[:, 1])]
                full = [p for p in full if len(p) >= SETTLE_MIN_SIGHTINGS]
                if not full:
                    continue
                spread = max(np.max(np.linalg.norm(p - np.median(p, axis=0), axis=1)) for p in full)
                if spread <= SETTLE_MAX_SPREAD_M:
                    return window_start
                window_start = time.time()
            logger.warning(f'Stretch test: gantry not still within {SETTLE_TIMEOUT_S:.0f}s, recording anyway')
            return None

        async def visit(f, z, leg):
            """Fly to height z, wait for it to settle, record, and return the stop's record."""
            nonlocal stops
            self.progress(100.0 * stops / planned_stops, f'{leg} to {z:.2f}m')
            await ob.seek_goal(np.array([start[0], start[1], z]), auto_altitude=False)
            ob.slow_stop_all_spools()
            settled_since = await settle()

            sample_start = time.time()
            samples = []
            while time.time() - sample_start < SAMPLE_S:
                await asyncio.sleep(SAMPLE_INTERVAL_S)
                samples.append({
                    't': time.time(),
                    # (time, length m, speed m/s, tension N) per line
                    'lines': ob.line_records().tolist(),
                    'hang': ob.hang_position().tolist(),
                    'fused': ob.gantry_position().tolist(),
                    'laser': ob.laser_range(),
                })
            sightings = ob.gantry_sightings(settled_since if settled_since is not None else sample_start)
            record = {
                'stop': stops,
                'leg': leg,
                'goal_z': z,
                'settled': settled_since is not None,
                'samples': samples,
                # (time, anchor_num, x, y, z) for every camera sighting since it settled
                'sightings': sightings.tolist(),
            }
            write(f, record)
            stops += 1
            tensions = np.array([s['lines'] for s in samples])[:, :, 3]
            lasers = [s['laser'] for s in samples if s['laser'] is not None]
            logger.info(f'Stretch test stop {stops} ({leg}) goal z={z:.2f}m: '
                        f'hang z={samples[-1]["hang"][2]:.3f} '
                        f'visual z={np.mean(sightings[:, 4]) if len(sightings) else float("nan"):.3f} '
                        f'laser={np.mean(lasers) if lasers else float("nan"):.3f} '
                        f'max tension={tensions.max():.1f}N')
            return tensions.max(), (min(lasers) if lasers else None)

        with open(path, 'w') as f:
            write(f, {
                'header': True,
                'start_time': time.time(),
                'start': start.tolist(),
                'step_m': step_m,
                'top_z': top_z,
                'anchor_points': ob.anchor_points().tolist(),
                'pole_offset': ob.pole_offset().tolist(),
                'laser_offset': model_constants.laser_offset,
                'max_safe_tension': ob.max_safe_tension,
                'holding': ob.is_holding(),
                'line_columns': ['t', 'length_m', 'speed_mps', 'tension_n'],
                'sighting_columns': ['t', 'anchor_num', 'x', 'y', 'z'],
            })
            try:
                z = float(start[2])
                max_tension, min_laser = await visit(f, z, 'start')

                # climb to the top, or until a line nears the safe tension
                while z + step_m <= top_z + 1e-6:
                    if max_tension > TENSION_STOP_FRACTION * ob.max_safe_tension:
                        logger.info(f'Stretch test: {max_tension:.1f}N is near the safe tension, '
                                    f'not climbing further')
                        break
                    z += step_m
                    max_tension, min_laser = await visit(f, z, 'up')

                # descend until the next step would bring the gripper too near the floor
                while True:
                    if min_laser is not None:
                        if min_laser - step_m < MIN_LASER_RANGE_M:
                            break
                    elif z - step_m < ob.pole_offset()[2] + NO_LASER_FLOOR_CLEARANCE_M:
                        break
                    z -= step_m
                    _, min_laser = await visit(f, z, 'down')

                # climb back to the start height, so the low stops are seen from both directions
                while z + step_m <= start[2] + 1e-6:
                    z += step_m
                    await visit(f, z, 'return')
                end_reason = 'finished'
            finally:
                write(f, {'end': True, 'reason': end_reason, 'stops': stops, 'end_time': time.time()})
                ob.slow_stop_all_spools()
                logger.info(f'Stretch test {end_reason} after {stops} stops, wrote {path}')

        self.finish(f'{stops} stops written')
        self.notify(f'Stretch test finished: {stops} stops written to {path}')
