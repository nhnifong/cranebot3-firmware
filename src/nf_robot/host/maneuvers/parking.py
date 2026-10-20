"""Parking: recording where the hook is, settling the gantry onto it, and lifting it off."""

import asyncio
import logging
import time

import cv2
import numpy as np

import nf_robot.common.definitions as model_constants
import nf_robot.generated.nf.config as nf_config
from nf_robot.common.cv_common import get_inward_wall_normal, get_wall_escape_direction
from nf_robot.common.pose_functions import average_pose
from nf_robot.common.util import fromnp, poseProtoToTuple, poseTupleToProto, tonp
from nf_robot.generated.nf import control, telemetry
from nf_robot.host.maneuver import Maneuver, TiltWatch, command, startup_step

logger = logging.getLogger(__name__)

# (metres) how far out along the escape direction the mouth of the fork is: from there the
# gantry goes straight down the track into the fork with no room to correct. record_park
# records the gantry tag pose from here and park steers back to it.
MOUTH_OFFSET_M = 0.085
# (degrees) how far the track in and out of the parking hook leans along the wall, off the
# wall's inward normal and towards the anchor end of it. Unpark leaves along this line and
# park comes back down it, so one constant sets both.
ESCAPE_TILT_DEG = 45.0
# (degrees, -90 open to 90 closed) the fingers are held at around the hook. Closed past about
# 60 they cross the rangefinder's beam, and every range park and record_park take would be
# the distance to the fingers instead of to what is under the hook.
FINGER_ANGLE_PARKED = 50
# (seconds) for the fingers to get there and the rangefinder to report past them
FINGER_SETTLE_S = 1.5
# Sightings of the gantry tag one anchor has to deliver to describe where it is now.
MOUTH_TAG_MIN_SIGHTINGS = 5
# Shown when record_park cannot see the gantry tag from the mouth of the fork, which park
# needs to steer by.
TAG_NOT_SEEN_MESSAGE = ('This parking hook location or lighting does not allow the marker to '
                        'be seen. See parking hook install guide '
                        'https://neufangled.com/docs/arp_install_guide/#mounting-the-parking-hook')


class Parking(Maneuver):
    name = 'parking'
    title = 'Parking'
    config_field = 'park_data'

    def publish_parking_state(self):
        """Tell the UI whether the gantry is parked and whether there is a hook to park on."""
        data = self.data
        self.ob.send_ui(parking_state=telemetry.ParkingState(
            parked=bool(data is not None and data.parked),
            location_recorded=bool(data is not None and data.pos is not None),
            present=True,
        ))

    def send_setup_telemetry(self):
        self.publish_parking_state()
        if self.data is not None and self.data.pos is not None:
            self.ob.send_ui(named_position=telemetry.NamedObjectPosition(
                name='parking_location',
                position=self.data.pos,
            ))

    @startup_step('unpark')
    async def unpark_if_parked(self):
        # Only if the robot was left on the hook. Unparking one that is already flying
        # drops it 10cm and shoves it at the nearest wall, on the strength of a position
        # estimate that has nothing to do with where it is.
        if self.data is not None and self.data.parked:
            await self.unpark()

    @startup_step('park')
    async def park_if_recorded(self):
        # Parking needs somewhere to park; without a recorded location the robot is better
        # left hanging.
        if self.data is not None and self.data.pos is not None:
            await self.park()
        else:
            logger.info('No parking location recorded, so leaving the robot where it is')

    def set_parked(self, parked):
        """Record whether the gantry is on the hook, and write it out.

        Saved rather than held in memory because the question outlives the process: a host
        restarted while the robot hangs on the wall has no way to look and see.
        """
        if self.data is None:
            self.data = nf_config.ParkData()
        if self.data.parked == parked:
            return
        self.data.parked = parked
        self.save_data()
        self.publish_parking_state()
        logger.info(f'Robot is {"parked" if parked else "not parked"}')

    def gantry_tag_now(self, window_s, after=None, anchor=None, min_sightings=MOUTH_TAG_MIN_SIGHTINGS):
        """The gantry tag as one anchor camera sees it now: (anchor_num, (rvec, tvec)).

        The average of that camera's sightings in the window, in its own frame. anchor picks
        the camera; otherwise it is whichever saw the tag most. None if no camera delivered
        min_sightings.
        """
        seen = self.ob.raw_gantry_tag_sightings(window_s, after=after)
        if anchor is not None:
            seen = {anchor: seen.get(anchor, [])}
        if not seen:
            return None
        num, rows = max(seen.items(), key=lambda kv: len(kv[1]))
        if len(rows) < min_sightings:
            return None
        poses = [(np.asarray(p[0], dtype=float).reshape(3), np.asarray(p[1], dtype=float).reshape(3))
                 for _, p in rows]
        return num, average_pose(poses)

    def tag_offset_to_room(self, anchor, now_pose):
        """The room-frame move that would put the gantry tag back where it was recorded.

        Both poses are from the same camera, so the difference of their translations is how
        far the tag has to move in that camera's frame, and only the camera's room rotation
        is needed to say which way that is in the room. The camera's room position, the
        position estimate and the tag-to-gantry offset all cancel out; the last one because
        the gantry does not turn, so the tag and the gantry move together. None if the
        anchor is not connected.
        """
        camera = self.ob.anchor_camera_pose(anchor)
        if camera is None:
            return None
        rotation, _ = cv2.Rodrigues(np.asarray(camera[0], dtype=float))
        recorded = poseProtoToTuple(self.data.mouth_tag_pose)
        return rotation @ (recorded[1] - now_pose[1])

    @command(control.Command.RECORD_PARK, motion=True)
    async def record_park(self):
        """Record where the parking hook is, and how an anchor camera sees the gantry tag
        from the mouth of it.

        Ends back down on the hook
        This is a motion task.
        """
        LIFT_M = 0.10               # up off the hook, the same move unpark starts with
        LIFT_SPEED_MPS = 0.04
        SETTLE_S = 2.0              # let the pole stop swinging before reading the tag
        CAPTURE_TIMEOUT_S = 5.0
        CONFIRM_TIMEOUT_S = 120.0   # long enough to walk over and look at the hook

        ob = self.ob
        try:
            if not ob.gripper_connected():
                logger.warning('Cannot record a parking location without a connected gripper')
                return

            # Everything recorded here is measured from wherever the gantry happens to be,
            # and none of it can tell whether that is the hook or thin air a foot below it.
            # Ask, rather than write a parking location that will fly the robot at a wall.
            answer = await self.ask(
                'Is the marker resting in the parking hook now?',
                buttons=['Yes', 'No'], timeout=CONFIRM_TIMEOUT_S)
            if answer != 0:
                logger.info('Record park: answered no' if answer == 1 else
                            'Record park: nobody answered, so nothing was recorded')
                return

            hook_pos = ob.gantry_position()
            await ob.set_finger_angle(FINGER_ANGLE_PARKED)
            await asyncio.sleep(FINGER_SETTLE_S)
            # taken now, resting on the hook, which is the one moment this reading means
            # "the height the hook holds the gantry at"
            parked_range = ob.laser_range()
            if parked_range is None:
                logger.warning('Record park: no rangefinder reading on the hook; park will '
                               'have nothing to confirm its height against')

            # All of this happens within 10cm of the hook, so swing cancellation stays off
            # throughout and is not restored afterwards, the same as park.
            ob.set_swing_cancellation(False)

            # 1. up off the hook.
            logger.info(f'Record park: lifting {LIFT_M * 100:.0f}cm from '
                        f'{np.round(hook_pos, 3)}')
            await ob.nudge_gantry(np.array([0.0, 0.0, LIFT_M]), speed=LIFT_SPEED_MPS)

            # 2. nose away from the wall. park turns back to the angle this leaves the wrist
            # at, so the gripper meets the hook the way it sat on it here.
            away = get_inward_wall_normal(hook_pos, ob.anchor_points())
            heading = ob.wrist_angle_for_heading(float(np.arctan2(away[0], away[1])))
            logger.info(f'Record park: turning the nose to face {np.round(away, 3)}, '
                        f'wrist heading {heading:.0f} degrees')
            await ob.settle_wrist_to_heading(heading)
            await asyncio.sleep(SETTLE_S)
            wrist_angle = ob.wrist_angle()

            # The track in and out of the hook, worked out here from the hook itself so park
            # gets the line unpark would have left along without rederiving it from wherever
            # it happens to be standing when it is asked to come home.
            escape = get_wall_escape_direction(hook_pos, ob.anchor_points(),
                                               anchor_indices=model_constants.ANCHOR_MOUNTED_POINTS,
                                               tilt_deg=ESCAPE_TILT_DEG)

            # 3. out to the mouth of the fork, which is where the tag pose has to be taken
            # from: it is the last place park can still correct itself.
            hover_pos = ob.gantry_position()
            logger.info(f'Record park: stepping {MOUTH_OFFSET_M * 100:.1f}cm out to the '
                        f'mouth of the fork along {np.round(escape, 3)}')
            await ob.nudge_gantry(np.array([escape[0], escape[1], 0.0]) * MOUTH_OFFSET_M,
                                  speed=LIFT_SPEED_MPS)
            settled_at = time.time()
            await asyncio.sleep(SETTLE_S)

            # The gantry tag as the anchor nearest the hook sees it from here. Parked it is
            # partly hidden and goes undetected, but at the mouth it is in view, and park
            # can servo back onto this one camera's view of it. Only frames from after the
            # move stopped, and the settle, count.
            mouth_tag = None
            tag_deadline = time.time() + CAPTURE_TIMEOUT_S
            while mouth_tag is None and time.time() < tag_deadline:
                mouth_tag = self.gantry_tag_now(window_s=SETTLE_S, after=settled_at + SETTLE_S / 2)
                if mouth_tag is None:
                    await asyncio.sleep(0.2)
            # the range from here, which is what park sets its own altitude by
            mouth_range = ob.laser_range()
            if mouth_range is None:
                logger.warning('Record park: no rangefinder reading at the mouth; park will '
                               'have to take its altitude from the position estimate')

            if mouth_tag is None:
                # park has nothing to steer by without it, so a location recorded without it
                # would only fly the robot at the hook blind
                logger.warning('Record park: no anchor camera saw the gantry tag at the mouth; '
                               'parking location not saved')
                self.notify(TAG_NOT_SEEN_MESSAGE)
            else:
                logger.info(f'Record park: anchor {mouth_tag[0]} sees the gantry tag at the mouth '
                            f'at {np.round(mouth_tag[1][1], 3)} in its camera frame')
                # the hovering position, not this one: the mouth is derived from it and the
                # escape direction, and the descent is measured from it
                self.data.pos = fromnp(hover_pos)
                self.data.escape_direction = fromnp(np.array([escape[0], escape[1], 0.0]))
                self.data.wrist_angle = wrist_angle
                self.data.parked_range = parked_range or 0.0
                self.data.mouth_range = mouth_range or 0.0
                self.data.mouth_tag_anchor = int(mouth_tag[0])
                self.data.mouth_tag_pose = poseTupleToProto(mouth_tag[1])
                logger.info(f'Record park: range {parked_range} on the hook, '
                            f'{mouth_range} at the mouth')
                self.save_data()
                self.publish_parking_state()
                ob.send_ui(named_position=telemetry.NamedObjectPosition(
                    name='parking_location',
                    position=self.data.pos,
                ))

            # 5. back in and down onto the hook, the way park comes in rather than straight
            # at it: in along the track first, then down, measured against where this
            # started so the drift the moves picked up is undone.
            await ob.nudge_gantry(np.array([-escape[0], -escape[1], 0.0]) * MOUTH_OFFSET_M,
                                  speed=LIFT_SPEED_MPS)
            await ob.nudge_gantry(hook_pos - ob.gantry_position(), speed=LIFT_SPEED_MPS)
            # it went back to where it was, and the operator just confirmed that was the hook
            self.set_parked(True)
        except asyncio.CancelledError:
            logger.info('Record park cancelled')
            raise
        finally:
            ob.slow_stop_all_spools()
            ob.set_swing_cancellation(False)

    async def _straighten_pole(self, tilt, caught):
        """Move the gantry sideways until the pole stands up again. None once it does, or why
        it could not - caught (the tilt watch's complaint) included.

        Only used while entering the fork, where the plausible cause of a lean is the pole
        pivoting on the hook. The gantry is a pivot with the gripper hanging below it, so the
        lean is the gantry no longer being over the gripper; moving the gantry back over it
        straightens the pole, where retrying from scratch only repeats whatever caused it.
        Elsewhere a lean means something else, and the attempt is failed instead. Slow, and bounded in distance
        and in how many times one tilt watch may ask for it, since a pole that will not
        straighten is caught on something moving it cannot fix.
        """
        STRAIGHT_FRACTION = 0.5       # of the watch's limit, the lean to stop at
        SPEED_MPS = 0.01
        MAX_TRAVEL_M = 0.05
        TIMEOUT_S = 10.0
        LOOP_S = 0.1
        MAX_STRAIGHTENS = 3           # per tilt watch

        ob = self.ob
        tilt.straightens = getattr(tilt, 'straightens', 0) + 1
        if tilt.straightens > MAX_STRAIGHTENS:
            return f'{caught}, and it has already been straightened {MAX_STRAIGHTENS} times'
        target = tilt.tilt_deg * STRAIGHT_FRACTION
        start = ob.gantry_position()
        deadline = time.time() + TIMEOUT_S
        logger.info(f'Park: {caught}; moving the gantry to straighten it')
        try:
            while True:
                lean = ob.pole_tilt()
                direction = ob.pole_straighten_direction()
                if lean is None or direction is None:
                    return f'{caught}, and the gripper does not say which way'
                if lean <= target:
                    moved = float(np.linalg.norm(ob.gantry_position()[:2] - start[:2]))
                    logger.info(f'Park: pole straightened to {lean:.1f} degrees after moving '
                                f'{moved * 100:.1f}cm')
                    tilt.leaning_since = None
                    return None
                here = ob.gantry_position()
                if float(np.linalg.norm(here[:2] - start[:2])) > MAX_TRAVEL_M:
                    return (f'{caught}, and it was still {lean:.0f} degrees after moving '
                            f'{MAX_TRAVEL_M * 100:.0f}cm to straighten it')
                if time.time() > deadline:
                    return (f'{caught}, and it was still {lean:.0f} degrees after '
                            f'{TIMEOUT_S:.0f}s straightening it')
                await ob.move_direction_speed(np.array([direction[0], direction[1], 0.0]) * SPEED_MPS,
                                              None, here, key=self.velocity_key())
                await asyncio.sleep(LOOP_S)
        finally:
            await ob.move_direction_speed(np.zeros(3), 0, key=self.velocity_key())

    async def _enter_hook(self, track, tilt, distance):
        """Move in along the track by distance, from wherever the visual close left off.

        Relative to where this move starts rather than towards a recorded position: the
        close just moved the gantry by however much the estimate had wrong, and the estimate
        did not see an error, only a move. Aiming at a stored point from here would take that
        correction straight back out.

        Nothing steers during this: from the mouth of the fork there is no room to correct,
        so all that is left is to go in and watch the pole. None if it got the whole way.
        """
        SPEED_MPS = 0.03              # slower than the approach: this is the part that is blind
        LOOP_S = 0.1

        start = self.ob.gantry_position()
        logger.info(f'Park: entering the fork, {distance * 100:.1f}cm in along the track')
        while True:
            caught = tilt.check()
            if caught:
                failed = await self._straighten_pole(tilt, caught)
                if failed:
                    return f'{failed} entering the fork'
                continue
            here = self.ob.gantry_position()
            travelled = float(np.dot(here[:2] - start[:2], -track))
            if travelled >= distance:
                logger.info(f'Park: {travelled * 100:.1f}cm in, steepest lean '
                            f'{tilt.worst_tilt:.1f} degrees')
                return None
            await self.ob.move_direction_speed(np.array([-track[0], -track[1], 0.0]) * SPEED_MPS,
                                               None, here, key=self.velocity_key())
            await asyncio.sleep(LOOP_S)

    async def _approach_hook_on_tag(self, lateral, tilt):
        """Close the last of the approach by steering the gantry tag back to where one anchor
        camera saw it when the parking location was recorded.

        The error each pass is the recorded camera-frame tag position minus the current one,
        from the same camera, turned into the room by that camera's rotation - see
        tag_offset_to_room. The position estimate is not used to decide where to go, only by
        move_direction_speed to work out line speeds. All three axes come from the tag, so
        this also sets the height.

        lateral is the attempt's offset from the recorded spot.
        None once it arrives, or why it stopped.
        """
        LOOP_S = 0.2
        TIMEOUT_S = 30.0
        WINDOW_S = 0.6                # sightings this recent describe where the tag is now
        MIN_SIGHTINGS = 2             # ...and this many of them, from the recorded anchor
        ARRIVED_M = 0.005
        ARRIVED_PASSES = 3
        STALL_S = 6.0
        IMPROVEMENT_M = 0.001
        BLIND_LIMIT_S = 6.0
        GAIN = 1.0
        MIN_SPEED_MPS = 0.006
        MAX_SPEED_MPS = 0.05
        MAX_ERROR_M = 0.45            # a tag reading this far off is not one to fly at
        SMOOTHING = 0.5

        anchor = self.data.mouth_tag_anchor
        target = np.array([lateral[0], lateral[1], 0.0])
        logger.info(f'Park: closing on the gantry tag pose anchor {anchor} recorded')
        deadline = time.time() + TIMEOUT_S
        last_reading = time.time()
        close_passes = 0
        smoothed = None
        best = np.inf
        best_at = time.time()
        while True:
            caught = tilt.check()
            if caught:
                return f'{caught} on the way in'
            if time.time() > deadline:
                return 'took too long to close on the recorded gantry tag pose'

            move = None
            seen = self.gantry_tag_now(WINDOW_S, anchor=anchor, min_sightings=MIN_SIGHTINGS)
            if seen is not None:
                move = self.tag_offset_to_room(anchor, seen[1])
            if move is None:
                if time.time() - last_reading > BLIND_LIMIT_S:
                    return (f'anchor {anchor} has not seen the gantry tag for '
                            f'{BLIND_LIMIT_S:.0f}s')
                await self.ob.move_direction_speed(np.zeros(3), 0, key=self.velocity_key())
                await asyncio.sleep(LOOP_S)
                continue
            last_reading = time.time()

            move = move + target
            smoothed = move if smoothed is None else smoothed * SMOOTHING + move * (1 - SMOOTHING)
            move = smoothed
            error = float(np.linalg.norm(move))
            logger.debug(f'Park: tag says move {np.round(move * 100, 1)}cm')
            if error > MAX_ERROR_M:
                return f'the gantry tag puts the mouth {error:.2f}m away, too far to trust'

            if error < best - IMPROVEMENT_M:
                best, best_at = error, time.time()
            elif time.time() - best_at > STALL_S:
                logger.warning(f'Park: the tag stopped closing at {error * 100:.1f}cm from this '
                               f'attempt\'s aim point; steepest lean on the way in was '
                               f'{tilt.worst_tilt:.1f} degrees')
                return None

            if error < ARRIVED_M:
                close_passes += 1
                if close_passes >= ARRIVED_PASSES:
                    from_mouth = float(np.linalg.norm(move - target))
                    logger.info(f"Park: {error * 100:.1f}cm from this attempt's aim point and "
                                f"{from_mouth * 100:.1f}cm from the recorded mouth by the tag; "
                                f"steepest lean on the way in was {tilt.worst_tilt:.1f} degrees")
                    return None
            else:
                close_passes = 0

            speed = min(MAX_SPEED_MPS, max(MIN_SPEED_MPS, error * GAIN))
            await self.ob.move_direction_speed(move / error * speed, None,
                                               self.ob.gantry_position(), key=self.velocity_key())
            await asyncio.sleep(LOOP_S)

    @command(control.Command.PARK, motion=True)
    async def park(self):
        """Fly home and settle the gantry onto the parking hook.

        The hook is a diagonal slot built around the flat marker.
        It is entered along the same line unpark leaves by.
        First we fly to a point HANDOVER_M short of the mouth of the fork.
        The gantry tag, as the anchor camera that recorded it sees it, steers the rest of the way
        to the mouth. From there it goes in along the track blind and lowers,
        looking for the signature of a successful park, using tension and laser rangefinder.

        Swing cancellation is worth having for the flight out and is switched off for good once the
        tag takes over. it's very important that it isn't turned on in the vicinity of the hook.
        That would almost certainly damage something.

        This is a motion task.
        """
        HANDOVER_M = 0.10             # how far out beyond the mouth the flight ends and
                                      # the tag takes over.
        DESCENT_M = 0.30              # how far down to look for the hook
        DESCENT_SPEED_MPS = 0.03      # to start with, and ramping up the longer the hook
        DESCENT_ACCEL_MPS2 = 0.015    # takes to take the weight: lines that are nearly slack
        DESCENT_MAX_SPEED_MPS = 0.12  # but not quite need paying out faster to let go
        DESCENT_LOOP_S = 0.1
        # Maximum permissible pole tilt while trying to park
        TILT_DEG = 10.0
        TILT_CONFIRM_S = 0.2
        # offsets to apply to subsequent attempts. from the recorded parking spot
        # y is aligned to the track into the mouth of the hook. X is purpendicular to the track.
        ATTEMPT_OFFSETS_M = (
            (0.0, 0.0, 0.0),
            (0.0, 0.02, 0.0),
            (0.0, -0.02, 0.0),
        )
        PARK_ATTEMPTS = len(ATTEMPT_OFFSETS_M)
        BACKOUT_LIFT_M = 0.10         # up off whatever it is touching before retreating
        BACKOUT_SPEED_MPS = 0.05
        PARKED_FRACTION = 0.3         # total line tension under this share of the total
                                      # free-hanging tension is the hook having taken the
                                      # weight. Judged on the sum, not line by line: a
                                      # lightly loaded line's limit can sit under what its
                                      # sensor reads fully slack, and one line's jitter
                                      # would keep resetting the confirm timer
        PARKED_CONFIRM_S = 0.4
        PARKED_RANGE_TOL_M = 0.15     # how far off the recorded parked range still counts
        RANGE_TRIM_TOL_M = 0.01
        RANGE_TRIM_STEPS = 5
        RANGE_TRIM_TRAVEL_M = 0.30
        SWING_SETTLE_S = 3.0          # after the approach, before steering by the tag
        # Lined up at the mouth is the last thing that can be checked - past it the gantry
        # is inside the fork and the tag is partly hidden.
        ALIGNED_M = 0.10             # lined up at the mouth, before going in
        SETTLE_S = 2.0

        ob = self.ob

        def stop_short(reason):
            logger.warning(f'Park stopped: {reason}')

        try:
            # TODO check if holding something, if so warn user and do not proceed.
            park_data = self.data
            if park_data is None or park_data.pos is None or park_data.escape_direction is None:
                stop_short('no parking location has been recorded; run Set Parking Location '
                           'from the hook first')
                return

            parkpos = tonp(park_data.pos)
            track = tonp(park_data.escape_direction)[:2]
            track = track / (np.linalg.norm(track) + 1e-9)  # outward along the slot
            if park_data.mouth_tag_pose is None:
                # a location recorded before parking steered by the tag, or one where the tag
                # could not be seen; either way there is nothing to steer onto the hook by
                stop_short('no gantry tag pose was recorded at the mouth of the hook')
                self.notify('The parking location needs to be recorded again. Rest the marker '
                            'in the parking hook and run Set Parking Location.')
                return
            if ob.anchor_camera_pose(park_data.mouth_tag_anchor) is None:
                stop_short(f'anchor {park_data.mouth_tag_anchor}, which recorded the gantry '
                           f'tag at the mouth, is not connected')
                return

            # across the track, so an attempt offset can be written in the track's frame
            sideways = np.array([-track[1], track[0]])

            def in_room(offset):
                """An (across, along, up) attempt offset as a room-frame vector."""
                across, along, up = offset
                return np.array([sideways[0] * across + track[0] * along,
                                 sideways[1] * across + track[1] * along,
                                 up])

            # the flight out is long enough for the fingers to clear the rangefinder
            await ob.set_finger_angle(FINGER_ANGLE_PARKED)

            # 1. Fly to standoff position
            def handover_point(offset):
                return (parkpos + offset + np.array([track[0], track[1], 0.0])
                        * (MOUTH_OFFSET_M + HANDOVER_M))

            first = handover_point(in_room(ATTEMPT_OFFSETS_M[0]))
            logger.info(f'Park: flying to {np.round(first, 3)}, where the tag takes over, '
                        f'{(MOUTH_OFFSET_M + HANDOVER_M) * 100:.1f}cm out from '
                        f'{np.round(parkpos, 3)} along {np.round(track, 3)}')
            async with ob.prefer_swing_cancellation():
                await ob.seek_goal(first)
                await asyncio.sleep(SETTLE_S) # Allow swing cancellation to damp

            # 2. Turn swing cancellation off for good, and the wrist back
            # to the angle record_park left it at.
            ob.set_swing_cancellation(False)
            if ob.gripper_connected() and park_data.wrist_angle:
                await ob.settle_wrist_to_heading(park_data.wrist_angle)
            await asyncio.sleep(SETTLE_S)

            # move to the laser range that was recorded at the hook mouth.
            # If anything was placed below the hook, it invalidates this method.
            if park_data.mouth_range:
                before_z = float(ob.gantry_position()[2])
                await ob.trim_altitude_to_range(park_data.mouth_range, tol_m=RANGE_TRIM_TOL_M,
                                                max_steps=RANGE_TRIM_STEPS,
                                                max_travel_m=RANGE_TRIM_TRAVEL_M)
                moved = float(ob.gantry_position()[2]) - before_z
                parkpos[2] += moved
                logger.info(f'Park: laser put the hover height {moved * 100:+.1f}cm from the '
                            f'recorded one; working from {np.round(parkpos, 3)}')
            else:
                logger.warning('Park: no mouth range recorded, so the altitude is whatever '
                               'the position estimate says')

            def check_tag_alignment(lateral):
                """Why the tag says this attempt is not lined up at the mouth, or None."""
                anchor = park_data.mouth_tag_anchor
                seen = self.gantry_tag_now(SETTLE_S, anchor=anchor)
                off = None if seen is None else self.tag_offset_to_room(anchor, seen[1])
                if off is None:
                    logger.warning(f'Park: anchor {anchor} cannot see the gantry tag to '
                                   f'check the alignment; going in without checking it')
                    return None
                off = off + np.array([lateral[0], lateral[1], 0.0])
                logger.info(f"Park: by the tag, this attempt's aim point is "
                            f"{np.round(off * 100, 1)}cm away")
                if float(np.linalg.norm(off)) > ALIGNED_M:
                    return (f"the tag puts this attempt's aim point "
                            f"{np.linalg.norm(off) * 100:.1f}cm away - not lined up with it")
                return None

            async def attempt(offset):
                """One go at getting onto the hook, aiming offset metres from the recorded
                spot. None if it worked, or why it did not."""
                # the mouth of the fork, which is where the tag pose was recorded from
                lateral = offset[:2]
                mouth = parkpos + offset + np.array([track[0], track[1], 0.0]) * MOUTH_OFFSET_M
                tilt = TiltWatch(ob, tilt_deg=TILT_DEG, confirm_s=TILT_CONFIRM_S)
                logger.info(f'Park: closing on the mouth of the fork at {np.round(mouth, 3)}')

                # 2. Close the distance using the gantry tag
                ob.slow_stop_all_spools()
                await asyncio.sleep(SWING_SETTLE_S)
                failed = await self._approach_hook_on_tag(lateral, tilt)
                if failed:
                    return failed

                ob.slow_stop_all_spools()
                await asyncio.sleep(SETTLE_S)

                # Report alignment numbers before moving in.
                failed = check_tag_alignment(lateral)
                if failed:
                    return failed

                # 4. Blindly move in along a track. we may or may not be aligned.
                # use various readings to look for the signature of a successful park
                failed = await self._enter_hook(track, tilt, MOUTH_OFFSET_M)
                if failed:
                    return failed
                ob.slow_stop_all_spools()
                await asyncio.sleep(SETTLE_S)

                # 5. lower until the hook takes the weight.
                free = await ob.measure_free_tension()
                parked_limit = float(np.sum(free)) * PARKED_FRACTION
                tilt = TiltWatch(ob, tilt_deg=TILT_DEG, confirm_s=TILT_CONFIRM_S)
                logger.info(f'Park: lowering up to {DESCENT_M * 100:.0f}cm, parked when the '
                            f'lines total under {parked_limit:.2f}N (free '
                            f'{np.round(free, 2)}N)')
                start_z = float(ob.gantry_position()[2])
                tension = ob.line_tensions()
                slack_since = None
                landed = False
                descent_started = time.time()
                while float(ob.gantry_position()[2]) > start_z - DESCENT_M:
                    caught = tilt.check()
                    if caught:
                        return f'{caught} on the way down'
                    tension = tension * 0.7 + ob.line_tensions() * 0.3
                    # how far the total still is above the parked limit; zero is parked
                    total = float(np.sum(tension))
                    residual = max(total - parked_limit, 0.0)
                    speed = min(DESCENT_MAX_SPEED_MPS, DESCENT_SPEED_MPS
                                + DESCENT_ACCEL_MPS2 * (time.time() - descent_started))
                    logger.debug(f'Park: {start_z - float(ob.gantry_position()[2]):.3f}m down '
                                 f'at {speed * 100:.1f}cm/s, residual tension {residual:.2f}N over the limit '
                                 f'(total {total:.2f}N, lines {np.round(tension, 2)}N)')
                    if total < parked_limit:
                        if slack_since is None:
                            slack_since = time.time()
                        elif time.time() - slack_since > PARKED_CONFIRM_S:
                            landed = True
                            logger.info(
                                f'Park: the hook has it, {np.round(tension, 2)}N on the lines '
                                f'after {start_z - float(ob.gantry_position()[2]):.3f}m down, '
                                f'steepest lean {tilt.worst_tilt:.1f} degrees')
                            break
                    else:
                        slack_since = None
                    await ob.move_direction_speed(np.array([0.0, 0.0, -speed]),
                                                  None, ob.gantry_position(),
                                                  key=self.velocity_key(), downward_bias=0)
                    await asyncio.sleep(DESCENT_LOOP_S)
                ob.slow_stop_all_spools()

                if not landed:
                    return (f'lowered the full {DESCENT_M * 100:.0f}cm without the lines going '
                            f'slack ({np.sum(tension):.2f}N total, '
                            f'{np.round(tension, 2)}N per line)')

                # Slack lines say something is holding the gantry up. The range says it is
                # being held at the height the hook holds it at, which the front edge of the
                # fork and anything else it might have come to rest on are not.
                if park_data.parked_range:
                    measured = ob.laser_range()
                    if measured is None:
                        logger.warning('Park: no rangefinder reading to confirm the height with')
                    elif abs(measured - park_data.parked_range) > PARKED_RANGE_TOL_M:
                        return (f'the lines went slack at range {measured:.3f}m, '
                                f'{abs(measured - park_data.parked_range) * 100:.1f}cm off the '
                                f'{park_data.parked_range:.3f}m recorded on the hook')
                    else:
                        logger.info(f'Park: range {measured:.3f}m confirms the parked height')

                return None

            async def back_out(offset):
                """Lift clear of the hook and retreat to where the next attempt takes over
                from, which is offset metres from the first one's."""
                logger.info(f'Park: backing off the hook, next try aims '
                            f'{np.round(offset * 100, 1)}cm off the recorded spot')
                # up first: everything that ends an attempt leaves the pole touching something
                await ob.nudge_gantry(np.array([0.0, 0.0, BACKOUT_LIFT_M]),
                                      speed=BACKOUT_SPEED_MPS)
                target = handover_point(offset)
                await ob.nudge_gantry(target - ob.gantry_position(), speed=BACKOUT_SPEED_MPS,
                                      max_step=0.5)
                await asyncio.sleep(SETTLE_S)

            parked = False
            for attempt_no, step in enumerate(ATTEMPT_OFFSETS_M, start=1):
                failed = await attempt(in_room(step))
                if failed is None:
                    parked = True
                    break
                stop_short(f'attempt {attempt_no} of {PARK_ATTEMPTS} '
                           f'(across {step[0] * 100:+.1f}, along {step[1] * 100:+.1f}, '
                           f'up {step[2] * 100:+.1f} cm): {failed}')
                if attempt_no < PARK_ATTEMPTS:
                    await back_out(in_room(ATTEMPT_OFFSETS_M[attempt_no]))
            if not parked:
                logger.warning(f'Park: giving up after {PARK_ATTEMPTS} attempts')
                return

            # for looks, as well as to let me know it finished.
            await ob.set_finger_angle(FINGER_ANGLE_PARKED)
            self.set_parked(True)
            logger.info('Park complete')
        except asyncio.CancelledError:
            logger.info('Park cancelled')
            raise
        finally:
            # slow_stop_all_spools only zeroes the default source, and a velocity left on
            # this one would be summed back into the next move anything else commands.
            await ob.move_direction_speed(np.zeros(3), 0, key=self.velocity_key())
            ob.slow_stop_all_spools()
            await ob.clear_goal()
            # Deliberately not restoring swing cancellation: however this ended, the gantry
            # is at or near the hook, which is the one place it must not come back on.
            ob.set_swing_cancellation(False)

    @command(control.Command.UNPARK, motion=True)
    async def unpark(self):
        """Lift the gantry off the parking hook and fly it clear into the room.

        Parked, no anchor camera can see the gantry marker, so the position estimate is
        running on where the robot was shut down: enough to rise, step clear of the wall and
        creep towards the middle of the work volume until a camera picks the marker up, and nothing
        is trusted for more than that until the half calibration at the end. The step out
        leans along the wall toward the anchor holding that end of it, which walks out from
        under the hook rather than only backwards off it, and towards the camera that has to
        find the marker again.
        This is a motion task.
        """
        LIFT_M = 0.10                 # straight up, to clear the hook
        LIFT_SPEED_MPS = 0.05
        CLEAR_M = 0.20                # diagonal step out from the wall
        CLEAR_SPEED_MPS = 0.05
        CRUISE_SPEED_MPS = 0.10       # the creep in towards the middle of the work volume
        CRUISE_LOOP_S = 0.1
        CRUISE_TIMEOUT_S = 90.0
        CENTER_PROXIMITY_M = 0.3      # near enough the middle that there is no more room to use
        UNPARK_TILT_DEG = 15.0        # looser than park's: a traverse swings the pole about
        SIGHTING_WINDOW_S = 1.0       # how recent a sighting has to be to describe now
        MIN_SIGHTINGS = 3             # this many inside that window means the marker is in view

        ob = self.ob

        def stop_short(reason):
            logger.warning(f'Unpark stopped: {reason}')

        try:
            # Wherever the estimate says the gantry is, is where it is started from: the
            # filter was seeded with the last position of the previous session and, parked,
            # has had nothing to revise it with.
            start_pos = ob.gantry_position()

            # 1. straight up, off the hook.
            logger.info(f'Unpark: lifting {LIFT_M * 100:.0f}cm from {np.round(start_pos, 3)}')
            await ob.nudge_gantry(np.array([0.0, 0.0, LIFT_M]), speed=LIFT_SPEED_MPS)

            # 2. clear of the wall, diagonally, toward the anchor end of it.
            anchor_points = ob.anchor_points()
            escape = get_wall_escape_direction(start_pos, anchor_points,
                                               anchor_indices=model_constants.ANCHOR_MOUNTED_POINTS,
                                               tilt_deg=ESCAPE_TILT_DEG)
            logger.info(f'Unpark: stepping {CLEAR_M * 100:.0f}cm clear of the wall '
                        f'along {np.round(escape, 3)}')
            await ob.nudge_gantry(np.array([escape[0], escape[1], 0.0]) * CLEAR_M,
                                  speed=CLEAR_SPEED_MPS)
            # clear of the hook from here, whatever becomes of the rest of this
            self.set_parked(False)

            # 3. in and down towards the middle of the work volume until a camera finds the
            # marker. The pole hangs free by now, so it leaning means the gantry has run into
            # something on its way out. The target is the middle of the volume rather than of
            # the floor plan: the hook is high on a wall, so descending as it comes in crosses
            # more of what the anchor cameras cover. The floor is z=0, so half the anchor
            # plane height is the middle of it.
            center = np.array([*np.mean(anchor_points[:, :2], axis=0),
                               float(np.mean(anchor_points[:, 2])) / 2.0])
            tilt = TiltWatch(ob, tilt_deg=UNPARK_TILT_DEG)
            logger.info(f'Unpark: moving in towards {np.round(center, 2)}')

            started = time.time()
            deadline = started + CRUISE_TIMEOUT_S
            while True:
                if time.time() > deadline:
                    stop_short('took too long to bring the gantry marker back into sight')
                    return

                caught = tilt.check()
                if caught:
                    stop_short(f'{caught}; the gantry is caught on something')
                    return

                if len(ob.fresh_gantry_sightings(SIGHTING_WINDOW_S, after=started)) >= MIN_SIGHTINGS:
                    logger.info(f'Unpark: gantry marker back in sight after '
                                f'{time.time() - started:.0f}s')
                    break

                here = ob.gantry_position()
                to_center = center - here
                distance = float(np.linalg.norm(to_center))
                if distance < CENTER_PROXIMITY_M:
                    stop_short('reached the middle of the room and the gantry marker never came into sight')
                    return

                velocity = to_center / distance * CRUISE_SPEED_MPS
                await ob.move_direction_speed(velocity, None, here, key=self.velocity_key())
                await asyncio.sleep(CRUISE_LOOP_S)

            ob.slow_stop_all_spools()
            if not await ob.settle_visual_estimate(window_s=SIGHTING_WINDOW_S,
                                                   min_sightings=MIN_SIGHTINGS):
                stop_short('the position estimate never settled onto the marker sightings')
                return

            await ob.half_auto_calibration()
            logger.info('Unpark complete')
        except asyncio.CancelledError:
            logger.info('Unpark cancelled')
            raise
        finally:
            # slow_stop_all_spools only zeroes the default source, and a velocity left on
            # this one would be summed back into the next move anything else commands.
            await ob.move_direction_speed(np.zeros(3), 0, key=self.velocity_key())
            ob.slow_stop_all_spools()
            await ob.clear_goal()
