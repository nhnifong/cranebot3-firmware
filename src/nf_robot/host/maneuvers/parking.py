"""Parking: recording where the hook is, settling the gantry onto it, and lifting it off."""

import asyncio
import logging
import time

import cv2
import numpy as np

import nf_robot.common.definitions as model_constants
import nf_robot.generated.nf.config as nf_config
from nf_robot.common.cv_common import get_inward_wall_normal, get_wall_escape_direction
from nf_robot.common.image_motion import image_shift
from nf_robot.common.util import fromnp, tonp
from nf_robot.generated.nf import control, telemetry
from nf_robot.host.maneuver import Maneuver, TiltWatch, command, startup_step

logger = logging.getLogger(__name__)

# Gripper camera views of the parking hook; park_data.reference_image names the one this
# robot parks against.
# Share of the frame, from the top, that park compares against that reference. The rest is
# fingers and whatever is a hand's width under the camera: the part that slides furthest for
# a given move and agrees with nothing else in the view.
PARK_COMPARE_CROP = 0.62
PARK_COMPARE_MIN_MATCHES = 20   # fewer than this and there is nothing to fit a move to
# (metres) how far out along the escape direction the mouth of the fork is: as far in as
# the gripper camera is any use, since from there the gantry goes straight down the track
# into the fork with no room to correct. record_park photographs from here and park steers
# to here.
MOUTH_OFFSET_M = 0.075
# (degrees) fingers held here through both record_park and park. Fully open is fully
# retracted, which takes them out of the camera's view altogether: the reference image and
# the live view then agree about the whole frame instead of agreeing everywhere the fingers
# are not. The server clamps to -90.
PARK_FINGER_ANGLE = -90
PARK_REFERENCE_DIR = 'park_reference'
# (degrees) how far the track in and out of the parking hook leans along the wall, off the
# wall's inward normal and towards the anchor end of it. Unpark leaves along this line and
# park comes back down it, so one constant sets both.
ESCAPE_TILT_DEG = 45.0


class Parking(Maneuver):
    name = 'parking'
    title = 'Parking'
    config_field = 'park_data'

    def __init__(self, ob):
        super().__init__(ob)
        # (name, mtime) -> grayscale image, so a servo loop reading the reference several
        # times a second is not decoding a JPEG every pass.
        self._reference_cache = {}

    def send_setup_telemetry(self):
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
        logger.info(f'Robot is {"parked" if parked else "not parked"}')

    def save_reference_image(self, frame):
        """Write the gripper camera frame out as the parking reference, and return its name.

        A config that already names one keeps that name and has the file overwritten, so
        re-recording replaces the reference instead of leaving orphans behind.
        """
        directory = self.output_dir(PARK_REFERENCE_DIR)
        name = (self.data.reference_image
                or f'park_reference_{time.strftime("%Y%m%d_%H%M%S")}.jpg')
        # frames arrive from the decoder as RGB; cv2 writes BGR
        cv2.imwrite(str(directory / name), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        logger.info(f'Saved park reference image {directory / name}')
        return name

    def compare_to_reference(self, frame, name=None):
        """Where this gripper frame sits relative to a parking reference image.

        Returns (dx, dy, confidence, inliers) as image_motion.image_shift does: how far the
        scene has slid in pixels between the two, the share of feature matches that agree
        on that (0 to 1), and how many that was.
        """
        name = name or self.data.reference_image
        if not name:
            return None
        reference = self._reference_gray(name)
        if reference is None:
            return None
        return image_shift(reference, cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY),
                           crop=PARK_COMPARE_CROP, min_matches=PARK_COMPARE_MIN_MATCHES)

    def _reference_gray(self, name):
        """A parking reference image in grayscale, or None if it cannot be read."""
        path = self.output_dir(PARK_REFERENCE_DIR) / name
        try:
            mtime = path.stat().st_mtime
        except OSError:
            logger.warning(f'Parking reference image {path} is missing')
            return None
        if (name, mtime) not in self._reference_cache:
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                logger.warning(f'Parking reference image {path} could not be read')
                return None
            self._reference_cache = {(name, mtime): image}
        return self._reference_cache[(name, mtime)]

    def offset_to_room(self, dx, dy):
        """The room-frame XY move that would undo a (dx, dy) pixel slide of the scene.

        A camera that translates by t sees the scene slide by -f*t/depth, so the move that
        puts the scene back is (dx, dy) * depth / f in the camera's own frame. Depth is what
        the rangefinder reads, which is the distance straight down rather than along an
        optical axis that also looks forward, so this is a direction with roughly the right
        size rather than a measurement - which is fine for something that measures again
        after every step, and is why the gain on it is well under one.
        """
        distance = self.ob.laser_range()
        if distance is None:
            return None
        intrinsics = self.ob.gripper_camera_intrinsics()
        # camera optical frame: x right, y down, z along the axis
        in_camera = np.array([dx * distance / intrinsics[0][0],
                              dy * distance / intrinsics[1][1],
                              0.0])
        return self.ob.gripper_camera_to_room(in_camera)[:2]

    async def measure_reference_offset(self, name=None):
        """How far the gripper's view is from a parking reference, as (offset_px, move, conf).

        None when there is nothing to compare: no gripper, no reference recorded, or no
        frame. A low confidence is still returned rather than swallowed - the caller has to
        decide whether "cannot tell" is a reason to stop, and that differs by caller.
        """
        if not self.ob.gripper_connected():
            return None
        name = name or self.data.reference_image
        if not name:
            return None
        frame = await self.ob.gripper_frame(timeout=3.0)
        if frame is None:
            return None
        result = self.compare_to_reference(frame, name)
        if result is None:
            return None
        dx, dy, confidence, inliers = result
        return float(np.hypot(dx, dy)), self.offset_to_room(dx, dy), confidence

    @command(control.Command.RECORD_PARK, motion=True)
    async def record_park(self):
        """Record where the parking hook is, and what the camera sees from over it.

        Ends back down on the hook
        This is a motion task.
        """
        LIFT_M = 0.10               # up off the hook, the same move unpark starts with
        LIFT_SPEED_MPS = 0.04
        SETTLE_S = 2.0              # let the pole stop swinging before the photograph
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
            # taken now, resting on the hook, which is the one moment this reading means
            # "the height the hook holds the gantry at"
            parked_range = ob.laser_range()
            if parked_range is None:
                logger.warning('Record park: no rangefinder reading on the hook; park will '
                               'have nothing to confirm its height against')

            # All of this happens within 10cm of the hook, so swing cancellation stays off
            # throughout and is not restored afterwards, the same as park.
            ob.set_swing_cancellation(False)

            # Fingers fully open, which retracts them clear of the camera, and is where park
            # holds them too: the reference image has to be framed the way park will see it.
            # Waited on rather than fired off, since the photograph is the whole point here.
            await ob.settle_fingers(PARK_FINGER_ANGLE)

            # 1. up off the hook.
            logger.info(f'Record park: lifting {LIFT_M * 100:.0f}cm from '
                        f'{np.round(hook_pos, 3)}')
            await ob.nudge_gantry(np.array([0.0, 0.0, LIFT_M]), speed=LIFT_SPEED_MPS)

            # 2. nose away from the wall. The camera looks out from under the nose, so this
            # is what frames the reference image, and park turns back to the angle this
            # leaves the wrist at.
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

            # 3. out to the mouth of the fork, which is where the reference image has to be
            # taken from: it is the last place park can still correct itself, so it is the
            # place park has to be able to recognise.
            hover_pos = ob.gantry_position()
            logger.info(f'Record park: stepping {MOUTH_OFFSET_M * 100:.1f}cm out to the '
                        f'mouth of the fork along {np.round(escape, 3)}')
            await ob.nudge_gantry(np.array([escape[0], escape[1], 0.0]) * MOUTH_OFFSET_M,
                                  speed=LIFT_SPEED_MPS)
            await asyncio.sleep(SETTLE_S)

            # 4. the photograph, from a frame captured after everything stopped moving, and
            # the range from here, which is what park sets its own altitude by.
            mouth_range = ob.laser_range()
            if mouth_range is None:
                logger.warning('Record park: no rangefinder reading at the mouth; park will '
                               'have to take its altitude from the position estimate')
            frame = await ob.gripper_frame(timeout=CAPTURE_TIMEOUT_S)
            if frame is None:
                logger.warning('No frame arrived from the gripper camera; '
                               'parking location not saved')
            else:
                # the hovering position, not this one: the mouth is derived from it and the
                # escape direction, and the descent is measured from it
                self.data.pos = fromnp(hover_pos)
                self.data.escape_direction = fromnp(np.array([escape[0], escape[1], 0.0]))
                self.data.wrist_angle = wrist_angle
                self.data.parked_range = parked_range or 0.0
                self.data.mouth_range = mouth_range or 0.0
                logger.info(f'Record park: range {parked_range} on the hook, '
                            f'{mouth_range} at the mouth')
                self.data.reference_image = self.save_reference_image(frame)
                self.save_data()
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
                return f'{caught} entering the fork'
            here = self.ob.gantry_position()
            travelled = float(np.dot(here[:2] - start[:2], -track))
            if travelled >= distance:
                logger.info(f'Park: {travelled * 100:.1f}cm in, steepest lean '
                            f'{tilt.worst_tilt:.1f} degrees')
                return None
            await self.ob.move_direction_speed(np.array([-track[0], -track[1], 0.0]) * SPEED_MPS,
                                               None, here, key=self.velocity_key())
            await asyncio.sleep(LOOP_S)

    async def _approach_hook_on_reference(self, aim, lateral, tilt):
        """Close the last of the approach by steering the gripper's view onto the reference.

        Each pass measures how far the view has slid from the image record_park took over
        the hook and moves against it, so the thing being closed is the picture rather than
        a position estimate that only has to be a couple of centimetres out to miss a slot.
        Height stays on the estimate: an image says nothing about it that a change of scene
        could not equally explain.

        A reading nothing agrees on moves nothing - a low confidence is what a view half
        full of somebody walking past looks like, and the next frame is a fifth of a second
        away. Excursion is bounded against the recorded position for the same reason: a
        confident wrong answer should not be able to fly the gantry into a wall.

        lateral shifts what counts as arrived: the image says where the recorded spot is,
        and a retry searching to one side of it wants to stop that far short of matching.

        Meant for the last few centimetres, after the track approach has done the coarse
        work. Thirty centimetres out the two views barely overlap and the fit is poor - one
        run measured the right distance in a direction 47 degrees wrong, on a tenth of its
        matches agreeing - while over the hook the same fit agrees two thirds of the way and
        lands within a couple of centimetres. Readings are smoothed for that last part: a
        couple of centimetres of noise on consecutive frames would otherwise be chased.

        None once it arrives, or why it stopped.
        """
        LOOP_S = 0.2                  # the camera is ~0.2s behind, so a faster loop only
                                      # steers on staler pictures
        TIMEOUT_S = 20.0
        ARRIVED_M = 0.005             # how close the view has to put us, in both axes
        ARRIVED_PASSES = 3            # ...held for this many readings, since one is noise
        # A servo that is closing gets closer. One that has stopped getting closer has found
        # its limit - the measurement noise, a bias, a cable creeping - and waiting out the
        # timeout from there just hovers, so a stall ends this stage. It never fails the
        # attempt: this loop is steering on a smoothed error that a bad patch of readings can
        # leave sitting well off, while the check at the mouth measures again from scratch
        # and is the thing that decides whether to go in.
        STALL_S = 6.0                 # no improvement for this long means it is done moving
        IMPROVEMENT_M = 0.001         # ...and this much closer is what counts as improving
        GOOD_ENOUGH_M = 0.055         # a stall further out than this is worth remarking on
        MIN_CONFIDENCE = 0.5          # share of matches that must agree to be worth moving on
        BLIND_LIMIT_S = 6.0          # give up if nothing readable arrives for this long
        GAIN = 1.0                    # 1/s on the room-frame error. Not lower: at 0.4 a
                                      # half-centimetre error asks for 2mm/s, which
                                      # move_direction_speed rounds to a dead stop
        MIN_SPEED_MPS = 0.006         # ...and this is the speed it stops rounding away
        MAX_SPEED_MPS = 0.05
        MAX_EXCURSION_M = 0.45        # never steer further than this from the recorded spot
        CLIMB_GAIN = 0.3              # 1/s holding the recorded altitude, on the estimate
        SMOOTHING = 0.5               # of the previous error estimate kept each pass

        logger.info('Park: closing on the reference image')
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
                return 'took too long to close on the reference image'

            # the newest frame rather than a fresh one: it is a fifth of a second behind
            # whatever the gantry is doing, and waiting for a newer one only makes it two
            move = error = None
            frame = await self.ob.gripper_frame(after=0, timeout=1.0)
            result = self.compare_to_reference(frame) if frame is not None else None
            if result is not None:
                dx, dy, confidence, inliers = result
                logger.debug(f'Park: offset ({dx:+.1f}, {dy:+.1f})px, '
                             f'{inliers} matches agreed ({confidence:.2f})')
                if confidence >= MIN_CONFIDENCE:
                    move = self.offset_to_room(dx, dy)
                    if move is not None:
                        move = move + lateral
                        smoothed = (move if smoothed is None
                                    else smoothed * SMOOTHING + move * (1 - SMOOTHING))
                        move = smoothed
                        error = float(np.linalg.norm(move))
            if move is None:
                if time.time() - last_reading > BLIND_LIMIT_S:
                    return ('the gripper view has not matched the reference image for '
                            f'{BLIND_LIMIT_S:.0f}s')
                await self.ob.move_direction_speed(np.zeros(3), 0, key=self.velocity_key())
                await asyncio.sleep(LOOP_S)
                continue
            last_reading = time.time()

            if error < best - IMPROVEMENT_M:
                best, best_at = error, time.time()
            elif time.time() - best_at > STALL_S:
                where = (f'{error * 100:.1f}cm from this attempt\'s aim point; steepest lean '
                         f'on the way in was {tilt.worst_tilt:.1f} degrees')
                if error < GOOD_ENOUGH_M:
                    logger.info(f'Park: the view stopped closing at {where}')
                else:
                    logger.warning(f'Park: the view stopped closing well out, at {where} - '
                                   f'going on to the check at the mouth anyway')
                return None

            if error < ARRIVED_M:
                close_passes += 1
                if close_passes >= ARRIVED_PASSES:
                    # move carries the lateral offset, so taking it back out gives the
                    # distance from the recorded mouth itself
                    from_mouth = float(np.linalg.norm(move - lateral))
                    logger.info(f"Park: {error * 100:.1f}cm from this attempt's aim "
                                f"point and {from_mouth * 100:.1f}cm from the recorded "
                                f"mouth; steepest lean on the way in was "
                                f"{tilt.worst_tilt:.1f} degrees")
                    return None
            else:
                close_passes = 0

            here = self.ob.gantry_position()
            excursion = here[:2] + move - aim[:2]
            if float(np.linalg.norm(excursion)) > MAX_EXCURSION_M:
                return (f'the reference image is steering {np.linalg.norm(excursion):.2f}m '
                        f'away from the recorded parking location')

            speed = min(MAX_SPEED_MPS, max(MIN_SPEED_MPS, error * GAIN))
            velocity = np.array([move[0] / error * speed, move[1] / error * speed,
                                 (aim[2] - here[2]) * CLIMB_GAIN])
            await self.ob.move_direction_speed(velocity, None, here, key=self.velocity_key())
            await asyncio.sleep(LOOP_S)

    @command(control.Command.PARK, motion=True)
    async def park(self):
        """Fly home and settle the gantry onto the parking hook.

        The hook is a diagonal slot built around the flat marker.
        It is entered along the same line unpark leaves by.
        First we fly to a point HANDOVER_M short of the mouth of the fork.
        The gripper camera steers the rest of the way onto the hook using a different from a reference image.
        After getting as close as this can take us, it goes in along the track blind and lowers
        Looking for the signature of a successful park, using tension and laser rangefinder.

        Swing cancellation is worth having for the flight out and is switched off for good once the
        camera takes over. it's very important that it isn't turned on in the vicinity of the hook.
        That would almost certainly damage something.

        This is a motion task.
        """
        FINGER_ANGLE_PARKED = 70      # a neutral-looking hand position.
        HANDOVER_M = 0.10             # how far out beyond the mouth the flight ends and
                                      # the camera takes over.
        DESCENT_M = 0.30              # how far down to look for the hook
        DESCENT_SPEED_MPS = 0.03
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
        PARKED_FRACTION = 0.3         # every line under this share of its free-hanging
                                      # tension is the hook having taken the weight
        PARKED_CONFIRM_S = 0.4
        PARKED_RANGE_TOL_M = 0.05     # how far off the recorded parked range still counts
        RANGE_TRIM_TOL_M = 0.01
        RANGE_TRIM_STEPS = 5
        RANGE_TRIM_TRAVEL_M = 0.30
        SWING_SETTLE_S = 3.0          # after the approach, before steering by the camera
        # Lined up at the mouth is the last thing that can be checked - past it the gantry
        # is inside the fork and the camera is looking at something it has no reference for.
        # In metres: pixels mean different distances at different heights.
        ALIGNED_M = 0.10             # lined up at the mouth, before going in
        VERIFY_CONFIDENCE = 0.4       # under this the view has not been recognised at all
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
            if not ob.gripper_connected() or not park_data.reference_image:
                stop_short('parking needs a connected gripper and a recorded reference image')
                return

            # fingers fully open to clear view
            await ob.set_finger_angle(PARK_FINGER_ANGLE)

            # across the track, so an attempt offset can be written in the track's frame
            sideways = np.array([-track[1], track[0]])

            def in_room(offset):
                """An (across, along, up) attempt offset as a room-frame vector."""
                across, along, up = offset
                return np.array([sideways[0] * across + track[0] * along,
                                 sideways[1] * across + track[1] * along,
                                 up])

            # 1. Fly to standoff position
            def handover_point(offset):
                return (parkpos + offset + np.array([track[0], track[1], 0.0])
                        * (MOUTH_OFFSET_M + HANDOVER_M))

            first = handover_point(in_room(ATTEMPT_OFFSETS_M[0]))
            logger.info(f'Park: flying to {np.round(first, 3)}, where the camera takes over, '
                        f'{(MOUTH_OFFSET_M + HANDOVER_M) * 100:.1f}cm out from '
                        f'{np.round(parkpos, 3)} along {np.round(track, 3)}')
            async with ob.prefer_swing_cancellation():
                await ob.seek_goal(first)
                await asyncio.sleep(SETTLE_S) # Allow swing cancellation to damp

            # 2. Turn swing cancellation off for good, and the wrist back
            # to the angle the reference image was taken at.
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

            async def attempt(offset):
                """One go at getting onto the hook, aiming offset metres from the recorded
                spot. None if it worked, or why it did not."""
                # the mouth of the fork, which is what the reference image was taken from
                lateral = offset[:2]   # the servo only steers horizontally; z is held
                mouth = parkpos + offset + np.array([track[0], track[1], 0.0]) * MOUTH_OFFSET_M
                tilt = TiltWatch(ob, tilt_deg=TILT_DEG, confirm_s=TILT_CONFIRM_S)
                logger.info(f'Park: closing on the mouth of the fork at {np.round(mouth, 3)}')

                # 2. Close the distance using the reference image
                ob.slow_stop_all_spools()
                await asyncio.sleep(SWING_SETTLE_S)
                failed = await self._approach_hook_on_reference(mouth, lateral, tilt)
                if failed:
                    return failed

                ob.slow_stop_all_spools()
                await asyncio.sleep(SETTLE_S)

                # Report visual alignment numbers before moving in.
                aligned = await self.measure_reference_offset()
                if aligned is None:
                    logger.warning('Park: nothing to check the alignment against before going in')
                elif aligned[2] < VERIFY_CONFIDENCE:
                    logger.warning(f'Park: the view at the mouth was not recognised '
                                   f'(confidence {aligned[2]:.2f}); going in without checking it')
                elif aligned[1] is None:
                    logger.warning('Park: no rangefinder reading, so the view offset cannot be '
                                   'turned into a distance; going in without checking it')
                elif float(np.linalg.norm(aligned[1] + lateral)) > ALIGNED_M:
                    # against where this attempt meant to end up, which is the recorded
                    # mouth shifted by lateral; measuring against the recorded mouth itself
                    # would fail every attempt that is deliberately searching to one side
                    return (f"the view puts this attempt's aim point "
                            f"{np.linalg.norm(aligned[1] + lateral) * 100:.1f}cm away "
                            f"({np.linalg.norm(aligned[1]) * 100:.1f}cm from the recorded "
                            f"mouth) - not lined up with it")

                # 4. Blindly move in along a track. we may or may not be aligned.
                # use various readings to look for the signature of a successful park
                failed = await self._enter_hook(track, tilt, MOUTH_OFFSET_M)
                if failed:
                    return failed
                ob.slow_stop_all_spools()
                await asyncio.sleep(SETTLE_S)

                # 5. lower until the hook takes the weight.
                free = await ob.measure_free_tension()
                parked_limit = free * PARKED_FRACTION
                tilt = TiltWatch(ob, tilt_deg=TILT_DEG, confirm_s=TILT_CONFIRM_S)
                logger.info(f'Park: lowering up to {DESCENT_M * 100:.0f}cm, parked when every '
                            f'line falls under {np.round(parked_limit, 2)}N')
                start_z = float(ob.gantry_position()[2])
                tension = ob.line_tensions()
                slack_since = None
                landed = False
                while float(ob.gantry_position()[2]) > start_z - DESCENT_M:
                    caught = tilt.check()
                    if caught:
                        return f'{caught} on the way down'
                    tension = tension * 0.7 + ob.line_tensions() * 0.3
                    if np.all(tension < parked_limit):
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
                    await ob.move_direction_speed(np.array([0.0, 0.0, -DESCENT_SPEED_MPS]),
                                                  None, ob.gantry_position(),
                                                  key=self.velocity_key(), downward_bias=0)
                    await asyncio.sleep(DESCENT_LOOP_S)
                ob.slow_stop_all_spools()

                if not landed:
                    return (f'lowered the full {DESCENT_M * 100:.0f}cm without every line going '
                            f'slack ({np.round(tension, 2)}N)')

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
