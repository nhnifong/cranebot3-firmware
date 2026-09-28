"""Plates: capture runs for the synthetic visual servoing dataset; see ml/visual_servoing/readme.md."""

import asyncio
import logging
import time
import uuid

from nf_robot.host.arp_gripper_client import CAPTURE_RESOLUTION_SIZE
from nf_robot.host.maneuver import Maneuver, verb

logger = logging.getLogger(__name__)

PLATE_OUTPUT_DIR = 'plates'
# Finger sweep bounds. The server clamps finger angle to -90 (open) .. 90 (closed), and a
# plate is wanted at every aperture the fingers are actually driven to during a grasp.
FINGERPLATE_ANGLE_MIN = -76
FINGERPLATE_ANGLE_MAX = 90
FINGERPLATE_ANGLE_STEP = 2
# Frames per wrist turn. The matte keys each one and takes the median, so it wants enough
# of them that anything which rotated past is outvoted - and they are cheap next to the
# finger moves between turns.
FINGERPLATE_WRIST_STEPS = 18
# (seconds) extra wait after the wrist reports arrival, due to the higher video latency from this format
FINGERPLATE_SETTLE_S = 0.3
# How long to wait for the camera to come back at the capture resolution. rpicam-vid is
# killed and relaunched to change resolution, and the client retries the connection a few
# times before giving up, so this has to cover all of that.
CAPTURE_STREAM_TIMEOUT_S = 30.0
# Consecutive missing frames that mean the stream has gone rather than lagged.
FINGERPLATE_MAX_MISSES = 5
# (metres) rangefinder readings to capture floor and object plates at. Spans the heights
# a gripper actually approaches from, and is what calibrates a plate's apparent scale
# against the range it was taken at, so the compositor can rescale it to any simulated
# height. Trimmed to by measurement, not by commanded altitude.
PLATE_RANGES_M = (0.12, 0.28, 0.44, 0.60, 0.74)
# (degrees/second, degrees) the continuous wrist sweep floor and object plates are
# captured during. Slow enough that the pole does not swing and frames stay sharp.
PLATE_WRIST_SPEED_DPS = 30.0
PLATE_SWEEP_DEGREES = 360.0
# (seconds) how often a wrist speed command is repeated to keep the sweep going.
WRIST_SPEED_REFRESH_S = 0.1
# (seconds) how often the robot's state is sampled beside a recorded video sweep.
TELEMETRY_SAMPLE_S = 0.05
# (seconds) grace for the demux loop to notice a recording has been asked to stop.
RECORDING_CLOSE_S = 1.0
# (degrees) fingers parked out of frame while capturing floor and object plates. -90 is
# fully open; anything the camera can see would be composited into every synthetic frame
# built from these plates.
PLATE_FINGERS_RETRACTED = -90.0


class Plates(Maneuver):
    name = 'plates'
    title = 'Plates'

    @verb('fingerplates', motion=True)
    async def collect_fingerplates(self, finger_angles=None, wrist_steps=FINGERPLATE_WRIST_STEPS,
                                   output_dir=PLATE_OUTPUT_DIR, settle_s=FINGERPLATE_SETTLE_S):
        """Capture the frames a finger matte is extracted from, one wrist turn per finger angle.

        Park the gripper over the green backdrop first: the matte is a chroma key, so
        anything ungreen under the fingers comes out as hardware.

        The wrist turn is what makes that robust. The camera is in the palm and turns with
        the wrist, so the fingers stay on the same pixels while the world rotates behind
        them; keying every frame and taking the median leaves anything that passed
        underneath outvoted.

        Only the raw frames are written. Deciding what is finger is an offline judgement
        with thresholds nobody has tuned, and it should be revisable without asking the
        robot to do this again.
        """
        from nf_robot.ml.visual_servoing.plates import PlateWriter, provenance

        ob = self.ob
        if not ob.gripper_connected():
            logger.error('No gripper connected; cannot collect fingerplates')
            return None
        if finger_angles is None:
            finger_angles = list(range(FINGERPLATE_ANGLE_MIN, FINGERPLATE_ANGLE_MAX + 1,
                                       FINGERPLATE_ANGLE_STEP))

        start_wrist = ob.wrist_angle()
        # A full turn has to fit inside the wrist's 0-1080 range without winding the cable
        # up against its limit, so start low enough that base + 360 still fits.
        base_wrist = float(min(max(start_wrist, 0.0), 1080.0 - 360.0))
        wrist_angles = [base_wrist + 360.0 * i / wrist_steps for i in range(wrist_steps)]

        writer = PlateWriter(output_dir, 'fingerplates',
                             notes='wrist turn per finger angle; matte offline by chroma key')
        logger.info(f'Fingerplates: {len(finger_angles)} finger angles x {wrist_steps} wrist '
                    f'steps = {len(finger_angles) * wrist_steps} frames, wrist {base_wrist:.0f}'
                    f'-{base_wrist + 360:.0f}, writing to {output_dir}')

        expect = CAPTURE_RESOLUTION_SIZE
        await ob.use_gripper_capture_stream()
        try:
            # Hold out for a frame at the capture resolution, not merely a recent one: the
            # old stream keeps delivering for seconds after the new settings are sent, and
            # accepting those would run the whole sweep at 684x384 while reporting success.
            _, probe = await ob.gripper_capture(timeout=CAPTURE_STREAM_TIMEOUT_S, expect_size=expect)
            if probe is None:
                logger.error(
                    f'Fingerplates: no {expect[0]}x{expect[1]} frames within '
                    f'{CAPTURE_STREAM_TIMEOUT_S}s of switching to the capture stream. '
                    f'Check the gripper log for rpicam-vid "ERROR: ***" lines - it may not '
                    f'be able to start at this resolution.')
                return None
            logger.info(f'Fingerplates: capture stream up at {probe.shape[1]}x{probe.shape[0]}')

            missed = 0
            for wrist_angle in wrist_angles:
                actual_wrist = await ob.settle_wrist(wrist_angle)
                # Telemetry reports the motor arrived before the video shows it: the
                # capture stream's settings put the frames further behind than that.
                await asyncio.sleep(settle_s)
                # Always the same direction, never serpentine. There is enough slop in
                # the finger gearing that the same commanded angle approached from above
                # and from below puts the hardware in visibly different places, which
                # comes out of the matte as doubled fingers.
                for finger_angle in finger_angles:
                    actual_finger = await ob.settle_fingers(finger_angle)
                    timestamp, frame = await ob.gripper_capture(expect_size=expect)
                    if frame is None:
                        missed += 1
                        logger.warning(f'Fingerplates: no frame at finger {finger_angle} '
                                       f'wrist {wrist_angle:.0f} ({missed} in a row)')
                        if missed >= FINGERPLATE_MAX_MISSES:
                            # The stream is gone, not merely late. Continuing means half an
                            # hour of moving the wrist around for nothing.
                            logger.error(f'Fingerplates: {missed} consecutive frames missing; '
                                         f'abandoning the run with {len(writer)} captured')
                            return None
                        continue
                    missed = 0
                    writer.add(
                        frame, captured_at=timestamp,
                        finger_angle=actual_finger, wrist_angle=actual_wrist,
                        laser_rangefinder=ob.laser_range(),
                        finger_pressure=ob.finger_pad_voltage(),
                        commanded_finger_angle=finger_angle, commanded_wrist_angle=wrist_angle,
                    )
                logger.info(f'Fingerplates: wrist {wrist_angle:.0f} done ({len(writer)} frames)')
        finally:
            # the capture stream stays selected for the rest of the session; switching
            # back costs a stream restart and the next plate command would undo it
            await ob.settle_wrist(start_wrist)

        return writer.close(
            finger_angles=list(finger_angles), wrist_steps=wrist_steps,
            base_wrist_angle=base_wrist, **provenance(ob.robot_id()),
        )

    async def _sweep_wrist_sampling(self, kind, writer, degrees, speed_dps, extra,
                                    timeout_margin=1.5):
        """Turn the wrist steadily through `degrees`, sampling telemetry as it goes.

        The frames themselves are being recorded as video by the client; what this adds
        is the state track they get matched against. The speed command is repeated
        because the gripper zeroes it after ACTION_TIMEOUT, which is also what stops the
        wrist if this is cancelled.
        """
        ob = self.ob
        start = ob.wrist_angle()
        direction = 1.0 if degrees >= 0 else -1.0
        deadline = time.time() + abs(degrees) / speed_dps + timeout_margin
        next_command = 0.0
        samples = 0

        try:
            while time.time() < deadline:
                now = time.time()
                if now >= next_command:
                    await ob.set_wrist_speed(direction * speed_dps)
                    next_command = now + WRIST_SPEED_REFRESH_S

                writer.add_telemetry(
                    captured_at=time.time(),
                    wrist_angle=ob.wrist_angle(),
                    finger_angle=ob.finger_angle(),
                    finger_pressure=ob.finger_pad_voltage(),
                    laser_rangefinder=ob.laser_range(),
                    **extra,
                )
                samples += 1

                travelled = (ob.wrist_angle() - start) * direction
                if travelled >= abs(degrees):
                    break
                await asyncio.sleep(TELEMETRY_SAMPLE_S)
        finally:
            await ob.set_wrist_speed(0.0)

        actual = ob.wrist_angle()
        logger.info(f'{kind}: swept wrist {start:.0f} -> {actual:.0f} '
                    f'({samples} telemetry samples at {speed_dps:.0f} deg/s)')
        return samples

    async def _height_wrist_sweep(self, kind, ranges, output_dir, settle_s,
                                  notes='', run_attrs=None, frame_attrs=None,
                                  speed_dps=PLATE_WRIST_SPEED_DPS,
                                  degrees=PLATE_SWEEP_DEGREES):
        """Frames at each of several heights, sweeping the wrist through a circle at each.

        The shape floorplates and objectplates share. Heights are reached by trimming to
        a measured rangefinder reading, so what each plate records is how far away its
        subject actually was. The fingers are parked out of frame first, since hardware
        in the corner of a plate would be composited into every frame built from it.
        """
        from nf_robot.ml.visual_servoing.plates import VideoRunWriter, provenance

        ob = self.ob
        if not ob.gripper_connected():
            logger.error(f'No gripper connected; cannot collect {kind}')
            return None

        start_wrist = ob.wrist_angle()
        # start low enough in the wrist's 0-1080 range that a full sweep fits
        base_wrist = float(min(max(start_wrist, 0.0), 1080.0 - abs(degrees)))

        writer = VideoRunWriter(output_dir, kind, notes=notes)
        seconds = len(ranges) * abs(degrees) / speed_dps
        logger.info(f'{kind}: {len(ranges)} heights, {degrees:.0f} deg at {speed_dps:.0f} '
                    f'deg/s each, about {seconds / 60:.0f} min of sweeping, '
                    f'writing to {output_dir}')

        expect = CAPTURE_RESOLUTION_SIZE
        await ob.use_gripper_capture_stream()
        packets, stream_start_ts = 0, None
        try:
            _, probe = await ob.gripper_capture(timeout=CAPTURE_STREAM_TIMEOUT_S, expect_size=expect)
            if probe is None:
                logger.error(f'{kind}: no {expect[0]}x{expect[1]} frames within '
                             f'{CAPTURE_STREAM_TIMEOUT_S}s of switching to the capture '
                             f'stream. Check the gripper log for rpicam-vid errors.')
                return None
            logger.info(f'{kind}: capture stream up at {probe.shape[1]}x{probe.shape[0]}')

            await ob.settle_fingers(PLATE_FINGERS_RETRACTED)
            await ob.settle_wrist(base_wrist)

            ob.start_gripper_recording(writer.video_path)
            heading = 1.0
            for target_range in ranges:
                reached = await ob.trim_altitude_to_range(target_range)
                if reached is None:
                    logger.warning(f'{kind}: no rangefinder reading at target '
                                   f'{target_range:.2f}m; skipping this height')
                    continue
                await asyncio.sleep(settle_s)
                await self._sweep_wrist_sampling(
                    kind, writer, heading * degrees, speed_dps,
                    extra={'target_range_m': target_range,
                           'start_wrist_angle': start_wrist,
                           **(frame_attrs or {})})
                heading = -heading
                logger.info(f'{kind}: range {target_range:.2f}m done '
                            f'({ob.gripper_recorded_packets()} packets recorded)')
        finally:
            if ob.gripper_connected():
                packets, stream_start_ts = ob.stop_gripper_recording()
            # the demux loop closes the file when it next sees a packet
            await asyncio.sleep(RECORDING_CLOSE_S)
            # the capture stream stays selected for the rest of the session; see
            # collect_fingerplates
            await ob.settle_wrist(start_wrist)

        return writer.close(stream_start_ts or 0.0, packets=packets,
                            target_ranges=list(ranges), sweep_degrees=degrees,
                            sweep_speed_dps=speed_dps, start_wrist_angle=start_wrist,
                            **provenance(ob.robot_id()), **(run_attrs or {}))

    @verb('floorplates', motion=True)
    async def collect_floorplates(self, ranges=None, output_dir=PLATE_OUTPUT_DIR,
                                  settle_s=FINGERPLATE_SETTLE_S):
        """Capture bare floor at a range of heights, for synthetic backgrounds.

        The operator flies the gripper somewhere clean and clear and only then triggers
        this; it moves nothing but height and wrist. An autonomous room sweep would come
        back with a library of beds, furniture and feet, none of which is a floor plate.
        """
        return await self._height_wrist_sweep(
            'floorplates', ranges or PLATE_RANGES_M, output_dir, settle_s,
            notes='bare floor at several heights; fingers retracted')

    @verb('objectplates', motion=True)
    async def collect_objectplates(self, ranges=None,
                                   output_dir=PLATE_OUTPUT_DIR,
                                   settle_s=FINGERPLATE_SETTLE_S):
        """Capture one object on the green board, for compositing onto floor plates.

        Two things the operator sets before triggering this, and both are labels rather
        than settings: the object's intended grasp point goes under the camera, which
        makes the grasp point the principal point by construction, and the wrist is
        turned to the ideal grasping angle, which makes the grasp axis zero at the start
        and a known offset at every later frame. Neither needs marks on the board.
        """
        label = f'object-{time.strftime("%Y%m%d-%H%M%S")}-{uuid.uuid4().hex[:4]}'
        start_wrist = self.ob.wrist_angle()
        logger.info(f'objectplates: labelling this object {label}')
        result = await self._height_wrist_sweep(
            'objectplates', ranges or PLATE_RANGES_M, output_dir, settle_s,
            notes=f'object on green board: {label}',
            run_attrs={'label': label, 'grasp_axis_wrist_angle': start_wrist},
            frame_attrs={'label': label})
        if result is not None:
            self.notify('Objectplate capture complete.')
        return result
