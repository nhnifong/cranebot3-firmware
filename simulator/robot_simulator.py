#!/usr/bin/env python3
"""
Local robot component simulator — run without physical hardware.

requires all optional dependencies of nf_robot
pip install nf_robot[host,dev,pi]

Spins up two ARP anchor servers and one ARP gripper server on localhost with all
I2C / CAN-bus hardware stubbed out.  Each server also gets an ffmpeg test-pattern
video stream (h264 mpegts over TCP) matching the resolution, framerate, and codec
that rpicam-vid would normally produce.

Port assignments
  Anchor 0 (power):  websocket 9865, video 9888
  Anchor 1:          websocket 9866, video 9889
  Gripper:           websocket 9864, video 9890

To connect observer.py to the simulator you can either:
  - let it discover via mDNS on localhost (the simulator advertises on 127.0.0.1), or
  - manually set the service addresses in the observer config.
"""

import argparse
import asyncio
import logging
import signal
from unittest.mock import Mock, patch

from zeroconf import IPVersion
from zeroconf.asyncio import AsyncZeroconf

from nf_robot.robot.anchor_arp_server import AnchorArpServer
from nf_robot.robot.gripper_arp_server import GripperArpServer


# ── Stub Damiao CAN-bus motor objects ──────────────────────────────────────────

class _StubDaMiaoMotor:
    def enable(self): pass
    def disable(self): pass
    def ensure_control_mode(self, mode): pass
    def set_acceleration(self, a): pass
    def set_deceleration(self, d): pass
    def send_cmd_vel(self, target_velocity=0.0): pass
    def get_states(self):
        # slight negative torque keeps anti-tangle logic quiet
        return {'pos': 0.0, 'vel': 0.0, 'torq': -0.01}


class _StubDaMiaoController:
    def __init__(self, **kwargs): pass
    def add_motor(self, **kwargs):
        return _StubDaMiaoMotor()
    def shutdown(self): pass


# ── ffmpeg test-pattern stream ─────────────────────────────────────────────────

async def _video_stream_loop(port: int, width: int, height: int, fps: int, bitrate: str):
    """
    Keep an ffmpeg h264/mpegts test-pattern stream listening on TCP port.
    ffmpeg exits after each client disconnects; this loop restarts it.
    """
    cmd = [
        'ffmpeg', '-re',
        '-f', 'lavfi', '-i', f'testsrc=size={width}x{height}:rate={fps}',
        '-vcodec', 'libx264',
        '-preset', 'ultrafast',
        '-tune', 'zerolatency',
        '-b:v', bitrate,
        '-f', 'mpegts',
        f'tcp://0.0.0.0:{port}?listen=1',
        '-y', '-loglevel', 'warning',
    ]
    proc = None
    try:
        while True:
            logging.info('ffmpeg test stream starting on port %d (%dx%d @ %d fps)', port, width, height, fps)
            proc = await asyncio.create_subprocess_exec(*cmd)
            await proc.wait()
            proc = None
            logging.info('ffmpeg exited on port %d, restarting in 1 s', port)
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        if proc is not None:
            proc.kill()
            await proc.wait()
        raise


# ── main ───────────────────────────────────────────────────────────────────────

ANCHOR_WS_PORTS   = [9865, 9866]
ANCHOR_VID_PORTS  = [9888, 9889]
GRIPPER_WS_PORT   = 9864
GRIPPER_VID_PORT  = 9890


async def main(no_video=False, mujoco_model=None, realtime=1.0,
               room_side=None, cam_tilt=None):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)-8s %(message)s',
    )

    # ── patches ────────────────────────────────────────────────────────────────
    patchers = []
    world = None

    if mujoco_model is not None:
        # Physics-backed hardware: the same interfaces, driven by the MuJoCo model, so
        # the servers below run unmodified against a robot that actually swings and goes
        # slack. See experiments/mujoco_bridge.py.
        from nf_robot.robot import anchor_arp_server as _aas  # noqa: F401  (patch target)
        import mujoco_bridge
        kw = {}
        if room_side is not None:
            kw['room_side'] = room_side
        if cam_tilt is not None:
            kw['cam_tilt_deg'] = cam_tilt
        world = mujoco_bridge.MujocoWorld(mujoco_model, realtime=realtime, **kw)
        patchers.extend(mujoco_bridge.make_patchers(world))
        for p in patchers:
            p.start()
        world.start()
        return await _run_servers(no_video, patchers, world)

    # Damiao CAN-bus controller (used by AnchorArpServer)
    patchers.append(patch('nf_robot.robot.anchor_arp_server.DaMiaoController', _StubDaMiaoController))

    # Gripper I2C hardware (mirrors observer_integration_test.py)
    mock_servo_class = Mock()
    mock_servo_class.return_value.get_feedback.return_value = {
        'position': 2000, 'speed': 0, 'load': 0, 'voltage': 7.4, 'temp': 20, 'moving': 0,
    }
    patchers.append(patch('nf_robot.robot.gripper_arp_server.SimpleSTS3215', mock_servo_class))
    patchers.append(patch('nf_robot.robot.gripper_arp_server.board.SCL', None, create=True))
    patchers.append(patch('nf_robot.robot.gripper_arp_server.board.SDA', None, create=True))
    patchers.append(patch('nf_robot.robot.gripper_arp_server.busio.I2C', lambda a, b: None))

    mock_imu_class = Mock()
    mock_imu_class.return_value.gyro = (0, 0, 0)
    patchers.append(patch('nf_robot.robot.gripper_arp_server.MPU6050', mock_imu_class))

    patchers.append(patch('nf_robot.robot.gripper_arp_server.ADS1015', Mock()))

    mock_analog_class = Mock()
    mock_analog_class.return_value.voltage = 2.5
    patchers.append(patch('nf_robot.robot.gripper_arp_server.AnalogIn', mock_analog_class))

    mock_range_class = Mock()
    mock_range_class.return_value.model_info = (1, 2, 3)
    mock_range_class.return_value.data_ready = True
    mock_range_class.return_value.distance = 30.0
    patchers.append(patch('nf_robot.robot.gripper_arp_server.VL53L1X', mock_range_class))

    for p in patchers:
        p.start()

    return await _run_servers(no_video, patchers, None)


async def _run_servers(no_video, patchers, world):
    """Bring up the anchor and gripper servers. `world` is a mujoco_bridge.MujocoWorld
    when the simulator is physics-backed, otherwise None."""

    # ── zeroconf on localhost only ──────────────────────────────────────────────
    zc = AsyncZeroconf(ip_version=IPVersion.All, interfaces=['127.0.0.1'])

    # ── servers ────────────────────────────────────────────────────────────────
    anchors = []
    for i, (ws_port, vid_port) in enumerate(zip(ANCHOR_WS_PORTS, ANCHOR_VID_PORTS)):
        server = AnchorArpServer(power=(i == 0))
        server.zc = zc
        # leave mock_camera_port unset so the server never advertises a video stream
        if not no_video:
            server.mock_camera_port = vid_port
        anchors.append(server)

    gripper = GripperArpServer()
    gripper.zc = zc
    if not no_video:
        gripper.mock_camera_port = GRIPPER_VID_PORT

    # ── launch everything ──────────────────────────────────────────────────────
    tasks = []
    streamer = None

    if not no_video and world is not None:
        # Real camera feeds: MuJoCo renders from the model's own cameras, sized from
        # component_server.stream_modes so they match what the hardware produces, and at
        # the field of view baked into the model from the config's camera calibration.
        import mujoco_bridge
        specs = mujoco_bridge.stream_specs(ANCHOR_VID_PORTS, GRIPPER_VID_PORT)
        for spec in specs:
            tasks.append(asyncio.create_task(mujoco_bridge.serve_stream(spec)))
        # queues are created by serve_stream on this loop, so let those start first
        await asyncio.sleep(0)
        streamer = mujoco_bridge.CameraStreamer(world, specs, asyncio.get_running_loop())
        streamer.start()

    elif not no_video:
        # video streams — anchor: 1920×1080 @ 10 fps / 520 kbps (matches stream_command in anchor_server.py)
        for vid_port in ANCHOR_VID_PORTS:
            tasks.append(asyncio.create_task(_video_stream_loop(vid_port, 1920, 1080, 10, '520k')))
        # gripper: 384×384 @ 60 fps / 1200 kbps (matches stream_command in gripper_arp_server.py)
        tasks.append(asyncio.create_task(_video_stream_loop(GRIPPER_VID_PORT, 384, 384, 60, '1200k')))

    for i, (server, ws_port) in enumerate(zip(anchors, ANCHOR_WS_PORTS)):
        tasks.append(asyncio.create_task(
            server.main(port=ws_port, name=f'cranebot-anchor-arpeggio-service.sim{i}')
        ))
    tasks.append(asyncio.create_task(
        gripper.main(port=GRIPPER_WS_PORT, name='cranebot-gripper-arpeggio-service.sim')
    ))

    if world is not None:
        # give the spool loops a moment to take their first reading, then tell them how
        # much line is really out (see mujoco_bridge.seed_reference_lengths)
        import mujoco_bridge
        await asyncio.sleep(0.5)
        mujoco_bridge.seed_reference_lengths(anchors)
        logging.info('mujoco: gantry at %s', world.gantry_position().round(3))

    if no_video:
        logging.info(
            'Simulator ready (no video). Anchor WS ports: %s | Gripper WS: %d',
            ANCHOR_WS_PORTS, GRIPPER_WS_PORT,
        )
    else:
        logging.info(
            'Simulator ready. '
            'Anchor WS ports: %s  video ports: %s | Gripper WS: %d  video: %d',
            ANCHOR_WS_PORTS, ANCHOR_VID_PORTS, GRIPPER_WS_PORT, GRIPPER_VID_PORT,
        )

    # ── run until SIGINT / SIGTERM ─────────────────────────────────────────────
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGINT, stop.set)
    loop.add_signal_handler(signal.SIGTERM, stop.set)

    await stop.wait()
    logging.info('Shutting down simulator...')

    if streamer is not None:
        logging.info('camera frames sent/dropped: %s', streamer.stats())
        streamer.stop()

    if world is not None:
        world.stop()

    for server in anchors:
        server.shutdown()
    gripper.shutdown()

    for t in tasks:
        t.cancel()
    try:
        await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=5)
    except asyncio.TimeoutError:
        logging.warning('Some tasks did not stop within 5 s')

    await zc.async_unregister_all_services()
    await zc.async_close()

    for p in patchers:
        p.stop()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Local robot component simulator.')
    parser.add_argument('--no-video', action='store_true',
                        help='Run without the ffmpeg test-pattern video streams.')
    parser.add_argument('--mujoco', nargs='?', const=True, default=None, metavar='MODEL.xml',
                        help='Back the stubbed hardware with the MuJoCo model instead of '
                             'constants, so the simulated robot actually moves. Optionally '
                             'takes a path; defaults to the model beside this script.')
    parser.add_argument('--realtime', type=float, default=1.0, metavar='X',
                        help='Physics speed multiplier for --mujoco. The servers are '
                             'real-time, so values far from 1.0 desynchronise them.')
    parser.add_argument('--room-side', type=float, default=None, metavar='M',
                        help='Side length of the square room in metres (default 4). The '
                             'four corner posts move to match, and the fixed run each '
                             'indirect line carries is remeasured with them.')
    parser.add_argument('--cam-tilt', type=float, default=None, metavar='DEG',
                        help='Anchor camera tilt below horizontal, i.e. which tilt '
                             'adapter is fitted (default 30; the anchor model also has '
                             '22 and 26 degree variants). The host must be told the same '
                             'angle - indirectLine.camTilt in the robot config - or '
                             'calibration solves against a camera pointing elsewhere.')
    args = parser.parse_args()

    model = None
    if args.mujoco is not None:
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from mujoco_bridge import DEFAULT_MODEL
        model = DEFAULT_MODEL if args.mujoco is True else args.mujoco

    if model is None and (args.room_side is not None or args.cam_tilt is not None):
        parser.error('--room-side and --cam-tilt only mean anything with --mujoco')

    asyncio.run(main(no_video=args.no_video, mujoco_model=model, realtime=args.realtime,
                     room_side=args.room_side, cam_tilt=args.cam_tilt))
