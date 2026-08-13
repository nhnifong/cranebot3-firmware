"""
Measure the offset between the appearance of the "Output #0..." log line from rpicam-vid
and the 0-point of the DTS clock observed by a stream consumer on the same machine.
The measured constant should be in the neighborhood of 0.5 seconds
this constant is dependent on the Raspberry Pi Zero 2W hardware and the system load.

Measures every named stream mode a component accepts, because the offset is a property of
the exact rpicam-vid command that gets launched: sensor mode, resolution, bitrate and
framerate all move it, and a named mode is exactly one such command. The mode names and
the stream commands are imported from the servers that run them, and combined with the
same function the servers use, so that this stays honest when those change.

Run this on the pi whose constants you want, and paste the printed values into the
dts_zero_offset field of those modes in component_server.stream_modes. The gripper's modes
need the Camera Module 3 Wide for their 2304:1296 sensor mode, so those measurements only
mean something on a gripper, and the anchor's only on an anchor.
"""

import time
import asyncio
import av
import re
from asyncio.subprocess import PIPE, STDOUT

from nf_robot.robot.component_server import apply_stream_mode, stream_modes

# each server module pulls in the hardware libraries for its own component at import time,
# which are not installed on the other one. A missing one should skip that server's
# measurements, not lose the other's.
import_errors = {}
try:
    from nf_robot.robot.anchor_arp_server import anchor_stream_modes
    from nf_robot.robot.component_server import stream_command as anchor_stream_command
except ImportError as e:
    anchor_stream_modes, anchor_stream_command = (), None
    import_errors['anchor'] = e

try:
    from nf_robot.robot.gripper_arp_server import (
        stream_command as gripper_stream_command,
        gripper_stream_modes)
except ImportError as e:
    gripper_stream_modes, gripper_stream_command = (), None
    import_errors['gripper'] = e

# the line we are looking for looks like this
#Output #0, mpegts, to 'tcp://0.0.0.0:8888?listen=1&tcp_nodelay=1':
# the query string is matched loosely so that adding url options does not break this.
ready_line_re = re.compile(r"^\s*Output #0, mpegts, to 'tcp://([^:]+):(\d+)\?[^']*':")
# regex to strip colors
ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

line_timeout = 30

async def run_rpicam_vid(event, stream_command_args):
    ready_wall_time = None
    rpicam_process = await asyncio.create_subprocess_exec(stream_command_args[0], *stream_command_args[1:], stdout=PIPE, stderr=STDOUT)
    # read all the lines of output
    while True:
        try:
            line = await asyncio.wait_for(rpicam_process.stdout.readline(), 30)
        except asyncio.TimeoutError:
            print(f'rpicam-vid wrote no lines for {line_timeout} seconds')
            rpicam_process.kill()
            break
        if not line: # EOF.
            print('rpicam-vid exited without showing the line we wanted to see')
            break
        line = line.decode()
        # remove color codes
        line = ansi_escape.sub('', line)
        print(line[:-1])

        # Look for the line indicating the stream is ready
        match = ready_line_re.match(line)
        if match:
            ready_wall_time = time.time()
            event.set()
            break

    # wait for the subprocess to exit.
    # it isn't going to exit until our client connects and disconnects.
    result_of_process = await rpicam_process.wait()
    return ready_wall_time

def connect_client():
    video_uri = f'tcp://127.0.0.1:8888'
    print(f'Connecting to {video_uri}')

    options = {
        'rtsp_transport': 'tcp',
        'fflags': 'nobuffer',
        'flags': 'low_delay',
        'fast': '1',
    }
    count = 30

    container = av.open(video_uri, options=options, mode='r')
    stream = next(s for s in container.streams if s.type == 'video')
    stream.thread_type = "SLICE"

    # collect 30 frames, average the measurement
    dts_zero_estimates = []

    for av_frame in container.decode(stream):
        dts_zero_time = time.time() - av_frame.time
        dts_zero_estimates.append(dts_zero_time)
        if len(dts_zero_estimates) == count:
            break

    container.close()
    return sum(dts_zero_estimates)/count

async def measure(name, stream_command_args):
    print(f'\n===== {name} =====')
    print(' '.join(stream_command_args))
    event = asyncio.Event()
    server = asyncio.create_task(run_rpicam_vid(event, stream_command_args))
    await event.wait()
    await asyncio.sleep(2)
    dts_zero_walltime = await asyncio.to_thread(connect_client)
    ready_line_walltime = await server
    offset = dts_zero_walltime - ready_line_walltime
    print(f'{name}: DTS zero occurs {offset}s after the ready line is printed by rpicam-vid')
    return offset

async def measure_modes(stream_command, mode_names, results):
    """Measure each named mode, in the command that mode really launches."""
    for name in mode_names:
        # rpicam-vid holds port 8888 for a few seconds after it exits, and the next
        # measurement cannot start until it lets go
        if results:
            await asyncio.sleep(5)
        results[name] = await measure(name, apply_stream_mode(stream_command, stream_modes[name]))

async def run_experiment():
    results = {}

    await measure_modes(anchor_stream_command, anchor_stream_modes, results)
    await measure_modes(gripper_stream_command, gripper_stream_modes, results)

    for component, error in import_errors.items():
        print(f'\nskipping {component} modes, could not import them: {error}')

    print('\n===== summary =====')
    print('paste each value into the dts_zero_offset of that mode in stream_modes')
    for name, offset in results.items():
        print(f'{name}: {offset}')

asyncio.run(run_experiment())
