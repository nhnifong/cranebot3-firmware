"""
Measure the offset between the appearance of the "Output #0..." log line from rpicam-vid
and the 0-point of the DTS clock observed by a stream consumer on the same machine.
The measured constant should be in the neighborhood of 0.5 seconds
this constant is dependent on the Raspberry Pi Zero 2W hardware and the system load.

Measures every named stream mode a component accepts, because the offset is a property of
the exact rpicam-vid command that gets launched: sensor mode, resolution, bitrate and
framerate all move it, and a named mode is exactly one such command. The mode names and
the stream commands are imported from the servers that run them, and combined with the
same function the servers use, so that this stays honest when those change. Pass
--mode NAME to measure a single mode, for when only one of them needs a number.

Each mode is measured several times and averaged, because the offset is mostly camera and
encoder startup and that costs a different amount each launch.

Run this on the pi whose constants you want, and paste the printed averages into the
dts_zero_offset field of those modes in component_server.stream_modes. The gripper's modes
need the Camera Module 3 Wide for their 2304:1296 sensor mode, so those measurements only
mean something on a gripper, and the anchor's only on an anchor.
"""

import argparse
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
    from nf_robot.robot.anchor_arp_server import (
        anchor_stream_modes, anchor_stream_modes_3a_plus)
    from nf_robot.robot.component_server import stream_command as anchor_stream_command
    # every mode an anchor can run on any board, not only the ones the board this is
    # running on starts in: measuring a mode is how it earns an offset, and a faster
    # board's mode still has to be measured on that board before it can be used there.
    anchor_stream_modes = tuple(dict.fromkeys(
        anchor_stream_modes + anchor_stream_modes_3a_plus))
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

# each component's stream command with the modes that command can be given. A component
# whose import failed contributes no modes, so nothing ever reaches its None command.
components = (
    (anchor_stream_command, anchor_stream_modes),
    (gripper_stream_command, gripper_stream_modes),
)

# the line we are looking for looks like this
#Output #0, mpegts, to 'tcp://0.0.0.0:8888?listen=1&tcp_nodelay=1':
# the query string is matched loosely so that adding url options does not break this.
ready_line_re = re.compile(r"^\s*Output #0, mpegts, to 'tcp://([^:]+):(\d+)\?[^']*':")
# regex to strip colors
ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

line_timeout = 30

# how many times each mode is launched and measured. Four is enough to see whether a mode's
# offset is a constant at all without making a full run take all afternoon - every sample
# is a camera startup plus the five second wait for the port to come back.
samples_per_mode = 4

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
    """Measure each named mode samples_per_mode times, in the command it really launches.

    Averaging whole runs rather than only frames within one run is what covers the part
    that actually varies: camera startup is a different amount of work every time, and a
    single run cannot tell a representative startup from an unlucky one.
    """
    for name in mode_names:
        command = apply_stream_mode(stream_command, stream_modes[name])
        for sample in range(samples_per_mode):
            # rpicam-vid holds port 8888 for a few seconds after it exits, and the next
            # run cannot start until it lets go
            if results:
                await asyncio.sleep(5)
            results.setdefault(name, []).append(
                await measure(f'{name} sample {sample + 1}/{samples_per_mode}', command))

def select(only_mode):
    """The (command, mode names) pairs to measure, narrowed to one mode if one was named.

    A named mode is looked up in the components rather than in the whole stream_modes table
    so that asking for another component's mode fails here, saying so, instead of measuring
    the wrong camera command and printing a plausible number for it.
    """
    if only_mode is None:
        return components
    for command, mode_names in components:
        if only_mode in mode_names:
            return ((command, (only_mode,)),)
    known = ', '.join(sorted({n for _, names in components for n in names})) or 'none'
    skipped = ''.join(f'\n{c} modes could not be imported here: {e}'
                      for c, e in import_errors.items())
    raise SystemExit(f'no component here accepts stream mode {only_mode!r}.'
                     f'\nmeasurable on this machine: {known}{skipped}')


async def run_experiment(only_mode=None):
    results = {}

    for command, mode_names in select(only_mode):
        await measure_modes(command, mode_names, results)

    for component, error in import_errors.items():
        print(f'\nskipping {component} modes, could not import them: {error}')

    print('\n===== summary =====')
    print('paste each average into the dts_zero_offset of that mode in stream_modes')
    for name, offsets in results.items():
        average = sum(offsets)/len(offsets)
        # the spread is the honest error bar on the average: a mode whose runs disagree by
        # more than a frame interval is not a constant, and averaging it harder won't help
        spread = max(offsets) - min(offsets)
        print(f'{name}: {average} (spread {spread:.4f} over {len(offsets)} samples)')
        print(f'    samples: {", ".join(f"{o:.4f}" for o in offsets)}')

parser = argparse.ArgumentParser(
    description='Measure the dts zero offset of this component\'s stream modes.')
parser.add_argument('--mode', metavar='NAME', default=None,
                    help='measure only this named stream mode, instead of every mode a '
                         'component on this machine accepts')
args = parser.parse_args()

asyncio.run(run_experiment(args.mode))
