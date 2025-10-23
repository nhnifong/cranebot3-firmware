"""
Measure the offset between the appearance of the "Output #0..." log line from rpicam-vid
and the 0-point of the DTS clock observed by a stream consumer on the same machine.
The measured constant should be in the neighborhood of 0.5 seconds
this constant is dependent on the Raspberry Pi Zero 2W hardware and the system load.
"""

import time
import asyncio
import av
import re
from asyncio.subprocess import PIPE, STDOUT

stream_command_args = [
    "/usr/bin/rpicam-vid", "-t", "0", "-n",
    "--width=1920", "--height=1080",
    "-o", "tcp://0.0.0.0:8888?listen=1",
    "--codec", "libav",
    "--libav-format", "mpegts",
    "--vflip", "--hflip",
    "--autofocus-mode", "continuous",
    "--low-latency"
]

# the line we are looking for looks like this
#Output #0, mpegts, to 'tcp://0.0.0.0:8888?listen=1':
ready_line_re = re.compile(r"^\s*Output #0, mpegts, to 'tcp://([^:]+):(\d+)\?listen=1':")
# regex to strip colors
ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

line_timeout = 30

async def run_rpicam_vid(event):
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

async def run_experiment():
	event = asyncio.Event()
	server = asyncio.create_task(run_rpicam_vid(event))
	await event.wait()
	dts_zero_walltime = await asyncio.to_thread(connect_client)
	ready_line_walltime = await server
	offset = dts_zero_walltime - ready_line_walltime
	print(f'DTS zero occurs {offset}s after the ready line is printed by rpicam-vid')

asyncio.run(run_experiment())