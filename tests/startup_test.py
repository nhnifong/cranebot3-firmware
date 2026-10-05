"""Launch the installed stringman-headless command the way a user would and check that it
comes up and opens its local telemetry websocket. The rest of the suite drives AsyncObserver
directly, so nothing else catches a failure in main() itself: a missing ffmpeg, a
platform-specific signal API, a console script that doesn't resolve."""
import os
import shutil
import signal
import socket
import subprocess
import sys
import time

import pytest

# torch and friends import slowly on a cold Windows runner
STARTUP_TIMEOUT = 120
# the local telemetry websocket, opened near the end of AsyncObserver.main()'s startup
TELEMETRY_PORT = 4245


def _port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('127.0.0.1', port)) == 0


def test_stringman_headless_starts(tmp_path):
    exe = shutil.which('stringman-headless')
    if exe is None:
        pytest.skip('stringman-headless is not installed (pip install ".[host]")')
    # the local telemetry websocket's port is fixed; a real observer on this machine holds it
    if _port_in_use(TELEMETRY_PORT):
        pytest.skip(f'port {TELEMETRY_PORT} is in use, probably by a running observer')

    log_path = tmp_path / 'observer.log'
    with open(log_path, 'w') as log:
        proc = subprocess.Popen(
            [exe, '--no_ortho', '--config', str(tmp_path / 'configuration.json'),
             # CI has no built playroom-ui to serve
             '--no_serve_ui'],
            cwd=tmp_path, stdout=log, stderr=subprocess.STDOUT,
            # otherwise its prints sit in a buffer that the kill below throws away
            env={**os.environ, 'PYTHONUNBUFFERED': '1'},
        )
        try:
            deadline = time.monotonic() + STARTUP_TIMEOUT
            served = False
            while time.monotonic() < deadline and proc.poll() is None:
                if _port_in_use(TELEMETRY_PORT):
                    served = True
                    break
                time.sleep(1)
            exited = proc.poll()
            if exited is None and not served and hasattr(signal, 'SIGUSR1'):
                # main() dumps every thread's stack and pending asyncio task on SIGUSR1,
                # which shows where startup hung
                proc.send_signal(signal.SIGUSR1)
                time.sleep(3)
        finally:
            proc.kill()
            proc.wait()

    output = log_path.read_text(errors='replace')
    assert exited is None, f'stringman-headless exited with {exited} on {sys.platform}:\n{output}'
    assert served, f'telemetry port {TELEMETRY_PORT} never opened within {STARTUP_TIMEOUT}s:\n{output}'
