"""Launch the installed stringman-headless command the way a user would and check that it
comes up and serves the UI. The rest of the suite drives AsyncObserver directly, so nothing
else catches a failure in main() itself: a missing ffmpeg, a platform-specific signal API,
a console script that doesn't resolve."""
import shutil
import socket
import subprocess
import sys
import time
import urllib.request

import pytest

from port_utils import free_port

# torch and friends import slowly on a cold Windows runner
STARTUP_TIMEOUT = 120


def _port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('127.0.0.1', port)) == 0


def test_stringman_headless_starts(tmp_path):
    exe = shutil.which('stringman-headless')
    if exe is None:
        pytest.skip('stringman-headless is not installed (pip install ".[host]")')
    # the local telemetry websocket's port is fixed; a real observer on this machine holds it
    if _port_in_use(4245):
        pytest.skip('port 4245 is in use, probably by a running observer')

    ui_port = free_port()
    log_path = tmp_path / 'observer.log'
    with open(log_path, 'w') as log:
        proc = subprocess.Popen(
            [exe, '--no_ortho', '--config', str(tmp_path / 'configuration.json'),
             '--ui_port', str(ui_port)],
            cwd=tmp_path, stdout=log, stderr=subprocess.STDOUT,
        )
        try:
            deadline = time.monotonic() + STARTUP_TIMEOUT
            served = False
            while time.monotonic() < deadline and proc.poll() is None:
                try:
                    with urllib.request.urlopen(f'http://127.0.0.1:{ui_port}/', timeout=2) as r:
                        served = r.status == 200
                    break
                except OSError:
                    time.sleep(1)
            exited = proc.poll()
        finally:
            proc.kill()
            proc.wait()

    output = log_path.read_text(errors='replace')
    assert exited is None, f'stringman-headless exited with {exited} on {sys.platform}:\n{output}'
    assert served, f'UI never came up on port {ui_port} within {STARTUP_TIMEOUT}s:\n{output}'
