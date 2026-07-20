import asyncio
import time
from unittest.mock import patch

import pytest

# Several test suites are dominated by real-time sleeps in the code under test (polling
# loops, retry/restart waits, calibration delays) plus the tests sleeping to wait on them.
# Scaling sleep durations down by a constant keeps the *relative* ordering of events intact
# while cutting wall-clock time dramatically. asyncio.wait_for() timeouts are unaffected
# because they use loop.call_later, not asyncio.sleep, so they remain real-time safety nets
# that still catch genuine hangs.
#
# Only sleeps *longer* than a per-group threshold are compressed. Sleeps at or under the
# threshold are left alone because they are load-bearing at real time scale:
#   - short yields that let real websocket telemetry round-trip back to the test
#   - a test server's heartbeat whose 0.9s spacing must stay slower than a 0.1s check window
#   - waits for a server timeout that is measured against the real time.time() clock
#     (e.g. gripper ACTION_TIMEOUT=0.2), which scaling the wait alone would desync
# The observer suites have no such sub-second load-bearing sleeps and need their 0.5s
# polling ticks compressed, so they use a lower threshold than the client/server suites.
SLEEP_SPEEDUP = 20.0
SLEEP_SCALE_THRESHOLD = 0.15
_SLEEP_SCALED_MODULES = (
    "observer_connection_test.py",
    "observer_integration_test.py",
    "_client_test.py",   # arp_anchor_client_test.py
    "_server_test.py",   # anchor_arp_server_test.py, gripper_arp_server_test.py
)

# Individual tests opted out of sleep scaling: their timing is load-bearing at real-time
# scale and can't be uniformly compressed alongside the rest of the suite.
#   - test_line_record relies on the test server's 0.9s heartbeat staying slower than its
#     own 0.1s check window, so the heartbeat doesn't overwrite the record under test.
#   - test_wrist_speed_timeout_safety waits for a server timeout measured against the real
#     time.time() clock (ACTION_TIMEOUT=0.2); scaling that wait alone desyncs it.
# Both are already sub-second, so leaving them at real speed costs little.
_SLEEP_SCALE_EXCLUDE = (
    "test_line_record",
    "test_wrist_speed_timeout_safety",
)


@pytest.fixture(autouse=True)
def _fast_sleep(request):
    nodeid = request.node.nodeid
    if any(x in nodeid for x in _SLEEP_SCALE_EXCLUDE):
        yield
        return
    if not any(m in nodeid for m in _SLEEP_SCALED_MODULES):
        yield
        return

    real_async_sleep = asyncio.sleep
    real_time_sleep = time.sleep

    def scaled(delay):
        if delay and delay > SLEEP_SCALE_THRESHOLD:
            return delay / SLEEP_SPEEDUP
        return delay

    async def fast_async_sleep(delay, *args, **kwargs):
        return await real_async_sleep(scaled(delay), *args, **kwargs)

    def fast_time_sleep(secs):
        return real_time_sleep(scaled(secs))

    with patch("asyncio.sleep", fast_async_sleep), patch("time.sleep", fast_time_sleep):
        yield


# observer_integration_test.py spins up a full observer against mocked hardware servers.
# When something fundamental in observer.py is broken, that suite tends to hang (e.g. waiting
# forever on startup_complete) instead of failing fast. observer_connection_test.py exercises
# much of the same startup/connection machinery with lighter mocks and fails fast.
#
# So: if observer_connection_test.py had any failures, skip observer_integration_test.py
# rather than letting it hang. This relies on observer_connection_test.py running first,
# which it does because pytest collects test files in (alphabetical) path order and
# "observer_connection_test.py" sorts before "observer_integration_test.py".
CONNECTION_TEST_FILE = "observer_connection_test.py"
INTEGRATION_TEST_FILE = "observer_integration_test.py"

_connection_tests_failed = False


def pytest_runtest_logreport(report):
    global _connection_tests_failed
    if report.when == "call" and report.failed and CONNECTION_TEST_FILE in report.nodeid:
        _connection_tests_failed = True


def pytest_runtest_setup(item):
    if INTEGRATION_TEST_FILE in item.nodeid and _connection_tests_failed:
        pytest.skip(f"Skipping {INTEGRATION_TEST_FILE} because {CONNECTION_TEST_FILE} had failures")
