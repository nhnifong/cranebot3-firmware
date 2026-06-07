import pytest

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
