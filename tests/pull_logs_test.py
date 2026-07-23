import os
import tempfile
import zipfile
import unittest
from unittest.mock import AsyncMock, patch

from nf_robot.host.observer import AsyncObserver
from nf_robot.host.component_client import ComponentClient
from nf_robot.robot import component_server


class TestPullLogsToZip(unittest.IsolatedAsyncioTestCase):
    def _make_observer(self):
        return AsyncObserver(terminate_with_ui=False, config_path=None, port=0)

    async def test_pull_logs_to_zip_writes_one_entry_per_client(self):
        ob = self._make_observer()
        clients = {}
        # one client with a thermal log, one without (e.g. older firmware)
        for ip, logs in [('10.0.0.1', ('anchor0 log\n', 'thermal0 log\n')),
                         ('10.0.0.2', ('anchor1 log\n', None))]:
            c = AsyncMock()
            c.address = ip
            c.pull_logs = AsyncMock(return_value=logs)
            clients[ip] = c
        ob.bot_clients = clients

        cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)
            try:
                await ob.pull_logs_to_zip()
                zips = [f for f in os.listdir(tmpdir) if f.startswith('pulled_logs_') and f.endswith('.zip')]
                self.assertEqual(len(zips), 1)
                with zipfile.ZipFile(zips[0]) as zf:
                    self.assertEqual(set(zf.namelist()),
                                     {'10.0.0.1.log', '10.0.0.1_thermal.log', '10.0.0.2.log'})
                    self.assertEqual(zf.read('10.0.0.1.log').decode(), 'anchor0 log\n')
                    self.assertEqual(zf.read('10.0.0.1_thermal.log').decode(), 'thermal0 log\n')
            finally:
                os.chdir(cwd)

    async def test_pull_logs_to_zip_skips_clients_that_didnt_respond(self):
        ob = self._make_observer()
        c1 = AsyncMock(); c1.address = '10.0.0.1'; c1.pull_logs = AsyncMock(return_value=('ok\n', None))
        c2 = AsyncMock(); c2.address = '10.0.0.2'; c2.pull_logs = AsyncMock(return_value=(None, None))
        ob.bot_clients = {'a': c1, 'b': c2}

        cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)
            try:
                await ob.pull_logs_to_zip()
                [zip_name] = [f for f in os.listdir(tmpdir) if f.endswith('.zip')]
                with zipfile.ZipFile(zip_name) as zf:
                    self.assertEqual(zf.namelist(), ['10.0.0.1.log'])
            finally:
                os.chdir(cwd)


class TestComponentClientPullLogs(unittest.IsolatedAsyncioTestCase):
    def _make_client(self):
        client = ComponentClient.__new__(ComponentClient)
        client.connected = True
        client.address = '10.0.0.5'
        client.pulled_logs = None
        client.pulled_thermal = None
        return client

    async def test_pull_logs_returns_response(self):
        client = self._make_client()

        async def fake_send(update):
            self.assertEqual(update, {'get_logs': None})
            client.pulled_logs = 'line1\nline2\n'
            client.pulled_thermal = 'thermal1\n'
        client.send_commands = AsyncMock(side_effect=fake_send)

        result = await client.pull_logs(timeout=2)
        self.assertEqual(result, ('line1\nline2\n', 'thermal1\n'))

    async def test_pull_logs_returns_none_if_disconnected_before_response(self):
        client = self._make_client()
        client.connected = False  # simulate dropping before ever getting a reply
        client.send_commands = AsyncMock()

        result = await client.pull_logs(timeout=2)
        self.assertEqual(result, (None, None))


class TestServerReadRecentLogs(unittest.TestCase):
    def _make_server(self):
        return component_server.RobotComponentServer.__new__(component_server.RobotComponentServer)

    def test_tails_last_n_lines(self):
        server = self._make_server()
        with tempfile.NamedTemporaryFile('w', suffix='.log', delete=False) as f:
            for i in range(10):
                f.write(f'line {i}\n')
            path = f.name
        try:
            with patch.object(component_server, 'log_path', path), \
                 patch.object(component_server, 'log_tail_lines', 3):
                result = server.read_recent_logs()
            self.assertEqual(result, 'line 7\nline 8\nline 9\n')
        finally:
            os.unlink(path)

    def test_missing_file_returns_empty_string(self):
        server = self._make_server()
        with patch.object(component_server, 'log_path', '/nonexistent/path/does/not/exist.log'):
            result = server.read_recent_logs()
        self.assertEqual(result, '')


if __name__ == '__main__':
    unittest.main()
