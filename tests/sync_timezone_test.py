import unittest
from unittest.mock import AsyncMock, patch

from nf_robot.host.observer import AsyncObserver


class TestSyncTimezone(unittest.IsolatedAsyncioTestCase):
    """Timezone sync must work on every host OS the observer runs on, including
    Windows, which has no `timedatectl`. These tests run in CI on windows-latest."""

    def _make_observer(self):
        # config_path=None gives a fresh default config; nothing here touches the
        # network or the event loop, so no heavy mocking of clients is needed.
        return AsyncObserver(terminate_with_ui=False, config_path=None, port=0)

    def test_get_local_timezone_name_returns_iana(self):
        """The bots expect an IANA name (e.g. 'America/New_York'); we must be able
        to resolve one on the host regardless of platform."""
        tz = AsyncObserver._get_local_timezone_name()
        self.assertIsInstance(tz, str)
        self.assertTrue(tz)

    async def test_sync_timezone_to_bots_sends_to_all_clients(self):
        ob = self._make_observer()
        clients = {name: AsyncMock() for name in ('anchor0', 'anchor1', 'gripper')}
        ob.bot_clients = clients

        await ob.sync_timezone_to_bots()

        tz = AsyncObserver._get_local_timezone_name()
        for client in clients.values():
            client.send_commands.assert_awaited_once_with({'set_timezone': tz})

    async def test_sync_timezone_skips_when_unresolved(self):
        """If the timezone can't be determined, we should skip rather than crash
        the caller (this path used to raise FileNotFoundError on Windows)."""
        ob = self._make_observer()
        client = AsyncMock()
        ob.bot_clients = {'anchor0': client}

        with patch.object(AsyncObserver, '_get_local_timezone_name', return_value=None):
            await ob.sync_timezone_to_bots()

        client.send_commands.assert_not_awaited()


if __name__ == '__main__':
    unittest.main()
