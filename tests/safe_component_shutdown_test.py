import unittest
from unittest.mock import AsyncMock

from nf_robot.generated.nf import control
from nf_robot.host.observer import AsyncObserver


class TestSafeComponentShutdown(unittest.IsolatedAsyncioTestCase):
    def _make_observer(self):
        ob = AsyncObserver(terminate_with_ui=False, config_path=None, port=0)
        ob.stop_all = AsyncMock()
        ob.send_ui = lambda **kwargs: self.ui_messages.append(kwargs)
        self.ui_messages = []
        return ob

    def _client(self):
        c = AsyncMock()
        c.send_commands = AsyncMock()
        return c

    async def test_command_asks_every_component_to_halt(self):
        ob = self._make_observer()
        ob.bot_clients = {'anchor0': self._client(), 'gripper': self._client()}

        await ob._handle_common_command(control.Command.SAFE_COMPONENT_SHUTDOWN)

        for client in ob.bot_clients.values():
            client.send_commands.assert_awaited_once_with({'shutdown_pi': True})
        # nothing may still be commanding motion into a halting component
        ob.stop_all.assert_awaited_once()
        self.assertEqual(len(self.ui_messages), 1)

    async def test_one_unreachable_component_does_not_stop_the_rest(self):
        ob = self._make_observer()
        dead = self._client()
        dead.send_commands = AsyncMock(side_effect=ConnectionError('gone'))
        alive = self._client()
        ob.bot_clients = {'anchor0': dead, 'gripper': alive}

        await ob.shutdown_all_bots()

        alive.send_commands.assert_awaited_once_with({'shutdown_pi': True})
        [message] = self.ui_messages
        self.assertIn('anchor0', message['operation_progress'].current_action)

    async def test_no_components_connected(self):
        ob = self._make_observer()
        ob.bot_clients = {}

        await ob.shutdown_all_bots()

        ob.stop_all.assert_not_awaited()
        self.assertEqual(len(self.ui_messages), 1)


if __name__ == '__main__':
    unittest.main()
