"""Ferry: carrying things between named places, and following one."""

import asyncio
import logging

from nf_robot.host.maneuver import Maneuver, prefer_swing_cancellation, verb
from nf_robot.host.maneuvers.pick_and_place import GRIPPER_HEIGHT_OVER_TARGET

logger = logging.getLogger(__name__)


class Ferry(Maneuver):
    name = 'ferry'
    title = 'Ferry'

    async def _wait_for(self, name):
        """The named place's position, once it has been seen."""
        while self.ob.named_position(name) is None:
            await asyncio.sleep(0.5)
        return self.ob.named_position(name)

    @verb('ferry', motion=True)
    @prefer_swing_cancellation
    async def ferry(self, source='hamper', dest='trash'):
        """Carry objectes between one named tag and another.
        Moves to source, attempt auto grasp, move to test, drop, repeat.
        'ferry' alone carries from the hamper to the trash; 'ferry SOURCE DEST' names them."""
        ob = self.ob
        try:
            while True:
                await asyncio.sleep(0.1)

                # wait for source position to be seen, then go there
                goal = await self._wait_for(source) + ob.pole_offset() + GRIPPER_HEIGHT_OVER_TARGET
                await ob.seek_goal(goal)

                await ob.grasp()

                # wait for destination position to be seen, then go there
                goal = await self._wait_for(dest) + ob.pole_offset() + GRIPPER_HEIGHT_OVER_TARGET
                await ob.seek_goal(goal)

                # drop
                await ob.set_finger_angle(-30)
                await asyncio.sleep(1)

        except asyncio.CancelledError:
            raise

    async def chase_tag(self, name):
        """Keep the gripper at the named location, following it as it moves."""
        while True:
            await asyncio.sleep(0.1)
            position = self.ob.named_position(name)
            if position is None:
                continue
            # re-aims the one flight rather than starting another each time
            await self.ob.seek_goal(position + self.ob.pole_offset(), timeout=0)
