"""Basket episodes: recording lerobot episodes of the gripper leaving a drop position, for
learning to center over baskets."""

import asyncio
import logging
import time

import numpy as np

from nf_robot.generated.nf import common
from nf_robot.host.maneuver import Maneuver, TiltWatch, prefer_swing_cancellation, verb

logger = logging.getLogger(__name__)

MAX_EPISODES = 50
# (metres) how far each episode carries the gripper from the drop position. Lateral is in a
# random direction; vertical is up or down.
LATERAL_MAX_M = 1.0
VERTICAL_MAX_M = 0.5
# (metres) a goal that would put the gripper lower than this over the floor is redrawn.
MIN_GRIPPER_Z_M = 0.15
# Goals are redrawn this many times looking for one inside the work area before giving up.
GOAL_DRAWS = 30
# (degrees) the wrist turns by up to this much either way before each episode, and is kept
# inside its range of travel with this much to spare.
WRIST_OFFSET_MAX_DEG = 180.0
WRIST_MIN_DEG = 20.0
WRIST_MAX_DEG = 1060.0
# (seconds) wait after the wrist turn before an episode starts, so the pole has stopped
# swinging and the episode's first frame is a still, centred view of the basket.
SETTLE_S = 1.5
# A lean this steep held this long is a strike, not a swing. Free moves with swing
# cancellation lean a few degrees.
TILT_DEG = 12.0
TILT_CONFIRM_S = 0.3
# (seconds) an episode is abandoned when no anchor camera has seen the gantry marker in this
# long. Far stricter than the observer's own warning, because a stretch of episode with no
# visual fix is a stretch the position in the data was only estimated from line lengths.
MARKER_UNSEEN_S = 3.0
# (seconds) how long the lerobot session may take to answer.
START_TIMEOUT_S = 10.0
# Saving an episode encodes its video, so becoming ready again can take a while.
READY_TIMEOUT_S = 120.0
WATCH_POLL_S = 0.05


class BasketEpisodes(Maneuver):
    name = 'basket_episodes'
    title = 'Basket episodes'

    def __init__(self, ob):
        super().__init__(ob)
        self.rng = np.random.default_rng()
        # (kind, anchor_num) of a component that dropped during a run, or None
        self.disconnected = None

    def on_component_disconnected(self, kind, anchor_num=None):
        self.disconnected = (kind, anchor_num)

    @verb('basketdata', motion=True)
    @prefer_swing_cancellation
    async def record(self, count=MAX_EPISODES):
        """Record episodes of the gripper leaving the drop position, for centring over baskets.

        Put the gripper at an ideal drop position over a basket, start a lerobot recording
        session, then type 'basketdata' (or 'basketdata N' for other than 50 episodes). Before
        each episode the wrist turns to a random angle and the pole is left to settle, so the
        episode's first frame is a centred view of the basket. The episode itself is one
        flight up to a metre aside and half a metre up or down. Between episodes it flies back
        to the drop position, unrecorded, and leaves the wrist where it is. Stopping it ends
        the recording session.
        """
        ob = self.ob
        count = int(count)
        start = ob.gantry_position()
        self.disconnected = None
        in_episode = False
        recorded = abandoned = 0
        try:
            if not await ob.maneuver('lerobot').session_connected():
                self.notify('No lerobot session is connected. Start a recording session first.')
                return
            if not await self._wait_status(common.LerobotStatus.REC_READY, START_TIMEOUT_S):
                self.notify(f'The lerobot session is not ready to record '
                            f'(status {self._session_status()!r}).')
                return
            logger.info(f'Basket episodes: {count} from drop position {np.round(start, 3)}')

            for attempt in range(count):
                self.progress(100.0 * attempt / count,
                              f'Episode {attempt + 1} of {count}: {recorded} recorded, '
                              f'{abandoned} abandoned')
                # unrecorded: a new wrist angle, held still at the drop position
                problem = await self._turn_wrist()
                if problem:
                    self.notify(f'Stopping basket episodes while turning the wrist: {problem}')
                    return
                await asyncio.sleep(SETTLE_S)
                goal = self._away_goal(start)
                if goal is None:
                    self.notify('No goal within reach of the drop position is inside the work '
                                'area; stopping basket episodes.')
                    return
                if not self._marker_seen():
                    self.notify(f'No anchor camera has seen the gripper marker in '
                                f'{MARKER_UNSEEN_S:.0f}s, so no episode was started.')
                    return

                self._episode_command(common.EpCommand.EVAL_START)
                await ob.flush_tele_buffer()
                if not await self._wait_status(common.LerobotStatus.RECORDING, START_TIMEOUT_S):
                    self.notify(f'The lerobot session did not start an episode '
                                f'(status {self._session_status()!r}).')
                    return
                in_episode = True

                problem = await self._watch(ob.seek_goal(goal, auto_altitude=False),
                                            TiltWatch(ob, TILT_DEG, TILT_CONFIRM_S))
                ob.slow_stop_all_spools()
                await ob.clear_goal()
                if problem:
                    logger.warning(f'Basket episodes: abandoning episode {attempt + 1}: {problem}')
                    self._episode_command(common.EpCommand.ABANDON)
                    abandoned += 1
                else:
                    self._episode_command(common.EpCommand.EVAL_STOP)
                    recorded += 1
                in_episode = False
                await ob.flush_tele_buffer()

                if self.disconnected is not None:
                    kind, anchor_num = self.disconnected
                    which = kind if anchor_num is None else f'{kind} {anchor_num}'
                    self.notify(f'The {which} disconnected; stopping basket episodes.')
                    return

                # back to the drop position, unrecorded; the wrist stays where it is
                problem = await self._watch(ob.seek_goal(start, auto_altitude=False),
                                            TiltWatch(ob, TILT_DEG, TILT_CONFIRM_S))
                ob.slow_stop_all_spools()
                await ob.clear_goal()
                if problem:
                    self.notify(f'Stopping basket episodes on the way back to the drop '
                                f'position: {problem}')
                    return

                if not await self._wait_status(common.LerobotStatus.REC_READY, READY_TIMEOUT_S):
                    self.notify(f'The lerobot session did not become ready for another episode '
                                f'(status {self._session_status()!r}).')
                    return

            self.notify(f'Basket episodes done: {recorded} recorded, {abandoned} abandoned.')
        finally:
            if in_episode:
                self._episode_command(common.EpCommand.ABANDON)
            self._episode_command(common.EpCommand.END_RECORDING)
            logger.info(f'Basket episodes ended: {recorded} recorded, {abandoned} abandoned; '
                        f'ending the recording session')
            self.finish(f'{recorded} recorded, {abandoned} abandoned')
            ob.slow_stop_all_spools()
            await ob.clear_goal()
            await ob.flush_tele_buffer()

    async def _turn_wrist(self):
        """Turn the wrist by a random offset. None if it went cleanly, or what went wrong."""
        wrist = self.ob.wrist_angle()
        offset = self.rng.uniform(-WRIST_OFFSET_MAX_DEG, WRIST_OFFSET_MAX_DEG)
        if not WRIST_MIN_DEG <= wrist + offset <= WRIST_MAX_DEG:
            offset = -offset
        target = float(np.clip(wrist + offset, WRIST_MIN_DEG, WRIST_MAX_DEG))
        return await self._watch(self.ob.ease_wrist(target),
                                 TiltWatch(self.ob, TILT_DEG, TILT_CONFIRM_S))

    def _away_goal(self, start):
        """A random gantry goal up to LATERAL_MAX_M aside and VERTICAL_MAX_M above or below start,
        inside the work area and with the gripper clear of the floor, or None if none was
        found."""
        # the gantry goal minus this is where the gripper ends up
        pole_offset = self.ob.pole_offset()
        for _ in range(GOAL_DRAWS):
            heading = self.rng.uniform(0, 2 * np.pi)
            lateral = self.rng.uniform(0, LATERAL_MAX_M)
            goal = start + np.array([lateral * np.cos(heading), lateral * np.sin(heading),
                                     self.rng.uniform(-VERTICAL_MAX_M, VERTICAL_MAX_M)])
            if (self.ob.inside_work_area_2d(goal)
                    and (goal - pole_offset)[2] >= MIN_GRIPPER_Z_M):
                return goal
        return None

    async def _watch(self, coro, tilt):
        """Await a motion coroutine, checking for anomalies while it runs. None if it finished
        cleanly, or what went wrong, in which case it has been cancelled."""
        task = asyncio.ensure_future(coro)
        try:
            while not task.done():
                problem = self._anomaly(tilt)
                if problem:
                    return problem
                await asyncio.wait([task], timeout=WATCH_POLL_S)
            if task.result() is False:
                return 'the move ended before arriving'
            return None
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    def _anomaly(self, tilt):
        """Why the robot's state makes the current motion unusable, or None."""
        if self.disconnected is not None:
            kind, anchor_num = self.disconnected
            return f'the {kind if anchor_num is None else f"{kind} {anchor_num}"} disconnected'
        if not self.ob.gripper_connected():
            return 'the gripper is not connected'
        caught = tilt.check()
        if caught:
            return caught
        if not self._marker_seen():
            return f'no anchor camera has seen the gantry marker in {MARKER_UNSEEN_S:.0f}s'
        status = self._session_status()
        if status == common.LerobotStatus.ERROR:
            return 'the lerobot session reported an error'
        return None

    def _marker_seen(self):
        return len(self.ob.fresh_gantry_sightings(MARKER_UNSEEN_S)) > 0

    def _episode_command(self, command):
        self.ob.send_ui(episode_control=common.EpisodeControl(command=command))

    def _session_status(self):
        """The LerobotStatus the connected session last reported."""
        status = self.ob.maneuver('lerobot').last_status
        if isinstance(status, common.LerobotSessionStatus):
            return status.status
        return status

    async def _wait_status(self, wanted, timeout):
        """Wait for the lerobot session to report wanted. False on timeout or an error."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            status = self._session_status()
            if status == wanted:
                return True
            if status == common.LerobotStatus.ERROR:
                return False
            await asyncio.sleep(WATCH_POLL_S)
        return False
