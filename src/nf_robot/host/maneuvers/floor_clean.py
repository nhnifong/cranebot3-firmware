"""Floor clean: clearing a floor into containers from a queue of jobs.

The robot works through the queue on its own, and an agent watching it adds jobs and changes
where a carried item goes without stopping it. An item is carried on the assumption that the
grasp got what the job expected, so the agent looks at the gripper image while the robot is
already on its way, and redirects only if the grasp got something else.

    cleandest pit -1.5 -1.75 1.0 0.4       a big target: open on the fly inside the circle
    cleandest bin 0.66 -0.04 0.6           a small target: center, descend, then open
    cleanadd 0.2,0.3,pit -0.1,0.5,bin      queue jobs
    clean                                  run the queue as the motion task
    cleanto bin                            send what is being carried somewhere else

See "Clearing a floor" in mcp_bridge/GUIDE.md.
"""

import asyncio
import json
import logging
import math
import time

import numpy as np

from nf_robot.host.maneuver import Maneuver, verb

logger = logging.getLogger(__name__)

# (seconds) with an empty queue, how long 'clean' waits for more jobs before it finishes
IDLE_LIMIT_S = 45.0
# (meters) gripper height to start a grasp from, unless the job says otherwise
DEFAULT_HOVER_M = 0.55
# how far the visual servo may steer from where the grasp started, unless the job says otherwise
DEFAULT_WANDER_M = 0.5
DEFAULT_TRIES = 2
# Half calibrate when the last one is older than this: before a grasp, and before a drop that
# needs centering. Height drifts over repeated dives.
CAL_MAX_AGE_S = 120.0
CAL_MAX_AGE_BEFORE_DROP_S = 45.0

# (meters) minimum gripper height to fly over a container at, and how close counts as over it
CARRY_HEIGHT_M = 0.85
OVER_CONTAINER_TOL_M = 0.06
# A big target's opening can be entered this far below its drop height.
RADIUS_DROP_HEIGHT_MARGIN_M = 0.25
OPEN_ANGLE = -60
OPEN_BETWEEN_TRIES_ANGLE = -40
# (seconds) re-aiming interval while carrying, so a redirect takes effect without a stop
REAIM_S = 0.1
SEEK_TIMEOUT_S = 15.0
SEEK_TO_DROP_TIMEOUT_S = 8.0
# (seconds) to let a swing settle before opening over a small target, and to let the item fall
SETTLE_S = 0.5
RELEASE_S = 0.5
RELEASE_ON_THE_FLY_S = 0.4

# Fingers closed this far with this little pad force means nothing is held. Thin items such as
# a rubber glove look the same, so a job flagged 'trust' skips this check.
EMPTY_FINGER_ANGLE = 85
EMPTY_PAD_VOLTAGE = 0.06


class FloorClean(Maneuver):
    name = 'floorclean'
    title = 'Cleaning the floor'

    def __init__(self, ob):
        super().__init__(ob)
        self.dests = {}
        self.jobs = []
        self.job = None
        self.dest = None
        # whether cleanto can still change where the current item goes
        self.can_redirect = False
        self.last_cal = 0.0
        self.stats_path = None
        self.rows = []

    @verb('cleandest')
    async def set_dest(self, name, x, y, z, radius='-', wrist='-'):
        """Set a place to let go, as a gripper position. Give a radius (or '-') for a big target
        that is dropped into as soon as the gripper is inside the circle, and a wrist angle
        (or '-') to turn to first, which lines up long items with a container."""
        dest = dict(xyz=[float(x), float(y), float(z)])
        if radius != '-':
            dest['radius'] = float(radius)
        if wrist != '-':
            dest['wrist'] = float(wrist)
        self.dests[name] = dest
        return dest

    @verb('cleanadd')
    async def add_jobs(self, *jobs):
        """Queue jobs, each x,y,dest[,tries,wander,hover,trust] with empty fields for defaults,
        such as 0.2,0.3,bin or 0.2,0.3,bin,1,0.25,0.35,trust. Works during 'clean'."""
        parsed = [self._parse_job(job) for job in jobs]
        self.jobs.extend(parsed)
        return dict(queued=len(self.jobs))

    def _parse_job(self, text):
        fields = text.split(',') + [''] * 7
        if fields[2] not in self.dests:
            raise ValueError(f'unknown destination {fields[2]!r}; set it with cleandest')
        if fields[6] not in ('', 'trust'):
            raise ValueError(f'unknown job flag {fields[6]!r}')
        return dict(x=float(fields[0]), y=float(fields[1]), dest=fields[2],
                    tries=int(fields[3] or DEFAULT_TRIES),
                    wander=float(fields[4] or DEFAULT_WANDER_M),
                    hover=float(fields[5] or DEFAULT_HOVER_M),
                    trust=fields[6] == 'trust')

    @verb('cleanclear')
    async def clear_jobs(self):
        """Empty the queue. The job in progress carries on."""
        self.jobs.clear()
        return dict(queued=0)

    @verb('cleanshow')
    async def show(self):
        """The queue, the job in progress, and the destinations."""
        return dict(jobs=self.jobs, current=self.job, dest=self.dest, dests=self.dests)

    @verb('cleanto')
    async def redirect(self, dest):
        """Change where the item being carried goes, without stopping. Too late once the gripper
        has started down over a small target; the reply says if it was ignored."""
        if dest not in self.dests:
            raise ValueError(f'unknown destination {dest!r}; set it with cleandest')
        if not self.can_redirect:
            return dict(ignored='nothing is being carried that can still be redirected')
        self.dest = dest
        return dict(dest=dest)

    @verb('cleanstats')
    async def stats(self):
        """Totals for the jobs run since the observer started."""
        rows = self.rows
        return dict(jobs=len(rows), held=sum(r['held'] for r in rows),
                    attempts=sum(r['attempts'] for r in rows),
                    rejected_holds=sum(r['rejected_holds'] for r in rows),
                    **{k: round(sum(r[k] for r in rows), 1)
                       for k in ('total_s', 'fly_s', 'grasp_s', 'carry_s', 'cal_s')},
                    log=str(self.stats_path) if self.stats_path else None)

    def _record(self, row):
        row = {k: round(v, 2) if isinstance(v, float) else v for k, v in row.items()}
        self.rows.append(row)
        if self.stats_path is None:
            self.stats_path = self.output_dir('floor_clean') / f'{time.strftime("%Y%m%d-%H%M%S")}.jsonl'
        with open(self.stats_path, 'a') as f:
            f.write(json.dumps(row) + '\n')

    def _next_job(self):
        """The job with the least travel from here, through the item, to its container."""
        here = self.ob.gripper_position()[:2]
        def travel(job):
            item = (job['x'], job['y'])
            return math.dist(here, item) + math.dist(item, self.dests[job['dest']]['xyz'][:2])
        job = min(self.jobs, key=travel)
        self.jobs.remove(job)
        return job

    async def _fly(self, xyz, timeout):
        # A flight in progress is re-aimed rather than restarted, and outlives the timeout.
        goal = np.array(xyz, dtype=float) + self.ob.pole_offset()
        return await self.ob.seek_goal(goal, auto_altitude=False, timeout=timeout)

    async def _calibrate_if_older_than(self, max_age):
        if time.time() - self.last_cal < max_age:
            return 0.0
        started = time.time()
        await self.ob.clear_goal()
        await self.ob.half_auto_calibration()
        self.last_cal = time.time()
        return self.last_cal - started

    async def _grasp_job(self, job, row):
        """True once something is held."""
        for _ in range(job['tries']):
            started = time.time()
            await self._fly([job['x'], job['y'], job['hover']], SEEK_TIMEOUT_S)
            arrived = time.time()
            held = await self.ob.grasp(max_wander_m=job['wander'])
            row['fly_s'] += arrived - started
            row['grasp_s'] += time.time() - arrived
            row['attempts'] += 1
            angle, pad = self.ob.finger_angle(), self.ob.finger_pad_voltage() or 0
            row['finger'], row['pad'] = round(angle, 1), round(pad, 3)
            if held and not job['trust'] and angle > EMPTY_FINGER_ANGLE and pad < EMPTY_PAD_VOLTAGE:
                row['rejected_holds'] += 1
                await self.ob.set_finger_angle(OPEN_BETWEEN_TRIES_ANGLE)
                held = False
            if held:
                row['grasped_at'] = self.ob.gripper_position().round(2).tolist()
                return True
        return False

    async def _deliver(self, row):
        """Carry to self.dest, which cleanto can change on the way, and let go."""
        started = time.time()
        while True:
            dest = self.dest
            d = self.dests[dest]
            if 'wrist' in d:
                await self.ob.set_wrist_angle(d['wrist'])
            p = self.ob.gripper_position()
            if 'radius' in d:
                await self._fly(d['xyz'], REAIM_S)
                inside = math.dist(p[:2], d['xyz'][:2]) < d['radius']
                if inside and p[2] > d['xyz'][2] - RADIUS_DROP_HEIGHT_MARGIN_M:
                    self.can_redirect = False
                    await self.ob.set_finger_angle(OPEN_ANGLE)
                    await asyncio.sleep(RELEASE_ON_THE_FLY_S)
                    break
            else:
                high = [d['xyz'][0], d['xyz'][1], max(CARRY_HEIGHT_M, d['xyz'][2])]
                await self._fly(high, REAIM_S)
                if self.dest != dest or math.dist(p, high) > OVER_CONTAINER_TOL_M:
                    continue
                self.can_redirect = False
                row['cal_s'] += await self._calibrate_if_older_than(CAL_MAX_AGE_BEFORE_DROP_S)
                await self._fly(high, SEEK_TO_DROP_TIMEOUT_S)
                await self._fly(d['xyz'], SEEK_TO_DROP_TIMEOUT_S)
                await asyncio.sleep(SETTLE_S)
                await self.ob.set_finger_angle(OPEN_ANGLE)
                await asyncio.sleep(RELEASE_S)
                await self._fly(high, REAIM_S)
                break
        row['dest'] = dest
        row['carry_s'] = time.time() - started - row['cal_s']

    @verb('clean', motion=True)
    async def clean(self):
        """Work through the queue until it has stayed empty a while. Reports each grasp as a
        log event; jobs can be added and items redirected while this runs."""
        done = 0
        idle_since = None
        try:
            async with self.ob.prefer_swing_cancellation():
                while True:
                    if not self.jobs:
                        await self.ob.clear_goal()
                        idle_since = idle_since or time.time()
                        if time.time() - idle_since > IDLE_LIMIT_S:
                            return dict(delivered=done)
                        await asyncio.sleep(0.2)
                        continue
                    idle_since = None
                    self.job = self._next_job()
                    self.dest = self.job['dest']
                    started = time.time()
                    row = dict(job=self.job, fly_s=0.0, grasp_s=0.0, carry_s=0.0, cal_s=0.0,
                               attempts=0, rejected_holds=0, held=False)
                    row['cal_s'] += await self._calibrate_if_older_than(CAL_MAX_AGE_S)
                    row['held'] = await self._grasp_job(self.job, row)
                    if row['held']:
                        self.can_redirect = True
                        logger.info('HELD at %s bound for %s finger %s pad %s',
                                    row['grasped_at'], self.dest, row['finger'], row['pad'])
                        await self._deliver(row)
                        done += 1
                    else:
                        logger.info('MISS %s,%s after %d attempts',
                                    self.job['x'], self.job['y'], row['attempts'])
                    row['total_s'] = time.time() - started
                    self._record(row)
        finally:
            self.job = None
            self.can_redirect = False
            await self.ob.clear_goal()
            self.ob.slow_stop_all_spools()
