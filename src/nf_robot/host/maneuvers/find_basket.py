"""Find basket: turning a rough guess at where a basket is into a fix good enough to drop on."""

import asyncio
import logging
import time

import numpy as np

from nf_robot.common.util import fromnp, tonp
from nf_robot.generated.nf import telemetry
from nf_robot.host.maneuver import Maneuver, TiltWatch, prefer_swing_cancellation, verb

logger = logging.getLogger(__name__)

# Where the result is shown in the UI. Not saved: it is one run's answer.
BASKET_FIX_NAME = 'basket_fix'
# Looks at the basket before giving up on converging.
MAX_LOOKS = 6
# Frames per look; the move is the median of their predictions.
FRAMES_PER_LOOK = 5
# How many of those have to see a basket at all, by SCORE_MIN, for the look to count.
MIN_GOOD_FRAMES = 3
# The cell softmax's peak probability below which a frame is taken to show no basket. The
# softmax is over 900 cells, so even a confident answer spreads over a few of them.
SCORE_MIN = 0.05
# (metres) the frames of one look disagreeing by more than this, median absolute deviation
# of the lateral move, means the model is not looking at one basket.
SPREAD_MAX_M = 0.1
# (metres) close enough to call it centred.
LATERAL_TOL_M = 0.05
VERTICAL_TOL_M = 0.07
# Fraction of each predicted move taken, so a wrong first guess from far off does not
# carry the gripper all the way past the basket.
GAIN = 0.8
# (metres) cap on one step, which is about how far off the training data's moves went.
MAX_STEP_M = 1.0
# (metres) how much higher than the usual drop height the search starts over a guess, so a
# guess that is well off still has the basket in view.
START_EXTRA_HEIGHT_M = 0.3
# (metres) never put the gripper lower than this over the floor while searching.
MIN_GRIPPER_Z_M = 0.2
# (seconds) wait after each move for the pole to stop swinging before looking.
SETTLE_S = 1.0
TILT_DEG = 12.0
TILT_CONFIRM_S = 0.3
SEEK_POLL_S = 0.1
# (metres) a fix taken from a coarse position this far from where that place is now seen is
# a fix on where it used to be.
FIX_STALE_M = 0.3


def fix_marker_name(key):
    """The named position a stored fix is shown in the UI under. Display only: fixes are
    kept in this maneuver's settings."""
    return f'{key}_fix'


class FindBasket(Maneuver):
    name = 'find_basket'
    title = 'Find basket'

    def __init__(self, ob):
        super().__init__(ob)
        self.model = None

    def send_setup_telemetry(self):
        for key, entry in self._fixes().items():
            self.ob.send_ui(named_position=telemetry.NamedObjectPosition(
                name=fix_marker_name(key), position=fromnp(np.array(entry['fix']))))

    # -- fixes ----------------------------------------------------------------
    #
    # A fix is the gantry position to drop into a basket from, found by visiting it. Fixes
    # are kept by key - a named place such as 'hamper', or a room grid cell - in this
    # maneuver's settings, so they outlast a restart, each beside the coarse position it was
    # taken from so that a basket which has since moved does not keep its old fix.

    def _fixes(self):
        return (self.settings_json or {}).get('fixes', {})

    def fix_for(self, key, coarse=None):
        """The fix stored under key, or None if there is none or coarse - where that place
        is seen now - has moved FIX_STALE_M from where it was when the fix was taken."""
        entry = self._fixes().get(key)
        if entry is None:
            return None
        if coarse is not None and entry.get('coarse') is not None:
            moved = np.linalg.norm((np.asarray(coarse) - np.asarray(entry['coarse']))[:2])
            if moved > FIX_STALE_M:
                logger.info(f'Fix for {key} is stale: the place has moved {moved:.2f}m since')
                return None
        return np.array(entry['fix'], dtype=float)

    def record_fix(self, key, fix, coarse=None):
        fixes = dict(self._fixes())
        fixes[key] = {'fix': [float(c) for c in fix],
                      'coarse': None if coarse is None else [float(c) for c in coarse]}
        self.save_settings_json({**(self.settings_json or {}), 'fixes': fixes})
        self.ob.send_ui(named_position=telemetry.NamedObjectPosition(
            name=fix_marker_name(key), position=fromnp(np.asarray(fix, dtype=float))))

    def forget_fixes(self):
        for key in self._fixes():
            # a named position with no position is hidden by the UI
            self.ob.send_ui(named_position=telemetry.NamedObjectPosition(name=fix_marker_name(key)))
        self.save_settings_json({**(self.settings_json or {}), 'fixes': {}})

    async def fix_location(self, key, guess, coarse=None, silent=True):
        """Visit guess with find_basket and store what it finds under key. The fix, or None.
        This is a motion task."""
        fix = await self.find_basket(guess, silent=silent)
        if fix is not None:
            self.record_fix(key, fix, coarse)
            logger.info(f'Fix for {key}: drop from {np.round(fix, 3)}')
        return fix

    @verb('findbasket', motion=True)
    @prefer_swing_cancellation
    async def find_basket_verb(self, *where):
        """Debug: find a basket and show the fix as the 'basket_fix' named position.

        'findbasket' alone starts from wherever the gripper is, so move it over the rough
        guess first. 'findbasket X Y' flies to a floor point first, and 'findbasket NAME' to
        a named position such as hamper, storing the result as that place's fix.
        """
        guess = None
        key = None
        if len(where) == 2:
            guess = np.array([float(where[0]), float(where[1]), 0.0])
        elif len(where) == 1:
            key = where[0]
            guess = self.ob.named_position(key)
            if guess is None:
                self.notify(f'No named position called {key}')
                return None
        elif where:
            self.notify("Usage: findbasket, findbasket X Y, or findbasket NAME")
            return None
        try:
            if key is not None:
                fix = await self.fix_location(key, guess, coarse=guess, silent=False)
            else:
                fix = await self.find_basket(guess, silent=False)
        finally:
            self.ob.slow_stop_all_spools()
            await self.ob.clear_goal()
        if fix is not None:
            self.notify(f'Basket found: drop from gantry position {np.round(fix, 3)}')
        return fix

    async def find_basket(self, guess=None, silent=True):
        """Look at a basket from near guess and work out exactly where to drop into it from.

        guess is a rough floor position of the basket, which the gantry flies over at the
        usual drop height plus START_EXTRA_HEIGHT_M first; with None, the search starts from wherever the gripper is.
        Then it looks, moves by most of what the basket centering model says, and looks
        again, until the model says it is centred. Returns the gantry position to drop
        from, or None if no basket was found or it did not settle on one. This is a
        motion task, and leaves the gripper wherever the search ended.

        Silent by default, for other maneuvers that call it as a step of their own: it then
        only logs. silent=False adds progress bars and popups, for the debug command.
        """
        ob = self.ob

        def say(message):
            logger.info(message)
            if not silent:
                self.notify(message)

        def progress(percent, action):
            if not silent:
                self.progress(percent, action)

        def finish(action):
            if not silent:
                self.finish(action)

        if not ob.gripper_connected():
            say('Find basket needs the gripper connected')
            return None
        if not await self.ensure_model(silent):
            return None

        if guess is not None:
            hover = np.array(guess, dtype=float)
            hover[2] = 0.0
            hover += tonp(ob.config.pick_and_place.gantry_height_over_dropoff)
            hover[2] += START_EXTRA_HEIGHT_M
            progress(0, 'Flying to the rough basket position')
            problem = await self._fly(hover)
            if problem:
                say(f'Find basket: {problem} on the way to the rough position')
                return None

        for look in range(MAX_LOOKS):
            progress(100.0 * (look + 1) / (MAX_LOOKS + 1), f'Looking ({look + 1})')
            await asyncio.sleep(SETTLE_S)
            move = await self._look()
            if move is None:
                say('Find basket: no basket in view, or the model could not settle on one')
                finish('No basket found')
                return None
            gantry = ob.gantry_position()
            lateral = float(np.linalg.norm(move[:2]))
            logger.info(f'Find basket look {look + 1}: move {np.round(move, 3)} from '
                        f'{np.round(gantry, 3)}')
            if lateral < LATERAL_TOL_M and abs(move[2]) < VERTICAL_TOL_M:
                fix = gantry + move
                if not silent:
                    ob.set_named_position(BASKET_FIX_NAME, fix, save=False)
                finish(f'Centred after {look + 1} look(s)')
                return fix

            step = move * GAIN
            length = float(np.linalg.norm(step))
            if length > MAX_STEP_M:
                step *= MAX_STEP_M / length
            goal = gantry + step
            if not ob.inside_work_area_2d(goal):
                say('Find basket: the basket looks to be outside the work area')
                finish('Basket out of reach')
                return None
            gripper_z = float((goal - ob.pole_offset())[2])
            if gripper_z < MIN_GRIPPER_Z_M:
                goal[2] += MIN_GRIPPER_Z_M - gripper_z
            problem = await self._fly(goal)
            if problem:
                say(f'Find basket: {problem}')
                finish('Stopped')
                return None

        say(f'Find basket: not centred after {MAX_LOOKS} looks')
        finish('Did not converge')
        return None

    async def _look(self):
        """The median room-frame move to centre the basket over FRAMES_PER_LOOK fresh
        frames, or None if too few saw it or they disagree."""
        from nf_robot.ml.basket.model import predict_frame

        ob = self.ob
        moves = []
        for _ in range(FRAMES_PER_LOOK):
            asked = time.time()
            rgb = await ob.gripper_frame(after=asked)
            if rgb is None:
                continue
            state = {
                'laser_rangefinder': ob.laser_range() or 0.0,
                'finger_angle': ob.finger_angle(),
                'target_force': ob.grip_target_force(),
            }
            prediction = await self.run_in_thread(
                predict_frame, self.model, rgb, state, ob.torch_device(), ob.gripper_spin())
            logger.debug(f'Find basket frame: score {prediction["score"]:.3f} move '
                         f'{np.round(prediction["return_room"], 3)}')
            if prediction['score'] >= SCORE_MIN:
                moves.append(prediction['return_room'])
        if len(moves) < MIN_GOOD_FRAMES:
            logger.info(f'Find basket: only {len(moves)} of {FRAMES_PER_LOOK} frames saw a basket')
            return None
        moves = np.array(moves)
        median = np.median(moves, axis=0)
        spread = float(np.median(np.linalg.norm(moves[:, :2] - median[:2], axis=1)))
        if spread > SPREAD_MAX_M:
            logger.info(f'Find basket: frames disagree by {spread * 100:.0f}cm')
            return None
        return median

    async def _fly(self, goal):
        """Fly the gantry to goal, watching for the pole hitting something. None when it
        arrived, or what went wrong."""
        tilt = TiltWatch(self.ob, TILT_DEG, TILT_CONFIRM_S)
        while not await self.ob.seek_goal(goal, auto_altitude=False, timeout=SEEK_POLL_S):
            caught = tilt.check()
            if caught:
                return caught
        return None

    async def ensure_model(self, silent=True):
        """Load the basket centering model if it is not loaded. True if there is one."""
        if self.model is not None:
            return True

        def load_sync():
            from nf_robot.common.model_revisions import pinned_revision
            from nf_robot.ml.basket.model import BASKET_MODEL_REPOID, load_model

            revision = None if self.ob.local_models else pinned_revision(BASKET_MODEL_REPOID)
            return load_model(self.ob.torch_device(), local_models=self.ob.local_models,
                              revision=revision)

        if not silent:
            self.progress(0, 'Loading the basket centering model')
        try:
            model, checkpoint = await self.run_in_thread(load_sync)
        except Exception as e:
            logger.error(f'Could not load the basket centering model: {e!r}')
            if not silent:
                self.notify(f'Could not load the basket centering model: {e}')
                self.finish('No model')
            return False
        logger.info(f'Basket centering model ready: epoch {checkpoint.get("epoch")}, '
                    f'metrics {checkpoint.get("metrics")}')
        self.model = model
        return True
