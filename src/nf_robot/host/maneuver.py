"""The interface a maneuver is written against.

A maneuver is one high-level behavior of the robot - parking, say - kept in its own module
and registered with AsyncObserver.add_maneuver. It drives the robot only through the
public methods of the observer (every name without a leading underscore), and declares how
it is reached with the decorators below:

    class Parking(Maneuver):
        name = 'parking'
        config_field = 'park_data'

        @command(control.Command.PARK, motion=True)
        async def park(self): ...

        @startup_step('park')
        async def park_if_recorded(self): ...

This module never imports the observer, so a maneuver module can import it freely.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from enum import Enum
from functools import wraps
from pathlib import Path
from typing import NamedTuple

import numpy as np

from nf_robot.generated.nf import common, telemetry

logger = logging.getLogger(__name__)

# The name the drop position is saved under. Not a tag: nothing ever sees it, it is only ever
# what was recorded there, where the others are re-observed whenever a camera catches the tag.
DROP_POSITION_NAME = "drop_position"
# The drop point model (ml/placer/model.md) predicts where the item now being picked up
# belongs, and its answer is saved under this name like any other place to fly to.
PREDICTED_DROP_NAME = "predicted_drop"
# The named position each route point flies to; see AsyncObserver.route_point_position.
ROUTE_POINT_TAG_NAMES = {
    common.RoutePoint.HAMPER: "hamper",
    common.RoutePoint.TOYBOX: "toys",
    common.RoutePoint.TRASH: "trash",
    common.RoutePoint.GAMEPAD: "gamepad",
    common.RoutePoint.DROP_POSITION: DROP_POSITION_NAME,
    common.RoutePoint.PREDICTED_DROP: PREDICTED_DROP_NAME,
}

# Velocity source keys a maneuver commands moves under. Inbound moves carry a free-form
# source_key (relay users send the integer id of their account), so these prefixes are
# refused on inbound moves: nobody driving remotely can land on a maneuver's key.
MANEUVER_KEY_PREFIX = 'maneuver:'
RESERVED_KEY_PREFIXES = (MANEUVER_KEY_PREFIX, 'ob:')


class OverTension(Enum):
    """What happens to the running motion task when a line goes over the safe tension.

    passive_safety sheds the tension by cycling torque whatever this says; this only
    decides the fate of the task that was running when it happened.
    """
    ABORT = 'abort'     # cancel it, with abort_reason 'tension'
    NOTIFY = 'notify'   # leave it running and call its maneuver's on_over_tension
    IGNORE = 'ignore'   # leave it running


class ItemImage(NamedTuple):
    """A gripper camera frame of whatever was under it at grasping distance."""
    image_rgb: np.ndarray
    timestamp: float            # capture time
    laser_range: float          # metres from the camera to the item
    gantry_position: np.ndarray


@dataclass(frozen=True)
class SafetyPolicy:
    on_over_tension: OverTension = OverTension.ABORT
    # A task that turns sightings of the gantry marker into stored results is aborted on a
    # marker fault rather than left to fit them.
    needs_gantry_marker: bool = False


def prefer_swing_cancellation(func):
    """Run a maneuver's coroutine method under the observer's prefer_swing_cancellation, for
    the long tasks that want it from their first move to their last. Keeps __name__, which
    the motion task is named by."""
    @wraps(func)
    async def wrapper(self, *args, **kwargs):
        async with self.ob.prefer_swing_cancellation():
            return await func(self, *args, **kwargs)
    return wrapper


def _entry(kind, key, motion=False, safety=None):
    def mark(fn):
        entries = fn.__dict__.setdefault('_maneuver_entries', [])
        entries.append((kind, key, motion, safety))
        return fn
    return mark


def command(cmd, motion=False, safety=None):
    """Handle a control.Command. motion=True runs the handler as the motion task."""
    return _entry('command', cmd, motion, safety)


def control_item(field):
    """Handle a ControlItem oneof field, by its proto field name. The handler gets the item."""
    return _entry('control_item', field)


def verb(word, motion=False, safety=None):
    """Handle a Debug action whose first word is word. The handler gets the rest as strings."""
    return _entry('verb', word, motion, safety)


def startup_step(name, safety=None):
    """Offer this method as a step AsyncObserver.set_startup_sequence can list by name.

    Steps decide for themselves whether they apply, since a sequence is written once and
    run on every start.
    """
    return _entry('startup_step', name, True, safety)


class Maneuver:
    """Base class for a maneuver. Override only the hooks you need."""

    name: str = None
    title: str = None           # for progress bars; defaults to name
    config_field: str = None    # the StringmanPilotConfig field this maneuver owns
    safety = SafetyPolicy()     # for motion handlers that do not give their own

    def __init__(self, ob):
        self.ob = ob
        # why the last run of this maneuver's motion task was cancelled by a safety
        # monitor: 'tension', 'marker', or None for anything else, a stop included
        self.abort_reason = None
        self._spawned = set()

    # -- hooks ---------------------------------------------------------------

    async def start(self):
        """Called once the observer is serving telemetry."""

    async def stop(self):
        """Called during shutdown, after stop_all. Spawned tasks are already cancelled."""

    def on_component_connected(self, kind, anchor_num=None):
        """A component's websocket came up. kind is 'gripper' or 'anchor'."""

    def on_component_disconnected(self, kind, anchor_num=None):
        """A component's websocket that was up went down."""

    def send_setup_telemetry(self):
        """Replay this maneuver's state to a UI that just connected."""

    def on_stop_all(self):
        """Called when everything is being stopped."""

    def on_over_tension(self, tensions):
        """Called while this maneuver's task runs under OverTension.NOTIFY and a line is over."""

    # -- helpers -------------------------------------------------------------

    @property
    def data(self):
        """This maneuver's own section of the robot config."""
        return getattr(self.ob.config, self.config_field)

    @data.setter
    def data(self, value):
        setattr(self.ob.config, self.config_field, value)

    def save_data(self):
        self.ob.save_config()

    @property
    def settings(self):
        """What this maneuver last stored with save_settings, or '' if nothing. For maneuvers
        from outside nf_robot, which have no typed config field of their own."""
        return self.ob.config.maneuver_settings.get(self.name, '')

    def save_settings(self, value):
        self.ob.config.maneuver_settings[self.name] = value
        self.ob.save_config()

    @property
    def settings_json(self):
        """settings decoded as JSON, or None if nothing has been stored."""
        return json.loads(self.settings) if self.settings else None

    def save_settings_json(self, value):
        self.save_settings(json.dumps(value))

    def override_safety(self, **changes):
        """A context manager changing the running motion task's SafetyPolicy for one phase:

            with self.override_safety(on_over_tension=OverTension.IGNORE):
                ...
        """
        return self.ob.override_safety(**changes)

    def velocity_key(self, sub=''):
        """A move_direction_speed source key of this maneuver's own."""
        return f'{MANEUVER_KEY_PREFIX}{self.name}' + (f':{sub}' if sub else '')

    def progress(self, percent, action):
        self.ob.send_ui(operation_progress=telemetry.OperationProgress(
            percent_complete=percent, name=self.title or self.name, current_action=action))

    def finish(self, action=''):
        self.progress(100.0, action)

    def notify(self, message):
        self.ob.send_ui(pop_message=telemetry.Popup(message=message))

    async def ask(self, message, buttons=None, timeout=None):
        """Index of the button pressed, or None if nobody answered in time."""
        return await self.ob.send_popup_and_await_answer(message, buttons=buttons, timeout=timeout)

    def spawn(self, coro, name=None):
        """Run coro in the background until it ends or this maneuver is stopped."""
        task = asyncio.create_task(coro, name=name)
        self._spawned.add(task)
        task.add_done_callback(self._spawned.discard)
        return task

    async def run_in_thread(self, fn, *args):
        return await asyncio.to_thread(fn, *args)

    def output_dir(self, sub):
        """A directory for files this maneuver writes, created if need be."""
        path = Path(self.ob.output_root) / sub
        path.mkdir(parents=True, exist_ok=True)
        return path

    async def _cancel_spawned(self):
        tasks = list(self._spawned)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


class TiltWatch:
    """Trips when the pole leans past tilt_deg for longer than confirm_s.

    A pole a pole's length below the gantry is the first thing to touch anything the gantry
    is steered into, and a strike leans it. Confirming over a window keeps a swing from
    reading as a strike, and a stale reading - the gripper gone quiet - never trips it
    rather than tripping it forever.
    """

    def __init__(self, ob, tilt_deg=8.0, confirm_s=0.3, max_age_s=1.0):
        self.ob = ob
        self.tilt_deg = tilt_deg
        self.confirm_s = confirm_s
        self.max_age_s = max_age_s
        self.leaning_since = None
        # steepest lean seen so far, for tuning the threshold from a run that went well
        self.worst_tilt = 0.0

    def check(self):
        """The reason to believe the pole has hit something, or None. Call it steadily: the
        confirmation window is counted in calls."""
        tilt = self.ob.pole_tilt(max_age=self.max_age_s)
        if tilt is None:
            return None
        self.worst_tilt = max(self.worst_tilt, tilt)
        if tilt <= self.tilt_deg:
            self.leaning_since = None
            return None
        if self.leaning_since is None:
            self.leaning_since = time.time()
            return None
        if time.time() - self.leaning_since > self.confirm_s:
            return f'the pole is leaning {tilt:.0f} degrees off vertical'
        return None
