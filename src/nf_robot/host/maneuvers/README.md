# Maneuvers

A maneuver is one high-level behavior of a Stringman robot, such as parking, pick and place or anything you write yourself, kept in its own class and registered with the `AsyncObserver`. The observer owns the connections, telemetry, config, calibration and motion primitives. A maneuver drives the robot only through the observer's public methods.

A maneuver is *not* a program that drives the robot through it's telemetry link. The main UI and lerobot/strigman.py are examples of code that communicates that way.

## Getting started

```sh
pip install "nf_robot[host]"
```

```python
# hello_maneuver.py
import asyncio
import numpy as np
from nf_robot.host.observer import AsyncObserver
from nf_robot.host.maneuver import Maneuver, verb

class Hello(Maneuver):
    name = 'hello'

    @verb('hello', motion=True)        # type "hello" in the UI's debug box
    async def wave(self):
        try:
            self.notify(f'Hello from {self.ob.gantry_position().round(2)}')
            await self.ob.nudge_gantry(np.array([0.0, 0.0, 0.05]))    # up 5 cm
            await self.ob.nudge_gantry(np.array([0.0, 0.0, -0.05]))   # and back
        finally:
            self.ob.slow_stop_all_spools()

async def run():
    ob = AsyncObserver(terminate_with_ui=False, config_path='configuration.json')
    ob.add_maneuver(Hello)
    await ob.main()                    # runs until shutdown

asyncio.run(run())
```

Run it the way you would run `stringman-headless`, open the UI it prints, and type `hello` into the debug box. The stop button cancels it like any other motion.

## A fuller example: sorting cans as the startup task

Cans arrive at one fixed pickup spot. The robot grasps each one and classifies it from the gripper camera's view of it just before the fingers closed. It then drops the can at the bin for that class. After a few empty grasps in a row it stops, and the startup sequence parks.

```python
# can_sorter.py
import asyncio, time
import numpy as np
import torch
from nf_robot.host.observer import AsyncObserver
from nf_robot.host.maneuver import Maneuver, verb, startup_step, prefer_swing_cancellation
from nf_robot.generated.nf import telemetry

PICKUP = 'can_pickup'   # named positions are stored in the robot config
BINS = {'aluminum': 'bin_aluminum', 'steel': 'bin_steel', 'reject': 'bin_reject'}
EMPTY_GRASPS_TO_STOP = 3

class CanSorter(Maneuver):
    name = 'sort_cans'
    title = 'Sorting cans'

    def __init__(self, ob, checkpoint):
        super().__init__(ob)
        self.checkpoint = checkpoint
        self.model = None

    # -- controls, typed into the debug box --
    @verb('canpickup')
    async def record_pickup(self):
        """Hover over the pickup spot and type 'canpickup'."""
        self.ob.set_named_position(PICKUP, self.ob.gantry_position() - self.hover_over_target())

    @verb('canbin')
    async def record_bin(self, cls):
        """Hover over a bin and type 'canbin steel'."""
        self.ob.set_named_position(BINS[cls], self.ob.gantry_position() - self.hover_over_dropoff())

    # -- replayed to every UI that connects --
    def send_setup_telemetry(self):
        for name in (PICKUP, *BINS.values()):
            if (pos := self.ob.named_position(name)) is not None:
                self.ob.send_ui(named_position=telemetry.NamedObjectPosition(name=name, position=pos))

    # -- the work: a startup step, and also runnable by hand --
    @startup_step('sort_cans')
    @verb('sortcans', motion=True)
    @prefer_swing_cancellation
    async def sort(self):
        pickup = self.ob.named_position(PICKUP)
        if pickup is None or any(self.ob.named_position(b) is None for b in BINS.values()):
            self.notify('Record the pickup spot and every bin first (canpickup, canbin <class>)')
            return
        await self.ensure_model()
        tally = self.settings_json or {}      # lifetime counts, kept in the robot config
        empty = count = 0
        try:
            while empty < EMPTY_GRASPS_TO_STOP:
                await self.ob.seek_goal(pickup + self.hover_over_target())
                started = time.time()
                if not await self.ob.grasp():
                    empty += 1
                    continue
                empty = 0
                seen = self.ob.last_clear_item_image()     # the can just before the fingers closed
                if seen is None or seen.timestamp < started:
                    cls = 'reject'
                else:
                    cls = await self.run_in_thread(self.classify, seen.image_rgb)
                await self.ob.seek_goal(self.ob.named_position(BINS[cls]) + self.hover_over_dropoff())
                await self.ob.set_finger_angle(self.ob.config.pick_and_place.relaxed_open)
                count += 1
                tally[cls] = tally.get(cls, 0) + 1
                self.progress(0, f'{count} sorted, last was {cls}')
            self.save_settings_json(tally)
            self.notify(f'Pickup is empty. Sorted {count} cans.')
        finally:
            self.finish()
            self.ob.slow_stop_all_spools()
            await self.ob.clear_goal()

    async def ensure_model(self):
        if self.model is None:
            self.progress(0, 'Loading can classifier')
            device = await self.run_in_thread(self.ob.torch_device)
            self.model = await self.run_in_thread(torch.jit.load, self.checkpoint, device)

    def classify(self, image_rgb):
        x = torch.from_numpy(image_rgb).permute(2, 0, 1).float().div(255).unsqueeze(0)
        return list(BINS)[int(self.model(x.to(self.ob.torch_device())).argmax())]

    def hover_over_target(self):  return np.array(self.ob.config.pick_and_place.gantry_height_over_target)
    def hover_over_dropoff(self): return np.array(self.ob.config.pick_and_place.gantry_height_over_dropoff)

async def run():
    ob = AsyncObserver(terminate_with_ui=False, config_path='configuration.json', auto_start=True)
    ob.add_maneuver(CanSorter, checkpoint='can_classifier.pt')
    ob.set_startup_sequence(['unpark', 'sort_cans', 'park'])
    await ob.main()

asyncio.run(run())
```

Run it from the directory holding the robot's `configuration.json`, with the classifier checkpoint beside it:

```sh
python can_sorter.py
```

With `auto_start=True`, the startup sequence runs as soon as every component connects. Before that first run, record the pickup spot and the bins from the debug box.

Things worth copying from it:
- **Wrap every motion method in `try`/`finally`** and bring the robot to rest in the `finally`. The stop button, a new command, or a safety monitor can cancel it at any `await`.
- **Keep slow work off the event loop.** Model loading and inference go through `run_in_thread`; on the loop they would stall telemetry and every other motion.
- **Named positions** persist in the robot config and appear in the UI, so they are the natural place to keep spots in the room.

## Rules

- **Stay on the public surface.** Call only the observer methods listed below, the ones without a leading underscore. Do not touch `ob.pe`, `ob.datastore`, `ob.gripper_client` or `ob.anchors`, and never write calibration or geometry fields of `ob.config`. Those belong to the observer.
- **One motion task at a time.** Starting one cancels whatever was running. Inside a motion task, call other motion coroutines directly with `await`, never through `invoke_motion_task`, because that would cancel the task you are in.

## API reference

### Registering and running (`AsyncObserver`)

| Method | |
|---|---|
| `AsyncObserver(terminate_with_ui, config_path, telemetry_env=None, run_ortho=True, auto_start=False, local_models=False, port=4245, debug=False, bind_address="127.0.0.1", rec_diagnostics=False, serve_ui=True, ui_port=8090, diamond_size=DIAMOND_SIZE, lerobot_grasp=False)` | Construct inside a running event loop. `config_path=None` uses an in-memory config that is never saved. |
| `add_maneuver(maneuver_class, **options)` | Constructs `maneuver_class(ob, **options)`, routes everything it declared, and returns the instance. Raises `ValueError` on a duplicate name, command, control field, verb or startup step. Call before `main()`. |
| `maneuver(name)` | A registered maneuver, built-in or yours. |
| `maneuvers` | Dict of every registered maneuver by name. |
| `set_startup_sequence(names)` | Startup step names an `auto_start` robot runs in order once every component connects. The default is `['unpark', 'pick_and_place', 'park']`. Names are checked when `main()` starts. |
| `await main()` | Runs until shutdown, then closes everything. |

Built-in maneuvers are `parking`, `drop_point`, `lerobot`, `pick_and_place`, `plates`, `diagnostics`, `ferry` and `cluster_sort`. Built-in startup steps are `unpark`, `park` and `pick_and_place`.

### `nf_robot.host.maneuver`

**`class Maneuver`** is the base class. Override only what you need.

| Class attribute | |
|---|---|
| `name` | Required and unique. It is also the key for `settings` and for `velocity_key()`. |
| `title` | Shown in progress bars; defaults to `name`. |
| `config_field` | For maneuvers inside nf_robot: the typed `StringmanPilotConfig` field this maneuver owns. |
| `safety` | The default `SafetyPolicy` for this maneuver's motion handlers. |

| Hook | Called |
|---|---|
| `__init__(self, ob, **options)` | By `add_maneuver`. Call `super().__init__(ob)`. |
| `async start()` | Once the observer is serving telemetry. |
| `async stop()` | During shutdown, after `stop_all`. Spawned tasks are already cancelled. |
| `send_setup_telemetry()` | Whenever a UI connects. Replay whatever it should see. |
| `on_stop_all()` | When everything is being stopped. |
| `on_over_tension(tensions)` | While your motion task runs under `OverTension.NOTIFY` and a line goes over the safe limit. `tensions` is a (4,) array in newtons. |
| `on_component_connected(kind, anchor_num=None)` | A component's websocket came up. `kind` is `'gripper'` or `'anchor'`. |
| `on_component_disconnected(kind, anchor_num=None)` | A component's websocket that was up went down. |

| Helper | |
|---|---|
| `self.ob` | The observer. |
| `settings` / `save_settings(value)` | An arbitrary string of your own, persisted in the robot config (`maneuver_settings[name]`). `settings` is `''` until something is saved. |
| `settings_json` / `save_settings_json(value)` | The same, stored as JSON. `settings_json` is `None` until something is saved. |
| `data` / `save_data()` | The typed `config_field` section, for maneuvers inside nf_robot. |
| `velocity_key(sub='')` | `'maneuver:<name>[:<sub>]'`, a `move_direction_speed` source key of your own. |
| `progress(percent, action)` / `finish(action='')` | Operation progress in the UI under `title`. `finish` sends 100%. |
| `notify(message)` | A popup. |
| `await ask(message, buttons=None, timeout=None)` | A popup with buttons. Returns the index of the button pressed, or `None` on timeout. |
| `spawn(coro, name=None)` | Runs a background task that is cancelled when the observer shuts down. |
| `await run_in_thread(fn, *args)` | Runs `fn(*args)` in a worker thread. |
| `output_dir(sub)` | A directory for files you write, created if need be. |
| `override_safety(**changes)` | A context manager that changes the running motion task's `SafetyPolicy` until the block exits. |
| `abort_reason` | After your motion task was cancelled by a safety monitor: `'tension'` or `'marker'`. `None` for anything else, a stop included. |

**Decorators.** They stack, so one method can answer several ways.

| Decorator | |
|---|---|
| `@command(cmd, motion=False, safety=None)` | Handle a `control.Command` the observer does not handle itself. |
| `@control_item(field)` | Handle a `ControlItem` oneof field, by proto field name. The handler gets the payload. Handlers are awaited in the dispatch loop, so `spawn` anything slow. |
| `@verb(word, motion=False, safety=None)` | Handle a debug-box action whose first word is `word`. The handler gets the remaining words as string arguments. |
| `@startup_step(name, safety=None)` | Offer the method as a step `set_startup_sequence` can list. Steps run inside the startup motion task and decide for themselves whether they apply. |
| `@prefer_swing_cancellation` | Run the method with swing cancellation on, if this robot has it verified. |

`motion=True` runs the handler as the motion task, owned by your maneuver and held to `safety` (or the class's `safety`).

**Safety**

```python
class OverTension(Enum):
    ABORT   # cancel the motion task, abort_reason = 'tension' (default)
    NOTIFY  # keep running and call on_over_tension
    IGNORE  # keep running

@dataclass(frozen=True)
class SafetyPolicy:
    on_over_tension: OverTension = OverTension.ABORT
    needs_gantry_marker: bool = False   # abort on a gantry marker fault, abort_reason = 'marker'
```

Whatever the policy, the observer always sheds an over-tension by briefly cycling motor torque. The policy only decides what happens to your task.

**Other**

| Name | |
|---|---|
| `TiltWatch(ob, tilt_deg=8.0, confirm_s=0.3, max_age_s=1.0)` | Call `check()` steadily. It returns a reason once the pole has leaned past `tilt_deg` for `confirm_s`, else `None`. `worst_tilt` is the steepest lean seen. |
| `ItemImage` | Named tuple: `image_rgb`, `timestamp`, `laser_range`, `gantry_position`. |
| `ROUTE_POINT_TAG_NAMES` | Maps each `common.RoutePoint` to the named position it flies to. |
| `DROP_POSITION_NAME`, `PREDICTED_DROP_NAME` | Named positions for the recorded drop position and the drop point model's prediction. |

### Observer: position and state

Positions are room-frame numpy arrays in metres, with z up and the floor at z=0. Every getter returns a copy.

| Method | |
|---|---|
| `gantry_position()` | Where the position estimate puts the gantry. |
| `visual_gantry_position()` | The anchor cameras' running average of the gantry marker. |
| `gripper_position()` | Where the estimate puts the gripper. |
| `pole_offset()` | The offset from the gantry down to the gripper. Add it to a gripper goal to get a gantry goal. |
| `anchor_points()` | (4, 3) points the lines leave from. |
| `inside_work_area_2d(point)` | Whether a point lies inside the floor area the robot can reach. |
| `line_tensions()` | (4,) newtons, or `None` before any report. |
| `max_safe_tension` | The tension (newtons) at which the observer sheds torque. This is a property. |
| `await measure_free_tension(samples=5, interval_s=0.1)` | (4,) median tension while hanging free. |
| `fresh_gantry_sightings(window_s, after=None)` | Anchor camera sightings of the gantry marker from the last `window_s` seconds. |
| `await settle_visual_estimate(tol_m=0.05, window_s=1.0, min_sightings=3, timeout=15.0)` | Wait until the visual estimate agrees with live sightings. Returns `True`, or `False` on timeout. |
| `is_holding()` | Whether the fingers are closed on something. |
| `named_position(name)` | A named place's last known position, or `None`. |
| `set_named_position(name, position, save=True)` | Remember a named place and show it in the UI. |
| `route()` | `(source, destination)` as `common.RoutePoint`s, the UI's From: and To:. |
| `set_route(source=None, destination=None)` | Change either end; it is saved and shown in the UI. |
| `route_point_position(route_point)` | A route point's floor position, or `None`. |

### Observer: gripper and cameras

| Method | |
|---|---|
| `gripper_connected()` | |
| `laser_range()` | The rangefinder's distance, in metres, to whatever is under the gripper, or `None` if stale. |
| `wrist_angle()` | Degrees, 0 to 1080. |
| `finger_angle()` | Degrees, -90 open to 90 closed. |
| `finger_pad_voltage()` | The finger pressure pad reading. |
| `reset_finger_pressure_rising()` / `finger_pressure_rose()` | Whether the finger pressure has risen since the reset. |
| `pole_tilt(max_age=1.0)` | Degrees off vertical, or `None` if stale. |
| `wrist_angle_for_heading(heading)` | The wrist angle pointing the gripper's nose along a room heading, in radians. |
| `gripper_camera_intrinsics()` | 3x3 intrinsic matrix. |
| `gripper_camera_to_room(vec)` | Rotate a vector from the camera's optical frame (x right, y down, z forward) into the room frame. |
| `await gripper_frame(after=None, timeout=3.0)` | An RGB frame captured after `after` (now, by default), or `None`. `after=0` takes the newest frame. |
| `await gripper_capture(after=None, timeout=3.0, expect_size=None)` | `(timestamp, RGB frame)`, optionally only at a given `(width, height)`. |
| `last_clear_item_image()` | The newest `ItemImage` taken while the laser read 12 to 25 cm: an item under the gripper, in view, before the fingers close. |
| `latest_ortho()` | The newest orthographic floor view (RGB), or `None`. |
| `ortho_enabled()` | Whether the floor view is being rendered. |
| `anchor_pixel_to_floor(anchor_num, norm_xy)` | Where a normalized point in an anchor camera image lands on the floor. |
| `route_tag_samples(name, since)` | The gripper camera's `(timestamp, pose)` sightings of a floor tag. |
| `gantry_minus_card(pose, timestamp)` | Room-frame gantry offset from a tag, from one sighting. |
| `await use_gripper_capture_stream()` | Switch the gripper camera to full resolution for the rest of the session. |
| `start_gripper_recording(path)` / `gripper_recorded_packets()` / `stop_gripper_recording()` | Record the camera's compressed stream. `stop` returns `(packets, stream_start_ts)`. |

### Observer: motion

Coroutines marked *motion* move the robot and may be cancelled at any `await`.

| Method | |
|---|---|
| `await seek_goal(goal_pos, head_turn=False, auto_altitude=True, timeout=None)` | *Motion.* Fly the gantry to `goal_pos`. Returns `True` on arrival. A call while a flight is under way re-aims it. With `timeout`, it returns `False` after that many seconds and leaves the flight going. |
| `await clear_goal()` | End the flight in progress. |
| `await nudge_gantry(delta, speed=0.12, max_step=0.35)` | *Motion.* A small eased move by `delta` metres, then stop. |
| `await move_direction_speed(uvec, speed=None, starting_pos=None, downward_bias=-0.04, key=...)` | Command a velocity under a source key; pass `key=self.velocity_key()`. Velocities from all keys are summed, and a key expires 2 s after its last command. Zero your key when done. |
| `slow_stop_all_spools()` | Bring every spool to rest. |
| `await settle_wrist(target, tol=2.0, timeout=6.0)` | Move the wrist and wait for it to arrive. |
| `await settle_wrist_to_heading(angle_deg, tol=2.0, peak_dps=90.0)` | The same, choosing the nearest equivalent angle and easing in and out. |
| `await settle_fingers(target, tol=2.0, timeout=6.0)` | Move the fingers and wait for them to arrive. |
| `await set_finger_angle(angle)` / `await set_wrist_angle(angle)` / `await set_wrist_speed(dps)` | Command without waiting. The wrist speed has to be repeated to keep turning. |
| `await trim_altitude_to_range(target_range_m, tol_m=0.02, max_steps=4, ceiling_z=None, max_travel_m=None)` | *Motion.* Adjust height until the laser reads `target_range_m`. Returns the final reading. |
| `await tension_and_wait()` | Tighten all lines and wait until they read tight. |
| `await grasp()` | *Motion.* Grasp whatever is under the gripper. Returns `True` if it is held. |
| `await servo_center()` | *Motion.* Steer sideways over the object the visual servoing model sees and turn the wrist to it, never descending. Runs until cancelled. Its velocity is summed with your own keys, so you can set the height alongside it. |
| `servo_center_offset(max_age=0.5)` | Metres between the jaws and the object `servo_center` is steering for, or `None` if no object is confidently seen or it is not running. |
| `prefer_swing_cancellation()` | An async context manager: swing cancellation on for the block, if verified. |
| `set_swing_cancellation(enabled)` | Returns whether it was on. |
| `await half_auto_calibration()` | *Motion.* Re-tension and re-reference the lines from the cameras' view of the gantry. |
| `await invoke_motion_task(coro, owner=None, safety=None)` | Start `coro` as the motion task, cancelling the current one. Not from inside a motion task. |
| `await stop_all()` | Stop everything. Not from inside a motion task; use `slow_stop_all_spools` there. |
| `override_safety(**changes)` | See `Maneuver.override_safety`. |
| `speed_limit()` | Maximum gantry speed, in m/s. |
| `feature_supported(feature_key)` | Whether every connected component's firmware supports a feature. |

### Observer: UI, config and compute

| Name | |
|---|---|
| `send_ui(**item)` | Queue one telemetry item (one `TelemetryItem` field) for every connected UI. |
| `await flush_tele_buffer()` | Send queued telemetry now instead of on the next estimator tick. |
| `await send_popup_and_await_answer(message, buttons=None, timeout=None)` | See `Maneuver.ask`. |
| `config` | The robot config (`StringmanPilotConfig`). Read freely; write only your own settings. |
| `save_config()` | Write the config out. |
| `torch_device()` | The torch device all models on this host share. The first call imports torch, so make it from `run_in_thread`. |
| `local_models` | Whether `--local_models` asked for models from `models/` instead of the hub. |
| `robot_id()` | This robot's id with its control plane, or `None`. |
