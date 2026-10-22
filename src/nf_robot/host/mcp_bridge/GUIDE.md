# Driving Stringman through MCP

You are connected to the observer: the process that runs a Stringman robot. A Stringman is a four-line cable driven parallel robot. One pull-point is in each corner of a room, two corners are powered anchors featuring cameras in opposite corners. the other two corners contain passive eyelets that redirect lines. Where the lines meet is an apriltag marker, referred to as the gantry or marker tag. A pole hangs from the gantry, and a gripper on the end of the pole has a palm camera, a wrist that rotates around the pole, two symmetric fingers, an IMU, and a downward laser rangefinder. Anchor camera views are also fused into an image called the "ortho" image which is a projection at floor level of both views where 5 meters is 512 pixels.

Anchors and gripper are thin servers running on raspberry pi's on the wifi. you don't need to connect to them directly. The observer process manages that and serves as "the robot" from your perspective.

This server runs inside the observer itself. Your tools call the same methods the robot's own behaviors use, and you can load new behaviors (*maneuvers*) into the running process without restarting it. That makes an experimental loop possible: observe, try something, watch what happened, change the code, try again.

This is a real machine in a real room, often a home. Read **Safety** before you move anything.

## Safety

- **Move only with permission.** Before your first move in a session, ask the user whether it is safe to begin. Assume permission is revoked if the observer process is stopped, or your session ends, or if they explicitly say it is revoked.
- **`stop` always works.** It cancels whatever is running and brings every spool to rest, the same as the stop button in the UI. A person can press the UI's stop button at any time too, and your task will end with reason `stopped`.
- **Start small.** Use `nudge` (at most 0.5 m) and short `drive` bursts before `move_to`. Look at an image first.
- **Never weaken a safety monitor without the user's say-so.** Don't use `override_safety(on_over_tension=IGNORE)` or a `SafetyPolicy` that ignores over-tension unless the user agrees. The monitors are explained under **How motion ends**.
- **Don't answer popups for the user.** `answer_popup` exists so you can act on what the user tells you. A popup usually asks a person to confirm something physical, such as "are the cards on the floor?". Ask the user.
- **The robot refuses to move while parked on its hook.** Ask the user before unparking (`run_command('UNPARK')`).
- **Some commands are left to the person.** Shutdown and firmware updates are refused here. They are for the user to run from the UI.
- **Prefer to move with swing cancellation turned on.** Swing cancellation uses the gripper IMU to cancel swing in the pole and makes motion more reliable. Swing cancellation only works well after good calibration. The robot's config records whether swing cancelltion was calibrated and found to work well. But it can always unreliable near the ceiling.

## Coordinates and units

- **Frame and units:** room frame, metres, z up, the floor at z = 0. Angles are in degrees unless a docstring says radians. Tensions are in newtons. The origin is wherver the "origin" card was placed during calibration, and it is the center of the ortho image.
- **Gantry vs gripper:** `gantry_position` is the gantry, and `gripper_position` is the gripper at the end of the pole below it. To put the gripper at a point, aim the gantry at `point + pole_offset`. `move_to(frame='gripper')` does this for you.
- **Fingers and wrist:** fingers run from -90 (open) to 90 (closed) but these are not real angles, just arbitrary command units.  The wrist runs from 0 to 1080, these are real angles in degrees.
- **Named positions:** places in the room such as `hamper`, `trash` and `parking_location`. They appear in `get_state()['named_positions']`. Only try to park and unpark robots with saved parking positions. If its not saved the user will handle it manually.

## First steps

1. **`get_state`** shows what is connected, where the gantry is, whether it is parked, what is running, and anything waiting on a person.
2. **`capabilities`** lists every command and verb the robot answers to, with the docstring of each.
3. **`get_image('ortho')`** gives a top-down view of the floor. `get_image('gripper')` shows what is under the gripper, and `anchor0` to `anchor3` are the corner cameras.
4. **Check the safety basics with the user** before moving anything.

## Tools

| Purpose | Tools |
|---|---|
| Read | `get_state`, `get_events`, `capabilities`, `api_docs`, `guide` |
| See | `get_image`, `get_images`, `get_frames`, `record_clip` |
| Move (return at once) | `move_to`, `nudge`, `drive`, `run_command`, `run_verb` |
| Gripper | `set_fingers`, `set_wrist`, `run_command('GRASP')` |
| Wait and learn the outcome | `wait`, `get_events` |
| Stop | `stop` |
| People | `answer_popup` |
| Code (needs `--mcp_allow_code`) | `load_maneuver`, `unload_maneuver`, `get_maneuver_source`, `save_maneuver`, `run_script` |

## How motion ends

Only one *motion task* runs at a time, and starting one cancels the one before. Every tool that moves the robot starts a motion task and **returns at once**. To find out how it went:

```
nudge(0, 0, 0.05)                   -> {"started": true, "motion_task": "nudge", ...}
wait(until="motion_done", timeout=30)
  -> {"motion_results": [{"name": "nudge", "outcome": "finished",
                          "message": "Nudge finished after 1.8 s.", ...}],
      "events": [...], "gantry_position": [...]}
```

`wait` reports everything since your last move. A task that ended before you called `wait` is still in the report.

Every result has an `outcome`, a `reason` and a plain-language `message`:

| outcome | reason | what happened | what to do |
|---|---|---|---|
| finished | | It ran to the end. `value` is what it returned. | |
| failed | | It raised. `error` holds the traceback; lines in your own code show as `<agent_maneuvers.NAME>` or `<agent_scripts.script_N>`. | Fix the code and load it again. |
| cancelled | stopped | `stop` or the UI's stop button. | If it wasn't you, a person stopped it. Ask why before retrying. |
| cancelled | replaced | Another motion task started. | Normal if you started it. |
| cancelled | tension | **Safety trip.** A line went over the safe tension. The motors were put into a damped mode for 1 s to relax it. | The move pulled against something, or went too far toward an edge or the floor. Look at an image and the line tensions, then try a gentler move. Don't just retry. |
| cancelled | marker | **Safety trip.** The anchor cameras lost track of the gantry marker, or saw it in two places, during a task that needs it. | Tell the user. Something may be blocking the cameras. |
| cancelled | removed | You unloaded or reloaded the maneuver that owned it. | |
| refused | parked | The robot is on its parking hook. | Ask the user about unparking. |

`safety_trip` is true for `tension` and `marker`. Safety monitors also file `safety` events when they act, even when no task was running or the task ran on.

**`get_events(since)`** is the full log. It holds:

- `motion_result`
- `safety`
- `popup` (with an `id` and `buttons`)
- `progress` (the progress bar a maneuver drives)
- `component` (connections coming and going)
- `log` (every warning or error, and INFO logs from your own code)

Pass back `next_event_seq` to get only what is new.

## Seeing

- **Cost:** images cost tokens, about 350 for a 512 px frame.
- **`get_image(camera, fresh=True)`** waits for a frame captured after the call, so it shows the result of a move that just ended. Without `fresh` you get the newest frame already received, which may be a moment old.
- **`wait(..., snapshot='gripper')`** waits and then captures a fresh frame in the same call.
- **`get_images(views)`** takes several cameras in one call, each with an optional crop as fractions of the frame, e.g. the gripper view beside a close-up of part of the ortho. Small crops are enlarged so they stay legible.
- **`get_frames(camera, count, interval_s)`** returns one grid image of a short sequence, for things that move: a swing settling, fingers closing, an object sliding.
- **`record_clip(camera, seconds)`** writes an mp4 for the user to watch, and gives you six frames from it.
- **Live streams:** the user can watch live video at the URLs in `get_state()['video_streams']`.

You cannot react to video in real time from here, because each of your turns takes seconds. **Anything that has to respond to what a camera sees, such as tracking, centering or waiting until something appears, belongs in a maneuver.** The maneuver runs inside the observer at full camera rate. You write it, run it, and judge the result from stills and events.

## Using automatic grasping

A small visual servoing model is used when running `run_command('GRASP')` which is reliable for grasping a prominent object in the gripper's frame. It can steer some distance toward whatever it finds most prominent; from code, `await ob.grasp(max_wander_m=0.15)` keeps it within that distance of where it started and gives up on a target beyond it.

## The experimental loop

Use this loop to develop a behavior:

1. **Observe.** Call `get_state` and look with `get_image`. Use `run_script(motion=False)` to read anything the getters expose:
   ```python
   async def run(ob, m):
       return dict(range=ob.laser_range(), tilt=ob.pole_tilt(), holding=ob.is_holding())
   ```
2. **Probe with primitives.** Try `nudge`, `move_to`, `set_fingers` and `run_command('GRASP')`, then `wait`. Learn how the robot responds before you automate anything.
3. **Write a maneuver.** Save it to a file and load it with `load_maneuver(path=...)`, or pass `source` inline. It answers to the verbs you declare.
4. **Run it** with `run_verb('yourverb args')`, then call `wait(until='motion_done', snapshot='gripper')`.
5. **Judge the run.** Read the motion result, the `log` events your code wrote, and the images. Did it do what you meant? If not, form one specific hypothesis about why.
6. **Make changes and reload** with `load_maneuver` under the same `module_name` (with a path, edit the file and load it again). The old version is removed and the new one registered, with no restart. If the new source fails to import, the old version stays loaded and you get the error.
7. **Repeat**, keeping the moves small until the behavior is right.
8. **Save it** when you have accomplished the goal: `save_maneuver(module_name)` writes it to the maneuver folder, and it loads on every start from then on.

Tell the user what you are trying and why as you iterate.

## Writing a maneuver

```python
import asyncio, logging
import numpy as np
from nf_robot.host.maneuver import Maneuver, verb, TiltWatch

logger = logging.getLogger(__name__)    # INFO and above reach you as `log` events

class Hover(Maneuver):
    name = 'hover'                       # unique; also the owner name in results
    title = 'Hover'                      # shown on the UI's progress bar

    @verb('hover', motion=True)          # run_verb('hover 0.3') runs this as the motion task
    async def hover(self, height='0.3'):
        """Hold the gripper at a height above whatever is under it."""
        target = float(height)
        watch = TiltWatch(self.ob)       # trips if the pole leans as if it hit something
        try:
            for step in range(40):
                if (why := watch.check()) is not None:
                    raise RuntimeError(why)          # ends as outcome 'failed' with this message
                rng = self.ob.laser_range()
                if rng is None:
                    await asyncio.sleep(0.1)
                    continue
                error = rng - target
                logger.info('step %d range %.3f error %.3f', step, rng, error)
                if abs(error) < 0.01:
                    return dict(range=rng, steps=step)   # becomes the result's `value`
                await self.ob.nudge_gantry(np.array([0.0, 0.0, -np.clip(error, -0.05, 0.05)]))
            return dict(range=self.ob.laser_range(), steps=40, converged=False)
        finally:
            self.ob.slow_stop_all_spools()   # cancellation can land at any await
```

**Rules:**

- **Use only the public API.** Call observer methods without a leading underscore, through `self.ob`. Don't touch `ob.pe`, `ob.datastore`, `ob.gripper_client` or `ob.anchors`, and never write calibration or geometry fields of `ob.config`. `api_docs()` lists the public API, and `api_docs('seek_goal')` gives one method in full. `api_docs('maneuver_readme')` is the complete maneuver guide with a longer example.
- **Bring the robot to rest when a motion method ends.** Wrap the body in `try`/`finally` and call `self.ob.slow_stop_all_spools()` in the `finally`. A stop, a new command or a safety monitor can cancel it at any `await`. This leaves the robot hanging somwhere, which is fine. Parking should be done only at the end of the session or if asked.
- **Inside a motion task, `await` other motion coroutines directly** (`seek_goal`, `nudge_gantry`, `grasp`, ...). Never use `invoke_motion_task` there, because it would cancel the task you are in.
- **Keep heavy work off the event loop.** Model inference or big numpy work goes through `await self.run_in_thread(fn, ...)`. The loop also runs telemetry and the safety monitors, its very full already.
- **For velocity control**, call `await self.ob.move_direction_speed(velocity, key=self.velocity_key())` repeatedly; a key expires 2 s after its last command. Zero it when done. Respect `self.ob.speed_limit()`.
- **Return a value from a motion method** to have it land in the result's `value`. A small dict of what happened is easier to judge than logs alone.
- **Use the other decorators where they fit.** `@command(control.Command.X)` handles a UI command, `@startup_step('name')` adds a startup step, and `self.spawn(coro)` starts a background task. Failures in spawned tasks arrive as `log` events.
- **One module can define several maneuvers.** Reloading the module replaces all of them.

## Scripts

`run_script(source)` runs `async def run(ob, m)` once. `ob` is the observer and `m` is a maneuver for helpers like `m.notify` and `m.velocity_key()`.

- **`motion=True` (the default)**: use this whenever the script can command motion of any kind (spools, gantry, wrist or fingers) at any point in its run, directly or through anything it calls. It runs as the motion task, replacing whatever was moving, and the stop button and safety monitors can cancel it. The tool returns at once; follow it with `wait`.
- **`motion=False`**: use this only for a script that never moves anything, such as one reading getters or computing something. It runs beside the motion task without disturbing it and returns its value directly. Nothing can stop it, so a script that moves the robot must never run this way.

Scripts are good for one-off measurements. For anything you will run more than once, write a maneuver so the user can run it from the UI's debug box too.

## When things go wrong

- **`no frame from ... within 3 s`:** that camera isn't streaming. Check `components` and `anchor_cameras` in `get_state`.
- **Code tools refused:** the user started the server without `--mcp_allow_code`. Ask them whether they want to enable it.
- **A command did nothing:** look at `get_events` for a `log` warning, and at `last_motion_result` in `get_state`.
- **Hang position desync:** if positions look wrong (an image shows the gripper somewhere other than where `gripper_position` says), or movement is unreliable or causing unwanted slack lines or swinging, the first thing to do is to try getting the gantry to a position where it can be seen and running `run_command('HALF_CAL')` re-references the lines from the cameras.
- **If swing cancellation is pumping instead of damping:** Turn it off and notify the user.

## Setup (for the person running the robot)

```sh
pip install "nf_robot[host,mcp]"
stringman-headless --mcp --mcp_allow_code --maneuver_dir ./agent_maneuvers
claude mcp add --transport http stringman http://127.0.0.1:4250/mcp
```

- **Port:** `--mcp` takes an optional port (4250 by default).
- **Code tools:** `--mcp_allow_code` lets the agent run Python inside the observer process. Leave it off to allow driving and watching only.
- **Saved maneuvers:** `--maneuver_dir` is where saved maneuvers live and are loaded from at startup.
- **Remote machines:** the server listens on 127.0.0.1 only. To reach it from another machine, use an SSH tunnel: `ssh -L 4250:127.0.0.1:4250 robot-host`.
