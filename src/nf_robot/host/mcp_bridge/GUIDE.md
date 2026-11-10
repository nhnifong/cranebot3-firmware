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
- **Fingers and wrist:** fingers run from -90 (open) to 90 (closed) but these are not real angles, just arbitrary command units. Finger angles are for opening and shaping the empty hand; grasping is done by force (see **Gripping**). The wrist runs from 0 to 1080, these are real angles in degrees.
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
| Gripper | `grip` (close and hold by force), `set_fingers` (open or shape the empty hand only), `set_wrist`, `run_command('GRASP')` |
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

- **What it goes for depends on where it starts.** It picks the most prominent, high-contrast thing in the palm camera's view. Put the object you want in the middle of the frame at about 0.5 m before calling it. One distracting object nearby (a wide flat disc, a bright tray, a book with an eye on the cover) can pull every attempt toward it; ask the user to move it if it keeps happening.
- **A tight `max_wander_m` makes it give up rather than try.** If the model's chosen target is outside the limit, the grasp ends at once. Use 0.25 to 0.5 unless something nearby must not be touched.
- **Grasping and checking repeatedly beats aiming carefully.** The fastest loop is: call `grasp()` until it says it holds something, then look at the gripper image to see what it got, and decide where it goes. Hand-built dives (open, descend, `close_grip`) did worse than auto grasp on every class of object tried, including flat cards and cushions, which auto grasp can pick up.
- **Its held/not-held answer is unreliable in both directions:**
  - Hard objects (plastic trucks, discs) often do not press the finger pads hard enough to register. The grasp then reports failure and opens the fingers, dropping an object it had.
  - It sometimes reports held with the fingers fully closed on nothing (finger angle about 89-90).
  - Thin things (paintbrush, marker, dowel) also close to about 89, but the pad reading tells them apart: `ob.finger_pad_voltage()` is about 0.08 or more on a thin object and 0.05 or less when empty. Use the angle and the pad voltage together, and look at the image when in doubt.
- **Small balls and flat, thin sticks are the hardest.** They tend to roll or slip out as the fingers close; expect several attempts.

## Gripping

**Close the fingers at a constant rate; never command a closed finger angle.** The gripper has a low-level controller for this. While the fingers close at the commanded finger speed, the pads sense pressure; on contact the gripper switches itself to force mode. From then on a finger speed is the rate the holding force changes (normalized 0 to 1, held by a PID loop): positive tightens, 0 holds the force where it is, and negative loosens until, below zero, it lets go. Finger speed commands expire after 0.2 s, so they have to be repeated.

An angle command defeats all of this. The object stops the fingers short of the angle, the servo keeps pushing toward it, and its overload cutout drops whatever it held, typically mid-carry when the load rises. An angle command also takes the gripper out of force mode.

- **`grip(hold_force)`** closes at a constant rate until the force is reached and then holds it. `await ob.close_grip(hold_force=0.25)` does the same from code. About 0.25 suits most things; soft or floppy things like cloth take less, heavy things more.
- **`set_fingers(angle)`** is for opening (letting go) and for shaping the empty hand before a dive.
- **`run_command('GRASP')`** (or `await ob.grasp()`) is the automatic grasp: it centres over an object with the visual servo model and closes by force itself.

A grasp in a maneuver:

```python
await self.ob.set_finger_angle(-60)                  # open, before reaching the object
...                                                  # descend until the jaws are around it
grip = await self.ob.close_grip(hold_force=0.25)     # close at a constant rate, hold by force
if not grip['held']:
    raise RuntimeError(f'nothing between the fingers: {grip}')
...                                                  # carry it; the gripper keeps the force
await self.ob.set_finger_angle(-60)                  # let go
```

## Practical tricks

Lessons from driving the robot through MCP. Assume about 90% of them still hold: any of them may have become obsolete or wrong since it was written.

### Grasping

- **Grip by asking for constant force, not by constant finger angle.** Use `grip` / `ob.close_grip()`: it closes at a constant speed, the gripper switches to force mode on contact, and the force is held. Closing to a fixed angle on an object seldom works and can even overload the finger servo if you catch something the wrong way.
- **In clutter, it helps to dive into something with fingers nerly closed** Close to ~80 before descending, and open them only near the object (laser range ~0.08 m). Opening pushes neighbours aside instead of a finger landing on them. Objects have to be between fingers to grasp.
- **Be opportunistic in a pile.** Auto GRASP over a pile reliably comes up with one or two pieces in ~18 s, much faster than aiming at a particular one. Take whatever comes; if it is more than one, drop them from ~0.3 m onto open floor so they bounce apart, then pick them singly.
- **Open fingertips reach ~0.10 m below the rangefinder.** With the fingers open the laser never reads less than ~0.10 over bare floor; open them at ~0.15 m, then lower in small steps and stop when the pole starts to lean (fingertips on the floor) and close there.
- **People and children may come into the work area.** Watch the frames for hands and feet, and stop when someone is reaching in. They are used to the robot though and it's fairly harmless and slow.
- **A held object blinds the rangefinder** and fills the middle of the palm camera. Use the estimated height while carrying, and ignore blobs in the middle of the frame when looking for things on the floor.
- **Before an auto grasp, put the target at the image center.** The servo model steers from there. Releasing is different: things fall from the fingertips, which sit 10-15 cm from the image center (see "Clearing a floor").
- **Recenter at ~0.3 m range before the final drop.** The gantry drifts several centimetres during a descent.
- **Image right in the room frame** is `ob.gripper_camera_to_room([1, 0, 0])`; use it to turn a pixel offset into a nudge. One pixel of the 684 px frame is about depth / 302 m, where depth is the camera's distance to the thing in that pixel, not necessarily the laser range.
- **Fingers close along image horizontal.** Rotate the wrist so the object's long axis is image vertical.
- **Auto grasp may fail near walls** and other motion may too. due to a tangle prevention mechanism, spools refuse to let out line if they are compeltely slack, which has the side effect of phantom forces within about 20cm of walls that can prevent you from getting close to them. the longest lines are the loose ones and the spool won't let them out any further because it's reading approximately zero tension. worse in tall rooms
- **Automatic GRASP can wander** to whatever the servo model finds most prominent, up to ~0.7 m away. Near things that must not be touched, call `ob.grasp(max_wander_m=...)` from a script, but below about 0.25 it mostly gives up instead of grasping.
- **A tilt over ~7 deg near the floor means the gripper is resting on something.** Lines go slack and lateral nudges barely move it. Lift a few cm before moving sideways.

### Moving things

- **Lay long floppy objects straight by lift-and-sweep.** Lift until it hangs nearly free, carry it over open floor, then move horizontally while descending so it unrolls behind the gripper.
- **Try to set things down at the same height from a surface you picked them up with**

### Seeing

- **The position estimate drifts 5-20 cm** between visits, worst after a marker dropout, and height can drift by 0.3 m or more over many dives; a half calibration resets both. Absolute coordinates are not good enough for tight layouts; place relative to what the palm camera sees, and check the result in the ortho view.
- **The gantry marker drops out in predictable places** such as very low, where it is viewed with a steep angle, or obscured by the "cross" on the top of the gantry. it can also be in one of the blind spots of either camera. encoder based positioning (aka hang point prediction) keeps working and feeding the kalman filter in the position estimator.
- **Ortho pixels to room:** on the 1000 px ortho, x = (u - 500) / 200, y = (500 - v) / 200 (`ml/ortho_target/model.py`).
- **The robot itself shows up in the ortho** as a smeared shape with a line: the gripper and pole projected from the air. It is not an obstacle on the floor. It can be white, copper, or pink.
- **Crop the ortho at full resolution** (`ob.latest_ortho()`, save a crop to a file and read it) to count small objects; the 512 px tool image is too coarse.
- **The marker-fault popup can clear on its own** once the gantry rises.

## Clearing a floor

Lessons from tidying a cluttered playroom into bins.

- **Plan the layout before picking anything up.** Every drop is a round trip, and the trip to the container usually costs more than the grasp. Light, empty bins can be carried: pick one up by its rim and set it down beside the pile before you start, and put each container close to the kind of thing it will collect. Containers that cannot move (a ball pit in a corner) are worth a separate batch at the end.
- **Run a half calibration before every drop.** The height estimate drifts during repeated dives, by up to 0.35 m in one session, and lateral position drifts too. `await ob.half_auto_calibration()` takes about 2 s near the middle of the room and keeps drops and grasp heights honest. If `gripper_position()` reports the gripper below the floor, it is overdue.
- **Things fall from the fingertips, not from under the image center.** In the palm camera, the closed fingertips sit near the bottom middle of the frame (about pixel 256, 215 of a 512 x 288 view at 0.6 m), roughly 10-15 cm from the image center, and that offset turns with the wrist. To find a drop position, move until the container's center is under the fingertip point, then record `gripper_position()`. A position recorded by centering the container in the image will miss by the offset, by a different amount at each wrist angle.
- **Use the right depth when turning a pixel into a move.** One pixel is about depth / 302 m in the 684 px frame, where depth is the distance from the camera to the thing in that pixel. The rangefinder reads whatever is under the center, often clutter or a bin's contents, so it is the wrong depth for a pixel off-center. Use the estimated gripper height minus the target's height; bin rims are about 0.25 m above the floor, which halves the depth.
- **Drop from low, aligned with the container.** About 0.3 m above the rim is enough. Higher drops bounce things out. Turn the wrist so a long object lies along the container's long axis before letting go; one that lies across it bounces out and can roll into a dead zone by a wall.
- **Lower floppy things in while spinning the wrist.** A long plush toy dropped from above drapes over the rim. Lowering it to just above the bottom while turning the wrist one or two full turns coils it inside.
- **Lines can hit tall furniture.** Away from the middle of the room, the lines pass low enough to touch things like a playhouse roof. Over-tension trips near tall furniture usually mean this; work elsewhere, or ask the user to move things into the open.
- **Ask the user about placement.** Whose things they are and where they belong (papers by the computer, for example) is the user's call. Ask before treating anything as trash.
- **Keep a short list of what is left and why.** Out of reach, too flat, distracting the grasp: when the user offers to help, that list tells them exactly what to move.

### Throughput

A profiled two-hour playroom session cleared about 70 items, roughly 1.4 minutes per item once a working loop was running. Of each roughly 60 s cycle, about 45 s went to grasping (most of it retries) and 13-15 s to carrying and dropping. The first quarter of the session produced almost nothing. In order of payoff:

- **Get to the working loop first.** Build "grasp until held, look, drop, grasp the next one" as a single maneuver verb before trying anything clever. One call should drop the held item and start the next grasp, so the robot is never waiting on you between items. In that session, blind picks at absolute coordinates, hand-built dives and careful bin measuring took the first 24 minutes and delivered a handful of items.
- **Cut grasp retries.** Retries cost far more than travel. Start each grasp with something graspable in the middle of the frame. Reject an empty hand by finger angle and pad voltage together (see "Using automatic grasping"), or thin objects get thrown away as misses.
- **Cut travel.** Move light containers beside the pile at the start. Grasp near the last drop point instead of always returning to one spot. For a far, fixed container (a ball pit in a corner), clear what goes into the near containers first, so that late trips to the far one are not mixed in with everything else.
- **Measure container drop points once, correctly.** Use a half calibration, then center the container under the fingertip point (not the image center) and save the position. Re-measuring after missed drops and fetching things that bounced out wasted most of an hour in that session.
- **Keep your own turns cheap.** One snapshot per cycle, taken by the `wait` that ends the motion, is enough to identify the item and choose its container. Batch maneuver edits, since each reload and test costs a minute or more. Time between motion tasks was about 16% of wall time once the loop was running.
- **Avoid long side trips mid-task.** `spin-recover` (about 2 minutes) and other calibrations belong at the start, or after a clear sign that something is wrong.

### The floorclean maneuver

`floorclean` is a built-in maneuver that runs the loop described above, so you don't have to write one. It works without `--mcp_allow_code`. The robot works through a queue on its own while you look at images, so it is never waiting on you.

```
run_verb('cleandest pit -1.5 -1.75 1.0 0.4')       # big target: opens on the fly inside the 0.4 m circle
run_verb('cleandest bin 0.66 -0.04 0.6')           # small target: centers, descends, then opens
run_verb('cleanadd 0.2,0.3,pit -0.1,0.5,bin')      # queue jobs
run_verb('clean')                                  # start; returns at once
wait(until='event', timeout=60, snapshot='gripper')
  -> a log event "HELD at [...] bound for pit ...", with the gripper image of what it got
run_verb('cleanto bin')                            # only if the grasp got something else
```

| Verb | What it does |
|---|---|
| `clean` | Motion task. Runs the queue, and finishes once the queue has stayed empty for 45 s. |
| `cleandest NAME X Y Z [RADIUS] [WRIST]` | Sets a place to let go, as a gripper position. Use `-` to skip the radius. A wrist angle turns the wrist first, to line long items up with a container. |
| `cleanadd JOB ...` | Queues jobs, also during `clean`. A job is `x,y,dest[,tries,wander,hover,trust]`, with empty fields for the defaults (2 tries, 0.5 m wander, 0.55 m hover). |
| `cleanto NAME` | Changes where the carried item goes, without stopping. |
| `cleanclear`, `cleanshow` | Empties the queue, or shows it with the current job and destinations. |
| `cleanstats` | Totals for jobs, grasp attempts and time per phase, and the path of the per-job log. |

- **Order is chosen for you.** The next job is the one with the least travel from the gripper, through the item, to its container.
- **Expect `wait(until='event')` to be how you follow it.** Pass the `next_event_seq` from the last reply as `since`, or an event that arrived between calls can be missed.
- **Redirect fast on short carries.** A carry can take only a few seconds, and looking and replying took 4-10 s in testing. `cleanto` is refused once the gripper has started down over a small target, and the redirect is lost if the item has been released. Set the likely destination in the job to begin with.
- **Grabs can be something large nearby.** With a large `wander`, a grasp near a fabric laundry bin picked the bin up by its rim. Use a `wander` of about 0.25 and a lower `hover` near things that must not be picked up or that distract the model.
- **Flag thin items `trust`.** A rubber glove or a sock can close the fingers past the "closed on nothing" check. `trust` keeps the hold, at the cost of sometimes carrying nothing, so look at the image.
- **Coordinates are starting points.** A grasp often gets a neighbor within `wander`, so refresh the queue from a new ortho view as it empties, and clear jobs whose items have gone. A miss costs 15-60 s.

Measured on one cluttered playroom: about 66 items in 45 minutes, a median cycle of 26 s into a ball pit and 32 s into a bin, with 56% of the time spent in the grasp.

### Fast cleanup

From playroom sessions. A queue-driven loop cut a roughly 70-item cleanup from 129 min to 45 min. Grasping was still more than half of that time, and retries were about 40% of grasping.

- **Run a job queue in a maneuver, and feed it while it works.** One motion verb pops the cheapest job (travel from here to the item plus item to its container), flies to a hover point, grasps, carries, drops, and starts the next. The agent only adds jobs (`x,y,dest` from the ortho) and redirects carried items. Queue several ahead so the robot never waits on you. The built-in `floorclean` maneuver is a working version (see "The floorclean maneuver" above); it logs a per-job row of fly, grasp, carry and calibration time for reporting.
- **Assume the grasp got what you queued.** Carry at once, and look at the snapshot while it flies. Redirect only if it grabbed something unexpected. An agent turn takes 10-15 s, so redirects only work for containers more than a few seconds away.
- **Drop into big targets on the fly, center over small ones.** For a ball pit, open as soon as the gripper is inside its circle. Small bins need a half calibration, centering, a short settle, then the release.
- **Near an on-the-fly target, don't let the servo choose.** Items grabbed beside the ball pit landed in it before a redirect could arrive (a booklet, a fabric item, a toy phone). Near the pit, queue only what is clearly a ball, use `max_wander_m` about 0.25 and a lower hover. Don't queue items against the rim either: the gripper can grab the rim and tip the pit over.
- **One `grasp()` call per job, then requeue misses for later.** Each call already makes up to 3 internal attempts. A second call at once usually repeats the first failure; coming back after nearby items have moved works better.
- **Reject false holds yourself.** The holding head can read 0.9-1.0 with the fingers shut on nothing (angle about 88-90, pad voltage at most 0.06). The servo then reports success and skips its own retries. Count those as misses. For thin items, such as a rubber glove, that sensors cannot tell from empty, mark the job as trusted.
- **Start lower over a distractor.** A dark tripod foot or a printed cover pulls the model. Hovering at 0.35-0.45 m instead of 0.55 m narrows the view to the target.
- **Merge soft things going to the same place.** Dropping one sock on another and grabbing both saves a trip. This does not work for hard toys.
- **Work out the image axes before measuring a drop point.** Nudge +x by 0.15 m and see which way the scene shifts. The mapping turns with the wrist, so measure again if the wrist moves.
- **Capture the visual servo's own log** to learn why grasps end. From `run_script(motion=False)`, add a `logging.FileHandler` to the `nf_robot.host.visual_servo` logger. Its lines (`Close head says go`, `approach ended: ...`, `Close ended empty ...`) separate profile problems from model problems.
- **Check redirects with a script.** `run_verb` does not return a non-motion verb's value, so `cleanto` gives no sign of whether it was accepted. Read the maneuver's state from `run_script(motion=False)`.
- **Try a grab profile on a few items before committing.** `grab-profile fast` scored 1 hold in 5 on 2026-11-10, against 5 in 10 for `default` the same day; its misses shut the fingers on nothing. Switch back after about 5 misses rather than running a whole job on it.
- **A high, idle robot is normal.** Users park it about 2 m up so people can walk under it, and a stop you did not send is usually the user. Ask why before restarting.
- **Leave alone what the user says to leave.** Large flat mats, brooms and other awkward things cost more than they're worth. Keep a list of skipped items for the report.
- **Report per subtask.** Give time spent flying, grasping, carrying and calibrating, plus holds per `grasp()` call, false holds rejected, and items that went to the wrong place.

## The experimental loop

Use this loop to develop a behavior:

1. **Observe.** Call `get_state` and look with `get_image`. Use `run_script(motion=False)` to read anything the getters expose:
   ```python
   async def run(ob, m):
       return dict(range=ob.laser_range(), tilt=ob.pole_tilt(), holding=ob.is_holding())
   ```
2. **Probe with primitives.** Try `nudge`, `move_to`, `grip`, `set_fingers` and `run_command('GRASP')`, then `wait`. Learn how the robot responds before you automate anything.
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
- **`spin-recover` can switch swing cancellation off.** It ends with a damping check, and if that check trips the tension limit it marks swing cancellation unverified. `prefer_swing_cancellation()` then leaves it off for every maneuver. Tell the user; they decide whether to turn it back on with `ob.set_swing_cancellation(True)`.
- **A "line looks derailed" stop can be a false positive**, for example while the gripper is pressed against something. Stop and ask the user to check the spool before moving again.

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
