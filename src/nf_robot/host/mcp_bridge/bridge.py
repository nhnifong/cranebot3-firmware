"""An MCP server inside the observer, so an agent can watch and drive the robot and load
maneuvers it writes without the observer restarting.

The bridge is an ordinary maneuver: everything it does goes through the observer's public
methods, and every move it makes is a motion task, so the stop button, the tension and marker
monitors and the parked check apply to the agent exactly as to anything else. Tools that
start motion return at once; the agent learns how it ended from wait or get_events.

See GUIDE.md, which the guide tool serves, for how an agent is meant to use it.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import math
import socket
import threading
import time
import traceback
from collections import deque
from contextlib import contextmanager
from functools import wraps
from pathlib import Path

import cv2
import numpy as np
from pydantic import BaseModel, Field

from nf_robot.generated.nf import control
from nf_robot.host.maneuver import Maneuver
from nf_robot.host.mcp_bridge.agent_code import MANEUVER_PREFIX, SCRIPT_PREFIX, AgentCode

logger = logging.getLogger(__name__)

DEFAULT_PORT = 4250
GUIDE_PATH = Path(__file__).with_name('GUIDE.md')
MANEUVER_README_PATH = Path(__file__).parent.parent / 'maneuvers' / 'README.md'
CAMERAS = ('gripper', 'anchor0', 'anchor1', 'anchor2', 'anchor3', 'ortho')
MAX_WAIT_S = 120.0
MAX_DRIVE_S = 5.0
# how long run_command and run_verb wait on a handler before leaving it running in the
# background; motion handlers return at once anyway, since they only start the motion task
HANDLER_WAIT_S = 10.0
EVENT_HISTORY = 1000
MAX_ERROR_CHARS = 6000
# a crop is enlarged toward max_px by at most this much; past it the pixels are only blur
MAX_CROP_ENLARGE = 4.0

INSTRUCTIONS = """\
Controls a Stringman cable robot through the observer process that runs it. Call `guide`
first: it explains the tools, the safety rules, and how to iterate on behaviors by loading
maneuver code into the running observer."""


def _jsonable(value, digits=4):
    """value with numpy arrays, numpy scalars and enums made into plain JSON types."""
    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        return round(value, digits) if math.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist(), digits)
    if isinstance(value, np.generic):
        return _jsonable(value.item(), digits)
    if isinstance(value, dict):
        return {str(k): _jsonable(v, digits) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v, digits) for v in value]
    if hasattr(value, 'name'):
        return value.name
    return repr(value)


def _vec3(v):
    return None if v is None else [round(v.x, 4), round(v.y, 4), round(v.z, 4)]


def _jpeg(rgb, max_px, quality=80, max_enlarge=1.0):
    h, w = rgb.shape[:2]
    scale = min(max_enlarge, max_px / max(h, w))
    if scale != 1.0:
        rgb = cv2.resize(rgb, (max(1, int(w * scale)), max(1, int(h * scale))),
                         interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC)
    ok, buf = cv2.imencode('.jpg', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError('JPEG encoding failed')
    return buf.tobytes()


class View(BaseModel):
    """One image for get_images."""
    camera: str = Field(description='gripper, anchor0-anchor3 or ortho')
    crop: list[float] | None = Field(
        default=None, description='[left, top, right, bottom] as fractions of the frame, 0 to 1; '
                                  'omit for the whole frame')
    max_px: int = Field(default=512, description='bound on the longer side of the result')


def crop_frame(frame, crop):
    """The part of frame inside crop, [left, top, right, bottom] as fractions of its size."""
    if crop is None:
        return frame
    if len(crop) != 4:
        raise ValueError(f'crop is [left, top, right, bottom], got {crop}')
    left, top, right, bottom = (min(1.0, max(0.0, float(c))) for c in crop)
    if right <= left or bottom <= top:
        raise ValueError(f'crop {crop} is empty: right must exceed left and bottom exceed top')
    h, w = frame.shape[:2]
    x0, y0 = int(left * w), int(top * h)
    x1, y1 = max(x0 + 1, int(round(right * w))), max(y0 + 1, int(round(bottom * h)))
    return frame[y0:y1, x0:x1]


def contact_sheet(frames, labels, max_px):
    """One RGB image of frames laid out in a grid, each scaled to fit max_px and labelled."""
    cells = []
    for frame, label in zip(frames, labels):
        h, w = frame.shape[:2]
        scale = min(1.0, max_px / max(h, w))
        cell = cv2.resize(frame, (max(1, int(w * scale)), max(1, int(h * scale))),
                          interpolation=cv2.INTER_AREA)
        cell = np.ascontiguousarray(cell)
        cv2.putText(cell, label, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
        cv2.putText(cell, label, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)
        cells.append(cell)
    cell_h = max(c.shape[0] for c in cells)
    cell_w = max(c.shape[1] for c in cells)
    cols = math.ceil(math.sqrt(len(cells)))
    rows = math.ceil(len(cells) / cols)
    sheet = np.zeros((rows * cell_h, cols * cell_w, 3), np.uint8)
    for i, cell in enumerate(cells):
        r, c = divmod(i, cols)
        sheet[r * cell_h:r * cell_h + cell.shape[0], c * cell_w:c * cell_w + cell.shape[1]] = cell
    return sheet


def _result_dict(result):
    if result is None:
        return None
    d = dict(name=result.name, owner=result.owner, outcome=result.outcome, reason=result.reason,
             safety_trip=result.safety_trip, message=result.message,
             duration_s=round(result.ended_at - result.started_at, 2), ended_at=result.ended_at)
    if result.error:
        d['error'] = result.error[-MAX_ERROR_CHARS:]
    if result.value is not None:
        d['value'] = _jsonable(result.value)
    return d


class _EventLogHandler(logging.Handler):
    """Hands the bridge every warning, and the INFO logs of agent code."""

    def __init__(self, bridge):
        super().__init__(level=logging.INFO)
        self.bridge = bridge

    def emit(self, record):
        if record.levelno < logging.WARNING and not record.name.startswith(
                (MANEUVER_PREFIX, SCRIPT_PREFIX)):
            return
        if record.name.startswith(('uvicorn', 'mcp', 'httpx', 'sse_starlette')) \
                and record.levelno < logging.ERROR:
            return
        try:
            text = record.getMessage()
            if record.exc_info and record.exc_info[1] is not None:
                text += '\n' + ''.join(traceback.format_exception(*record.exc_info))[-MAX_ERROR_CHARS:]
        except Exception:
            text = str(record.msg)
        self.bridge.add_event('log', level=record.levelname, logger=record.name, message=text)


class McpBridge(Maneuver):
    name = 'mcp'
    title = 'Agent'

    def __init__(self, ob, port=DEFAULT_PORT, host='127.0.0.1', allow_code=False, agent_code=None):
        super().__init__(ob)
        self.port = port
        self.host = host
        self.allow_code = allow_code
        self.agent_code = agent_code or AgentCode(ob)
        self._events = deque(maxlen=EVENT_HISTORY)
        self._events_lock = threading.Lock()
        self._seq = 0
        # the event seq just before the last tool call that started something, which a wait
        # with no `since` reports from, so a task that ended before the wait is not missed
        self._wait_from = None
        # latest telemetry, filed by what it describes
        self._components = {}
        self._named_positions = {}
        self._video = {}
        self._toggles = {}
        self._targets = []
        self._last_progress = None
        self._popups = {}
        self._log_handler = _EventLogHandler(self)
        self._server = None
        self._server_task = None
        self.mcp = self._build_server()
        ob.telemetry.add_listener(self._on_telemetry)

    # -- maneuver hooks -----------------------------------------------------------

    async def start(self):
        logging.getLogger().addHandler(self._log_handler)
        for prefix in (MANEUVER_PREFIX, SCRIPT_PREFIX):
            logging.getLogger(prefix).setLevel(logging.INFO)
        # state sent before the bridge was listening, replayed as it is to a new UI
        await self.ob.send_setup_telemetry()
        await self._start_server()

    async def stop(self):
        logging.getLogger().removeHandler(self._log_handler)
        self.ob.telemetry.remove_listener(self._on_telemetry)
        if self._server is not None:
            self._server.should_exit = True
        if self._server_task is not None:
            try:
                await asyncio.wait_for(self._server_task, timeout=3.0)
            except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
                self._server_task.cancel()

    def on_motion_task_done(self, result):
        self.add_event('motion_result', **_result_dict(result))

    def on_safety_event(self, kind, message):
        self.add_event('safety', safety_kind=kind, message=message)

    # -- events ---------------------------------------------------------------------

    def add_event(self, kind, **fields):
        """File an event for get_events and wait. Thread safe."""
        with self._events_lock:
            self._seq += 1
            self._events.append(dict(seq=self._seq, time=round(time.time(), 3), kind=kind, **fields))

    def events_since(self, since, limit=200):
        with self._events_lock:
            events = [e for e in self._events if e['seq'] > since]
            first_kept = self._events[0]['seq'] if self._events else self._seq + 1
            seq = self._seq
        dropped = since + 1 < first_kept and since < seq
        return events[:limit], seq, dropped

    @property
    def seq(self):
        return self._seq

    def _on_telemetry(self, key, msg):
        """Telemetry listener: runs on whichever thread sent the item."""
        if key == 'pop_message':
            if msg.id is not None:
                self._popups[msg.id] = dict(id=msg.id, message=msg.message, buttons=list(msg.buttons))
            self.add_event('popup', id=msg.id, message=msg.message, buttons=list(msg.buttons))
        elif key == 'operation_progress':
            progress = (msg.name, msg.current_action)
            if progress != self._last_progress:
                self._last_progress = progress
                self.add_event('progress', name=msg.name, action=msg.current_action,
                               percent=round(msg.percent_complete, 1))
        elif key == 'component_conn_status':
            name = 'gripper' if msg.is_gripper else f'anchor{msg.anchor_num}'
            status = dict(websocket=_jsonable(msg.websocket_status), video=_jsonable(msg.video_status))
            if msg.error_message:
                status['error'] = msg.error_message
            if msg.motor_enabled is not None:
                status['motor'] = _jsonable(msg.motor_enabled)
            previous = self._components.get(name)
            self._components[name] = status
            if previous is None or previous['websocket'] != status['websocket'] \
                    or previous.get('error') != status.get('error'):
                self.add_event('component', component=name, **status)
        elif key == 'named_position':
            if msg.position is None:
                self._named_positions.pop(msg.name, None)
            else:
                self._named_positions[msg.name] = _vec3(msg.position)
        elif key == 'video_ready':
            name = 'gripper' if msg.is_gripper else (
                'ortho' if msg.anchor_num is None else f'anchor{msg.anchor_num}')
            self._video[f'{name} (feed {msg.feed_number})'] = msg.local_uri
        elif key in ('swing_cancellation_state', 'tension_regulation_state', 'torque_state',
                     'auto_targeting_state'):
            self._toggles[key.removesuffix('_state')] = msg.enabled
        elif key == 'target_list':
            self._targets = [dict(id=t.id, position=_vec3(t.position), status=_jsonable(t.status),
                                  source=t.source) for t in msg.targets]

    def pending_popups(self):
        """Popups still waiting on an answer from somebody."""
        waiting = self.ob.pending_popup_acks
        for popup_id in [i for i in self._popups if i not in waiting]:
            self._popups.pop(popup_id, None)
        return list(self._popups.values())

    # -- state ----------------------------------------------------------------------

    def state(self):
        ob = self.ob

        def safe(fn):
            try:
                return _jsonable(fn())
            except Exception:
                return None

        running = ob.current_motion_task()
        return dict(
            time=round(time.time(), 3),
            gantry_position=safe(ob.gantry_position),
            gripper_position=safe(ob.gripper_position),
            pole_offset=safe(ob.pole_offset),
            line_tensions_n=safe(ob.line_tensions),
            max_safe_tension_n=safe(lambda: ob.max_safe_tension),
            laser_range_m=safe(ob.laser_range),
            finger_angle_deg=safe(ob.finger_angle),
            wrist_angle_deg=safe(ob.wrist_angle),
            pole_tilt_deg=safe(ob.pole_tilt),
            holding=safe(ob.is_holding),
            gripper_connected=safe(ob.gripper_connected),
            speed_limit_mps=safe(ob.speed_limit),
            parked=safe(lambda: ob.maneuver('parking').data.parked),
            components=dict(self._components),
            anchor_cameras=safe(ob.anchor_cameras),
            toggles=dict(self._toggles),
            motion_task=None if running is None else dict(
                name=running.name, owner=running.owner,
                running_s=round(time.time() - running.started_at, 1) if running.started_at else None),
            last_motion_result=_result_dict(ob.last_motion_result()),
            route=safe(ob.route),
            named_positions=dict(self._named_positions),
            targets=self._targets[:20],
            target_count=len(self._targets),
            pending_popups=self.pending_popups(),
            video_streams=dict(self._video),
            agent_modules={name: m.maneuver_names for name, m in self.agent_code.modules.items()},
            next_event_seq=self._seq,
        )

    # -- cameras --------------------------------------------------------------------

    async def capture(self, camera, fresh=False, timeout=3.0):
        """(timestamp, RGB frame) from a camera, or raise if there is none."""
        after = None if fresh else 0
        if camera == 'gripper':
            ts, frame = await self.ob.gripper_capture(after=after, timeout=timeout)
        elif camera.startswith('anchor') and camera[6:].isdigit():
            ts, frame = await self.ob.anchor_capture(int(camera[6:]), after=after, timeout=timeout)
        elif camera == 'ortho':
            frame = self.ob.latest_ortho()
            ts = time.time() if frame is not None else None
        else:
            raise ValueError(f'unknown camera {camera!r}; cameras are {", ".join(CAMERAS)}')
        if frame is None:
            raise RuntimeError(f'no frame from {camera} within {timeout} s; check `components` '
                               f'and `anchor_cameras` in get_state')
        return ts, frame

    # -- acting ---------------------------------------------------------------------

    async def send_control(self, **item):
        """Dispatch one ControlItem the way a UI's would arrive, waiting HANDLER_WAIT_S at most."""
        self._wait_from = self._seq
        message = bytes(control.ControlBatchUpdate(updates=[control.ControlItem(**item)]))
        task = asyncio.create_task(self.ob.handle_command(message))
        done, _ = await asyncio.wait([task], timeout=HANDLER_WAIT_S)
        if task in done:
            task.result()
            return 'done'
        return f'still running after {HANDLER_WAIT_S:.0f} s; it carries on in the background'

    async def start_motion(self, coro):
        """Start coro as the motion task, owned by the bridge, and say what happened."""
        self._wait_from = self._seq
        task = await self.ob.invoke_motion_task(coro, owner=self)
        if task is None:
            return dict(started=False, result=_result_dict(self.ob.last_motion_result()))
        return dict(started=True, motion_task=task.get_name(), next_event_seq=self._seq,
                    hint='call wait(until="motion_done") to learn how it ends')

    async def wait(self, until, timeout, since):
        """Wait for the motion task to end, for any event, or out the timeout; then report."""
        timeout = max(0.0, min(float(timeout), MAX_WAIT_S))
        if since is not None:
            start_seq = since
        elif self._wait_from is not None:
            start_seq = self._wait_from
        else:
            start_seq = self._seq
        self._wait_from = None
        started = time.monotonic()
        watched = self.ob.current_motion_task()
        satisfied = False
        while True:
            if until == 'motion_done':
                now = self.ob.current_motion_task()
                satisfied = watched is None or now is None or now != watched
            elif until == 'event':
                satisfied = self._seq > start_seq
            if satisfied or time.monotonic() - started >= timeout:
                break
            await asyncio.sleep(0.1)
        if until == 'motion_done' and satisfied:
            # the done callback runs one loop turn after the task finishes
            await asyncio.sleep(0.05)
        events, seq, dropped = self.events_since(start_seq)
        state = self.state()
        report = dict(
            satisfied=satisfied if until != 'seconds' else True,
            waited_s=round(time.monotonic() - started, 2),
            motion_results=[e for e in events if e['kind'] == 'motion_result'],
            events=[e for e in events if e['kind'] != 'motion_result'],
            next_event_seq=seq,
            motion_task=state['motion_task'],
            gantry_position=state['gantry_position'],
        )
        if dropped:
            report['note'] = 'older events were dropped from the history'
        return report

    # -- docs -----------------------------------------------------------------------

    def api_docs(self, name=''):
        from nf_robot.host.observer import AsyncObserver
        if name:
            for owner in (AsyncObserver, Maneuver):
                member = getattr(owner, name, None)
                if member is not None and not name.startswith('_'):
                    try:
                        signature = str(inspect.signature(member))
                    except (TypeError, ValueError):
                        signature = ''
                    return f'{owner.__name__}.{name}{signature}\n\n{inspect.getdoc(member) or ""}'
            if name in ('maneuver_readme', 'README'):
                return MANEUVER_README_PATH.read_text()
            return f'nothing public named {name!r} on AsyncObserver or Maneuver'
        lines = []
        for owner in (AsyncObserver, Maneuver):
            lines.append(f'## {owner.__name__}')
            for attr, member in inspect.getmembers(owner):
                if attr.startswith('_') or not callable(member) and not isinstance(member, property):
                    continue
                if inspect.isclass(member):
                    continue
                try:
                    signature = str(inspect.signature(member))
                except (TypeError, ValueError):
                    signature = ''
                doc = (inspect.getdoc(member) or '').split('\n\n')[0].replace('\n', ' ')
                lines.append(f'- {attr}{signature}: {doc}')
        lines.append("\nCall api_docs(name) for one method's full docstring, or "
                     "api_docs('maneuver_readme') for the maneuver guide with examples.")
        return '\n'.join(lines)

    # -- the MCP server -------------------------------------------------------------

    def _build_server(self):
        from mcp.server.mcpserver import Image, MCPServer
        from mcp.server.mcpserver.exceptions import ToolError
        bridge = self
        ob = self.ob
        server = MCPServer('stringman', instructions=INSTRUCTIONS, log_level='WARNING')

        def tool(fn):
            """server.tool(), with any exception passed to the agent as a ToolError: the SDK
            reports anything else as a bare 'Error executing tool', dropping the message."""
            def as_tool_error(e):
                return ToolError(f'{type(e).__name__}: {e}')
            if inspect.iscoroutinefunction(fn):
                @wraps(fn)
                async def wrapper(*args, **kwargs):
                    try:
                        return await fn(*args, **kwargs)
                    except ToolError:
                        raise
                    except Exception as e:
                        raise as_tool_error(e) from e
            else:
                @wraps(fn)
                def wrapper(*args, **kwargs):
                    try:
                        return fn(*args, **kwargs)
                    except ToolError:
                        raise
                    except Exception as e:
                        raise as_tool_error(e) from e
            return server.tool()(wrapper)

        def image(rgb, max_px, max_enlarge=1.0):
            return Image(data=_jpeg(rgb, max_px, max_enlarge=max_enlarge), format='jpeg')

        @tool
        def guide() -> str:
            """How to use this server: the experimental loop, safety, and writing maneuvers.
            Read it before anything else."""
            return GUIDE_PATH.read_text()

        @tool
        def get_state() -> dict:
            """A snapshot of the robot: positions (metres, room frame, z up, floor z=0), line
            tensions, gripper sensors, connections, the running motion task and how the last
            one ended, named positions, targets, popups awaiting an answer, video stream URLs,
            and next_event_seq for get_events."""
            return bridge.state()

        @tool
        async def get_image(camera: str = 'gripper', fresh: bool = False, max_px: int = 512):
            """One camera frame as JPEG. camera is gripper, anchor0-anchor3 or ortho (the
            floor seen from above, stitched from the anchor cameras). fresh=True waits for a
            frame captured after this call, so it shows the result of a move that just ended.
            max_px bounds the longer side; 512 costs a few hundred tokens."""
            ts, frame = await bridge.capture(camera, fresh=fresh)
            caption = f'{camera} {frame.shape[1]}x{frame.shape[0]}, captured {time.time() - ts:.2f} s ago'
            return [caption, await asyncio.to_thread(image, frame, max(64, min(max_px, 1600)))]

        @tool
        async def get_images(views: list[View], fresh: bool = False):
            """Several camera frames in one call, each optionally cropped, e.g. the gripper view
            beside a close-up of one corner of the ortho floor view. Frames are captured
            concurrently. A crop is [left, top, right, bottom] as fractions of the frame and is
            enlarged up to 4x toward max_px, so a small region comes back legible. On the
            1000 px ortho, room point (x, y) is at fraction (0.5 + x/5, 0.5 - y/5). fresh=True
            waits for frames captured after the call. A camera with no frame is reported in
            its caption rather than failing the others."""
            if not views:
                raise ValueError('give at least one view')
            if len(views) > 8:
                raise ValueError('at most 8 views per call')

            async def one(view):
                try:
                    ts, frame = await bridge.capture(view.camera, fresh=fresh)
                    part = crop_frame(frame, view.crop)
                except Exception as e:
                    return [f'{view.camera}: {type(e).__name__}: {e}']
                where = '' if view.crop is None else f' crop {[round(c, 3) for c in view.crop]}'
                caption = (f'{view.camera} {frame.shape[1]}x{frame.shape[0]}{where} '
                           f'({part.shape[1]}x{part.shape[0]} px), captured {time.time() - ts:.2f} s ago')
                enlarge = MAX_CROP_ENLARGE if view.crop is not None else 1.0
                return [caption, await asyncio.to_thread(
                    image, part, max(64, min(view.max_px, 1600)), enlarge)]

            blocks = []
            for result in await asyncio.gather(*(one(v) for v in views)):
                blocks.extend(result)
            return blocks

        @tool
        async def get_frames(camera: str = 'gripper', count: int = 6, interval_s: float = 0.3,
                             max_px: int = 320):
            """A short sequence from one camera as a single labelled grid image, to see motion
            such as a swing settling or fingers closing. Each frame is captured after the one
            before, interval_s apart."""
            count = max(2, min(count, 16))
            frames, labels = [], []
            t0, ts = None, time.time() - interval_s
            for i in range(count):
                ts, frame = await _capture_after(camera, ts + interval_s)
                t0 = t0 or ts
                frames.append(frame)
                labels.append(f'{i}  +{ts - t0:.2f}s')
            after = ts
            sheet = await asyncio.to_thread(contact_sheet, frames, labels, max(64, min(max_px, 800)))
            return [f'{camera}: {count} frames over {after - t0:.2f} s, left to right then down',
                    await asyncio.to_thread(image, sheet, 1600)]

        async def _capture_after(camera, after_ts):
            """A frame captured after after_ts. The floor view carries no capture time, so it
            is waited for by the clock instead."""
            if camera == 'ortho':
                await asyncio.sleep(max(0.0, after_ts - time.time()))
                return await bridge.capture(camera)
            if camera == 'gripper':
                ts, frame = await ob.gripper_capture(after=after_ts, timeout=3.0)
            else:
                ts, frame = await ob.anchor_capture(int(camera[6:]), after=after_ts, timeout=3.0)
            if frame is None:
                raise RuntimeError(f'{camera} stopped delivering frames')
            return ts, frame

        @tool
        async def record_clip(camera: str = 'gripper', seconds: float = 5.0, fps: float = 10.0):
            """Record seconds of one camera to an mp4 for a person to review, and return its
            path with a grid of six evenly spaced frames from it."""
            seconds = max(0.5, min(seconds, 60.0))
            frames, stamps = [], []
            end = time.time() + seconds
            after = time.time()
            while time.time() < end:
                ts, frame = await _capture_after(camera, after + 1.0 / fps)
                after = ts
                frames.append(frame)
                stamps.append(ts)
            path = bridge.output_dir('mcp') / f'{camera}_{int(time.time())}.mp4'

            def write():
                h, w = frames[0].shape[:2]
                writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                for f in frames:
                    if f.shape[:2] == (h, w):
                        writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
                writer.release()
            await asyncio.to_thread(write)
            picks = np.linspace(0, len(frames) - 1, min(6, len(frames))).round().astype(int)
            sheet = await asyncio.to_thread(
                contact_sheet, [frames[i] for i in picks],
                [f'+{stamps[i] - stamps[0]:.1f}s' for i in picks], 320)
            return [f'{len(frames)} frames written to {path.resolve()}',
                    await asyncio.to_thread(image, sheet, 1600)]

        @tool
        def get_events(since: int = 0, limit: int = 200) -> dict:
            """Events after sequence number `since`: motion_result (how each motion task
            ended, with a message, reason and any traceback), safety (over tension, marker
            faults), popup, progress, component (connections) and log (warnings, and INFO from
            agent code). Pass back next_event_seq to get only newer ones."""
            events, seq, dropped = bridge.events_since(since, max(1, min(limit, 1000)))
            return dict(events=events, next_event_seq=seq, older_events_dropped=dropped)

        @tool
        async def wait(until: str = 'motion_done', timeout: float = 30.0, since: int | None = None,
                       snapshot: str | None = None):
            """Block until the motion task ends (until='motion_done'), any event arrives
            ('event'), or simply for timeout seconds ('seconds'), at most 120. Returns the
            motion results and other events since `since`; by default, since just before the
            last tool call that started something, so a task that already ended is reported.
            Also returns the gantry position. snapshot names a camera to capture a fresh frame from
            once the wait is over."""
            if until not in ('motion_done', 'event', 'seconds'):
                raise ValueError("until must be 'motion_done', 'event' or 'seconds'")
            report = await bridge.wait(until, timeout, since)
            if snapshot:
                try:
                    _, frame = await bridge.capture(snapshot, fresh=True)
                    return [report, await asyncio.to_thread(image, frame, 512)]
                except Exception as e:
                    report['snapshot_error'] = str(e)
            return report

        @tool
        def capabilities() -> dict:
            """Every command, debug verb, control item and startup step the observer and its
            maneuvers answer to, with the maneuver that owns it and its docstring. Commands
            are run with run_command and verbs with run_verb."""
            return dict(entries=ob.maneuver_entries(), maneuvers=sorted(ob.maneuvers),
                        agent_modules=sorted(bridge.agent_code.modules),
                        cameras=list(CAMERAS), code_tools_enabled=bridge.allow_code)

        @tool
        def api_docs(name: str = '') -> str:
            """The public API maneuver code is written against: with no name, every method of
            AsyncObserver (reached as self.ob) and Maneuver with its first doc line; with a
            name, that method's full docstring. 'maneuver_readme' gives the maneuver guide."""
            return bridge.api_docs(name)

        @tool
        async def stop() -> dict:
            """Stop everything: cancels the motion task and swing cancellation and brings
            every spool to rest. The same as the stop button."""
            await ob.stop_all()
            return dict(stopped=True, last_motion_result=_result_dict(ob.last_motion_result()))

        @tool
        async def move_to(x: float, y: float, z: float, frame: str = 'gripper') -> dict:
            """Fly to a point in room coordinates (metres). frame='gripper' puts the gripper
            there; 'gantry' puts the gantry there. Starts a motion task and returns at once."""
            goal = np.array([x, y, z], dtype=float)
            if frame == 'gripper':
                goal = goal + ob.pole_offset()
            elif frame != 'gantry':
                raise ValueError("frame must be 'gripper' or 'gantry'")
            if not ob.inside_work_area_2d(goal):
                raise ValueError(f'{goal[:2].round(3).tolist()} is outside the work area')

            async def move_to():
                return await ob.seek_goal(goal)
            return await bridge.start_motion(move_to())

        @tool
        async def nudge(dx: float, dy: float, dz: float) -> dict:
            """A small eased move of the gantry by (dx, dy, dz) metres, then stop. Starts a
            motion task and returns at once."""
            delta = np.array([dx, dy, dz], dtype=float)
            if np.linalg.norm(delta) > 0.5:
                raise ValueError('a nudge is at most 0.5 m; use move_to for longer moves')

            async def nudge():
                return await ob.nudge_gantry(delta)
            return await bridge.start_motion(nudge())

        @tool
        async def drive(vx: float, vy: float, vz: float, seconds: float = 1.0) -> dict:
            """Hold a gantry velocity (m/s, room frame, z up) for at most 5 seconds, then stop.
            Bounded because a model's turn takes seconds; for steering by what a camera sees,
            write a maneuver instead. Starts a motion task and returns at once."""
            velocity = np.array([vx, vy, vz], dtype=float)
            limit = ob.speed_limit()
            if np.linalg.norm(velocity) > limit:
                velocity *= limit / np.linalg.norm(velocity)
            seconds = max(0.0, min(seconds, MAX_DRIVE_S))
            key = bridge.velocity_key('drive')

            async def drive():
                try:
                    end = time.monotonic() + seconds
                    while time.monotonic() < end:
                        await ob.move_direction_speed(velocity, key=key)
                        await asyncio.sleep(0.2)
                finally:
                    ob.slow_stop_all_spools()
            return await bridge.start_motion(drive())

        @tool
        async def set_fingers(angle: float) -> str:
            """Command the fingers to an angle in degrees, -90 open to 90 closed, without
            waiting. Use run_command('GRASP') for an automatic grasp."""
            await ob.set_finger_angle(float(np.clip(angle, -90, 90)))
            return 'sent'

        @tool
        async def set_wrist(angle: float) -> str:
            """Command the wrist to an absolute angle in degrees, 0 to 1080, without waiting."""
            await ob.set_wrist_angle(float(np.clip(angle, 0, 1080)))
            return 'sent'

        @tool
        async def run_command(name: str) -> dict:
            """Run a control.Command by name, as the UI's buttons do: e.g. PARK, UNPARK, GRASP,
            HALF_CAL, PICK_AND_DROP, TIGHTEN_LINES. See capabilities() for the full list.
            Motion commands start a motion task and return at once."""
            try:
                command = control.Command[name.upper().removeprefix('COMMAND_')]
            except KeyError:
                raise ValueError(f'no command {name!r}; one of '
                                 f'{[c.name for c in control.Command if c.name != "UNUSED"]}')
            if command in (control.Command.SHUTDOWN, control.Command.SAFE_COMPONENT_SHUTDOWN,
                           control.Command.UPDATE_FIRMWARE):
                raise ValueError(f'{command.name} is for a person to run from the UI')
            seq = bridge.seq
            status = await bridge.send_control(command=control.CommonCommand(name=command))
            return dict(status=status, motion_task=bridge.state()['motion_task'],
                        next_event_seq=seq)

        @tool
        async def run_verb(text: str) -> dict:
            """Run a debug verb, as typed in the UI's debug box: the first word picks the
            handler and the rest are its arguments, e.g. 'ferry hamper trash'. Verbs from
            maneuvers you load appear here. Motion verbs return at once."""
            seq = bridge.seq
            status = await bridge.send_control(debug=control.Debug(action=text))
            return dict(status=status, motion_task=bridge.state()['motion_task'],
                        next_event_seq=seq)

        @tool
        async def answer_popup(popup_id: int, button: int) -> str:
            """Answer a popup that is waiting on a person (see pending_popups in get_state) by
            the index of the button to press. Answer only what the user has told you to."""
            await bridge.send_control(popup_ack=control.PopupAck(id=popup_id, button=button))
            return 'answered'

        @tool
        async def load_maneuver(module_name: str = '', source: str = '', path: str = '') -> dict:
            """Load Python source defining one or more Maneuver subclasses into the running
            observer, replacing whatever an earlier load of module_name registered. Give the
            source inline, or the path of a .py file on the observer's machine to read it from;
            with a path, module_name defaults to the file's name. Iterate by editing the file
            and loading it again. The new source is imported before the old version is
            removed, so code that fails to import leaves the old one in place. Returns what
            the maneuvers answer to, or the error with its traceback. Requires
            --mcp_allow_code."""
            bridge.require_code()
            if bool(source) == bool(path):
                raise ValueError('give exactly one of source and path')
            if path:
                file = Path(path).expanduser()
                source = await asyncio.to_thread(file.read_text)
                module_name = module_name or file.stem
            if not module_name:
                raise ValueError('module_name is required with inline source')
            try:
                entry = await bridge.agent_code.load(module_name, source)
            except Exception as e:
                return dict(loaded=False, error=f'{type(e).__name__}: {e}',
                            traceback=traceback.format_exc()[-MAX_ERROR_CHARS:])
            entries = [e for e in ob.maneuver_entries() if e['owner'] in entry.maneuver_names]
            return dict(loaded=True, maneuvers=entry.maneuver_names, entries=entries)

        @tool
        async def unload_maneuver(module_name: str) -> str:
            """Remove the maneuvers a module registered, cancelling a motion task they own.
            Requires --mcp_allow_code."""
            bridge.require_code()
            await bridge.agent_code.unload(module_name)
            return f'unloaded {module_name}'

        @tool
        def save_maneuver(module_name: str) -> str:
            """Write a loaded module's source to the --maneuver_dir folder, where it is loaded
            on every start. Requires --mcp_allow_code."""
            bridge.require_code()
            return f'saved to {bridge.agent_code.save(module_name).resolve()}'

        @tool
        def get_maneuver_source(module_name: str) -> str:
            """The source last loaded for a module."""
            return bridge.agent_code.modules[module_name].source

        @tool
        async def run_script(source: str, motion: bool = True, timeout: float = 30.0) -> dict:
            """Run source defining `async def run(ob, m)`, where ob is the observer and m is a
            Maneuver for helpers like m.notify and m.velocity_key. Requires --mcp_allow_code.

            motion: set True whenever the script can command any motion at any point in
            its run: spools, gantry, wrist or fingers, directly or through anything it calls.
            It then runs as the motion task, where the stop button and the safety monitors can
            cancel it, and it replaces whatever motion task was running. This returns at once;
            wait() gives its MotionResult, whose value is what run returned.

            Set False only for a script that never moves anything, such as one reading getters
            or computing something. It runs beside the motion task without disturbing it, to
            completion or timeout seconds, and its return value comes back here. Nothing can
            stop it, so a motion=False script that moves the robot moves it beyond the reach of
            the stop button and the safety monitors."""
            bridge.require_code()
            try:
                run = bridge.agent_code.script(source)
            except Exception as e:
                return dict(started=False, error=f'{type(e).__name__}: {e}',
                            traceback=traceback.format_exc()[-MAX_ERROR_CHARS:])
            coro = run(ob, bridge)
            if motion:
                coro.__name__ = 'script'
                return await bridge.start_motion(coro)
            try:
                value = await asyncio.wait_for(coro, timeout=min(timeout, MAX_WAIT_S))
            except Exception as e:
                return dict(finished=False, error=f'{type(e).__name__}: {e}',
                            traceback=traceback.format_exc()[-MAX_ERROR_CHARS:])
            return dict(finished=True, value=_jsonable(value))

        return server

    def require_code(self):
        if not self.allow_code:
            raise PermissionError('code tools are off; the user can enable them by starting '
                                  'stringman-headless with --mcp_allow_code')

    async def _start_server(self):
        import uvicorn

        class Server(uvicorn.Server):
            @contextmanager
            def capture_signals(self):
                # the observer owns SIGINT; uvicorn's handler would replace it
                yield

        # bound here so a port in use is a logged error, not uvicorn's sys.exit
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((self.host, self.port))
        except OSError as e:
            sock.close()
            logger.error(f'MCP server could not listen on {self.host}:{self.port}: {e}')
            return
        app = self.mcp.streamable_http_app(host=self.host)
        config = uvicorn.Config(app, log_level='warning', lifespan='on')
        self._server = Server(config)

        async def serve():
            try:
                await self._server.serve(sockets=[sock])
            except SystemExit:
                logger.error('MCP server exited')
            except Exception:
                logger.exception('MCP server failed')

        self._server_task = asyncio.create_task(serve(), name='mcp_server')
        logger.info(f'MCP server at http://{self.host}:{self.port}/mcp')
