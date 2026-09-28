"""Pick and place: finding targets on the floor, and carrying them along the route."""

import asyncio
import logging
import time
from functools import partial

import numpy as np

from nf_robot.common.model_revisions import pinned_revision
from nf_robot.common.util import tonp
from nf_robot.generated.nf import common, control, telemetry
from nf_robot.host.fake_progress import FakeProgress
from nf_robot.host.maneuver import (ROUTE_POINT_TAG_NAMES, Maneuver, command, control_item,
                                    prefer_swing_cancellation, startup_step)
from nf_robot.host.target_queue import TargetQueue

logger = logging.getLogger(__name__)

# where Go Here puts the gripper above the point it was sent to
GRIPPER_HEIGHT_OVER_TARGET = np.array([0, 0, 0.3])
# (seconds) quiet time after the last manual target add, move or delete before the targets
# are submitted to the ortho target dataset on their own
AUTO_SUBMIT_TARGETS_S = 5.0
# The prediction is a point on the floor, so a drop aims this far above it. A tall hamper
# needs the clearance, and the model does not predict a height yet.
PREDICTED_DROP_HEIGHT_M = 0.0
FIND_TARGETS_INTERVAL_S = 0.5
# targets this close to the route destination, horizontally, are left where they are
DROPOFF_EXCLUSION_M = 0.10


class PickAndPlace(Maneuver):
    name = 'pick_and_place'
    title = 'Pick and Place'

    def __init__(self, ob):
        super().__init__(ob)
        self.target_queue = TargetQueue()
        self.target_model = None
        self.last_snapshot_hash = None # to spare the UI from too many updates
        # pending auto-submit of manually edited targets; see _schedule_target_submit
        self._target_submit_task = None

    async def start(self):
        self.spawn(self._find_targets_forever(), name='find_targets')

    def send_setup_telemetry(self):
        self.ob.send_ui(auto_targeting_state=telemetry.AutoTargetingState(
            enabled=self.target_model is not None, present=True))
        # a UI that connects later has seen none of the targets
        self.last_snapshot_hash = None
        self.send_tq_to_ui()

    def send_tq_to_ui(self):
        snapshot = self.target_queue.get_queue_snapshot()
        # Create a deterministic hash
        current_hash = hash(bytes(snapshot))
        if current_hash != self.last_snapshot_hash:
            self.ob.send_ui(target_list=snapshot)
            self.last_snapshot_hash = current_hash

    # -- targets placed by hand ------------------------------------------------

    @control_item('delete_target')
    async def delete_target(self, item: control.DeleteTarget):
        if item.clear_all:
            self.target_queue.remove_all_targets()
            self._schedule_target_submit()
        elif item.target_id is not None:
            self.target_queue.remove_target(item.target_id);
            self._schedule_target_submit()
        self.send_tq_to_ui()
        await self.ob.flush_tele_buffer()

    @control_item('add_cam_target')
    def add_cam_target(self, item: control.AddTargetFromAnchorCam):
        floor_point = self.ob.anchor_pixel_to_floor(item.anchor_num, (item.img_norm_x, item.img_norm_y))
        logger.info(f'Adding target at floor point ({floor_point}) from image point '
                    f'({item.img_norm_x}, {item.img_norm_y}) in anchor cam {item.anchor_num}')
        if floor_point is not None:
            if item.target_id is not None:
                self.target_queue.set_target_position(item.target_id, floor_point)
            else:
                self.target_queue.add_user_target(floor_point, dropoff='hamper')
            self._schedule_target_submit()
        self.send_tq_to_ui()

    @control_item('add_room_target')
    def add_room_target(self, item: control.AddTargetInRoom):
        # Used when the position arrives already in room coordinates.
        logger.info(f'Adding target at floor point ({item.x}, {item.y}) from the 3d view')
        self.target_queue.add_user_target((item.x, item.y), dropoff='hamper')
        self._schedule_target_submit()
        self.send_tq_to_ui()

    def _schedule_target_submit(self):
        """Submit the targets to the dataset once manual edits stop for AUTO_SUBMIT_TARGETS_S.

        Each add, move or delete restarts the wait, so a burst of edits becomes one row
        carrying the finished set rather than one per click.
        """
        if self._target_submit_task is not None:
            self._target_submit_task.cancel()

        async def submit_when_quiet():
            await asyncio.sleep(AUTO_SUBMIT_TARGETS_S)
            # past the wait, so a later edit schedules a new submit instead of cancelling this one
            self._target_submit_task = None
            await self.submit_targets_to_dataset()

        self._target_submit_task = self.spawn(submit_when_quiet(), name='submit_targets')

    @command(control.Command.SUBMIT_TARGETS_TO_DATASET)
    async def submit_targets_to_dataset(self):
        """Save the ortho floor view and every user-placed target as ortho_target rows.

        The ortho target model's labels normally come from teleop: wherever an operator's
        grasp landed is by construction a place worth reaching for. A target placed by hand
        in the UI says the same thing about the same projection without anyone having to
        drive there, so it goes to the same row format - into ortho_target.USER_LABEL_ROOT,
        which nothing trains on until someone decides these labels are worth merging.

        One row per user target, all carrying the frame they were placed on; AI targets are
        the model's own output and would only teach it what it already believes.
        """
        frame = self.ob.latest_ortho()
        if frame is None:
            logger.warning('No orthographic floor view to label yet'
                           + ('; --no_ortho disables it' if not self.ob.ortho_enabled() else ''))
            return
        # the ortho worker replaces this on every anchor frame, so take the pixels now
        frame = frame.copy()

        targets = [t.position for t in self.target_queue.get_user_targets()]
        if not targets:
            logger.warning('No user-placed targets to save; click the floor to add one first')
            return

        def save():
            # imported in the thread: it pulls torch, and this is the only thing on the
            # host that wants it before a target model is loaded
            from nf_robot.ml.ortho_target.dataset import write_user_labels
            return write_user_labels(frame, targets)

        try:
            path, written = await self.run_in_thread(save)
        except Exception:
            logger.exception('Could not save targets to the ortho target dataset')
            return

        if written:
            logger.info(f'Saved {written} target label(s) on the ortho frame to {path}')
        else:
            logger.warning(f'None of the {len(targets)} targets are inside the ortho map; nothing saved')

    @control_item('move_gripper_to')
    async def go_here(self, item: control.MoveGripperTo):
        """Handle the Go Here command"""
        goal_pos = None
        if item.target_id is not None:
            # derive target position from target
            target = self.target_queue.get_target_info(item.target_id)
            if target is not None:
                goal_pos = tonp(target.position) + GRIPPER_HEIGHT_OVER_TARGET + self.ob.pole_offset()
        elif item.pos is not None:
            goal_pos = tonp(item.pos) + GRIPPER_HEIGHT_OVER_TARGET + self.ob.pole_offset()

        if goal_pos is None:
            return
        r = await self.ob.invoke_motion_task(self.seek_goal_where_asked(goal_pos),
                                             owner=self, safety=self.safety)

    @prefer_swing_cancellation
    async def seek_goal_where_asked(self, goal_pos):
        """seek_goal but only when someone calls it with go here"""
        return await self.ob.seek_goal(goal_pos)

    # -- targets found by the model ------------------------------------------

    def _set_target_model(self, model):
        """Set self.target_model, notifying the UI via auto_targeting_state whenever
        whether a model is loaded (not the model itself) changes."""
        was_loaded = self.target_model is not None
        self.target_model = model
        if (model is not None) != was_loaded:
            self.ob.send_ui(auto_targeting_state=telemetry.AutoTargetingState(enabled=model is not None, present=True))

    async def load_target_model(self):
        """Load the ortho target model and make it the active one."""
        device = await self.run_in_thread(self.ob.torch_device)

        if device == "cpu":
            logger.warning("Refusing to load targeting model on CPU; hardware acceleration required.")
            self._set_target_model(None)
            self.notify("Automatic target identification (targeting model) cannot be used without "
                        "some kind of hardware acceleration. Loading was aborted because the torch "
                        "device is CPU.")
            return

        def load_sync():
            from huggingface_hub import hf_hub_download
            from nf_robot.ml.ortho_target import model as ortho_target
            filename = ortho_target.TARGETING_MODEL_FILENAME
            repo_id = ortho_target.TARGETING_MODEL_REPOID
            path = (f"models/{filename}" if self.ob.local_models
                    else hf_hub_download(repo_id=repo_id, filename=filename,
                                         revision=pinned_revision(repo_id)))
            logger.info(f"Loading ortho target model from {path}...")
            model, _ = ortho_target.load_checkpoint(path, device)
            return model

        # The checkpoint fetch and torch import inside load_sync report nothing, so the
        # bar is on a 5s timer; it holds at 99% for as long as the load actually takes.
        async with FakeProgress(
            self.ob.send_ui,
            name="Target Model",
            current_action="Loading target model...",
            done_action="Target model ready",
            failed_action="Could not load the target model",
            expected_s=5.0,
            interval_s=0.2,
            suppress_completion_popup=True,
        ):
            model = await self.run_in_thread(load_sync)
        self._set_target_model(model)

    @control_item('set_target_model')
    def set_target_model(self, item: control.SetTargetModel):
        # ortho_target is the only target model; every enable action loads it. The enum
        # still carries the retired per-model choices, which are all treated as the default.
        if item.action == control.TargetModelAction.TARGET_MODEL_DISABLE:
            self._set_target_model(None)
            logger.info('Target model disabled')
        elif item.action != control.TargetModelAction.TARGET_MODEL_ACTION_UNUSED:
            # in the background: a load takes seconds, and every command behind this one
            # would wait for it
            async def load():
                logger.info('Loading target model...')
                await self.load_target_model()
                logger.info('Target model ready')
            self.spawn(load(), name='load_target_model')

    async def _find_targets_forever(self):
        """Run the target model on the floor view while one is loaded, keeping the AI targets
        in the queue up to date. It reads the floor projection the ortho worker renders, so
        run_ortho must be on for it to see anything."""
        while True:
            await asyncio.sleep(FIND_TARGETS_INTERVAL_S)
            if self.target_model is None:
                continue
            floor_targets = await self._find_targets_ortho()
            # None means "no opinion this round" (no input frame yet), which must not be
            # confused with the empty list, which retires every AI target in the queue.
            if floor_targets is None:
                continue
            floor_targets = self._reject_targets_at_dropoff(floor_targets)
            self.target_queue.add_ai_targets(floor_targets)
            self.send_tq_to_ui()

    def _reject_targets_at_dropoff(self, targets):
        """Drop targets sitting on the route destination, whatever model proposed them.

        This is to prevent the robot from repeatedly picking and dropping the same thing forver.
        """
        dst = self.ob.route_point_position(self.ob.route()[1])
        if dst is None:
            return targets
        kept = []
        for t in targets:
            # Horizontal distance only
            if np.linalg.norm(np.asarray(t['position'])[:2] - dst[:2]) < DROPOFF_EXCLUSION_M:
                logger.debug(f'discarding target at {t["position"]}, inside the dropoff exclusion')
                continue
            kept.append(t)
        return kept

    def _floor_target(self, x, y):
        """A target dict for the queue, or None if it lies outside the work area."""
        position = np.array([x, y, 0])
        if not self.ob.inside_work_area_2d(position):
            return None
        return {'position': position, 'dropoff': 'hamper'}

    async def _find_targets_ortho(self):
        """Every confident target in the ortho floor view, per the ortho_target model.

        The model reads the same projection the ortho worker already renders, so nothing
        per-camera is inferred and no warping is needed.
        """
        from nf_robot.ml.ortho_target import model as ortho_target

        # Each cell carries its own objectness, decided without reference to the rest of
        # the map, so one absolute bar holds on a bare floor and a crowded one alike and a
        # second object does not dilute the first. The bar comes from the checkpoint, which
        # records the operating point its training run scored best at: what counts as
        # confident depends on the pos_weight it trained under, so a constant here would be
        # right for one model and wrong for the next. The fallback is for checkpoints from
        # before the threshold was swept.
        ORTHO_MIN_PROBABILITY = getattr(self.target_model, 'threshold', 0.5)
        ORTHO_MAX_CANDIDATES = 16  # NMS peaks to consider before thresholding

        ortho_frame = self.ob.latest_ortho()
        if ortho_frame is None:
            if not self.ob.ortho_enabled():
                logger.warning('ortho target model needs the floor projection, which run_ortho disables')
            return None

        predictions = await self.run_in_thread(
            partial(ortho_target.predict_room_targets, self.target_model, ortho_frame,
                    self.ob.torch_device(), top_k=ORTHO_MAX_CANDIDATES,
                    min_probability=ORTHO_MIN_PROBABILITY),
        )
        targets = [self._floor_target(x, y) for x, y, _ in predictions]
        return [t for t in targets if t is not None]

    # -- the loop ----------------------------------------------------------------

    @startup_step('pick_and_place')
    async def work_until_clean(self):
        """Pick and place until no targets have appeared for a while, with the target model
        doing the finding."""
        # enable auto target selection
        await self.load_target_model()
        await self.pick_and_place_loop()

    @command(control.Command.PICK_AND_DROP, motion=True)
    @prefer_swing_cancellation
    async def pick_and_place_loop(self):
        """
        Long running motion task that repeatedly identifies targets picks them up and drops them over the hamper
        """
        ob = self.ob
        ppc = ob.config.pick_and_place
        GANTRY_HEIGHT_OVER_TARGET = tonp(ppc.gantry_height_over_target)
        GANTRY_HEIGHT_OVER_DROPOFF = tonp(ppc.gantry_height_over_dropoff)
        RELAXED_OPEN = ppc.relaxed_open # Open enough to drop and that fingers cannot be seen in frame
        DELAY_AFTER_DROP = ppc.delay_after_drop # long enough that the payload is not visible anymore in the hand
        LOOP_DELAY = ppc.loop_delay
        END_LOOP_TIMEOUT = ppc.end_loop_timeout

        # Where each item goes is predicted while it is being picked up, so the model is
        # loaded here rather than at startup, and the prediction runs for as long as this
        # loop does. A destination of PREDICTED_DROP is what acts on it; every other
        # destination just gets the marker to look at.
        drop_point_maneuver = ob.maneuvers.get('drop_point')
        if drop_point_maneuver is not None and await drop_point_maneuver.ensure_model():
            drop_point_maneuver.start_watch()

        # Only --lerobot_grasp needs a session; the default servoing grasp does not, and
        # grasp falls back to it anyway, so there is nothing to prompt about.
        lerobot = ob.maneuvers.get('lerobot')
        if lerobot is not None and lerobot.use_for_grasp and not await lerobot.session_connected():
            answer = await self.ask(
                "--lerobot_grasp is set but no session is connected. Start a subprocess of "
                "stringman-headless to run the grasping model? Answering No grasps with the "
                "visual servoing model instead.",
                buttons=["Yes", "No"],
            )
            if answer == 0:
                lerobot.start_eval_session("naavox/dit-grasp-3")

        drop_point = np.zeros(3)
        target_seen_t = time.time()
        try:
            while True:
                src, dst = ob.route()

                if src in (common.RoutePoint.ALL_TARGETS, common.RoutePoint.USER_TARGETS):
                    next_target = self.target_queue.get_best_target()
                    if next_target is None:
                        await ob.clear_goal()
                        if time.time() > target_seen_t + END_LOOP_TIMEOUT:
                            logger.info('Looks clean enough to me!')
                            return
                        await asyncio.sleep(LOOP_DELAY)
                        continue
                    target_seen_t = time.time()

                    self.target_queue.set_target_status(next_target.id, telemetry.TargetStatus.SELECTED)
                    self.send_tq_to_ui()

                    # pick Z position for gantry
                    # if we are too close to the drop point right now, the z position has to be our current z so we don't get hung up on the basket by going down too soon.
                    # otherwise use the normal value
                    here = ob.gantry_position()
                    if np.linalg.norm(here - (drop_point + GANTRY_HEIGHT_OVER_DROPOFF[2])) < 0.5:
                        z_pos = here[2]
                    else:
                        z_pos = GANTRY_HEIGHT_OVER_TARGET[2]
                    goal_pos = next_target.position + np.array([0, 0, z_pos])

                elif src in ROUTE_POINT_TAG_NAMES or src == common.RoutePoint.ORIGIN:
                    next_target = None
                    source = ob.route_point_position(src)
                    if source is None:
                        logger.warning(f'No saved position for the route source '
                                       f'{ROUTE_POINT_TAG_NAMES[src]}; nothing to pick up from')
                        return
                    goal_pos = source + GANTRY_HEIGHT_OVER_TARGET
                else:
                    logger.warning(f'Pick and place cannot pick up from {src!r}')
                    return

                # re-aims the seek already in flight onto the newly chosen target. If it does
                # not arrive in one second, run target selection again since a better one might
                # have appeared or the user might have put one in their queue
                if not await ob.seek_goal(goal_pos, timeout=1):
                    if next_target is not None:
                        self.target_queue.set_target_status(next_target.id, telemetry.TargetStatus.SEEN)
                    continue

                if not ob.gripper_connected():
                    logger.warning('Pick and place aborted because we lost the gripper connection')
                    break

                # when we reach this point we arrived over the item. commit to it unless it proves impossible to pick up.
                logger.info('Attempt grasp')
                start = time.time()
                success = await ob.grasp()
                logger.info(f'Grasp succeeded={success} took {time.time() - start:.2f}s')
                if not success:
                    if next_target is not None:
                        # just pick another target, but consider downranking this object or something.
                        self.target_queue.set_target_status(next_target.id, telemetry.TargetStatus.SEEN)
                        self.send_tq_to_ui()
                    await asyncio.sleep(LOOP_DELAY)
                    continue
                else:
                    if next_target is not None:
                        self.target_queue.set_target_status(next_target.id, telemetry.TargetStatus.PICKED_UP)
                        self.send_tq_to_ui()
                    logger.info('Object picked up')

                # Choose drop point. default to origin
                drop_point = np.zeros(3)

                if dst == common.RoutePoint.NA and next_target is not None:
                    # read drop point from target
                    # TODO currently these are not populated with useful data.
                    if not isinstance(next_target.dropoff, str):
                        drop_point = next_target.dropoff
                    # otherwise go to the named drop point
                    elif ob.named_position(next_target.dropoff) is not None:
                        drop_point = ob.named_position(next_target.dropoff)

                elif dst in ROUTE_POINT_TAG_NAMES:
                    # Typical path. A destination whose tag has never been seen, or a drop
                    # position never recorded, has nothing saved to fly to; say so rather than
                    # raising out of the middle of a pick.
                    saved = ob.route_point_position(dst)
                    if saved is None:
                        logger.warning(f'No saved position for the route destination '
                                       f'{ROUTE_POINT_TAG_NAMES[dst]}; dropping at the origin')
                    else:
                        drop_point = saved
                        if dst == common.RoutePoint.PREDICTED_DROP:
                            # A prediction is a point on the floor; let go above it.
                            drop_point = drop_point + np.array([0, 0, PREDICTED_DROP_HEIGHT_M])
                elif dst == common.RoutePoint.ORIGIN:
                    drop_point = np.zeros(3)

                # fly to to drop point
                logger.info(f'Flying to drop point {drop_point}')
                await ob.seek_goal(drop_point + GANTRY_HEIGHT_OVER_DROPOFF)
                # open gripper
                open_target = max(-90, min(RELAXED_OPEN, ob.finger_angle() - 10))
                asyncio.create_task(ob.set_finger_angle(open_target))
                if next_target is not None:
                    # don't immediately select a new target, because there's a chance it'll be the sock you're holding.
                    await asyncio.sleep(DELAY_AFTER_DROP)
                    self.target_queue.set_target_status(next_target.id, telemetry.TargetStatus.DROPPED)
                    self.send_tq_to_ui()

        except asyncio.CancelledError:
            logger.info('Pick and place cancelled')
            raise
        finally:
            if drop_point_maneuver is not None:
                drop_point_maneuver.stop_watch()
            ob.slow_stop_all_spools()
            await ob.clear_goal()
