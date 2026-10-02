"""Cluster sort: sorting objects into bins by how they look.

Nothing says up front what the groups are. A run has two phases:

    collect  fly over every target the target model finds, without picking anything up,
             center over it with the visual servo model and take a picture at a fixed
             laser range. The pictures are clustered in the latent space of the visual
             servo network's frozen DINOv2 trunk, one cluster per card.
    sort     pick the targets up one by one and drop each at its cluster's card.

Which object the grasp model goes after at a target is not guaranteed to be the one
photographed there, so an object is sorted by the view the grasp itself had of it, matched
to the nearest collected picture. Usually that is the picture of the same object.
"""

import asyncio
import json
import logging
import time
from typing import NamedTuple

import cv2
import numpy as np

from nf_robot.common.util import tonp
from nf_robot.generated.nf import telemetry
from nf_robot.host.maneuver import Maneuver, prefer_swing_cancellation, verb

logger = logging.getLogger(__name__)

# Drop points, in the order bins are used: 'cluster 3' sorts into the first three.
BIN_NAMES = ('gamepad', 'trash', 'toys', 'hamper')
# Targets this close to any card, horizontally, are left alone, or the robot would pick
# back up what it just sorted.
CARD_EXCLUSION_M = 0.4
# A target this close to one already flown over in the collect phase is the same object.
VISITED_RADIUS_M = 0.15

# Framing a picture: centered by the servo model at this laser range, and held there.
FRAME_RANGE_M = 0.30
FRAME_RANGE_TOL_M = 0.02
FRAME_CENTER_TOL_M = 0.02
FRAME_SETTLE_S = 0.5
FRAME_TIMEOUT_S = 12.0
# vertical speed per metre of range error, and its limit
FRAME_RANGE_GAIN = 1.0
FRAME_MAX_VERTICAL_SPEED = 0.08
FRAME_POLL_S = 0.1

# How far the fingers may close past the most open they were during the grasp before a
# clear view stops counting as the item seen whole.
FINGER_CLOSE_TOLERANCE_DEG = 10.0
CLEAR_VIEW_POLL_S = 0.1


class NoServoModel(Exception):
    pass


class Shot(NamedTuple):
    image_rgb: np.ndarray
    laser_range: float
    position: np.ndarray  # the gripper's floor position when it was taken


def embed(trunk, image_rgb, device):
    """The L2-normalized [CLS] token of the frozen trunk for one gripper frame."""
    import torch

    from nf_robot.ml.image_input import input_batch
    from nf_robot.ml.visual_servoing.model import DEFAULT_IMAGE_SIZE

    with torch.no_grad():
        cls = trunk(input_batch(image_rgb, DEFAULT_IMAGE_SIZE, device)).last_hidden_state[0, 0]
    v = cls.float().cpu().numpy()
    return v / (np.linalg.norm(v) + 1e-8)


def cluster_bins(embeddings, k):
    """The bin of each embedding, from ward clustering into at most k groups. The largest
    group gets bin 0."""
    from scipy.cluster.hierarchy import fcluster, linkage

    n = len(embeddings)
    if n == 1:
        return [0]
    labels = fcluster(linkage(np.asarray(embeddings), 'ward'), min(k, n), 'maxclust')
    by_size = sorted(set(labels), key=lambda label: -np.sum(labels == label))
    return [by_size.index(label) for label in labels]


class ClusterSort(Maneuver):
    name = 'cluster_sort'
    title = 'Cluster sort'

    def __init__(self, ob):
        super().__init__(ob)
        self.trunk = None

    async def ensure_trunk(self):
        if self.trunk is None:
            def load():
                from nf_robot.ml.dino_trunk import shared_backbone
                from nf_robot.ml.visual_servoing.model import DEFAULT_BACKBONE
                return shared_backbone(DEFAULT_BACKBONE).to(self.ob.torch_device())
            self.progress(0, 'Loading image encoder')
            self.trunk = await self.run_in_thread(load)

    @staticmethod
    def _near(position, points, radius):
        return any(np.linalg.norm(np.asarray(position)[:2] - np.asarray(p)[:2]) < radius
                   for p in points)

    def _cards(self):
        # every card, in use or not, so nothing sorted earlier is picked back up
        return [p for p in map(self.ob.named_position, BIN_NAMES) if p is not None]

    async def _frame_shot(self):
        """Center over the object below and settle at FRAME_RANGE_M, then take a picture.

        servo_center steers sideways under the default velocity key; the height is held
        under this maneuver's own key, and the two are summed. Returns a Shot, or None if it
        did not settle within FRAME_TIMEOUT_S.
        """
        ob = self.ob
        key = self.velocity_key('frame')
        center = asyncio.create_task(ob.servo_center())
        settled_since = None
        deadline = time.time() + FRAME_TIMEOUT_S
        try:
            while time.time() < deadline:
                if center.done():
                    raise NoServoModel()
                laser = ob.laser_range()
                vz = 0.0 if laser is None else float(np.clip(
                    FRAME_RANGE_GAIN * (FRAME_RANGE_M - laser),
                    -FRAME_MAX_VERTICAL_SPEED, FRAME_MAX_VERTICAL_SPEED))
                await ob.move_direction_speed([0.0, 0.0, vz], downward_bias=0.0, key=key)

                offset = ob.servo_center_offset()
                if (laser is not None and abs(laser - FRAME_RANGE_M) < FRAME_RANGE_TOL_M
                        and offset is not None and offset < FRAME_CENTER_TOL_M):
                    settled_since = settled_since or time.time()
                    if time.time() - settled_since >= FRAME_SETTLE_S:
                        image = await ob.gripper_frame()
                        if image is not None:
                            return Shot(image, laser, ob.gripper_position())
                else:
                    settled_since = None
                await asyncio.sleep(FRAME_POLL_S)
            return None
        finally:
            await ob.move_direction_speed([0.0, 0.0, 0.0], key=key)
            center.cancel()
            await asyncio.gather(center, return_exceptions=True)

    async def _collect(self, run_dir, device):
        """Fly over every target without picking anything up and photograph it. Returns the
        seed list of (Shot, embedding, image name)."""
        ob = self.ob
        ppc = ob.config.pick_and_place
        queue = ob.maneuvers['pick_and_place'].target_queue
        hover_z = tonp(ppc.gantry_height_over_target)[2]
        seeds, visited = [], []
        target_seen_t = time.time()
        while True:
            cards = self._cards()
            target = queue.get_best_target(accept=lambda t: not (
                self._near(t.position, cards, CARD_EXCLUSION_M)
                or self._near(t.position, visited, VISITED_RADIUS_M)))
            if target is None:
                await ob.clear_goal()
                if time.time() > target_seen_t + ppc.end_loop_timeout:
                    return seeds
                await asyncio.sleep(ppc.loop_delay)
                continue
            target_seen_t = time.time()
            # re-chooses each second, as the queue may have changed
            if not await ob.seek_goal(target.position + np.array([0, 0, hover_z]), timeout=1):
                continue
            visited.append(target.position)
            shot = await self._frame_shot()
            if shot is None:
                logger.info(f'Could not frame the object at {target.position[:2].round(2)}; skipping it')
                continue
            visited.append(shot.position)
            name = f'seed_{len(seeds):04d}.jpg'
            cv2.imwrite(str(run_dir / name), cv2.cvtColor(shot.image_rgb, cv2.COLOR_RGB2BGR))
            seeds.append((shot, await self.run_in_thread(embed, self.trunk, shot.image_rgb, device), name))
            self.progress(0, f'Photographed {len(seeds)} objects')

    async def _grasp_watching(self):
        """Grasp, keeping the clear view of the item nearest FRAME_RANGE_M from before the
        fingers closed.

        last_clear_item_image also fills during the lift, with the item in the fingers, so
        a view counts only while the fingers are still about as open as they got.
        Returns (grasped, ItemImage or None).
        """
        ob = self.ob
        started = time.time()
        grasp = asyncio.create_task(ob.grasp())
        most_open = ob.finger_angle()
        best = None
        try:
            while not grasp.done():
                angle = ob.finger_angle()
                most_open = min(most_open, angle)
                view = ob.last_clear_item_image()
                if (angle <= most_open + FINGER_CLOSE_TOLERANCE_DEG and view is not None
                        and view.timestamp >= started
                        and (best is None or abs(view.laser_range - FRAME_RANGE_M)
                             < abs(best.laser_range - FRAME_RANGE_M))):
                    best = view
                await asyncio.wait([grasp], timeout=CLEAR_VIEW_POLL_S)
        finally:
            if not grasp.done():
                grasp.cancel()
                # the grasp stops the spools on its way out; wait for that before moving on
                await asyncio.gather(grasp, return_exceptions=True)
        return grasp.result(), best

    @verb('cluster', motion=True)
    @prefer_swing_cancellation
    async def cluster_sort(self, k='4'):
        """Photograph what the target model finds, cluster the pictures, then pick it all up
        and sort it among the cards. 'cluster' sorts into all four cards, 'cluster K' into
        the first K."""
        ob = self.ob
        k = int(k) if k.isdigit() else 0
        if not 1 <= k <= len(BIN_NAMES):
            self.notify(f'cluster takes 1 to {len(BIN_NAMES)} bins')
            return
        bins = BIN_NAMES[:k]
        missing = [b for b in bins if ob.named_position(b) is None]
        if missing:
            self.notify(f'No position yet for the {", ".join(missing)} card(s); show them to a camera first')
            return

        pnp = ob.maneuvers['pick_and_place']
        if pnp.target_model is None:
            await pnp.load_target_model()
            if pnp.target_model is None:
                return
        await self.ensure_trunk()
        device = await self.run_in_thread(ob.torch_device)

        run_dir = self.output_dir(f'cluster_sort/{time.strftime("%Y%m%d-%H%M%S")}')
        logger.info(f'Cluster sorting into {", ".join(bins)}; pictures in {run_dir}')
        try:
            try:
                seeds = await self._collect(run_dir, device)
            except NoServoModel:
                self.notify('Cluster sort needs the visual servoing model to frame its pictures')
                return
            if not seeds:
                self.notify('Found nothing to photograph')
                return
            seed_embeddings = np.stack([e for _, e, _ in seeds])
            seed_bins = await self.run_in_thread(cluster_bins, seed_embeddings, k)
            collection = {
                'bins': list(bins),
                'seeds': [{
                    'image': name,
                    'position': [round(float(c), 4) for c in shot.position[:2]],
                    'laser_range': round(shot.laser_range, 4),
                    'bin': bins[b],
                } for (shot, _, name), b in zip(seeds, seed_bins)],
                'sorted': [],
            }
            np.save(run_dir / 'seed_embeddings.npy', seed_embeddings)
            self._save_collection(run_dir, collection)
            sizes = {name: seed_bins.count(b) for b, name in enumerate(bins)}
            logger.info(f'Clustered {len(seeds)} pictures: {sizes}')
            self.notify(f'Clustered {len(seeds)} objects: '
                        + ', '.join(f'{n} for {name}' for name, n in sizes.items()))

            await self._sort(run_dir, device, bins, seeds, seed_embeddings, seed_bins, collection)
        except asyncio.CancelledError:
            logger.info('Cluster sort cancelled')
            raise
        finally:
            self.finish()
            ob.slow_stop_all_spools()
            await ob.clear_goal()

    async def _sort(self, run_dir, device, bins, seeds, seed_embeddings, seed_bins, collection):
        """Pick up targets until none are left, dropping each at the card of the collected
        picture its grasp-time view is nearest to."""
        ob = self.ob
        ppc = ob.config.pick_and_place
        pnp = ob.maneuvers['pick_and_place']
        GANTRY_HEIGHT_OVER_TARGET = tonp(ppc.gantry_height_over_target)
        GANTRY_HEIGHT_OVER_DROPOFF = tonp(ppc.gantry_height_over_dropoff)
        drop_point = None
        target_seen_t = time.time()
        while True:
            cards = self._cards()
            target = pnp.target_queue.get_best_target(
                accept=lambda t: not self._near(t.position, cards, CARD_EXCLUSION_M))
            if target is None:
                await ob.clear_goal()
                if time.time() > target_seen_t + ppc.end_loop_timeout:
                    self.notify(f'Looks clean enough. Sorted {len(collection["sorted"])} objects.')
                    return
                await asyncio.sleep(ppc.loop_delay)
                continue
            target_seen_t = time.time()
            pnp.target_queue.set_target_status(target.id, telemetry.TargetStatus.SELECTED)
            pnp.send_tq_to_ui()

            # leaving a card, stay at this height until clear of it
            here = ob.gantry_position()
            if drop_point is not None and np.linalg.norm(here - (drop_point + GANTRY_HEIGHT_OVER_DROPOFF)) < 0.5:
                z_pos = here[2]
            else:
                z_pos = GANTRY_HEIGHT_OVER_TARGET[2]
            # re-chooses the target each second, in case a better one appeared
            if not await ob.seek_goal(target.position + np.array([0, 0, z_pos]), timeout=1):
                pnp.target_queue.set_target_status(target.id, telemetry.TargetStatus.SEEN)
                continue

            if not ob.gripper_connected():
                logger.warning('Cluster sort aborted because we lost the gripper connection')
                return

            grasped, seen = await self._grasp_watching()
            if not grasped:
                pnp.target_queue.set_target_status(target.id, telemetry.TargetStatus.SEEN)
                pnp.send_tq_to_ui()
                await asyncio.sleep(ppc.loop_delay)
                continue
            pnp.target_queue.set_target_status(target.id, telemetry.TargetStatus.PICKED_UP)
            pnp.send_tq_to_ui()

            record = {'pickup_position': [round(float(c), 4) for c in target.position[:2]]}
            if seen is not None:
                e = await self.run_in_thread(embed, self.trunk, seen.image_rgb, device)
                nearest = int(np.argmax(seed_embeddings @ e))
                record['image'] = f'sorted_{len(collection["sorted"]):04d}.jpg'
                record['laser_range'] = round(float(seen.laser_range), 4)
                cv2.imwrite(str(run_dir / record['image']), cv2.cvtColor(seen.image_rgb, cv2.COLOR_RGB2BGR))
            else:
                # no view to go by, so trust that the object is the one photographed nearest
                # to where it was picked up
                logger.warning('No clear view of the object before the grasp; sorting it by where it was')
                nearest = int(np.argmin([np.linalg.norm(s.position[:2] - target.position[:2])
                                         for s, _, _ in seeds]))
            b = seed_bins[nearest]
            record.update(nearest_seed=seeds[nearest][2], bin=bins[b])
            collection['sorted'].append(record)
            self._save_collection(run_dir, collection)
            self.progress(0, f'{len(collection["sorted"])} sorted, last into {bins[b]}')
            logger.info(f'Object looks most like {seeds[nearest][2]}; it goes to the {bins[b]} card')

            drop_point = ob.named_position(bins[b])
            await ob.seek_goal(drop_point + GANTRY_HEIGHT_OVER_DROPOFF)
            asyncio.create_task(ob.set_finger_angle(
                max(-90, min(ppc.relaxed_open, ob.finger_angle() - 10))))
            # don't immediately select a new target, because it may be what was just let go
            await asyncio.sleep(ppc.delay_after_drop)
            pnp.target_queue.set_target_status(target.id, telemetry.TargetStatus.DROPPED)
            pnp.send_tq_to_ui()

    def _save_collection(self, run_dir, collection):
        """Rewrite the run's collection.json, so a run cut short still has it."""
        with open(run_dir / 'collection.json', 'w') as f:
            json.dump(collection, f, indent=2)
