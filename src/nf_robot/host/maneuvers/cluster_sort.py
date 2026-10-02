"""Cluster sort: picking up objects and sorting them into bins by how they look.

Nothing says up front what the groups are. Each object picked up adds the gripper camera's
clear view of it to a collection, the whole collection is clustered in the latent space of
the visual servo network's frozen DINOv2 trunk, and the object is dropped at the card its
cluster belongs to. The set is never all visible at once, so the clusters are redrawn on
every pick; each is matched to the card that holds most of its earlier members, so a card
keeps its meaning as the collection grows, although items dropped earlier may by then
belong elsewhere.
"""

import asyncio
import json
import logging
import time

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
# How far the fingers may close past the most open they were during the grasp before a
# clear view stops counting as the item seen whole.
FINGER_CLOSE_TOLERANCE_DEG = 10.0
CLEAR_VIEW_POLL_S = 0.1


def embed(trunk, image_rgb, device):
    """The L2-normalized [CLS] token of the frozen trunk for one gripper frame."""
    import torch

    from nf_robot.ml.image_input import input_batch
    from nf_robot.ml.visual_servoing.model import DEFAULT_IMAGE_SIZE

    with torch.no_grad():
        cls = trunk(input_batch(image_rgb, DEFAULT_IMAGE_SIZE, device)).last_hidden_state[0, 0]
    v = cls.float().cpu().numpy()
    return v / (np.linalg.norm(v) + 1e-8)


def assign_bins(embeddings, previous_bins, k):
    """Cluster every embedding into at most k groups, and give each group a bin.

    previous_bins holds the bin each earlier item was dropped in, one per embedding but the
    last. Groups are matched to bins to keep as many earlier items as possible where they
    are. Returns the bin of every item under this clustering.
    """
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.optimize import linear_sum_assignment

    n = len(embeddings)
    if n == 1:
        labels = np.zeros(1, dtype=int)
    else:
        labels = fcluster(linkage(np.asarray(embeddings), 'ward'), min(k, n), 'maxclust') - 1
    overlap = np.zeros((k, k))
    for label, b in zip(labels, previous_bins):
        overlap[label, b] += 1
    groups, bins = linear_sum_assignment(overlap, maximize=True)
    to_bin = dict(zip(groups, bins))
    return [int(to_bin[label]) for label in labels]


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

    def _near_card(self, position, cards):
        return any(np.linalg.norm(np.asarray(position)[:2] - c[:2]) < CARD_EXCLUSION_M for c in cards)

    async def _grasp_watching(self):
        """Grasp, keeping the newest clear view of the item from before the fingers closed.

        last_clear_item_image also fills during the lift, with the item in the fingers, so
        a view counts only while the fingers are still about as open as they got.
        Returns (grasped, ItemImage or None).
        """
        ob = self.ob
        started = time.time()
        grasp = asyncio.create_task(ob.grasp())
        most_open = ob.finger_angle()
        seen = None
        try:
            while not grasp.done():
                angle = ob.finger_angle()
                most_open = min(most_open, angle)
                view = ob.last_clear_item_image()
                if (angle <= most_open + FINGER_CLOSE_TOLERANCE_DEG and view is not None
                        and view.timestamp >= started):
                    seen = view
                await asyncio.wait([grasp], timeout=CLEAR_VIEW_POLL_S)
        finally:
            if not grasp.done():
                grasp.cancel()
                # the grasp stops the spools on its way out; wait for that before moving on
                await asyncio.gather(grasp, return_exceptions=True)
        return grasp.result(), seen

    @verb('cluster', motion=True)
    @prefer_swing_cancellation
    async def cluster_sort(self, k='4'):
        """Pick up what the target model finds and sort it among the cards by appearance.
        'cluster' sorts into all four cards, 'cluster K' into the first K."""
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

        ppc = ob.config.pick_and_place
        GANTRY_HEIGHT_OVER_TARGET = tonp(ppc.gantry_height_over_target)
        GANTRY_HEIGHT_OVER_DROPOFF = tonp(ppc.gantry_height_over_dropoff)

        run_dir = self.output_dir(f'cluster_sort/{time.strftime("%Y%m%d-%H%M%S")}')
        embeddings, dropped_in, records = [], [], []
        drop_point = None
        target_seen_t = time.time()
        logger.info(f'Cluster sorting into {", ".join(bins)}; collection in {run_dir}')
        try:
            while True:
                # every card, in use or not, so nothing sorted earlier is picked back up
                cards = [p for p in map(ob.named_position, BIN_NAMES) if p is not None]
                target = pnp.target_queue.get_best_target(
                    accept=lambda t: not self._near_card(t.position, cards))
                if target is None:
                    await ob.clear_goal()
                    if time.time() > target_seen_t + ppc.end_loop_timeout:
                        self.notify(f'Looks clean enough. Sorted {len(records)} objects.')
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

                if seen is None:
                    # nothing to sort it by, and it must not go back where it was found
                    logger.warning('No clear view of the object before the grasp; dropping it in the first bin')
                    b = 0
                else:
                    embeddings.append(await self.run_in_thread(embed, self.trunk, seen.image_rgb, device))
                    current = await self.run_in_thread(assign_bins, embeddings, dropped_in, k)
                    b = current[-1]
                    dropped_in.append(b)
                    image_name = f'{len(records):04d}.jpg'
                    cv2.imwrite(str(run_dir / image_name), cv2.cvtColor(seen.image_rgb, cv2.COLOR_RGB2BGR))
                    records.append({
                        'image': image_name,
                        'timestamp': seen.timestamp,
                        'laser_range': seen.laser_range,
                        'pickup_position': [round(float(c), 4) for c in target.position[:2]],
                        'dropped_in': bins[b],
                    })
                    for record, c in zip(records, current):
                        record['current_bin'] = bins[c]
                    self._save_collection(run_dir, bins, records, embeddings)
                    self.progress(0, f'{len(records)} sorted, last into {bins[b]}')
                logger.info(f'Object goes to the {bins[b]} card')

                drop_point = ob.named_position(bins[b])
                await ob.seek_goal(drop_point + GANTRY_HEIGHT_OVER_DROPOFF)
                asyncio.create_task(ob.set_finger_angle(
                    max(-90, min(ppc.relaxed_open, ob.finger_angle() - 10))))
                # don't immediately select a new target, because it may be what was just let go
                await asyncio.sleep(ppc.delay_after_drop)
                pnp.target_queue.set_target_status(target.id, telemetry.TargetStatus.DROPPED)
                pnp.send_tq_to_ui()
        except asyncio.CancelledError:
            logger.info('Cluster sort cancelled')
            raise
        finally:
            self.finish()
            ob.slow_stop_all_spools()
            await ob.clear_goal()

    def _save_collection(self, run_dir, bins, records, embeddings):
        """Rewrite the run's collection.json and embeddings.npy, so a run cut short still has both."""
        with open(run_dir / 'collection.json', 'w') as f:
            json.dump({'bins': list(bins), 'objects': records}, f, indent=2)
        np.save(run_dir / 'embeddings.npy', np.stack(embeddings))
