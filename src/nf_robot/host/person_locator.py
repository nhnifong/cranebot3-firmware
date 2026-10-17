"""Locating the operator in the anchor cameras, as an alternative to the 'gamepad' tag.

The UI can orbit its controls around either the gamepad tag or the named position
'person' published here. RF-DETR (COCO weights, from the `rfdetr` package) runs on every
anchor camera about once a second, and the person found is placed at the floor point
under their feet - the bottom centre of their box, cast to the floor - averaged over the
cameras that see them.

With more than one person in the room the one wanted is the one holding the gamepad.
COCO has no gamepad class, but the detector reliably calls one a "remote", or now and
then a "cell phone", and puts that box inside the hands of whoever holds it. So the
people with such a detection inside their box are the candidates, and since a phone
looks the same to it, the one chosen among several is the one nearest the last known
position of the gamepad tag. The other cameras each contribute their person nearest
that one, if any lies close enough to be the same person.

Off until the UI sends COMMAND_ENABLE_PERSON_LOCATOR.
"""

import asyncio
import logging
import time

import numpy as np

from nf_robot.common.cv_common import project_pixels_to_floor
from nf_robot.generated.nf import telemetry

logger = logging.getLogger(__name__)

NAMED_POSITION = 'person'
# The tag that breaks ties between several people holding something gamepad-like.
TAG_POSITION = 'gamepad'
PERIOD_S = 1.0
PERSON_THRESHOLD = 0.5
# Low, because a gamepad across the room scores 0.2 to 0.45. Only boxes inside a person
# count, which keeps the clutter this lets through from mattering.
HELD_THRESHOLD = 0.15
HELD_CLASSES = ('remote', 'cell phone')
# Feet estimates from two cameras further apart than this are taken to be two people.
SAME_PERSON_RADIUS_M = 1.0


class PersonLocator:
    def __init__(self, observer):
        self.ob = observer
        self.model = None
        self.class_names = None
        self.task = None

    @property
    def enabled(self):
        return self.task is not None and not self.task.done()

    def start(self):
        if not self.enabled:
            self.task = asyncio.create_task(self.run())

    async def stop(self):
        if self.task is not None:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
            self.task = None

    def _load_sync(self):
        from rfdetr import RFDETRMedium
        from rfdetr.assets.coco_classes import COCO_CLASSES
        self.model = RFDETRMedium(device=self.ob.torch_device())
        self.class_names = COCO_CLASSES

    async def run(self):
        logger.info('person locator starting')
        try:
            if self.model is None:
                try:
                    await asyncio.to_thread(self._load_sync)
                except Exception as e:
                    logger.error(f'Could not load the person detector: {e!r}')
                    self.ob.send_ui(pop_message=telemetry.Popup(
                        message=f'The person locator could not load RF-DETR: {e}'))
                    return
            while True:
                started = time.monotonic()
                try:
                    position = await self.locate_once()
                    if position is not None:
                        self.ob.set_named_position(NAMED_POSITION, position, save=False)
                except Exception:
                    logger.exception('person locator pass failed')
                await asyncio.sleep(max(0.0, PERIOD_S - (time.monotonic() - started)))
        finally:
            logger.info('person locator stopped')

    async def locate_once(self):
        """Floor position of the person to orbit around, or None if nobody is seen."""
        views = [
            (client.anchor_num, client.last_output_frame, np.array(client.camera_pose))
            for client in list(self.ob.anchors.values())
            if client.last_output_frame is not None
            and client.anchor_num in self.ob.config.preferred_cameras
        ]
        if not views:
            return None
        detections = await asyncio.to_thread(
            self.model.predict, [frame for _, frame, _ in views], threshold=HELD_THRESHOLD)
        if not isinstance(detections, list):
            detections = [detections]

        # (anchor, feet xy, person score, held score), every person in every camera
        people = []
        for (anchor_num, frame, pose), det in zip(views, detections):
            h, w = frame.shape[:2]
            names = [self.class_names.get(int(c)) for c in det.class_id]
            boxes = [(b, s) for b, s, n in zip(det.xyxy, det.confidence, names)
                     if n == 'person' and s >= PERSON_THRESHOLD]
            held = [(b, s) for b, s, n in zip(det.xyxy, det.confidence, names) if n in HELD_CLASSES]
            for box, score in boxes:
                held_score = max((float(s) for b, s in held if _centre_inside(b, box)), default=0.0)
                feet = project_pixels_to_floor(
                    [[(box[0] + box[2]) / 2 / w, box[3] / h]], pose, self.ob.config.camera_cal)
                if len(feet) == 1:
                    people.append((anchor_num, feet[0], float(score), held_score))
        if not people:
            return None

        primary = self._choose(people)
        chosen = [primary]
        for anchor_num in {p[0] for p in people} - {primary[0]}:
            others = [p for p in people if p[0] == anchor_num]
            nearest = min(others, key=lambda p: np.linalg.norm(p[1] - primary[1]))
            if np.linalg.norm(nearest[1] - primary[1]) <= SAME_PERSON_RADIUS_M:
                chosen.append(nearest)
        xy = np.mean([p[1] for p in chosen], axis=0)
        logger.debug(f'person locator: {len(people)} people, holder score {primary[3]:.2f}, '
                     f'cameras {[p[0] for p in chosen]}, at {xy.round(2)}')
        return np.array([xy[0], xy[1], 0.0])

    def _choose(self, people):
        """The sighting to locate: among those holding something gamepad-like, the one
        nearest the gamepad tag, or the best holder if the tag has never been seen. With
        nobody holding anything, the surest person."""
        holders = [p for p in people if p[3] > 0]
        if not holders:
            return max(people, key=lambda p: p[2])
        tag = self.ob.named_position(TAG_POSITION)
        if tag is None:
            return max(holders, key=lambda p: (p[3], p[2]))
        return min(holders, key=lambda p: np.linalg.norm(p[1] - tag[:2]))


def _centre_inside(inner, outer):
    cx, cy = (inner[0] + inner[2]) / 2, (inner[1] + inner[3]) / 2
    return outer[0] <= cx <= outer[2] and outer[1] <= cy <= outer[3]
