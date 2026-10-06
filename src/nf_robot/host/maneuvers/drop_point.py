"""Drop point: predicting where the item being picked up belongs."""

import asyncio
import logging
import time

import numpy as np

from nf_robot.common.model_revisions import pinned_revision
from nf_robot.host.maneuver import PREDICTED_DROP_NAME, Maneuver, verb

logger = logging.getLogger(__name__)

DROP_POINT_INTERVAL_S = 0.25


class DropPoint(Maneuver):
    name = 'drop_point'
    title = 'Drop Point'

    def __init__(self, ob):
        super().__init__(ob)
        # the model and the task that runs it, both lazy
        self.model = None
        self.watch_task = None
        # (time it was made, floor position) of the newest prediction, or None
        self.last_prediction = None

    async def ensure_model(self):
        """Load the drop point model if it is not loaded. True if there is one to run.

        Everything slow happens in a worker thread - the torch import and, on the hub path,
        a download - because on the event loop either one stalls telemetry and every motion
        task for as long as it takes. Nothing raises: a model that will not load is a
        prediction the robot goes without, not a traceback out of a pick and place.
        """
        if self.model is not None:
            return True

        def load_sync():
            from nf_robot.ml.placer.model import DROP_POINT_MODEL_REPOID, load_model

            # The trunk is the shared frozen one (ml/dino_trunk.py), so this adds a head
            # rather than a second backbone: about 0.5GB of VRAM and 30ms a frame, or the
            # head alone when another model is already loaded. It has to be the observer's
            # own device for that reason - loading this one somewhere else would drag the
            # trunk the other models are using along with it.
            device = self.ob.torch_device()
            model, checkpoint = load_model(device, local_models=self.ob.local_models,
                                           revision=pinned_revision(DROP_POINT_MODEL_REPOID))
            return model, checkpoint, device

        try:
            model, checkpoint, device = await self.run_in_thread(load_sync)
        except Exception as e:
            logger.error(f'Could not load the drop point model: {e!r}')
            self.notify(f'Could not load the drop point model, so nothing will predict where '
                        f'items go: {e}')
            return False
        self.model = model
        logger.info(f'Drop point model ready on {device}: epoch {checkpoint.get("epoch")}, '
                    f'metrics {checkpoint.get("metrics")}')
        return True

    @verb('droppoint')
    async def toggle_preview(self):
        """Debug: run the drop point model on its own, without a pick and place.

        The same loop pick and place runs, so what it writes is what a pick would fly to.
        Send the command again to stop.
        """
        if self.watch_task is not None and not self.watch_task.done():
            self.stop_watch()
            self.notify('Drop point prediction stopped')
            return False
        if not await self.ensure_model():
            return False
        self.start_watch()
        return True

    def start_watch(self):
        """Run the drop point model in the background, if it is loaded and not already running."""
        if self.model is None:
            return
        if self.watch_task is None or self.watch_task.done():
            self.watch_task = self.spawn(self._watch(), name='drop_point_watch')

    def stop_watch(self):
        if self.watch_task is not None:
            self.watch_task.cancel()
            self.watch_task = None

    def _predict(self, item_rgb, ortho_rgb):
        """The model's drop point for one pair of frames, as normalized overhead (u, v)."""
        from nf_robot.ml.image_input import input_batch
        from nf_robot.ml.placer.model import predict

        model = self.model
        # The model's own device, so the frames cannot arrive somewhere its weights are not.
        device = next(model.parameters()).device
        item = input_batch(item_rgb, model.item_size, device)
        overhead = input_batch(ortho_rgb, model.overhead_size, device)
        uv = predict(model, item, overhead)["uv"][0, 0]
        return float(uv[0]), float(uv[1])

    async def _watch(self, interval_s=DROP_POINT_INTERVAL_S):
        """Predict where the item in front of the gripper goes, each time there is a new view
        of one.

        Only from last_clear_item_image, whose frames are taken at the distance the model was
        trained at: further out it is looking at the floor, closer the fingers are across it,
        and in both cases the answer is not worth the frame. The prediction is published as
        the PREDICTED_DROP_NAME position, which is what the UI draws and what a route flies to.
        """
        from nf_robot.ml.ortho_target.model import ortho_px_to_room

        last_used = None
        try:
            while True:
                await asyncio.sleep(interval_s)
                seen = self.ob.last_clear_item_image()
                if seen is None or seen.timestamp == last_used:
                    continue
                ortho_rgb = self.ob.latest_ortho()
                if ortho_rgb is None:
                    continue
                last_used = seen.timestamp
                u, v = await self.run_in_thread(self._predict, seen.image_rgb, ortho_rgb)
                x, y = ortho_px_to_room(u, v, 1.0, 1.0)
                self.ob.set_named_position(PREDICTED_DROP_NAME, np.array([x, y, 0.0]), save=False)
                self.last_prediction = (time.time(), np.array([x, y, 0.0]))
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.exception('drop point prediction failed')
            self.notify(f'Drop point prediction failed: {e}')
        finally:
            logger.debug('drop point prediction ended')
