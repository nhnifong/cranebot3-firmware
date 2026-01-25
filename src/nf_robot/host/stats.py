import asyncio
import numpy as np
import time

from nf_robot.generated.nf.telemetry import VidStats

class StatCounter:
    def __init__(self, ob):
        self.ob = ob
        self.detection_count = 0
        self.pending_frames_in_pool = 0
        self.latency = []
        self.framerate = []
        self.last_update = time.time()
        self.run = True
        self.mean_latency = 0

    async def stat_main(self):
        while self.run:
            now = time.time()
            elapsed = now-self.last_update
            if len(self.latency) > 0:
                self.mean_latency = np.mean(np.array(self.latency))
            mean_framerate = 0
            if len(self.framerate) > 0:
                mean_framerate = np.mean(np.array(self.framerate))
            detection_rate = self.detection_count / elapsed
            self.last_update = now
            self.latency = []
            self.framerate = []
            self.detection_count = 0
            self.ob.send_ui(vid_stats=VidStats(
                detection_rate=detection_rate,
                video_latency=self.mean_latency,
                video_framerate=mean_framerate,
            ))
            await asyncio.sleep(0.5)