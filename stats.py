import asyncio
import numpy as np
import time

class StatCounter:
    def __init__(self, to_ui_q):
        self.to_ui_q = to_ui_q
        self.detection_count = 0
        self.latency = []
        self.framerate = []
        self.last_update = time.time()
        self.run = True

    async def stat_main(self):
        while self.run:
            now = time.time()
            elapsed = now-self.last_update
            mean_latency = np.mean(np.array(self.latency))
            mean_framerate = np.mean(np.array(self.framerate))
            detection_rate = self.detection_count / elapsed
            self.last_update = now
            self.latency = []
            self.framerate = []
            self.detection_count = 0
            self.pending_frames_in_pool = 0
            self.to_ui_q.put({'vid_stats':{
                'detection_rate':detection_rate,
                'video_latency':mean_latency,
                'video_framerate':mean_framerate,
                'pending_frames': self.pending_frames_in_pool,
                }})
            await asyncio.sleep(0.5)