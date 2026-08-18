"""Tests for the plate recording the demux loop writes.

A recording is a remux of the live stream, started whenever a caller sets recording_path.
That instant is not a frame boundary the encoder knows about, which is what these are
about: an h264 recording that does not begin at a keyframe is undecodable up to the next
one, and the decoder says so at length ("non-existing PPS 0 referenced").
"""

import tempfile
import types
import unittest
from pathlib import Path

import av
import numpy as np

from nf_robot.host.component_client import ComponentClient


def make_stream(path, frames=120, gop=60):
    """A short h264 mpegts file with a known keyframe interval, standing in for the pi."""
    container = av.open(str(path), "w", format="mpegts")
    stream = container.add_stream("libx264", rate=30)
    stream.width, stream.height, stream.pix_fmt = 160, 128, "yuv420p"
    # no scene cuts, so keyframes land only where keyint puts them
    stream.options = {"x264-params": f"scenecut=0:keyint={gop}:min-keyint={gop}"}
    base = np.random.default_rng(0).integers(0, 255, (128, 160, 3), dtype=np.uint8)
    for i in range(frames):
        image = np.roll(base, i, axis=1)
        for packet in stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def read_packets(path):
    container = av.open(str(path))
    stream = next(s for s in container.streams if s.type == "video")
    packets = [p for p in container.demux(stream) if p.dts is not None]
    return stream, packets, container


def decode_count(path):
    """How many frames come out of a file. Fewer than it has packets means the leading
    ones referenced parameter sets that were never written."""
    container = av.open(str(path))
    stream = next(s for s in container.streams if s.type == "video")
    try:
        return sum(1 for _ in container.decode(stream))
    finally:
        container.close()


class FakeClient:
    """The attributes _service_recording touches, and nothing else."""

    def __init__(self, path):
        self.recording_path = path
        self._recording = None
        self.recorded_packets = 0
        self.recording_stream_start_ts = None
        self._recording_skipped = 0
        self.stream_start_ts = 1000.0

    _service_recording = ComponentClient._service_recording
    _close_recording = ComponentClient._close_recording


class TestRecordingStartsOnAKeyframe(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dir = Path(tempfile.mkdtemp())
        cls.source = cls.dir / "source.ts"
        make_stream(cls.source)

    def _record_from(self, index, name):
        """Feed the source's packets to the real _service_recording, starting at `index`,
        the way the demux loop would if recording were switched on at that moment."""
        stream, packets, container = read_packets(self.source)
        client = FakeClient(self.dir / name)
        try:
            for packet in packets[index:]:
                client._service_recording(stream, packet)
        finally:
            client._close_recording()
            container.close()
        return client

    def test_source_has_a_long_gop(self):
        """Otherwise the rest of these prove nothing."""
        _, packets, container = read_packets(self.source)
        keys = [i for i, p in enumerate(packets) if p.is_keyframe]
        container.close()
        self.assertEqual(keys[0], 0)
        self.assertGreaterEqual(keys[1], 30)

    def test_waits_for_a_keyframe_when_switched_on_mid_gop(self):
        _, source_packets, container = read_packets(self.source)
        keys = [i for i, p in enumerate(source_packets) if p.is_keyframe]
        container.close()

        client = self._record_from(5, "mid.ts")
        _, packets, container = read_packets(self.dir / "mid.ts")
        container.close()
        self.assertTrue(packets[0].is_keyframe)
        # it skipped forward from packet 5 to the next keyframe rather than writing them
        self.assertEqual(client.recorded_packets, len(source_packets) - keys[1])

    def test_every_recorded_packet_decodes(self):
        """The bug: a mid-GOP start wrote 115 packets that decoded to 60 frames."""
        self._record_from(5, "clean.ts")
        _, packets, container = read_packets(self.dir / "clean.ts")
        container.close()
        self.assertEqual(decode_count(self.dir / "clean.ts"), len(packets))

    def test_a_recording_switched_on_at_a_keyframe_keeps_that_frame(self):
        """Nothing is dropped when the timing happens to be right."""
        _, packets, container = read_packets(self.source)
        keys = [i for i, p in enumerate(packets) if p.is_keyframe]
        container.close()
        client = self._record_from(keys[1], "onkey.ts")
        self.assertEqual(client.recorded_packets, len(packets) - keys[1])


if __name__ == "__main__":
    unittest.main()
