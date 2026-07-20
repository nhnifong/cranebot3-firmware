"""
End-to-end tests of the compressed-passthrough paths in video_streamer.py against a real
H264/mpegts TCP source, shaped like what rpicam-vid produces (see anchor_server.py's
stream_command): RTMPStreamer's stream-copy remux (cloud/telemetry-relay path), and
CompressedStreamer's raw TCP broadcast (LAN/same-machine path, e.g. a lerobot recording
process).

These exercise the actual production classes and the actual ffmpeg binary, not mocks: a
synthetic source is opened with PyAV using the exact same options anchor_client.py's
receive_video() uses, demuxed (not decoded) to grab each packet's raw compressed bytes.

Also covers the routing regression this fix could easily have introduced: NfVideoStreamer's
synthesized-frame producers (e.g. observer.py's orthographic floor projection) have no
pre-existing compressed bytes and must keep using the original encode-from-raw-frames path,
never passthrough.
"""
import shutil
import subprocess
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import av

from nf_robot.host.video_streamer import CompressedStreamer, NfVideoStreamer, RTMPStreamer

FFMPEG_AVAILABLE = shutil.which('ffmpeg') is not None
SOURCE_PORT = 18899  # distinct from mock_rpicam_vid.py's 8888-8891 range and the real 8888
COMPRESSED_PORT = 18898


def _receive_options():
    """The exact options anchor_client.py's receive_video() opens the component's video
    stream with, so this test proves the fix works under the real low-latency config."""
    return {
        'rtsp_transport': 'tcp',
        'fflags': 'nobuffer',
        'flags': 'low_delay',
        'fast': '1',
    }


class _SourceBackedTestCase(unittest.TestCase):
    """Base for tests that need a real, live H264/mpegts source. Shaped like rpicam-vid: H264
    in mpegts, listening on a TCP socket, low latency. Software libx264 here only stands in
    for the Pi's hardware encoder -- what's under test is the host's demux/passthrough, not
    the source's own encoder."""

    source_port = SOURCE_PORT

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="video_streamer_test_")
        self.source_proc = subprocess.Popen(
            [
                'ffmpeg', '-y', '-re', '-f', 'lavfi', '-i', 'testsrc=size=640x480:rate=20',
                '-c:v', 'libx264', '-preset', 'ultrafast', '-tune', 'zerolatency', '-g', '20',
                '-b:v', '520k',
                '-f', 'mpegts', f'tcp://127.0.0.1:{self.source_port}?listen=1',
            ],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

    def tearDown(self):
        self.source_proc.terminate()
        try:
            self.source_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.source_proc.kill()
            self.source_proc.wait(timeout=5)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _open_source(self, retries=20):
        # The ffmpeg source needs a moment to start listening after Popen returns.
        last_error = None
        for _ in range(retries):
            try:
                return av.open(f'tcp://127.0.0.1:{self.source_port}', options=_receive_options(), mode='r')
            except av.error.ConnectionRefusedError as e:
                last_error = e
                time.sleep(0.2)
        raise last_error


@unittest.skipUnless(FFMPEG_AVAILABLE, "ffmpeg CLI not installed")
class TestRTMPStreamerPassthrough(_SourceBackedTestCase):
    """Real ffmpeg source -> PyAV demux -> RTMPStreamer(passthrough=True) -> real ffmpeg remux."""

    def test_send_packet_produces_decodable_output_without_reencoding(self):
        """RTMPStreamer(passthrough=True) fed raw demuxed packets should produce output
        that decodes to real video, via a pure stream-copy remux (no decode/re-encode)."""
        container = self._open_source()
        stream = next(s for s in container.streams if s.type == 'video')
        stream.thread_type = "SLICE"

        out_path = str(Path(self.tmpdir) / "passthrough_out.flv")
        streamer = RTMPStreamer(width=640, height=480, fps=20, rtmp_url=out_path, passthrough=True)
        streamer.start()
        time.sleep(0.3)  # let ffmpeg spin up, same as production

        n_packets = 0
        n_frames_decoded_locally = 0
        try:
            for packet in container.demux(stream):
                if packet.dts is None:
                    continue
                # The path under test: raw compressed bytes forwarded untouched.
                streamer.send_packet(bytes(packet))
                n_packets += 1
                # The unaffected path: still decode the same packets for CV/local-frame use.
                for _frame in packet.decode():
                    n_frames_decoded_locally += 1
                if n_packets >= 100:
                    break
        finally:
            container.close()
            streamer.stop()

        self.assertGreater(n_packets, 0, "never received any packets from the synthetic source")
        self.assertGreater(n_frames_decoded_locally, 0,
                            "demuxing for send_packet() broke decoding the same packets for CV/local use")

        check = av.open(out_path, mode='r')
        try:
            check_stream = next(s for s in check.streams if s.type == 'video')
            n_out = sum(1 for _ in check.decode(check_stream))
        finally:
            check.close()

        self.assertGreater(n_out, 0, "RTMPStreamer(passthrough=True) produced no decodable output")
        # Stream-copy is lossless at the packet level: every packet fed in should be
        # recoverable as a frame out (allowing a little slack for decoder priming/EOF flush).
        self.assertGreaterEqual(n_out, n_frames_decoded_locally - 2)
        self.assertEqual(streamer.connection_status, 'ok')


@unittest.skipUnless(FFMPEG_AVAILABLE, "ffmpeg CLI not installed")
class TestCompressedStreamer(_SourceBackedTestCase):
    """Real ffmpeg source -> PyAV demux -> CompressedStreamer -> multiple real PyAV clients
    over raw TCP, for the LAN/same-machine passthrough path (e.g. a lerobot recorder).

    Unlike RTMPStreamer (one persistent process, fed from the start of its own lifetime),
    CompressedStreamer's whole point is serving clients that connect after the stream is
    already live -- so the case that matters here isn't "does it forward bytes" (that's the
    same mechanism RTMPStreamer already proves), it's "can a client that joins mid-GOP
    bootstrap correctly from the buffered backlog," and "does one client joining affect
    another already-connected one." Both are exercised with two concurrent real clients.
    """

    source_port = COMPRESSED_PORT

    def test_concurrent_midstream_clients_bootstrap_from_gop_backlog(self):
        container = self._open_source()
        stream = next(s for s in container.streams if s.type == 'video')
        stream.thread_type = "SLICE"

        streamer = CompressedStreamer(port=COMPRESSED_PORT + 1, bind_address='127.0.0.1')
        streamer.start()

        fed = {'n': 0}

        def feed(count):
            for packet in container.demux(stream):
                if packet.dts is None:
                    continue
                streamer.send_packet(bytes(packet), packet.is_keyframe)
                fed['n'] += 1
                count -= 1
                if count <= 0:
                    return

        def client(results, key, target_frames):
            c = av.open(f'tcp://127.0.0.1:{COMPRESSED_PORT + 1}', mode='r')
            try:
                cstream = next(s for s in c.streams if s.type == 'video')
                got = 0
                for _frame in c.decode(cstream):
                    got += 1
                    if got >= target_frames:
                        break
                results[key] = got
            finally:
                c.close()

        try:
            # Establish a real backlog before anyone connects -- this is the scenario the
            # GOP buffer exists for: a client joining after the stream has been running.
            feed(25)
            self.assertGreater(fed['n'], 0, "never received any packets from the synthetic source")

            results = {}
            client_a = threading.Thread(target=client, args=(results, 'A', 6))
            client_b = threading.Thread(target=client, args=(results, 'B', 6))
            client_a.start()
            client_b.start()
            time.sleep(0.3)  # let both connect and get seeded from the backlog

            feed(60)  # keep the live feed going while both clients are reading

            client_a.join(timeout=10)
            client_b.join(timeout=10)
        finally:
            container.close()
            streamer.stop()

        self.assertGreater(results.get('A', 0), 0,
                            "client A (joined mid-stream) decoded nothing -- GOP backlog broken")
        self.assertGreater(results.get('B', 0), 0,
                            "client B (joined mid-stream, concurrent with A) decoded nothing")


class TestNfVideoStreamerPassthroughRouting(unittest.TestCase):
    """Fast, no-network regression guard for the passthrough/encode routing split.

    This is exactly the bug this fix could have introduced silently: observer.py's
    orthographic floor projection has no pre-existing compressed bytes (it's synthesized
    from multiple camera views), so it must keep going through send_frame() -> encode, on
    both the local and remote sides, while camera-sourced streams must route send_frame()
    to local only and send_packet() to both remote and compressed.
    """

    def _make_streamer(self, passthrough):
        vs = NfVideoStreamer.__new__(NfVideoStreamer)
        vs._local = MagicMock()
        vs._remote = MagicMock()
        vs._compressed = MagicMock()
        vs._passthrough = passthrough
        vs._frames_sent = 0
        vs._frames_before_ready = 999
        vs._ready_sent = False
        vs._on_ready = None
        return vs

    def test_passthrough_stream_sends_frames_local_only_and_packets_remote_and_compressed(self):
        vs = self._make_streamer(passthrough=True)
        vs.send_frame("FRAME")
        vs.send_packet(b"PACKET", is_keyframe=True)

        vs._local.send_frame.assert_called_once_with("FRAME")
        vs._remote.send_frame.assert_not_called()
        vs._remote.send_packet.assert_called_once_with(b"PACKET")
        vs._compressed.send_packet.assert_called_once_with(b"PACKET", True)

    def test_encode_stream_sends_frames_to_local_and_remote_only(self):
        """Regression guard: synthesized-frame producers (e.g. the ortho floor projection
        in observer.py) have no compressed bytes to pass through and must keep encoding."""
        vs = self._make_streamer(passthrough=False)
        vs.send_frame("FRAME")
        vs.send_packet(b"PACKET")  # must be a no-op in encode mode: no support for it

        vs._local.send_frame.assert_called_once_with("FRAME")
        vs._remote.send_frame.assert_called_once_with("FRAME")
        vs._remote.send_packet.assert_not_called()
        vs._compressed.send_packet.assert_not_called()


if __name__ == '__main__':
    unittest.main()
