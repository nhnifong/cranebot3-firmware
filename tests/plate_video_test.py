"""Tests for the mp4 previews the matte tools write beside their contact sheets."""

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from nf_robot.ml.visual_servoing.plates import (
    VIDEO_WIDTH, checkerboard, over_checkerboard, write_video)


def probe(path):
    """(frames, fps, width, height) of a written video, read back through cv2."""
    cap = cv2.VideoCapture(str(path))
    try:
        return (int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS),
                int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    finally:
        cap.release()


class TestWriteVideo(unittest.TestCase):

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())

    def test_writes_every_frame_at_the_asked_for_rate(self):
        frames = [np.full((240, 320, 3), i * 8, np.uint8) for i in range(30)]
        path = write_video(self.dir / "a.mp4", frames, fps=60)
        self.assertIsNotNone(path)
        count, fps, width, height = probe(path)
        self.assertEqual(count, 30)
        self.assertEqual(fps, 60.0)
        self.assertEqual((width, height), (320, 240))

    def test_downscales_a_1080p_capture(self):
        frames = [np.zeros((1080, 1920, 3), np.uint8) for _ in range(3)]
        _, _, width, height = probe(write_video(self.dir / "b.mp4", frames))
        self.assertEqual((width, height), (VIDEO_WIDTH, 540))

    def test_dimensions_come_out_even(self):
        """Odd dimensions are legal in the container but choke 4:2:0 players."""
        frames = [np.zeros((385, 683, 3), np.uint8) for _ in range(2)]
        _, _, width, height = probe(write_video(self.dir / "c.mp4", frames))
        self.assertEqual(width % 2, 0)
        self.assertEqual(height % 2, 0)

    def test_frames_of_differing_size_are_squared_up(self):
        """The object cutouts each have their own shape; a video has one."""
        frames = [np.zeros((240, 320, 3), np.uint8), np.zeros((100, 150, 3), np.uint8)]
        count, _, width, height = probe(write_video(self.dir / "d.mp4", frames))
        self.assertEqual(count, 2)
        self.assertEqual((width, height), (320, 240))

    def test_nothing_to_write_writes_nothing(self):
        self.assertIsNone(write_video(self.dir / "e.mp4", iter([])))
        self.assertFalse((self.dir / "e.mp4").exists())


class TestCheckerboard(unittest.TestCase):

    def test_transparent_shows_the_board_and_opaque_hides_it(self):
        bgra = np.zeros((32, 32, 4), np.uint8)
        bgra[:16, :, :3] = 200
        bgra[:16, :, 3] = 255  # opaque half
        board = checkerboard(32, 32)
        out = over_checkerboard(bgra, board)
        np.testing.assert_array_equal(out[:16], np.full((16, 32, 3), 200, np.uint8))
        np.testing.assert_array_equal(out[16:], board[16:])

    def test_half_alpha_is_a_half_mix(self):
        bgra = np.zeros((16, 16, 4), np.uint8)
        bgra[:, :, :3] = 100
        bgra[:, :, 3] = 128
        board = np.full((16, 16, 3), 200, np.uint8)
        out = over_checkerboard(bgra, board)
        self.assertTrue(np.all(np.abs(out.astype(int) - 150) <= 1))


if __name__ == "__main__":
    unittest.main()
