"""Tests for the soft target the cell head is trained against.

The head is a 1792-way softmax over 12px cells and the mined labels are good to rather
less than that, so the target is spread over the cells a label plausibly covers. These
check the spreading is a real distribution, sits where the label is, and still collapses
to the old behaviour at sigma 0.
"""

import math
import unittest

import torch

from nf_robot.ml.visual_servoing.train import CELL_SIGMA, cell_loss, soft_cell_target

GRID = (32, 56)          # rows, cols - the model's canvas grid
ROWS, COLS = GRID


def centre_of(cell_x, cell_y):
    """A label sitting exactly at the centre of one cell."""
    return torch.tensor([[cell_x + 0.5, cell_y + 0.5]])


class TestSoftCellTarget(unittest.TestCase):

    def test_it_is_a_distribution(self):
        cell = torch.tensor([[10.3, 4.7], [55.9, 31.9], [0.1, 0.1]])
        target = soft_cell_target(cell, GRID, CELL_SIGMA)
        self.assertEqual(target.shape, (3, ROWS * COLS))
        torch.testing.assert_close(target.sum(dim=1), torch.ones(3))
        self.assertTrue((target >= 0).all())

    def test_the_peak_is_at_the_label(self):
        target = soft_cell_target(centre_of(20, 12), GRID, CELL_SIGMA)
        peak = int(target[0].argmax())
        self.assertEqual((peak // COLS, peak % COLS), (12, 20))

    def test_the_peak_follows_the_label_between_cells(self):
        """Centred on where the label is, not on the cell it lands in - two labels in the
        same cell but at opposite corners should not produce the same target."""
        left = soft_cell_target(torch.tensor([[20.05, 12.5]]), GRID, CELL_SIGMA)
        right = soft_cell_target(torch.tensor([[20.95, 12.5]]), GRID, CELL_SIGMA)
        self.assertFalse(torch.allclose(left, right))
        # mass shifts toward the neighbouring cell the label leans into
        def at(target, x, y):
            return float(target[0, y * COLS + x])
        self.assertGreater(at(left, 19, 12), at(right, 19, 12))
        self.assertGreater(at(right, 21, 12), at(left, 21, 12))

    def test_a_wider_sigma_spreads_more(self):
        cell = centre_of(20, 12)
        peak_at = lambda s: float(soft_cell_target(cell, GRID, s).max())
        self.assertGreater(peak_at(0.5), peak_at(1.5))
        self.assertGreater(peak_at(1.5), peak_at(4.0))

    def test_a_target_in_the_corner_still_sums_to_one(self):
        """Most of its Gaussian is outside the canvas; that mass has to be redistributed
        rather than silently dropped."""
        corner = soft_cell_target(torch.tensor([[0.0, 0.0]]), GRID, CELL_SIGMA)
        torch.testing.assert_close(corner.sum(dim=1), torch.ones(1))
        peak = int(corner[0].argmax())
        self.assertEqual((peak // COLS, peak % COLS), (0, 0))


class TestCellLoss(unittest.TestCase):

    def _logits(self, batch=1):
        return torch.zeros(batch, 1, ROWS, COLS)

    def test_sigma_zero_is_the_old_one_hot_cross_entropy(self):
        logits = torch.randn(4, 1, ROWS, COLS)
        cell = torch.tensor([[10.3, 4.7], [30.0, 20.0], [1.2, 2.8], [55.5, 31.5]])
        index = (cell[:, 1].floor().long() * COLS + cell[:, 0].floor().long())
        expected = torch.nn.functional.cross_entropy(
            logits.flatten(1), index, reduction="none")
        torch.testing.assert_close(cell_loss(logits, cell, GRID, index, 0.0), expected)

    def test_a_perfect_prediction_costs_nothing(self):
        """Reported as a KL, so the floor is zero however wide the target is - otherwise
        runs at different sigma could not be compared."""
        cell = torch.tensor([[20.5, 12.5]])
        target = soft_cell_target(cell, GRID, CELL_SIGMA)
        logits = target.clamp(min=1e-12).log().view(1, 1, ROWS, COLS)
        loss = cell_loss(logits, cell, GRID, torch.tensor([12 * COLS + 20]), CELL_SIGMA)
        self.assertLess(float(loss[0]), 1e-5)

    def test_being_wrong_costs_more_than_being_close(self):
        cell = torch.tensor([[20.5, 12.5]])
        index = torch.tensor([12 * COLS + 20])

        def loss_for_peak(x, y):
            logits = torch.full((1, 1, ROWS, COLS), -10.0)
            logits[0, 0, y, x] = 10.0
            return float(cell_loss(logits, cell, GRID, index, CELL_SIGMA)[0])

        self.assertLess(loss_for_peak(20, 12), loss_for_peak(23, 12))
        self.assertLess(loss_for_peak(23, 12), loss_for_peak(50, 30))

    def test_it_still_has_a_gradient(self):
        cell = torch.tensor([[20.5, 12.5]])
        logits = torch.zeros(1, 1, ROWS, COLS, requires_grad=True)
        cell_loss(logits, cell, GRID, torch.tensor([12 * COLS + 20]), CELL_SIGMA).sum().backward()
        self.assertTrue(logits.grad.abs().sum() > 0)
        # steepest where the target says the answer is
        peak = int(logits.grad[0, 0].neg().argmax())
        self.assertEqual((peak // COLS, peak % COLS), (12, 20))


if __name__ == "__main__":
    unittest.main()
