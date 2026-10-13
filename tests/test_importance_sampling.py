"""CPU tests for the narrow token-level TIS reference."""
import unittest
import torch

from agentic_text2sql.algorithms.importance_sampling import (
    SAFETY_BOUND,
    token_truncated_is_weights,
)


class ImportanceSamplingTests(unittest.TestCase):
    def test_unit_ratios(self):
        x = torch.zeros(2, 3)
        mask = torch.ones_like(x)
        torch.testing.assert_close(
            token_truncated_is_weights(x, mask), torch.ones_like(x)
        )

    def test_upper_truncation(self):
        x = torch.tensor([[.25, 1., 3.]], dtype=torch.float64).log()
        out = token_truncated_is_weights(x, torch.ones_like(x))
        torch.testing.assert_close(
            out, torch.tensor([[.25, 1., 2.]], dtype=torch.float64)
        )

    def test_mask_zero(self):
        x = torch.tensor([[0., 4., -3.]])
        mask = torch.tensor([[True, False, True]])
        out = token_truncated_is_weights(x, mask)
        self.assertEqual(out[0, 1].item(), 0.)

    def test_weights_detached(self):
        x = torch.zeros(1, 3, requires_grad=True)
        out = token_truncated_is_weights(x, torch.ones_like(x))
        self.assertFalse(out.requires_grad)
        self.assertIsNone(out.grad_fn)

    def test_no_batch_normalization(self):
        x = torch.full((2, 3), .5, dtype=torch.float64).log()
        out = token_truncated_is_weights(x, torch.ones_like(x))
        self.assertEqual(out.mean().item(), .5)

    def test_safety_clamp(self):
        x = torch.tensor(
            [[-2. * SAFETY_BOUND, 2. * SAFETY_BOUND]], dtype=torch.float64
        )
        out = token_truncated_is_weights(x, torch.ones_like(x))
        expected = torch.tensor(
            [[-SAFETY_BOUND, SAFETY_BOUND]], dtype=torch.float64
        ).exp().clamp(max=2.)
        torch.testing.assert_close(out, expected)

    def test_all_masked(self):
        x = torch.zeros(2, 3)
        out = token_truncated_is_weights(x, torch.zeros_like(x))
        self.assertEqual(torch.count_nonzero(out).item(), 0)

    def test_inputs_unchanged(self):
        x = torch.tensor([[-1., 0., 1.]])
        mask = torch.tensor([[1., 0., 1.]])
        before_x, before_mask = x.clone(), mask.clone()
        token_truncated_is_weights(x, mask)
        torch.testing.assert_close(x, before_x)
        torch.testing.assert_close(mask, before_mask)

    def test_invalid_inputs(self):
        x = torch.zeros(1, 3)
        mask = torch.ones_like(x)
        for cap in (0., -1., float("nan"), float("inf"), True):
            with self.subTest(cap=cap):
                with self.assertRaises((ValueError, TypeError)):
                    token_truncated_is_weights(x, mask, cap)
        with self.assertRaises(ValueError):
            token_truncated_is_weights(x, torch.ones(1, 2))
        with self.assertRaises(ValueError):
            token_truncated_is_weights(x, torch.full_like(x, .5))
        with self.assertRaises(TypeError):
            token_truncated_is_weights(x.long(), mask)
        with self.assertRaises(ValueError):
            token_truncated_is_weights(
                torch.full_like(x, float("nan")), mask
            )


if __name__ == "__main__":
    unittest.main()
