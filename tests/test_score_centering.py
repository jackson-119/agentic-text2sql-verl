"""CPU reference tests, not online verl integration tests."""
import unittest
import torch

from agentic_text2sql.algorithms.score_centering import apply_sc_correction


def fixture():
    generator = torch.Generator(device="cpu").manual_seed(1729)
    logits = torch.randn(
        1, 7, 5, generator=generator, dtype=torch.float64
    ).requires_grad_()
    selected = torch.tensor([[0, 1, 2, 3, 4, 0, 1]])
    auxiliary = torch.tensor([[
        [0, 0, 1, 2], [1, 2, 2, 4], [0, 1, 2, 3],
        [3, 3, 4, 4], [0, 1, 1, 4], [0, 2, 3, 4],
        [1, 1, 1, 1],
    ]])
    mask = torch.tensor([[True, True, True, True, True, True, False]])
    log_all = logits.log_softmax(-1)
    logp = log_all.gather(-1, selected.unsqueeze(-1)).squeeze(-1)
    mean_logp = log_all.gather(-1, auxiliary).mean(-1)

    ratio_target = torch.tensor(
        [[1., 1., 1.5, .5, 4., .4, 1.]], dtype=torch.float64
    )
    advantages = torch.tensor(
        [[1., -1., 1., -1., -1., 1., 1.]], dtype=torch.float64
    )
    old = logp.detach() - ratio_target.log()
    ratio = (logp - old).clamp(-20., 20.).exp()
    plain = -advantages * ratio
    clipped = -advantages * ratio.clamp(.8, 1.2)
    ordinary = torch.maximum(plain, clipped)
    dual = torch.minimum(-advantages * 3., ordinary)
    per_token = torch.where(advantages < 0, dual, ordinary)
    loss = (per_token * mask).sum() / mask.sum()
    coefficient = torch.tensor(
        [[-1., 1., 0., 0., 0., -.4, 0.]], dtype=torch.float64
    ) / 6.
    return logits, selected, auxiliary, mask, logp, mean_logp, loss, coefficient


class ScoreCenteringTests(unittest.TestCase):
    def test_reference_ppo_coefficient(self):
        _, _, _, _, logp, _, loss, expected = fixture()
        actual = torch.autograd.grad(loss, logp)[0]
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=0)

    def test_forward_loss_unchanged(self):
        _, _, _, mask, logp, mean, loss, _ = fixture()
        corrected, metrics = apply_sc_correction(loss, logp, mean, mask)
        self.assertEqual(corrected.item(), loss.item())
        self.assertEqual(metrics["actor/sc_forward_delta"], 0.0)

    def test_centered_logit_gradient(self):
        logits, chosen, auxiliary, mask, logp, mean, loss, c = fixture()
        corrected, _ = apply_sc_correction(loss, logp, mean, mask)
        actual = torch.autograd.grad(corrected, logits)[0]
        onehot = torch.nn.functional.one_hot(chosen, 5).double()
        frequency = torch.nn.functional.one_hot(
            auxiliary, 5
        ).double().mean(-2)
        expected = c.unsqueeze(-1) * (onehot - frequency)
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=0)

    def test_clipped_and_masked_positions_zero(self):
        logits, _, _, mask, logp, mean, loss, _ = fixture()
        corrected, _ = apply_sc_correction(loss, logp, mean, mask)
        gradient = torch.autograd.grad(corrected, logits)[0]
        inactive = gradient[:, [2, 3, 4, 6], :]
        torch.testing.assert_close(
            inactive, torch.zeros_like(inactive), atol=1e-12, rtol=0
        )
        self.assertGreater(gradient[:, [0, 1, 5], :].norm().item(), 0)

    def test_all_auxiliary_equal_selected_cancels(self):
        logits, _, _, mask, logp, _, loss, _ = fixture()
        corrected, _ = apply_sc_correction(loss, logp, logp, mask)
        gradient = torch.autograd.grad(corrected, logits)[0]
        torch.testing.assert_close(
            gradient, torch.zeros_like(gradient), atol=1e-12, rtol=0
        )

    def test_zero_coefficient(self):
        logits, _, _, mask, logp, mean, _, _ = fixture()
        corrected, _ = apply_sc_correction(
            (logp * 0).sum(), logp, mean, mask
        )
        gradient = torch.autograd.grad(corrected, logits)[0]
        self.assertEqual(torch.count_nonzero(gradient).item(), 0)

    def test_missing_or_detached_auxiliary_rejected(self):
        for absent in (True, False):
            with self.subTest(absent=absent):
                _, _, _, mask, logp, mean, loss, _ = fixture()
                with self.assertRaises(RuntimeError):
                    apply_sc_correction(
                        loss, logp, None if absent else mean.detach(), mask
                    )

    def test_invalid_shape_and_empty_mask_rejected(self):
        _, _, _, mask, logp, mean, loss, _ = fixture()
        with self.assertRaises(ValueError):
            apply_sc_correction(loss, logp, mean[:, :-1], mask)
        with self.assertRaises(ValueError):
            apply_sc_correction(loss, logp, mean, torch.zeros_like(mask))

    def test_nonfinite_auxiliary_handling(self):
        mask = torch.tensor([[True, False]])
        logp = torch.zeros(1, 2, dtype=torch.float64, requires_grad=True)
        mean = torch.tensor(
            [[-.5, float("nan")]], dtype=torch.float64, requires_grad=True
        )
        loss = logp[0, 0] * 2
        corrected, _ = apply_sc_correction(loss, logp, mean, mask)
        gradient = torch.autograd.grad(corrected, mean)[0]
        torch.testing.assert_close(
            gradient, torch.tensor([[-2., 0.]], dtype=torch.float64)
        )

        logp2 = torch.zeros(1, 2, dtype=torch.float64, requires_grad=True)
        bad = torch.tensor(
            [[float("nan"), 0.]], dtype=torch.float64, requires_grad=True
        )
        with self.assertRaises(FloatingPointError):
            apply_sc_correction(logp2[0, 0], logp2, bad, mask)

    def test_nonzero_masked_coefficient_rejected(self):
        logp = torch.zeros(1, 2, dtype=torch.float64, requires_grad=True)
        mean = torch.zeros(1, 2, dtype=torch.float64, requires_grad=True)
        with self.assertRaises(RuntimeError):
            apply_sc_correction(
                logp.sum(), logp, mean, torch.tensor([[True, False]])
            )


if __name__ == "__main__":
    unittest.main()
