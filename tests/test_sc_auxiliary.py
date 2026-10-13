"""CPU tests for packed auxiliary log-prob computation."""
import unittest
import torch

from agentic_text2sql.algorithms.sc_auxiliary import (
    shifted_auxiliary_ids,
    auxiliary_mean_logp_packed,
)


def fixture():
    ids = torch.tensor([
        [-1, -1, -1, -1],
        [1, 1, 2, 3],
        [0, 2, 2, 4],
        [-1, -1, -1, -1],
        [4, 4, 4, 4],
        [-1, -1, -1, -1],
        [0, 1, 3, 4],
    ], dtype=torch.long)
    offsets = torch.tensor([0, 3, 7], dtype=torch.long)
    generator = torch.Generator().manual_seed(2026)
    logits = torch.randn(
        7, 5, generator=generator, dtype=torch.float64
    ).requires_grad_()
    return logits, ids, offsets


def reference(logits, ids, offsets):
    output = []
    for start, end in zip(offsets[:-1].tolist(), offsets[1:].tolist()):
        for position in range(start, end):
            if position + 1 < end and bool((ids[position + 1] >= 0).all()):
                value = logits[position].log_softmax(-1)[
                    ids[position + 1]
                ].mean()
            else:
                value = logits[position, 0] * 0.0
            output.append(value)
    return torch.stack(output)


class SCAuxiliaryTests(unittest.TestCase):
    def test_causal_shift_and_sequence_boundary(self):
        _, ids, offsets = fixture()
        expected = torch.tensor([
            [1, 1, 2, 3],
            [0, 2, 2, 4],
            [-1, -1, -1, -1],
            [4, 4, 4, 4],
            [-1, -1, -1, -1],
            [0, 1, 3, 4],
            [-1, -1, -1, -1],
        ])
        before = ids.clone()
        actual = shifted_auxiliary_ids(ids, offsets)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(ids, before)

    def test_forward_matches_independent_reference(self):
        logits, ids, offsets = fixture()
        actual = auxiliary_mean_logp_packed(logits, ids, offsets, 2)
        expected = reference(logits, ids, offsets)
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=0)

    def test_gradient_matches_independent_reference(self):
        logits, ids, offsets = fixture()
        weights = torch.tensor(
            [1., -.3, 2., .7, -1., .2, 3.], dtype=torch.float64
        )
        actual = auxiliary_mean_logp_packed(logits, ids, offsets, 2)
        expected = reference(logits, ids, offsets)
        ga = torch.autograd.grad((actual * weights).sum(), logits)[0]
        ge = torch.autograd.grad((expected * weights).sum(), logits)[0]
        torch.testing.assert_close(ga, ge, atol=1e-12, rtol=0)

    def test_absent_positions_have_zero_gradient(self):
        logits, ids, offsets = fixture()
        actual = auxiliary_mean_logp_packed(logits, ids, offsets)
        gradient = torch.autograd.grad(actual.sum(), logits)[0]
        self.assertEqual(torch.count_nonzero(gradient[[2, 4, 6]]).item(), 0)

    def test_duplicate_samples_preserved(self):
        logits, ids, offsets = fixture()
        actual = auxiliary_mean_logp_packed(logits, ids, offsets)
        logp = logits[0].log_softmax(-1)
        expected = (2 * logp[1] + logp[2] + logp[3]) / 4
        torch.testing.assert_close(actual[0], expected, atol=1e-12, rtol=0)

    def test_chunk_size_and_no_grad_consistency(self):
        logits, ids, offsets = fixture()
        a = auxiliary_mean_logp_packed(logits, ids, offsets, 1)
        b = auxiliary_mean_logp_packed(logits, ids, offsets, 32)
        with torch.no_grad():
            c = auxiliary_mean_logp_packed(logits, ids, offsets, 2)
        torch.testing.assert_close(a, b, atol=1e-12, rtol=0)
        torch.testing.assert_close(a, c, atol=1e-12, rtol=0)

    def test_all_absent(self):
        logits, ids, offsets = fixture()
        output = auxiliary_mean_logp_packed(
            logits, torch.full_like(ids, -1), offsets
        )
        self.assertEqual(torch.count_nonzero(output).item(), 0)
        gradient = torch.autograd.grad(output.sum(), logits)[0]
        self.assertEqual(torch.count_nonzero(gradient).item(), 0)

    def test_invalid_inputs_rejected(self):
        logits, ids, offsets = fixture()
        partial = ids.clone()
        partial[1, 0] = -1
        invalid_sentinel = ids.clone()
        invalid_sentinel[1] = -2
        out_of_vocab = ids.clone()
        out_of_vocab[1] = 5

        cases = [
            (ids[:, :3], offsets, 32),
            (ids.float(), offsets, 32),
            (partial, offsets, 32),
            (invalid_sentinel, offsets, 32),
            (out_of_vocab, offsets, 32),
            (ids, torch.tensor([0, 3, 3, 7]), 32),
            (ids, torch.tensor([0, 6]), 32),
            (ids, offsets, 0),
        ]
        for index, (candidate, boundaries, chunk) in enumerate(cases):
            with self.subTest(case=index):
                with self.assertRaises((ValueError, TypeError)):
                    auxiliary_mean_logp_packed(
                        logits, candidate, boundaries, chunk
                    )


if __name__ == "__main__":
    unittest.main()
