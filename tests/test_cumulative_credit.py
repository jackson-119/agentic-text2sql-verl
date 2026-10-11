import unittest
import torch
from agentic_text2sql.algorithms.cumulative_credit import (
    compose_turn_aware_advantages as compose,
    infer_assistant_turn_ids, select_advantages,
    sequence_mean_token_mean,
)

class CumulativeCreditTests(unittest.TestCase):
    def base(self, **updates):
        args = dict(
            terminal_scores=torch.tensor([0.98, -0.1], dtype=torch.float64),
            response_mask=torch.tensor([[1,0,1,0,1,0,1,0]] * 2),
            group_ids=["a", "a"],
            process_turn_scores=torch.tensor(
                [[.004,.004,.008,.004], [9,9,9,9]], dtype=torch.float64),
            execution_correct=torch.tensor([True, False]),
        )
        args.update(updates)
        return compose(**args)

    def test_cumulative_and_terminal_sum(self):
        out = self.base()
        expected = torch.tensor([.02,0,.016,0,.012,0,.004,0], dtype=torch.float64)
        torch.testing.assert_close(out.process_advantages[0], expected)
        torch.testing.assert_close(out.process_advantages[1], torch.zeros_like(expected))
        mask = torch.tensor([1,0,1,0,1,0,1,0], dtype=torch.float64)
        torch.testing.assert_close(out.terminal_advantages[0], .54 * mask)
        torch.testing.assert_close(out.terminal_advantages[1], -.54 * mask)
        torch.testing.assert_close(out.advantages,
            out.terminal_advantages + out.process_advantages + out.format_advantages)

    def test_mask_and_inferred_turns(self):
        out = self.base()
        self.assertEqual(out.turn_ids[0].tolist(), [1,0,2,0,3,0,4,0])
        self.assertTrue(torch.equal(out.advantages[:,1::2],
                                    torch.zeros_like(out.advantages[:,1::2])))

    def test_local_signal_independent_of_peer(self):
        a = self.base()
        b = self.base(terminal_scores=torch.tensor([.98,.9], dtype=torch.float64),
                      execution_correct=torch.tensor([True,True]))
        torch.testing.assert_close(a.process_advantages[0], b.process_advantages[0])

    def test_missing_turn_removed_before_cumsum(self):
        out = compose(
            torch.tensor([.98]), torch.tensor([[1,0,1]]), ["a"],
            process_turn_scores=torch.tensor([[1.,99.,3.]]),
            execution_correct=torch.tensor([True]),
            turn_ids=torch.tensor([[1,0,3]]),
        )
        torch.testing.assert_close(out.process_advantages, torch.tensor([[4.,0.,3.]]))

    def test_future_penalty_and_coefficient(self):
        scores = torch.tensor([[1.,0.,0.,-2.], [0.,0.,0.,0.]])
        out = self.base(process_turn_scores=scores, process_coefficient=.5)
        expected = torch.tensor([-.5,0,-1.,0,-1.,0,-1.,0], dtype=torch.float64)
        torch.testing.assert_close(out.process_advantages[0], expected)

    def test_format_stays_local_and_not_correctness_gated(self):
        out = self.base(format_turn_scores=torch.tensor(
            [[1.,0.,0.,-.02], [0.,-.03,0.,0.]]))
        expected = torch.tensor(
            [[0,0,0,0,0,0,-.02,0], [0,0,-.03,0,0,0,0,0]], dtype=torch.float64)
        torch.testing.assert_close(out.format_advantages, expected)

    def test_process_normalization_rejected(self):
        with self.assertRaises(ValueError):
            self.base(normalize_process_by_std=True)

    def test_singleton_and_all_wrong(self):
        singleton = self.base(group_ids=["a","b"])
        torch.testing.assert_close(singleton.terminal_advantages,
                                    torch.zeros_like(singleton.terminal_advantages))
        self.assertGreater(singleton.process_advantages[0].abs().sum().item(), 0)
        wrong = self.base(terminal_scores=torch.tensor([-.1,-.1], dtype=torch.float64),
                          execution_correct=torch.tensor([False,False]))
        torch.testing.assert_close(wrong.advantages, torch.zeros_like(wrong.advantages))

    def test_legacy_and_inputs_unchanged(self):
        scores = torch.tensor([[1.,2.,3.,4.], [5.,6.,7.,8.]])
        saved = scores.clone()
        self.base(process_turn_scores=scores)
        torch.testing.assert_close(scores, saved)
        legacy = torch.ones(2,3)
        self.assertIs(select_advantages(False, legacy), legacy)

    def test_sequence_mean_reduction(self):
        loss = torch.tensor([[2.,4.,99.], [6.,99.,99.]])
        mask = torch.tensor([[1,1,0], [1,0,0]])
        torch.testing.assert_close(sequence_mean_token_mean(loss, mask), torch.tensor(4.5))
        self.assertEqual(sequence_mean_token_mean(loss, torch.zeros_like(mask)).item(), 0)

if __name__ == "__main__":
    unittest.main()
