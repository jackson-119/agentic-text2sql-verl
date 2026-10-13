"""Independent branch checks for the reward composition layer."""
import unittest

from agentic_text2sql.rewards.correctness_gated import compose_reward


def eligible_metrics():
    return {
        "execution_correct": 1.0,
        "format_compliance": 1.0,
        "answer_found": 1.0,
        "tool_called": 1.0,
        "tool_protocol_valid": 1.0,
        "malformed_tool_call": 0.0,
        "stopped_after_tool_error": 0.0,
        "list_tables_succeeded": 1.0,
        "get_table_schema_succeeded": 1.0,
        "execute_sql_succeeded": 1.0,
        "ordered_tool_chain": 1.0,
        "v3_final_no_duplicate_execute": 1.0,
    }


class CorrectnessGatedTests(unittest.TestCase):
    def test_incorrect_is_uniform(self):
        metrics = eligible_metrics()
        metrics["execution_correct"] = 0.0
        metrics["v3_duplicate_execute_count"] = 100.0
        result = compose_reward(metrics)
        self.assertEqual(result["score"], -0.1)
        self.assertEqual(result["v10_process_bonus"], 0.0)
        self.assertEqual(result["v10_correct_penalty"], 0.0)

    def test_complete_process(self):
        result = compose_reward(eligible_metrics())
        self.assertAlmostEqual(result["score"], 1.0)
        self.assertAlmostEqual(result["v10_terminal_reward"], 0.98)
        self.assertAlmostEqual(result["v10_process_bonus"], 0.02)

    def test_partial_process(self):
        metrics = eligible_metrics()
        metrics["ordered_tool_chain"] = 0.0
        metrics["v3_final_no_duplicate_execute"] = 0.0
        self.assertAlmostEqual(compose_reward(metrics)["score"], 0.992)

    def test_duplicate_penalty_cap(self):
        metrics = eligible_metrics()
        metrics["v3_duplicate_execute_count"] = 3.0
        metrics["v3_duplicate_schema_count"] = 3.0
        result = compose_reward(metrics)
        self.assertAlmostEqual(result["v10_correct_penalty"], 0.02)
        self.assertAlmostEqual(result["score"], 0.98)

    def test_formula_lower_bound(self):
        metrics = eligible_metrics()
        for name in (
            "list_tables_succeeded", "get_table_schema_succeeded",
            "execute_sql_succeeded", "ordered_tool_chain",
            "v3_final_no_duplicate_execute",
        ):
            metrics[name] = 0.0
        metrics["v3_duplicate_execute_count"] = 10.0
        # 合成指标验证公式边界，不宣称真实轨迹中必然可达。
        self.assertAlmostEqual(compose_reward(metrics)["score"], 0.96)

    def test_direct_correct(self):
        metrics = eligible_metrics()
        metrics["tool_called"] = 0.0
        self.assertEqual(compose_reward(metrics)["score"], 0.9)

    def test_invalid_protocol_precedes_fallback(self):
        metrics = eligible_metrics()
        metrics["malformed_tool_call"] = 1.0
        metrics["format_compliance"] = 0.0
        self.assertEqual(compose_reward(metrics)["score"], 0.8)

    def test_non_strict_correct_fallback(self):
        metrics = eligible_metrics()
        metrics["format_compliance"] = 0.0
        self.assertEqual(compose_reward(metrics)["score"], 0.9)

    def test_stopped_after_error_fallback(self):
        metrics = eligible_metrics()
        metrics["stopped_after_tool_error"] = 1.0
        self.assertEqual(compose_reward(metrics)["score"], 0.9)

    def test_missing_metrics(self):
        self.assertEqual(compose_reward({})["score"], -0.1)

    def test_input_not_mutated(self):
        metrics = eligible_metrics()
        original = dict(metrics)
        compose_reward(metrics)
        self.assertEqual(metrics, original)


if __name__ == "__main__":
    unittest.main(verbosity=2)
