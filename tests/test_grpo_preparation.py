import copy
import unittest
from agentic_text2sql.training.prepare_grpo import (
    BASELINE_ENV, adapt_ray, require_baseline,
)

class GRPOPreparationTests(unittest.TestCase):
    def config(self):
        return {
            "ray_init": {"num_cpus": 8, "_temp_dir": "/tmp/toy"},
            "ray_kwargs": {"ray_init": {"num_cpus": 8}},
            "algorithm": {
                "adv_estimator": "grpo",
                "rollout_correction": {"rollout_is": None},
            },
            "trainer": {
                "nnodes": 1, "n_gpus_per_node": 1,
                "resume_mode": "disable", "use_v1": False,
            },
            "reward": {"unused": "???"},
        }

    def test_actual_ray_location_and_input_preserved(self):
        cfg = self.config()
        before = copy.deepcopy(cfg)
        result = adapt_ray(cfg, {"PYTHONPATH": "/repo/src:/verl"})
        self.assertEqual(result["ray_kwargs"]["ray_init"]["_temp_dir"], "/tmp/toy")
        self.assertEqual(result["ray_kwargs"]["ray_init"]["address"], "local")
        self.assertEqual(cfg, before)
        self.assertEqual(result["algorithm"], cfg["algorithm"])

    def test_conflicting_ray_values_rejected(self):
        cfg = self.config()
        cfg["ray_kwargs"]["ray_init"]["num_cpus"] = 2
        with self.assertRaises(ValueError):
            adapt_ray(cfg, {})

    def test_baseline_overrides_existing_sc_flags(self):
        cfg = self.config()
        cfg["ray_kwargs"]["ray_init"]["runtime_env"] = {
            "env_vars": {"VERL_SC_ENABLED": "1", "VERL_SC_DIAG_DIR": "/bad"}
        }
        env = adapt_ray(cfg, {})["ray_kwargs"]["ray_init"]["runtime_env"]["env_vars"]
        for key, value in BASELINE_ENV.items():
            self.assertEqual(env[key], value)

    def test_unused_missing_field_preserved(self):
        result = adapt_ray(self.config(), {})
        self.assertEqual(result["reward"]["unused"], "???")

    def test_nonbaseline_and_external_cluster_rejected(self):
        cfg = self.config()
        require_baseline(cfg)
        cfg["algorithm"]["rollout_correction"]["rollout_is"] = "token"
        with self.assertRaises(ValueError):
            require_baseline(cfg)
        cfg = self.config()
        cfg["ray_kwargs"]["ray_init"]["address"] = "auto"
        with self.assertRaises(ValueError):
            adapt_ray(cfg, {})

if __name__ == "__main__":
    unittest.main()
