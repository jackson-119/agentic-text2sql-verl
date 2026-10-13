import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from agentic_text2sql.training import launch_grpo as m

class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        source = self.root / "input"
        source.write_text("fixture")
        self.item = {
            "command": ["never-executed"], "cwd": str(self.root),
            "run_directory": str(self.root),
            "environment": {"VERL_SC_ENABLED": "0"},
            "input_hashes": {str(source): m.sha(source)},
        }
        self.gpu = patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"})
        self.gpu.start()
        self.addCleanup(self.gpu.stop)

    def test_default_dry_run(self):
        with patch.object(m, "plan", return_value=self.item), patch.object(m, "execute_plan") as call:
            self.assertEqual(m.launch("fixture")["mode"], "dry_run")
            call.assert_not_called()

    def test_environment_and_nonzero_exit(self):
        with patch.object(m.subprocess, "run", return_value=subprocess.CompletedProcess([], 7)) as call:
            result = m.execute_plan(self.item)
        self.assertEqual(result["returncode"], 7)
        self.assertEqual(call.call_args.kwargs["env"]["VERL_SC_ENABLED"], "0")
        self.assertEqual(call.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"], "0")
        saved = json.loads((self.root / "launch_result.json").read_text())
        self.assertEqual(saved["returncode"], 7)

    def test_duplicate_launch_rejected(self):
        (self.root / "launch_claim.json").write_text("{}")
        with self.assertRaises(FileExistsError):
            m.execute_plan(self.item)

    def test_gpu_required(self):
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}):
            with self.assertRaises(ValueError):
                m.execute_plan(self.item)
        self.assertFalse((self.root / "launch_claim.json").exists())

    def test_changed_input_rejected_before_spawn(self):
        (self.root / "input").write_text("changed")
        with patch.object(m.subprocess, "run") as call:
            with self.assertRaises(ValueError):
                m.execute_plan(self.item)
            call.assert_not_called()

    def test_spawn_failure_recorded(self):
        with patch.object(m.subprocess, "run", side_effect=OSError("fixture")):
            with self.assertRaises(OSError):
                m.execute_plan(self.item)
        saved = json.loads((self.root / "launch_result.json").read_text())
        self.assertIn("OSError", saved["launcher_error"])

if __name__ == "__main__":
    unittest.main()
