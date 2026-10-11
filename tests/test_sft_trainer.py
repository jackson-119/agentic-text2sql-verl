import contextlib
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import torch
from agentic_text2sql.sft import trainer as t


class SFTTrainerTests(unittest.TestCase):
    def argv(self):
        return [
            "--model-path", "model", "--train-file", "train.parquet",
            "--validation-file", "val.parquet", "--output-dir", "output",
            "--learning-rate", "0.000002",
        ]

    def test_cli_explicit_learning_rate_and_reporting(self):
        args = t.parse_args(self.argv())
        self.assertEqual(args.learning_rate, 0.000002)
        self.assertEqual(args.report_to, "none")

    def test_cli_rejects_missing_or_invalid_learning_rate(self):
        for argv in (
            self.argv()[:-2],
            self.argv()[:-1] + ["nan"],
            self.argv()[:-1] + ["0"],
            self.argv()[:-1] + ["-1"],
        ):
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    t.parse_args(argv)

    def test_collator_padding_and_labels(self):
        features = [
            {"input_ids": [1, 2, 3], "labels": [-100, 2, 3],
             "attention_mask": [1, 1, 1]},
            {"input_ids": [4], "labels": [4], "attention_mask": [1]},
        ]
        batch = t.AgentSFTCollator(99)(features)
        self.assertEqual(tuple(batch["input_ids"].shape), (2, 8))
        self.assertEqual(batch["input_ids"][1].tolist(), [4] + [99] * 7)
        self.assertEqual(batch["labels"][1].tolist(), [4] + [-100] * 7)
        self.assertEqual(batch["attention_mask"][1].tolist(), [1] + [0] * 7)
        self.assertEqual(batch["labels"].dtype, torch.long)
        self.assertEqual(features[0]["labels"], [-100, 2, 3])

    def test_dataset_delegates_mask_and_excludes_metadata(self):
        row = {
            "messages_json": json.dumps([{"role": "assistant", "content": "x"}]),
            "tools_json": "[]", "trajectory_id": "toy",
        }
        item = {"input_ids": [1, 2], "labels": [-100, 2],
                "attention_mask": [1, 1]}
        fake = types.ModuleType("datasets")
        fake.load_dataset = lambda *a, **k: [row]
        with patch.dict("sys.modules", {"datasets": fake}):
            with patch.object(t, "tokenize_with_mask", return_value=item.copy()) as fn:
                with contextlib.redirect_stdout(io.StringIO()):
                    ds = t.AgentSFTDataset(Path("toy.parquet"), object(), 64)
        self.assertEqual(len(ds), 1)
        self.assertEqual(ds[0], item)
        self.assertNotIn("trajectory_id", ds[0])
        self.assertEqual(fn.call_args.args[3], 64)

    def test_empty_dataset_rejected(self):
        fake = types.ModuleType("datasets")
        fake.load_dataset = lambda *a, **k: []
        with patch.dict("sys.modules", {"datasets": fake}):
            with self.assertRaises(RuntimeError):
                t.AgentSFTDataset(Path("empty.parquet"), None, 64)

    def test_output_guard_and_distinct_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "model").mkdir()
            (root / "train").touch()
            (root / "val").touch()
            args = types.SimpleNamespace(
                model_path=root / "model", train_file=root / "train",
                validation_file=root / "val", output_dir=root / "out",
            )
            with patch.dict("os.environ", {"WORLD_SIZE": "1"}):
                with patch.object(t.torch.cuda, "device_count", return_value=1):
                    t.validate_run_paths(args)
                    self.assertFalse(args.output_dir.exists())
                    args.output_dir.mkdir()
                    (args.output_dir / "existing").touch()
                    with self.assertRaises(RuntimeError):
                        t.validate_run_paths(args)
                    args.output_dir = root / "fresh"
                    args.validation_file = args.train_file
                    with self.assertRaises(ValueError):
                        t.validate_run_paths(args)

    def test_non_single_process_rejected(self):
        with patch.dict("os.environ", {"WORLD_SIZE": "2"}):
            with self.assertRaises(RuntimeError):
                t.validate_run_paths(types.SimpleNamespace())

    def test_no_visible_gpu_rejected(self):
        with patch.dict("os.environ", {"WORLD_SIZE": "1"}):
            with patch.object(t.torch.cuda, "device_count", return_value=0):
                with self.assertRaises(RuntimeError):
                    t.validate_run_paths(types.SimpleNamespace())


if __name__ == "__main__":
    unittest.main()
