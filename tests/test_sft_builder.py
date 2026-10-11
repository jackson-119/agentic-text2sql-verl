import contextlib
import io
import json
import random
import sqlite3
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from agentic_text2sql.sft import builder as b


class ToyTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return [1] * (len(messages) * 8)


class SFTBuilderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.database = self.root / "raw" / "database"
        self.make_db("toy")

    def make_db(self, db_id):
        directory = self.database / db_id
        directory.mkdir(parents=True)
        path = directory / (db_id + ".sqlite")
        connection = sqlite3.connect(path)
        connection.execute("CREATE TABLE items (id INTEGER, name TEXT)")
        connection.executemany(
            "INSERT INTO items VALUES (?, ?)", [(1, "a"), (2, "b")]
        )
        connection.commit()
        connection.close()
        return path

    def record(self, **overrides):
        args = dict(
            source_index=0,
            example={
                "db_id": "toy", "question": "List names",
                "query": "SELECT name FROM items",
                "sql": {"from": {"table_units": [["table_unit", 0]]}},
            },
            schema={"table_names_original": ["items"]},
            database_root=self.database, tokenizer=ToyTokenizer(),
            split="train", rng=random.Random(42), max_length=4096,
            max_schema_tables=4, sample_rows=2, max_result_rows=20,
            recovery_ratio=0,
        )
        args.update(overrides)
        return b.build_record(**args)

    def test_standard_trajectory(self):
        row = self.record()
        messages = json.loads(row["messages_json"])
        calls = [
            m["tool_calls"][0]["function"]["name"]
            for m in messages if m.get("tool_calls")
        ]
        self.assertEqual(calls, ["list_tables", "get_table_schema", "execute_sql"])
        self.assertEqual(row["trajectory_type"], "standard_success")
        self.assertEqual(row["assistant_trainable_turns"], 4)
        payload = json.loads(messages[-2]["content"])
        self.assertEqual(payload["rows"], [["a"], ["b"]])
        self.assertEqual(messages[-1]["content"], "FINAL_SQL: SELECT name FROM items")

    def test_recovery_is_unsupervised_context(self):
        row = self.record(recovery_ratio=1)
        messages = json.loads(row["messages_json"])
        masked = [m for m in messages if m.get("trainable") is False]
        self.assertEqual(len(masked), 1)
        self.assertEqual(
            masked[0]["tool_calls"][0]["function"]["arguments"]["table_name"],
            "__missing_table__",
        )
        self.assertEqual(row["trajectory_type"], "schema_error_recovery")
        self.assertEqual(row["assistant_trainable_turns"], 4)

    def test_validation_has_no_recovery(self):
        row = self.record(split="validation", recovery_ratio=1)
        self.assertEqual(row["trajectory_type"], "standard_success")

    def test_overlength_rejected(self):
        with self.assertRaisesRegex(ValueError, "Overlong"):
            self.record(max_length=1)

    def test_table_id_extraction_and_limits(self):
        structure = {
            "a": [["table_unit", 1], ["table_unit", 0]],
            "b": {"nested": [["table_unit", 1]]},
        }
        self.assertEqual(b.extract_table_ids(structure), [1, 0])
        with self.assertRaisesRegex(ValueError, "Too many"):
            self.record(max_schema_tables=0)

    def test_readonly_and_result_truncation(self):
        path = b.database_path(self.database, "toy")
        before = path.read_bytes()
        connection = b.open_database(path)
        try:
            with self.assertRaises(sqlite3.OperationalError):
                connection.execute("DELETE FROM items")
            with self.assertRaises(ValueError):
                b.execute_response(connection, "DELETE FROM items", 20)
            result = b.execute_response(connection, "SELECT * FROM items", 1)
            self.assertTrue(result["truncated"])
            self.assertEqual(result["returned_rows"], 1)
        finally:
            connection.close()
        self.assertEqual(path.read_bytes(), before)

    def test_unsafe_database_and_output_rejected(self):
        with self.assertRaises(ValueError):
            b.database_path(self.database, "../toy")
        args = types.SimpleNamespace(
            raw_dir=self.root / "raw", model_path=self.root / "model",
            output_dir=self.root / "raw" / "output",
        )
        with self.assertRaises(ValueError):
            b.validate_output(args)
        args.output_dir = self.root / "output"
        args.output_dir.mkdir()
        (args.output_dir / "keep").touch()
        with self.assertRaises(RuntimeError):
            b.validate_output(args)

    def test_cli_rejects_invalid_parameters(self):
        base = ["--raw-dir", "raw", "--model-path", "model", "--output-dir", "out"]
        for extra in (
            ["--train-size", "0"], ["--sample-rows", "-1"],
            ["--recovery-ratio", "nan"], ["--validation-db-ratio", "1"],
        ):
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    b.parse_args(base + extra)

    def test_main_split_and_repeatability_with_mock_io(self):
        examples = []
        schemas = []
        for index in range(8):
            db_id = "db_" + str(index)
            self.make_db(db_id)
            examples.append({
                "db_id": db_id, "question": "List names",
                "query": "SELECT name FROM items",
                "sql": {"from": {"table_units": [["table_unit", 0]]}},
            })
            schemas.append({"db_id": db_id, "table_names_original": ["items"]})

        raw_dir = self.root / "raw"
        (raw_dir / "train_spider.json").write_text(json.dumps(examples))
        (raw_dir / "tables.json").write_text(json.dumps(schemas))
        model = self.root / "model"
        model.mkdir()
        written = {}

        class FakeDataset:
            def __init__(self, rows):
                self.rows = rows

            @classmethod
            def from_list(cls, rows):
                return cls(rows)

            def to_parquet(self, path):
                written[path] = self.rows

        datasets = types.ModuleType("datasets")
        datasets.Dataset = FakeDataset
        transformers = types.ModuleType("transformers")
        transformers.AutoTokenizer = types.SimpleNamespace(
            from_pretrained=lambda *a, **k: ToyTokenizer()
        )

        outputs = []
        for run in range(2):
            output = self.root / ("out_" + str(run))
            argv = [
                "builder", "--raw-dir", str(raw_dir), "--model-path", str(model),
                "--output-dir", str(output), "--train-size", "3",
                "--validation-size", "2", "--validation-db-ratio", "0.25",
                "--recovery-ratio", "0.5",
            ]
            with patch.dict("sys.modules", {
                "datasets": datasets, "transformers": transformers,
            }):
                with patch("sys.argv", argv):
                    with contextlib.redirect_stdout(io.StringIO()):
                        b.main()
            train = written[str(output / "train.parquet")]
            val = written[str(output / "validation.parquet")]
            self.assertEqual((len(train), len(val)), (3, 2))
            self.assertFalse({r["db_id"] for r in train} & {r["db_id"] for r in val})
            self.assertFalse(
                {r["source_index"] for r in train}
                & {r["source_index"] for r in val}
            )
            self.assertTrue(all(r["trajectory_type"] == "standard_success" for r in val))
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["train_size"], 3)
            outputs.append((train, val))
        self.assertEqual(outputs[0], outputs[1])


if __name__ == "__main__":
    unittest.main()
