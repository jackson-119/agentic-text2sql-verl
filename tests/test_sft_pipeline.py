"""Real Parquet integration; synthetic SQLite and tokenizer, no model."""
import hashlib
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from datasets import load_dataset

from agentic_text2sql.sft import builder
from agentic_text2sql.sft.trainer import AgentSFTDataset, AgentSFTCollator


class SyntheticTokenizer:
    pad_token_id = 0
    roles = {"system": 1, "user": 2, "assistant": 3, "tool": 4}

    def header(self, role):
        return [10, self.roles[role]]

    def body(self, message):
        payload = {
            key: value for key, value in message.items()
            if key not in ("role", "trainable")
        }
        text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return [32 + value for value in text.encode("utf-8")] + [11]

    def apply_chat_template(
        self, messages, tools=None, tokenize=True,
        add_generation_prompt=False, **kwargs
    ):
        if not tokenize:
            raise ValueError("Synthetic tokenizer only supports token output")
        result = []
        for message in messages:
            result.extend(self.header(message["role"]))
            result.extend(self.body(message))
        if add_generation_prompt:
            result.extend(self.header("assistant"))
        return result


class SFTPipelineTests(unittest.TestCase):
    def test_real_parquet_builder_dataset_collator(self):
        tokenizer = SyntheticTokenizer()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw"
            raw.mkdir()
            model = root / "tokenizer"
            model.mkdir()
            cache = root / "cache"
            output = root / "output"
            examples, schemas, hashes = [], [], {}

            for index in range(8):
                db_id = f"toy_{index}"
                db_dir = raw / "database" / db_id
                db_dir.mkdir(parents=True)
                db_path = db_dir / f"{db_id}.sqlite"
                connection = sqlite3.connect(db_path)
                try:
                    connection.execute(
                        "CREATE TABLE items (id INTEGER, name TEXT)"
                    )
                    connection.executemany(
                        "INSERT INTO items VALUES (?, ?)",
                        [(1, "a"), (2, "longer value")],
                    )
                    connection.commit()
                finally:
                    connection.close()
                hashes[db_path] = hashlib.sha256(db_path.read_bytes()).hexdigest()
                examples.append({
                    "db_id": db_id,
                    "question": "List names " + "x" * index,
                    "query": "SELECT name FROM items ORDER BY id",
                    "sql": {"from": {"table_units": [["table_unit", 0]]}},
                })
                schemas.append({
                    "db_id": db_id,
                    "table_names_original": ["items"],
                })

            (raw / "train_spider.json").write_text(json.dumps(examples))
            (raw / "tables.json").write_text(json.dumps(schemas))
            argv = [
                "builder", "--raw-dir", str(raw),
                "--model-path", str(model), "--output-dir", str(output),
                "--train-size", "3", "--validation-size", "2",
                "--validation-db-ratio", "0.25", "--recovery-ratio", "1",
                "--max-length", "20000",
            ]

            # Only tokenizer loading is replaced; SQLite and Parquet are real.
            with patch(
                "transformers.AutoTokenizer.from_pretrained",
                return_value=tokenizer,
            ):
                with patch("sys.argv", argv):
                    builder.main()

            loaded = {}
            features = []
            masked_recovery_count = 0

            for split, count in (("train", 3), ("validation", 2)):
                path = output / f"{split}.parquet"
                self.assertTrue(path.is_file())
                table = load_dataset(
                    "parquet", data_files=str(path), split="train",
                    cache_dir=str(cache),
                )
                loaded[split] = list(table)
                self.assertEqual(len(table), count)

                # Keep the production Dataset loader, but isolate its cache.
                def cached_load(*args, **kwargs):
                    kwargs["cache_dir"] = str(cache)
                    return load_dataset(*args, **kwargs)

                with patch("datasets.load_dataset", side_effect=cached_load):
                    dataset = AgentSFTDataset(path, tokenizer, 20000)

                self.assertEqual(len(dataset), count)
                for index, row in enumerate(table):
                    messages = json.loads(row["messages_json"])
                    self.assertEqual(
                        row["trajectory_type"],
                        "schema_error_recovery" if split == "train"
                        else "standard_success",
                    )

                    # Independently construct expected labels from message
                    # pieces, without calling tokenize_with_mask().
                    ids, labels = [], []
                    for message in messages:
                        header = tokenizer.header(message["role"])
                        body = tokenizer.body(message)
                        supervised = (
                            message["role"] == "assistant"
                            and message.get("trainable", True)
                        )
                        ids.extend(header + body)
                        labels.extend([-100] * len(header))
                        labels.extend(body if supervised else [-100] * len(body))
                        if message.get("trainable") is False:
                            masked_recovery_count += 1

                    item = dataset[index]
                    self.assertEqual(item["input_ids"], ids)
                    self.assertEqual(item["labels"], labels)
                    self.assertEqual(item["attention_mask"], [1] * len(ids))
                    self.assertEqual(row["token_count"], len(ids))
                    self.assertTrue(any(value != -100 for value in labels))
                    features.append(item)

            self.assertEqual(masked_recovery_count, 3)
            for key in ("db_id", "source_index", "trajectory_id"):
                self.assertFalse(
                    {row[key] for row in loaded["train"]}
                    & {row[key] for row in loaded["validation"]}
                )

            batch = AgentSFTCollator(tokenizer.pad_token_id)(features)
            self.assertEqual(batch["input_ids"].shape[0], 5)
            self.assertEqual(batch["input_ids"].shape[1] % 8, 0)
            self.assertEqual(batch["labels"].dtype, torch.long)

            for index, feature in enumerate(features):
                length = len(feature["input_ids"])
                self.assertEqual(
                    batch["labels"][index, :length].tolist(), feature["labels"]
                )
                self.assertTrue(torch.all(batch["labels"][index, length:] == -100))
                self.assertTrue(torch.all(batch["attention_mask"][index, length:] == 0))
                self.assertTrue(torch.all(batch["input_ids"][index, length:] == 0))

            for path, before in hashes.items():
                self.assertEqual(
                    hashlib.sha256(path.read_bytes()).hexdigest(), before
                )


if __name__ == "__main__":
    unittest.main()
