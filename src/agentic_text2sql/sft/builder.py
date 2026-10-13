#!/usr/bin/env python
"""Build successful Spider tool-use trajectories for assistant-only SFT."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import re
import sqlite3
import statistics
import time
from typing import Any

# Dataset and tokenizer dependencies are imported only by main().


SYSTEM_PROMPT = """You are a Text-to-SQL agent with access to SQLite database tools.

Follow this workflow:
1. Call list_tables before guessing any table name.
2. Inspect every relevant table with get_table_schema.
3. Construct a read-only SQLite SELECT or WITH query.
4. Call execute_sql to verify the query.
5. If a tool returns an error, use the error information and continue.
6. End with exactly one line in this form:
FINAL_SQL: <one SQLite SELECT or WITH query>

Never invent tool responses. Never use INSERT, UPDATE, DELETE, DROP, ALTER, ATTACH, PRAGMA, or multiple SQL statements."""


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_tables",
            "description": (
                "List all user tables in the attached SQLite database."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_table_schema",
            "description": (
                "Inspect one table, including columns, keys and "
                "several sample rows."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "table_name": {
                        "type": "string",
                        "description": (
                            "Exact table name returned by list_tables."
                        ),
                    }
                },
                "required": ["table_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "execute_sql",
            "description": (
                "Execute one read-only SQLite SELECT query and "
                "return its columns and rows."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": (
                            "One SQLite SELECT or WITH query."
                        ),
                    }
                },
                "required": ["sql"],
            },
        },
    },
]


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-size", type=int, default=1000)
    parser.add_argument("--validation-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--max-schema-tables", type=int, default=4)
    parser.add_argument("--sample-rows", type=int, default=2)
    parser.add_argument("--max-result-rows", type=int, default=20)
    parser.add_argument("--recovery-ratio", type=float, default=0.15)
    parser.add_argument("--validation-db-ratio", type=float, default=0.15)

    args = parser.parse_args(argv)
    for name in (
        "train_size", "validation_size", "max_length",
        "max_schema_tables", "max_result_rows",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"{name} must be positive")
    if args.sample_rows < 0:
        parser.error("sample_rows must be nonnegative")
    if not 0 <= args.recovery_ratio <= 1:
        parser.error("recovery_ratio must be between 0 and 1")
    if not 0 < args.validation_db_ratio < 1:
        parser.error("validation_db_ratio must be between 0 and 1")
    return args


def json_text(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    )


def clean_messages(messages: list[dict]) -> list[dict]:
    return [
        {
            key: value
            for key, value in message.items()
            if key != "trainable"
        }
        for message in messages
    ]


def normalize_sql(sql: str) -> str:
    return re.sub(r"\s+", " ", sql.strip()).rstrip(";")


def normalize_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value

    if isinstance(value, bytes):
        return {
            "type": "bytes",
            "hex": value.hex()[:400],
        }

    text = str(value)

    if len(text) > 240:
        text = text[:237] + "..."

    return text


def quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def database_path(database_root: Path, db_id: str) -> Path:
    if not isinstance(db_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", db_id):
        raise ValueError("Unsafe database ID")
    root = database_root.resolve()
    path = (root / db_id / f"{db_id}.sqlite").resolve()
    if root not in path.parents:
        raise ValueError("Database path escapes configured root")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def open_database(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(
        path.resolve().as_uri() + "?mode=ro",
        uri=True,
        timeout=1.0,
    )
    connection.text_factory = lambda raw: raw.decode(
        "utf-8",
        errors="replace",
    )
    connection.execute("PRAGMA query_only = ON")
    return connection


def list_tables(connection: sqlite3.Connection) -> list[str]:
    rows = connection.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table'
          AND name NOT LIKE 'sqlite_%'
        ORDER BY name
        """
    ).fetchall()

    return [str(row[0]) for row in rows]


def list_tables_response(
    connection: sqlite3.Connection,
) -> dict[str, Any]:
    return {
        "ok": True,
        "tables": list_tables(connection),
    }


def schema_response(
    connection: sqlite3.Connection,
    table_name: str,
    sample_rows: int,
) -> dict[str, Any]:
    available = list_tables(connection)

    if table_name not in available:
        return {
            "ok": False,
            "error_type": "ValueError",
            "error": (
                f"Unknown table {table_name!r}. "
                f"Available tables: {available}"
            ),
        }

    create_row = connection.execute(
        """
        SELECT sql
        FROM sqlite_master
        WHERE type = 'table' AND name = ?
        """,
        (table_name,),
    ).fetchone()

    create_sql = (
        create_row[0]
        if create_row and create_row[0]
        else ""
    )

    pragma_rows = connection.execute(
        f"PRAGMA table_info({quote_identifier(table_name)})"
    ).fetchall()

    columns = [
        {
            "cid": int(row[0]),
            "name": str(row[1]),
            "type": str(row[2] or ""),
            "not_null": bool(row[3]),
            "default_value": normalize_value(row[4]),
            "primary_key": bool(row[5]),
        }
        for row in pragma_rows
    ]

    foreign_key_rows = connection.execute(
        f"PRAGMA foreign_key_list({quote_identifier(table_name)})"
    ).fetchall()

    foreign_keys = [
        {
            "id": int(row[0]),
            "sequence": int(row[1]),
            "referenced_table": str(row[2]),
            "from_column": str(row[3]),
            "to_column": str(row[4]),
            "on_update": str(row[5]),
            "on_delete": str(row[6]),
        }
        for row in foreign_key_rows
    ]

    cursor = connection.execute(
        f"SELECT * FROM {quote_identifier(table_name)} "
        f"LIMIT {int(sample_rows)}"
    )
    sample_columns = [
        item[0] for item in cursor.description
    ]
    rows = cursor.fetchall()

    sample_data = [
        {
            column: normalize_value(value)
            for column, value in zip(
                sample_columns,
                row,
                strict=True,
            )
        }
        for row in rows
    ]

    return {
        "ok": True,
        "table_name": table_name,
        "create_sql": create_sql,
        "columns": columns,
        "foreign_keys": foreign_keys,
        "sample_rows": sample_data,
    }


def execute_response(
    connection: sqlite3.Connection,
    sql: str,
    max_rows: int,
    timeout_seconds: float = 3.0,
) -> dict[str, Any]:
    normalized = normalize_sql(sql)

    if not re.match(r"(?is)^(SELECT|WITH)\b", normalized):
        raise ValueError("Gold SQL is not SELECT or WITH")

    deadline = time.monotonic() + timeout_seconds

    def stop_long_query() -> int:
        return int(time.monotonic() > deadline)

    connection.set_progress_handler(
        stop_long_query,
        1000,
    )

    try:
        cursor = connection.execute(normalized)

        if cursor.description is None:
            raise ValueError("SQL returned no result set")

        columns = [
            item[0] for item in cursor.description
        ]
        rows = cursor.fetchmany(max_rows + 1)
        truncated = len(rows) > max_rows
        rows = rows[:max_rows]

        return {
            "ok": True,
            "columns": columns,
            "rows": [
                [
                    normalize_value(value)
                    for value in row
                ]
                for row in rows
            ],
            "returned_rows": len(rows),
            "truncated": truncated,
        }
    finally:
        connection.set_progress_handler(None, 0)


def extract_table_ids(sql_structure: Any) -> list[int]:
    found: list[int] = []

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for child in value.values():
                visit(child)
            return

        if isinstance(value, list):
            if (
                len(value) >= 2
                and value[0] == "table_unit"
                and isinstance(value[1], int)
            ):
                if value[1] not in found:
                    found.append(value[1])

            for child in value:
                visit(child)

    visit(sql_structure)
    return found


def assistant_tool_call(
    name: str,
    arguments: dict[str, Any],
    *,
    trainable: bool = True,
) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": "",
        "trainable": trainable,
        "tool_calls": [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": arguments,
                },
            }
        ],
    }


def tool_response(
    name: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    return {
        "role": "tool",
        "name": name,
        "content": json_text(payload),
    }


def render_token_count(
    tokenizer,
    messages: list[dict],
) -> int:
    token_ids = tokenizer.apply_chat_template(
        clean_messages(messages),
        tools=TOOLS,
        tokenize=True,
        add_generation_prompt=False,
    )
    return len(token_ids)


def build_record(
    *,
    source_index: int,
    example: dict[str, Any],
    schema: dict[str, Any],
    database_root: Path,
    tokenizer,
    split: str,
    rng: random.Random,
    max_length: int,
    max_schema_tables: int,
    sample_rows: int,
    max_result_rows: int,
    recovery_ratio: float,
) -> dict[str, Any]:
    db_id = example["db_id"]
    question = example["question"].strip()
    gold_sql = normalize_sql(example["query"])

    table_ids = extract_table_ids(example["sql"])

    if not table_ids:
        raise ValueError("No table_unit found")

    if len(table_ids) > max_schema_tables:
        raise ValueError("Too many relevant tables")

    original_names = schema["table_names_original"]

    relevant_tables = []

    for table_id in table_ids:
        if table_id < 0 or table_id >= len(original_names):
            raise ValueError("Invalid table id")

        table_name = original_names[table_id]

        if table_name not in relevant_tables:
            relevant_tables.append(table_name)

    db_path = database_path(database_root, db_id)
    connection = open_database(db_path)

    try:
        available_tables = list_tables(connection)

        for table_name in relevant_tables:
            if table_name not in available_tables:
                raise ValueError(
                    f"Structured SQL table missing: {table_name}"
                )

        list_payload = list_tables_response(connection)
        schema_payloads = {
            table_name: schema_response(
                connection,
                table_name,
                sample_rows,
            )
            for table_name in relevant_tables
        }
        execution_payload = execute_response(
            connection,
            gold_sql,
            max_result_rows,
        )
    finally:
        connection.close()

    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": question,
        },
        assistant_tool_call(
            "list_tables",
            {},
        ),
        tool_response(
            "list_tables",
            list_payload,
        ),
    ]

    recovery_example = (
        split == "train"
        and rng.random() < recovery_ratio
    )

    if recovery_example:
        missing_table = "__missing_table__"

        messages.extend(
            [
                # Included as context but masked from SFT loss.
                assistant_tool_call(
                    "get_table_schema",
                    {
                        "table_name": missing_table,
                    },
                    trainable=False,
                ),
                tool_response(
                    "get_table_schema",
                    {
                        "ok": False,
                        "error_type": "ValueError",
                        "error": (
                            f"Unknown table {missing_table!r}. "
                            f"Available tables: {available_tables}"
                        ),
                    },
                ),
            ]
        )

    for table_name in relevant_tables:
        messages.extend(
            [
                assistant_tool_call(
                    "get_table_schema",
                    {
                        "table_name": table_name,
                    },
                ),
                tool_response(
                    "get_table_schema",
                    schema_payloads[table_name],
                ),
            ]
        )

    messages.extend(
        [
            assistant_tool_call(
                "execute_sql",
                {
                    "sql": gold_sql,
                },
            ),
            tool_response(
                "execute_sql",
                execution_payload,
            ),
            {
                "role": "assistant",
                "content": f"FINAL_SQL: {gold_sql}",
                "trainable": True,
            },
        ]
    )

    token_count = render_token_count(
        tokenizer,
        messages,
    )

    if token_count > max_length:
        raise ValueError(
            f"Overlong trajectory: {token_count}"
        )

    assistant_turns = sum(
        message["role"] == "assistant"
        and message.get("trainable", True)
        for message in messages
    )

    return {
        "trajectory_id": (
            f"spider_train_{source_index:05d}"
        ),
        "source_split": "spider_train",
        "source_index": source_index,
        "split": split,
        "db_id": db_id,
        "question": question,
        "gold_sql": gold_sql,
        "relevant_tables_json": json_text(
            relevant_tables
        ),
        "trajectory_type": (
            "schema_error_recovery"
            if recovery_example
            else "standard_success"
        ),
        "messages_json": json_text(messages),
        "tools_json": json_text(TOOLS),
        "token_count": token_count,
        "assistant_trainable_turns": assistant_turns,
    }



def validate_output(args):
    """Refuse overwriting artifacts or placing output inside input directories."""
    output = args.output_dir.resolve()
    for path in (args.raw_dir, args.model_path):
        resolved = path.resolve()
        if output == resolved or output in resolved.parents or resolved in output.parents:
            raise ValueError("Output directory overlaps an input directory")
    if args.output_dir.exists():
        if not args.output_dir.is_dir() or any(args.output_dir.iterdir()):
            raise RuntimeError("Output directory must be absent or empty")


def main() -> None:
    args = parse_args()
    validate_output(args)
    from datasets import Dataset
    from transformers import AutoTokenizer

    raw_dir = args.raw_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = raw_dir / "train_spider.json"
    tables_path = raw_dir / "tables.json"
    database_root = raw_dir / "database"

    for required in (
        train_path,
        tables_path,
        database_root,
        args.model_path,
    ):
        if not required.exists():
            raise FileNotFoundError(required)

    train_data = json.loads(
        train_path.read_text(encoding="utf-8")
    )
    tables_data = json.loads(
        tables_path.read_text(encoding="utf-8")
    )
    schema_by_db = {
        item["db_id"]: item
        for item in tables_data
    }

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=True,
    )

    rng = random.Random(args.seed)

    db_ids = sorted(
        {
            example["db_id"]
            for example in train_data
        }
    )
    rng.shuffle(db_ids)

    validation_db_count = max(
        1,
        round(len(db_ids) * args.validation_db_ratio),
    )
    validation_db_ids = set(
        db_ids[:validation_db_count]
    )

    candidates = list(enumerate(train_data))
    rng.shuffle(candidates)

    targets = {
        "train": args.train_size,
        "validation": args.validation_size,
    }
    records = {
        "train": [],
        "validation": [],
    }
    skipped = Counter()

    for source_index, example in candidates:
        split = (
            "validation"
            if example["db_id"] in validation_db_ids
            else "train"
        )

        if len(records[split]) >= targets[split]:
            continue

        schema = schema_by_db.get(example["db_id"])

        if schema is None:
            skipped["missing_schema"] += 1
            continue

        try:
            record = build_record(
                source_index=source_index,
                example=example,
                schema=schema,
                database_root=database_root,
                tokenizer=tokenizer,
                split=split,
                rng=rng,
                max_length=args.max_length,
                max_schema_tables=args.max_schema_tables,
                sample_rows=args.sample_rows,
                max_result_rows=args.max_result_rows,
                recovery_ratio=args.recovery_ratio,
            )
        except Exception as exc:
            skipped[
                f"{type(exc).__name__}: {str(exc)[:80]}"
            ] += 1
            continue

        records[split].append(record)

        if all(
            len(records[name]) >= targets[name]
            for name in targets
        ):
            break

    for split, target in targets.items():
        if len(records[split]) != target:
            raise RuntimeError(
                f"{split} only built "
                f"{len(records[split])}/{target}"
            )

    train_db_ids = {
        record["db_id"]
        for record in records["train"]
    }
    val_db_ids = {
        record["db_id"]
        for record in records["validation"]
    }

    if train_db_ids & val_db_ids:
        raise RuntimeError(
            "Train and validation databases overlap"
        )

    for split in ("train", "validation"):
        dataset = Dataset.from_list(records[split])
        output_path = output_dir / f"{split}.parquet"
        dataset.to_parquet(str(output_path))

        preview_path = (
            output_dir / f"{split}_preview.json"
        )
        preview_path.write_text(
            json.dumps(
                records[split][0],
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    manifest = {
        "version": 1,
        "source": str(train_path),
        "model_path": str(args.model_path),
        "seed": args.seed,
        "max_length": args.max_length,
        "train_size": len(records["train"]),
        "validation_size": len(records["validation"]),
        "train_database_count": len(train_db_ids),
        "validation_database_count": len(val_db_ids),
        "train_database_ids": sorted(train_db_ids),
        "validation_database_ids": sorted(val_db_ids),
        "recovery_ratio_requested": args.recovery_ratio,
        "recovery_train_count": sum(
            record["trajectory_type"]
            == "schema_error_recovery"
            for record in records["train"]
        ),
        "token_statistics": {
            split: {
                "min": min(
                    record["token_count"]
                    for record in records[split]
                ),
                "mean": statistics.fmean(
                    record["token_count"]
                    for record in records[split]
                ),
                "max": max(
                    record["token_count"]
                    for record in records[split]
                ),
            }
            for split in records
        },
        "skipped": dict(skipped),
    }

    (
        output_dir / "manifest.json"
    ).write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        )
    )
    print()
    print("train.parquet：", output_dir / "train.parquet")
    print(
        "validation.parquet：",
        output_dir / "validation.parquet",
    )


if __name__ == "__main__":
    main()
