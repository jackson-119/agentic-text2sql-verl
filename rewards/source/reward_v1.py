from __future__ import annotations

import json
import math
import os
import re
import sqlite3
import time
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import quote


MARKER_PATH = Path(
    "/home/archlab/yb/data/spider/SPIDER_ROOT.txt"
)

MAX_RESULT_ROWS = 50000
QUERY_TIMEOUT_SECONDS = 5.0

ACCURACY_WEIGHT = 0.90
EXECUTABLE_WEIGHT = 0.05
FORMAT_WEIGHT = 0.05


def _database_root() -> Path:
    configured = os.environ.get("SPIDER_DATABASE_ROOT")

    if configured:
        root = Path(configured)
    else:
        spider_root = Path(
            MARKER_PATH.read_text(encoding="utf-8").strip()
        )
        root = spider_root / "database"

    root = root.expanduser().resolve()

    if not root.is_dir():
        raise FileNotFoundError(
            f"Spider database directory does not exist: {root}"
        )

    return root


def _database_path(db_id: str) -> Path:
    if not isinstance(db_id, str):
        raise TypeError("db_id must be a string")

    if not re.fullmatch(r"[A-Za-z0-9_]+", db_id):
        raise ValueError(f"Unsafe db_id: {db_id!r}")

    root = _database_root()
    path = (root / db_id / f"{db_id}.sqlite").resolve()

    if root not in path.parents:
        raise ValueError("Database path escaped database root")

    if not path.is_file():
        raise FileNotFoundError(path)

    return path


def _remove_leading_comments(sql: str) -> str:
    return re.sub(
        r"(?is)^\s*(?:(?:--[^\n]*\n)|(?:/\*.*?\*/\s*))*",
        "",
        sql,
    )


def _validate_read_only_sql(sql: str) -> str:
    if not isinstance(sql, str) or not sql.strip():
        raise ValueError("SQL is empty")

    sql = sql.strip()
    cleaned = _remove_leading_comments(sql)

    if not re.match(r"(?is)^(SELECT|WITH)\b", cleaned):
        raise ValueError(
            "Only SELECT or WITH queries are allowed"
        )

    dangerous_function = re.search(
        r"(?is)\b("
        r"load_extension|readfile|writefile"
        r")\s*\(",
        cleaned,
    )

    if dangerous_function:
        raise ValueError(
            "Forbidden SQLite function: "
            f"{dangerous_function.group(1)}"
        )

    return sql


def _strip_code_fence(text: str) -> str:
    text = text.strip()

    if not text.startswith("```"):
        return text

    lines = text.splitlines()

    if lines:
        lines = lines[1:]

    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]

    return "\n".join(lines).strip()


def extract_final_sql(
    solution_str: str,
) -> tuple[str | None, bool, str]:
    if not isinstance(solution_str, str):
        return None, False, "none"

    text = solution_str.strip()

    if not text:
        return None, False, "none"

    strict_match = re.search(
        r"(?im)^\s*FINAL_SQL\s*:\s*"
        r"((?:SELECT|WITH)\b[^\r\n]*)\s*$",
        text,
    )

    strict_format = bool(
        strict_match
        and not text[strict_match.end():].strip()
    )

    label_matches = list(
        re.finditer(
            r"(?im)^\s*FINAL_SQL\s*:\s*",
            text,
        )
    )

    if label_matches:
        candidate = text[
            label_matches[-1].end():
        ].strip()

        candidate = _strip_code_fence(candidate)

        if candidate:
            return candidate, strict_format, "final_sql"

    fenced_matches = re.findall(
        r"(?is)```(?:sql|sqlite)?\s*(.*?)```",
        text,
    )

    for candidate in reversed(fenced_matches):
        candidate = candidate.strip()

        if re.match(
            r"(?is)^(SELECT|WITH)\b",
            _remove_leading_comments(candidate),
        ):
            return candidate, False, "sql_fence"

    direct = _strip_code_fence(text)

    if re.match(
        r"(?is)^(SELECT|WITH)\b",
        _remove_leading_comments(direct),
    ):
        return direct, False, "direct_sql"

    return None, False, "none"


def _normalize_value(value: Any) -> tuple[str, Any]:
    if value is None:
        return ("null", None)

    if isinstance(value, bytes):
        return ("bytes", value.hex())

    if isinstance(value, bool):
        return ("integer", int(value))

    if isinstance(value, int):
        return ("integer", value)

    if isinstance(value, float):
        if math.isnan(value):
            return ("float", "nan")

        if math.isinf(value):
            return (
                "float",
                "inf" if value > 0 else "-inf",
            )

        rounded = round(value, 8)

        if rounded == 0:
            rounded = 0.0

        return ("float", rounded)

    if isinstance(value, str):
        return ("text", value)

    return ("other", str(value))


def _execute_query(
    db_path: Path,
    sql: str,
) -> tuple[tuple[str, ...], list[tuple]]:
    sql = _validate_read_only_sql(sql)

    encoded_path = quote(str(db_path), safe="/")
    uri = (
        f"file:{encoded_path}"
        f"?mode=ro&immutable=1"
    )

    connection = sqlite3.connect(
        uri,
        uri=True,
        timeout=1.0,
    )

    # Spider wta_1 contains two malformed UTF-8 player names.
    # Decode valid text normally and replace only malformed bytes.
    connection.text_factory = lambda raw: raw.decode(
        "utf-8",
        errors="replace",
    )

    try:
        connection.execute("PRAGMA query_only = ON")

        deadline = (
            time.monotonic() + QUERY_TIMEOUT_SECONDS
        )

        def stop_long_query() -> int:
            return int(time.monotonic() > deadline)

        connection.set_progress_handler(
            stop_long_query,
            1000,
        )

        cursor = connection.execute(sql)

        if cursor.description is None:
            raise ValueError(
                "SQL did not return a result set"
            )

        columns = tuple(
            item[0] for item in cursor.description
        )

        raw_rows = cursor.fetchmany(
            MAX_RESULT_ROWS + 1
        )

        if len(raw_rows) > MAX_RESULT_ROWS:
            raise ValueError(
                "SQL result exceeded "
                f"{MAX_RESULT_ROWS} rows"
            )

        rows = [
            tuple(
                _normalize_value(value)
                for value in row
            )
            for row in raw_rows
        ]

        return columns, rows

    finally:
        connection.close()


def _gold_requires_order(gold_sql: str) -> bool:
    return bool(
        re.search(
            r"(?is)\bORDER\s+BY\b",
            gold_sql,
        )
    )


def _same_execution_result(
    predicted_columns: tuple[str, ...],
    predicted_rows: list[tuple],
    gold_columns: tuple[str, ...],
    gold_rows: list[tuple],
    gold_sql: str,
) -> bool:
    if len(predicted_columns) != len(gold_columns):
        return False

    if _gold_requires_order(gold_sql):
        return predicted_rows == gold_rows

    return Counter(predicted_rows) == Counter(gold_rows)


def _parse_ground_truth(
    ground_truth: Any,
    extra_info: dict | None,
) -> tuple[str, str]:
    if isinstance(ground_truth, str):
        try:
            parsed = json.loads(ground_truth)
        except json.JSONDecodeError:
            parsed = None
    elif isinstance(ground_truth, dict):
        parsed = ground_truth
    else:
        parsed = None

    if isinstance(parsed, dict):
        db_id = parsed.get("db_id")
        gold_sql = parsed.get("query")
    else:
        db_id = None
        gold_sql = None

    if extra_info:
        db_id = db_id or extra_info.get("db_id")
        gold_sql = gold_sql or extra_info.get(
            "gold_sql"
        )

    if not isinstance(db_id, str) or not db_id:
        raise ValueError(
            "Ground truth does not contain db_id"
        )

    if not isinstance(gold_sql, str) or not gold_sql:
        raise ValueError(
            "Ground truth does not contain query"
        )

    return db_id, gold_sql


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict | None = None,
) -> dict[str, float]:
    del data_source

    from verl.utils.text2sql_tool_metrics import extract_tool_process_metrics

    prediction, strict_format, method = (
        extract_final_sql(solution_str)
    )

    answer_found = float(prediction is not None)
    format_compliance = float(strict_format)
    tool_metrics = extract_tool_process_metrics(solution_str)
    sql_executable = 0.0
    execution_correct = 0.0
    gold_executable = 0.0

    extraction_codes = {
        "none": 0.0,
        "final_sql": 1.0,
        "sql_fence": 2.0,
        "direct_sql": 3.0,
    }

    try:
        db_id, gold_sql = _parse_ground_truth(
            ground_truth,
            extra_info,
        )
        db_path = _database_path(db_id)

        gold_columns, gold_rows = _execute_query(
            db_path,
            gold_sql,
        )
        gold_executable = 1.0

        if prediction is not None:
            predicted_columns, predicted_rows = (
                _execute_query(
                    db_path,
                    prediction,
                )
            )
            sql_executable = 1.0

            execution_correct = float(
                _same_execution_result(
                    predicted_columns,
                    predicted_rows,
                    gold_columns,
                    gold_rows,
                    gold_sql,
                )
            )

    except Exception:
        pass

    accuracy_reward = (
        ACCURACY_WEIGHT * execution_correct
    )
    executable_reward = (
        EXECUTABLE_WEIGHT * sql_executable
    )
    format_reward = (
        FORMAT_WEIGHT * format_compliance
    )

    score = (
        accuracy_reward
        + executable_reward
        + format_reward
    )

    return {
        "score": float(score),
        "execution_correct": execution_correct,
        "sql_executable": sql_executable,
        "format_compliance": format_compliance,
        "answer_found": answer_found,
        "gold_executable": gold_executable,
        "accuracy_reward": accuracy_reward,
        "executable_reward": executable_reward,
        "format_reward": format_reward,
        "extraction_method": extraction_codes[method],
        **tool_metrics,
    }
