"""Project-specific SQL execution evaluator.

Extracted helper behavior is preserved from the historical evaluator.
This is not the official Spider evaluation implementation.
"""
from __future__ import annotations

import math
import re
import sqlite3
import time
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import quote

MAX_RESULT_ROWS = 50000

QUERY_TIMEOUT_SECONDS = 5.0

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


def evaluate_answer(
    db_path: str | Path,
    solution_str: str,
    gold_sql: str,
) -> dict[str, Any]:
    """Evaluate one answer using an explicitly supplied database path.

    Exceptions are recorded as diagnostics. They still produce zero
    correctness, matching the historical execution-metric behavior.
    This is not a verl compute_score entry point.
    """
    prediction, strict_format, method = extract_final_sql(solution_str)
    result = {
        "answer_found": float(prediction is not None),
        "format_compliance": float(strict_format),
        "gold_executable": 0.0,
        "sql_executable": 0.0,
        "execution_correct": 0.0,
        "extraction_method": {
            "none": 0.0,
            "final_sql": 1.0,
            "sql_fence": 2.0,
            "direct_sql": 3.0,
        }[method],
        "evaluation_error_type": None,
    }
    try:
        gold_columns, gold_rows = _execute_query(Path(db_path), gold_sql)
        result["gold_executable"] = 1.0
        if prediction is not None:
            columns, rows = _execute_query(Path(db_path), prediction)
            result["sql_executable"] = 1.0
            result["execution_correct"] = float(
                _same_execution_result(
                    columns, rows, gold_columns, gold_rows, gold_sql
                )
            )
    except Exception as exc:
        result["evaluation_error_type"] = type(exc).__name__
    return result
