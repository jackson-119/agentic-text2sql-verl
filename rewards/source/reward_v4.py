from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import re
import sqlite3
from typing import Any


_V3_PATH = Path(__file__).with_name(
    "spider_process_reward_v3.py"
)

_SPEC = importlib.util.spec_from_file_location(
    "spider_process_reward_v3_for_v4",
    _V3_PATH,
)

if _SPEC is None or _SPEC.loader is None:
    raise ImportError(
        f"无法加载Reward v3：{_V3_PATH}"
    )

_V3 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_V3)


DATABASE_ROOT = Path(
    os.environ.get(
        "SPIDER_DATABASE_ROOT",
        (
            "/home/archlab/yb/data/spider/"
            "raw/spider_data/database"
        ),
    )
)

V4_TABLE_WEIGHT = 0.04
V4_COLUMN_WEIGHT = 0.04
V4_JOIN_WEIGHT = 0.02

V4_CHANGED_RECOVERY_WEIGHT = 0.04
V4_FINAL_CONSISTENCY_WEIGHT = 0.03

V4_FAILED_FINAL_PENALTY = 0.04
V4_UNVERIFIED_FINAL_PENALTY = 0.02
V4_FAILED_SQL_REUSE_UNIT = 0.02
V4_FAILED_SQL_REUSE_CAP = 0.06

V4_MIN_SCORE = -0.20
V4_MAX_SCORE = 1.00


_EVENT_PATTERN = re.compile(
    r"<tool_call>\s*(.*?)\s*</tool_call>"
    r"|<tool_response>\s*(.*?)\s*</tool_response>",
    flags=re.IGNORECASE | re.DOTALL,
)

_FINAL_SQL_PATTERN = re.compile(
    r"FINAL_SQL\s*:\s*([^\r\n]+)",
    flags=re.IGNORECASE,
)

_SQL_FENCE_PATTERN = re.compile(
    r"```sql\s*(.*?)```",
    flags=re.IGNORECASE | re.DOTALL,
)

_IDENTIFIER = (
    r'(?:`[^`]+`|"[^"]+"|\[[^\]]+\]'
    r"|[A-Za-z_][A-Za-z0-9_$]*)"
)

_RESERVED = {
    "select",
    "from",
    "join",
    "left",
    "right",
    "inner",
    "outer",
    "cross",
    "full",
    "where",
    "group",
    "order",
    "having",
    "limit",
    "union",
    "intersect",
    "except",
    "on",
    "as",
    "and",
    "or",
}


def _safe_float(
    value: Any,
    default: float = 0.0,
) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default

    if numeric != numeric:
        return default

    return numeric


def _parse_json_object(
    text: str,
) -> dict[str, Any] | None:
    try:
        value = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None

    if isinstance(value, dict):
        return value

    return None


def _normalize_sql(sql: str | None) -> str:
    if not isinstance(sql, str):
        return ""

    normalized = sql.strip().rstrip(";")
    normalized = normalized.replace("`", "")
    normalized = normalized.replace("[", "")
    normalized = normalized.replace("]", "")
    normalized = re.sub(
        r"\s+",
        " ",
        normalized,
    )

    return normalized.casefold()


def _unquote(identifier: str) -> str:
    identifier = identifier.strip()

    if len(identifier) >= 2:
        pairs = {
            ("`", "`"),
            ('"', '"'),
            ("[", "]"),
        }

        if (
            identifier[0],
            identifier[-1],
        ) in pairs:
            return identifier[1:-1]

    return identifier


def _extract_final_sql(
    solution_str: str,
) -> str | None:
    if not isinstance(solution_str, str):
        return None

    matches = list(
        _FINAL_SQL_PATTERN.finditer(
            solution_str
        )
    )

    if matches:
        sql = matches[-1].group(1).strip()

        if sql:
            return sql

    fences = list(
        _SQL_FENCE_PATTERN.finditer(
            solution_str
        )
    )

    if fences:
        sql = fences[-1].group(1).strip()

        if sql:
            return sql

    return None


def _parse_ground_truth(
    ground_truth: Any,
    extra_info: dict | None,
) -> tuple[str | None, str | None]:
    value = ground_truth

    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = None

        if isinstance(parsed, dict):
            value = parsed

    db_id = None
    gold_sql = None

    if isinstance(value, dict):
        db_id = value.get("db_id")
        gold_sql = (
            value.get("query")
            or value.get("sql")
            or value.get("gold_sql")
        )
    elif isinstance(value, str):
        gold_sql = value

    if isinstance(extra_info, dict):
        db_id = (
            db_id
            or extra_info.get("db_id")
        )

        gold_sql = (
            gold_sql
            or extra_info.get("query")
            or extra_info.get("gold_sql")
        )

    if not isinstance(db_id, str):
        db_id = None

    if not isinstance(gold_sql, str):
        gold_sql = None

    return db_id, gold_sql


def _parse_tool_trace(
    solution_str: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    for match in _EVENT_PATTERN.finditer(
        solution_str or ""
    ):
        call_payload = match.group(1)
        response_payload = match.group(2)

        if call_payload is not None:
            parsed = _parse_json_object(
                call_payload
            )

            name = None
            arguments: Any = {}

            if parsed is not None:
                name = parsed.get("name")
                arguments = parsed.get(
                    "arguments",
                    {},
                )

            if isinstance(arguments, str):
                try:
                    arguments = json.loads(
                        arguments
                    )
                except json.JSONDecodeError:
                    arguments = {}

            if not isinstance(arguments, dict):
                arguments = {}

            records.append(
                {
                    "name": name,
                    "arguments": arguments,
                    "response_seen": False,
                    "response": None,
                    "response_text": "",
                    "success": False,
                    "error": False,
                }
            )
            continue

        pending = next(
            (
                record
                for record in reversed(records)
                if not record["response_seen"]
            ),
            None,
        )

        if pending is None:
            continue

        response_text = (
            response_payload or ""
        )

        response = _parse_json_object(
            response_text
        )

        success = False
        error = False

        if response is not None:
            if response.get("ok") is True:
                success = True
            elif response.get("ok") is False:
                error = True
            elif response.get("error") is not None:
                error = True
        else:
            lowered = response_text.casefold()

            error = any(
                marker in lowered
                for marker in (
                    "error",
                    "exception",
                    "no such",
                    "unknown",
                )
            )
            success = not error and bool(
                response_text.strip()
            )

        pending["response_seen"] = True
        pending["response"] = response
        pending["response_text"] = (
            response_text
        )
        pending["success"] = success
        pending["error"] = error

    return records


def _analyze_execution_trace(
    solution_str: str,
    final_sql: str | None,
) -> dict[str, float]:
    records = _parse_tool_trace(
        solution_str
    )

    executes = [
        record
        for record in records
        if record.get("name") == "execute_sql"
    ]

    failed_sqls: list[str] = []
    successful_sqls: list[str] = []
    failed_seen: set[str] = set()

    failed_sql_reuse_count = 0
    unknown_column_failures: list[
        tuple[int, str]
    ] = []
    failed_executes: list[
        tuple[int, str]
    ] = []
    successful_executes: list[
        tuple[int, str]
    ] = []

    for index, record in enumerate(executes):
        raw_sql = record.get(
            "arguments",
            {},
        ).get("sql")

        normalized = _normalize_sql(raw_sql)

        if not normalized:
            continue

        if normalized in failed_seen:
            failed_sql_reuse_count += 1

        if record["error"]:
            failed_seen.add(normalized)
            failed_sqls.append(normalized)
            failed_executes.append(
                (index, normalized)
            )

            response_text = str(
                record.get(
                    "response_text",
                    "",
                )
            ).casefold()

            if (
                "no such column"
                in response_text
                or "unknown column"
                in response_text
            ):
                unknown_column_failures.append(
                    (index, normalized)
                )

        if record["success"]:
            successful_sqls.append(
                normalized
            )
            successful_executes.append(
                (index, normalized)
            )

    changed_recovery = any(
        success_index > failure_index
        and success_sql != failure_sql
        for failure_index, failure_sql
        in failed_executes
        for success_index, success_sql
        in successful_executes
    )

    unknown_column_recovery = any(
        success_index > failure_index
        and success_sql != failure_sql
        for failure_index, failure_sql
        in unknown_column_failures
        for success_index, success_sql
        in successful_executes
    )

    normalized_final = _normalize_sql(
        final_sql
    )

    final_matches_success = bool(
        normalized_final
        and successful_sqls
        and normalized_final
        == successful_sqls[-1]
    )

    final_was_failed = bool(
        normalized_final
        and normalized_final in failed_sqls
        and normalized_final
        not in successful_sqls
    )

    final_unverified = bool(
        normalized_final
        and not final_matches_success
    )

    return {
        "v4_changed_sql_recovery": float(
            changed_recovery
        ),
        "v4_unknown_column_recovery": float(
            unknown_column_recovery
        ),
        "v4_final_matches_successful_execute": (
            float(final_matches_success)
        ),
        "v4_final_was_failed": float(
            final_was_failed
        ),
        "v4_final_unverified": float(
            final_unverified
        ),
        "v4_failed_sql_reuse_count": float(
            failed_sql_reuse_count
        ),
        "v4_has_failed_sql_reuse": float(
            failed_sql_reuse_count > 0
        ),
    }


def _read_schema(
    db_id: str | None,
) -> tuple[
    dict[str, str],
    dict[str, set[str]],
]:
    if not db_id:
        return {}, {}

    db_path = (
        DATABASE_ROOT
        / db_id
        / f"{db_id}.sqlite"
    )

    if not db_path.is_file():
        return {}, {}

    connection = sqlite3.connect(
        f"file:{db_path}?mode=ro",
        uri=True,
        timeout=5,
    )

    try:
        table_rows = connection.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
              AND name NOT LIKE 'sqlite_%'
            """
        ).fetchall()

        table_lookup: dict[str, str] = {}
        columns_by_table: dict[
            str,
            set[str],
        ] = {}

        for (raw_table,) in table_rows:
            table = str(raw_table)
            table_lookup[
                table.casefold()
            ] = table

            escaped = table.replace(
                '"',
                '""',
            )

            column_rows = connection.execute(
                f'PRAGMA table_info("{escaped}")'
            ).fetchall()

            columns_by_table[table] = {
                str(row[1]).casefold()
                for row in column_rows
            }

        return (
            table_lookup,
            columns_by_table,
        )
    finally:
        connection.close()


def _extract_tables_and_aliases(
    sql: str,
    table_lookup: dict[str, str],
) -> tuple[set[str], dict[str, str]]:
    pattern = re.compile(
        rf"\b(?:FROM|JOIN)\s+"
        rf"(?P<table>{_IDENTIFIER})"
        rf"(?:\s+(?:AS\s+)?"
        rf"(?P<alias>{_IDENTIFIER}))?",
        flags=re.IGNORECASE,
    )

    tables: set[str] = set()
    aliases: dict[str, str] = {}

    for match in pattern.finditer(sql):
        raw_table = _unquote(
            match.group("table")
        )
        raw_alias = match.group("alias")

        canonical = table_lookup.get(
            raw_table.casefold(),
            raw_table,
        )

        tables.add(
            canonical.casefold()
        )

        aliases[
            raw_table.casefold()
        ] = canonical.casefold()

        aliases[
            canonical.casefold()
        ] = canonical.casefold()

        if raw_alias:
            alias = _unquote(
                raw_alias
            ).casefold()

            if alias not in _RESERVED:
                aliases[alias] = (
                    canonical.casefold()
                )

    return tables, aliases


def _extract_columns(
    sql: str,
    columns_by_table: dict[
        str,
        set[str],
    ],
) -> set[str]:
    all_columns = {
        column
        for columns in columns_by_table.values()
        for column in columns
    }

    found: set[str] = set()

    for column in all_columns:
        pattern = re.compile(
            rf"(?<![A-Za-z0-9_$])"
            rf"[`\"\[]?{re.escape(column)}"
            rf"[`\"\]]?"
            rf"(?![A-Za-z0-9_$])",
            flags=re.IGNORECASE,
        )

        if pattern.search(sql):
            found.add(column)

    return found


def _extract_join_edges(
    sql: str,
    aliases: dict[str, str],
) -> set[tuple[str, str]]:
    pattern = re.compile(
        rf"(?P<a1>{_IDENTIFIER})\."
        rf"(?P<c1>{_IDENTIFIER})\s*=\s*"
        rf"(?P<a2>{_IDENTIFIER})\."
        rf"(?P<c2>{_IDENTIFIER})",
        flags=re.IGNORECASE,
    )

    edges: set[tuple[str, str]] = set()

    for match in pattern.finditer(sql):
        alias1 = _unquote(
            match.group("a1")
        ).casefold()
        alias2 = _unquote(
            match.group("a2")
        ).casefold()
        column1 = _unquote(
            match.group("c1")
        ).casefold()
        column2 = _unquote(
            match.group("c2")
        ).casefold()

        table1 = aliases.get(
            alias1,
            alias1,
        )
        table2 = aliases.get(
            alias2,
            alias2,
        )

        left = f"{table1}.{column1}"
        right = f"{table2}.{column2}"

        edges.add(
            tuple(sorted((left, right)))
        )

    return edges


def _f1(
    predicted: set,
    gold: set,
) -> float:
    if not predicted and not gold:
        return 1.0

    if not predicted or not gold:
        return 0.0

    intersection = len(
        predicted & gold
    )

    precision = intersection / len(
        predicted
    )
    recall = intersection / len(gold)

    if precision + recall == 0:
        return 0.0

    return (
        2
        * precision
        * recall
        / (precision + recall)
    )


def _structural_metrics(
    db_id: str | None,
    predicted_sql: str | None,
    gold_sql: str | None,
) -> dict[str, float]:
    if not predicted_sql or not gold_sql:
        return {
            "v4_table_f1": 0.0,
            "v4_column_f1": 0.0,
            "v4_join_f1": 0.0,
        }

    try:
        (
            table_lookup,
            columns_by_table,
        ) = _read_schema(db_id)

        pred_tables, pred_aliases = (
            _extract_tables_and_aliases(
                predicted_sql,
                table_lookup,
            )
        )

        gold_tables, gold_aliases = (
            _extract_tables_and_aliases(
                gold_sql,
                table_lookup,
            )
        )

        pred_columns = _extract_columns(
            predicted_sql,
            columns_by_table,
        )
        gold_columns = _extract_columns(
            gold_sql,
            columns_by_table,
        )

        pred_joins = _extract_join_edges(
            predicted_sql,
            pred_aliases,
        )
        gold_joins = _extract_join_edges(
            gold_sql,
            gold_aliases,
        )

        join_f1 = (
            _f1(pred_joins, gold_joins)
            if gold_joins
            else 0.0
        )

        return {
            "v4_table_f1": _f1(
                pred_tables,
                gold_tables,
            ),
            "v4_column_f1": _f1(
                pred_columns,
                gold_columns,
            ),
            "v4_join_f1": join_f1,
        }
    except Exception:
        return {
            "v4_table_f1": 0.0,
            "v4_column_f1": 0.0,
            "v4_join_f1": 0.0,
        }


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict | None = None,
) -> dict[str, float]:
    base = _V3.compute_score(
        data_source=data_source,
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
    )

    if not isinstance(base, dict):
        raise TypeError(
            "Reward v3必须返回dict"
        )

    result = dict(base)

    db_id, gold_sql = (
        _parse_ground_truth(
            ground_truth,
            extra_info,
        )
    )

    final_sql = _extract_final_sql(
        solution_str
    )

    trace = _analyze_execution_trace(
        solution_str,
        final_sql,
    )

    structure = _structural_metrics(
        db_id,
        final_sql,
        gold_sql,
    )

    correct = (
        _safe_float(
            base.get("execution_correct")
        )
        > 0.5
    )

    if correct:
        structural_reward = 0.0
    else:
        structural_reward = (
            V4_TABLE_WEIGHT
            * structure["v4_table_f1"]
            + V4_COLUMN_WEIGHT
            * structure["v4_column_f1"]
            + V4_JOIN_WEIGHT
            * structure["v4_join_f1"]
        )

    changed_recovery_reward = (
        V4_CHANGED_RECOVERY_WEIGHT
        * trace[
            "v4_changed_sql_recovery"
        ]
    )

    final_consistency_reward = (
        V4_FINAL_CONSISTENCY_WEIGHT
        * trace[
            "v4_final_matches_successful_execute"
        ]
    )

    failed_final_penalty = (
        V4_FAILED_FINAL_PENALTY
        * trace["v4_final_was_failed"]
    )

    unverified_final_penalty = (
        V4_UNVERIFIED_FINAL_PENALTY
        * trace["v4_final_unverified"]
        * (
            1.0
            - trace["v4_final_was_failed"]
        )
    )

    failed_sql_reuse_penalty = min(
        V4_FAILED_SQL_REUSE_CAP,
        V4_FAILED_SQL_REUSE_UNIT
        * trace[
            "v4_failed_sql_reuse_count"
        ],
    )

    old_recovery_reward = _safe_float(
        base.get("recovery_reward")
    )

    old_final_efficiency = _safe_float(
        base.get(
            "final_efficiency_reward"
        )
    )

    terminal_reward = _safe_float(
        base.get("terminal_reward")
    )

    process_reward = (
        _safe_float(
            base.get("process_reward")
        )
        - old_recovery_reward
        + changed_recovery_reward
    )

    efficiency_reward = (
        _safe_float(
            base.get("efficiency_reward")
        )
        - old_final_efficiency
        + final_consistency_reward
    )

    total_penalty = (
        _safe_float(
            base.get("total_penalty")
        )
        + failed_final_penalty
        + unverified_final_penalty
        + failed_sql_reuse_penalty
    )

    score_pre_clip = (
        terminal_reward
        + process_reward
        + efficiency_reward
        + structural_reward
        - total_penalty
    )

    score = min(
        V4_MAX_SCORE,
        max(
            V4_MIN_SCORE,
            score_pre_clip,
        ),
    )

    result.update(structure)
    result.update(trace)

    result.update(
        {
            "reward_version": 4.0,
            "v3_score": _safe_float(
                base.get("score")
            ),
            "v3_score_pre_clip": (
                _safe_float(
                    base.get(
                        "score_pre_clip"
                    )
                )
            ),
            "v3_generic_recovery_reward": (
                old_recovery_reward
            ),
            "v3_final_efficiency_reward": (
                old_final_efficiency
            ),
            "structural_reward": (
                structural_reward
            ),
            "v4_recovery_reward": (
                changed_recovery_reward
            ),
            "v4_final_consistency_reward": (
                final_consistency_reward
            ),
            "v4_failed_final_penalty": (
                failed_final_penalty
            ),
            "v4_unverified_final_penalty": (
                unverified_final_penalty
            ),
            "v4_failed_sql_reuse_penalty": (
                failed_sql_reuse_penalty
            ),
            "v4_new_penalty": (
                failed_final_penalty
                + unverified_final_penalty
                + failed_sql_reuse_penalty
            ),
            "terminal_reward": (
                terminal_reward
            ),
            "process_reward": (
                process_reward
            ),
            "efficiency_reward": (
                efficiency_reward
            ),
            "recovery_reward": (
                changed_recovery_reward
            ),
            "final_efficiency_reward": (
                final_consistency_reward
            ),
            "total_penalty": total_penalty,
            "penalty_reward": (
                -total_penalty
            ),
            "score_pre_clip": (
                score_pre_clip
            ),
            "score": score,
        }
    )

    return result
