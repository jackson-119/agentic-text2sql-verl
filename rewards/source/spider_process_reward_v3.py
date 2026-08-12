from __future__ import annotations

from collections import Counter
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any
import json
import re
import sys


# Terminal rewards: correctness remains dominant.
V3_ACCURACY_WEIGHT = 0.75
V3_EXECUTABLE_WEIGHT = 0.05
V3_FORMAT_WEIGHT = 0.02

# Positive process rewards are capped binary events.
V3_LIST_TABLES_WEIGHT = 0.015
V3_SCHEMA_WEIGHT = 0.025
V3_EXECUTE_TOOL_WEIGHT = 0.04
V3_RECOVERY_WEIGHT = 0.02
V3_ORDERED_CHAIN_WEIGHT = 0.01

# Efficiency bonuses.
V3_CORRECT_NO_DUPLICATE_WEIGHT = 0.06
V3_FINAL_NO_DUPLICATE_EXECUTE_WEIGHT = 0.03

# Penalties apply only to exact normalized duplicates.
V3_DUPLICATE_LIST_TABLES_PENALTY = 0.005
V3_DUPLICATE_LIST_TABLES_PENALTY_CAP = 0.015

V3_DUPLICATE_SCHEMA_PENALTY = 0.01
V3_DUPLICATE_SCHEMA_PENALTY_CAP = 0.03

V3_DUPLICATE_EXECUTE_PENALTY = 0.02
V3_DUPLICATE_EXECUTE_PENALTY_CAP = 0.08

V3_MALFORMED_TOOL_CALL_PENALTY = 0.02
V3_MALFORMED_TOOL_CALL_PENALTY_CAP = 0.04

V3_STOPPED_AFTER_ERROR_PENALTY = 0.02

V3_MIN_SCORE = -0.20
V3_MAX_SCORE = 1.00


_TOOL_CALL_PATTERN = re.compile(
    r"<tool_call>\s*(.*?)\s*</tool_call>",
    flags=re.IGNORECASE | re.DOTALL,
)


def _load_reward_v2():
    module_path = Path(__file__).with_name(
        "spider_process_reward_v2.py"
    )
    module_name = (
        "verl_text2sql_spider_process_reward_v2_for_v3"
    )

    existing = sys.modules.get(module_name)

    if existing is not None:
        return existing

    spec = spec_from_file_location(
        module_name,
        module_path,
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Could not load Reward v2 from {module_path}"
        )

    module = module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_REWARD_V2 = _load_reward_v2()


def _as_float(
    mapping: dict[str, Any],
    key: str,
) -> float:
    try:
        return float(mapping.get(key, 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _normalize_sql(sql: Any) -> str:
    if not isinstance(sql, str):
        return repr(sql)

    normalized = re.sub(
        r"\s+",
        " ",
        sql.strip(),
    )

    return normalized.rstrip(";").strip()


def _canonical_signature(
    name: str,
    arguments: Any,
) -> str:
    normalized_arguments = arguments

    if isinstance(arguments, dict):
        normalized_arguments = dict(arguments)

        if (
            name == "execute_sql"
            and "sql" in normalized_arguments
        ):
            normalized_arguments["sql"] = _normalize_sql(
                normalized_arguments["sql"]
            )

        if (
            name == "get_table_schema"
            and isinstance(
                normalized_arguments.get("table_name"),
                str,
            )
        ):
            normalized_arguments["table_name"] = (
                normalized_arguments["table_name"]
                .strip()
                .lower()
            )

    try:
        argument_text = json.dumps(
            normalized_arguments,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except Exception:
        argument_text = repr(normalized_arguments)

    return f"{name}:{argument_text}"


def extract_v3_efficiency_metrics(
    solution_str: str,
) -> dict[str, float]:
    """Extract exact-duplicate metrics without penalizing valid revisions."""

    if not isinstance(solution_str, str):
        solution_str = ""

    calls: list[dict[str, Any]] = []
    locally_malformed = 0

    for match in _TOOL_CALL_PATTERN.finditer(solution_str):
        payload = match.group(1)

        try:
            parsed = json.loads(payload)
        except Exception:
            locally_malformed += 1
            continue

        if not isinstance(parsed, dict):
            locally_malformed += 1
            continue

        name = parsed.get("name")

        if not isinstance(name, str) or not name:
            locally_malformed += 1
            continue

        arguments = parsed.get("arguments", {})

        calls.append(
            {
                "name": name,
                "arguments": arguments,
                "signature": _canonical_signature(
                    name,
                    arguments,
                ),
            }
        )

    signature_counts = Counter(
        call["signature"]
        for call in calls
    )

    duplicate_counts = {
        signature: count - 1
        for signature, count in signature_counts.items()
        if count > 1
    }

    duplicate_list_tables_count = sum(
        count
        for signature, count in duplicate_counts.items()
        if signature.startswith("list_tables:")
    )

    duplicate_schema_count = sum(
        count
        for signature, count in duplicate_counts.items()
        if signature.startswith("get_table_schema:")
    )

    duplicate_execute_count = sum(
        count
        for signature, count in duplicate_counts.items()
        if signature.startswith("execute_sql:")
    )

    duplicate_other_count = sum(
        count
        for signature, count in duplicate_counts.items()
        if not signature.startswith("list_tables:")
        and not signature.startswith("get_table_schema:")
        and not signature.startswith("execute_sql:")
    )

    duplicate_tool_call_count = sum(
        duplicate_counts.values()
    )

    unique_tool_call_count = len(signature_counts)

    tool_call_count = len(calls)

    tool_call_efficiency = (
        unique_tool_call_count / tool_call_count
        if tool_call_count
        else 0.0
    )

    return {
        "v3_valid_tool_call_count": float(tool_call_count),
        "v3_unique_tool_call_count": float(
            unique_tool_call_count
        ),
        "v3_duplicate_tool_call_count": float(
            duplicate_tool_call_count
        ),
        "v3_duplicate_list_tables_count": float(
            duplicate_list_tables_count
        ),
        "v3_duplicate_schema_count": float(
            duplicate_schema_count
        ),
        "v3_duplicate_execute_count": float(
            duplicate_execute_count
        ),
        "v3_duplicate_other_count": float(
            duplicate_other_count
        ),
        "v3_has_exact_duplicate": float(
            duplicate_tool_call_count > 0
        ),
        "v3_has_duplicate_schema": float(
            duplicate_schema_count > 0
        ),
        "v3_has_duplicate_execute": float(
            duplicate_execute_count > 0
        ),
        "v3_tool_call_efficiency": float(
            tool_call_efficiency
        ),
        "v3_locally_malformed_tool_call_count": float(
            locally_malformed
        ),
    }


def compute_v3_from_base_result(
    base_result: dict[str, Any],
    solution_str: str,
) -> dict[str, float]:
    """Reweight a v1/v2 result and add exact-duplicate efficiency shaping."""

    result = dict(base_result)

    efficiency_metrics = extract_v3_efficiency_metrics(
        solution_str
    )

    execution_correct = _as_float(
        result,
        "execution_correct",
    )
    sql_executable = _as_float(
        result,
        "sql_executable",
    )
    format_compliance = _as_float(
        result,
        "format_compliance",
    )
    answer_found = _as_float(
        result,
        "answer_found",
    )

    list_tables_success = _as_float(
        result,
        "list_tables_succeeded",
    )
    schema_success = _as_float(
        result,
        "get_table_schema_succeeded",
    )
    execute_after_schema_success = _as_float(
        result,
        "execute_sql_after_successful_schema",
    )
    recovery_success = _as_float(
        result,
        "tool_error_recovered_successfully",
    )
    ordered_chain = _as_float(
        result,
        "ordered_tool_chain",
    )

    malformed_count = max(
        _as_float(
            result,
            "malformed_tool_call_count",
        ),
        efficiency_metrics[
            "v3_locally_malformed_tool_call_count"
        ],
    )

    stopped_after_error = _as_float(
        result,
        "stopped_after_tool_error",
    )

    duplicate_count = efficiency_metrics[
        "v3_duplicate_tool_call_count"
    ]
    duplicate_list_count = efficiency_metrics[
        "v3_duplicate_list_tables_count"
    ]
    duplicate_schema_count = efficiency_metrics[
        "v3_duplicate_schema_count"
    ]
    duplicate_execute_count = efficiency_metrics[
        "v3_duplicate_execute_count"
    ]

    no_exact_duplicate = float(
        duplicate_count == 0.0
    )
    no_duplicate_execute = float(
        duplicate_execute_count == 0.0
    )

    correct_no_duplicate = (
        execution_correct * no_exact_duplicate
    )

    final_no_duplicate_execute = (
        answer_found * no_duplicate_execute
    )

    accuracy_reward = (
        V3_ACCURACY_WEIGHT * execution_correct
    )
    executable_reward = (
        V3_EXECUTABLE_WEIGHT * sql_executable
    )
    format_reward = (
        V3_FORMAT_WEIGHT * format_compliance
    )

    list_tables_reward = (
        V3_LIST_TABLES_WEIGHT * list_tables_success
    )
    schema_reward = (
        V3_SCHEMA_WEIGHT * schema_success
    )
    execute_tool_reward = (
        V3_EXECUTE_TOOL_WEIGHT
        * execute_after_schema_success
    )
    recovery_reward = (
        V3_RECOVERY_WEIGHT * recovery_success
    )
    ordered_chain_reward = (
        V3_ORDERED_CHAIN_WEIGHT * ordered_chain
    )

    correct_efficiency_reward = (
        V3_CORRECT_NO_DUPLICATE_WEIGHT
        * correct_no_duplicate
    )
    final_efficiency_reward = (
        V3_FINAL_NO_DUPLICATE_EXECUTE_WEIGHT
        * final_no_duplicate_execute
    )

    duplicate_list_tables_penalty = min(
        V3_DUPLICATE_LIST_TABLES_PENALTY_CAP,
        V3_DUPLICATE_LIST_TABLES_PENALTY
        * duplicate_list_count,
    )
    duplicate_schema_penalty = min(
        V3_DUPLICATE_SCHEMA_PENALTY_CAP,
        V3_DUPLICATE_SCHEMA_PENALTY
        * duplicate_schema_count,
    )
    duplicate_execute_penalty = min(
        V3_DUPLICATE_EXECUTE_PENALTY_CAP,
        V3_DUPLICATE_EXECUTE_PENALTY
        * duplicate_execute_count,
    )
    malformed_penalty = min(
        V3_MALFORMED_TOOL_CALL_PENALTY_CAP,
        V3_MALFORMED_TOOL_CALL_PENALTY
        * malformed_count,
    )
    stopped_after_error_penalty = (
        V3_STOPPED_AFTER_ERROR_PENALTY
        * float(stopped_after_error > 0.0)
    )

    terminal_reward = (
        accuracy_reward
        + executable_reward
        + format_reward
    )

    process_reward = (
        list_tables_reward
        + schema_reward
        + execute_tool_reward
        + recovery_reward
        + ordered_chain_reward
    )

    efficiency_reward = (
        correct_efficiency_reward
        + final_efficiency_reward
    )

    total_penalty = (
        duplicate_list_tables_penalty
        + duplicate_schema_penalty
        + duplicate_execute_penalty
        + malformed_penalty
        + stopped_after_error_penalty
    )

    score_pre_clip = (
        terminal_reward
        + process_reward
        + efficiency_reward
        - total_penalty
    )

    score = min(
        V3_MAX_SCORE,
        max(V3_MIN_SCORE, score_pre_clip),
    )

    result.update(efficiency_metrics)

    result.update(
        {
            "reward_version": 3.0,
            "score": float(score),
            "score_pre_clip": float(score_pre_clip),
            "terminal_reward": float(terminal_reward),
            "process_reward": float(process_reward),
            "efficiency_reward": float(efficiency_reward),
            "total_penalty": float(total_penalty),
            "penalty_reward": float(-total_penalty),
            "accuracy_reward": float(accuracy_reward),
            "executable_reward": float(executable_reward),
            "format_reward": float(format_reward),
            "list_tables_reward": float(
                list_tables_reward
            ),
            "schema_reward": float(schema_reward),
            "execute_tool_reward": float(
                execute_tool_reward
            ),
            "recovery_reward": float(recovery_reward),
            "ordered_chain_reward": float(
                ordered_chain_reward
            ),
            "correct_efficiency_reward": float(
                correct_efficiency_reward
            ),
            "final_efficiency_reward": float(
                final_efficiency_reward
            ),
            "duplicate_list_tables_penalty": float(
                duplicate_list_tables_penalty
            ),
            "duplicate_schema_penalty": float(
                duplicate_schema_penalty
            ),
            "duplicate_execute_penalty": float(
                duplicate_execute_penalty
            ),
            "malformed_penalty": float(
                malformed_penalty
            ),
            "stopped_after_error_penalty": float(
                stopped_after_error_penalty
            ),
            "v3_correct_no_duplicate": float(
                correct_no_duplicate
            ),
            "v3_final_no_duplicate_execute": float(
                final_no_duplicate_execute
            ),
        }
    )

    return result


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict | None = None,
):
    """Compute Reward v2 terminal/tool metrics, then apply Reward v3."""

    base_result = _REWARD_V2.compute_score(
        data_source=data_source,
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
    )

    if not isinstance(base_result, dict):
        raise TypeError(
            "Reward v2 compute_score must return a dict, "
            f"received {type(base_result)!r}"
        )

    return compute_v3_from_base_result(
        base_result=base_result,
        solution_str=solution_str,
    )
