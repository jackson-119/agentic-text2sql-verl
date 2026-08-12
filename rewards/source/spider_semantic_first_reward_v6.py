from __future__ import annotations

from pathlib import Path
from typing import Any
import importlib.util


_V5_PATH = (
    Path(__file__).with_name(
        "spider_verified_structural_reward_v5.py"
    )
)

_SPEC = importlib.util.spec_from_file_location(
    "verl_text2sql_reward_v5_for_v6",
    _V5_PATH,
)

if _SPEC is None or _SPEC.loader is None:
    raise ImportError(
        f"无法加载Reward v5：{_V5_PATH}"
    )

_V5_MODULE = importlib.util.module_from_spec(
    _SPEC
)

_SPEC.loader.exec_module(
    _V5_MODULE
)


def _float(
    mapping: dict[str, Any],
    key: str,
    default: float = 0.0,
) -> float:
    value = mapping.get(
        key,
        default,
    )

    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def compute_v6_from_v5(
    base_result: dict[str, Any],
) -> dict[str, float]:
    result = dict(base_result)

    execution_correct = float(
        _float(
            result,
            "execution_correct",
        )
        > 0.5
    )

    sql_executable = float(
        _float(
            result,
            "sql_executable",
        )
        > 0.5
    )

    format_compliance = float(
        _float(
            result,
            "format_compliance",
        )
        > 0.5
    )

    verified_eligible = float(
        _float(
            result,
            "v5_verified_structural_eligible",
        )
        > 0.5
    )

    final_ok = float(
        _float(
            result,
            "v5_final_matches_successful_execute",
        )
        > 0.5
    )

    final_failed = float(
        _float(
            result,
            "v5_final_was_failed",
        )
        > 0.5
    )

    final_unverified = float(
        _float(
            result,
            "v5_final_unverified",
        )
        > 0.5
    )

    tool_error = float(
        _float(
            result,
            "tool_error",
        )
        > 0.5
    )

    duplicate_execute = float(
        _float(
            result,
            "v3_has_duplicate_execute",
        )
        > 0.5
    )

    malformed_tool_call = float(
        _float(
            result,
            "malformed_tool_call",
        )
        > 0.5
    )

    ordered_tool_chain = float(
        _float(
            result,
            "ordered_tool_chain",
        )
        > 0.5
    )

    tool_efficiency = max(
        0.0,
        min(
            1.0,
            _float(
                result,
                "v3_tool_call_efficiency",
            ),
        ),
    )

    changed_recovery = float(
        _float(
            result,
            "v5_changed_sql_recovery",
        )
        > 0.5
    )

    table_f1 = max(
        0.0,
        min(
            1.0,
            _float(
                result,
                "v5_table_f1",
            ),
        ),
    )

    column_f1 = max(
        0.0,
        min(
            1.0,
            _float(
                result,
                "v5_column_f1",
            ),
        ),
    )

    join_f1 = max(
        0.0,
        min(
            1.0,
            _float(
                result,
                "v5_join_f1",
            ),
        ),
    )

    executable_but_wrong = float(
        sql_executable > 0.5
        and execution_correct < 0.5
    )

    terminal_reward = (
        0.77 * execution_correct
    )

    executable_reward = (
        0.02 * sql_executable
    )

    format_reward = (
        0.01 * format_compliance
    )

    semantic_reward = (
        verified_eligible
        * (
            0.06 * table_f1
            + 0.06 * join_f1
            + 0.03 * column_f1
        )
    )

    verification_reward = (
        0.03 * final_ok
    )

    process_reward = (
        0.01 * ordered_tool_chain
        + 0.01 * tool_efficiency
    )

    recovery_reward = (
        0.02 * changed_recovery
    )

    semantic_gap_penalty = (
        0.04 * executable_but_wrong
    )

    reliability_penalty = (
        0.03 * tool_error
        + 0.03 * duplicate_execute
        + 0.03 * final_failed
        + 0.02 * final_unverified
        + 0.02 * malformed_tool_call
    )

    total_penalty = (
        semantic_gap_penalty
        + reliability_penalty
    )

    score_pre_clip = (
        terminal_reward
        + executable_reward
        + format_reward
        + semantic_reward
        + verification_reward
        + process_reward
        + recovery_reward
        - total_penalty
    )

    incorrect_cap_applied = 0.0
    correct_floor_applied = 0.0

    if execution_correct < 0.5:
        if score_pre_clip > 0.25:
            incorrect_cap_applied = 1.0

        score_pre_clip = min(
            score_pre_clip,
            0.25,
        )
    else:
        if score_pre_clip < 0.70:
            correct_floor_applied = 1.0

        score_pre_clip = max(
            score_pre_clip,
            0.70,
        )

    score = max(
        -0.20,
        min(
            1.0,
            score_pre_clip,
        ),
    )

    base_v5_score = _float(
        result,
        "v5_score",
        _float(
            result,
            "score",
        ),
    )

    result.update(
        {
            "reward_version": 6.0,
            "score": score,
            "score_pre_clip": score_pre_clip,
            "v6_score": score,
            "v6_score_pre_clip": score_pre_clip,
            "v6_base_v5_score": base_v5_score,
            "v6_score_delta_from_v5": (
                score - base_v5_score
            ),
            "v6_execution_correct": (
                execution_correct
            ),
            "v6_executable_but_wrong": (
                executable_but_wrong
            ),
            "v6_verified_semantic_eligible": (
                verified_eligible
            ),
            "v6_final_ok": final_ok,
            "v6_tool_error": tool_error,
            "v6_duplicate_execute": (
                duplicate_execute
            ),
            "v6_terminal_reward": (
                terminal_reward
            ),
            "v6_executable_reward": (
                executable_reward
            ),
            "v6_format_reward": (
                format_reward
            ),
            "v6_semantic_reward": (
                semantic_reward
            ),
            "v6_verification_reward": (
                verification_reward
            ),
            "v6_process_reward": (
                process_reward
            ),
            "v6_recovery_reward": (
                recovery_reward
            ),
            "v6_semantic_gap_penalty": (
                semantic_gap_penalty
            ),
            "v6_reliability_penalty": (
                reliability_penalty
            ),
            "v6_total_penalty": (
                total_penalty
            ),
            "v6_incorrect_cap_applied": (
                incorrect_cap_applied
            ),
            "v6_correct_floor_applied": (
                correct_floor_applied
            ),
        }
    )

    return result


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict[str, Any] | None = None,
) -> dict[str, float]:
    base_result = _V5_MODULE.compute_score(
        data_source=data_source,
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
    )

    if not isinstance(
        base_result,
        dict,
    ):
        raise TypeError(
            "Reward v5必须返回dict，"
            f"实际返回{type(base_result)!r}"
        )

    return compute_v6_from_v5(
        base_result
    )
