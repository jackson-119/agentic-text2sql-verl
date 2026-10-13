"""Correctness-gated reward composition.

Consumes precomputed metrics; does not execute SQL.
Historical v10_* output keys are retained for compatibility.
"""
from __future__ import annotations

INCORRECT_SCORE = -0.10

CORRECT_PROTOCOL_BASE = 0.98

CORRECT_DIRECT_SCORE = 0.90

CORRECT_INVALID_PROTOCOL_SCORE = 0.80

MAX_PROCESS_BONUS = 0.02

MAX_CORRECT_PENALTY = 0.02

def _flag(
    metrics: dict,
    key: str,
) -> float:
    try:
        return float(
            float(metrics.get(key, 0.0))
            > 0.5
        )
    except Exception:
        return 0.0

def _number(
    metrics: dict,
    key: str,
) -> float:
    try:
        return float(
            metrics.get(key, 0.0)
        )
    except Exception:
        return 0.0

def compose_reward(
    metrics: dict,
) -> dict[str, float]:
    correct = _flag(
        metrics,
        "execution_correct",
    )
    strict_format = _flag(
        metrics,
        "format_compliance",
    )
    answer_found = _flag(
        metrics,
        "answer_found",
    )
    tool_called = _flag(
        metrics,
        "tool_called",
    )
    protocol_valid = _flag(
        metrics,
        "tool_protocol_valid",
    )
    malformed = _flag(
        metrics,
        "malformed_tool_call",
    )
    stopped_after_error = _flag(
        metrics,
        "stopped_after_tool_error",
    )

    protocol_invalid = float(
        malformed > 0.5
        or (
            tool_called > 0.5
            and protocol_valid < 0.5
        )
    )

    direct = float(
        tool_called < 0.5
    )

    eligible = float(
        correct > 0.5
        and strict_format > 0.5
        and answer_found > 0.5
        and tool_called > 0.5
        and protocol_valid > 0.5
        and malformed < 0.5
        and stopped_after_error < 0.5
    )

    process_components = [
        _flag(
            metrics,
            "list_tables_succeeded",
        ),
        _flag(
            metrics,
            "get_table_schema_succeeded",
        ),
        _flag(
            metrics,
            "execute_sql_succeeded",
        ),
        _flag(
            metrics,
            "ordered_tool_chain",
        ),
        _flag(
            metrics,
            "v3_final_no_duplicate_execute",
        ),
    ]

    process_quality = (
        sum(process_components)
        / len(process_components)
    )

    process_bonus = (
        MAX_PROCESS_BONUS
        * process_quality
        * eligible
    )

    duplicate_count = (
        _number(
            metrics,
            "v3_duplicate_execute_count",
        )
        + _number(
            metrics,
            "v3_duplicate_schema_count",
        )
        + _number(
            metrics,
            "v3_duplicate_list_tables_count",
        )
    )

    correct_penalty = (
        min(
            MAX_CORRECT_PENALTY,
            0.005 * duplicate_count,
        )
        if eligible > 0.5
        else 0.0
    )

    if correct < 0.5:
        # 核心不变量：所有错误轨迹完全同分。
        score = INCORRECT_SCORE
        terminal_reward = INCORRECT_SCORE
        branch_code = 0.0

    elif protocol_invalid > 0.5:
        score = (
            CORRECT_INVALID_PROTOCOL_SCORE
        )
        terminal_reward = score
        branch_code = 1.0

    elif eligible > 0.5:
        raw_score = (
            CORRECT_PROTOCOL_BASE
            + process_bonus
            - correct_penalty
        )
        score = min(
            1.0,
            max(0.96, raw_score),
        )
        terminal_reward = (
            CORRECT_PROTOCOL_BASE
        )
        branch_code = 3.0

    else:
        score = CORRECT_DIRECT_SCORE
        terminal_reward = score
        branch_code = 2.0

    positive_incorrect = float(
        correct < 0.5
        and score >= 0.0
    )

    wrong_process_nonzero = float(
        correct < 0.5
        and abs(process_bonus) > 1e-12
    )

    incorrect_uniform = float(
        correct > 0.5
        or abs(
            score - INCORRECT_SCORE
        ) <= 1e-12
    )

    return {
        "score": float(score),
        "v10_score": float(score),
        "v10_terminal_reward": float(
            terminal_reward
        ),
        "v10_process_quality": float(
            process_quality
        ),
        "v10_process_bonus": float(
            process_bonus
        ),
        "v10_correct_penalty": float(
            correct_penalty
        ),
        "v10_eligible_correct_process": (
            eligible
        ),
        "v10_protocol_invalid": (
            protocol_invalid
        ),
        "v10_direct": direct,
        "v10_malformed": malformed,
        "v10_stopped_after_error": (
            stopped_after_error
        ),
        "v10_positive_incorrect": (
            positive_incorrect
        ),
        "v10_wrong_process_nonzero": (
            wrong_process_nonzero
        ),
        "v10_incorrect_uniform": (
            incorrect_uniform
        ),
        "v10_branch_code": branch_code,
    }
