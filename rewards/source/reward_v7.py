from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path
from typing import Any


_REWARD_V5_PATH = Path(__file__).with_name(
    "spider_verified_structural_reward_v5.py"
)


def _load_reward_v5():
    spec = importlib.util.spec_from_file_location(
        "spider_verified_structural_reward_v5_for_v7",
        _REWARD_V5_PATH,
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"无法加载Reward v5：{_REWARD_V5_PATH}"
        )

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    return module


_REWARD_V5 = _load_reward_v5()


def _number(
    metrics: dict[str, Any],
    *keys: str,
    default: float = 0.0,
) -> float:
    for key in keys:
        value = metrics.get(key)

        if value is None:
            continue

        try:
            return float(value)
        except (TypeError, ValueError):
            continue

    return float(default)


def _flag(
    metrics: dict[str, Any],
    *keys: str,
) -> bool:
    return _number(
        metrics,
        *keys,
    ) > 0.5


def _compute_v7_from_metrics(
    base_metrics: dict[str, Any],
) -> dict[str, Any]:
    result = dict(base_metrics)

    correct = _flag(
        result,
        "execution_correct",
    )
    executable = _flag(
        result,
        "sql_executable",
    )
    answer_found = _flag(
        result,
        "answer_found",
    )

    duplicate_execute = _flag(
        result,
        "v3_has_duplicate_execute",
    ) or (
        _number(
            result,
            "v3_duplicate_execute_count",
        ) > 0
    )

    duplicate_schema = _flag(
        result,
        "v3_has_duplicate_schema",
    ) or (
        _number(
            result,
            "v3_duplicate_schema_count",
        ) > 0
    )

    malformed = _flag(
        result,
        "malformed_tool_call",
    ) or (
        _number(
            result,
            "malformed_tool_call_count",
            "v3_locally_malformed_tool_call_count",
        ) > 0
    )

    tool_error = _flag(
        result,
        "tool_error",
    )

    stopped_after_error = _flag(
        result,
        "stopped_after_tool_error",
    )

    final_was_failed = _flag(
        result,
        "v5_final_was_failed",
        "v4_final_was_failed",
    )

    final_unverified = _flag(
        result,
        "v5_final_unverified",
        "v4_final_unverified",
    )

    table_f1 = _number(
        result,
        "v5_table_f1",
        "v4_table_f1",
    )
    column_f1 = _number(
        result,
        "v5_column_f1",
        "v4_column_f1",
    )
    join_f1 = _number(
        result,
        "v5_join_f1",
        "v4_join_f1",
    )

    correctness_reward = (
        1.0
        if correct
        else 0.0
    )

    # Reward v7不再为结构相似、SQL可执行或工具链完整
    # 提供任何正奖励。这些指标只用于监控。
    structural_reward = 0.0
    semantic_reward = 0.0
    process_bonus = 0.0

    if correct:
        wrong_base_penalty = 0.0
        final_state_penalty = 0.0
        reliability_penalty = (
            0.02 * float(duplicate_execute)
            + 0.01 * float(duplicate_schema)
            + 0.02 * float(malformed)
        )

        reliability_penalty = min(
            reliability_penalty,
            0.10,
        )
    else:
        # 所有错误轨迹至少为-0.10。
        # GRPO不会因为“结构看起来接近”而偏好错误SQL。
        wrong_base_penalty = 0.10

        if executable:
            final_state_penalty = 0.0
        elif answer_found:
            final_state_penalty = 0.02
        else:
            final_state_penalty = 0.04

        reliability_penalty = (
            0.03 * float(tool_error)
            + 0.03 * float(final_was_failed)
            + 0.02 * float(final_unverified)
            + 0.03 * float(duplicate_execute)
            + 0.01 * float(duplicate_schema)
            + 0.03 * float(malformed)
            + 0.02 * float(stopped_after_error)
        )

        reliability_penalty = min(
            reliability_penalty,
            0.20,
        )

    total_penalty = (
        wrong_base_penalty
        + final_state_penalty
        + reliability_penalty
    )

    score_pre_clip = (
        correctness_reward
        + structural_reward
        + semantic_reward
        + process_bonus
        - total_penalty
    )

    if correct:
        score = max(
            0.90,
            min(
                1.0,
                score_pre_clip,
            ),
        )
    else:
        score = max(
            -0.30,
            min(
                0.0,
                score_pre_clip,
            ),
        )

    result.update(
        {
            "reward_version": 7.0,
            "v7_reward_version": 7.0,
            "v7_score": score,
            "v7_score_pre_clip": score_pre_clip,
            "v7_correct": float(correct),
            "v7_wrong_executable": float(
                executable and not correct
            ),
            "v7_correctness_reward": (
                correctness_reward
            ),
            "v7_structural_reward": (
                structural_reward
            ),
            "v7_semantic_reward": (
                semantic_reward
            ),
            "v7_process_bonus": process_bonus,
            "v7_wrong_base_penalty": (
                wrong_base_penalty
            ),
            "v7_final_state_penalty": (
                final_state_penalty
            ),
            "v7_reliability_penalty": (
                reliability_penalty
            ),
            "v7_total_penalty": total_penalty,
            "v7_negative_trajectory": float(
                score < 0
            ),
            "v7_no_positive_wrong_reward": float(
                correct or score <= 0
            ),
            "v7_table_f1_monitor": table_f1,
            "v7_column_f1_monitor": column_f1,
            "v7_join_f1_monitor": join_f1,
            "v7_structure_used_for_reward": 0.0,
            "v7_semantic_used_for_reward": 0.0,
            "score_pre_clip": score_pre_clip,
            "score": score,
            "reward": score,
        }
    )

    return result


def compute_score(
    data_source=None,
    solution_str=None,
    ground_truth=None,
    extra_info=None,
    **kwargs,
):
    candidate_arguments = {
        "data_source": data_source,
        "solution_str": solution_str,
        "ground_truth": ground_truth,
        "extra_info": extra_info,
        **kwargs,
    }

    signature = inspect.signature(
        _REWARD_V5.compute_score
    )

    accepts_kwargs = any(
        parameter.kind
        == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )

    if accepts_kwargs:
        call_arguments = candidate_arguments
    else:
        call_arguments = {
            key: value
            for key, value
            in candidate_arguments.items()
            if key in signature.parameters
        }

    base_result = _REWARD_V5.compute_score(
        **call_arguments
    )

    if not isinstance(base_result, dict):
        raise TypeError(
            "Reward v5必须返回dict，实际类型为："
            f"{type(base_result)!r}"
        )

    return _compute_v7_from_metrics(
        base_result
    )
