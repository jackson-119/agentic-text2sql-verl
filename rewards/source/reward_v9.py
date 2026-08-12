from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path
from typing import Any

from verl.utils.text2sql_tool_metrics import (
    extract_tool_process_metrics,
)


REWARD_VERSION = 9.0

# 正确性奖励与错误基准。
CORRECT_BASE = 0.85
INCORRECT_BASE = -0.25

# 只使用可以由真实工具轨迹验证的过程事件。
PROCESS_WEIGHTS = {
    "tool_protocol_valid": 0.01,
    "list_tables_succeeded": 0.02,
    "schema_after_list_tables": 0.03,
    "execute_sql_after_successful_schema": 0.03,
    "execute_sql_succeeded": 0.04,
    "tool_error_recovered_successfully": 0.02,
}

# 过程违规只做小幅惩罚。
PENALTY_WEIGHTS = {
    "malformed_tool_call": 0.04,
    "stopped_after_tool_error": 0.03,
    "duplicate_execute": 0.03,
}


def _load_binary_reward():
    reward_path = (
        Path(__file__).with_name(
            "spider_binary_execution_reward_v8.py"
        )
    )

    spec = importlib.util.spec_from_file_location(
        "spider_binary_execution_reward_v8_for_v9",
        reward_path,
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"无法加载Reward v8：{reward_path}"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module.compute_score


_BINARY_COMPUTE_SCORE = _load_binary_reward()


def _number(
    values: dict[str, Any],
    key: str,
) -> float:
    value = values.get(key, 0.0)

    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _flag(
    values: dict[str, Any],
    key: str,
) -> float:
    return float(
        _number(values, key) > 0.5
    )


def _duplicate_execute_flag(
    metrics: dict[str, Any],
) -> float:
    direct_keys = (
        "v3_has_duplicate_execute",
        "has_duplicate_execute",
        "duplicate_execute",
    )

    for key in direct_keys:
        if _flag(metrics, key):
            return 1.0

    count_keys = (
        "v3_duplicate_execute_count",
        "duplicate_execute_count",
    )

    for key in count_keys:
        if _number(metrics, key) > 0:
            return 1.0

    return 0.0


def compose_score(
    execution_correct: float,
    process_metrics: dict[str, Any],
) -> dict[str, float]:
    correct = float(
        float(execution_correct) > 0.5
    )

    process_components = {
        key: (
            weight
            * _flag(process_metrics, key)
        )
        for key, weight in PROCESS_WEIGHTS.items()
    }

    process_reward = sum(
        process_components.values()
    )

    duplicate_execute = (
        _duplicate_execute_flag(
            process_metrics
        )
    )

    penalty_components = {
        "malformed_tool_call": (
            PENALTY_WEIGHTS[
                "malformed_tool_call"
            ]
            * _flag(
                process_metrics,
                "malformed_tool_call",
            )
        ),
        "stopped_after_tool_error": (
            PENALTY_WEIGHTS[
                "stopped_after_tool_error"
            ]
            * _flag(
                process_metrics,
                "stopped_after_tool_error",
            )
        ),
        "duplicate_execute": (
            PENALTY_WEIGHTS[
                "duplicate_execute"
            ]
            * duplicate_execute
        ),
    }

    process_penalty = sum(
        penalty_components.values()
    )

    if correct:
        score_pre_clip = (
            CORRECT_BASE
            + process_reward
            - process_penalty
        )

        # 正确轨迹始终高于任何错误轨迹。
        score = min(
            1.0,
            max(0.75, score_pre_clip),
        )
    else:
        score_pre_clip = (
            INCORRECT_BASE
            + process_reward
            - process_penalty
        )

        # 即使过程完整，错误轨迹仍然保持负分。
        score = min(
            -0.10,
            max(-0.35, score_pre_clip),
        )

    result = {
        "score": float(score),
        "v9_score": float(score),
        "v9_score_pre_clip": float(
            score_pre_clip
        ),
        "v9_execution_correct": correct,
        "v9_terminal_reward": correct,
        "v9_process_reward": float(
            process_reward
        ),
        "v9_process_penalty": float(
            process_penalty
        ),
        "v9_has_process_signal": float(
            process_reward > 0
        ),
        "v9_incorrect_with_process": float(
            (not correct)
            and process_reward > 0
        ),
        "v9_duplicate_execute": float(
            duplicate_execute
        ),
        "reward_version": REWARD_VERSION,
    }

    for key, value in process_components.items():
        result[
            f"v9_process_{key}"
        ] = float(value)

    for key, value in penalty_components.items():
        result[
            f"v9_penalty_{key}"
        ] = float(value)

    return result


def _call_binary_reward(
    data_source: Any,
    solution_str: str,
    ground_truth: Any,
    extra_info: Any,
    additional_kwargs: dict[str, Any],
):
    available_arguments = {
        "data_source": data_source,
        "solution_str": solution_str,
        "ground_truth": ground_truth,
        "extra_info": extra_info,
        **additional_kwargs,
    }

    signature = inspect.signature(
        _BINARY_COMPUTE_SCORE
    )

    accepts_arbitrary_kwargs = any(
        parameter.kind
        == inspect.Parameter.VAR_KEYWORD
        for parameter
        in signature.parameters.values()
    )

    if accepts_arbitrary_kwargs:
        call_arguments = available_arguments
    else:
        call_arguments = {
            key: value
            for key, value
            in available_arguments.items()
            if key in signature.parameters
        }

    return _BINARY_COMPUTE_SCORE(
        **call_arguments
    )


def compute_score(
    data_source: Any,
    solution_str: str,
    ground_truth: Any,
    extra_info: Any = None,
    **kwargs: Any,
):
    binary_result = _call_binary_reward(
        data_source=data_source,
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
        additional_kwargs=kwargs,
    )

    if isinstance(binary_result, dict):
        base_result = dict(binary_result)
        execution_correct = float(
            base_result.get(
                "execution_correct",
                base_result.get(
                    "score",
                    0.0,
                ),
            )
        )
    else:
        execution_correct = float(
            binary_result
        )

        base_result = {
            "binary_score": execution_correct,
            "execution_correct": (
                execution_correct
            ),
        }

    try:
        process_metrics = (
            extract_tool_process_metrics(
                solution_str
            )
        )

        extraction_failed = 0.0
    except Exception:
        process_metrics = {}
        extraction_failed = 1.0

    composed = compose_score(
        execution_correct=execution_correct,
        process_metrics=process_metrics,
    )

    binary_score = float(
        base_result.get(
            "score",
            execution_correct,
        )
    )

    return {
        **base_result,
        **process_metrics,
        **composed,
        "binary_score": binary_score,
        "v9_metric_extraction_failed": (
            extraction_failed
        ),
    }
