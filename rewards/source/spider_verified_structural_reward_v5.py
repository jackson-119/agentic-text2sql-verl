from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from typing import Any


_REWARD_V4_PATH = (
    Path(__file__).resolve().parent
    / "spider_structural_reward_v4.py"
)


def _load_reward_v4():
    module_name = (
        "verl_text2sql_spider_structural_reward_v4_for_v5"
    )

    existing = sys.modules.get(
        module_name
    )

    if existing is not None:
        return existing

    spec = importlib.util.spec_from_file_location(
        module_name,
        _REWARD_V4_PATH,
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"无法加载Reward v4：{_REWARD_V4_PATH}"
        )

    module = importlib.util.module_from_spec(
        spec
    )

    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    return module


_REWARD_V4 = _load_reward_v4()


_SCORE_MIN = -0.20
_SCORE_MAX = 1.00

_TABLE_WEIGHT = 0.025
_COLUMN_WEIGHT = 0.025
_JOIN_WEIGHT = 0.010

_RECOVERY_REWARD = 0.040
_FINAL_CONSISTENCY_REWARD = 0.050

_FAILED_FINAL_PENALTY = 0.080
_UNVERIFIED_FINAL_PENALTY = 0.060

_FAILED_SQL_REUSE_UNIT_PENALTY = 0.020
_FAILED_SQL_REUSE_MAX_PENALTY = 0.060


def _number(
    mapping: dict[str, Any],
    *names: str,
    default: float = 0.0,
) -> float:
    for name in names:
        value = mapping.get(name)

        if value is None:
            continue

        try:
            number = float(value)
        except (TypeError, ValueError):
            continue

        if math.isfinite(number):
            return number

    return float(default)


def _flag(
    mapping: dict[str, Any],
    *names: str,
) -> bool:
    return _number(
        mapping,
        *names,
    ) > 0.5


def _clip(value: float) -> float:
    return min(
        _SCORE_MAX,
        max(
            _SCORE_MIN,
            float(value),
        ),
    )


def _resolve_v4_pre_clip(
    result: dict[str, Any],
) -> float:
    v4_score = _number(
        result,
        "score",
    )

    candidates = []

    for name in (
        "v4_score_pre_clip",
        "score_pre_clip",
    ):
        if result.get(name) is None:
            continue

        candidate = _number(
            result,
            name,
        )

        candidates.append(
            (
                name,
                candidate,
            )
        )

        if abs(
            _clip(candidate)
            - v4_score
        ) <= 1e-6:
            return candidate

    if (
        _SCORE_MIN + 1e-6
        < v4_score
        < _SCORE_MAX - 1e-6
    ):
        return v4_score

    raise RuntimeError(
        "无法确定Reward v4裁剪前分数："
        f"score={v4_score}, "
        f"candidates={candidates}"
    )


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict | None = None,
) -> dict[str, float]:
    result = dict(
        _REWARD_V4.compute_score(
            data_source=data_source,
            solution_str=solution_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
        )
    )

    v4_score = _number(
        result,
        "score",
    )

    v4_pre_clip = _resolve_v4_pre_clip(
        result
    )

    old_structural_reward = _number(
        result,
        "structural_reward",
        "v4_structural_reward",
    )

    old_recovery_reward = _number(
        result,
        "v4_recovery_reward",
    )

    old_final_consistency_reward = _number(
        result,
        "v4_final_consistency_reward",
    )

    old_new_penalty = _number(
        result,
        "v4_new_penalty",
        default=(
            _number(
                result,
                "v4_failed_final_penalty",
            )
            + _number(
                result,
                "v4_unverified_final_penalty",
            )
            + _number(
                result,
                "v4_failed_sql_reuse_penalty",
            )
        ),
    )

    base_score = (
        v4_pre_clip
        - old_structural_reward
        - old_recovery_reward
        - old_final_consistency_reward
        + old_new_penalty
    )

    table_f1 = max(
        0.0,
        min(
            1.0,
            _number(
                result,
                "v4_table_f1",
            ),
        ),
    )

    column_f1 = max(
        0.0,
        min(
            1.0,
            _number(
                result,
                "v4_column_f1",
            ),
        ),
    )

    join_f1 = max(
        0.0,
        min(
            1.0,
            _number(
                result,
                "v4_join_f1",
            ),
        ),
    )

    execution_correct = _flag(
        result,
        "execution_correct",
    )

    final_matches_successful_execute = _flag(
        result,
        "v4_final_matches_successful_execute",
    )

    final_was_failed = _flag(
        result,
        "v4_final_was_failed",
    )

    final_unverified = _flag(
        result,
        "v4_final_unverified",
    )

    changed_sql_recovery = _flag(
        result,
        "v4_changed_sql_recovery",
    )

    unknown_column_recovery = _flag(
        result,
        "v4_unknown_column_recovery",
    )

    has_failed_sql_reuse = _flag(
        result,
        "v4_has_failed_sql_reuse",
    )

    failed_sql_reuse_count = max(
        0.0,
        _number(
            result,
            "v4_failed_sql_reuse_count",
        ),
    )

    if (
        has_failed_sql_reuse
        and failed_sql_reuse_count < 1.0
    ):
        failed_sql_reuse_count = 1.0

    verified_structural_eligible = (
        not execution_correct
        and final_matches_successful_execute
        and not final_was_failed
        and not final_unverified
    )

    if verified_structural_eligible:
        structural_table_component = (
            _TABLE_WEIGHT
            * table_f1
        )

        structural_column_component = (
            _COLUMN_WEIGHT
            * column_f1
        )

        structural_join_component = (
            _JOIN_WEIGHT
            * join_f1
        )
    else:
        structural_table_component = 0.0
        structural_column_component = 0.0
        structural_join_component = 0.0

    structural_reward = (
        structural_table_component
        + structural_column_component
        + structural_join_component
    )

    recovery_reward = (
        _RECOVERY_REWARD
        if changed_sql_recovery
        else 0.0
    )

    final_consistency_reward = (
        _FINAL_CONSISTENCY_REWARD
        if final_matches_successful_execute
        else 0.0
    )

    failed_final_penalty = (
        _FAILED_FINAL_PENALTY
        if final_was_failed
        else 0.0
    )

    unverified_final_penalty = (
        _UNVERIFIED_FINAL_PENALTY
        if (
            final_unverified
            and not final_was_failed
        )
        else 0.0
    )

    failed_sql_reuse_penalty = min(
        _FAILED_SQL_REUSE_MAX_PENALTY,
        (
            _FAILED_SQL_REUSE_UNIT_PENALTY
            * failed_sql_reuse_count
        ),
    )

    new_penalty = (
        failed_final_penalty
        + unverified_final_penalty
        + failed_sql_reuse_penalty
    )

    score_pre_clip = (
        base_score
        + structural_reward
        + recovery_reward
        + final_consistency_reward
        - new_penalty
    )

    score = _clip(
        score_pre_clip
    )

    result.update(
        {
            "v4_score_original": v4_score,
            "v4_score_pre_clip_original": v4_pre_clip,
            "v4_structural_reward_original": (
                old_structural_reward
            ),
            "v4_recovery_reward_original": (
                old_recovery_reward
            ),
            "v4_final_consistency_reward_original": (
                old_final_consistency_reward
            ),
            "v4_new_penalty_original": old_new_penalty,
            "v5_base_score": base_score,
            "v5_verified_structural_eligible": float(
                verified_structural_eligible
            ),
            "v5_structural_table_component": (
                structural_table_component
            ),
            "v5_structural_column_component": (
                structural_column_component
            ),
            "v5_structural_join_component": (
                structural_join_component
            ),
            "v5_structural_reward": structural_reward,
            "v5_changed_sql_recovery": float(
                changed_sql_recovery
            ),
            "v5_unknown_column_recovery": float(
                unknown_column_recovery
            ),
            "v5_recovery_reward": recovery_reward,
            "v5_final_matches_successful_execute": float(
                final_matches_successful_execute
            ),
            "v5_final_consistency_reward": (
                final_consistency_reward
            ),
            "v5_final_was_failed": float(
                final_was_failed
            ),
            "v5_failed_final_penalty": (
                failed_final_penalty
            ),
            "v5_final_unverified": float(
                final_unverified
            ),
            "v5_unverified_final_penalty": (
                unverified_final_penalty
            ),
            "v5_has_failed_sql_reuse": float(
                has_failed_sql_reuse
            ),
            "v5_failed_sql_reuse_count": (
                failed_sql_reuse_count
            ),
            "v5_failed_sql_reuse_penalty": (
                failed_sql_reuse_penalty
            ),
            "v5_new_penalty": new_penalty,
            "v5_table_f1": table_f1,
            "v5_column_f1": column_f1,
            "v5_join_f1": join_f1,
            "v5_score_pre_clip": score_pre_clip,
            "v5_score": score,
            "structural_reward": structural_reward,
            "score_pre_clip": score_pre_clip,
            "score": score,
        }
    )

    return result
