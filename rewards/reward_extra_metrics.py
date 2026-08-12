"""Aggregate custom reward and GRPO metrics for training loggers."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np


_RATE_KEYS = (
    "execution_correct",
    "sql_executable",
    "format_compliance",
    "answer_found",
    "gold_executable",
    "tool_called",
    "list_tables_called",
    "get_table_schema_called",
    "execute_sql_called",
    "tool_error",
    "tool_error_recovered",
    "stopped_after_tool_error",
    "execute_sql_succeeded",
    "execute_sql_after_schema",
    "tool_protocol_valid",
    "list_tables_succeeded",
    "get_table_schema_succeeded",
    "schema_after_list_tables",
    "execute_sql_after_successful_schema",
    "tool_error_recovered_successfully",
    "ordered_tool_chain",
    "malformed_tool_call",
    "tool_failure",
)

_COUNT_KEYS = (
    "tool_call_count",
    "tool_success_count",
    "tool_error_count",
    "list_tables_call_count",
    "get_table_schema_call_count",
    "execute_sql_call_count",
    "malformed_tool_call_count",
)

_COMPONENT_KEYS = (
    "score",
    "terminal_reward",
    "process_reward",
    "accuracy_reward",
    "executable_reward",
    "format_reward",
    "list_tables_reward",
    "schema_reward",
    "execute_tool_reward",
    "recovery_reward",
    "ordered_chain_reward",
)


def _as_float_array(values: Any) -> np.ndarray:
    if values is None:
        return np.asarray([], dtype=np.float64)

    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()

    try:
        array = np.asarray(values, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        converted = []
        for value in values:
            if hasattr(value, "detach"):
                value = value.detach().cpu().item()
            converted.append(float(value))
        array = np.asarray(converted, dtype=np.float64)

    return array[np.isfinite(array)]


def _conditional_rate(
    numerator: np.ndarray,
    condition: np.ndarray,
) -> float | None:
    size = min(numerator.size, condition.size)

    if size == 0:
        return None

    numerator = numerator[:size]
    mask = condition[:size] > 0.5

    if not np.any(mask):
        return None

    return float(np.mean(numerator[mask]))


def compute_reward_extra_metrics(
    reward_extra_infos_dict: dict[str, Any] | None,
    batch: Any,
) -> dict[str, float]:
    """Convert per-trajectory reward extras into per-step scalar metrics."""

    if not reward_extra_infos_dict:
        return {}

    arrays = {
        key: _as_float_array(values)
        for key, values in reward_extra_infos_dict.items()
    }

    metrics: dict[str, float] = {}

    # Preserve the mean of every scalar returned by the reward function.
    for key, values in arrays.items():
        if values.size:
            metrics[f"reward_extra/{key}/mean"] = float(np.mean(values))

    # Important binary trajectory statistics.
    for key in _RATE_KEYS:
        values = arrays.get(key)

        if values is not None and values.size:
            metrics[f"text2sql/{key}_rate"] = float(np.mean(values))

    # Tool-call counts.
    for key in _COUNT_KEYS:
        values = arrays.get(key)

        if values is not None and values.size:
            metrics[f"text2sql/{key}/mean"] = float(np.mean(values))
            metrics[f"text2sql/{key}/max"] = float(np.max(values))

    # Reward distributions, not only one mean.
    for key in _COMPONENT_KEYS:
        values = arrays.get(key)

        if values is not None and values.size:
            prefix = f"text2sql/reward/{key}"
            metrics[f"{prefix}/mean"] = float(np.mean(values))
            metrics[f"{prefix}/std"] = float(np.std(values))
            metrics[f"{prefix}/min"] = float(np.min(values))
            metrics[f"{prefix}/max"] = float(np.max(values))

    score = arrays.get("score")

    if score is not None and score.size:
        metrics["text2sql/reward/zero_trajectory_rate"] = float(
            np.mean(np.isclose(score, 0.0))
        )
        metrics["text2sql/reward/positive_trajectory_rate"] = float(
            np.mean(score > 0.0)
        )

    extraction_method = arrays.get("extraction_method")

    if extraction_method is not None and extraction_method.size:
        extraction_names = {
            0: "none",
            1: "final_sql",
            2: "sql_fence",
            3: "direct_sql",
        }

        for code, name in extraction_names.items():
            metrics[f"text2sql/extraction/{name}_rate"] = float(
                np.mean(np.isclose(extraction_method, code))
            )

    # Conditional agent metrics.
    conditional_specs = (
        (
            "text2sql/tool_error_recovery_rate_given_error",
            "tool_error_recovered",
            "tool_error",
        ),
        (
            "text2sql/stopped_after_tool_error_rate_given_error",
            "stopped_after_tool_error",
            "tool_error",
        ),
        (
            "text2sql/execute_sql_success_rate_given_called",
            "execute_sql_succeeded",
            "execute_sql_called",
        ),
        (
            "text2sql/execute_sql_after_schema_rate_given_called",
            "execute_sql_after_schema",
            "execute_sql_called",
        ),
    )

    for metric_name, numerator_key, condition_key in conditional_specs:
        numerator = arrays.get(numerator_key)
        condition = arrays.get(condition_key)

        if numerator is None or condition is None:
            continue

        value = _conditional_rate(numerator, condition)

        if value is not None:
            metrics[metric_name] = value

    # GRPO group statistics. uid identifies different samples from one prompt.
    if score is not None and score.size:
        non_tensor_batch = getattr(batch, "non_tensor_batch", {})
        uid_values = non_tensor_batch.get("uid")

        if uid_values is not None:
            uid_values = np.asarray(uid_values, dtype=object).reshape(-1)
            size = min(uid_values.size, score.size)

            groups: dict[str, list[float]] = defaultdict(list)

            for uid, value in zip(
                uid_values[:size],
                score[:size],
                strict=True,
            ):
                groups[str(uid)].append(float(value))

            if groups:
                group_arrays = [
                    np.asarray(values, dtype=np.float64)
                    for values in groups.values()
                ]

                group_stds = np.asarray(
                    [np.std(values) for values in group_arrays],
                    dtype=np.float64,
                )
                group_ranges = np.asarray(
                    [np.max(values) - np.min(values) for values in group_arrays],
                    dtype=np.float64,
                )
                group_sizes = np.asarray(
                    [values.size for values in group_arrays],
                    dtype=np.float64,
                )
                all_zero = np.asarray(
                    [
                        bool(np.all(np.isclose(values, 0.0)))
                        for values in group_arrays
                    ],
                    dtype=np.float64,
                )
                all_same = np.asarray(
                    [value <= 1e-8 for value in group_ranges],
                    dtype=np.float64,
                )
                informative = 1.0 - all_same

                metrics.update(
                    {
                        "grpo/group_count": float(len(group_arrays)),
                        "grpo/group_size_mean": float(np.mean(group_sizes)),
                        "grpo/group_reward_std_mean": float(
                            np.mean(group_stds)
                        ),
                        "grpo/group_reward_std_max": float(
                            np.max(group_stds)
                        ),
                        "grpo/group_reward_range_mean": float(
                            np.mean(group_ranges)
                        ),
                        "grpo/group_reward_range_max": float(
                            np.max(group_ranges)
                        ),
                        "grpo/all_zero_group_rate": float(
                            np.mean(all_zero)
                        ),
                        "grpo/uniform_reward_group_rate": float(
                            np.mean(all_same)
                        ),
                        "grpo/informative_group_rate": float(
                            np.mean(informative)
                        ),
                    }
                )

    # REWARD_V4_METRICS_START
    v4_rate_keys = (
        "v4_changed_sql_recovery",
        "v4_unknown_column_recovery",
        "v4_final_matches_successful_execute",
        "v4_final_was_failed",
        "v4_final_unverified",
        "v4_has_failed_sql_reuse",
    )

    for key in v4_rate_keys:
        values = arrays.get(key)

        if values is not None and values.size:
            metrics[f"text2sql/{key}_rate"] = float(
                np.mean(values)
            )

    v4_count_keys = (
        "v4_failed_sql_reuse_count",
    )

    for key in v4_count_keys:
        values = arrays.get(key)

        if values is not None and values.size:
            metrics[f"text2sql/{key}/mean"] = float(
                np.mean(values)
            )
            metrics[f"text2sql/{key}/max"] = float(
                np.max(values)
            )

    v4_component_keys = (
        "structural_reward",
        "v4_recovery_reward",
        "v4_final_consistency_reward",
        "v4_failed_final_penalty",
        "v4_unverified_final_penalty",
        "v4_failed_sql_reuse_penalty",
        "v4_new_penalty",
        "v4_table_f1",
        "v4_column_f1",
        "v4_join_f1",
    )

    for key in v4_component_keys:
        values = arrays.get(key)

        if values is not None and values.size:
            prefix = f"text2sql/reward/{key}"

            metrics[f"{prefix}/mean"] = float(
                np.mean(values)
            )
            metrics[f"{prefix}/std"] = float(
                np.std(values)
            )
            metrics[f"{prefix}/min"] = float(
                np.min(values)
            )
            metrics[f"{prefix}/max"] = float(
                np.max(values)
            )

    changed_recovery = arrays.get(
        "v4_changed_sql_recovery"
    )
    tool_error = arrays.get("tool_error")

    if (
        changed_recovery is not None
        and tool_error is not None
    ):
        value = _conditional_rate(
            changed_recovery,
            tool_error,
        )

        if value is not None:
            metrics[
                "text2sql/"
                "v4_changed_recovery_rate_given_error"
            ] = value

    unknown_recovery = arrays.get(
        "v4_unknown_column_recovery"
    )

    if (
        unknown_recovery is not None
        and tool_error is not None
    ):
        value = _conditional_rate(
            unknown_recovery,
            tool_error,
        )

        if value is not None:
            metrics[
                "text2sql/"
                "v4_unknown_column_recovery_rate_given_error"
            ] = value
    # REWARD_V4_METRICS_END

    # REWARD_V5_METRICS_START
    v5_rate_keys = (
        "v5_changed_sql_recovery",
        "v5_verified_structural_eligible",
        "v5_unknown_column_recovery",
        "v5_final_matches_successful_execute",
        "v5_final_was_failed",
        "v5_final_unverified",
        "v5_has_failed_sql_reuse",
    )

    for key in v5_rate_keys:
        values = arrays.get(key)

        if values is not None and values.size:
            metrics[f"text2sql/{key}_rate"] = float(
                np.mean(values)
            )

    v5_count_keys = (
        "v5_failed_sql_reuse_count",
    )

    for key in v5_count_keys:
        values = arrays.get(key)

        if values is not None and values.size:
            metrics[f"text2sql/{key}/mean"] = float(
                np.mean(values)
            )
            metrics[f"text2sql/{key}/max"] = float(
                np.max(values)
            )

    v5_component_keys = (
        "structural_reward",
        "v5_recovery_reward",
        "v5_final_consistency_reward",
        "v5_failed_final_penalty",
        "v5_unverified_final_penalty",
        "v5_failed_sql_reuse_penalty",
        "v5_new_penalty",
        "v5_table_f1",
        "v5_column_f1",
        "v5_join_f1",
    )

    for key in v5_component_keys:
        values = arrays.get(key)

        if values is not None and values.size:
            prefix = f"text2sql/reward/{key}"

            metrics[f"{prefix}/mean"] = float(
                np.mean(values)
            )
            metrics[f"{prefix}/std"] = float(
                np.std(values)
            )
            metrics[f"{prefix}/min"] = float(
                np.min(values)
            )
            metrics[f"{prefix}/max"] = float(
                np.max(values)
            )

    changed_recovery = arrays.get(
        "v5_changed_sql_recovery"
    )
    tool_error = arrays.get("tool_error")

    if (
        changed_recovery is not None
        and tool_error is not None
    ):
        value = _conditional_rate(
            changed_recovery,
            tool_error,
        )

        if value is not None:
            metrics[
                "text2sql/"
                "v5_changed_recovery_rate_given_error"
            ] = value

    unknown_recovery = arrays.get(
        "v5_unknown_column_recovery"
    )

    if (
        unknown_recovery is not None
        and tool_error is not None
    ):
        value = _conditional_rate(
            unknown_recovery,
            tool_error,
        )

        if value is not None:
            metrics[
                "text2sql/"
                "v5_unknown_column_recovery_rate_given_error"
            ] = value
    # REWARD_V5_METRICS_END

    return metrics

# REWARD_V3_WANDB_METRICS_BEGIN
# Reward v3 fields are appended instead of replacing existing v1/v2 metrics.
_RATE_KEYS = tuple(
    dict.fromkeys(
        (
            *_RATE_KEYS,
            "v3_has_exact_duplicate",
            "v3_has_duplicate_schema",
            "v3_has_duplicate_execute",
            "v3_correct_no_duplicate",
            "v3_final_no_duplicate_execute",
        )
    )
)

_COUNT_KEYS = tuple(
    dict.fromkeys(
        (
            *_COUNT_KEYS,
            "v3_valid_tool_call_count",
            "v3_unique_tool_call_count",
            "v3_duplicate_tool_call_count",
            "v3_duplicate_list_tables_count",
            "v3_duplicate_schema_count",
            "v3_duplicate_execute_count",
            "v3_duplicate_other_count",
            "v3_locally_malformed_tool_call_count",
        )
    )
)

_COMPONENT_KEYS = tuple(
    dict.fromkeys(
        (
            *_COMPONENT_KEYS,
            "score_pre_clip",
            "terminal_reward",
            "process_reward",
            "efficiency_reward",
            "total_penalty",
            "penalty_reward",
            "accuracy_reward",
            "executable_reward",
            "format_reward",
            "list_tables_reward",
            "schema_reward",
            "execute_tool_reward",
            "recovery_reward",
            "ordered_chain_reward",
            "correct_efficiency_reward",
            "final_efficiency_reward",
            "duplicate_list_tables_penalty",
            "duplicate_schema_penalty",
            "duplicate_execute_penalty",
            "malformed_penalty",
            "stopped_after_error_penalty",
        )
    )
)

# REWARD_V6_METRICS_START
_RATE_KEYS = tuple(
    dict.fromkeys(
        (
            *_RATE_KEYS,
            "v6_execution_correct",
            "v6_executable_but_wrong",
            "v6_verified_semantic_eligible",
            "v6_final_ok",
            "v6_tool_error",
            "v6_duplicate_execute",
            "v6_incorrect_cap_applied",
            "v6_correct_floor_applied",
        )
    )
)

_COMPONENT_KEYS = tuple(
    dict.fromkeys(
        (
            *_COMPONENT_KEYS,
            "v6_score",
            "v6_score_pre_clip",
            "v6_base_v5_score",
            "v6_score_delta_from_v5",
            "v6_terminal_reward",
            "v6_executable_reward",
            "v6_format_reward",
            "v6_semantic_reward",
            "v6_verification_reward",
            "v6_process_reward",
            "v6_recovery_reward",
            "v6_semantic_gap_penalty",
            "v6_reliability_penalty",
            "v6_total_penalty",
        )
    )
)
# REWARD_V6_METRICS_END

# REWARD_V3_WANDB_METRICS_END


# REWARD_V7_METRICS_START
_RATE_KEYS = tuple(
    dict.fromkeys(
        (
            *_RATE_KEYS,
            "v7_correct",
            "v7_wrong_executable",
            "v7_negative_trajectory",
            "v7_no_positive_wrong_reward",
            "v7_structure_used_for_reward",
            "v7_semantic_used_for_reward",
        )
    )
)

_COMPONENT_KEYS = tuple(
    dict.fromkeys(
        (
            *_COMPONENT_KEYS,
            "v7_score",
            "v7_score_pre_clip",
            "v7_correctness_reward",
            "v7_structural_reward",
            "v7_semantic_reward",
            "v7_process_bonus",
            "v7_wrong_base_penalty",
            "v7_final_state_penalty",
            "v7_reliability_penalty",
            "v7_total_penalty",
            "v7_table_f1_monitor",
            "v7_column_f1_monitor",
            "v7_join_f1_monitor",
        )
    )
)
# REWARD_V7_METRICS_END

# REWARD_V9_METRICS_START
_RATE_KEYS = tuple(
    dict.fromkeys(
        (
            *_RATE_KEYS,
            "v9_execution_correct",
            "v9_has_process_signal",
            "v9_incorrect_with_process",
            "v9_duplicate_execute",
            "v9_metric_extraction_failed",
        )
    )
)

_COMPONENT_KEYS = tuple(
    dict.fromkeys(
        (
            *_COMPONENT_KEYS,
            "v9_score",
            "v9_score_pre_clip",
            "v9_terminal_reward",
            "v9_process_reward",
            "v9_process_penalty",
            "v9_process_tool_protocol_valid",
            "v9_process_list_tables_succeeded",
            "v9_process_schema_after_list_tables",
            "v9_process_execute_sql_after_successful_schema",
            "v9_process_execute_sql_succeeded",
            "v9_process_tool_error_recovered_successfully",
            "v9_penalty_malformed_tool_call",
            "v9_penalty_stopped_after_tool_error",
            "v9_penalty_duplicate_execute",
        )
    )
)
# REWARD_V9_METRICS_END

# REWARD_V10_METRICS_START
_V10_RATE_KEYS = (
    "v10_eligible_correct_process",
    "v10_protocol_invalid",
    "v10_direct",
    "v10_malformed",
    "v10_stopped_after_error",
    "v10_positive_incorrect",
    "v10_wrong_process_nonzero",
    "v10_incorrect_uniform",
)

_V10_COMPONENT_KEYS = (
    "v10_score",
    "v10_terminal_reward",
    "v10_process_quality",
    "v10_process_bonus",
    "v10_correct_penalty",
    "v10_branch_code",
)

_RATE_KEYS = tuple(
    dict.fromkeys(
        (
            *_RATE_KEYS,
            *_V10_RATE_KEYS,
        )
    )
)

_COMPONENT_KEYS = tuple(
    dict.fromkeys(
        (
            *_COMPONENT_KEYS,
            *_V10_COMPONENT_KEYS,
        )
    )
)
# REWARD_V10_METRICS_END
