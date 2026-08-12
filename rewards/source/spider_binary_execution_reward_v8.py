#!/usr/bin/env python
"""Binary execution-correctness reward for Spider Text-to-SQL.

Reward definition:
    1.0 if the recognized final SQL has the same execution
    result as the gold SQL.
    0.0 otherwise.

The robust SQL extraction and execution evaluator is reused from
Reward v5, but no format, process, structural, executability, or
efficiency component contributes to the optimization score.
"""

from __future__ import annotations

from importlib.util import (
    module_from_spec,
    spec_from_file_location,
)
from pathlib import Path
from typing import Any


_BASE_REWARD_PATH = Path(__file__).with_name(
    "spider_verified_structural_reward_v5.py"
)

_SPEC = spec_from_file_location(
    "_spider_reward_v5_for_binary_execution",
    _BASE_REWARD_PATH,
)

if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(
        f"Cannot load base reward: {_BASE_REWARD_PATH}"
    )

_BASE_MODULE = module_from_spec(_SPEC)
_SPEC.loader.exec_module(_BASE_MODULE)

_BASE_COMPUTE_SCORE = _BASE_MODULE.compute_score


def binary_from_base_result(
    base_result: dict[str, Any],
) -> dict[str, Any]:
    """Collapse the rich evaluator result to a binary score."""

    result = dict(base_result)

    execution_correct = float(
        result.get(
            "execution_correct",
            0.0,
        )
        or 0.0
    )

    binary_score = (
        1.0
        if execution_correct >= 0.5
        else 0.0
    )

    original_score = float(
        result.get(
            "score",
            0.0,
        )
        or 0.0
    )

    result.update(
        {
            "score": binary_score,
            "score_pre_clip": binary_score,
            "binary_execution_reward": binary_score,
            "binary_execution_correct": binary_score,
            "binary_reward_version": 1.0,
            "binary_base_v5_score": original_score,
            "binary_positive_incorrect": (
                1.0
                if (
                    binary_score > 0.0
                    and execution_correct < 0.5
                )
                else 0.0
            ),
        }
    )

    return result


def compute_score(
    *args,
    **kwargs,
) -> dict[str, Any]:
    """Evaluate task correctness, then return only 0/1 reward."""

    base_result = _BASE_COMPUTE_SCORE(
        *args,
        **kwargs,
    )

    if not isinstance(
        base_result,
        dict,
    ):
        raise TypeError(
            "Reward v5 compute_score must return a dict, "
            f"got {type(base_result).__name__}"
        )

    return binary_from_base_result(
        base_result
    )
