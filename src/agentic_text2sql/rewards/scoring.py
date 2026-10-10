"""Standalone correctness-gated scoring entry.

Historical score semantics are retained for valid configured inputs.
Historical structural-reward diagnostic fields are intentionally omitted.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from agentic_text2sql.evaluation.ground_truth import _parse_ground_truth
from agentic_text2sql.evaluation.sql_execution import evaluate_answer
from agentic_text2sql.evaluation.tool_process import extract_tool_process_metrics
from agentic_text2sql.evaluation.tool_efficiency import extract_v3_efficiency_metrics
from agentic_text2sql.rewards.correctness_gated import compose_reward


def _resolve_database(db_id: str) -> Path:
    if not isinstance(db_id, str) or not re.fullmatch(r"[A-Za-z0-9_]+", db_id):
        raise ValueError(f"Unsafe db_id: {db_id!r}")

    configured = os.environ.get("SPIDER_DATABASE_ROOT")
    if not configured:
        raise RuntimeError("SPIDER_DATABASE_ROOT must be explicitly configured")

    root = Path(configured).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)

    path = (root / db_id / f"{db_id}.sqlite").resolve()
    if root not in path.parents:
        raise ValueError("Database path escaped database root")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict | None = None,
) -> dict[str, float]:
    del data_source

    # Configuration and metadata failures are explicit, not silently zero-scored.
    db_id, gold_sql = _parse_ground_truth(ground_truth, extra_info)
    db_path = _resolve_database(db_id)

    evaluated = evaluate_answer(db_path, solution_str, gold_sql)
    error_type = evaluated.pop("evaluation_error_type")
    result = dict(evaluated)
    result["evaluation_error"] = float(error_type is not None)

    result.update(extract_tool_process_metrics(solution_str))
    result.update(extract_v3_efficiency_metrics(solution_str))

    # This is the exact derived metric used by historical V3 and inherited by V10.
    result["v3_final_no_duplicate_execute"] = (
        result["answer_found"]
        * float(result["v3_duplicate_execute_count"] == 0.0)
    )

    result.update(compose_reward(result))
    result.update({
        "reward_version": 10.0,
        "score_pre_clip": result["v10_score"],
        "terminal_reward": result["v10_terminal_reward"],
        "process_reward": result["v10_process_bonus"],
        "total_penalty": result["v10_correct_penalty"],
        "penalty_reward": -result["v10_correct_penalty"],
        "binary_execution_reward": float(result["execution_correct"] > 0.5),
    })
    return result
