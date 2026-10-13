"""Historical ground-truth field parsing."""
from __future__ import annotations
import json
from typing import Any

def _parse_ground_truth(
    ground_truth: Any,
    extra_info: dict | None,
) -> tuple[str, str]:
    if isinstance(ground_truth, str):
        try:
            parsed = json.loads(ground_truth)
        except json.JSONDecodeError:
            parsed = None
    elif isinstance(ground_truth, dict):
        parsed = ground_truth
    else:
        parsed = None

    if isinstance(parsed, dict):
        db_id = parsed.get("db_id")
        gold_sql = parsed.get("query")
    else:
        db_id = None
        gold_sql = None

    if extra_info:
        db_id = db_id or extra_info.get("db_id")
        gold_sql = gold_sql or extra_info.get(
            "gold_sql"
        )

    if not isinstance(db_id, str) or not db_id:
        raise ValueError(
            "Ground truth does not contain db_id"
        )

    if not isinstance(gold_sql, str) or not gold_sql:
        raise ValueError(
            "Ground truth does not contain query"
        )

    return db_id, gold_sql
