"""Exact-duplicate metrics extracted without reward reweighting."""
from __future__ import annotations
from collections import Counter
from typing import Any
import json
import re

_TOOL_CALL_PATTERN = re.compile(
    r"<tool_call>\s*(.*?)\s*</tool_call>",
    flags=re.IGNORECASE | re.DOTALL,
)

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
