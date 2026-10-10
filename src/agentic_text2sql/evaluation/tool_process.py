"""Extract scalar metrics from Text-to-SQL agent trajectories."""

from __future__ import annotations

import json
import re
from typing import Any


_EVENT_PATTERN = re.compile(
    r"(?is)<tool_call>\s*(.*?)\s*</tool_call>|"
    r"<tool_response>\s*(.*?)\s*</tool_response>"
)


def _parse_json_object(payload: str) -> dict[str, Any] | None:
    if not isinstance(payload, str):
        return None

    try:
        value = json.loads(payload.strip())
    except (json.JSONDecodeError, TypeError):
        return None

    return value if isinstance(value, dict) else None


def _fallback_tool_name(payload: str) -> str:
    match = re.search(
        r"(?is)[\"']name[\"']\s*:\s*[\"']([^\"']+)[\"']",
        payload,
    )
    return match.group(1) if match else ""


def _response_status(
    payload: str,
    parsed: dict[str, Any] | None,
) -> tuple[bool, bool, bool]:
    """Return recognized, error, success."""

    if parsed is not None:
        error = bool(
            parsed.get("ok") is False
            or parsed.get("error_type")
            or parsed.get("error")
        )
        return True, error, not error

    error = bool(
        re.search(
            r"(?is)[\"']ok[\"']\s*:\s*false|"
            r"[\"']error(?:_type)?[\"']\s*:",
            payload,
        )
    )
    success = bool(
        re.search(
            r"(?is)[\"']ok[\"']\s*:\s*true",
            payload,
        )
    )

    return error or success, error, success and not error


def extract_tool_process_metrics(
    solution_str: str,
) -> dict[str, float]:
    """Return capped tool-use metrics without changing reward score."""

    if not isinstance(solution_str, str):
        solution_str = ""

    records: list[dict[str, Any]] = []

    for match in _EVENT_PATTERN.finditer(solution_str):
        call_payload = match.group(1)
        response_payload = match.group(2)

        if call_payload is not None:
            parsed_call = _parse_json_object(call_payload)
            name = (
                parsed_call.get("name")
                if parsed_call is not None
                else None
            )

            if not isinstance(name, str) or not name:
                name = _fallback_tool_name(call_payload)

            records.append(
                {
                    "name": name,
                    "call_valid": bool(parsed_call and name),
                    "response_seen": False,
                    "response_valid": False,
                    "response_error": False,
                    "response_success": False,
                }
            )
            continue

        pending = next(
            (
                record
                for record in reversed(records)
                if not record["response_seen"]
            ),
            None,
        )

        if pending is None or response_payload is None:
            continue

        parsed_response = _parse_json_object(response_payload)
        recognized, response_error, response_success = (
            _response_status(
                response_payload,
                parsed_response,
            )
        )

        pending["response_seen"] = True
        pending["response_valid"] = recognized
        pending["response_error"] = response_error
        pending["response_success"] = response_success

    names = [record["name"] for record in records]

    error_indices = [
        index
        for index, record in enumerate(records)
        if record["response_error"]
    ]
    malformed_indices = [
        index
        for index, record in enumerate(records)
        if not record["call_valid"]
    ]
    success_indices = [
        index
        for index, record in enumerate(records)
        if record["response_success"]
    ]

    if error_indices:
        last_error_index = error_indices[-1]
        continued_after_error = any(
            index > last_error_index
            for index in range(len(records))
        )
        successfully_recovered = any(
            index > last_error_index
            for index in success_indices
        )
    else:
        continued_after_error = False
        successfully_recovered = False

    def indices(name: str, require_success: bool) -> list[int]:
        return [
            index
            for index, record in enumerate(records)
            if record["name"] == name
            and (
                not require_success
                or record["response_success"]
            )
        ]

    list_calls = indices("list_tables", False)
    schema_calls = indices("get_table_schema", False)
    execute_calls = indices("execute_sql", False)

    list_successes = indices("list_tables", True)
    schema_successes = indices("get_table_schema", True)
    execute_successes = indices("execute_sql", True)

    schema_after_list = any(
        list_index < schema_index
        for list_index in list_successes
        for schema_index in schema_successes
    )

    execute_after_schema = any(
        schema_index < execute_index
        for schema_index in schema_calls
        for execute_index in execute_calls
    )

    execute_after_successful_schema = any(
        schema_index < execute_index
        for schema_index in schema_successes
        for execute_index in execute_successes
    )

    ordered_tool_chain = any(
        list_index < schema_index < execute_index
        for list_index in list_successes
        for schema_index in schema_successes
        for execute_index in execute_successes
    )

    protocol_valid = bool(
        records
        and all(
            record["call_valid"]
            and record["response_seen"]
            and record["response_valid"]
            for record in records
        )
    )

    def count(name: str) -> float:
        return float(sum(item == name for item in names))

    return {
        "tool_called": float(bool(records)),
        "tool_call_count": float(len(records)),
        "tool_success_count": float(len(success_indices)),
        "tool_error_count": float(len(error_indices)),
        "tool_error": float(bool(error_indices)),
        "tool_error_recovered": float(
            bool(error_indices) and continued_after_error
        ),
        "tool_error_recovered_successfully": float(
            successfully_recovered
        ),
        "stopped_after_tool_error": float(
            bool(error_indices) and not continued_after_error
        ),
        "malformed_tool_call": float(bool(malformed_indices)),
        "malformed_tool_call_count": float(len(malformed_indices)),
        "tool_failure": float(
            bool(error_indices or malformed_indices)
        ),
        "list_tables_called": float(bool(list_calls)),
        "list_tables_call_count": count("list_tables"),
        "list_tables_succeeded": float(bool(list_successes)),
        "get_table_schema_called": float(bool(schema_calls)),
        "get_table_schema_call_count": count(
            "get_table_schema"
        ),
        "get_table_schema_succeeded": float(
            bool(schema_successes)
        ),
        "schema_after_list_tables": float(schema_after_list),
        "execute_sql_called": float(bool(execute_calls)),
        "execute_sql_call_count": count("execute_sql"),
        "execute_sql_succeeded": float(
            bool(execute_successes)
        ),
        "execute_sql_after_schema": float(
            execute_after_schema
        ),
        "execute_sql_after_successful_schema": float(
            execute_after_successful_schema
        ),
        "ordered_tool_chain": float(ordered_tool_chain),
        "tool_protocol_valid": float(protocol_valid),
    }
