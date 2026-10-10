from __future__ import annotations

import asyncio
import json
import re
import sqlite3
import time
from pathlib import Path
from typing import Any, Optional
from urllib.parse import quote
from uuid import uuid4

from verl.tools.base_tool import (
    BaseTool,
    OpenAIFunctionToolSchema,
    ToolResponse,
)
from verl.utils.rollout_trace import rollout_trace_op


class SpiderDatabaseTool(BaseTool):
    """Read-only SQLite tools for the Spider text-to-SQL dataset."""

    def __init__(
        self,
        config: dict,
        tool_schema: OpenAIFunctionToolSchema,
    ):
        super().__init__(config, tool_schema)

        self.action = str(config["action"])
        self.database_root = Path(
            config["database_root"]
        ).expanduser().resolve()

        self.max_rows = int(config.get("max_rows", 50))
        self.sample_rows = int(config.get("sample_rows", 3))
        self.timeout_seconds = float(
            config.get("timeout_seconds", 3.0)
        )

        if self.action not in {
            "list_tables",
            "get_table_schema",
            "execute_sql",
        }:
            raise ValueError(f"Unsupported action: {self.action}")

        if not self.database_root.is_dir():
            raise FileNotFoundError(
                f"Spider database root does not exist: "
                f"{self.database_root}"
            )

        self._instances: dict[str, dict[str, Any]] = {}

    async def create(
        self,
        instance_id: Optional[str] = None,
        db_id: Optional[str] = None,
        **kwargs,
    ) -> tuple[str, ToolResponse]:
        create_kwargs = kwargs.get("create_kwargs", {}) or {}

        if db_id is None:
            db_id = create_kwargs.get("db_id")

        if not isinstance(db_id, str) or not db_id:
            raise ValueError("create_kwargs.db_id is required")

        if not re.fullmatch(r"[A-Za-z0-9_]+", db_id):
            raise ValueError(f"Unsafe db_id: {db_id!r}")

        db_path = (
            self.database_root
            / db_id
            / f"{db_id}.sqlite"
        ).resolve()

        if self.database_root not in db_path.parents:
            raise ValueError("Database path escaped database root")

        if not db_path.is_file():
            raise FileNotFoundError(
                f"SQLite database does not exist: {db_path}"
            )

        if instance_id is None:
            instance_id = str(uuid4())

        self._instances[instance_id] = {
            "db_id": db_id,
            "db_path": db_path,
        }

        return instance_id, ToolResponse()

    @rollout_trace_op
    async def execute(
        self,
        instance_id: str,
        parameters: dict[str, Any],
        **kwargs,
    ) -> tuple[ToolResponse, float, dict]:
        if instance_id not in self._instances:
            return (
                ToolResponse(text=json.dumps({
                    "ok": False,
                    "error": "Unknown or released tool instance",
                })),
                0.0,
                {"sqlite_tool_success": 0.0},
            )

        state = self._instances[instance_id]

        payload, success = await asyncio.to_thread(
            self._execute_sync,
            state["db_path"],
            parameters,
        )

        return (
            ToolResponse(
                text=json.dumps(
                    payload,
                    ensure_ascii=False,
                    default=self._json_default,
                )
            ),
            0.0,
            {
                "sqlite_tool_success": float(success),
                "sqlite_tool_action": self.action,
            },
        )

    def _execute_sync(
        self,
        db_path: Path,
        parameters: dict[str, Any],
    ) -> tuple[dict[str, Any], bool]:
        connection = None

        try:
            encoded_path = quote(str(db_path), safe="/")
            uri = (
                f"file:{encoded_path}"
                f"?mode=ro&immutable=1"
            )

            connection = sqlite3.connect(
                uri,
                uri=True,
                timeout=1.0,
            )

            # Spider wta_1 contains two malformed UTF-8 player
            # names. Preserve tool availability using replacement
            # decoding without changing the original database.
            connection.text_factory = lambda raw: raw.decode(
                "utf-8",
                errors="replace",
            )

            connection.execute("PRAGMA query_only = ON")

            deadline = time.monotonic() + self.timeout_seconds

            def stop_long_query() -> int:
                return int(time.monotonic() > deadline)

            connection.set_progress_handler(
                stop_long_query,
                1000,
            )

            if self.action == "list_tables":
                payload = self._list_tables(connection)

            elif self.action == "get_table_schema":
                payload = self._get_table_schema(
                    connection,
                    parameters,
                )

            elif self.action == "execute_sql":
                payload = self._execute_sql(
                    connection,
                    parameters,
                )

            else:
                raise ValueError(
                    f"Unsupported action: {self.action}"
                )

            payload["ok"] = True
            return payload, True

        except Exception as exc:
            return {
                "ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }, False

        finally:
            if connection is not None:
                connection.close()

    def _list_tables(
        self,
        connection: sqlite3.Connection,
    ) -> dict[str, Any]:
        rows = connection.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
              AND name NOT LIKE 'sqlite_%'
            ORDER BY name
            """
        ).fetchall()

        return {
            "tables": [row[0] for row in rows],
            "table_count": len(rows),
        }

    def _get_table_schema(
        self,
        connection: sqlite3.Connection,
        parameters: dict[str, Any],
    ) -> dict[str, Any]:
        table_name = parameters.get("table_name")

        if not isinstance(table_name, str) or not table_name:
            raise ValueError("table_name must be a non-empty string")

        table_row = connection.execute(
            """
            SELECT sql
            FROM sqlite_master
            WHERE type = 'table' AND name = ?
            """,
            (table_name,),
        ).fetchone()

        if table_row is None:
            available = [
                row[0]
                for row in connection.execute(
                    """
                    SELECT name
                    FROM sqlite_master
                    WHERE type = 'table'
                      AND name NOT LIKE 'sqlite_%'
                    ORDER BY name
                    """
                ).fetchall()
            ]
            raise ValueError(
                f"Unknown table {table_name!r}. "
                f"Available tables: {available}"
            )

        quoted_table = self._quote_identifier(table_name)

        column_rows = connection.execute(
            f"PRAGMA table_info({quoted_table})"
        ).fetchall()

        foreign_key_rows = connection.execute(
            f"PRAGMA foreign_key_list({quoted_table})"
        ).fetchall()

        sample_cursor = connection.execute(
            f"SELECT * FROM {quoted_table} "
            f"LIMIT {self.sample_rows}"
        )

        sample_columns = [
            item[0] for item in sample_cursor.description
        ]
        sample_values = sample_cursor.fetchall()

        columns = [
            {
                "name": row[1],
                "type": row[2],
                "not_null": bool(row[3]),
                "default": row[4],
                "primary_key_position": row[5],
            }
            for row in column_rows
        ]

        foreign_keys = [
            {
                "from_column": row[3],
                "to_table": row[2],
                "to_column": row[4],
            }
            for row in foreign_key_rows
        ]

        sample_rows = [
            {
                name: self._json_default(value)
                for name, value in zip(
                    sample_columns,
                    row,
                    strict=True,
                )
            }
            for row in sample_values
        ]

        return {
            "table_name": table_name,
            "create_sql": table_row[0],
            "columns": columns,
            "foreign_keys": foreign_keys,
            "sample_rows": sample_rows,
        }

    def _execute_sql(
        self,
        connection: sqlite3.Connection,
        parameters: dict[str, Any],
    ) -> dict[str, Any]:
        sql = parameters.get("sql")

        if not isinstance(sql, str) or not sql.strip():
            raise ValueError("sql must be a non-empty string")

        sql = sql.strip()

        without_leading_comments = re.sub(
            r"(?is)^\s*(?:(?:--[^\n]*\n)|(?:/\*.*?\*/\s*))*",
            "",
            sql,
        )

        if not re.match(
            r"(?is)^(SELECT|WITH)\b",
            without_leading_comments,
        ):
            raise ValueError(
                "Only read-only SELECT or WITH queries are allowed"
            )

        forbidden = re.search(
            r"(?is)\b("
            r"INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|REPLACE|"
            r"ATTACH|DETACH|VACUUM|REINDEX|ANALYZE|PRAGMA"
            r")\b",
            without_leading_comments,
        )

        if forbidden:
            raise ValueError(
                f"Forbidden SQL keyword: {forbidden.group(1)}"
            )

        cursor = connection.execute(sql)

        if cursor.description is None:
            raise ValueError(
                "The SQL query did not return a result set"
            )

        columns = [item[0] for item in cursor.description]
        fetched = cursor.fetchmany(self.max_rows + 1)

        truncated = len(fetched) > self.max_rows
        rows = fetched[: self.max_rows]

        serializable_rows = [
            [
                self._json_default(value)
                for value in row
            ]
            for row in rows
        ]

        return {
            "columns": columns,
            "rows": serializable_rows,
            "returned_row_count": len(serializable_rows),
            "truncated": truncated,
        }

    @staticmethod
    def _quote_identifier(identifier: str) -> str:
        return '"' + identifier.replace('"', '""') + '"'

    @staticmethod
    def _json_default(value: Any) -> Any:
        if value is None:
            return None

        if isinstance(value, (str, int, float, bool)):
            return value

        if isinstance(value, bytes):
            return value.hex()

        return str(value)

    async def calc_reward(
        self,
        instance_id: str,
        **kwargs,
    ) -> float:
        return 0.0

    async def release(
        self,
        instance_id: str,
        **kwargs,
    ) -> None:
        self._instances.pop(instance_id, None)
