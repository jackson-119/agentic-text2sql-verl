"""SQLite backend regression tests; not full AgentLoop integration."""
import hashlib
import sqlite3
import tempfile
import unittest
from pathlib import Path

from agentic_text2sql.agent.spider_db_tool import SpiderDatabaseTool


class SQLiteBackendTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "fixture.sqlite"
        with sqlite3.connect(self.db) as connection:
            connection.executescript("""
                CREATE TABLE departments(
                    id INTEGER PRIMARY KEY,
                    name TEXT
                );
                CREATE TABLE employees(
                    id INTEGER PRIMARY KEY,
                    name TEXT,
                    department_id INTEGER,
                    FOREIGN KEY(department_id) REFERENCES departments(id)
                );
                INSERT INTO departments VALUES (1, 'research');
                INSERT INTO employees VALUES
                    (1, 'Alice', 1),
                    (2, 'Bob', 1),
                    (3, 'Carol', 1);
            """)
        self.before = hashlib.sha256(self.db.read_bytes()).hexdigest()

    def tearDown(self):
        after = hashlib.sha256(self.db.read_bytes()).hexdigest()
        try:
            self.assertEqual(self.before, after, "工具修改了数据库文件")
        finally:
            self.tmp.cleanup()

    def run_tool(self, action, parameters, max_rows=50):
        # 仅测试实际同步SQLite后端。
        # 不绕过测试去宣称BaseTool初始化或异步生命周期已经通过。
        tool = SpiderDatabaseTool.__new__(SpiderDatabaseTool)
        tool.action = action
        tool.max_rows = max_rows
        tool.sample_rows = 2
        tool.timeout_seconds = 3.0
        return tool._execute_sync(self.db, parameters)

    def test_list_tables(self):
        result, ok = self.run_tool("list_tables", {})
        self.assertTrue(ok)
        self.assertEqual(result["tables"], ["departments", "employees"])
        self.assertEqual(result["table_count"], 2)

    def test_schema_and_foreign_key(self):
        result, ok = self.run_tool(
            "get_table_schema", {"table_name": "employees"}
        )
        self.assertTrue(ok)
        self.assertEqual(
            [column["name"] for column in result["columns"]],
            ["id", "name", "department_id"],
        )
        self.assertEqual(result["foreign_keys"], [{
            "from_column": "department_id",
            "to_table": "departments",
            "to_column": "id",
        }])
        self.assertEqual(len(result["sample_rows"]), 2)

    def test_execute_select(self):
        result, ok = self.run_tool(
            "execute_sql", {"sql": "SELECT name FROM employees ORDER BY id"}
        )
        self.assertTrue(ok)
        self.assertEqual(result["rows"], [["Alice"], ["Bob"], ["Carol"]])

    def test_row_truncation(self):
        result, ok = self.run_tool(
            "execute_sql",
            {"sql": "SELECT name FROM employees ORDER BY id"},
            max_rows=2,
        )
        self.assertTrue(ok)
        self.assertTrue(result["truncated"])
        self.assertEqual(result["returned_row_count"], 2)

    def test_with_query(self):
        result, ok = self.run_tool(
            "execute_sql",
            {"sql": "WITH x AS (SELECT 7 AS n) SELECT n FROM x"},
        )
        self.assertTrue(ok)
        self.assertEqual(result["rows"], [[7]])

    def test_write_rejected(self):
        result, ok = self.run_tool(
            "execute_sql", {"sql": "DELETE FROM employees"}
        )
        self.assertFalse(ok)
        self.assertFalse(result["ok"])

    def test_multiple_statements_rejected(self):
        result, ok = self.run_tool(
            "execute_sql", {"sql": "SELECT 1; SELECT 2"}
        )
        self.assertFalse(ok)
        self.assertFalse(result["ok"])

    def test_unknown_table(self):
        result, ok = self.run_tool(
            "get_table_schema", {"table_name": "missing"}
        )
        self.assertFalse(ok)
        self.assertIn("Unknown table", result["error"])

    def test_invalid_sql(self):
        result, ok = self.run_tool(
            "execute_sql", {"sql": "SELECT missing_column FROM employees"}
        )
        self.assertFalse(ok)
        self.assertIn("error_type", result)

    def test_identifier_quoting(self):
        self.assertEqual(
            SpiderDatabaseTool._quote_identifier('a"b'),
            '"a""b"',
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
