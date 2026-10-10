"""Tests for historical SQL extraction and execution semantics."""
import hashlib
import sqlite3
import tempfile
import unittest
from pathlib import Path

from agentic_text2sql.evaluation.sql_execution import (
    evaluate_answer,
    extract_final_sql,
    _normalize_value,
)


class SQLExecutionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "fixture.sqlite"
        with sqlite3.connect(self.db) as connection:
            connection.executescript("""
                CREATE TABLE t(x INTEGER);
                INSERT INTO t VALUES (1), (1), (2);
            """)
        self.before = hashlib.sha256(self.db.read_bytes()).hexdigest()

    def tearDown(self):
        try:
            self.assertEqual(
                self.before,
                hashlib.sha256(self.db.read_bytes()).hexdigest(),
            )
        finally:
            self.tmp.cleanup()

    def check(self, prediction, gold):
        return evaluate_answer(self.db, prediction, gold)

    def test_strict_final_sql(self):
        self.assertEqual(
            extract_final_sql("FINAL_SQL: SELECT 1"),
            ("SELECT 1", True, "final_sql"),
        )

    def test_fenced_sql(self):
        self.assertEqual(
            extract_final_sql("```sql\nSELECT 1\n```"),
            ("SELECT 1", False, "sql_fence"),
        )

    def test_direct_sql(self):
        self.assertEqual(
            extract_final_sql("SELECT 1"),
            ("SELECT 1", False, "direct_sql"),
        )

    def test_no_answer(self):
        result = self.check("I cannot answer.", "SELECT x FROM t")
        self.assertEqual(result["answer_found"], 0.0)
        self.assertEqual(result["gold_executable"], 1.0)
        self.assertEqual(result["execution_correct"], 0.0)

    def test_correct_answer(self):
        result = self.check("FINAL_SQL: SELECT x FROM t", "SELECT x FROM t")
        self.assertEqual(result["execution_correct"], 1.0)
        self.assertEqual(result["format_compliance"], 1.0)

    def test_unordered_rows(self):
        result = self.check(
            "SELECT x FROM t ORDER BY x DESC", "SELECT x FROM t"
        )
        self.assertEqual(result["execution_correct"], 1.0)

    def test_ordered_rows(self):
        result = self.check(
            "SELECT x FROM t ORDER BY x DESC",
            "SELECT x FROM t ORDER BY x ASC",
        )
        self.assertEqual(result["execution_correct"], 0.0)

    def test_duplicate_multiplicity(self):
        result = self.check("SELECT DISTINCT x FROM t", "SELECT x FROM t")
        self.assertEqual(result["execution_correct"], 0.0)

    def test_column_alias_ignored(self):
        result = self.check("SELECT x AS renamed FROM t", "SELECT x FROM t")
        self.assertEqual(result["execution_correct"], 1.0)

    def test_column_count(self):
        result = self.check("SELECT x, x FROM t", "SELECT x FROM t")
        self.assertEqual(result["execution_correct"], 0.0)

    def test_integer_and_float_distinct(self):
        result = self.check("SELECT 1.0", "SELECT 1")
        self.assertEqual(result["execution_correct"], 0.0)

    def test_float_rounding(self):
        self.assertEqual(
            _normalize_value(1.000000001),
            _normalize_value(1.0),
        )

    def test_invalid_prediction(self):
        result = self.check("SELECT missing FROM t", "SELECT x FROM t")
        self.assertEqual(result["gold_executable"], 1.0)
        self.assertEqual(result["sql_executable"], 0.0)
        self.assertEqual(result["execution_correct"], 0.0)
        self.assertIsNotNone(result["evaluation_error_type"])

    def test_invalid_gold(self):
        result = self.check("SELECT x FROM t", "SELECT missing FROM t")
        self.assertEqual(result["gold_executable"], 0.0)
        self.assertEqual(result["sql_executable"], 0.0)
        self.assertIsNotNone(result["evaluation_error_type"])

    def test_write_rejected(self):
        result = self.check("FINAL_SQL: DELETE FROM t", "SELECT x FROM t")
        self.assertEqual(result["execution_correct"], 0.0)
        self.assertEqual(result["sql_executable"], 0.0)

    def test_multiple_statements_rejected(self):
        result = self.check("SELECT 1; SELECT 2", "SELECT 1")
        self.assertEqual(result["sql_executable"], 0.0)

    def test_empty_results_equal(self):
        result = self.check(
            "SELECT x FROM t WHERE x > 100",
            "SELECT x FROM t WHERE x < 0",
        )
        self.assertEqual(result["execution_correct"], 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
