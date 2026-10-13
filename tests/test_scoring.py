"""Synthetic trace tests; not historical rollout equivalence."""
import hashlib
import json
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agentic_text2sql.rewards.scoring import compute_score


def event(name, arguments, ok=True):
    call = json.dumps({"name": name, "arguments": arguments})
    response = json.dumps({"ok": True} if ok else {"ok": False, "error": "test"})
    return f"<tool_call>{call}</tool_call>\n<tool_response>{response}</tool_response>\n"


def prefix():
    return (
        event("list_tables", {})
        + event("get_table_schema", {"table_name": "t"})
    )


def success():
    return prefix() + event("execute_sql", {"sql": "SELECT x FROM t"})


class ScoringTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        (root / "fixture").mkdir()
        self.db = root / "fixture/fixture.sqlite"
        with sqlite3.connect(self.db) as connection:
            connection.executescript(
                "CREATE TABLE t(x INTEGER); INSERT INTO t VALUES (1), (2);"
            )
        self.before = hashlib.sha256(self.db.read_bytes()).hexdigest()
        self.env = patch.dict(os.environ, {"SPIDER_DATABASE_ROOT": str(root)})
        self.env.start()
        self.addCleanup(self.tmp.cleanup)
        self.addCleanup(self.env.stop)

    def tearDown(self):
        self.assertEqual(self.before, hashlib.sha256(self.db.read_bytes()).hexdigest())

    def score(self, trace, gold=None, extra=None):
        if gold is None:
            gold = {"db_id": "fixture", "query": "SELECT x FROM t"}
        result = compute_score("spider", trace, gold, extra)
        self.assertTrue(all(isinstance(v, float) for v in result.values()))
        return result

    def test_reward_branches(self):
        cases = [
            ("full", success() + "FINAL_SQL: SELECT x FROM t", 1.0, 3.0),
            ("direct", "SELECT x FROM t", 0.9, 2.0),
            ("wrong", success() + "FINAL_SQL: SELECT 99", -0.1, 0.0),
            ("missing", success(), -0.1, 0.0),
            (
                "duplicate_execute",
                success()
                + event("execute_sql", {"sql": "SELECT x FROM t;"})
                + "FINAL_SQL: SELECT x FROM t",
                0.991, 3.0,
            ),
            (
                "duplicate_schema",
                prefix()
                + event("get_table_schema", {"table_name": " T "})
                + event("execute_sql", {"sql": "SELECT x FROM t"})
                + "FINAL_SQL: SELECT x FROM t",
                0.995, 3.0,
            ),
            (
                "missing_response",
                '<tool_call>{"name":"list_tables","arguments":{}}</tool_call>\n'
                + "FINAL_SQL: SELECT x FROM t",
                0.8, 1.0,
            ),
            (
                "stopped_after_error",
                success()
                + event("execute_sql", {"sql": "SELECT missing FROM t"}, ok=False)
                + "FINAL_SQL: SELECT x FROM t",
                0.9, 2.0,
            ),
            (
                "recovered",
                prefix()
                + event("execute_sql", {"sql": "SELECT missing FROM t"}, ok=False)
                + event("execute_sql", {"sql": "SELECT x FROM t"})
                + "FINAL_SQL: SELECT x FROM t",
                1.0, 3.0,
            ),
        ]
        for name, trace, expected, branch in cases:
            with self.subTest(name=name):
                result = self.score(trace)
                self.assertAlmostEqual(result["score"], expected)
                self.assertEqual(result["v10_branch_code"], branch)
                if result["execution_correct"] == 0:
                    self.assertEqual(result["process_reward"], 0.0)
                    self.assertEqual(result["total_penalty"], 0.0)

    def test_ground_truth_json(self):
        result = self.score(
            "SELECT x FROM t",
            json.dumps({"db_id": "fixture", "query": "SELECT x FROM t"}),
        )
        self.assertEqual(result["execution_correct"], 1.0)

    def test_extra_info_fallback(self):
        result = self.score(
            "SELECT x FROM t", {},
            {"db_id": "fixture", "gold_sql": "SELECT x FROM t"},
        )
        self.assertEqual(result["execution_correct"], 1.0)

    def test_invalid_gold_diagnostic(self):
        result = self.score(
            "SELECT x FROM t",
            {"db_id": "fixture", "query": "SELECT missing FROM t"},
        )
        self.assertEqual(result["evaluation_error"], 1.0)
        self.assertEqual(result["score"], -0.1)

    def test_unsafe_database_id(self):
        with self.assertRaises(ValueError):
            self.score("SELECT 1", {"db_id": "../fixture", "query": "SELECT 1"})

    def test_missing_database_configuration(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(RuntimeError):
                self.score("SELECT x FROM t")


if __name__ == "__main__":
    unittest.main(verbosity=2)
