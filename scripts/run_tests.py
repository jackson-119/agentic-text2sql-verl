"""Run curated test groups in isolated subprocesses."""
import argparse
import os
import subprocess
import sys
from pathlib import Path

GROUPS = {
    "core": [
        "test_correctness_gated.py",
        "test_sql_execution.py",
        "test_scoring.py",
        "test_sft_masking.py",
    ],
    "algorithms": [
        "test_score_centering.py",
        "test_sc_auxiliary.py",
        "test_importance_sampling.py",
    ],
    "tools": [
        "test_sqlite_backend.py",
    ],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--group",
        choices=[*GROUPS, "all"],
        default="core",
        help="Default: core. tools/all require the verl tool dependencies.",
    )
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    groups = list(GROUPS) if args.group == "all" else [args.group]

    selected = [(group, name) for group in groups for name in GROUPS[group]]
    for _, name in selected:
        if not (repo / "tests" / name).is_file():
            parser.error(f"Required test file is missing: tests/{name}")

    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = ""
    paths = [str(repo / "src")]
    if env.get("PYTHONPATH"):
        paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(paths)

    failures = []
    for group, name in selected:
        print(f"\n===== {group}: {name} =====", flush=True)
        completed = subprocess.run(
            [
                sys.executable, "-m", "unittest", "discover",
                "-s", str(repo / "tests"), "-p", name, "-v",
            ],
            cwd=repo,
            env=env,
        )
        if completed.returncode:
            failures.append((name, completed.returncode))

    print("\n===== Test runner summary =====", flush=True)
    print(f"Selected test modules: {len(selected)}", flush=True)
    print(f"Failed modules: {len(failures)}", flush=True)
    for name, code in failures:
        print(f"FAIL {name}: exit={code}", flush=True)
    print(
        "Scope: curated CPU tests; not online distributed training verification.",
        flush=True,
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
