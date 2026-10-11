"""Fresh-run GRPO launcher. Dry-run is the default."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone
from omegaconf import OmegaConf
from .prepare_grpo import BASELINE_ENV, require_baseline

ENTRY_SHA = "d5085f4307190fb3645182eed831dcc1229983635acc1de70f815a7d2fee0ec9"

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def plan(path):
    path = Path(path).resolve(strict=True)
    run = path.parent
    record = json.loads(path.read_text())
    if record.get("status") != "prepared_not_launched":
        raise ValueError("Not a fresh preparation")
    env = record["driver_environment"]
    if not isinstance(env, dict) or not all(
        isinstance(k, str) and isinstance(v, str)
        and k and "=" not in k and "\0" not in k + v
        for k, v in env.items()
    ):
        raise ValueError("Invalid environment")
    for key, value in BASELINE_ENV.items():
        if env.get(key) != value:
            raise ValueError("Baseline flag mismatch: " + key)
    repo = Path(env["TEXT2SQL_REPO_ROOT"]).resolve(strict=True)
    paths = env["PYTHONPATH"].split(os.pathsep)
    if len(paths) != 2 or Path(paths[0]).resolve() != repo / "src":
        raise ValueError("Unexpected PYTHONPATH")
    verl = Path(paths[1]).resolve(strict=True)
    if Path(env["TEXT2SQL_RUN_DIR"]).resolve() != run:
        raise ValueError("Run path mismatch")
    for key, kind in {
        "TEXT2SQL_MODEL_PATH": "dir",
        "TEXT2SQL_CRITIC_MODEL_PATH": "dir",
        "TEXT2SQL_TRAIN_PARQUET": "file",
        "TEXT2SQL_DEV_PARQUET": "file",
        "SPIDER_DATABASE_ROOT": "dir",
    }.items():
        p = Path(env[key])
        valid = p.is_file() if kind == "file" else p.is_dir()
        if not p.is_absolute() or not valid:
            raise ValueError("Invalid input: " + key)
    config = run / "train.yaml"
    tools = run / "tools.yaml"
    entry = verl / "verl/trainer/main_ppo.py"
    template = repo / "configs/grpo/baseline_smoke5.yaml"
    for p, expected in (
        (config, record["prepared_config_sha256"]),
        (template, record["source_template_sha256"]),
        (entry, ENTRY_SHA),
    ):
        if sha(p) != expected:
            raise ValueError("Hash mismatch: " + str(p))
    if record["training_entry_sha256"] != ENTRY_SHA:
        raise ValueError("Unsupported training entry")
    if Path(env["TEXT2SQL_TOOL_CONFIG"]).resolve() != tools or not tools.is_file():
        raise ValueError("Invalid tool config")
    cfg = OmegaConf.to_container(OmegaConf.load(config), resolve=False)
    require_baseline(cfg)
    ray = cfg["ray_kwargs"]["ray_init"]
    temp = Path(env["TEXT2SQL_RAY_TEMP_DIR"])
    if (
        not temp.is_absolute() or temp.exists() or temp.is_symlink()
        or ray.get("address") != "local"
        or ray["_temp_dir"] != str(temp)
        or record["effective_ray_temp_dir"] != str(temp)
    ):
        raise ValueError("Ray target is not fresh and local")
    worker = ray["runtime_env"]["env_vars"]
    if any(worker.get(k) != v for k, v in env.items()):
        raise ValueError("Driver/worker environment mismatch")
    for reward in (cfg["custom_reward_function"], cfg["reward"]["custom_reward_function"]):
        if (
            reward["name"] != "compute_score"
            or reward["path"] != str(repo / "src/agentic_text2sql/rewards/scoring.py")
        ):
            raise ValueError("Unexpected reward entry")
    for name in ("launch_claim.json", "training.log", "launch_result.json"):
        p = run / name
        if p.exists() or p.is_symlink():
            raise FileExistsError(str(p))
    return {
        "command": [sys.executable, "-m", "verl.trainer.main_ppo",
                    "--config-path", str(run), "--config-name", "train"],
        "cwd": str(verl), "run_directory": str(run),
        "environment": env,
        "input_hashes": {str(p): sha(p) for p in (path, config, tools, entry, template)},
    }

def execute_plan(item):
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible or "," in visible or visible == "-1":
        raise ValueError("Select exactly one GPU with CUDA_VISIBLE_DEVICES")
    run = Path(item["run_directory"])
    for name in ("training.log", "launch_result.json"):
        p = run / name
        if p.exists() or p.is_symlink():
            raise FileExistsError(str(p))
    with (run / "launch_claim.json").open("x") as f:
        json.dump(item, f, indent=2)
    result = {"returncode": None, "subprocess_invoked": False}
    try:
        for p, expected in item["input_hashes"].items():
            if sha(p) != expected:
                raise ValueError("Input changed: " + p)
        env = os.environ.copy()
        env.update(item["environment"])
        env.pop("RAY_ADDRESS", None)
        env.pop("RAY_JOB_CONFIG_JSON_ENV_VAR", None)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["HYDRA_FULL_ERROR"] = "1"
        with (run / "training.log").open("x") as f:
            result["subprocess_invoked"] = True
            child = subprocess.run(
                item["command"], cwd=item["cwd"], env=env,
                stdout=f, stderr=subprocess.STDOUT, check=False,
            )
        result["returncode"] = child.returncode
    except BaseException as error:
        result["launcher_error"] = type(error).__name__ + ": " + str(error)
        raise
    finally:
        result["finished_at"] = datetime.now(timezone.utc).isoformat()
        with (run / "launch_result.json").open("x") as f:
            json.dump(result, f, indent=2)
    return result

def launch(path, execute=False):
    item = plan(path)
    return execute_plan(item) if execute else {"mode": "dry_run", **item}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    result = launch(args.preparation, args.execute)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.execute and result["returncode"] != 0:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
