"""Prepare an isolated baseline config. Never starts Ray or training."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


BASELINE_ENV = {
    "VERL_SC_ENABLED": "0",
    "VERL_SC_DIAG_DIR": "",
    "VERL_SC_Q_DIAG_DIR": "",
    "TURN_CREDIT_TRACE_DIR": "",
}

def adapt_ray(config, env):
    result = copy.deepcopy(config)
    kwargs = result.setdefault("ray_kwargs", {})
    active = kwargs.setdefault("ray_init", {})
    for key, value in result.get("ray_init", {}).items():
        if key in active and active[key] != value:
            raise ValueError(f"Conflicting Ray setting: {key}")
        active[key] = value
    if active.get("address") not in (None, "local"):
        raise ValueError("Existing Ray cluster is outside this baseline scope")
    active["address"] = "local"
    runtime = active.setdefault("runtime_env", {})
    supplied = runtime.setdefault("env_vars", {})
    protected = dict(env)
    protected.update(BASELINE_ENV)
    for key, value in protected.items():
        supplied[key] = value
    return result

def require_baseline(config):
    algorithm = config["algorithm"]
    trainer = config["trainer"]
    if algorithm["adv_estimator"] != "grpo":
        raise ValueError("Expected GRPO")
    correction = algorithm.get("rollout_correction", {})
    if correction.get("rollout_is") is not None:
        raise ValueError("Baseline must not enable rollout IS")
    if correction.get("rollout_rs") is not None or correction.get("bypass_mode", False):
        raise ValueError("Unsupported rollout correction mode")
    if algorithm.get("turn_aware_credit", {}).get("enabled", False):
        raise ValueError("Baseline must not enable turn-aware credit")
    if trainer["nnodes"] != 1 or trainer["n_gpus_per_node"] != 1:
        raise ValueError("Only single-node, single-GPU baseline supported")
    if trainer.get("use_v1", False):
        raise ValueError("This adapter targets the inspected legacy trainer")
    if trainer.get("resume_mode") != "disable":
        raise ValueError("Fresh-run adapter does not support resume")
    if config.get("transfer_queue", {}).get("enable", False):
        raise ValueError("Transfer queue is outside the inspected baseline")

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def prepare(repo, verl_root, run_dir, ray_temp):
    repo, verl_root, run_dir, ray_temp = [
        Path(p).resolve() for p in (repo, verl_root, run_dir, ray_temp)
    ]
    if run_dir.exists() or ray_temp.exists():
        raise ValueError("Run and Ray directories must not already exist")
    if len(str(ray_temp)) > 55:
        raise ValueError("Choose a shorter Ray temporary path")

    required = {
        "TEXT2SQL_MODEL_PATH": "dir",
        "TEXT2SQL_CRITIC_MODEL_PATH": "dir",
        "TEXT2SQL_TRAIN_PARQUET": "file",
        "TEXT2SQL_DEV_PARQUET": "file",
        "SPIDER_DATABASE_ROOT": "dir",
    }
    env = {}
    inputs = [repo, verl_root]
    for key, kind in required.items():
        value = os.environ.get(key)
        if not value:
            raise ValueError(f"Missing environment variable: {key}")
        path = Path(value).resolve()
        if not (path.is_dir() if kind == "dir" else path.is_file()):
            raise FileNotFoundError(f"{key}: {path}")
        env[key] = str(path)
        inputs.append(path)

    for output in (run_dir, ray_temp):
        for source in inputs:
            if output == source or output in source.parents or source in output.parents:
                raise ValueError("Output overlaps a repository or input path")
    if run_dir == ray_temp or run_dir in ray_temp.parents or ray_temp in run_dir.parents:
        raise ValueError("Run and Ray directories must be separate")

    entry = verl_root / "verl/trainer/main_ppo.py"
    expected = "d5085f4307190fb3645182eed831dcc1229983635acc1de70f815a7d2fee0ec9"
    if digest(entry) != expected:
        raise ValueError("Training entry differs from the inspected source")

    template = repo / "configs/grpo/baseline_smoke5.yaml"
    tools_template = repo / "configs/evaluation/spider_tools.yaml"
    if digest(template) != "3c17e3d5e936268255b99dcca6ad0053acb0d3de3bc83201cba8e092dc6b6b9f":
        raise ValueError("Baseline template differs from inspected version")

    env.update({
        "TEXT2SQL_REPO_ROOT": str(repo),
        "TEXT2SQL_RUN_DIR": str(run_dir),
        "TEXT2SQL_RAY_TEMP_DIR": str(ray_temp),
        "TEXT2SQL_TOOL_CONFIG": str(run_dir / "tools.yaml"),
        "PYTHONPATH": os.pathsep.join((str(repo / "src"), str(verl_root))),
    })

    # Temporary driver environment, restored even if preparation fails.
    original = {key: os.environ.get(key) for key in env}
    try:
        os.environ.update(env)
        tools = OmegaConf.to_container(
            OmegaConf.load(tools_template), resolve=True, throw_on_missing=True
        )
        with initialize_config_dir(
            config_dir=str(template.parent), version_base=None
        ):
            composed = compose(config_name=template.stem)
        plain = OmegaConf.to_container(
            composed, resolve=True, throw_on_missing=False
        )
    finally:
        for key, value in original.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    require_baseline(plain)
    reward_path = str(repo / "src/agentic_text2sql/rewards/scoring.py")
    for custom in (plain["custom_reward_function"],
                   plain["reward"]["custom_reward_function"]):
        if custom["path"] != reward_path or custom["name"] != "compute_score":
            raise ValueError("Unexpected reward entry")

    result = adapt_ray(plain, env)
    if result["algorithm"] != plain["algorithm"]:
        raise AssertionError("Algorithm configuration changed")
    result["hydra"] = {
        "run": {"dir": str(run_dir / "hydra")},
        "job": {"chdir": False},
    }

    # Atomically claim a new directory; failures leave evidence, never overwrite.
    run_dir.mkdir(parents=True, exist_ok=False)
    OmegaConf.save(OmegaConf.create(tools), run_dir / "tools.yaml")
    OmegaConf.save(OmegaConf.create(result), run_dir / "train.yaml")
    summary = {
        "status": "prepared_not_launched",
        "source_template_sha256": digest(template),
        "training_entry_sha256": digest(entry),
        "prepared_config_sha256": digest(run_dir / "train.yaml"),
        "effective_ray_temp_dir": result["ray_kwargs"]["ray_init"]["_temp_dir"],
        "driver_environment": dict(env, **BASELINE_ENV),
        "algorithm_settings_unchanged": True,
        "unused_missing_fields_preserved": True,
        "hydra_compose_passed": True,
        "worker_environment_verified": False,
        "ray_started": False,
        "training_started": False,
    }
    (run_dir / "preparation.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--verl-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--ray-temp-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(
        prepare(args.repo, args.verl_root, args.run_dir, args.ray_temp_dir),
        ensure_ascii=False, indent=2,
    ))

if __name__ == "__main__":
    main()
