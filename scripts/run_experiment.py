#!/usr/bin/env python3
"""
run_experiment.py

Single entry point for all experiments. Reads a YAML config and runs
all jobs sequentially. No tmux, no GPU topology assumptions — parallelism
is left to the user (run multiple configs in separate terminals, or use
gnu parallel / a cluster scheduler).

Usage:
    python run_experiment.py --config configs/main_perfect.yaml
    python run_experiment.py --config configs/main_imperfect.yaml
    python run_experiment.py --config configs/intervention.yaml
    python run_experiment.py --config configs/accuracy_sweep.yaml
    python run_experiment.py --config configs/ablations.yaml
    python run_experiment.py --config configs/timing.yaml
    python run_experiment.py --config configs/cub.yaml

    # Dry run — print commands without executing:
    python run_experiment.py --config configs/main_perfect.yaml --dry_run

    # Resume — skip jobs whose output already exists in results/:
    python run_experiment.py --config configs/main_perfect.yaml --resume
"""

import argparse
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

SCRIPTS_DIR = Path(__file__).parent
REPO_ROOT   = SCRIPTS_DIR.parent
NOTEBOOKS   = SCRIPTS_DIR / "notebooks"


def result_exists(out_folder: str, params: dict) -> bool:
    """Check whether a result matching these parameters already exists in results/."""
    results_dir = REPO_ROOT / "results" / out_folder
    if not results_dir.exists():
        return False
    for f in results_dir.glob("*.json"):
        try:
            data = json.loads(f.read_text())
            if all(data.get("parameters", {}).get(k) == v for k, v in params.items()):
                return True
        except Exception:
            pass
    return False


def build_jobs(config: dict) -> list[dict]:
    """Expand a config dict into a flat list of jobs."""
    experiment = config["experiment"]
    seeds      = config["seeds"]
    out_folder = config["out_folder"]
    jobs       = []

    # ── CUB ───────────────────────────────────────────────────────────────────
    if experiment == "cub":
        script = NOTEBOOKS / "supervised_learning.py"
        for seed, k in itertools.product(seeds, config["num_concepts_values"]):
            jobs.append({
                "script": script,
                "setting": None,
                "out_folder": out_folder,
                "params": {"seed": seed, "num_concepts_selected": k},
            })
        return jobs

    # ── Accuracy sweep (Figure 3) ─────────────────────────────────────────────
    if experiment == "accuracy_sweep":
        script = NOTEBOOKS / "accuracy_sweep.py"
        for env_name, env_cfg in config["environments"].items():
            for seed, accuracy, ts, k in itertools.product(
                seeds,
                config["concept_accuracies"],
                env_cfg["training_timesteps_values"],
                env_cfg["num_concepts_values"],
            ):
                jobs.append({
                    "script": script,
                    "setting": None,
                    "out_folder": out_folder,
                    "params": {
                        "seed":                  seed,
                        "environment_string":    env_name,
                        "gold_timesteps":        env_cfg["gold_timesteps"],
                        "training_timesteps":    ts,
                        "num_concepts_selected": k,
                        "concept_accuracy":      accuracy,
                    },
                })
        return jobs

    # ── Timing ────────────────────────────────────────────────────────────────
    if experiment == "timing":
        script = NOTEBOOKS / "get_runtimes.py"
        for env_name, env_cfg in config["environments"].items():
            for seed, method in itertools.product(seeds, config["methods"]):
                jobs.append({
                    "script": script,
                    "setting": None,
                    "out_folder": out_folder,
                    "params": {
                        "seed":                  seed,
                        "environment_string":    env_name,
                        "gold_timesteps":        env_cfg["gold_timesteps"],
                        "num_concepts_selected": env_cfg["num_concepts"],
                        "method":                method,
                    },
                })
        return jobs

    # ── Everything below uses run_comparison.py ───────────────────────────────
    script = NOTEBOOKS / "run_comparison.py"

    # ── Ablations ─────────────────────────────────────────────────────────────
    if experiment == "ablations":
        rho_cfg = config["rho_ablation"]
        for setting, env_name, seed, method in itertools.product(
            ("perfect", "imperfect"),
            rho_cfg["environments"],
            seeds,
            rho_cfg["methods"],
        ):
            env_cfg = rho_cfg["environments"][env_name]
            jobs.append({
                "script": script,
                "setting": setting,
                "out_folder": out_folder,
                "params": {
                    "seed":                  seed,
                    "environment_string":    env_name,
                    "gold_timesteps":        env_cfg["gold_timesteps"],
                    "training_timesteps":    env_cfg["training_timesteps"],
                    "num_concepts_selected": env_cfg["num_concepts"],
                    "method":                method,
                },
            })
        pq_cfg = config["policy_quality"]
        for seed, gold_ts, method in itertools.product(
            seeds, pq_cfg["gold_timesteps_values"], pq_cfg["methods"]
        ):
            jobs.append({
                "script": script,
                "setting": "imperfect",
                "out_folder": out_folder,
                "params": {
                    "seed":                  seed,
                    "environment_string":    pq_cfg["environment"],
                    "gold_timesteps":        gold_ts,
                    "training_timesteps":    pq_cfg["training_timesteps"],
                    "num_concepts_selected": pq_cfg["num_concepts"],
                    "method":                method,
                },
            })
        return jobs

    # ── Intervention ──────────────────────────────────────────────────────────
    if experiment == "intervention":
        for env_name, env_cfg in config["environments"].items():
            for seed, method, prob in itertools.product(
                seeds, config["methods"], config["intervention_probs"]
            ):
                jobs.append({
                    "script": script,
                    "setting": "intervention",
                    "out_folder": out_folder,
                    "params": {
                        "seed":                  seed,
                        "environment_string":    env_name,
                        "gold_timesteps":        env_cfg["gold_timesteps"],
                        "training_timesteps":    env_cfg["training_timesteps"],
                        "num_concepts_selected": env_cfg["num_concepts"],
                        "predictor_epochs":      env_cfg["predictor_epochs"],
                        "method":                method,
                        "intervention_prob":     prob,
                    },
                })
        return jobs

    # ── Main perfect / imperfect ──────────────────────────────────────────────
    setting = "perfect" if experiment == "main_perfect" else "imperfect"
    for env_name, env_cfg in config["environments"].items():
        for seed, method in itertools.product(seeds, config["methods"]):
            jobs.append({
                "script": script,
                "setting": setting,
                "out_folder": out_folder,
                "params": {
                    "seed":                  seed,
                    "environment_string":    env_name,
                    "gold_timesteps":        env_cfg["gold_timesteps"],
                    "training_timesteps":    env_cfg["training_timesteps"],
                    "num_concepts_selected": env_cfg["num_concepts"],
                    "method":                method,
                },
            })
    return jobs


def run_job(job: dict, dry_run: bool = False, resume: bool = False) -> bool:
    """Build and optionally execute the command for a single job."""
    script     = job["script"]
    setting    = job["setting"]
    out_folder = job["out_folder"]
    params     = job["params"]

    if resume and result_exists(out_folder, params):
        print(f"  [skip] {params}")
        return True

    cmd = [sys.executable, "-u", str(script)]
    if setting:
        cmd += ["--setting", setting]
    for k, v in params.items():
        cmd += [f"--{k}", str(v)]
    cmd += ["--out_folder", out_folder]

    print("  " + " ".join(cmd))
    if dry_run:
        return True

    result = subprocess.run(cmd, cwd=str(NOTEBOOKS))
    if result.returncode != 0:
        print(f"  [FAILED] exit code {result.returncode}")
        return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  required=True)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--resume",  action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        sys.exit(f"Config not found: {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    jobs  = build_jobs(config)
    total = len(jobs)
    print(f"Experiment : {config['experiment']}")
    print(f"Total jobs : {total}")
    if args.dry_run:
        print("(dry run)\n")

    failed = 0
    for i, job in enumerate(jobs, 1):
        print(f"\n[{i}/{total}]")
        if not run_job(job, dry_run=args.dry_run, resume=args.resume):
            failed += 1

    print(f"\nDone. {total - failed}/{total} jobs succeeded.")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()