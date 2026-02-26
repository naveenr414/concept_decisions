"""
get_runtimes.py

Times each concept selection method and records wall-clock duration.
Q-estimates must be pre-computed via train_prerequisites.py.
"""

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["TORCH_NUM_THREADS"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
os.environ["GRB_LICENSE_FILE"] = os.environ.get("GRB_LICENSE_FILE", "/usr0/home/naveenr/gurobi.lic")
os.environ["MKL_THREADING_LAYER"] = "GNU"

import time
import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

from stable_baselines3 import PPO

from concept_abstraction.training import train_ppo
from concept_abstraction.selection import (
    variance_selection, greedy_selection, drs, drs_log,
)
from concept_abstraction.concept_bank import get_concepts
from concept_abstraction.environments import get_environment
from concept_abstraction.utils import (
    get_save_path, delete_duplicate_results, get_results_matching_parameters,
)

import sys
import argparse
import numpy as np
import random
import pickle
import secrets
import ujson as json
from pathlib import Path 

is_jupyter = "ipykernel" in sys.modules
is_main = __name__ == "__main__"

# ── Arguments ─────────────────────────────────────────────────────────────────
if is_main:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",                  type=int, default=42)
    parser.add_argument("--environment_string",    type=str, default="mini_grid")
    parser.add_argument("--num_concepts_selected", type=int, default=0)
    parser.add_argument("--gold_timesteps",        type=int, default=0)
    parser.add_argument("--out_folder",            type=str, default="timing")
    parser.add_argument("--method",                type=str, default="drs",
        choices=["variance", "greedy", "drs", "drs_log"])
    args = parser.parse_args()

    seed                  = args.seed
    environment_string    = args.environment_string
    num_concepts_selected = args.num_concepts_selected
    gold_timesteps        = args.gold_timesteps
    method                = args.method
    out_folder            = args.out_folder

# ── Logging ───────────────────────────────────────────────────────────────────
if is_main:
    REPO_ROOT = Path(__file__).parent.parent
    results = {}
    results["parameters"] = {
        "seed":                  seed,
        "environment_string":    environment_string,
        "num_concepts_selected": num_concepts_selected,
        "method":                method,
        "experiment":            "runtimes",
    }
    print("Parameters {}".format(results["parameters"]))
    np.random.seed(seed)
    random.seed(seed)

# ── Environment and ground-truth policy ───────────────────────────────────────
if is_main:
    concept_list, processed_concepts = get_concepts(environment_string)
    num_concepts_selected = min(num_concepts_selected, len(concept_list))
    ground_truth_env, ground_truth_gym_env = get_environment(
        environment_string, concept_list=None, seed=seed)

    model_name = REPO_ROOT / "results/models/env={}_training={}_seed={}.zip".format(
        environment_string, gold_timesteps, seed)

    if os.path.exists(model_name):
        groundtruth_model = PPO.load(model_name)
    else:
        policy = "MlpPolicy" if environment_string == "glucose" else "CnnPolicy"
        groundtruth_model = train_ppo(
            ground_truth_env, environment_string,
            seed=seed, total_timesteps=gold_timesteps, policy=policy)
        groundtruth_model.save(model_name)

# drs_log requires accuracy estimates from a prior imperfect run
if is_main and method == "drs_log":
    params = get_results_matching_parameters(
        "intervention", "",
        dict(environment_string=environment_string,
             num_concepts_selected=num_concepts_selected,
             method="drs_log",
             intervention_prob=0.5))
    acc_list = np.array(params[0]["concept_accuracy"])

# ── Q-value estimation (must be pre-computed via train_prerequisites.py) ──────
if is_main:
    q_name = REPO_ROOT / "results/q_estimates/env={}_training={}_seed={}.pkl".format(
        environment_string, gold_timesteps, seed)
    if not os.path.exists(q_name):
        raise FileNotFoundError(
            f"Q-estimates not found: {q_name}\n"
            "Run train_prerequisites.py first.")
    q_estimates = pickle.load(open(q_name, "rb"))

# ── Timed concept selection ───────────────────────────────────────────────────
if is_main:
    start = time.time()

if is_main and method == "variance":
    _, idx = variance_selection(concept_list, num_concepts_selected, q_estimates)

if is_main and method == "greedy":
    _, idx = greedy_selection(concept_list, num_concepts_selected, q_estimates)

if is_main and method == "drs":
    _, idx = drs(
        ground_truth_gym_env, concept_list, num_concepts_selected,
        groundtruth_model, q_estimates, coverage_ratio=0.75)

if is_main and method == "drs_log":
    _, idx = drs_log(
        ground_truth_gym_env, concept_list, num_concepts_selected,
        groundtruth_model, q_estimates, acc_list)

if is_main:
    results["time_taken"] = time.time() - start

# ── Save ──────────────────────────────────────────────────────────────────────
if is_main:
    save_name = secrets.token_hex(4)
    save_path = get_save_path(out_folder, save_name)
    delete_duplicate_results(out_folder, "", results)
    json.dump(results, open(REPO_ROOT / "results/" / save_path, "w"))
    ground_truth_env.close()
    ground_truth_gym_env.close()