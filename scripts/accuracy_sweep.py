"""
accuracy_sweep.py

Runs DRS with synthetically injected concept accuracy rather than a trained
concept predictor. Produces Figure 3 (accuracy × num_concepts heatmaps)
and Figure 5 (training curves) by isolating each variable independently.

Distinct from run_comparison.py in two ways:
  - Takes --concept_accuracy as a direct float (no CNN predictor trained)
  - Only runs DRS; not a full method comparison
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

import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

from stable_baselines3 import PPO

from concept_abstraction.training import train_ppo, evaluate_model
from concept_abstraction.selection import drs
from concept_abstraction.concept_bank import get_concepts
from concept_abstraction.env_utils import estimate_q_values
from concept_abstraction.environments import get_environment
from concept_abstraction.utils import get_save_path, delete_duplicate_results

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
    parser.add_argument("--seed",                  type=int,   default=42)
    parser.add_argument("--environment_string",    type=str,   default="mini_grid")
    parser.add_argument("--training_timesteps",    type=int,   default=10000)
    parser.add_argument("--gold_timesteps",        type=int,   default=10000)
    parser.add_argument("--num_concepts_selected", type=int,   default=0)
    parser.add_argument("--concept_accuracy",      type=float, default=1.0,
        help="Synthetic per-concept accuracy injected uniformly (Figure 3 rows).")
    parser.add_argument("--out_folder",            type=str,   default="imperfect")
    args = parser.parse_args()

    seed                  = args.seed
    environment_string    = args.environment_string
    gold_timesteps        = args.gold_timesteps
    training_timesteps    = args.training_timesteps
    num_concepts_selected = args.num_concepts_selected
    concept_accuracy      = args.concept_accuracy
    out_folder            = args.out_folder

# ── Logging ───────────────────────────────────────────────────────────────────
if is_main:
    results = {}
    results["parameters"] = {
        "seed":                  seed,
        "environment_string":    environment_string,
        "training_timesteps":    training_timesteps,
        "gold_timesteps":        gold_timesteps,
        "num_concepts_selected": num_concepts_selected,
        "concept_accuracy":      concept_accuracy,
    }
    print("Parameters {}".format(results["parameters"]))
    np.random.seed(seed)
    random.seed(seed)

# ── Environment and ground-truth policy ───────────────────────────────────────
if is_main:
    REPO_ROOT = Path(__file__).parent.parent
    concept_list, processed_concepts = get_concepts(environment_string)
    num_concepts_selected = min(num_concepts_selected, len(concept_list))
    ground_truth_env, ground_truth_gym_env = get_environment(
        environment_string, concept_list=None, seed=seed)

    model_name = REPO_ROOT / "results/models/env={}_training={}_seed={}.zip".format(
        environment_string, gold_timesteps, seed)

    if os.path.exists(model_name):
        custom_objects = {
            "observation_space": ground_truth_env.observation_space,
            "action_space":      ground_truth_env.action_space,
        }
        groundtruth_model = PPO.load(model_name, custom_objects=custom_objects)
    else:
        groundtruth_model = train_ppo(
            ground_truth_env, environment_string,
            seed=seed, total_timesteps=gold_timesteps, policy="CnnPolicy")
        groundtruth_model.save(model_name)

# ── Q-value estimation ────────────────────────────────────────────────────────
if is_main:
    q_name = REPO_ROOT / "results/q_estimates/env={}_training={}_seed={}.pkl".format(
        environment_string, gold_timesteps, seed)
    if os.path.exists(q_name):
        q_estimates = pickle.load(open(q_name, "rb"))
    else:
        q_estimates = estimate_q_values(
            groundtruth_model, ground_truth_gym_env, concept_list)
        pickle.dump(q_estimates, open(q_name, "wb"))

# ── DRS with synthetic accuracy ───────────────────────────────────────────────
if is_main:
    _, idx = drs(
        ground_truth_gym_env, concept_list, num_concepts_selected,
        groundtruth_model, q_estimates, coverage_ratio=0.75)
    subset_concept = [concept_list[i] for i in idx]

    env, eval_env = get_environment(
        environment_string, subset_concept, seed,
        processed_concepts=processed_concepts,
        concept_idx=idx,
        concept_accuracy=concept_accuracy)

    model = train_ppo(
        env, environment_string, seed=seed, policy="MlpPolicy",
        total_timesteps=training_timesteps,
        custom_name="{}_accuracy_sweep_{}_{}_{}".format(
            environment_string, concept_accuracy, num_concepts_selected, seed))

    reward = evaluate_model(environment_string, eval_env, model, seed)
    results["drs"] = {"reward": reward, "concepts": idx}

# ── Save ──────────────────────────────────────────────────────────────────────
if is_main:
    save_name = secrets.token_hex(4)
    save_path = get_save_path(out_folder, save_name)
    delete_duplicate_results(out_folder, "", results)
    json.dump(results, open(REPO_ROOT / "results/" / save_path, "w"))
    ground_truth_env.close()
    ground_truth_gym_env.close()
    env.close()
    eval_env.close()