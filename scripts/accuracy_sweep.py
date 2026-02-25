"""
accuracy_sweep.py  (renamed from only_multiple.py)

Runs DRS with synthetically injected concept accuracy rather than a trained
concept predictor. Produces Figure 3 (accuracy × num_concepts heatmaps)
and Figure 5 (training curves) by isolating each variable independently.

Distinct from method_comparison_imperfect.py in two ways:
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
os.environ['MKL_THREADING_LAYER'] = "GNU"

import torch

from concept_abstraction.training import *
from concept_abstraction.selection import *
from concept_abstraction.concept_bank import *
from concept_abstraction.env_utils import *
from concept_abstraction.environments import *
from concept_abstraction.utils import *
import sys
import argparse
import numpy as np
import random
import os
import pickle
import secrets

is_jupyter = 'ipykernel' in sys.modules
is_main = __name__ == "__main__"

# ── Arguments ─────────────────────────────────────────────────────────────────
if is_main:
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed',                  type=int,   default=42)
    parser.add_argument('--environment_string',    type=str,   default="mini_grid")
    parser.add_argument('--training_timesteps',    type=int,   default=10000)
    parser.add_argument('--gold_timesteps',        type=int,   default=10000)
    parser.add_argument('--num_concepts_selected', type=int,   default=0)
    parser.add_argument('--concept_accuracy',      type=float, default=1.0,
        help="Synthetic per-concept accuracy injected uniformly (Figure 3 rows).")
    parser.add_argument('--out_folder',            type=str,   default="imperfect")
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
    results['parameters'] = {
        'seed':                  seed,
        'environment_string':    environment_string,
        'training_timesteps':    training_timesteps,
        'gold_timesteps':        gold_timesteps,
        'num_concepts_selected': num_concepts_selected,
        'concept_accuracy':      concept_accuracy,
    }
    print("Parameters {}".format(results['parameters']))
    np.random.seed(seed)
    random.seed(seed)

# ── Environment and ground-truth policy ───────────────────────────────────────
if is_main:
    concept_list, processed_concepts = get_concepts(environment_string, "human_selected_binary", seed)
    num_concepts_selected = min(num_concepts_selected, len(concept_list))
    ground_truth_env, ground_truth_gym_env = get_environment(environment_string, None, seed)

    model_name = "../../results/models/env={}_training={}_seed={}.zip".format(
        environment_string, gold_timesteps, seed)

    if os.path.exists(model_name):
        custom_objects = {
            "observation_space": ground_truth_env.observation_space,
            "action_space":      ground_truth_env.action_space,
        }
        groundtruth_model = PPO.load(model_name, custom_objects=custom_objects)
    else:
        policy = "CnnPolicy"
        groundtruth_model = train_ppo_model(
            ground_truth_env, environment_string,
            total_timesteps=gold_timesteps, policy=policy)
        groundtruth_model.save(model_name)

# ── Q-value estimation ────────────────────────────────────────────────────────
if is_main:
    q_name = "../../results/q_estimates/env={}_training={}_seed={}_selection={}_source={}.pkl".format(
        environment_string, gold_timesteps, seed, "q_value", "human_selected_binary")
    if os.path.exists(q_name):
        q_estimates = pickle.load(open(q_name, "rb"))
    else:
        q_estimates = rollout_q_estimates_td(groundtruth_model, ground_truth_gym_env, concept_list)
        pickle.dump(q_estimates, open(q_name, "wb"))

# ── DRS with synthetic accuracy ───────────────────────────────────────────────
if is_main:
    subset_concept, idx = policy_coverage_selection_lp_hybrid(
        ground_truth_gym_env, concept_list, num_concepts_selected,
        groundtruth_model, q_estimates, coverage_ratio=0.75)

    env, eval_env = get_environment(
        environment_string, subset_concept, seed,
        processed_concepts=processed_concepts,
        concept_idx=idx,
        concept_accuracy=concept_accuracy)

    model = train_ppo_model(
        env, environment_string, policy="MlpPolicy",
        total_timesteps=training_timesteps,
        custom_name="{}_accuracy_sweep_{}_{}_{}".format(
            environment_string, concept_accuracy, num_concepts_selected, seed))

    reward = evaluate_model(environment_string, eval_env, model, seed)
    results['drs'] = {'reward': reward, 'concepts': idx}

# ── Save ──────────────────────────────────────────────────────────────────────
if is_main:
    save_name = secrets.token_hex(4)
    save_path = get_save_path(out_folder, save_name)
    delete_duplicate_results(out_folder, "", results)
    json.dump(results, open('../../results/' + save_path, 'w'))
    ground_truth_env.close()
    ground_truth_gym_env.close()
    env.close()
    eval_env.close()