"""
train_prerequisites.py

Trains all shared prerequisites before running experiments:
  1. Base policy (pi*) for each environment × seed
  2. Q-value estimates derived from pi*
  3. Concept predictor CNN for each environment × seed
     (not needed for glucose — concepts are perfect there)

Must be run before run_experiment.py. All outputs are cached in results/models/
and results/q_estimates/ so subsequent runs skip already-completed work.

Usage:
    # Train everything for all environments and default seeds:
    python train_prerequisites.py

    # Train only specific environments:
    python train_prerequisites.py --envs cart_pole mini_grid

    # Train only specific seeds:
    python train_prerequisites.py --seeds 42 43 44

    # Skip concept predictor training (only base policy + Q-estimates):
    python train_prerequisites.py --skip_concept_predictors

    # Dry run:
    python train_prerequisites.py --dry_run
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

import argparse
import pickle
import secrets
import numpy as np
import random
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

from concept_abstraction.training import *
from concept_abstraction.selection import *
from concept_abstraction.concept_bank import *
from concept_abstraction.env_utils import *
from concept_abstraction.environments import *
from concept_abstraction.utils import *

# ── Default experiment matrix ─────────────────────────────────────────────────
# These match the values used throughout the paper. Override via CLI if needed.

GOLD_TIMESTEPS = {
    "cart_pole":  4_000_000,
    "mini_grid":  1_000_000,
    "pong":       15_000_000,
    "boxing":     30_000_000,
    "glucose":    4_000_000,
}

# Glucose uses perfect concept labels — no CNN predictor needed.
ENVS_NEEDING_CONCEPT_PREDICTOR = {"cart_pole", "mini_grid", "pong", "boxing"}

DEFAULT_SEEDS = [42, 43, 44, 45, 46, 47]

# ── Arguments ─────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--envs',   nargs='+', default=list(GOLD_TIMESTEPS.keys()),
    choices=list(GOLD_TIMESTEPS.keys()),
    help="Environments to train prerequisites for.")
parser.add_argument('--seeds',  nargs='+', type=int, default=DEFAULT_SEEDS)
parser.add_argument('--skip_concept_predictors', action='store_true',
    help="Skip concept predictor training (base policy + Q-estimates only).")
parser.add_argument('--force',  action='store_true',
    help="Retrain even if cached outputs already exist.")
parser.add_argument('--dry_run', action='store_true',
    help="Print what would be done without running anything.")
args = parser.parse_args()

# ── Helpers ───────────────────────────────────────────────────────────────────
def policy_for_env(environment_string: str) -> str:
    return "MlpPolicy" if any(x in environment_string for x in ("cyclic", "tree", "glucose")) \
           else "CnnPolicy"

def train_name_for_env(environment_string: str, policy: str) -> str:
    return environment_string + "_raw" if policy == "MlpPolicy" else environment_string


def train_base_policy(environment_string: str, seed: int, force: bool = False) -> None:
    """Train pi* and compute Q-value estimates. Skips if cached."""
    gold_timesteps = GOLD_TIMESTEPS[environment_string]
    model_path = "../../results/models/env={}_training={}_seed={}.zip".format(
        environment_string, gold_timesteps, seed)
    q_path = "../../results/q_estimates/env={}_training={}_seed={}_selection={}_source={}.pkl".format(
        environment_string, gold_timesteps, seed, "q_value", "human_selected_binary")

    if os.path.exists(model_path) and os.path.exists(q_path) and not force:
        print(f"  [skip] base policy already exists: {environment_string}, seed={seed}")
        return

    print(f"  Training base policy: {environment_string}, seed={seed}")
    np.random.seed(seed)
    random.seed(seed)

    concept_list, _ = get_concepts(environment_string, "human_selected_binary", seed)
    ground_truth_env, ground_truth_gym_env = get_environment(environment_string, None, seed)

    policy    = policy_for_env(environment_string)
    train_name = train_name_for_env(environment_string, policy)
    groundtruth_model = train_ppo_model(
        ground_truth_env, train_name, total_timesteps=gold_timesteps, policy=policy)
    groundtruth_model.save(model_path)

    groundtruth_reward = evaluate_model(
        environment_string, ground_truth_gym_env, groundtruth_model, seed)
    print(f"    Reward: {groundtruth_reward:.3f}")

    print(f"    Computing Q-estimates...")
    q_estimates = rollout_q_estimates_td(groundtruth_model, ground_truth_gym_env, concept_list)
    pickle.dump(q_estimates, open(q_path, "wb"))

    # Save reward to results/training/ for reference
    results = {
        'parameters': {
            'seed': seed,
            'gold_timesteps': gold_timesteps,
            'environment_string': environment_string,
            'experiment': 'training',
        },
        'ground_truth': {'reward': groundtruth_reward},
    }
    save_name = secrets.token_hex(4)
    save_path = get_save_path("training", save_name)
    delete_duplicate_results("training", "", results)
    json.dump(results, open('../../results/' + save_path, 'w'))

    ground_truth_env.close()
    ground_truth_gym_env.close()


def train_concept_predictor_for_env(environment_string: str, seed: int, force: bool = False) -> None:
    """Train CNN concept predictor. Skips if cached."""
    gold_timesteps = GOLD_TIMESTEPS[environment_string]
    predictor_path = "../../results/models/concept_predictor_env={}_training={}_seed={}.pth".format(
        environment_string, 25, seed)

    if os.path.exists(predictor_path) and not force:
        print(f"  [skip] concept predictor already exists: {environment_string}, seed={seed}")
        return

    print(f"  Training concept predictor: {environment_string}, seed={seed}")
    np.random.seed(seed)
    random.seed(seed)

    concept_list, _ = get_concepts(environment_string, "human_selected_binary", seed)
    ground_truth_env, ground_truth_gym_env = get_environment(environment_string, None, seed)

    model_path = "../../results/models/env={}_training={}_seed={}.zip".format(
        environment_string, gold_timesteps, seed)
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Base policy not found: {model_path}\n"
            "Run train_prerequisites.py without --skip_concept_predictors first.")

    groundtruth_model = PPO.load(model_path)

    height = width = 84
    num_frames = 1 if environment_string == "mini_grid" else 4
    if environment_string == "cart_pole":
        height, width = 160, 240

    concept_predictor, acc_list = train_concept_predictor(
        ground_truth_gym_env, groundtruth_model, concept_list,
        list(range(len(concept_list))), environment_string,
        epochs=25, max_episode_length=10_000)
    torch.save(concept_predictor.state_dict(), predictor_path)
    print(f"    Mean concept accuracy: {np.mean(acc_list):.3f}")
    print(f"    Per-concept accuracy: {[round(a, 3) for a in acc_list.tolist()]}")

    ground_truth_env.close()
    ground_truth_gym_env.close()


# ── Main ──────────────────────────────────────────────────────────────────────
os.makedirs("../../results/models",      exist_ok=True)
os.makedirs("../../results/q_estimates", exist_ok=True)
os.makedirs("../../results/training",    exist_ok=True)

total_base       = len(args.envs) * len(args.seeds)
total_predictors = len([e for e in args.envs if e in ENVS_NEEDING_CONCEPT_PREDICTOR]) * len(args.seeds)

print(f"Environments : {args.envs}")
print(f"Seeds        : {args.seeds}")
print(f"Base policies to train    : {total_base}")
if not args.skip_concept_predictors:
    print(f"Concept predictors to train: {total_predictors}")
if args.dry_run:
    print("(dry run — nothing will be executed)\n")

# Step 1: Base policies + Q-estimates
print("\n── Step 1: Base policies and Q-estimates ────────────────────────────────")
for env in args.envs:
    for seed in args.seeds:
        if args.dry_run:
            print(f"  Would train: base policy {env}, seed={seed}")
        else:
            train_base_policy(env, seed, force=args.force)

# Step 2: Concept predictors
if not args.skip_concept_predictors:
    print("\n── Step 2: Concept predictors ───────────────────────────────────────────")
    for env in args.envs:
        if env not in ENVS_NEEDING_CONCEPT_PREDICTOR:
            print(f"  [skip] {env} uses perfect concept labels")
            continue
        for seed in args.seeds:
            if args.dry_run:
                print(f"  Would train: concept predictor {env}, seed={seed}")
            else:
                train_concept_predictor_for_env(env, seed, force=args.force)

print("\nAll prerequisites complete.")
print("You can now run: python run_experiment.py --config configs/main_perfect.yaml")