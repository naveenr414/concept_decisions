"""
run_comparison.py

Unified entry point for all concept selection experiments.
Replaces method_comparison_perfect.py, method_comparison_imperfect.py,
and method_comparison_intervention.py.

The three settings differ in two ways:
  - perfect:      uses ground-truth concept labels (no CNN predictor)
  - imperfect:    uses a trained CNN concept predictor
  - intervention: same as imperfect, but injects intervention_prob at
                  test time and uses a weakly trained predictor

Usage:
    python run_comparison.py --setting perfect      --environment_string mini_grid ...
    python run_comparison.py --setting imperfect    --environment_string mini_grid ...
    python run_comparison.py --setting intervention --environment_string mini_grid \
                             --intervention_prob 0.5 --predictor_epochs 1
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
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

from concept_abstraction.training import *
from concept_abstraction.selection import *
from concept_abstraction.concept_bank import *
from concept_abstraction.env_utils import *
from concept_abstraction.environments import *
from concept_abstraction.utils import *
import argparse
import numpy as np
import random
import pickle
import secrets

# ── Arguments ─────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--setting', type=str, required=True,
    choices=['perfect', 'imperfect', 'intervention'],
    help="perfect: ground-truth concepts; imperfect: trained predictor; "
         "intervention: imperfect + test-time correction")
parser.add_argument('--seed',                  type=int,   default=42)
parser.add_argument('--environment_string',    type=str,   default="mini_grid")
parser.add_argument('--training_timesteps',    type=int,   default=10000)
parser.add_argument('--gold_timesteps',        type=int,   default=10000)
parser.add_argument('--num_concepts_selected', type=int,   default=0)
parser.add_argument('--out_folder',            type=str,   default="basic")
parser.add_argument('--method',                type=str,   default='random',
    choices=[
        # Baselines (Section 5.1)
        'all_concepts', 'random', 'variance', 'greedy',
        # Main methods (Section 4)
        'drs', 'drs_log',
        # Ablations: rho sweep and P1c relaxation (Appendix D)
        'drs_rho_0', 'drs_rho_05', 'drs_rho_099', 'drs_no_relax',
    ])
# Intervention-only arguments
parser.add_argument('--intervention_prob', type=float, default=0.0,
    help="Fraction of concepts corrected at test time (alpha in paper). "
         "Only used with --setting intervention.")
parser.add_argument('--predictor_epochs', type=int, default=25,
    help="Epochs to train concept predictor. Use 1 for weak predictor (Section 5.3). "
         "Only used with --setting imperfect or intervention.")
args = parser.parse_args()

setting               = args.setting
seed                  = args.seed
environment_string    = args.environment_string
gold_timesteps        = args.gold_timesteps
training_timesteps    = args.training_timesteps
num_concepts_selected = args.num_concepts_selected
out_folder            = args.out_folder
method                = args.method
intervention_prob     = args.intervention_prob
predictor_epochs      = args.predictor_epochs

# drs_log is only defined for imperfect concept predictors
if setting == 'perfect' and method == 'drs_log':
    raise ValueError("drs_log requires a trained concept predictor; use --setting imperfect or intervention.")

# ── Logging ───────────────────────────────────────────────────────────────────
results = {}
results['parameters'] = {
    'seed':                  seed,
    'environment_string':    environment_string,
    'training_timesteps':    training_timesteps,
    'gold_timesteps':        gold_timesteps,
    'num_concepts_selected': num_concepts_selected,
    'method':                method,
    'setting':               setting,
    **(({'intervention_prob': intervention_prob}) if setting == 'intervention' else {}),
}
print("Parameters {}".format(results['parameters']))

np.random.seed(seed)
random.seed(seed)

# ── Environment and ground-truth policy ───────────────────────────────────────
concept_list, processed_concepts = get_concepts(environment_string, "human_selected_binary", seed)
num_concepts_selected = min(num_concepts_selected, len(concept_list))
ground_truth_env, ground_truth_gym_env = get_environment(environment_string, None, seed)

model_name = "../../results/models/env={}_training={}_seed={}.zip".format(
    environment_string, gold_timesteps, seed)

if not os.path.exists(model_name):
    raise FileNotFoundError(
        f"Base policy not found: {model_name}\n"
        "Run train_prerequisites.py before running experiments.")
load_kwargs = {}
if setting == 'perfect':
    load_kwargs['custom_objects'] = {
        "observation_space": ground_truth_env.observation_space,
        "action_space":      ground_truth_env.action_space,
    }
groundtruth_model = PPO.load(model_name, **load_kwargs)

if method == 'all_concepts':
    groundtruth_reward = evaluate_model(
        environment_string, ground_truth_gym_env, groundtruth_model, seed)
    results['ground_truth'] = {'reward': groundtruth_reward}

# ── Concept predictor (imperfect and intervention only) ───────────────────────
if setting in ('imperfect', 'intervention'):
    epochs = predictor_epochs if setting == 'intervention' else 25
    predictor_name = "../../results/models/concept_predictor_env={}_training={}_seed={}.pth".format(
        environment_string, epochs, seed)

    height = width = 84
    num_frames = 1 if environment_string == "mini_grid" else 4
    if environment_string == "cart_pole":
        height, width = 160, 240

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if not os.path.exists(predictor_name):
        raise FileNotFoundError(
            f"Concept predictor not found: {predictor_name}\n"
            "Run train_prerequisites.py before running experiments.")
    concept_predictor = ConceptPredictorCNN(
        len(concept_list), num_frames=num_frames, height=height, width=width).to(device)
    concept_predictor.load_state_dict(torch.load(predictor_name, weights_only=True))
    concept_predictor.eval()

    # drs_log needs per-concept accuracy
    if method == 'drs_log' and 'concept_accuracy' not in results:
        if setting == 'intervention':
            # reuse accuracy from a prior imperfect run if available
            matching = get_results_matching_parameters(
                "training", "",
                {'environment_string': environment_string,
                 'gold_timesteps': gold_timesteps, 'seed': seed})
            if matching and 'concept_accuracy' in matching[0]:
                acc_list = matching[0]['concept_accuracy']
                results["concept_accuracy"] = acc_list
            else:
                acc_list = evaluate_concept_predictor(
                    concept_predictor, ground_truth_gym_env, groundtruth_model, concept_list)
                results["concept_accuracy"] = acc_list.tolist()
        else:
            acc_list = evaluate_concept_predictor(
                concept_predictor, ground_truth_gym_env, groundtruth_model, concept_list)
            results["concept_accuracy"] = acc_list.tolist()

    if torch.cuda.is_available():
        concept_predictor = concept_predictor.cuda()
        concept_predictor.eval()
        with torch.no_grad():
            _ = concept_predictor(torch.zeros(8, num_frames, height, width, device='cuda'))

# ── Q-value estimation ────────────────────────────────────────────────────────
q_name = "../../results/q_estimates/env={}_training={}_seed={}_selection={}_source={}.pkl".format(
    environment_string, gold_timesteps, seed, "q_value", "human_selected_binary")
if not os.path.exists(q_name):
    raise FileNotFoundError(
        f"Q-estimates not found: {q_name}\n"
        "Run train_prerequisites.py before running experiments.")
q_estimates = pickle.load(open(q_name, "rb"))

# ── Concept selection ─────────────────────────────────────────────────────────

if method == 'all_concepts':
    subset_concept = concept_list
    idx = list(range(len(concept_list)))

if method == 'random':
    subset_concept, idx = random_selection(concept_list, num_concepts_selected)

if method == 'variance':
    subset_concept, idx = basic_greedy_selection(
        concept_list, num_concepts_selected, "q_value", q_estimates, "human_selected_binary")

if method == 'greedy':
    subset_concept, idx = greedy_selection(
        concept_list, num_concepts_selected, "q_value", q_estimates, "human_selected_binary")

if method == 'drs':
    subset_concept, idx = policy_coverage_selection_lp_hybrid(
        ground_truth_gym_env, concept_list, num_concepts_selected,
        groundtruth_model, q_estimates, coverage_ratio=0.75)

if method == 'drs_log':
    subset_concept, idx = policy_coverage_selection_multiple_log(
        ground_truth_gym_env, concept_list, num_concepts_selected,
        groundtruth_model, q_estimates, acc_list)

# Rho ablation (Appendix D)
if method == 'drs_rho_0':
    subset_concept, idx = policy_coverage_selection_lp_hybrid(
        ground_truth_gym_env, concept_list, num_concepts_selected,
        groundtruth_model, q_estimates, coverage_ratio=0.0)

if method == 'drs_rho_05':
    subset_concept, idx = policy_coverage_selection_lp_hybrid(
        ground_truth_gym_env, concept_list, num_concepts_selected,
        groundtruth_model, q_estimates, coverage_ratio=0.5)

if method == 'drs_rho_099':
    subset_concept, idx = policy_coverage_selection_lp_hybrid(
        ground_truth_gym_env, concept_list, num_concepts_selected,
        groundtruth_model, q_estimates, coverage_ratio=0.99)

# P1c relaxation ablation (Appendix D)
if method == 'drs_no_relax':
    subset_concept, idx = policy_coverage_selection_lp_hybrid(
        ground_truth_gym_env, concept_list, num_concepts_selected,
        groundtruth_model, q_estimates, coverage_ratio=0.75, prefix=True)

# ── Train concept-based policy and evaluate ───────────────────────────────────
if setting == 'perfect':
    env_kwargs = dict(processed_concepts=processed_concepts, concept_idx=idx)
else:
    env_kwargs = dict(
        fast_predictor=concept_predictor, use_processed=True,
        concept_idx=idx, processed_concepts=processed_concepts,
        **(({'intervention_prob': intervention_prob}) if setting == 'intervention' else {}),
    )

two_stage_env, two_stage_gym_env = get_environment(
    environment_string,
    subset_concept if setting == 'perfect' else concept_list,
    seed, **env_kwargs)

model = train_ppo_model(
    two_stage_env, environment_string, policy="MlpPolicy",
    total_timesteps=training_timesteps,
    custom_name="{}_{}_{}_{}".format(environment_string, setting, method, seed))

reward = evaluate_model(environment_string, two_stage_gym_env, model, seed)
results[method] = {'reward': reward, 'concepts': idx}

# ── Save ──────────────────────────────────────────────────────────────────────
save_name = secrets.token_hex(4)
save_path = get_save_path(out_folder, save_name)
delete_duplicate_results(out_folder, "", results)
json.dump(results, open('../../results/' + save_path, 'w'))

ground_truth_env.close()
ground_truth_gym_env.close()
two_stage_env.close()
two_stage_gym_env.close()