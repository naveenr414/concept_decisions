"""
supervised_learning.py

Concept selection for supervised learning on CUB-200-2011 (Figure 6, Section 5.4).
Compares automatic concept selection methods against manually selected concepts.

Runs two evaluations in a single invocation:
  - imperfect:    train and evaluate on predicted (noisy) concept labels
  - intervention: test-time correction of a fraction of concepts

Output: results/cub/

Usage:
    python supervised_learning.py --seed 42 --num_concepts_selected 112
"""

import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["GRB_LICENSE_FILE"] = os.environ.get("GRB_LICENSE_FILE", "/usr0/home/naveenr/gurobi.lic")

from concept_abstraction.selection import (
    random_selection,
    variance_selection_supervised,
    greedy_selection_supervised,
    drs_supervised,
    drs_log_supervised,
)
from concept_abstraction.utils import (
    get_save_path, delete_duplicate_results,
)

import argparse
import secrets
import numpy as np
import random
import pickle
import ujson as json
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# ── Arguments ─────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--seed",                  type=int, default=42)
parser.add_argument("--num_concepts_selected", type=int, default=112,
    help="Max concepts to select. Sweeps from 20 to this value in steps of 20.")
parser.add_argument("--out_folder",            type=str, default="cub")
args = parser.parse_args()

seed                  = args.seed
num_concepts_selected = args.num_concepts_selected
out_folder            = args.out_folder

results = {}
results["parameters"] = {
    "seed":                  seed,
    "num_concepts_selected": num_concepts_selected,
}
print("Parameters {}".format(results["parameters"]))

np.random.seed(seed)
random.seed(seed)

# ── Load data ─────────────────────────────────────────────────────────────────
# Ground-truth concept labels
train_gt = pickle.load(open("../../data/cub/train.pkl", "rb"))
test_gt  = pickle.load(open("../../data/cub/test.pkl",  "rb"))
train_X  = np.array([i["attribute_label"] for i in train_gt])
train_Y  = np.array([i["class_label"]     for i in train_gt])
test_X   = np.array([i["attribute_label"] for i in test_gt])
test_Y   = np.array([i["class_label"]     for i in test_gt])

# Predicted (noisy) concept labels
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

train_pred = pickle.load(open("../../data/cub/train_error.pkl", "rb"))
test_pred  = pickle.load(open("../../data/cub/test_error.pkl",  "rb"))
pred_train_X = sigmoid(np.array([i["attribute_label"] for i in train_pred])).round()
pred_test_X  = sigmoid(np.array([i["attribute_label"] for i in test_pred])).round()

train_concept_accuracy = np.mean(pred_train_X == train_X, axis=0)

manually_selected_concepts = [
    int(i) for i in
    open("../../data/cub/manual_concepts.txt").read().strip().split("\n")
]

# ── Helpers ───────────────────────────────────────────────────────────────────
def evaluate(selected_concepts, eval_X, hidden_size=256):
    """Train MLP on predicted train concepts, evaluate on eval_X."""
    mlp = MLPClassifier(
        hidden_layer_sizes=(hidden_size,), activation="relu", solver="adam")
    mlp.fit(pred_train_X[:, selected_concepts], train_Y)
    y_pred = mlp.predict(eval_X[:, selected_concepts])
    return float(accuracy_score(test_Y.reshape(-1, 1), y_pred))

# ── Concept selection sweep ───────────────────────────────────────────────────
num_concepts_range = range(20, num_concepts_selected + 1, 20)

results["imperfect"] = {}

results["imperfect"]["manual"] = {
    "reward":   evaluate(manually_selected_concepts, pred_test_X),
    "concepts": manually_selected_concepts,
}

results["imperfect"]["random"] = {}
for c in num_concepts_range:
    # random_selection expects a list; for CUB we pass indices as a dummy list
    concepts = random.sample(list(range(312)), c)
    results["imperfect"]["random"][c] = {
        "reward":   evaluate(concepts, pred_test_X),
        "concepts": concepts,
    }

results["imperfect"]["variance"] = {}
for c in num_concepts_range:
    concepts = variance_selection_supervised(train_X, train_Y, c)
    results["imperfect"]["variance"][c] = {
        "reward":   evaluate(concepts, pred_test_X),
        "concepts": concepts,
    }

results["imperfect"]["greedy"] = {}
for c in num_concepts_range:
    concepts = greedy_selection_supervised(train_X, train_Y, c)
    results["imperfect"]["greedy"][c] = {
        "reward":   evaluate(concepts, pred_test_X),
        "concepts": concepts,
    }

results["imperfect"]["drs"] = {}
for c in num_concepts_range:
    concepts = drs_supervised(train_X, train_Y, c)
    results["imperfect"]["drs"][c] = {
        "reward":   evaluate(concepts, pred_test_X),
        "concepts": concepts,
    }

results["imperfect"]["drs_log"] = {}
for c in num_concepts_range:
    concepts = drs_log_supervised(train_X, train_Y, train_concept_accuracy, c)
    results["imperfect"]["drs_log"][c] = {
        "reward":   evaluate(concepts, pred_test_X),
        "concepts": concepts,
    }

# ── Intervention sweep ────────────────────────────────────────────────────────
# Fix k=num_concepts_selected for all methods, vary fraction of concepts corrected.
k = num_concepts_selected
selections = {
    "manual":   manually_selected_concepts,
    "random":   random.sample(list(range(312)), k),
    "variance": variance_selection_supervised(train_X, train_Y, k),
    "greedy":   greedy_selection_supervised(train_X, train_Y, k),
    "drs":      drs_supervised(train_X, train_Y, k),
    "drs_log":  drs_log_supervised(train_X, train_Y, train_concept_accuracy, k),
}

results["intervention"] = {name: {} for name in selections}

for intervention_frac in [0.2, 0.4, 0.6, 0.8, 1.0]:
    # Replace (1 - intervention_frac) of columns with predicted values
    num_cols_predicted = int((1 - intervention_frac) * test_X.shape[1])
    cols_predicted = np.random.choice(test_X.shape[1], num_cols_predicted, replace=False)
    intervention_test_X = test_X.copy()
    intervention_test_X[:, cols_predicted] = pred_test_X[:, cols_predicted]

    for name, concepts in selections.items():
        acc = evaluate(concepts, intervention_test_X)
        results["intervention"][name][intervention_frac] = {"reward": acc}
        print(f"  intervention {intervention_frac:.1f}  {name:10s}  {acc:.4f}")

# ── Save ──────────────────────────────────────────────────────────────────────
save_name = secrets.token_hex(4)
save_path = get_save_path(out_folder, save_name)
delete_duplicate_results(out_folder, "", results)
json.dump(results, open("../../results/" + save_path, "w"))