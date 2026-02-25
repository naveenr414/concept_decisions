import numpy as np
import gurobipy as gp
from gurobipy import GRB
import scipy
from copy import deepcopy
import random
import torch
from itertools import combinations
from collections import Counter
import torch.nn as nn
import torch.optim as optim


# ── RL concept selection ──────────────────────────────────────────────────────

def random_selection(concept_list, num_concepts):
    """Randomly select `num_concepts` from `concept_list`.

    Returns:
        concepts: Selected concept functions
        idx: Sorted list of selected indices
    """
    total = len(concept_list)
    idx = sorted(np.random.choice(total, num_concepts, replace=False).tolist())
    return [concept_list[i] for i in idx], idx


def variance_selection(concept_list, num_concepts, q_estimates):
    """Select concepts that minimise weighted conditional variance of Q-values.

    For each concept, compute the total variance of Q-values when the concept
    is 0 vs 1 (weighted by group size), then select the `num_concepts` with
    the lowest total variance.

    Args:
        concept_list: Full list of concept functions
        num_concepts: Number to select
        q_estimates: List of (concept_vector, action, q_value) triples

    Returns:
        concepts: Selected concept functions
        idx: List of selected indices
    """
    unique_actions = list(set(int(i[1]) for i in q_estimates))

    variance_by_concept = []
    for c_idx in range(len(concept_list)):
        total_variance = 0
        for action in unique_actions:
            x_0 = [i[2] for i in q_estimates if i[0][c_idx] == 0 and int(i[1]) == action]
            x_1 = [i[2] for i in q_estimates if i[0][c_idx] == 1 and int(i[1]) == action]
            total_variance += np.std(x_0) * len(x_0) + np.std(x_1) * len(x_1)
        variance_by_concept.append(total_variance)

    variance_by_concept = np.array(variance_by_concept)
    idx = np.argpartition(variance_by_concept, num_concepts - 1)[:num_concepts]
    idx = idx[np.argsort(variance_by_concept[idx])].tolist()
    idx = [int(i) for i in idx]
    return [concept_list[i] for i in idx], idx


def greedy_selection(concept_list, num_concepts, q_estimates):
    """Alias for variance_selection for backwards compatibility."""
    return variance_selection(concept_list, num_concepts, q_estimates)


def drs(
    ground_truth_gym_env,
    concept_list,
    num_concepts,
    groundtruth_model,
    q_estimates,
    num_pairs=20_000,
    rollout_steps=10_000,
    coverage_ratio=0.75,
    fixed_idx=None,
):
    """DRS: select concepts via a coverage LP over Q-value-distinguishing pairs.

    Maximises coverage of high-Q-value-difference state pairs subject to a
    cardinality budget, then maximises Q-value distinguishability as the
    secondary objective.

    Args:
        ground_truth_gym_env: Vectorised gym environment
        concept_list: Full concept list
        num_concepts: Budget k
        groundtruth_model: Policy used to collect rollout observations
        q_estimates: List of (concept_vector, action, q_value) triples
        num_pairs: Number of cross-action pairs for the coverage constraint
        rollout_steps: Steps to collect policy observations
        coverage_ratio: Minimum fraction of pairs that must be covered (rho)
        fixed_idx: Concept indices forced into the solution (for recursive calls)

    Returns:
        concepts: Selected concept functions
        idx: List of selected indices
    """
    if fixed_idx is None:
        fixed_idx = []

    unique_actions = list(set(int(i[1]) for i in q_estimates))
    actions_arr = np.array([i[1] for i in q_estimates])
    discretized_X = np.array([i[0] for i in q_estimates])
    q_values = np.array([i[2] for i in q_estimates])

    # Build Q-distinguishing pairs
    seen = set()
    final_vals = []
    for a in unique_actions:
        relevant_idx = np.where(actions_arr == a)[0]
        if len(relevant_idx) <= 500:
            relevant_low = relevant_high = relevant_idx
        else:
            relevant_low = np.argsort(np.abs(q_values))[:500]
            relevant_high = np.argsort(np.abs(q_values))[-500:]
        for low_idx in relevant_low:
            for high_idx in relevant_high:
                diff = abs(q_values[low_idx] - q_values[high_idx])
                diffs = tuple(
                    i for i, (l, h) in enumerate(zip(discretized_X[low_idx], discretized_X[high_idx]))
                    if l != h
                )
                if diffs not in seen and diffs:
                    seen.add(diffs)
                    final_vals.append((diff, diffs))
    final_vals = sorted(final_vals[:250_000], reverse=True)

    # Collect policy observations
    all_observations, all_actions_list = [], []
    obs, info = ground_truth_gym_env.reset()
    for _ in range(rollout_steps):
        actions = groundtruth_model.predict(obs)[0]
        for j in range(len(actions)):
            all_observations.append([c(info[j]["observation"]) for c in concept_list])
            all_actions_list.append(actions[j])
        obs, _, _, _, info = ground_truth_gym_env.step(actions)

    all_observations = np.asarray(all_observations, dtype=np.int8)
    all_actions_arr = np.asarray(all_actions_list)
    N, K = all_observations.shape

    # Sample cross-action pairs
    idx_i = np.random.randint(0, N, size=5 * num_pairs)
    idx_j = np.random.randint(0, N, size=5 * num_pairs)
    valid = all_actions_arr[idx_i] != all_actions_arr[idx_j]
    idx_i = idx_i[valid][:num_pairs]
    idx_j = idx_j[valid][:num_pairs]
    if len(idx_i) == 0:
        raise ValueError("No cross-action pairs sampled.")
    M = len(idx_i)
    disagreement = (all_observations[idx_i] != all_observations[idx_j]).astype(np.int8)

    # Build Gurobi model
    gmodel = gp.Model("drs")
    gmodel.Params.OutputFlag = 0

    x = gmodel.addVars(K, lb=0.0, ub=1.0, vtype=GRB.BINARY, name="x")
    y = gmodel.addVars(M, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="y")
    y2 = gmodel.addVars(len(final_vals), lb=0.0, ub=1.0, vtype=GRB.BINARY, name="y2")

    for i, (_, elems) in enumerate(final_vals):
        if elems:
            gmodel.addConstr(y2[i] <= gp.quicksum(x[e] for e in elems))
        else:
            gmodel.addConstr(y2[i] == 0)

    for p in range(M):
        gmodel.addConstr(y[p] <= gp.quicksum(disagreement[p, d] * x[d] for d in range(K)))

    gmodel.addConstr(gp.quicksum(x[d] for d in range(K)) <= num_concepts)
    gmodel.addConstr(gp.quicksum(y[p] for p in range(M)) / M >= coverage_ratio)

    for i in fixed_idx:
        gmodel.addConstr(x[i] == 1)

    weights = [v for v, _ in final_vals]
    gmodel.setObjective(gp.quicksum(weights[i] * y2[i] for i in range(len(final_vals))), GRB.MAXIMIZE)
    gmodel.optimize()

    if gmodel.Status not in (GRB.OPTIMAL, GRB.TIME_LIMIT):
        coverage_ratio -= 0.05
        if coverage_ratio < 0:
            return [], [0]
        return drs(ground_truth_gym_env, concept_list, num_concepts, groundtruth_model,
                   q_estimates, coverage_ratio=coverage_ratio)

    x_vals = np.array([x[d].X for d in range(K)])
    idx = [i for i in range(len(x_vals)) if x_vals[i] > 0.5]

    if len(idx) < num_concepts and not fixed_idx:
        return drs(ground_truth_gym_env, concept_list, num_concepts, groundtruth_model,
                   q_estimates, fixed_idx=idx)

    covered = disagreement[:, idx].any(axis=1)
    print(f"DRS coverage: {covered.mean():.3f}")

    return [concept_list[i] for i in idx], idx


def drs_log(
    ground_truth_gym_env,
    concept_list,
    num_concepts,
    groundtruth_model,
    q_estimates,
    acc_list,
    num_pairs=20_000,
    rollout_steps=10_000,
    coverage_ratio=0.75,
    fixed_idx=None,
    _final_vals=None,
    _all_observations=None,
    _all_actions=None,
):
    """DRS-Log: DRS variant using probabilistic coverage with log constraints.

    Accounts for imperfect concept accuracy by modelling the probability that
    a pair of states is distinguishable given binary concept noise.

    Args:
        ground_truth_gym_env: Vectorised gym environment
        concept_list: Full concept list
        num_concepts: Budget k
        groundtruth_model: Policy used to collect rollout observations
        q_estimates: List of (concept_vector, action, q_value) triples
        acc_list: Per-concept accuracy values in [0, 1]
        num_pairs: Number of cross-action pairs
        rollout_steps: Steps to collect policy observations
        coverage_ratio: Minimum expected coverage fraction (rho)
        fixed_idx: Concept indices forced into the solution
        _final_vals: Pre-computed Q-distinguishing pairs (internal, for recursion)
        _all_observations: Pre-collected observations (internal, for recursion)
        _all_actions: Pre-collected actions (internal, for recursion)

    Returns:
        concepts: Selected concept functions
        idx: List of selected indices
    """
    if fixed_idx is None:
        fixed_idx = []

    unique_actions = list(set(int(i[1]) for i in q_estimates))
    actions_arr = np.array([i[1] for i in q_estimates])
    discretized_X = np.array([i[0] for i in q_estimates])
    q_values = np.array([i[2] for i in q_estimates])
    acc_list = [min(a, 0.99) for a in acc_list]

    # Build Q-distinguishing pairs (reuse if provided)
    if _final_vals is None:
        seen = set()
        _final_vals = []
        for a in unique_actions:
            relevant_idx = np.where(actions_arr == a)[0]
            if len(relevant_idx) <= 500:
                relevant_low = relevant_high = relevant_idx
            else:
                q_sub = q_values[relevant_idx]
                order = np.argsort(np.abs(q_sub))
                relevant_low = relevant_idx[order[:500]]
                relevant_high = relevant_idx[order[-500:]]
            for low_idx in relevant_low:
                for high_idx in relevant_high:
                    diff = abs(q_values[low_idx] - q_values[high_idx])
                    diffs = tuple(
                        i for i, (l, h) in enumerate(zip(discretized_X[low_idx], discretized_X[high_idx]))
                        if l != h
                    )
                    if diffs not in seen and diffs:
                        seen.add(diffs)
                        _final_vals.append((diff, diffs))
        _final_vals = sorted(_final_vals[:100_000], reverse=True)

    # Collect policy observations (reuse if provided)
    if _all_observations is None:
        all_obs_list, all_act_list = [], []
        obs, info = ground_truth_gym_env.reset()
        for _ in range(rollout_steps):
            actions = groundtruth_model.predict(obs)[0]
            for j in range(len(actions)):
                all_obs_list.append([c(info[j]["observation"]) for c in concept_list])
                all_act_list.append(actions[j])
            obs, _, _, _, info = ground_truth_gym_env.step(actions)
        _all_observations = np.asarray(all_obs_list, dtype=np.int8)
        _all_actions = np.asarray(all_act_list)

    N, K = _all_observations.shape

    # Sample cross-action pairs
    idx_i_list, idx_j_list = [], []
    while len(idx_i_list) < num_pairs:
        i_batch = np.random.randint(0, N, size=num_pairs)
        j_batch = np.random.randint(0, N, size=num_pairs)
        mask = _all_actions[i_batch] != _all_actions[j_batch]
        idx_i_list.append(i_batch[mask])
        idx_j_list.append(j_batch[mask])
    idx_i = np.concatenate(idx_i_list)[:num_pairs]
    idx_j = np.concatenate(idx_j_list)[:num_pairs]
    if len(idx_i) == 0:
        raise ValueError("No cross-action pairs sampled.")
    M = len(idx_i)
    disagreement = (_all_observations[idx_i] != _all_observations[idx_j]).astype(np.uint8)

    # Precompute log constants
    acc = np.array(acc_list)
    p_same = acc ** 2 + (1 - acc) ** 2
    log_const = np.log(1 - np.minimum(p_same, 1 - 1e-9))

    # Build Gurobi model
    gmodel = gp.Model("drs_log")
    gmodel.Params.OutputFlag = 0
    gmodel.Params.Presolve = 2
    gmodel.Params.MIPFocus = 1
    gmodel.Params.Threads = 8

    x = gmodel.addVars(K, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="x")
    y = gmodel.addVars(M, lb=0.0, ub=1.0, name="y")
    y2 = gmodel.addVars(len(_final_vals), lb=0.0, ub=1.0, name="y2")
    z2 = gmodel.addVars(len(_final_vals), lb=-GRB.INFINITY, ub=0.0, name="z2")
    s2 = gmodel.addVars(len(_final_vals), lb=-GRB.INFINITY, ub=0.0, name="s2")

    bp = np.array([1e-6, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9])
    bp_vals = np.log(1 - bp)

    for i, (_, elems) in enumerate(_final_vals):
        if elems:
            coeffs = log_const[list(elems)]
            vars_ = [x[e] for e in elems]
            gmodel.addConstr(s2[i] == gp.LinExpr(coeffs, vars_))
            gmodel.addGenConstrPWL(y2[i], z2[i], bp.tolist(), bp_vals.tolist())
            gmodel.addConstr(z2[i] >= s2[i])
        else:
            gmodel.addConstr(y2[i] == 0.0)

    for p in range(M):
        nz = np.flatnonzero(disagreement[p])
        if len(nz):
            gmodel.addConstr(y[p] <= gp.quicksum(x[d] for d in nz))
        else:
            gmodel.addConstr(y[p] == 0.0)

    gmodel.addConstr(gp.quicksum(y[p] for p in range(M)) / M >= coverage_ratio)
    gmodel.addConstr(gp.quicksum(x[d] for d in range(K)) <= num_concepts)
    for d in fixed_idx:
        gmodel.addConstr(x[d] == 1.0)

    weights = np.array([w for w, _ in _final_vals])
    gmodel.setObjective(
        gp.quicksum(weights[i] * y2[i] for i in range(len(_final_vals))),
        GRB.MAXIMIZE,
    )
    gmodel.optimize()

    if gmodel.Status not in (GRB.OPTIMAL, GRB.TIME_LIMIT):
        coverage_ratio -= 0.05
        if coverage_ratio < 0:
            return [], [0]
        return drs_log(
            ground_truth_gym_env, concept_list, num_concepts, groundtruth_model,
            q_estimates, acc_list, coverage_ratio=coverage_ratio,
            _final_vals=_final_vals, _all_observations=_all_observations,
            _all_actions=_all_actions,
        )

    x_vals = np.array([x[d].X for d in range(K)])
    idx = sorted(np.argsort(-x_vals)[:num_concepts].tolist())
    idx = [i for i in idx if x_vals[i] > 0.01]

    if len(idx) < num_concepts and not fixed_idx:
        return drs_log(
            ground_truth_gym_env, concept_list, num_concepts, groundtruth_model,
            q_estimates, acc_list, fixed_idx=idx,
            _final_vals=_final_vals, _all_observations=_all_observations,
            _all_actions=_all_actions,
        )

    covered = disagreement[:, idx].any(axis=1)
    print(f"DRS-Log coverage: {covered.mean():.3f}")

    return [concept_list[i] for i in idx], idx


# ── Supervised (CUB) concept selection ───────────────────────────────────────

def variance_selection_supervised(train_X, train_Y, num_concepts):
    """Select concepts by highest marginal entropy (most balanced split).

    Args:
        train_X: (N, d) binary concept matrix
        train_Y: (N,) class labels (unused, kept for API consistency)
        num_concepts: Number to select

    Returns:
        idx: List of selected concept indices
    """
    entropy_by_concept = np.array([
        (train_X[:, i] == 0).mean() * (train_X[:, i] == 1).mean()
        for i in range(train_X.shape[1])
    ])
    idx = np.argpartition(-entropy_by_concept, num_concepts)[:num_concepts]
    return idx[np.argsort(-entropy_by_concept[idx])].tolist()


def greedy_selection_supervised(train_X, train_Y, num_concepts):
    """Select concepts that minimise conditional variance of Y.

    Args:
        train_X: (N, d) binary concept matrix
        train_Y: (N,) continuous labels
        num_concepts: Number to select

    Returns:
        idx: List of selected concept indices
    """
    N, d = train_X.shape
    conditional_variance = []
    for i in range(d):
        mask0 = train_X[:, i] == 0
        var = 0.0
        if mask0.any():
            var += np.var(train_Y[mask0]) * mask0.sum()
        if (~mask0).any():
            var += np.var(train_Y[~mask0]) * (~mask0).sum()
        conditional_variance.append(var / N)

    conditional_variance = np.array(conditional_variance)
    idx = np.argpartition(conditional_variance, num_concepts)[:num_concepts]
    return idx[np.argsort(conditional_variance[idx])].tolist()


def drs_supervised(train_X, train_Y, num_concepts):
    """Supervised DRS: LP selecting concepts that cover label-differing pairs.

    Args:
        train_X: (N, d) binary concept matrix
        train_Y: (N,) class labels
        num_concepts: Budget k

    Returns:
        idx: List of selected concept indices
    """
    train_X = np.asarray(train_X)
    train_Y = np.asarray(train_Y)

    rows_as_tuples = [tuple(row) for row in train_X]
    row_counts = Counter(rows_as_tuples)
    unique_rows = np.array(list(row_counts.keys()))
    unique_counts = np.array([row_counts[r] for r in row_counts])
    unique_labels = np.array([
        train_Y[np.all(train_X == r, axis=1)][0] for r in unique_rows
    ])

    pairs = [
        (i, j)
        for i, j in combinations(range(len(unique_rows)), 2)
        if unique_labels[i] != unique_labels[j]
    ]
    final_vals = [
        (unique_counts[i] * unique_counts[j],
         np.where(unique_rows[i] != unique_rows[j])[0].tolist())
        for i, j in pairs
    ]

    n = len(final_vals)
    m = max(max(elems) for _, elems in final_vals if elems) + 1

    gmodel = gp.Model("drs_supervised")
    gmodel.Params.OutputFlag = 0

    x = gmodel.addVars(m, vtype=GRB.BINARY, name="x")
    y = gmodel.addVars(n, lb=0.0, vtype=GRB.CONTINUOUS, name="y")

    for i, (_, elems) in enumerate(final_vals):
        if elems:
            gmodel.addConstr(y[i] <= gp.quicksum(x[e] for e in elems))
        else:
            gmodel.addConstr(y[i] == 0)

    gmodel.addConstr(gp.quicksum(x[i] for i in range(m)) == num_concepts)
    gmodel.setObjective(
        gp.quicksum(y[i] * final_vals[i][0] for i in range(n)),
        GRB.MAXIMIZE,
    )
    gmodel.optimize()

    return [i for i in range(m) if x[i].X > 0.5]


def drs_log_supervised(train_X, train_Y, concept_accuracy, num_concepts):
    """Supervised DRS-Log: probabilistic coverage LP for imperfect concepts.

    Args:
        train_X: (N, d) binary concept matrix
        train_Y: (N,) class labels
        concept_accuracy: Per-concept accuracy values in [0, 1]
        num_concepts: Budget k

    Returns:
        idx: List of selected concept indices
    """
    train_X = np.asarray(train_X)
    train_Y = np.asarray(train_Y)

    rows_as_tuples = [tuple(row) for row in train_X]
    row_counts = Counter(rows_as_tuples)
    unique_rows = np.array(list(row_counts.keys()))
    unique_counts = np.array([row_counts[r] for r in row_counts])
    unique_labels = np.array([
        train_Y[np.all(train_X == r, axis=1)][0] for r in unique_rows
    ])

    pairs = [
        (i, j)
        for i, j in combinations(range(len(unique_rows)), 2)
        if unique_labels[i] != unique_labels[j]
    ]
    final_vals = [
        (unique_counts[i] * unique_counts[j],
         np.where(unique_rows[i] != unique_rows[j])[0].tolist())
        for i, j in pairs
    ]

    n = len(final_vals)
    m = max(max(elems) for _, elems in final_vals if elems) + 1

    gmodel = gp.Model("drs_log_supervised")
    gmodel.Params.OutputFlag = 0

    x = gmodel.addVars(m, vtype=GRB.BINARY, name="x")
    y = gmodel.addVars(n, lb=0.0, vtype=GRB.CONTINUOUS, name="y")

    for i, (_, elems) in enumerate(final_vals):
        if elems:
            gmodel.addConstr(
                y[i] <= gp.quicksum(x[e] * concept_accuracy[e] for e in elems)
            )
        else:
            gmodel.addConstr(y[i] == 0)

    gmodel.addConstr(gp.quicksum(x[i] for i in range(m)) == num_concepts)
    gmodel.setObjective(
        gp.quicksum(y[i] * final_vals[i][0] for i in range(n)),
        GRB.MAXIMIZE,
    )
    gmodel.optimize()

    return [i for i in range(m) if x[i].X > 0.5]