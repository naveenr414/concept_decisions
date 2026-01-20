import numpy as np
import gurobipy as gp
from gurobipy import GRB
import scipy
from copy import deepcopy
import random
import torch
from itertools import combinations
from collections import Counter 
import math 
import copy 
import torch.nn as nn
import torch.optim as optim
from concept_abstraction.env_utils import rollout_q_estimates_td
from sklearn.feature_selection import mutual_info_regression

class ValueNet(nn.Module):
    def __init__(self, obs_dim, hidden_sizes=(64,64)):
        super().__init__()
        layers = []
        last_dim = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last_dim, h))
            layers.append(nn.ReLU())
            last_dim = h
        layers.append(nn.Linear(last_dim, 1))
        self.net = nn.Sequential(*layers)
        
        # Initialize final layer with small weights for stability
        nn.init.orthogonal_(self.net[-1].weight, gain=0.01)
        nn.init.constant_(self.net[-1].bias, 0.0)
    
    def forward(self, obs):
        # obs: (batch, obs_dim)
        return self.net(obs).squeeze(-1)


class SimpleMCTrainer:
    def __init__(self, value_net, env, policy, concept_list, gamma=0.995, lr=1e-4, device='cpu'):
        self.value_net = value_net.to(device)
        self.env = env
        self.policy = policy
        self.gamma = gamma
        self.device = device
        self.optimizer = optim.Adam(self.value_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.concept_list = concept_list

    def run_rollouts(self, num_episodes=50):
        """Collect data from multiple episodes."""
        obs_list, returns_list = [], []
        num_envs = 8

        for ep in range(num_episodes):
            print(f"Episode {ep+1}")
            obs, infos = self.env.reset()
            obs, _, _, _, infos = self.env.step([1 for i in range(8)])
            done = [False] * num_envs
            rewards = [[] for _ in range(num_envs)]
            ep_obs = [[] for _ in range(num_envs)]

            while not all(done):
                for i in range(num_envs):
                    if not done[i]:
                        ep_obs[i].append([c(infos[i]['observation']) for c in self.concept_list])
                action, _ = self.policy.predict(obs, deterministic=False)
                next_obs, reward, terminated, truncated, infos = self.env.step(action)
                obs = next_obs

                for i in range(num_envs):
                    if not done[i]:
                        rewards[i].append(reward[i])
                done = [done[i] or terminated[i] or truncated[i] for i in range(num_envs)]

            # Compute discounted returns
            for i in range(8):
                G = 0
                ep_returns = []
                for r in reversed(rewards[i]):
                    G = r + self.gamma * G
                    ep_returns.insert(0, G)

                obs_list.extend(ep_obs[i])
                returns_list.extend(ep_returns)

        obs_array = np.array(obs_list, dtype=np.float32)
        returns_array = np.array(returns_list, dtype=np.float32)
        return obs_array, returns_array

    def train_value_net(self, obs_array, returns_array, batch_size=64, epochs=25):
        t_obs = torch.as_tensor(obs_array, dtype=torch.float32).to(self.device)
        t_returns = torch.as_tensor(returns_array, dtype=torch.float32).to(self.device)

        dataset_size = len(t_obs)
        indices = np.arange(dataset_size)

        self.value_net.train()
        for _ in range(epochs):
            np.random.shuffle(indices)
            for start in range(0, dataset_size, batch_size):
                end = start + batch_size
                idx = indices[start:end]
                batch_x = t_obs[idx]
                batch_y = t_returns[idx]

                preds = self.value_net(batch_x)
                loss = self.loss_fn(preds, batch_y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
    def evaluate(self, num_episodes=20):
        """
        Evaluate the value function using vectorized episodes.
        Compares predicted values for initial states to actual Monte Carlo returns.
        """
        self.value_net.eval()
        predicted, actual = [], []

        num_envs = 8  # number of parallel environments
        for ep in range(num_episodes):
            print(f"Episode {ep+1}")
            obs, infos = self.env.reset()
            obs, _, _, _, infos = self.env.step([1 for i in range(8)])
            done = [False] * num_envs
            rewards = [[] for _ in range(num_envs)]
            ep_obs = [[] for _ in range(num_envs)]

            while not all(done):
                for i in range(num_envs):
                    if not done[i]:
                        ep_obs[i].append([c(infos[i]['observation']) for c in self.concept_list])

                action, _ = self.policy.predict(obs, deterministic=False)
                next_obs, reward, terminated, truncated, infos = self.env.step(action)
                obs = next_obs

                for i in range(num_envs):
                    if not done[i]:
                        rewards[i].append(reward[i])
                done = [done[i] or terminated[i] or truncated[i] for i in range(num_envs)]

            # Compute total discounted return for each env
            for i in range(num_envs):
                G = 0
                for idx,r in enumerate(reversed(rewards[i])):
                    G = r + self.gamma * G
                    # Predict value for initial observation
                    obs_t = torch.as_tensor(np.array(ep_obs[i][-idx], dtype=np.float32)).unsqueeze(0).to(self.device)
                    pred_val = self.value_net(obs_t).item()

                    predicted.append(pred_val)
                    actual.append(G)
        predicted = np.array(predicted)
        actual = np.array(actual)
        correlation = np.corrcoef(predicted, actual)[0, 1]

        print(f"Predicted vs Actual Correlation: {correlation:.3f}")
        return predicted, actual

def random_selection(concept_list,num_concepts):
    """Randomly select {num_concepts} from env.concepts
    
    Arguments:
        concept_list: Gymasium environment
        num_concepts: Integer, number of concepts to select
    
    Returns: List of size {num_concepts} of functions,
            each representing a concept"""
    
    total_concepts = len(concept_list)
    idx = np.random.choice(list(range(total_concepts)),num_concepts,replace=False)
    idx = sorted(idx)
    return [concept_list[i] for i in idx], [int(i) for i in idx]

def basic_greedy_selection(concept_list,num_concepts_selected,selection_function,q_estimates,concept_source):
    """Select {num_concepts} greedily
        by first learning the Q(s,a) values from a rollout
        Then selecting the concepts that reduce the standard deviation 
        across all partitions
    
    Arguments:
        env: Gymasium environment
        concept_list: List of functions
        num_concepts_selected: Integer, number of concepts to select
        reference_model: A policy that performs well
            which we are trying to distill
        selection_function: String, whether we're selecting according to 
            Q value, etc."""
    
    all_x_vals = np.array([i[0] for i in q_estimates])
    mean_vals = np.mean(all_x_vals,axis=0)
    mean_vals = mean_vals*(1-mean_vals)
    idx = np.argsort(mean_vals)[-num_concepts_selected:]
    concepts = [concept_list[i] for i in idx]

    return concepts, idx.tolist()


def greedy_selection(concept_list,num_concepts_selected,selection_function,q_estimates,concept_source):
    """Select {num_concepts} greedily
        by first learning the Q(s,a) values from a rollout
        Then selecting the concepts that reduce the standard deviation 
        across all partitions
    
    Arguments:
        env: Gymasium environment
        concept_list: List of functions
        num_concepts_selected: Integer, number of concepts to select
        reference_model: A policy that performs well
            which we are trying to distill
        selection_function: String, whether we're selecting according to 
            Q value, etc."""
    
    unique_actions = list(set([int(i[1]) for i in q_estimates]))

    # Continuous
    if concept_source == "human_selected":
        correlation_by_concept = []
        if selection_function == "q_value":
            for idx in range(len(concept_list)):
                correlations = []
                num_by_action = []

                for action in unique_actions:
                    x_y_pair = [(i[0][idx],i[2]) for i in q_estimates if int(i[1]) == action]
                    x,y = zip(*x_y_pair)
                    num_by_action.append(len(x))
                    if len(x) <= 2:
                        correlations.append(0)
                    else:
                        correlations.append(scipy.stats.pearsonr(x,y).statistic**2)
                avg_correlation = np.sum(np.array(correlations)*np.array(num_by_action))/np.sum(num_by_action)
                correlation_by_concept.append(avg_correlation)
        elif selection_function == "policy":
            for idx in range(len(concept_list)):
                x_y_pair = [(i[0][idx],i[1]) for i in q_estimates]
                x,y = zip(*x_y_pair)
                if len(x) <= 2:
                    correlation_by_concept.append(0)
                else:
                    avg_correlation = scipy.stats.pearsonr(x,y).statistic**2
                    correlation_by_concept.append(avg_correlation)
        correlation_by_concept = np.array(correlation_by_concept)
        idx = np.argpartition(-correlation_by_concept, num_concepts_selected-1)[:num_concepts_selected]
        idx = idx[np.argsort(-correlation_by_concept[idx])]
        idx = np.array(idx).tolist()
        concepts = [concept_list[i] for i in idx]
    else:
        correlation_by_concept = []
        if selection_function == "q_value":
            for idx in range(len(concept_list)):
                total_variance = 0
                for action in unique_actions:
                    x_0 = [i[2] for i in q_estimates if i[0][idx] == 0 and int(i[1]) == action]
                    x_1 = [i[2] for i in q_estimates if i[0][idx] == 1 and int(i[1]) == action]
                    total_variance += np.std(x_0)*len(x_0) + np.std(x_1)*len(x_1)
                correlation_by_concept.append(total_variance)
        elif selection_function == "policy":
            for idx in range(len(concept_list)):
                x_0 = [i[1] for i in q_estimates if i[0][idx] == 0]
                x_1 = [i[1] for i in q_estimates if i[0][idx] == 1]
                total_variance = np.std(x_0)*len(x_0) + np.std(x_1)*len(x_1)
                correlation_by_concept.append(total_variance)
        correlation_by_concept = np.array(correlation_by_concept)
        idx = np.argpartition(correlation_by_concept, num_concepts_selected-1)[:num_concepts_selected]
        idx = idx[np.argsort(correlation_by_concept[idx])]
        idx = np.array([int(i) for i in idx]).tolist()
        concepts = [concept_list[i] for i in idx]
    return concepts, idx

def mutual_information_selection(
    concept_list,
    num_concepts_selected,
    selection_function,
    q_estimates,
    concept_source,
    n_neighbors=5
):
    """
    Greedy concept selection using mutual information.
    """

    unique_actions = list(set(int(i[1]) for i in q_estimates))
    num_concepts = len(concept_list)

    mi_by_concept = []


    for idx in range(num_concepts):
        mi_vals = []
        weights = []

        for action in unique_actions:
            x = np.array([i[0][idx] for i in q_estimates if int(i[1]) == action])
            y = np.array([i[2] for i in q_estimates if int(i[1]) == action])

            if len(np.unique(x)) <= 1 or len(y) <= 2:
                continue

            mi = mutual_info_regression(
                x.reshape(-1, 1),
                y,
                n_neighbors=n_neighbors,
                random_state=0
            )[0]

            mi_vals.append(mi)
            weights.append(len(x))

        if len(mi_vals) == 0:
            mi_by_concept.append(0)
        else:
            mi_by_concept.append(
                np.average(mi_vals, weights=weights)
            )

    mi_by_concept = np.array(mi_by_concept)

    # Select top MI concepts
    idx = np.argpartition(-mi_by_concept, num_concepts_selected - 1)[:num_concepts_selected]
    idx = idx[np.argsort(-mi_by_concept[idx])]
    idx = idx.tolist()

    concepts = [concept_list[i] for i in idx]

    return concepts, idx

def max_prefix_gurobi(final_vals, num_concepts_selected,in_order=True,as_float=False,weighted=False,acc_list=None,min_accuracy=0,fixed_idx=[],has_equality=True):
    """Arguments:
        final_vals: list of tuples (value, elements_covering_value)
                    assumed sorted in decreasing priority (top first)
        num_concepts_selected: budget
    Returns: 
        List of concepet indexes

    """
    
    n = len(final_vals)
    m = max([max(i[1]) for i in final_vals])+1
    
    model = gp.Model("max_prefix_hitting")
    model.Params.OutputFlag = 0  # silence solver
    
    if as_float:
        x = model.addVars(m, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="x")
        y = model.addVars(n, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="y")
    elif weighted: 
        # Binary vars: x[e] = 1 if element e selected
        x = model.addVars(m, vtype=GRB.BINARY, name="x")
        
        # Binary vars: y[i] = 1 if value i is covered
        y = model.addVars(n, lb=0.0, vtype=GRB.CONTINUOUS, name="y")
    else:
        # Binary vars: x[e] = 1 if element e selected
        x = model.addVars(m, vtype=GRB.BINARY, name="x")
        
        # Binary vars: y[i] = 1 if value i is covered
        y = model.addVars(n, vtype=GRB.BINARY, name="y")
    model.update()
    # Link y[i] to coverage by selected elements
    for i, (_, elems) in enumerate(final_vals):
        if elems:  # make sure not empty
            model.addConstr(
                y[i] <= gp.quicksum(x[e] for e in elems),
                name=f"cover_{i}"
            )
        else:
            model.addConstr(y[i] == 0)  # cannot be covered
    
    if len(fixed_idx) > 0:
        for i in fixed_idx:
            model.addConstr(x[i] == 1)

    # Prefix constraints: enforce consecutive coverage
    # y[i] <= y[i-1] for i>0
    if in_order:
        for i in range(1, n):
            model.addConstr(y[i] <= y[i-1], name=f"prefix_{i}")
        
    if has_equality:
        model.addConstr(gp.quicksum(x[i] for i in range(m)) == num_concepts_selected, name="budget")
    else:
        model.addConstr(gp.quicksum(x[i] for i in range(m)) <= num_concepts_selected, name="budget")


    if acc_list is not None:
        model.addConstr(gp.quicksum(x[i]*acc_list[i] for i in range(m)) >= min_accuracy*num_concepts_selected)


    if weighted: 
        model.setObjective(gp.quicksum(y[i]*final_vals[i][0] for i in range(n)), GRB.MAXIMIZE)    
    else:
        model.setObjective(gp.quicksum(y[i] for i in range(n)), GRB.MAXIMIZE)    
    model.optimize()

    selected_elements = [i for i in range(m) if x[i].X > 0.5]
    max_prefix_len = sum(1 for i in range(n) if y[i].X > 0.5)
    return selected_elements, max_prefix_len


def policy_coverage_selection_lp_hybrid(
    ground_truth_gym_env,
    concept_list,
    num_concepts_selected,
    groundtruth_model,
    q_estimates,
    num_pairs_lp=20_000,
    rollout_steps=10_000,
    coverage_ratio=0.99,
    fixed_idx = [],
    prefix=False 
):
    unique_actions = list(set([int(i[1]) for i in q_estimates]))
    actions = np.array([i[1] for i in q_estimates])

    # Continuous
    discretized_X = np.array([i[0] for i in q_estimates])
    
    q_values = np.array([i[2] for i in q_estimates])

    final_vals = []
    num_actions = len(set([i[1] for i in q_estimates]))
    print(num_actions)
    seen = set() 
    for a in unique_actions:
            relevant_idx = np.where(actions == a)[0]
            if len(relevant_idx) <= 500:
                relevant_low = relevant_high = relevant_idx
            else:
                relevant_low = np.argsort(np.abs(q_values))[:500]
                relevant_high = np.argsort(np.abs(q_values))[-500:]
            for low_idx in relevant_low:
                for high_idx in relevant_high:
                    diff = abs(q_values[low_idx] - q_values[high_idx])
                    # tuple of differing concept indices
                    diffs = tuple(i for i, (l, h) in enumerate(zip(discretized_X[low_idx], discretized_X[high_idx])) if l != h)
                    tup = (diff, diffs)
                    if diffs not in seen and diffs != ():
                        seen.add(diffs)
                        final_vals.append(tup)
    final_vals = final_vals[:250_000]
    final_vals = sorted(final_vals,reverse=True)

    # --------------------------------------------------
    # Collect observations / actions (same as before)
    # --------------------------------------------------
    all_observations = []
    all_actions = []

    obs, info = ground_truth_gym_env.reset()

    for _ in range(rollout_steps):
        actions = groundtruth_model.predict(obs)[0]
        for j in range(len(actions)):
            all_observations.append([c(info[j]['observation']) for c in concept_list])
            all_actions.append(actions[j])
        obs, rew, t_1, t_2, info = ground_truth_gym_env.step(actions)

    all_observations = np.asarray(all_observations, dtype=np.int8)
    all_actions = np.asarray(all_actions)

    N, K = all_observations.shape

    print("There are {} observations".format(N))

    # --------------------------------------------------
    # Sample cross-action pairs
    # --------------------------------------------------
    idx_i = np.random.randint(low=0, high=N, size=5 * num_pairs_lp)
    idx_j = np.random.randint(low=0, high=N, size=5 * num_pairs_lp)

    valid = all_actions[idx_i] != all_actions[idx_j]
    idx_i = idx_i[valid][:num_pairs_lp]
    idx_j = idx_j[valid][:num_pairs_lp]

    if len(idx_i) == 0:
        raise ValueError("No cross-action pairs sampled.")

    M = len(idx_i)

    disagreement = (all_observations[idx_i] != all_observations[idx_j]).astype(np.int8)

    # --------------------------------------------------
    # Build LP in Gurobi
    # --------------------------------------------------
    model = gp.Model("max_coverage_lp")
    model.Params.OutputFlag = 0

    ub = 1.0
    len_x_vals = 0
    trials = 0

    # x_d variables (concept selection)
    x = model.addVars(K, lb=0.0, ub=1.0, vtype=GRB.BINARY, name="x")

    # y_p variables (pair covered)
    y = model.addVars(M, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="y")

    if len(fixed_idx) == 0:
        y_2 = model.addVars(len(final_vals), lb=0.0, ub=ub, vtype=GRB.BINARY, name="y_2")
    else:
        y_2 = model.addVars(len(final_vals), lb=0.0, vtype=GRB.CONTINUOUS, name="y_2")

    for i, (_, elems) in enumerate(final_vals):
        if elems:  # make sure not empty
            model.addConstr(
                y_2[i] <= gp.quicksum(x[e] for e in elems),
                name=f"cover_{i}"
            )
        else:
            model.addConstr(y_2[i] == 0)  # cannot be covered
    
    # Prefix constraints: enforce consecutive coverage

    if len(fixed_idx) == 0 and prefix:
        for i in range(1, len(final_vals)):
            model.addConstr(y_2[i] <= y_2[i-1], name=f"prefix_{i}")

    weights = [i[0] for i in final_vals]

    for p in range(M):
        model.addConstr(
            y[p] <= gp.quicksum(disagreement[p, d] * x[d] for d in range(K)),
            name=f"cover_{p}",
        )

    # Cardinality constraint
    model.addConstr(
        gp.quicksum(x[d] for d in range(K)) <= num_concepts_selected,
        name="budget",
    )
    if len(fixed_idx) > 0:
        for i in fixed_idx:
            model.addConstr(x[i] == 1)

    # Constraint: maximize covered pairs
    model.addConstr(gp.quicksum(y[p] for p in range(M))/M >= coverage_ratio)
    
    # if len(fixed_idx) > 0:
    #     model.setObjective(gp.quicksum(y[p] for p in range(M)), GRB.MAXIMIZE)    
    # else:
    model.setObjective(gp.quicksum(weights[i]*y_2[i] for i in range(len(final_vals))), GRB.MAXIMIZE)    

    model.optimize()

    if model.Status not in (GRB.OPTIMAL, GRB.TIME_LIMIT):
        coverage_ratio -= 0.05 
        
        if coverage_ratio < 0:
            return [], [0]
        else:
            return policy_coverage_selection_lp_hybrid(
                ground_truth_gym_env,
                concept_list,
                num_concepts_selected,
                groundtruth_model,
                q_estimates,
                coverage_ratio=coverage_ratio
            )

    # --------------------------------------------------
    # Rounding: take top-k x_d
    # --------------------------------------------------
    x_vals = np.array([x[d].X for d in range(K)])
    y_vals = np.array([y[p].X for p in range(M)])
    len_x_vals = sum(x_vals)


    print("There are {} x vals".format(len_x_vals))
    idx = [i for i in range(len(x_vals)) if x_vals[i] > 0.5]

    if len_x_vals < num_concepts_selected and fixed_idx == []:
        return policy_coverage_selection_lp_hybrid(ground_truth_gym_env,concept_list,
                                                   num_concepts_selected,groundtruth_model,
                                                   q_estimates,fixed_idx=idx)


    subset_concept = [concept_list[i] for i in idx]

    # Optional: compute achieved coverage on LP sample
    covered = disagreement[:, idx].any(axis=1)
    coverage_ratio = covered.mean()

    print("Coverage {}".format(coverage_ratio))

    return subset_concept, idx

def policy_coverage_selection_multiple_log(
    ground_truth_gym_env,
    concept_list,
    num_concepts_selected,
    groundtruth_model,
    q_estimates,
    acc_list,
    num_pairs_lp=20_000,
    rollout_steps=10_000,
    coverage_ratio=0.75,
    fixed_idx=None
):
    if fixed_idx is None:
        fixed_idx = []
    unique_actions = list(set([int(i[1]) for i in q_estimates]))
    actions = np.array([i[1] for i in q_estimates])

    # Continuous
    discretized_X = np.array([i[0] for i in q_estimates])
    
    q_values = np.array([i[2] for i in q_estimates])

    final_vals = []
    num_actions = len(set([i[1] for i in q_estimates]))
    print(num_actions)
    seen = set() 
    for a in unique_actions:
        relevant_idx = np.where(actions == a)[0]

        if len(relevant_idx) <= 500:
            relevant_low = relevant_idx
            relevant_high = relevant_idx
        else:
            # Restrict to this action only
            q_sub = q_values[relevant_idx]

            # Sort by absolute Q-value *within this action*
            order = np.argsort(np.abs(q_sub))

            relevant_low = relevant_idx[order[:500]]
            relevant_high = relevant_idx[order[-500:]]
        for low_idx in relevant_low:
            for high_idx in relevant_high:
                diff = abs(q_values[low_idx] - q_values[high_idx])
                # tuple of differing concept indices
                diffs = tuple(i for i, (l, h) in enumerate(zip(discretized_X[low_idx], discretized_X[high_idx])) if l != h)
                tup = (diff, diffs)
                if diffs not in seen and diffs != ():
                    seen.add(diffs)
                    final_vals.append(tup)
    final_vals = final_vals[:250_000]
    final_vals = sorted(final_vals,reverse=True)

    # --------------------------------------------------
    # Collect observations / actions (same as before)
    # --------------------------------------------------
    all_observations = []
    all_actions = []

    obs, info = ground_truth_gym_env.reset()

    for _ in range(rollout_steps):
        actions = groundtruth_model.predict(obs)[0]
        for j in range(len(actions)):
            all_observations.append([c(info[j]['observation']) for c in concept_list])
            all_actions.append(actions[j])
        obs, rew, t_1, t_2, info = ground_truth_gym_env.step(actions)

    all_observations = np.asarray(all_observations, dtype=np.int8)
    all_actions = np.asarray(all_actions)

    N, K = all_observations.shape

    print("There are {} observations".format(N))

    # --------------------------------------------------
    # Sample cross-action pairs
    # --------------------------------------------------
    idx_i = np.random.randint(low=0, high=N, size=5 * num_pairs_lp)
    idx_j = np.random.randint(low=0, high=N, size=5 * num_pairs_lp)

    valid = all_actions[idx_i] != all_actions[idx_j]
    idx_i = idx_i[valid][:num_pairs_lp]
    idx_j = idx_j[valid][:num_pairs_lp]

    if len(idx_i) == 0:
        raise ValueError("No cross-action pairs sampled.")

    M = len(idx_i)

    disagreement = (all_observations[idx_i] != all_observations[idx_j]).astype(np.int8)

    # --------------------------------------------------
    # Precompute log(1 - p_{p,d})
    # --------------------------------------------------
    log_p = np.zeros((M, K))
    eps = 1e-9

    for p in range(M):
        for d in range(K):
            if disagreement[p, d] == 1:
                prob = acc_list[d]**2 + (1 - acc_list[d])**2
            else:
                prob = 2 * acc_list[d] * (1 - acc_list[d])
            prob = min(prob, 1 - eps)
            log_p[p, d] = np.log(1 - prob)   # ≤ 0

    # --------------------------------------------------
    # Build Gurobi model
    # --------------------------------------------------
    model = gp.Model("max_coverage_log")
    model.Params.OutputFlag = 0

    # Concept selection
    x = model.addVars(K, vtype=GRB.BINARY, name="x")

    # Pair coverage probabilities
    y = model.addVars(M, lb=0.0, ub=1.0, name="y")
    y_2 = model.addVars(len(final_vals), lb=0.0, vtype=GRB.CONTINUOUS, name="y_2")

    eps = 1e-9

    one_minus_y2 = model.addVars(len(final_vals), lb=eps, ub=1.0, name="one_minus_y2")
    z2 = model.addVars(len(final_vals), lb=-GRB.INFINITY, ub=0.0, name="z2")
    s2 = model.addVars(len(final_vals), lb=-GRB.INFINITY, ub=0.0, name="s2")

    for i, (_, elems) in enumerate(final_vals):
        if elems:
            # one_minus_y2 = 1 - y_2
            model.addConstr(one_minus_y2[i] == 1 - y_2[i], name=f"one_minus_y2_{i}")

            # z2 = log(1 - y_2)
            model.addGenConstrLog(one_minus_y2[i], z2[i], name=f"log_y2_{i}")

            # s2 = sum x_e * log(1 - p_e)
            model.addConstr(
                s2[i] == gp.quicksum(
                    x[e] * np.log(1 - (acc_list[e]**2 + (1 - acc_list[e])**2))
                    for e in elems
                ),
                name=f"log_sum_{i}"
            )

            # main probabilistic constraint
            model.addConstr(z2[i] >= s2[i], name=f"cover_{i}")

        else:
            print("Setting to 0")
            model.addConstr(y_2[i] == 0, name=f"cover_{i}")

    for p in range(M):
        model.addConstr(
            y[p] <= gp.quicksum(disagreement[p, d] * x[d] for d in range(K)),
            name=f"cover_{p}",
        )

    # Prefix constraints: enforce consecutive coverage
    # if len(fixed_idx) == 0:
    #     for i in range(1, len(final_vals)):
    #         model.addConstr(y_2[i] <= y_2[i-1], name=f"prefix_{i}")


    model.addConstr(gp.quicksum(y[p] for p in range(M)) / M >= coverage_ratio)

    weights = [i[0] for i in final_vals]

    # Cardinality constraint
    model.addConstr(
        gp.quicksum(x[d] for d in range(K)) <= num_concepts_selected,
        name="budget",
    )
    if len(fixed_idx) > 0:
        for i in fixed_idx:
            model.addConstr(x[i] == 1)

    # Constraint: maximize covered pairs
    
    # if len(fixed_idx) > 0:
    #     model.setObjective(gp.quicksum(y[p] for p in range(M)), GRB.MAXIMIZE)    
    # else:
    model.setObjective(gp.quicksum(weights[i]*y_2[i] for i in range(len(final_vals))), GRB.MAXIMIZE)    

    model.optimize()

    if model.Status not in (GRB.OPTIMAL, GRB.TIME_LIMIT):
        coverage_ratio -= 0.05 
        
        if coverage_ratio < 0:
            return [], [0]
        else:
            return policy_coverage_selection_multiple_log(
                ground_truth_gym_env,
                concept_list,
                num_concepts_selected,
                groundtruth_model,
                q_estimates,
                acc_list,
                coverage_ratio=coverage_ratio
            )

    # --------------------------------------------------
    # Rounding: take top-k x_d
    # --------------------------------------------------
    x_vals = np.array([x[d].X for d in range(K)])
    y_vals = np.array([y[p].X for p in range(M)])
    y_2_vals = np.array([y_2[p].X for p in range(len(y_2))])

    print("Y Vals mean",np.mean(y_vals))
    print("Y 2 Vals maen {}".format(np.mean(y_2_vals)))
    len_x_vals = np.sum(x_vals >  0.5)


    print("There are {} x vals".format(len_x_vals))
    idx = [i for i in range(len(x_vals)) if x_vals[i] > 0.5]

    if len_x_vals < num_concepts_selected and fixed_idx == []:
        return policy_coverage_selection_multiple_log(
            ground_truth_gym_env=ground_truth_gym_env,
            concept_list=concept_list,
            num_concepts_selected=num_concepts_selected,
            groundtruth_model=groundtruth_model,
            q_estimates=q_estimates,
            acc_list=acc_list,
            num_pairs_lp=num_pairs_lp,
            rollout_steps=rollout_steps,
            coverage_ratio=coverage_ratio,
            fixed_idx=idx,
        )


    subset_concept = [concept_list[i] for i in idx]

    # Optional: compute achieved coverage on LP sample
    covered = disagreement[:, idx].any(axis=1)
    coverage_ratio = covered.mean()

    print("Coverage {}".format(coverage_ratio))

    return subset_concept, idx


def basic_greedy_selection_supervised(train_X,train_Y,num_concepts_selected):
    entropy_by_concept = []

    for i in range(len(train_X[0])):
        is_zero = len(train_X[train_X[:,i] == 0])/len(train_X)
        is_one = 1-is_zero 
        entropy = is_zero*is_one 
        entropy_by_concept.append(entropy)
    entropy_by_concept = np.array(entropy_by_concept)
    topk_idx = np.argpartition(-entropy_by_concept, num_concepts_selected)[:num_concepts_selected]
    topk_idx = topk_idx[np.argsort(-entropy_by_concept[topk_idx])]
    return topk_idx.tolist()

def greedy_selection_supervised(
    train_X,
    train_Y,
    num_concepts_selected,
):
    """
    Supervised analogue of greedy_selection.

    Selects concepts that minimize conditional variance of Y
    after splitting on the binary concept.

    Args:
        train_X: (N, d) binary concept matrix
        train_Y: (N,) continuous or discrete labels
        num_concepts_selected: number of concepts to select

    Returns:
        idx: indices of selected concepts
    """

    N, d = train_X.shape
    conditional_variance = []

    for i in range(d):
        mask0 = train_X[:, i] == 0
        mask1 = ~mask0

        var = 0.0

        if mask0.any():
            var += np.var(train_Y[mask0]) * mask0.sum()
        if mask1.any():
            var += np.var(train_Y[mask1]) * mask1.sum()

        # Normalize by total samples (optional, ranking unaffected)
        conditional_variance.append(var / N)

    conditional_variance = np.array(conditional_variance)

    # Select concepts that minimize conditional variance
    idx = np.argpartition(conditional_variance, num_concepts_selected)[:num_concepts_selected]
    idx = idx[np.argsort(conditional_variance[idx])]

    return idx.tolist()


def lp_selection_supervised(train_matrix,labels,num_concepts):
    """Select {num_concepts} greedily
        by selecting those that reduce the reward range within each partition
        For example, first select the concept
            so that, c_{i} = 0 and c_{i} = 1 each have
                small differences between max and min reward

    Arguments:
        env: Gymasium environment
        num_concepts: Integer, number of concepts to select
    
    Returns: List of size {num_concepts} of integers
        each representing a concept"""
    train_X = np.asarray(train_matrix)
    labels = np.asarray(labels)
    # Convert rows to tuples to make them hashable
    rows_as_tuples = [tuple(row) for row in train_X]
    row_counts = Counter(rows_as_tuples)   # counts of each unique row

    unique_rows = np.array(list(row_counts.keys()))
    unique_counts = np.array([row_counts[r] for r in row_counts])
    unique_labels = np.array([labels[np.all(train_X == r, axis=1)][0] for r in unique_rows])

    pairs = [
        (i, j) 
        for i, j in combinations(range(len(unique_rows)), 2) 
        if unique_labels[i] != unique_labels[j]
    ]

    per_train_constraint_weighted = []

    for i, j in pairs:
        elems_diff = np.where(unique_rows[i] != unique_rows[j])[0].tolist()
        weight = unique_counts[i] * unique_counts[j]   # multiplicity weight
        per_train_constraint_weighted.append((weight, elems_diff))
    
    selected_elements, _ = max_prefix_gurobi(per_train_constraint_weighted, num_concepts, in_order=False,as_float=False,weighted=True)

    return selected_elements

def multiple_log_selection_supervised(
    train_matrix,
    labels,
    concept_accuracy,
    num_concepts,
):
    """
    Supervised LP with probabilistic coverage.
    Minimal extension of lp_selection_supervised.
    """

    train_X = np.asarray(train_matrix)
    labels = np.asarray(labels)
    # Convert rows to tuples to make them hashable
    rows_as_tuples = [tuple(row) for row in train_X]
    row_counts = Counter(rows_as_tuples)   # counts of each unique row

    unique_rows = np.array(list(row_counts.keys()))
    unique_counts = np.array([row_counts[r] for r in row_counts])
    unique_labels = np.array([labels[np.all(train_X == r, axis=1)][0] for r in unique_rows])

    pairs = [
        (i, j) 
        for i, j in combinations(range(len(unique_rows)), 2) 
        if unique_labels[i] != unique_labels[j]
    ]

    per_train_constraint_weighted = []

    for i, j in pairs:
        elems_diff = np.where(unique_rows[i] != unique_rows[j])[0].tolist()
        weight = unique_counts[i] * unique_counts[j]   # multiplicity weight
        per_train_constraint_weighted.append((weight, elems_diff))
    final_vals = per_train_constraint_weighted
    n = len(final_vals)
    m = max([max(i[1]) for i in final_vals])+1
    
    model = gp.Model("max_prefix_hitting")
    model.Params.OutputFlag = 0  # silence solver

    # Binary vars: x[e] = 1 if element e selected
    x = model.addVars(m, vtype=GRB.BINARY, name="x")
    
    # Binary vars: y[i] = 1 if value i is covered
    y = model.addVars(n, lb=0.0, vtype=GRB.CONTINUOUS, name="y")
    model.update()
    # Link y[i] to coverage by selected elements
    for i, (_, elems) in enumerate(final_vals):
        if elems:  # make sure not empty
            model.addConstr(
                y[i] <= gp.quicksum(x[e]*concept_accuracy[e] for e in elems),
                name=f"cover_{i}"
            )
        else:
            model.addConstr(y[i] == 0)  # cannot be covered
    
    model.addConstr(gp.quicksum(x[i] for i in range(m)) == num_concepts, name="budget")

    model.setObjective(gp.quicksum(y[i]*final_vals[i][0] for i in range(n)), GRB.MAXIMIZE)    
    model.optimize()

    selected_elements = [i for i in range(m) if x[i].X > 0.5]

    return selected_elements

