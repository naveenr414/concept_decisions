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

def max_prefix_gurobi(final_vals, num_concepts_selected,in_order=True,as_float=False,weighted=False,acc_list=None,min_accuracy=0,fixed_idx=[]):
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


def lp_based_selection_old(env,concept_list,num_concepts_selected,selection_function,q_estimates,concept_source):
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
    actions = np.array([i[1] for i in q_estimates])

    # Continuous
    if concept_source == "human_selected":
        all_concepts = np.array([i[0] for i in q_estimates])
        median_by_column = np.median(all_concepts,axis=0)
        discretized_X = (all_concepts < median_by_column).astype(np.int8)
    else:
        discretized_X = np.array([i[0] for i in q_estimates])
    
    if selection_function == "q_value":
        q_values = np.array([i[2] for i in q_estimates])

        final_vals = []
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
        if len(final_vals) > 50_000:
            final_vals = random.sample(final_vals,50_000)

        final_vals = sorted(final_vals,reverse=True)
        idx, _ = max_prefix_gurobi(final_vals,num_concepts_selected)
        concepts = [concept_list[i] for i in idx]
    elif selection_function == "policy":
        sample_by_action = []

        for a in unique_actions:
            X_reduced = discretized_X[actions == a]
            sample_by_action.append(X_reduced)
        weights = [len(lst) for lst in sample_by_action]

        pairs = []
        for _ in range(10_000):
            i, j = random.choices(range(len(sample_by_action)), weights=weights, k=2)
            while i == j: 
                weights_non_action = [weights[j] for j in range(len(weights)) if j!=i]
                num_non_action = [i for i in range(len(sample_by_action)) if i!=j]
                if len(num_non_action) == 0:
                    break 
                j = random.choices(num_non_action, weights=weights_non_action, k=1)[0]
            a = random.choice(sample_by_action[i])
            b = random.choice(sample_by_action[j])
            pairs.append([i for i in range(len(a)) if a[i] != b[i]])
        pairs = [(0,i) for i in pairs]
        pairs = [i for i in pairs if len(i[1])>0]
        idx, _ = max_prefix_gurobi(pairs,num_concepts_selected,in_order=False)
        idx = [int(i) for i in idx]
        idx = np.array(idx).tolist()
        concepts = [concept_list[i] for i in idx]

    if len(idx) == 0:
        return [concept_list[0]],[0]

    return concepts, idx

def lp_based_selection(env,concept_list,num_concepts_selected,selection_function,q_estimates,concept_source):
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
    actions = np.array([i[1] for i in q_estimates])

    # Continuous
    if concept_source == "human_selected":
        all_concepts = np.array([i[0] for i in q_estimates])
        median_by_column = np.median(all_concepts,axis=0)
        discretized_X = (all_concepts < median_by_column).astype(np.int8)
    else:
        discretized_X = np.array([i[0] for i in q_estimates])
    
    if selection_function == "q_value":
        q_values = np.array([i[2] for i in q_estimates])

        final_vals = []
        num_actions = len(set([i[1] for i in q_estimates]))
        print(num_actions)
        state_diff = []
        for i in range(0,len(q_estimates),num_actions):
            temp = []
            for j in range(num_actions):
                for j_prime in range(j):
                    temp.append(q_estimates[i+j][2]-q_estimates[i+j_prime][2])
            state_diff.append((q_estimates[i][0],np.array(temp)))
        gap_diffs = []
        final_gap_diffs = []
        seen = set() 
        for i in range(len(state_diff)):
            if i%100 == 0:
                print(i)
            for j in range(len(state_diff)):
                things_diff = [k for k in range(len(state_diff[j][0])) if state_diff[j][0][k] != state_diff[i][0][k]]
                score = np.max(np.abs(state_diff[i][1]-state_diff[j][1]))
                if len(things_diff) > 0 and score > 0:
                    gap_diffs.append((score,things_diff))
        gap_diffs = sorted(gap_diffs,reverse=True,key=lambda k: k[0])
        for i in gap_diffs:
            if tuple(i[1]) in seen:
                continue 
            else:
                final_gap_diffs.append(i)
                seen.add(tuple(i[1]))
        gap_diffs = final_gap_diffs
        final_vals = gap_diffs[:100_000]
        print("Final vals {}".format(len(final_vals)))
        idx, num_covered = max_prefix_gurobi(final_vals,num_concepts_selected)
        if len(idx) < num_concepts_selected:
            idx, num_covered = max_prefix_gurobi(final_vals,num_concepts_selected,weighted=True,fixed_idx=idx)
        print("Idx {}".format(idx))
        print("{} covered out of {}".format(num_covered,len(final_vals)))
        concepts = [concept_list[i] for i in idx]
    elif selection_function == "policy":
        sample_by_action = []

        for a in unique_actions:
            X_reduced = discretized_X[actions == a]
            sample_by_action.append(X_reduced)
        weights = [len(lst) for lst in sample_by_action]

        pairs = []
        for _ in range(10_000):
            i, j = random.choices(range(len(sample_by_action)), weights=weights, k=2)
            while i == j: 
                weights_non_action = [weights[j] for j in range(len(weights)) if j!=i]
                num_non_action = [i for i in range(len(sample_by_action)) if i!=j]
                if len(num_non_action) == 0:
                    break 
                j = random.choices(num_non_action, weights=weights_non_action, k=1)[0]
            a = random.choice(sample_by_action[i])
            b = random.choice(sample_by_action[j])
            pairs.append([i for i in range(len(a)) if a[i] != b[i]])
        pairs = [(0,i) for i in pairs]
        pairs = [i for i in pairs if len(i[1])>0]
        idx, _ = max_prefix_gurobi(pairs,num_concepts_selected,in_order=False)
        idx = [int(i) for i in idx]
        idx = np.array(idx).tolist()
        concepts = [concept_list[i] for i in idx]

    if len(idx) == 0:
        return [concept_list[0]],[0]

    return concepts, idx


def solve_exp_relaxation_constant(final_vals, acc_list, num_concepts_selected, 
                                  time_limit=10*60,ub=1):
    """
    Faster surrogate for the constant-accuracy case.
    Version optimized for speed + with PWL in direct form.
    """
    # Validate
    if len(set(acc_list)) != 1:
        raise ValueError("acc_list must be constant")
    
    acc = acc_list[0]
    w = -math.log(max(1e-12, 1.0 - acc))
    n = len(final_vals)
    m = len(acc_list)
    
    # Precompute global max_t for all i
    max_D = max(len(D) for _, D in final_vals)
    global_max_t = w * max_D
    
    # Reduce PWL points for faster solve (3 points often sufficient)
    t_points = [0, 0.5 * global_max_t, global_max_t]
    u_points = [math.exp(-tp) for tp in t_points]
    
    model = gp.Model("exp_relax_constant_fast")
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = time_limit
    model.Params.PreCrush = 1
    model.Params.MIPFocus = 1  # Focus on finding good feasible solutions quickly
    model.Params.Threads = 0   # Use all available threads (default, but explicit)
    
    # Variables - create all at once for efficiency
    x = model.addVars(m, vtype=GRB.BINARY, name="x")
    
    # Only create t, u, y for indices with non-empty D
    active_indices = [i for i, (_, D) in enumerate(final_vals) if len(D) > 0]
    
    t = model.addVars(active_indices, vtype=GRB.CONTINUOUS, 
                      lb=0.0, ub=global_max_t, name="t")
    u = model.addVars(active_indices, vtype=GRB.CONTINUOUS, 
                      lb=0.0, ub=1.0, name="u")
    y = model.addVars(active_indices, vtype=GRB.CONTINUOUS, 
                      lb=0.0, ub=ub, name="y")
    
    # Build constraint expressions more efficiently
    # Batch constraints together where possible
    for i in active_indices:
        _, D = final_vals[i]
        
        # t[i] = w * Σ_{j in D} x_j
        model.addConstr(t[i] == w * gp.quicksum(x[j] for j in D), 
                        name=f"tdef_{i}")
        
        # PWL approximation
        model.addGenConstrPWL(t[i], u[i], t_points, u_points, 
                              name=f"exp_pwl_{i}")
        
        # y[i] = 1 - u[i]
        model.addConstr(y[i] == 1 - u[i], name=f"ydef_{i}")
    
    # Budget constraint
    model.addConstr(gp.quicksum(x) <= num_concepts_selected)
    
    # Objective - only sum over active indices
    obj_expr = gp.quicksum(y[i] * final_vals[i][0] for i in active_indices)
    
    # Add contribution from empty D cases (they contribute final_vals[i][0] * 0)
    # Actually, skip them entirely - they don't contribute to objective
    
    model.setObjective(obj_expr, GRB.MAXIMIZE)
    
    model.optimize()
    
    # Extract solution
    selected = [j for j in range(m) if x[j].X > 0.5]
    
    if len(selected) < num_concepts_selected:
        fracs = sorted(
            [(j, x[j].X) for j in range(m) if j not in selected],
            key=lambda a: a[1],
            reverse=True
        )
        needed = num_concepts_selected - len(selected)
        selected.extend([j for j, _ in fracs[:needed]])
    
    return selected[:num_concepts_selected]

def solve_exp_relaxation(final_vals, acc_list, num_concepts_selected, time_limit=10*60):
    """
    Correct exponential surrogate:
      y_i = 1 - exp(- sum_j w_j x_j )
    using Gurobi PWL approximation - optimized for speed.
    """
    n = len(final_vals)
    m = len(acc_list)
    
    # Precompute weights once
    w = [-math.log(max(1e-12, 1.0 - acc_list[j])) for j in range(m)]
    
    # Precompute which concepts appear in which pairs (for sparsity)
    concept_usage = [False] * m
    active_indices = []
    for i, (_, D) in enumerate(final_vals):
        if len(D) > 0:
            active_indices.append(i)
            for j in D:
                concept_usage[j] = True
    
    # Compute reasonable bounds for z based on actual data
    max_t = max(sum(w[j] for j in D) for _, D in final_vals if len(D) > 0) if active_indices else 0
    min_z = -max_t
    
    # Adaptive PWL breakpoints based on actual range
    if min_z < -10:
        breakpoints = [min_z, -8, -6, -4, -2, 0]
    else:
        breakpoints = [min_z, min_z/2, -2, -1, 0]
    exp_vals = [math.exp(z) for z in breakpoints]
    
    model = gp.Model("exp_relax_fixed")
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = time_limit
    model.Params.MIPFocus = 1  # Focus on finding good solutions
    model.Params.Presolve = 2  # Aggressive presolve
    model.Params.Cuts = 2      # Aggressive cuts
    
    # Only create x variables for concepts that are actually used
    used_concepts = [j for j in range(m) if concept_usage[j]]
    x = model.addVars(used_concepts, vtype=GRB.BINARY, name="x")
    
    # Add unused concepts with fixed value 0 (for budget constraint)
    unused_concepts = [j for j in range(m) if not concept_usage[j]]
    if unused_concepts:
        x.update(model.addVars(unused_concepts, vtype=GRB.BINARY, 
                               ub=0, name="x_unused"))
    
    # Only create variables for active pairs
    z = model.addVars(active_indices, lb=min_z, ub=0, 
                      vtype=GRB.CONTINUOUS, name="z")
    u = model.addVars(active_indices, lb=math.exp(min_z), ub=1.0, 
                      vtype=GRB.CONTINUOUS, name="u")
    y = model.addVars(active_indices, lb=0.0, ub=1.0, 
                      vtype=GRB.CONTINUOUS, name="y")
    
    # Build constraints more efficiently
    model.update()  # Process variable additions before constraints
    
    for i in active_indices:
        _, D = final_vals[i]
        
        # z_i = -sum_{j in D} w_j x_j
        # Combined constraint (skip intermediate t variable)
        if len(D) > 0:
            model.addConstr(
                z[i] == -gp.quicksum(w[j] * x[j] for j in D),
                name=f"zdef_{i}"
            )
        else:
            model.addConstr(z[i] == 0.0, name=f"zdef_empty_{i}")
        
        # PWL approximation: u_i ≈ exp(z_i)
        model.addGenConstrPWL(z[i], u[i], breakpoints, exp_vals, 
                              name=f"exp_{i}")
        
        # y_i = 1 - u_i
        model.addConstr(y[i] == 1 - u[i], name=f"ydef_{i}")
    
    # Budget constraint
    model.addConstr(gp.quicksum(x.values()) <= num_concepts_selected, 
                    name="budget")
    
    # Objective: maximize sum_i d_i * y_i (only over active indices)
    obj_expr = gp.quicksum(y[i] * final_vals[i][0] for i in active_indices)
    model.setObjective(obj_expr, GRB.MAXIMIZE)
    
    print("Optimizing")
    model.optimize()
    
    # Check if solution was found
    if model.SolCount == 0:
        print(f"No solution found within time limit. Status: {model.Status}")
        # Return greedy fallback or empty list
        return []
    
    # Extract selection
    selected = [j for j in range(m) if j in x and x[j].X > 0.5]
    
    # Enforce exactly num_concepts_selected
    if len(selected) < num_concepts_selected:
        fracs = sorted(
            [(j, x[j].X) for j in range(m) if j in x and j not in selected],
            key=lambda a: a[1],
            reverse=True
        )
        needed = num_concepts_selected - len(selected)
        selected.extend([j for j, _ in fracs[:needed]])
    
    return selected[:num_concepts_selected]

def multiple_lp_selection_old(env,concept_list,num_concepts_selected,selection_function,q_estimates,concept_source,acc_list):
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
    actions = np.array([i[1] for i in q_estimates])

    # Continuous
    if concept_source == "human_selected":
        all_concepts = np.array([i[0] for i in q_estimates])
        median_by_column = np.median(all_concepts,axis=0)
        discretized_X = (all_concepts < median_by_column).astype(np.int8)
    else:
        discretized_X = np.array([i[0] for i in q_estimates])
    
    if selection_function == "q_value":
        q_values = np.array([i[2] for i in q_estimates])

        final_vals = []
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
        if len(final_vals) > 50_000:
            final_vals = random.sample(final_vals,50_000)

        final_vals = sorted(final_vals,reverse=True)
        print("Final vals {}".format(len(final_vals)))
        if len(set(acc_list)) == 1:
            idx = solve_exp_relaxation_constant(final_vals, acc_list, num_concepts_selected)
        else:
            idx = solve_exp_relaxation(final_vals, acc_list, num_concepts_selected)
        concepts = [concept_list[i] for i in idx]
    elif selection_function == "policy":
        sample_by_action = []

        for a in unique_actions:
            X_reduced = discretized_X[actions == a]
            sample_by_action.append(X_reduced)
        weights = [len(lst) for lst in sample_by_action]

        pairs = []
        for i in range(100):
            i, j = random.choices(range(len(sample_by_action)), weights=weights, k=2)
            while i == j: 
                weights_non_action = [weights[j] for j in range(len(weights)) if j!=i]
                num_non_action = [i for i in range(len(sample_by_action)) if i!=j]
                if len(num_non_action) == 0:
                    break 
                j = random.choices(num_non_action, weights=weights_non_action, k=1)[0]
            if len(num_non_action) == 0:
                break 
            a = random.choice(sample_by_action[i])
            b = random.choice(sample_by_action[j])
            pairs.append([i for i in range(len(a)) if a[i] != b[i]])
        pairs = [(0,i) for i in pairs]
        idx, _ = max_prefix_gurobi(pairs,num_concepts_selected,in_order=False)
        idx = [int(i) for i in idx]
        idx = np.array(idx).tolist()
        concepts = [concept_list[i] for i in idx]

    if len(idx) == 0:
        return [concept_list[0]],[0]

    return concepts, idx

def multiple_lp_selection(env,concept_list,num_concepts_selected,selection_function,q_estimates,concept_source,acc_list):
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
    actions = np.array([i[1] for i in q_estimates])

    # Continuous
    if concept_source == "human_selected":
        all_concepts = np.array([i[0] for i in q_estimates])
        median_by_column = np.median(all_concepts,axis=0)
        discretized_X = (all_concepts < median_by_column).astype(np.int8)
    else:
        discretized_X = np.array([i[0] for i in q_estimates])
    
    if selection_function == "q_value":
        q_values = np.array([i[2] for i in q_estimates])

        final_vals = []
        num_actions = len(set([i[1] for i in q_estimates]))
        print(num_actions)
        state_diff = []
        for i in range(0,len(q_estimates),num_actions):
            temp = []
            for j in range(num_actions):
                for j_prime in range(j):
                    temp.append(q_estimates[i+j][2]-q_estimates[i+j_prime][2])
            state_diff.append((q_estimates[i][0],np.array(temp)))
        gap_diffs = []
        final_gap_diffs = []
        seen = set() 
        for i in range(len(state_diff)):
            if i%100 == 0:
                print(i)
            for j in range(len(state_diff)):
                things_diff = [k for k in range(len(state_diff[j][0])) if state_diff[j][0][k] != state_diff[i][0][k]]
                score = np.max(np.abs(state_diff[i][1]-state_diff[j][1]))
                if len(things_diff) > 0 and score > 0:
                    gap_diffs.append((score,things_diff))
        gap_diffs = sorted(gap_diffs,reverse=True,key=lambda k: k[0])
        for i in gap_diffs:
            if tuple(i[1]) in seen:
                continue 
            else:
                final_gap_diffs.append(i)
                seen.add(tuple(i[1]))
        gap_diffs = final_gap_diffs
        final_vals = gap_diffs
        print("Final vals {}".format(len(final_vals)))
        if len(set(acc_list)) == 1:
            idx = solve_exp_relaxation_constant(final_vals, acc_list, num_concepts_selected)
        else:
            idx = solve_exp_relaxation(final_vals, acc_list, num_concepts_selected)
        concepts = [concept_list[i] for i in idx]
    elif selection_function == "policy":
        sample_by_action = []

        for a in unique_actions:
            X_reduced = discretized_X[actions == a]
            sample_by_action.append(X_reduced)
        weights = [len(lst) for lst in sample_by_action]

        pairs = []
        for i in range(100):
            i, j = random.choices(range(len(sample_by_action)), weights=weights, k=2)
            while i == j: 
                weights_non_action = [weights[j] for j in range(len(weights)) if j!=i]
                num_non_action = [i for i in range(len(sample_by_action)) if i!=j]
                if len(num_non_action) == 0:
                    break 
                j = random.choices(num_non_action, weights=weights_non_action, k=1)[0]
            if len(num_non_action) == 0:
                break 
            a = random.choice(sample_by_action[i])
            b = random.choice(sample_by_action[j])
            pairs.append([i for i in range(len(a)) if a[i] != b[i]])
        pairs = [(0,i) for i in pairs]
        idx, _ = max_prefix_gurobi(pairs,num_concepts_selected,in_order=False)
        idx = [int(i) for i in idx]
        idx = np.array(idx).tolist()
        concepts = [concept_list[i] for i in idx]

    if len(idx) == 0:
        return [concept_list[0]],[0]

    return concepts, idx


def policy_coverage_selection_lp(
    ground_truth_gym_env,
    concept_list,
    num_concepts_selected,
    groundtruth_model,
    num_pairs_lp=20_000,
    rollout_steps=1_000,
):
    rng = np.random.default_rng()

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
    idx_i = rng.integers(0, N, size=5 * num_pairs_lp)
    idx_j = rng.integers(0, N, size=5 * num_pairs_lp)

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
    while len_x_vals < num_concepts_selected and trials < 5:

        # x_d variables (concept selection)
        x = model.addVars(K, lb=0.0, ub=1.0, vtype=GRB.BINARY, name="x")

        # y_p variables (pair covered)
        y = model.addVars(M, lb=0.0, ub=ub, vtype=GRB.CONTINUOUS, name="y")

        # Coverage constraints: y_p <= sum_d A[p,d] * x_d
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

        # Objective: maximize covered pairs
        model.setObjective(gp.quicksum(y[p] for p in range(M)), GRB.MAXIMIZE)

        model.optimize()

        if model.Status not in (GRB.OPTIMAL, GRB.TIME_LIMIT):
            raise RuntimeError("LP did not solve successfully")

        # --------------------------------------------------
        # Rounding: take top-k x_d
        # --------------------------------------------------
        x_vals = np.array([x[d].X for d in range(K)])
        len_x_vals = sum(x_vals)
        trials += 1
        ub += 1

    idx = np.argsort(x_vals)[-num_concepts_selected:].tolist()
    idx = list(sorted(idx))

    subset_concept = [concept_list[i] for i in idx]

    # Optional: compute achieved coverage on LP sample
    covered = disagreement[:, idx].any(axis=1)
    coverage_ratio = covered.mean()

    print("Coverage {}".format(coverage_ratio))

    return subset_concept, idx

def policy_coverage_selection_lp_hybrid(
    ground_truth_gym_env,
    concept_list,
    num_concepts_selected,
    groundtruth_model,
    q_estimates,
    num_pairs_lp=20_000,
    rollout_steps=10_000,
    coverage_ratio=0.95,
    fixed_idx = []
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
    weights = [i[0] for i in final_vals]

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

    if len(fixed_idx) > 0:
        y_2 = model.addVars(len(final_vals), lb=0.0, vtype=GRB.CONTINUOUS, name="y_2")
    else:
        y_2 = model.addVars(len(final_vals), lb=0.0, ub=ub, vtype=GRB.BINARY, name="y_2")

    for i, (_, elems) in enumerate(final_vals):
        if elems:  # make sure not empty
            model.addConstr(
                y_2[i] <= gp.quicksum(x[e] for e in elems),
                name=f"cover_{i}"
            )
        else:
            model.addConstr(y_2[i] == 0)  # cannot be covered
    
    # Prefix constraints: enforce consecutive coverage
    # y[i] <= y[i-1] for i>0
    if len(fixed_idx) == 0:
        for i in range(1, len(final_vals)):
            model.addConstr(y_2[i] <= y_2[i-1], name=f"prefix_{i}")

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
                num_pairs_lp=20_000,
                rollout_steps=1_000,
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

    if len_x_vals < num_concepts_selected:
        return policy_coverage_selection_lp_hybrid(ground_truth_gym_env,concept_list,
                                                   num_concepts_selected,groundtruth_model,
                                                   q_estimates,fixed_idx=idx)


    subset_concept = [concept_list[i] for i in idx]

    # Optional: compute achieved coverage on LP sample
    covered = disagreement[:, idx].any(axis=1)
    coverage_ratio = covered.mean()

    print("Coverage {}".format(coverage_ratio))

    return subset_concept, idx


def policy_coverage_selection_lp_weighted(
    ground_truth_gym_env,
    concept_list,
    num_concepts_selected,
    groundtruth_model,
    q_estimates,
    num_pairs_lp=20_000,
    rollout_steps=1_000,
    coverage_ratio=0.95,
):
    rng = np.random.default_rng()

    unique_actions = list(set([int(i[1]) for i in q_estimates]))
    actions = np.array([i[1] for i in q_estimates])

    # Continuous
    discretized_X = np.array([i[0] for i in q_estimates])
    
    q_values = np.array([i[2] for i in q_estimates])

    final_vals = []
    num_actions = len(set([i[1] for i in q_estimates]))
    print(num_actions)
    state_diff = []
    for i in range(0,len(q_estimates),num_actions):
        temp = []
        for j in range(num_actions):
            for j_prime in range(j):
                temp.append(q_estimates[i+j][2]-q_estimates[i+j_prime][2])
        state_diff.append((q_estimates[i][0],np.array(temp)))
    gap_diffs = []
    final_gap_diffs = []
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
    weights = [i[0] for i in final_vals]

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
    idx_i = rng.integers(0, N, size=5 * num_pairs_lp)
    idx_j = rng.integers(0, N, size=5 * num_pairs_lp)

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

    y_2 = model.addVars(len(final_vals), lb=0.0,vtype=GRB.CONTINUOUS, name="y_2")

    for i, (_, elems) in enumerate(final_vals):
        if elems:  # make sure not empty
            model.addConstr(
                y_2[i] <= gp.quicksum(x[e] for e in elems),
                name=f"cover_{i}"
            )
        else:
            model.addConstr(y_2[i] == 0)  # cannot be covered
    
    # Prefix constraints: enforce consecutive coverage
    # y[i] <= y[i-1] for i>0

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
    # Constraint: maximize covered pairs
    model.addConstr(gp.quicksum(y[p] for p in range(M))/M >= coverage_ratio)
    model.setObjective(gp.quicksum(weights[i]*y_2[i] for i in range(len(final_vals))), GRB.MAXIMIZE)    
    model.optimize()

    if model.Status not in (GRB.OPTIMAL, GRB.TIME_LIMIT):
        coverage_ratio -= 0.05 
        
        if coverage_ratio < 0:
            return [], [0]
        else:
            return policy_coverage_selection_lp_weighted(
                ground_truth_gym_env,
                concept_list,
                num_concepts_selected,
                groundtruth_model,
                q_estimates,
                num_pairs_lp=20_000,
                rollout_steps=1_000,
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


    subset_concept = [concept_list[i] for i in idx]

    # Optional: compute achieved coverage on LP sample
    covered = disagreement[:, idx].any(axis=1)
    coverage_ratio = covered.mean()

    print("Coverage {}".format(coverage_ratio))

    return subset_concept, idx



def policy_coverage_selection_lp_advantage(
    ground_truth_gym_env,
    concept_list,
    num_concepts_selected,
    groundtruth_model,
    gamma=0.995,
    num_pairs_lp=20_000,
    rollout_steps=1000,
    device="cpu"
):
    rng = np.random.default_rng()
    td_learner, _ = rollout_q_estimates_td(groundtruth_model, ground_truth_gym_env, concept_list,get_td_learner=True)

    # --------------------------------------------------
    # Collect observations / actions (same as before)
    # --------------------------------------------------
    
    all_observations = []
    all_actions = []
    delta_list_per_state = []

    obs, info = ground_truth_gym_env.reset()
    print(info)
    for t in range(rollout_steps):
        print("On rollout {}".format(t))
        actions = groundtruth_model.predict(obs)[0]
        values = []
        delta_val = td_learner.q_net(torch.tensor([[c(i['observation']) for c in concept_list] for i in info]).float()).detach().cpu().numpy()
        binary_obs = [[c(o) for c in concept_list] for o in [i['observation'] for i in info]]

        for i in range(8):
            all_observations.append(binary_obs[i]) 
            all_actions.append(actions[i])    

            temp = []
            for j in range(len(delta_val[i])):
                for k in range(j):
                    temp.append(delta_val[i][j]-delta_val[i][k])
            delta_list_per_state.append(np.array(temp))

        # Step the main env
        obs, _, _, _, info = ground_truth_gym_env.step(actions)
    all_observations = np.asarray(all_observations, dtype=np.int8)
    all_actions = np.asarray(all_actions)
    delta_list_per_state = np.array(delta_list_per_state)

    N, K = all_observations.shape

    print("There are {} observations".format(N))

    # --------------------------------------------------
    # Sample cross-action pairs
    # --------------------------------------------------
    idx_i = rng.integers(0, N, size=5 * num_pairs_lp)
    idx_j = rng.integers(0, N, size=5 * num_pairs_lp)

    valid = all_actions[idx_i] != all_actions[idx_j]
    idx_i = idx_i[valid][:num_pairs_lp]
    idx_j = idx_j[valid][:num_pairs_lp]

    if len(idx_i) == 0:
        raise ValueError("No cross-action pairs sampled.")

    M = len(idx_i)

    disagreement = (all_observations[idx_i] != all_observations[idx_j]).astype(np.int8)
    delta_array = np.max(np.abs(np.array([delta_list_per_state[i] for i in idx_i])
                            -np.array([delta_list_per_state[j] for j in idx_j])),axis=1)

    # --------------------------------------------------
    # Build LP in Gurobi
    # --------------------------------------------------
    model = gp.Model("max_coverage_lp")
    model.Params.OutputFlag = 0

    ub = 1.0
    len_x_vals = 0
    trials = 0
    while len_x_vals < num_concepts_selected and trials < 5:

        # x_d variables (concept selection)
        x = model.addVars(K, lb=0.0, ub=1.0, vtype=GRB.BINARY, name="x")

        # y_p variables (pair covered)
        y = model.addVars(M, lb=0.0, ub=ub, vtype=GRB.CONTINUOUS, name="y")

        # Coverage constraints: y_p <= sum_d A[p,d] * x_d
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

        # Objective: maximize covered pairs
        model.setObjective(gp.quicksum(delta_array[p]*y[p] for p in range(M)), GRB.MAXIMIZE)

        model.optimize()

        if model.Status not in (GRB.OPTIMAL, GRB.TIME_LIMIT):
            raise RuntimeError("LP did not solve successfully")

        # --------------------------------------------------
        # Rounding: take top-k x_d
        # --------------------------------------------------
        x_vals = np.array([x[d].X for d in range(K)])
        len_x_vals = sum(x_vals)
        trials += 1
        ub += 1

    idx = np.argsort(x_vals)[-num_concepts_selected:].tolist()
    idx = list(sorted(idx))

    subset_concept = [concept_list[i] for i in idx]

    # Optional: compute achieved coverage on LP sample
    covered = disagreement[:, idx].any(axis=1)
    coverage_ratio = covered.mean()

    print("Coverage {}".format(coverage_ratio))

    return subset_concept, idx


def policy_coverage_selection_exp_lp(
    ground_truth_gym_env,
    concept_list,
    acc_list,
    num_concepts_selected,
    groundtruth_model,
    num_pairs_lp=20_000,
    rollout_steps=1000,
    relax_x=False,
):
    rng = np.random.default_rng()

    # --------------------------------------------------
    # Rollout data collection
    # --------------------------------------------------
    all_obs = []
    all_actions = []

    obs, info = ground_truth_gym_env.reset()
    for _ in range(rollout_steps):
        actions = groundtruth_model.predict(obs)[0]
        for j in range(len(actions)):
            all_obs.append([c(info[j]['observation']) for c in concept_list])
            all_actions.append(actions[j])
        obs, _, _, _, info = ground_truth_gym_env.step(actions)

    all_obs = np.asarray(all_obs, dtype=np.int8)
    all_actions = np.asarray(all_actions)

    N, K = all_obs.shape

    # --------------------------------------------------
    # Sample cross-action pairs
    # --------------------------------------------------
    idx_i = rng.integers(0, N, size=5 * num_pairs_lp)
    idx_j = rng.integers(0, N, size=5 * num_pairs_lp)

    valid = all_actions[idx_i] != all_actions[idx_j]
    idx_i = idx_i[valid][:num_pairs_lp]
    idx_j = idx_j[valid][:num_pairs_lp]

    if len(idx_i) == 0:
        raise ValueError("No cross-action pairs sampled")

    disagreement = (all_obs[idx_i] != all_obs[idx_j]).astype(np.int8)
    M = disagreement.shape[0]

    # --------------------------------------------------
    # Precompute weights
    # --------------------------------------------------
    w = np.array([-math.log(max(1e-12, 1.0 - acc_list[d])) for d in range(K)])

    # Identify used concepts (sparsity)
    used_concepts = np.where(disagreement.sum(axis=0) > 0)[0]

    # --------------------------------------------------
    # PWL range estimation
    # --------------------------------------------------
    max_t = max(
        sum(w[d] for d in np.where(disagreement[p])[0])
        for p in range(M)
        if disagreement[p].any()
    )
    min_z = -max_t

    if min_z < -10:
        breakpoints = [min_z, -8, -6, -4, -2, 0]
    else:
        breakpoints = [min_z, min_z / 2, -2, -1, 0]

    exp_vals = [math.exp(z) for z in breakpoints]

    # --------------------------------------------------
    # Build Gurobi model
    # --------------------------------------------------
    model = gp.Model("stochastic_max_coverage")
    model.Params.OutputFlag = 0
    model.Params.Presolve = 2
    model.Params.MIPFocus = 1
    model.Params.Cuts = 2

    vtype = GRB.CONTINUOUS if relax_x else GRB.BINARY
    x = model.addVars(used_concepts, lb=0, ub=1, vtype=vtype, name="x")

    z = model.addVars(M, lb=min_z, ub=0, vtype=GRB.CONTINUOUS, name="z")
    u = model.addVars(M, lb=math.exp(min_z), ub=1.0, vtype=GRB.CONTINUOUS, name="u")
    y = model.addVars(M, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="y")

    model.update()

    # z_p = -sum_d A[p,d] w_d x_d
    for p in range(M):
        D = np.where(disagreement[p])[0]
        if len(D) > 0:
            model.addConstr(
                z[p] == -gp.quicksum(w[d] * x[d] for d in D if d in x),
                name=f"zdef_{p}",
            )
        else:
            model.addConstr(z[p] == 0.0)

        model.addGenConstrPWL(z[p], u[p], breakpoints, exp_vals, name=f"exp_{p}")
        model.addConstr(y[p] == 1 - u[p], name=f"ydef_{p}")

    # Budget
    model.addConstr(gp.quicksum(x[d] for d in x) == num_concepts_selected)

    # Objective
    model.setObjective(gp.quicksum(y[p] for p in range(M)), GRB.MAXIMIZE)

    model.optimize()

    # --------------------------------------------------
    # Extract solution
    # --------------------------------------------------
    if model.SolCount == 0:
        raise RuntimeError("No solution found")

    x_vals = np.zeros(K)
    for d in x:
        x_vals[d] = x[d].X
    
    print(sum(x_vals))

    # Enforce exactly k selections
    selected = np.argsort(x_vals)[-num_concepts_selected:].tolist()
    selected = list(sorted(selected))

    subset_concepts = [concept_list[i] for i in selected]

    return subset_concepts, selected


def concept_completeness_selection(env,concept_list,num_concepts_selected,selection_function,q_estimates,concept_source):
    
    X = []
    y = []
    w = []  # kernel weights

    for concepts, action, q in q_estimates:
        S = np.concatenate([concepts, np.array([action])])

        X.append(S)
        y.append(q)

        # kernel weight = 1 / (|S| * (n - |S|))
        k = S.sum()
        n = len(S)
        # avoid division by zero for all-zero or all-one coalitions:
        if k == 0 or k == n:
            weight = 1.0
        else:
            weight = 1.0 / (k * (n - k))

        w.append(weight)

    X = np.array(X)
    y = np.array(y)
    w = np.array(w)

    # -------------------------------------
    # Weighted linear regression
    # Solve:  (X^T W X) φ = X^T W y
    # -------------------------------------
    W = np.diag(w)
    XtW = X.T @ W
    beta = np.linalg.pinv(XtW @ X) @ (XtW @ y)
    beta = beta[:-1]
    sorted_vals = np.argsort(np.abs(beta))[::-1].copy()[:num_concepts_selected]
    sorted_vals = list(sorted_vals)
    sorted_vals = [int(i) for i in sorted_vals]

    return [concept_list[i] for i in sorted_vals], sorted_vals


def greedy_selection_supervised(train_X,train_Y,num_concepts_selected):
    entropy_by_concept = []

    for i in range(len(train_X[0])):
        is_zero = len(train_X[train_X[:,i] == 0])/len(train_X)
        is_one = 1-is_zero 
        entropy = is_zero*is_one 
        entropy_by_concept.append(entropy)
    entropy_by_concept = np.array(entropy_by_concept)
    topk_idx = np.argpartition(-entropy_by_concept, num_concepts_selected)[:num_concepts_selected]
    topk_idx = topk_idx[np.argsort(-entropy_by_concept[topk_idx])]
    print(entropy_by_concept[topk_idx],np.mean(entropy_by_concept))
    return topk_idx.tolist()


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
    
    selected_elements, _ = max_prefix_gurobi(per_train_constraint_weighted, num_concepts, in_order=False,as_float=False)

    return selected_elements

def multiple_selection_supervised(train_matrix,labels,num_concepts):
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
