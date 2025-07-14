import numpy as np
import gurobipy as gp
from gurobipy import GRB

def greedy_selection(env,num_concepts):
    total_concepts = []
    all_concepts = env.concepts
    reward_by_state = env.rewards
    groups = [0 for i in range(len(reward_by_state))]

    for _ in range(num_concepts):
        scores_by_group = []
        for k in range(len(all_concepts)):
            if k in total_concepts:
                scores_by_group.append(np.inf)
                continue   
            total_groups = max(groups)
            total_score = 0
            for g in range(total_groups+1):
                states_in_group = [i for i in range(len(groups)) if groups[i] == g]
                partition_0 = [i for i in states_in_group if env.concepts[k][i] == 0]
                rewards_0 = [reward_by_state[i] for i in partition_0]
                partition_1 = [i for i in states_in_group if env.concepts[k][i] == 1]
                rewards_1 = [reward_by_state[i] for i in partition_1]

                if len(rewards_0) > 0:
                    total_score = max(max(rewards_0)-min(rewards_0),total_score)
                if len(rewards_1) > 0:
                    total_score = max(max(rewards_1)-min(rewards_1),total_score)
            scores_by_group.append(total_score)
        selected_idx = np.argmin(scores_by_group)

        total_groups = max(groups)
        new_groups = max(groups)
        for g in range(total_groups+1):
            states_in_group = [i for i in range(len(groups))  if groups[i] == g]
            partition_0 = [i for i in states_in_group if env.concepts[selected_idx][i] == 0]
            partition_1 = [i for i in states_in_group if env.concepts[selected_idx][i] == 1]
            if len(partition_0) > 0 and len(partition_1) > 0:
                for i in partition_1:
                    groups[i] = new_groups + 1
                new_groups += 1
        total_concepts.append(selected_idx)
    return total_concepts 

def random_selection(env,num_concepts):
    total_concepts = len(env.concepts)
    return np.random.choice(list(range(total_concepts)),num_concepts,replace=False)

def human_centered_selection(env,accuracy_by_concept,target_abstraction):
    rewards = env.rewards
    state_pairs = []

    for i in range(len(rewards)):
        for j in range(i):
            state_pairs.append((abs(rewards[i]-rewards[j]),np.where(env.concepts[:,i] != env.concepts[:,j])[0]))
    state_pairs = [i for i in state_pairs if i[0] > target_abstraction]

    m = gp.Model("fractional_lp")
    n = len(accuracy_by_concept)

    # Variables: y[i] = t * x[i], where x[i] ∈ [0,1]
    y = m.addVars(n, lb=0.0, name="y")
    t = m.addVar(lb=0.0, name="t")

    # Objective: Maximize sum(c[i] * y[i])
    m.setObjective(gp.quicksum(accuracy_by_concept[i] * y[i] for i in range(n)), GRB.MAXIMIZE)

    # Charnes-Cooper constraint: sum(y) = 1
    m.addConstr(gp.quicksum(y[i] for i in range(n)) == 1)

    # Bounds: y[i] <= t
    for i in range(n):
        m.addConstr(y[i] <= t)

    for pair in state_pairs:
        m.addConstr(gp.quicksum(y[i] for i in pair[1] if i < n) >= t)

    # Optimize
    m.optimize()

    t_val = t.X
    x_vals = [y[i].X / t_val for i in range(n)]
    return np.array(x_vals)