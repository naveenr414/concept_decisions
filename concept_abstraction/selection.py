import numpy as np
import gurobipy as gp
from gurobipy import GRB
import scipy

from concept_abstraction.env_utils import rollout_q_estimates


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
    return [concept_list[i] for i in idx]

def greedy_selection(env,concept_list,num_concepts_selected,reference_model,selection_function):
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
    
    q_estimates = rollout_q_estimates(reference_model,env,concept_list)
    unique_actions = list(set([int(i[1]) for i in q_estimates]))

    correlation_by_concept = []
    if selection_function == "q_value":
        for idx in range(len(concept_list)):
            correlations = []
            num_by_action = []

            for action in unique_actions:
                x_y_pair = [(i[0][idx],i[2]) for i in q_estimates if int(i[1]) == action]
                x,y = zip(*x_y_pair)
                num_by_action.append(len(x))
                correlations.append(scipy.stats.pearsonr(x,y).statistic**2)
            avg_correlation = np.sum(np.array(correlations)*np.array(num_by_action))/np.sum(num_by_action)
            correlation_by_concept.append(avg_correlation)
    elif selection_function == "policy":
        for idx in range(len(concept_list)):
            x_y_pair = [(i[0][idx],i[1]) for i in q_estimates]
            x,y = zip(*x_y_pair)
            avg_correlation = scipy.stats.pearsonr(x,y).statistic**2
            correlation_by_concept.append(avg_correlation)
    correlation_by_concept = np.array(correlation_by_concept)
    idx = np.argpartition(-correlation_by_concept, num_concepts_selected)[:num_concepts_selected]
    idx = idx[np.argsort(-correlation_by_concept[idx])]
    concepts = [concept_list[i] for i in idx]

    return concepts 

def greedy_selection_real_world(env,num_concepts,concept_list,state_maps,metric='std'):
    """Select {num_concepts} greedily
        by selecting those that reduce the reward range within each partition
        For example, first select the concept
            so that, c_{i} = 0 and c_{i} = 1 each have
                small differences between max and min reward

    Arguments:
        env: Gymasium environment
        num_concepts: Integer, number of concepts to select
        concept_list: Map of those states to concepts
        state_maps: Either the actions (\pi(s)), Q values ([Q(s,0),Q(s,1)], or reward/transition [Transition List for each concept])
            for each of the state seen
    
    Returns: List of size {num_concepts} of integers
        each representing a concept"""

    total_concepts = []
    all_concepts = env.concepts
    groups = [0 for i in range(len(concept_list))]

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
                partition_0 = [i for i in states_in_group if concept_list[i][k] == 0]
                rewards_0 = np.array([state_maps[i] for i in partition_0])
                partition_1 = [i for i in states_in_group if concept_list[i][k] == 1]
                rewards_1 = np.array([state_maps[i] for i in partition_1])

                if len(rewards_0) > 0:
                    for action in range(len(rewards_0[0])):
                        if metric == 'min_max':
                            total_score = max(np.max(rewards_0[:,action])-np.min(rewards_0[:,action]),total_score) 
                        elif metric == 'std':
                            total_score = max(np.std(rewards_0[:,action]),total_score) 

                if len(rewards_1) > 0:
                    for action in range(len(rewards_1[0])):
                        if metric == 'min_max':
                            total_score = max(np.max(rewards_1[:,action])-np.min(rewards_1[:,action]),total_score) 
                        elif metric == 'std':
                            total_score = max(np.std(rewards_1[:,action]),total_score) 
            scores_by_group.append(total_score)
        selected_idx = int(np.argmin(scores_by_group))

        total_groups = max(groups)
        new_groups = max(groups)
        for g in range(total_groups+1):
            states_in_group = [i for i in range(len(groups)) if groups[i] == g]
            partition_0 = [i for i in states_in_group if concept_list[i][selected_idx] == 0]
            partition_1 = [i for i in states_in_group if concept_list[i][selected_idx] == 1]
            if len(partition_0) > 0 and len(partition_1) > 0:
                for i in partition_1:
                    groups[i] = new_groups + 1
                new_groups += 1
        total_concepts.append(selected_idx)
    return total_concepts 

def greedy_selection_supervised(train_matrix,labels,num_concepts):
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

    total_concepts = []
    all_concepts = train_matrix.shape[1]
    groups = [0 for i in range(len(labels))]

    for _ in range(num_concepts):
        scores_by_group = []
        for k in range(all_concepts):
            if k in total_concepts:
                scores_by_group.append(np.inf)
                continue   
            total_groups = max(groups)
            total_score = 0
            for g in range(total_groups+1):
                states_in_group = [i for i in range(len(groups)) if groups[i] == g]
                partition_0 = [i for i in states_in_group if train_matrix[i][k] == 0]
                rewards_0 = np.array([labels[i] for i in partition_0])
                partition_1 = [i for i in states_in_group if train_matrix[i][k] == 1]
                rewards_1 = np.array([labels[i] for i in partition_1])

                total_score = 0 
                
                if len(rewards_0) > 0:
                    total_score += float((1-np.max(np.bincount(rewards_0)) / len(rewards_0))*len(rewards_0))
                if len(rewards_1) > 0:
                    total_score += (1-np.max(np.bincount(rewards_1)) / len(rewards_1))*len(rewards_1)

            scores_by_group.append(total_score)
        selected_idx = int(np.argmin(scores_by_group))

        total_groups = max(groups)
        new_groups = max(groups)
        for g in range(total_groups+1):
            states_in_group = [i for i in range(len(groups))  if groups[i] == g]
            partition_0 = [i for i in states_in_group if train_matrix[i][selected_idx] == 0]
            partition_1 = [i for i in states_in_group if train_matrix[i][selected_idx] == 1]
            if len(partition_0) > 0 and len(partition_1) > 0:
                for i in partition_1:
                    groups[i] = new_groups + 1
                new_groups += 1
        total_concepts.append(selected_idx)
    return total_concepts 


def human_centered_selection(env,accuracy_by_concept,target_abstraction):
    """Select concepts based on human skill
        Solve a fractional linear programming problem
        that optimizes for the average accuracy
        subject to range <= target_abstraction

    Arguments:
        env: Gymasium environment
        accuracy_by_concept: List of floats, accuracy of 
            humans for each concept
        target_abstraction: float, threshold for 
            range of concepts
    
    Returns: List of size {num_concepts} of integers
        each representing a concept"""
    
    rewards = env.rewards
    state_pairs = []
    for i in range(len(rewards)):
        for j in range(i):
            state_pairs.append((max(abs(rewards[i]-rewards[j])),np.where(env.concepts[:,i] != env.concepts[:,j])[0]))
    state_pairs = [i for i in state_pairs if i[0] > target_abstraction]
    
    m = gp.Model("fractional_lp")
    n = len(accuracy_by_concept)
    y = m.addVars(n, lb=0.0, name="y")
    t = m.addVar(lb=0.0, name="t")

    # Maximize average accuracy
    m.setObjective(gp.quicksum(accuracy_by_concept[i] * y[i] for i in range(n)), GRB.MAXIMIZE)

    # Ensure minimum coverage
    m.addConstr(gp.quicksum(y[i] for i in range(n)) == 1, name="normalize")
    for i in range(n):
        m.addConstr(y[i] <= t, name=f"upper_bound_y_{i}")
    for idx, pair in enumerate(state_pairs):
        m.addConstr(gp.quicksum(y[i] for i in pair[1]) >= t, name=f"coverage_{idx}_{pair}")
    m.setParam(gp.GRB.Param.DualReductions, 0)
    m.optimize()

    if m.status == gp.GRB.INFEASIBLE:
        print("Model is infeasible")
        m.computeIIS()
        m.write("model.ilp")
        m.write("model.lp")
        m.write("model.mps")
        raise RuntimeError("Model infeasible")
    elif m.status != gp.GRB.OPTIMAL:
        raise RuntimeError(f"Solver ended with status {m.status}")

    # Extract solution
    t_val = t.X
    x_vals = [y[i].X / t_val for i in range(n)] if t_val > 0 else [0.0] * n

    return np.array(x_vals)


def human_centered_selection_real_world(env,accuracy_by_concept,target_abstraction,concept_list,state_maps,sample=10):
    """Select concepts based on human skill
        Solve a fractional linear programming problem
        that optimizes for the average accuracy
        subject to range <= target_abstraction

    Arguments:
        env: Gymasium environment
        accuracy_by_concept: List of floats, accuracy of 
            humans for each concept
        target_abstraction: float, threshold for 
            range of concepts
            concept_list: Map of those states to concepts
        state_maps: Either the actions (\pi(s)), Q values ([Q(s,0),Q(s,1)], or reward/transition [Transition List for each concept])
            for each of the state seen

    Returns: List of size {num_concepts} of integers
        each representing a concept"""
    
    if len(concept_list) > sample:
        indices = np.random.choice(len(concept_list), size=sample, replace=True)

        # Apply the same indices to both lists
        concept_list = np.array([concept_list[i] for i in indices])
        state_maps = np.array([state_maps[i] for i in indices])

    rewards = state_maps
    state_pairs = []
    for i in range(len(rewards)):
        for j in range(i):
            state_pairs.append((max(abs(rewards[i]-rewards[j])),np.where(concept_list[i,:] != concept_list[j,:])[0]))
    state_pairs = [i for i in state_pairs if i[0] > target_abstraction and len(i[1]) > 0]
    
    m = gp.Model("fractional_lp")
    n = len(accuracy_by_concept)
    y = m.addVars(n, lb=0.0, name="y")
    t = m.addVar(lb=0.0, name="t")

    # Maximize average accuracy
    m.setObjective(gp.quicksum(accuracy_by_concept[i] * y[i] for i in range(n)), GRB.MAXIMIZE)

    # Ensure minimum coverage
    m.addConstr(gp.quicksum(y[i] for i in range(n)) == 1, name="normalize")
    for i in range(n):
        m.addConstr(y[i] <= t, name=f"upper_bound_y_{i}")
    for idx, pair in enumerate(state_pairs):
        m.addConstr(gp.quicksum(y[i] for i in pair[1]) >= t, name=f"coverage_{idx}_{pair}")
    m.setParam(gp.GRB.Param.DualReductions, 0)
    m.optimize()

    if m.status == gp.GRB.INFEASIBLE:
        print("Model is infeasible")
        m.computeIIS()
        m.write("model.ilp")
        m.write("model.lp")
        m.write("model.mps")
        raise RuntimeError("Model infeasible")
    elif m.status != gp.GRB.OPTIMAL:
        raise RuntimeError(f"Solver ended with status {m.status}")

    # Extract solution
    t_val = t.X
    x_vals = [y[i].X / t_val for i in range(n)] if t_val > 0 else [0.0] * n

    return np.array(x_vals)