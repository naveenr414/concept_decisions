import numpy as np
import gurobipy as gp
from gurobipy import Model, GRB, quicksum
import scipy
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from skopt import Optimizer
from copy import deepcopy
import random
from math import ceil
from concept_abstraction.training import evaluate_model, train_ppo_model
from concept_abstraction.environments import get_environment
import torch
from itertools import combinations
from collections import defaultdict, Counter 
from sklearn.feature_selection import mutual_info_regression

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

    return concepts, idx



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


def greedy_iterative_selection(concept_list,num_concepts_selected,selection_function,q_estimates,concept_source):
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
        curr_concepts = []
        for _ in range(num_concepts_selected):
            val_by_next_concept = []
            for idx in range(len(concept_list)):
                if idx in curr_concepts:
                    val_by_next_concept.append(-10000)
                else:
                    if selection_function == "q_value":
                        correlations = []
                        num_by_action = []

                        for action in unique_actions:
                            X_list, y_list = [], []
                            for i in q_estimates:
                                if int(i[1]) == action:
                                    features = [i[0][idx]] + [i[0][j] for j in curr_concepts]
                                    X_list.append(features)
                                    y_list.append(i[2])

                            X = np.array(X_list)
                            y = np.array(y_list)
                            if len(X) <= 2:
                                r2 = 0
                            else:
                                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

                                model = LinearRegression()
                                model.fit(X_train, y_train)
                                # Predict
                                y_pred = model.predict(X_test)
                                # Evaluate
                                r2 = r2_score(y_test, y_pred)
                            correlations.append(r2)
                            num_by_action.append(len(X))
                        avg_correlation = np.sum(np.array(correlations)*np.array(num_by_action))/np.sum(num_by_action)
                        val_by_next_concept.append(avg_correlation)
                    elif selection_function == "policy":
                        X_list, y_list = [], []
                        for i in q_estimates:
                            features = [i[0][idx]] + [i[0][j] for j in curr_concepts]
                            X_list.append(features)
                            y_list.append(i[1])

                        X = np.array(X_list)
                        y = np.array(y_list)
                        if len(X) <= 2:
                            r2 = 0
                        else:
                            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

                            model = LinearRegression()
                            model.fit(X_train, y_train)
                            # Predict
                            y_pred = model.predict(X_test)
                            # Evaluate
                            r2 = r2_score(y_test, y_pred)
                        val_by_next_concept.append(r2)
            curr_concepts.append(int(np.argmax(val_by_next_concept))  )   
        idx = np.array(curr_concepts).tolist()    
        concepts = [concept_list[i] for i in idx]
    else:
        selected_concepts = []
        for i in range(num_concepts_selected):
            correlation_by_concept = []
            for idx in range(len(concept_list)):
                if idx in selected_concepts:
                    correlation_by_concept.append(1000000)
                    continue 
                full_concepts = selected_concepts + [idx]
                total_variance = 0
                if selection_function == "q_value":
                    for action in unique_actions:
                        group_pairs = [(str(np.array(i[0])[full_concepts]),i[2]) for i in q_estimates if int(i[1]) == action]
                        valid_str = set([i[0] for i in group_pairs])
                        for concept in valid_str:
                            subset = [i[1] for i in group_pairs if i[0] == concept]
                            total_variance += np.std(subset) * len(subset)
                elif selection_function == "policy":
                    group_pairs = [(str(np.array(i[0])[full_concepts]),i[1]) for i in q_estimates]
                    valid_str = set([i[0] for i in group_pairs])
                    for concept in valid_str:
                        subset = [i[1] for i in group_pairs if i[0] == concept]
                        total_variance += np.std(subset) * len(subset)
                correlation_by_concept.append(total_variance)
            selected_concepts.append(int(np.argmin(correlation_by_concept)))
        idx = np.array(selected_concepts).tolist()
        concepts = [concept_list[i] for i in idx]

    return concepts, idx

def max_prefix_gurobi(final_vals, num_concepts_selected,in_order=True,as_float=False,weighted=False):
    """Arguments:
        final_vals: list of tuples (value, elements_covering_value)
                    assumed sorted in decreasing priority (top first)
        num_concepts_selected: budget
    Returns: 
        List of concepet indexes

    """
    U = set()
    for _, elems in final_vals:
        U.update(elems)
    U = list(U)  # universe of elements
    elem_to_idx = {e:i for i,e in enumerate(U)}
    
    n = len(final_vals)
    m = len(U)
    
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
                y[i] <= gp.quicksum(x[elem_to_idx[e]] for e in elems),
                name=f"cover_{i}"
            )
        else:
            model.addConstr(y[i] == 0)  # cannot be covered
    
    # Prefix constraints: enforce consecutive coverage
    # y[i] <= y[i-1] for i>0
    if in_order:
        for i in range(1, n):
            model.addConstr(y[i] <= y[i-1], name=f"prefix_{i}")
    model.addConstr(gp.quicksum(x[i] for i in range(m)) <= num_concepts_selected, name="budget")
    
    if weighted: 
        model.setObjective(gp.quicksum(y[i]*final_vals[i][0] for i in range(n)), GRB.MAXIMIZE)    
    else:
        model.setObjective(gp.quicksum(y[i] for i in range(n)), GRB.MAXIMIZE)    
    model.optimize()

    selected_elements = [U[i] for i in range(m) if x[i].X > 0.5]
    max_prefix_len = sum(1 for i in range(n) if y[i].X > 0.5)
    return selected_elements, max_prefix_len

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


def select_concepts_by_mi(states, q_values,num_concepts_selected,concept_list):
    """
    Select concepts using mutual information between each concept dimension
    (binary feature) and the ΔQ value or policy target.
    
    Arguments:
        states: np.array shape (N, D), binary 0/1
        q_values: np.array shape (N,2) with Q(s,0), Q(s,1)
        k: number of concepts to select
    
    Returns:
        idx: indices of concepts selected
        scores: MI scores for all features
    """
    k = num_concepts_selected
    # Compute ΔQ as continuous target for MI
    delta_q = q_values[:, 1] - q_values[:, 0]

    # Compute MI between each feature X[:, i] and delta-q
    # mutual_info_regression works fine for binary inputs
    mi_scores = mutual_info_regression(states, delta_q, discrete_features=True)

    # Pick top-k concepts
    idx = list(np.argsort(mi_scores)[::-1][:k])

    return [concept_list[i] for i in idx], idx

def multiple_lp_selection(env,concept_list,num_concepts_selected,selection_function,q_estimates,concept_source):
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
                    if tup not in seen and diffs != ():
                        seen.add(tup)
                        final_vals.append(tup)
        final_vals = sorted(final_vals,reverse=True)
        final_vals = final_vals[:10000]
        idx, _ = max_prefix_gurobi(final_vals,num_concepts_selected,weighted=True,in_order=False)
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

def max_accuracy_selection(final_vals,accuracies,direction,num_concepts_selected,target_abstraction_percentage=1,multiplier=1):
    """Select the set of concepts (where there are len(accuracies)
        total concepts)
        So that the average accuracy is maximized (if direction = 'max')
            or minimized (if direction = 'min')
        Subject to each list in final_vals having at least 1 element selected
    
    Arguments:
        final_vals: List of lists, where each number corresponds
            to a concept/constraint
        accuracies: Float values for each concept
        direction: String, 'max' or 'min'
    
    Returns: Indices for the selected concepts"""

    n = len(accuracies)
    if direction == "max":
        best_avg = -float('inf')
    else:
        best_avg = float('inf')
    best_selection = []

    # Compute feasible range for number of selected concepts
    # Minimum number of concepts = max number of disjoint sets in final_vals
    k = num_concepts_selected
    m = Model()
    m.Params.OutputFlag = 0  # silent

    x = m.addVars(n, vtype=GRB.BINARY)
    y = m.addVars(len(final_vals), vtype=GRB.BINARY)  # y[j] = 1 if list j has >=1 selected
    for j, lst in enumerate(final_vals):
        m.addConstr(quicksum(x[i] for i in lst) >= multiplier * y[j])
    
    m.addConstr(quicksum(y[j] for j in range(len(final_vals))) >= ceil(target_abstraction_percentage * len(final_vals)))
    m.addConstr(quicksum(x[i] for i in range(n)) == k)
    if direction == 'max':
        m.setObjective(quicksum(accuracies[i] * x[i] for i in range(n)), GRB.MAXIMIZE)
    else:
        m.setObjective(quicksum(accuracies[i] * x[i] for i in range(n)), GRB.MINIMIZE)
    m.optimize()

    if m.Status == GRB.OPTIMAL:
        selected = [i for i in range(n) if x[i].X > 0.5]
        avg_acc = sum(accuracies[i] for i in selected) / k
        if direction == "max":
            if avg_acc >= best_avg:
                best_avg = avg_acc
                best_selection = selected
        else:
            if avg_acc <= best_avg:
                best_avg = avg_acc
                best_selection = selected
    else:
        print("Status",m.Status)

    return best_selection, best_avg

def imperfect_lp_selection(env,concept_list,q_estimates,selection_function,target_abstraction,num_concepts_selected,accuracies,concept_source,environment_string,additional_info,direction='max'):
    """Select {num_concepts} through an LP policy
        by first learning the Q(s,a) values from a rollout
        Then selecting the concepts that maximize the average accuracy
            While selecting according to the target abstraction
    
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
    if concept_source == "human_selection":
        # Discrete relaxation
        all_concepts = np.array([i[0] for i in q_estimates])
        median_by_column = np.median(all_concepts,axis=0)
        discretized_X = (all_concepts < median_by_column).astype(np.int8)
    else:
        discretized_X = np.array([i[0] for i in q_estimates])

    if selection_function == "q_value":
        q_values = np.array([i[2] for i in q_estimates])
        final_vals = []
        for a in unique_actions:
            relevant_q_estimates = q_values[actions == a]
            relevant_threshold = np.argsort(relevant_q_estimates)
            for low_idx in relevant_threshold[:500]:
                for high_idx in relevant_threshold[-500:]:
                    if abs(relevant_q_estimates[low_idx]-relevant_q_estimates[high_idx]) > target_abstraction:
                        tup = (abs(relevant_q_estimates[low_idx]-relevant_q_estimates[high_idx]),[i for i in range(len(concept_list)) if discretized_X[low_idx][i] != discretized_X[high_idx][i]])
                        if tup[-1] == [] or len(final_vals) > 10_000:
                            continue 
                        final_vals.append(tup[-1])
        if len(final_vals) > 10_000:
            final_vals = random.sample(final_vals, 10_000)
        idx, avg_acc = max_accuracy_selection(final_vals,accuracies,"max",num_concepts_selected,target_abstraction_percentage=0.5)
        concepts = [concept_list[i] for i in idx]
    elif selection_function == "policy":
        # TODO: Ensure the policy equivalent works
        if target_abstraction > 1:
            return [],[] 
        elif target_abstraction == 0:
            return concept_list, list(range(len(concept_list)))

        sample_by_action = []

        for a in unique_actions:
            X_reduced = discretized_X[actions == a]
            sample_by_action.append(X_reduced)
        weights = [len(lst) for lst in sample_by_action]

        pairs = []
        for i in range(100):
            i, j = random.choices(range(len(sample_by_action)), weights=weights, k=2)
            while i == j:  # ensure different lists
                j = random.choices(range(len(sample_by_action)), weights=weights, k=1)[0]
            a = random.choice(sample_by_action[i])
            b = random.choice(sample_by_action[j])
            pairs.append([i for i in range(len(a)) if a[i] != b[i]])
        idx, _ = max_accuracy_selection(pairs,accuracies,direction,num_concepts_selected,target_abstraction_percentage=target_abstraction)
        concepts = [concept_list[i] for i in idx]
    idx = np.array(idx).tolist()
    return concepts, idx


def greedy_max_coverage(final_vals, budget):
    # final_vals: list of (value, elements_covering_value)
    # budget: number of concepts to select
    
    # Build inverse map: element -> set indices it covers
    element_to_sets = defaultdict(set)
    for i, (_, elems) in enumerate(final_vals):
        for e in elems:
            element_to_sets[e].add(i)

    covered = set()
    selected = []

    for _ in range(budget):
        best_e, best_gain = None, -1
        # pick element covering the most *new* sets
        for e, sets in element_to_sets.items():
            gain = len(sets - covered)
            if gain > best_gain:
                best_e, best_gain = e, gain
        if best_gain <= 0:
            break  # no more gain
        selected.append(best_e)
        covered.update(element_to_sets[best_e])
        # optional: lazy update for speed
    return selected, len(covered)

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

def imperfect_lp_selection_supervised(train_matrix,labels,num_concepts,accuracies):
    """Select {num_concepts} through an LP policy
        by first learning the Q(s,a) values from a rollout
        Then selecting the concepts that maximize the average accuracy
            While selecting according to the target abstraction
    
    Arguments:
        env: Gymasium environment
        concept_list: List of functions
        num_concepts_selected: Integer, number of concepts to select
        reference_model: A policy that performs well
            which we are trying to distill
        selection_function: String, whether we're selecting according to 
            Q value, etc."""

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
        per_train_constraint_weighted.append(elems_diff)
    
    selected_elements, _ = max_accuracy_selection(per_train_constraint_weighted,accuracies,"max",num_concepts,target_abstraction_percentage=1,multiplier=1)

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


def lp_selection_supervised_imperfect(train_matrix,labels,num_concepts,accuracies,fraction_y):
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

    final_vals = per_train_constraint_weighted
    U = set()
    for _, elems in final_vals:
        U.update(elems)
    U = list(U)  # universe of elements
    elem_to_idx = {e:i for i,e in enumerate(U)}
    
    n = len(final_vals)
    m = len(U)
    
    model = gp.Model("max_prefix_hitting")
    model.Params.OutputFlag = 0  # silence solver
    
    # Binary vars: x[e] = 1 if element e selected
    x = model.addVars(m, vtype=GRB.BINARY, name="x")
    
    # Binary vars: y[i] = 1 if value i is covered
    y = model.addVars(n, vtype=GRB.BINARY, name="y")

    # Link y[i] to coverage by selected elements
    for i, (_, elems) in enumerate(final_vals):
        if elems:  # make sure not empty
            model.addConstr(
                y[i] <= gp.quicksum(x[elem_to_idx[e]] for e in elems),
                name=f"cover_{i}"
            )
        else:
            model.addConstr(y[i] == 0)  # cannot be covered
    
    # Prefix constraints: enforce consecutive coverage
    # y[i] <= y[i-1] for i>0
    model.addConstr(gp.quicksum(x[i] for i in range(m)) <= num_concepts, name="budget")
    model.addConstr(gp.quicksum(y[i] for i in range(n)) >= fraction_y*n)    
    model.setObjective(gp.quicksum(x[i]*accuracies[i] for i in range(m)), GRB.MAXIMIZE)    

    model.optimize()
    
    selected_elements = [U[i] for i in range(m) if x[i].X > 0.5]
    return selected_elements

def lp_vector_fqe_selection(states,fqe,state_mean,state_std,concept_list,num_concepts_selected):
    q_values, q_std = fqe.get_q_values(states)
    print(f"Q-values shape: {q_values.shape}")  # (100, 2)
    print(f"Q-values std shape: {q_std.shape}")  # (100, 2)


    diffs = []
    modified_q_estimates = []
    s = ((states*state_std)+state_mean).astype(np.int8)

    for idx,_ in enumerate(states):
        action_list =  q_values[idx]

        modified_q_estimates.append((tuple(s[idx]),tuple(action_list)))


    modified_q_estimates = list(set(modified_q_estimates))

    for i in range(len(modified_q_estimates)):
        for j in range(len(modified_q_estimates)):
            if i>=j:
                m_i = modified_q_estimates[i]
                m_j = modified_q_estimates[j]

                diff = 0

                for k in range(len(m_i[1])):
                    diff = max(m_i[1][k]-m_j[1][k],diff)

                diffs.append((diff,[k for k in range(len(m_i[0])) if m_i[0][k] != m_j[0][k]]))

    sorted_diffs = sorted(diffs,key=lambda k: k[0],reverse=True)
    idx, _ = max_prefix_gurobi(sorted_diffs,num_concepts_selected)
    concepts = [concept_list[i] for i in idx]
    return concepts, idx
