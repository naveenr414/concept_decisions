import numpy as np
import gurobipy as gp
from gurobipy import Model, GRB, quicksum
import scipy
import gymnasium.spaces as spaces
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from skopt import Optimizer
from copy import deepcopy
import random
from math import ceil

from concept_abstraction.env_utils import rollout_q_estimates_td, rollout_pi_estimates
from concept_abstraction.concept_bank import inaccurate_concepts_continuous
from concept_abstraction.training import train_two_stage_ppo_model, evaluate_model
from concept_abstraction.environments import InfoTransformWrapper


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

def greedy_selection(env,concept_list,num_concepts_selected,reference_model,selection_function,q_estimates):
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
    if type(env.observation_space) is spaces.Box:
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


def greedy_iterative_selection(env,concept_list,num_concepts_selected,reference_model,selection_function,q_estimates):
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
    if type(env.observation_space) is spaces.Box:
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

def max_prefix_gurobi(final_vals, num_concepts_selected,in_order=True):
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
    if in_order:
        for i in range(1, n):
            model.addConstr(y[i] <= y[i-1], name=f"prefix_{i}")
    
    model.addConstr(gp.quicksum(x[i] for i in range(m)) <= num_concepts_selected, name="budget")
    model.setObjective(gp.quicksum(y[i] for i in range(n)), GRB.MAXIMIZE)    
    model.optimize()
    
    selected_elements = [U[i] for i in range(m) if x[i].X > 0.5]
    max_prefix_len = sum(1 for i in range(n) if y[i].X > 0.5)
    return selected_elements, max_prefix_len


def lp_based_selection(env,concept_list,num_concepts_selected,reference_model,selection_function,target_abstraction,q_estimates):
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
    if type(env.observation_space) is spaces.Box:
        all_concepts = np.array([i[0] for i in q_estimates])
        median_by_column = np.median(all_concepts,axis=0)
        discretized_X = (all_concepts < median_by_column).astype(np.int8)
    else:
        discretized_X = np.array([i[0] for i in q_estimates])


    if selection_function == "q_value":
        q_values = np.array([i[2] for i in q_estimates])

        k = int(np.sqrt(len(discretized_X)))

        vals_by_action = []

        all_lower_list = []
        all_higher_list = []

        for a in unique_actions:
            relevant_idx = np.where(actions == a)[0]
            relevant_q = q_values[relevant_idx]
            order = np.argsort(relevant_q)
            lower_list = relevant_idx[order[:k]]
            higher_list = relevant_idx[order[-k:]]
            all_lower_list.append(lower_list)
            all_higher_list.append(higher_list)

        final_vals = []

        for a in range(len(unique_actions)):
            for low_idx in all_lower_list[a]:
                for high_idx in all_higher_list[a]:
                    if abs(q_values[low_idx]-q_values[high_idx]) > target_abstraction:
                        tup = (abs(q_values[low_idx]-q_values[high_idx]),[i for i in range(len(concept_list)) if discretized_X[low_idx][i] != discretized_X[high_idx][i]])
                        if tup not in final_vals:
                            final_vals.append(tup)
        final_vals = sorted(final_vals,reverse=True)

        selected_concepts, max_prefix_len = max_prefix_gurobi(final_vals,num_concepts_selected)
        vals_by_action.append((max_prefix_len,selected_concepts))
        idx = vals_by_action[np.argmin([i[0] for i in vals_by_action])][1]
        concepts = [concept_list[i] for i in idx]
    elif selection_function == "policy":
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

def max_accuracy_selection(final_vals,accuracies,direction,target_abstraction_percentage=1):
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
    min_k = 1
    max_k = n
    for k in range(min_k, max_k + 1):
        m = Model()
        m.Params.OutputFlag = 0  # silent

        x = m.addVars(n, vtype=GRB.BINARY)
        y = m.addVars(len(final_vals), vtype=GRB.BINARY)  # y[j] = 1 if list j has >=1 selected
        for j, lst in enumerate(final_vals):
            m.addConstr(quicksum(x[i] for i in lst) >= 1 * y[j])
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
                if avg_acc > best_avg:
                    best_avg = avg_acc
                    best_selection = selected
            else:
                if avg_acc < best_avg:
                    best_avg = avg_acc
                    best_selection = selected

    return best_selection, best_avg

def imperfect_lp_selection(env,concept_list,reference_model,selection_function,target_abstraction,accuracies,direction='max'):
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
    
    if selection_function == "q_value":
        q_estimates = rollout_q_estimates_td(reference_model,env,concept_list)
    elif selection_function == "policy":
        q_estimates = rollout_pi_estimates(reference_model,env,concept_list)


    unique_actions = list(set([int(i[1]) for i in q_estimates]))
    actions = np.array([i[1] for i in q_estimates])
    # Continuous
    if type(env.observation_space) is spaces.Box:
        # Discrete relaxation
        all_concepts = np.array([i[0] for i in q_estimates])
        median_by_column = np.median(all_concepts,axis=0)
        discretized_X = (all_concepts < median_by_column).astype(np.int8)
    else:
        discretized_X = np.array([i[0] for i in q_estimates])

    k = int(np.sqrt(len(discretized_X)))

    if selection_function == "q_value":
        q_values = np.array([i[2] for i in q_estimates])

        vals_by_action = []
        for a in unique_actions:
            relevant_q_estimates = q_values[actions == a]
            relevant_threshold = np.argsort(relevant_q_estimates)
            lower_list = list(relevant_threshold)[:k]
            higher_list = list(relevant_threshold)[-k:]
            
            final_vals = []

            for low_idx in lower_list:
                for high_idx in higher_list:
                    if abs(relevant_q_estimates[low_idx]-relevant_q_estimates[high_idx]) > target_abstraction:
                        tup = (abs(relevant_q_estimates[low_idx]-relevant_q_estimates[high_idx]),[i for i in range(len(concept_list)) if discretized_X[low_idx][i] != discretized_X[high_idx][i]])
                        if tup[-1] == []:
                            return concept_list, list(range(len(concept_list)))
                        final_vals.append(tup[-1])
            selected_concepts, avg_acc = max_accuracy_selection(final_vals,accuracies,direction)
            vals_by_action.append((avg_acc,selected_concepts))

        idx = vals_by_action[np.argmin([i[0] for i in vals_by_action])][1]
        concepts = [concept_list[i] for i in idx]
    elif selection_function == "policy":
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
        idx, _ = max_accuracy_selection(pairs,accuracies,direction,target_abstraction_percentage=target_abstraction)
        concepts = [concept_list[i] for i in idx]
    idx = np.array(idx).tolist()
    return concepts, idx

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

    train_matrix = np.asarray(train_matrix)
    labels = np.asarray(labels)
    n_samples, n_features = train_matrix.shape
    label_size = labels.max() + 1
    all_prefixes = []

    total_concepts = []
    remaining_concepts = set(range(n_features))
    
    # Initially, all samples in one group
    groups = {0: np.arange(n_samples)}
    
    for iteration in range(num_concepts):
        best_score = np.inf
        best_concept = -1
        best_partition = None  # store how groups would look if we choose this concept
        
        for k in remaining_concepts:
            score = 0
            candidate_groups = {}
            gid = 0
            
            # Split each current group by this concept
            for indices in groups.values():
                col_values = train_matrix[indices, k]
                for val in np.unique(col_values):
                    sub_idx = indices[col_values == val]
                    # label counts for this subgroup
                    lbl_counts = np.bincount(labels[sub_idx], minlength=label_size)
                    score += lbl_counts.sum() - lbl_counts.max()
                    candidate_groups[gid] = sub_idx
                    gid += 1
            
            if score < best_score:
                best_score = score
                best_concept = k
                best_partition = candidate_groups
        
        # Update chosen groups
        groups = best_partition
        total_concepts.append(best_concept)
        remaining_concepts.remove(best_concept)
        all_prefixes.append(deepcopy(total_concepts))
    
    return all_prefixes

def ucb(mu_hat,N,n,c=0.1):
    """UCB upper bound
    
    Arguments:
        mu_hat: Float, Mean seen so far
        N: Integer, total number of trials
        n: Total triasl with arm i
        c: Exploration constant, default 0.1
    
    Returns: Float, upper bound on \mu"""
    if n == 0:
        return 1
    return mu_hat + c*np.sqrt(np.log(N)/n)

def iterative_selection(environment_string,env,concept_list,groundtruth_model,selection_function,target_abstraction,num_iterations,training_timesteps):
    """Iteratively select which concepts to run based on true values
        for concept performance
        
    Arguments:
        env: Gymnasium environment
        concept_list: List of potential concepts
        groundtruth_model: Model trained on the raw states
        selection_function: String, q_value or policy
        target_abstraction: Float, how much of the state space we need to cover
        num_iterations: Total iterations we run for
    
    Returns: List of concepts and their indices"""
    trials = [[] for i in range(len(concept_list))]

    for iteration in range(num_iterations):
        predicted_state_loss = [ucb(np.mean(t),iteration,len(t)) for t in trials]
        modified_concept_predictors = [inaccurate_concepts_continuous(func,0,std) for func,std in zip(concept_list,predicted_state_loss)]
        subset_concept, imperfect_idx = imperfect_lp_selection(env,modified_concept_predictors,groundtruth_model,selection_function,target_abstraction,predicted_state_loss,direction='min')

        if iteration == num_iterations-1:
            return subset_concept, imperfect_idx
        ground_truth_concept_env = InfoTransformWrapper(env,subset_concept)
        model = train_two_stage_ppo_model(environment_string,ground_truth_concept_env,subset_concept,total_timesteps=training_timesteps)
        per_state_loss = model.per_state_loss
        for idx,i in enumerate(imperfect_idx):
            trials[i].append(per_state_loss[idx])


def bayesian_iterative_selection(env,eval_env,environment_string,additional_info,seed,concept_list,num_iterations,training_timesteps):
    def run(concept_idx):
        ground_truth_concept_env = InfoTransformWrapper(env,[concept_list[i] for i in concept_idx])
        ground_truth_eval_concept_env = InfoTransformWrapper(eval_env,[concept_list[i] for i in concept_idx])
        model = train_two_stage_ppo_model(environment_string,ground_truth_concept_env,[concept_list[i] for i in concept_idx],total_timesteps=training_timesteps)
        reward = evaluate_model(environment_string,ground_truth_eval_concept_env,additional_info,model,seed)
        return reward

    n_concepts = len(concept_list)
    space = [(0.0, 1.0)] * n_concepts
    opt = Optimizer(space, base_estimator="GP")  # Gaussian Process

    for _ in range(num_iterations):
        x = opt.ask()
        concept_idx = [i for i, xi in enumerate(x) if xi > 0.5]
        
        reward = run(concept_idx)
        
        opt.tell(x, -reward)
        
    # Best subset found
    best_x = opt.Xi[np.argmin(opt.yi)]
    best_subset = [i for i, xi in enumerate(best_x) if xi > 0.5]
    return [concept_list[i] for i in best_subset],best_subset 