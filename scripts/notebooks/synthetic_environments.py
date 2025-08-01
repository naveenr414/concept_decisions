# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: food
#     language: python
#     name: python3
# ---

# %load_ext autoreload
# %autoreload 2

from concept_abstraction.training import train_model
from concept_abstraction.selection import greedy_selection, random_selection, human_centered_selection
from concept_abstraction.env_utils import *
from concept_abstraction.utils import *
import sys 
import argparse
import secrets
import numpy as np 
import random 
import time 

is_jupyter = 'ipykernel' in sys.modules

# +
if is_jupyter: 
    seed        = 43
    environment_string = "cycle"
    environment_nodes = 4
    show_baseline = True 
    human_accuracy_by_concept = None 
    target_abstraction = 0
    out_folder = "synthetic"
    num_concepts_selected = 0
    cbm_accuracy_by_concept = None
    human_reliance_by_concept =None
    reward_error = 0.1
    transition_error = 0
else:
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', help='Random Seed', type=int, default=42)
    parser.add_argument('--environment_string', help='Which environment to create', type=str, default="tree")
    parser.add_argument('--environment_nodes', help='Size of the environment; number of nodes', type=int, default=4)
    parser.add_argument('--show-baseline', action='store_true', help='Whether to show the baseline')
    parser.add_argument('--num_concepts_selected', help='Number of concepts selected by greedy or random',type=int, default=0)
    parser.add_argument('--human_accuracy_by_concept', nargs='*', type=float, default=None)
    parser.add_argument('--target_abstraction', help='Value for the target abstraction with human performance', type=float, default=0.05)
    parser.add_argument('--cbm_accuracy_by_concept', help="What is the accuracy of AI per concept?", nargs='*', type=float, default=None)
    parser.add_argument('--human_reliance_by_concept', help="How much does AI rely on human intervention?",  nargs='*', type=float, default=None)
    parser.add_argument('--reward_error', help="How much to perturb the reward by?", type=float, default=0)
    parser.add_argument('--transition_error', help="How much to perturb the transition by?", type=float, default=0)
    parser.add_argument('--out_folder', help='Which folder', type=str, default="exploration")

    args = parser.parse_args()

    seed = args.seed
    environment_string = args.environment_string
    environment_nodes = args.environment_nodes 
    show_baseline = args.show_baseline
    num_concepts_selected = args.num_concepts_selected
    human_accuracy_by_concept = args.human_accuracy_by_concept
    human_reliance_by_concept = args.human_reliance_by_concept
    target_abstraction = args.target_abstraction
    cbm_accuracy_by_concept = args.cbm_accuracy_by_concept
    reward_error = args.reward_error
    transition_error = args.transition_error
    out_folder = args.out_folder

save_name = secrets.token_hex(4)  
# -

results = {}
results['parameters'] = {'seed'      : seed,
        'environment_string'    : environment_string, 
        'environment_nodes': environment_nodes, 
        'show_baseline': show_baseline,
        'num_concepts_selected': num_concepts_selected,
        'human_accuracy_by_concept': human_accuracy_by_concept, 
        'human_reliance_by_concept': human_reliance_by_concept, 
        'target_abstraction': target_abstraction,
        'cbm_accuracy_by_concept': cbm_accuracy_by_concept,
        'reward_error': reward_error, 
        'transition_error': transition_error,
}
print("Parameters {}".format(results['parameters']))

np.random.seed(seed)
random.seed(seed)

# ## Retrieving Concept Values

baseline_concepts = get_baseline_concept_sets(environment_string,environment_nodes)
results['baseline'] = {'concepts': baseline_concepts}
env = create_environment_from_string(environment_string,environment_nodes,baseline_concepts[-1],None)


def make_env_fn(concept_list,accuracies,reward_error=0,transition_error=0):
    env = create_environment_from_string(environment_string, environment_nodes, concept_list, accuracies)
    new_rewards = env.rewards+np.random.normal(0,reward_error,size=env.rewards.shape)
    new_transitions = env.transitions+np.random.normal(0,transition_error,size=env.transitions.shape) 
    if np.min(new_transitions) < 0:
        new_transitions -= np.min(new_transitions)
    for i in range(len(new_transitions)):
        for j in range(len(new_transitions[i])):
            new_transitions[i,j] /= np.sum(new_transitions[i,j])
    
    env.rewards = new_rewards 
    env.transitions = new_transitions 

    return env


if show_baseline:
    values_by_concept = []
    rewards = []
    transitions = []

    for concept_list in baseline_concepts:
        env = make_env_fn(concept_list,None,reward_error,transition_error)
        model = train_model(env)
        values_by_concept.append(get_values(env, model))
        rewards.append(env.rewards.tolist())
        transitions.append(env.transitions.tolist())

    results['baseline'] = {
        'concepts': baseline_concepts,
        'values': values_by_concept,
        'rewards': rewards, 
        'transitions': transitions 
    }

# ## Concept Selection

# +
selected_concepts = []
random_times = [] 
values_by_random_concept = []
rewards = []
transitions = []
start = time.time() 
for k in range(1,num_concepts_selected+1):
    if k > len(env.concepts):
        break 
    random_concepts = random_selection(env,k)

    env = make_env_fn(random_concepts,None,reward_error=reward_error,transition_error=transition_error)
    model = train_model(env)
    selected_concepts.append(random_concepts)
    values_by_random_concept.append(get_values(env,model))
    random_times.append(time.time()-start)
    rewards.append(env.rewards.tolist())
    transitions.append(env.transitions.tolist())

results['random_selection'] = {
    'concepts': [i.tolist() for i in selected_concepts], 
    'values': values_by_random_concept,
    'time': random_times, 
    'rewards': rewards, 
    'transitions': transitions 
}

# +
selected_concepts = []
values_by_greedy_concept = []
greedy_times = []
rewards = []
transitions = []
start = time.time() 
for k in range(1,num_concepts_selected+1):
    if k > len(env.concepts):
        break 
    greedy_concepts = greedy_selection(env,k)
    env = make_env_fn(greedy_concepts,None,reward_error=reward_error,transition_error=transition_error)
    model = train_model(env)
    selected_concepts.append(greedy_concepts)
    values_by_greedy_concept.append(get_values(env,model))
    greedy_times.append(time.time()-start)
    rewards.append(env.rewards.tolist())
    transitions.append(env.transitions.tolist())

results['greedy_selection'] = {
    'concepts': selected_concepts, 
    'values': values_by_greedy_concept,
    'time': greedy_times,
    'rewards': rewards, 
    'transitions': transitions 
}
# -

# ## Performance under Uncertainty

if human_accuracy_by_concept is not None or cbm_accuracy_by_concept is not None:
    if human_accuracy_by_concept is None:
        modified_acc_rate = cbm_accuracy_by_concept
    elif cbm_accuracy_by_concept is None:
        modified_acc_rate = human_accuracy_by_concept
    else:
        modified_acc_rate = [reliance_percent*human_acc + (1-reliance_percent)*machine_acc 
                for human_acc,machine_acc,reliance_percent in zip(human_accuracy_by_concept,
                                                                cbm_accuracy_by_concept,
                                                                human_reliance_by_concept)]

    selected_concepts = human_centered_selection(env,modified_acc_rate,target_abstraction)
    selected_concepts = [idx for idx,i in enumerate(selected_concepts) if i>=0.5]

    env = make_env_fn(selected_concepts,modified_acc_rate)
    model = train_model(env)
    human_perf = get_values(env,model)

    concepts = []
    values_error = []

    for idx,concept_list in enumerate(baseline_concepts):
        env = create_environment_from_string(environment_string,environment_nodes,concept_list,modified_acc_rate)
        q_net = train_model(make_env_fn(concept_list,modified_acc_rate))
        concepts.append(concept_list)
        values_error.append(get_values(env,q_net))

    results['uncertainty'] = {
        'values': values_error,
        'concepts': concepts, 
        'selected_concepts': selected_concepts,
        'combined_accuracies': modified_acc_rate,
        'combined_value': human_perf,
    }

# ## Save Data

save_path = get_save_path(out_folder,save_name)

delete_duplicate_results(out_folder,"",results)

json.dump(results,open('../../results/'+save_path,'w'))
