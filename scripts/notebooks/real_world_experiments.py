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

from concept_abstraction.training import train_ppo_model, SimpleQEstimator
from concept_abstraction.selection import random_selection,greedy_selection_real_world,human_centered_selection_real_world
from concept_abstraction.env_utils import *
from concept_abstraction.utils import *
import sys 
import argparse
import secrets
import numpy as np 
import random 
import time 
from collections import Counter

is_jupyter = 'ipykernel' in sys.modules

# +
if is_jupyter: 
    seed        = 42
    environment_string = "cart_pole_binary"
    training_timesteps = 10000
    num_concepts_selected = 0
    selection_function = "policy"
    human_accuracy_by_concept = None 
    cbm_accuracy_by_concept = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1] 
    human_reliance_by_concept = None  
    target_abstraction = 0.05
    out_folder = "cart_pole"
    reward_error = 0
else:
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', help='Random Seed', type=int, default=42)
    parser.add_argument('--environment_string', help='Which environment to create', type=str, default="tree")
    parser.add_argument('--training_timesteps', help='Number of training timesteps', type=int, default=10000)
    parser.add_argument('--num_concepts_selected', help='Number of concepts selected by greedy or random',type=int, default=0)
    parser.add_argument('--selection_function', help='When selecting, use q_value, policy, or transition?', type=str, default="policy")
    parser.add_argument('--target_abstraction', help='Value for the target abstraction with human performance', type=float, default=0.05)
    parser.add_argument('--human_accuracy_by_concept', nargs='*', type=float, default=None)
    parser.add_argument('--cbm_accuracy_by_concept', help="What is the accuracy of AI per concept?", nargs='*', type=float, default=None)
    parser.add_argument('--human_reliance_by_concept', help="How much does AI rely on human intervention?",  nargs='*', type=float, default=None)
    parser.add_argument('--reward_error', help="How much to perturb the reward by?", type=float, default=0)
    parser.add_argument('--out_folder', help='Which folder', type=str, default="exploration")

    args = parser.parse_args()

    seed = args.seed
    environment_string = args.environment_string
    training_timesteps = args.training_timesteps 
    selection_function = args.selection_function
    num_concepts_selected = args.num_concepts_selected
    human_accuracy_by_concept = args.human_accuracy_by_concept
    human_reliance_by_concept = args.human_reliance_by_concept
    target_abstraction = args.target_abstraction
    cbm_accuracy_by_concept = args.cbm_accuracy_by_concept
    reward_error = args.reward_error
    out_folder = args.out_folder

save_name = secrets.token_hex(4)  
# -

results = {}
results['parameters'] = {'seed'      : seed,
        'environment_string'    : environment_string, 
        'training_timesteps': training_timesteps, 
        'selection_function': selection_function,
        'num_concepts_selected': num_concepts_selected,
        'human_accuracy_by_concept': human_accuracy_by_concept, 
        'human_reliance_by_concept': human_reliance_by_concept, 
        'target_abstraction': target_abstraction,
        'cbm_accuracy_by_concept': cbm_accuracy_by_concept,
        'reward_error': reward_error, 
}
print("Parameters {}".format(results['parameters']))

np.random.seed(seed)
random.seed(seed)

# ## State Abstractions

golden_model = get_golden_model(environment_string,reward_error)

# +
all_concepts = get_all_concepts(environment_string)
env = create_environment_from_string_real_world("cart_pole",get_all_concepts("cart_pole"),accuracies=None,reward_error=0)

transitions,rewards = get_transition_reward_rollout(golden_model,env)
all_states = list(transitions.keys())
all_states_array = np.array([[float(j) for j in i.split(" ")] for i in all_states])
env = create_environment_from_string_real_world(environment_string,get_all_concepts(environment_string)) 
all_binarized_concepts = [convert_env_state_to_concept(environment_string,env,i) for i in all_states_array]
env = create_environment_from_string_real_world("cart_pole",get_all_concepts("cart_pole"),accuracies=None,reward_error=0)

if selection_function == 'policy':
    state_values = golden_model.predict(all_states_array)[0].reshape(-1,1)
elif selection_function == 'q_value':
    state_dim = env.observation_space.shape[0]
    action_dim = 1
    q_estimator = SimpleQEstimator(state_dim, action_dim, golden_model)
    q_estimator.collect_and_train(env, num_episodes=100)
    state_values = np.array([[q_estimator.get_q_value(all_states_array[i],[0]),q_estimator.get_q_value(all_states_array[i],[1])] for i in range(len(all_states_array))])
elif selection_function == 'transition':
    transitions_by_concept = {}
    env = create_environment_from_string_real_world(environment_string,get_all_concepts(environment_string)) 
    for idx,state in enumerate(all_states):
        concept = tuple(all_binarized_concepts[idx])
        if concept not in transitions_by_concept:
            transitions_by_concept[concept] = {'0': [], '1': []}
        for (action,next_state) in transitions[state]:
            next_concept = convert_env_state_to_concept(environment_string,env,np.array([float(i) for i in next_state.split(" ")]))
            transitions_by_concept[concept][action].append(tuple(next_concept))
    for key in transitions_by_concept:
        for action in ['0','1']:
            transitions_by_concept[key][action] = Counter(transitions_by_concept[key][action])
    all_concepts = set(transitions_by_concept.keys())

    for key in transitions_by_concept:
        for action in ['0','1']:
            for val in transitions_by_concept[key][action]:
                if val not in all_concepts:
                    all_concepts.add(val)
    concept_to_idx = {}
    for idx,concept in enumerate(all_concepts):
        concept_to_idx[concept] = idx
    transitions_vector_by_concept = {}
    for key in transitions_by_concept:
        vec = np.zeros(len(concept_to_idx)*2)

        for action in ['0','1']:
            for key_2 in transitions_by_concept[key][action]:
                vec[int(action)*len(concept_to_idx) + concept_to_idx[key_2]] += transitions_by_concept[key][action][key_2]
        vec[:len(concept_to_idx)] = vec[:len(concept_to_idx)]/np.sum(vec[:len(concept_to_idx)]+0.00001)
        vec[len(concept_to_idx):] = vec[len(concept_to_idx):]/np.sum(vec[len(concept_to_idx):]+0.00001)
        transitions_vector_by_concept[key] = vec
    state_values = np.array([transitions_vector_by_concept[tuple(k)] for k in all_binarized_concepts])
# -

# ## Concept Selection

# +
selected_concepts = []
random_times = [] 
average_rewards = []
start = time.time()
for k in range(1,num_concepts_selected+1):
    if k > len(all_concepts):
        break 
    env = create_environment_from_string_real_world(environment_string,[0]) 
    random_concepts = random_selection(env,k)
    env = create_environment_from_string_real_world(environment_string,random_concepts,accuracies=None,reward_error=reward_error)
    model = train_ppo_model(env,total_timesteps=training_timesteps)
    selected_concepts.append(random_concepts)
    env = create_environment_from_string_real_world(environment_string,random_concepts,accuracies=None,reward_error=0)
    average_rewards.append(get_average_reward(env,model))
    random_times.append(time.time()-start)

results['random_selection'] = {
    'concepts': [i.tolist() for i in selected_concepts], 
    'values': average_rewards,
    'time': random_times, 
}

# +
selected_concepts = []
greedy_average_rewards = []
greedy_times = []
start = time.time() 
for k in range(1,num_concepts_selected+1):
    if k > len(env.concepts):
        break 
    
    greedy_concepts = greedy_selection_real_world(env,k,all_binarized_concepts,state_values)
    env = create_environment_from_string_real_world(environment_string,greedy_concepts,accuracies=None,reward_error=reward_error)
    model = train_ppo_model(env,total_timesteps=training_timesteps)
    selected_concepts.append(greedy_concepts)
    env = create_environment_from_string_real_world(environment_string,greedy_concepts,accuracies=None,reward_error=0)
    greedy_average_rewards.append(get_average_reward(env,model))
    greedy_times.append(time.time()-start)

results['greedy_selection'] = {
    'concepts': selected_concepts, 
    'values': greedy_average_rewards,
    'time': greedy_times,
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

    selected_concepts = human_centered_selection_real_world(env,np.array(modified_acc_rate),target_abstraction,np.array(all_binarized_concepts),state_values)
    selected_concepts = [idx for idx,i in enumerate(selected_concepts) if i>=0.5]


    env = create_environment_from_string_real_world(environment_string,selected_concepts,accuracies=modified_acc_rate)
    model = train_ppo_model(env,total_timesteps=training_timesteps)
    env = create_environment_from_string_real_world(environment_string,selected_concepts,accuracies=None)
    human_perf = get_average_reward(env,model)

    results['uncertainty'] = {
        'selected_concepts': selected_concepts,
        'combined_accuracies': modified_acc_rate,
        'combined_value': human_perf,
    }

# ## Save Data

save_path = get_save_path(out_folder,save_name)

delete_duplicate_results(out_folder,"",results)

json.dump(results,open('../../results/'+save_path,'w'))


