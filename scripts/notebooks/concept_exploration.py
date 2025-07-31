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

from start_line.plotting import *
from concept_abstraction.training import train_model
from concept_abstraction.selection import greedy_selection, random_selection, human_centered_selection
from concept_abstraction.env_utils import *
from concept_abstraction.utils import *
import sys 
import argparse
import secrets
import numpy as np 
import random 

is_jupyter = 'ipykernel' in sys.modules

# +
if is_jupyter: 
    seed        = 43
    environment_string = "cycle"
    out_folder = "exploration"
else:
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', help='Random Seed', type=int, default=42)
    parser.add_argument('--environment_string', help='Which environment to create', type=str, default="tree")
    parser.add_argument('--out_folder', help='Which folder', type=str, default="exploration")

    args = parser.parse_args()

    seed = args.seed
    environment_string = args.environment_string
    out_folder = args.out_folder

save_name = secrets.token_hex(4)  
# -

results = {}
results['parameters'] = {'seed'      : seed,
        'environment_string'    : environment_string}

np.random.seed(seed)
random.seed(seed)

# ## Concept Baseline

values_by_concept = []
baseline_concepts = get_baseline_concept_sets(environment_string)
for concept_list in baseline_concepts:
    print(concept_list)
    env = create_environment_from_string(environment_string,concept_list,0)
    q_net = train_model(env)
    values_by_concept.append(get_values(env,q_net))
results['baseline'] = {
    'concepts': baseline_concepts,
    'values': values_by_concept
}

# ## Concept Selection

selected_concepts = []
values_by_random_concept = []
for k in range(1,round(len(env.concepts)**0.5)+1):
    random_concepts = random_selection(env,k)
    env = create_environment_from_string(environment_string,random_concepts,0)
    q_net = train_model(env)
    selected_concepts.append(random_concepts)
    values_by_random_concept.append(get_values(env,q_net))
results['random_selection'] = {
    'concepts': [i.tolist() for i in selected_concepts], 
    'values': values_by_random_concept
}

selected_concepts = []
values_by_greedy_concept = []
for k in range(1,round(len(env.concepts)**0.5)+1):
    greedy_concepts = greedy_selection(env,k)
    env = create_environment_from_string(environment_string,greedy_concepts,0)
    q_net = train_model(env)
    selected_concepts.append(greedy_concepts)
    values_by_greedy_concept.append(get_values(env,q_net))
results['greedy_selection'] = {
    'concepts': selected_concepts, 
    'values': values_by_greedy_concept
}

# +
concept_list = list(range(env.concepts.shape[0]))
accuracy_by_concept = np.random.random(len(concept_list))
target_abstraction = np.random.random()*0.25

selected_concepts = human_centered_selection(env,accuracy_by_concept,target_abstraction)
selected_concepts = [concept_list[idx] for idx,i in enumerate(selected_concepts) if i>=0.5]
selected_accuracies = [accuracy_by_concept[idx] for idx,i in enumerate(selected_concepts) if i>=0.5]

env = create_environment_from_string(environment_string,selected_concepts,1-np.mean(selected_accuracies))
q_net = train_model(env)
human_perf = get_values(env,q_net)
results['human_selection'] = {
    'accuracies': accuracy_by_concept.tolist(),
    'target': target_abstraction,
    'concepts': selected_concepts,
    'values': values_by_greedy_concept,
}
# -

# ## Performance under Uncertainty

# +
epsilons = [0,0.01,0.1,0.25,0.5]
values_error = [[[] for _ in baseline_concepts[1:]] for _ in epsilons]

for idx,e in enumerate(epsilons):
    for jdx,concept_list in enumerate(baseline_concepts[1:]):
        env = create_environment_from_string(environment_string,concept_list,e)
        q_net = train_model(env)
        values_error[idx][jdx] = get_values(env,q_net)
results['uncertainty'] = {
    'epsilons': epsilons, 
    'concepts': baseline_concepts[1:],
    'values': values_error,
}
# -

# ## Save Data

save_path = get_save_path(out_folder,save_name)

delete_duplicate_results(out_folder,"",results)

json.dump(results,open('../../results/'+save_path,'w'))


