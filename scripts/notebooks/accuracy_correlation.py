# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: food
#     language: python
#     name: python3
# ---

# %load_ext autoreload
# %autoreload 2

# +
import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["GRB_LICENSE_FILE"] = "/usr0/home/naveenr/gurobi.lic"
os.environ['MKL_THREADING_LAYER'] = "GNU"
# -

from concept_abstraction.training import *
from concept_abstraction.selection import *
from concept_abstraction.concept_bank import *
from concept_abstraction.env_utils import *
from concept_abstraction.environments import *
from concept_abstraction.utils import *
from concept_abstraction.environments import ConceptEnv
import sys 
import argparse
import secrets
import numpy as np 
import random 
import os
from stable_baselines3 import PPO
import pickle
import resource

torch.cuda.set_per_process_memory_fraction(0.5)
torch.set_num_threads(1)
resource.setrlimit(resource.RLIMIT_AS, (30 * 1024 * 1024 * 1024, -1))


is_jupyter = 'ipykernel' in sys.modules
is_main = __name__ == "__main__"

if is_main:
    if is_jupyter: 
        # Basics 
        seed        = 42
        environment_string = "mini_grid"
        gold_timesteps = 4_000_000
        training_timesteps = 250_000
        num_concepts_selected = 40
        num_trials = 10
        out_folder = "correlation"
        cbm_accuracy_by_concept = [0.75 for i in range(44)]
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument('--seed', help='Random Seed', type=int, default=42)
        parser.add_argument('--environment_string', help='Which environment to create', type=str, default="tree")
        parser.add_argument('--training_timesteps', help='Number of training timesteps', type=int, default=10000)
        parser.add_argument('--gold_timesteps', help='Number of training timesteps without concepts', type=int, default=10000)
        parser.add_argument('--num_concepts_selected', help='Number of concepts selected by greedy or random',type=int, default=0)
        parser.add_argument('--num_trials', help='How many trials to run this for',type=int, default=10)
        parser.add_argument('--cbm_accuracy_by_concept', help="What is the accuracy of AI per concept?", nargs='*', type=float, default=None)
        parser.add_argument('--out_folder', help='Which folder', type=str, default="exploration")

        args = parser.parse_args()

        seed = args.seed
        environment_string = args.environment_string
        training_timesteps = args.training_timesteps 
        gold_timesteps = args.gold_timesteps
        num_concepts_selected = args.num_concepts_selected
        num_trials = args.num_trials
        cbm_accuracy_by_concept = args.cbm_accuracy_by_concept
        out_folder = args.out_folder
    
    save_name = secrets.token_hex(4)  

if is_main:
        results = {}
        results['parameters'] = {'seed'      : seed,
                'environment_string'    : environment_string, 
                'training_timesteps': training_timesteps, 
                'gold_timesteps': gold_timesteps,
                'num_concepts_selected': num_concepts_selected,
                'num_trials': num_trials,
                'cbm_accuracy_by_concept': cbm_accuracy_by_concept,
        }
        print("Parameters {}".format(results['parameters']))

if is_main:
    np.random.seed(seed)
    random.seed(seed)

# ### Basic Setup

if is_main:
    concept_list = get_concepts(environment_string,"human_selected_binary",seed)
    num_concepts_selected = min(num_concepts_selected,len(concept_list))
    ground_truth_env, ground_truth_gym_env, additional_info = get_environment(environment_string, None, seed)   

if is_main:
    model_name = "../../results/models/env={}_training={}_seed={}.zip".format(environment_string,gold_timesteps,seed)
    
    if os.path.exists(model_name):
        print("Model exists!")
        groundtruth_model = PPO.load(model_name)
        additional_info['subset_concepts'] = get_concepts(environment_string,"human_selected",seed)
    else:
        if "cyclic" in environment_string or "tree" in environment_string or "glucose" in environment_string:
            policy = "MlpPolicy"
        else:
            policy = "CnnPolicy"
        
        if environment_string == "mimic":
            additional_info['subset_concepts'] = get_concepts(environment_string,"human_selected",seed)
            groundtruth_model = train_ppo_model(ground_truth_env,"mimic_raw",total_timesteps=gold_timesteps,policy=policy,)
        else:
            groundtruth_model = train_ppo_model(ground_truth_env,environment_string,total_timesteps=gold_timesteps,policy=policy)
        groundtruth_model.save(model_name)
    groundtruth_reward = evaluate_model(environment_string,ground_truth_gym_env,additional_info,groundtruth_model,seed)
    results['ground_truth'] = {'reward':groundtruth_reward}
    print("Basic:",results['ground_truth']['reward'])

# ### Basic Comparison
if is_main:
    all_results = []
    for i in range(num_trials):
        results['random_comparison'] = {}
        modified_concept_predictors = [inaccurate_concepts_binary(func,acc,seed) for (func,acc) in zip(concept_list,cbm_accuracy_by_concept)]
        subset_concept, random_idx = random_selection(modified_concept_predictors,num_concepts_selected)
        
        # Imperfect concepts
        env, eval_env, additional_info = get_environment(environment_string,subset_concept,seed)
        try:
            model = train_ppo_model(env,environment_string,total_timesteps=training_timesteps,policy="MlpPolicy",custom_name="{}_error_random_imperfect".format(environment_string),silent=True)
            reward_imperfect = evaluate_model(environment_string,eval_env,additional_info,model,seed)
        finally:
            # Clean up
            env.close()
            eval_env.close()
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Perfect concepts
        subset_concept = [concept_list[i] for i in random_idx]
        env, eval_env, additional_info = get_environment(environment_string,subset_concept,seed)
        try:
            model = train_ppo_model(env,environment_string,total_timesteps=training_timesteps,policy="MlpPolicy",custom_name="{}_error_random_perfect".format(environment_string),silent=True)
            reward_perfect = evaluate_model(environment_string,eval_env,additional_info,model,seed)
        finally:
            # Clean up
            env.close()
            eval_env.close()
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        all_results.append((reward_perfect,reward_imperfect))
    
    print(all_results)
    results['random_comparison'] = all_results


# ## Save Data

if is_main:
    save_path = get_save_path(out_folder,save_name)

if is_main:
    delete_duplicate_results(out_folder,"",results)

if is_main:
    json.dump(results,open('../../results/'+save_path,'w'))

if is_main:
    ground_truth_env.close()
    ground_truth_gym_env.close()
    env.close()
    eval_env.close()
