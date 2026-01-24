import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["TORCH_NUM_THREADS"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "0" 
os.environ["GRB_LICENSE_FILE"] = "/usr0/home/naveenr/gurobi.lic" 
os.environ['MKL_THREADING_LAYER'] = "GNU"

import torch 
from concept_abstraction.training import *
from concept_abstraction.selection import *
from concept_abstraction.concept_bank import *
from concept_abstraction.env_utils import *
from concept_abstraction.environments import *
from concept_abstraction.utils import *
import sys 
import argparse
import numpy as np 
import random 
import os
import pickle
import secrets 

import torch 
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
is_jupyter = 'ipykernel' in sys.modules
is_main = __name__ == "__main__"

if is_main:
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', help='Random Seed', type=int, default=42)
    parser.add_argument('--environment_string', help='Which environment to create', type=str, default="tree")
    parser.add_argument('--num_concepts_selected', help='Number of concepts selected by greedy or random',type=int, default=0)
    parser.add_argument('--gold_timesteps', help='Gold Timesteps',type=int, default=0)
    parser.add_argument('--method', help='Which method to use for comparison',type=str, default='random')    
    parser.add_argument('--out_folder', help='Which folder to write results to',type=str, default="basic")

    args = parser.parse_args()

    seed = args.seed
    environment_string = args.environment_string
    num_concepts_selected = args.num_concepts_selected
    gold_timesteps = args.gold_timesteps
    method = args.method 
    out_folder = args.out_folder


if is_main:
    results = {}
    results['parameters'] = {'seed'      : seed,
            'environment_string'    : environment_string, 
            'num_concepts_selected': num_concepts_selected,
            'method': method,
            'experiment': 'runtimes'
    }
    print("Parameters {}".format(results['parameters']))

if is_main:
    np.random.seed(seed)
    random.seed(seed)

if is_main:
    concept_list,processed_concepts = get_concepts(environment_string,"human_selected_binary",seed)
    num_concepts_selected = min(num_concepts_selected,len(concept_list))
    ground_truth_env, ground_truth_gym_env = get_environment(environment_string, None, seed)   
    model_name = "../../results/models/env={}_training={}_seed={}.zip".format(environment_string,gold_timesteps,seed)
    if os.path.exists(model_name) and method != "ground_truth":
        groundtruth_model = PPO.load(model_name)
    else:
        if "cyclic" in environment_string or "tree" in environment_string or "glucose" in environment_string:
            policy = "MlpPolicy"
        else:
            policy = "CnnPolicy"

        if policy == "MlpPolicy":
            groundtruth_model = train_ppo_model(ground_truth_env,environment_string+"_raw",total_timesteps=gold_timesteps,policy=policy)
        else:
            groundtruth_model = train_ppo_model(ground_truth_env,environment_string,total_timesteps=gold_timesteps,policy=policy)
        
        groundtruth_model.save(model_name)


if is_main and "multiple" in method:
    params = get_results_matching_parameters(
        "intervention", "",
        dict(
            environment_string=environment_string,
            num_concepts_selected=num_concepts_selected,
            method="multiple",
            intervention_prob=0.5,
        )
    )
    acc_list = np.array(params[0]["concept_accuracy"])


if is_main:    
    model_name = "../../results/q_estimates/env={}_training={}_seed={}_selection={}_source={}.pkl".format(environment_string,gold_timesteps,seed,"q_value","human_selected_binary")
    if os.path.exists(model_name) and method != "ground_truth":
        q_estimates = pickle.load(open(model_name,"rb"))

if is_main:
    start = time.time()

# # Basic Greedy
if is_main and method == 'entropy':
    subset_concept, idx = basic_greedy_selection(concept_list,num_concepts_selected,"q_value",q_estimates,"human_selected_binary")

# # Greedy
if is_main and method == 'greedy':
    subset_concept, idx = greedy_selection(concept_list,num_concepts_selected,"q_value",q_estimates,"human_selected_binary")

if is_main and method == 'multiple_log':
    subset_concept, idx = policy_coverage_selection_multiple_log(ground_truth_gym_env,concept_list,num_concepts_selected,groundtruth_model,q_estimates,acc_list)

if is_main and method == 'rho_075':
    subset_concept, idx = policy_coverage_selection_lp_hybrid(
    ground_truth_gym_env,
    concept_list,
    num_concepts_selected,
    groundtruth_model,
    q_estimates,coverage_ratio=0.75)

if is_main:
    time_taken = time.time()-start
    results['time_taken'] = time_taken 

if is_main:
    save_name = secrets.token_hex(4)  
    save_path = get_save_path(out_folder,save_name)
    delete_duplicate_results(out_folder,"",results)
    json.dump(results,open('../../results/'+save_path,'w'))

    ground_truth_env.close()
    ground_truth_gym_env.close()