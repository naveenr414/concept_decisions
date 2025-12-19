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

is_jupyter = 'ipykernel' in sys.modules
is_main = __name__ == "__main__"

if is_main:
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', help='Random Seed', type=int, default=42)
    parser.add_argument('--environment_string', help='Which environment to create', type=str, default="tree")
    parser.add_argument('--training_timesteps', help='Number of training timesteps without concepts', type=int, default=10000)
    parser.add_argument('--gold_timesteps', help='Number of training timesteps without concepts', type=int, default=10000)
    parser.add_argument('--num_concepts_selected', help='Number of concepts selected by greedy or random',type=int, default=0)
    parser.add_argument('--method', help='Which method to use for comparison',type=str, default='random')    
    parser.add_argument('--out_folder', help='Which folder to write results to',type=str, default="basic")

    args = parser.parse_args()

    seed = args.seed
    environment_string = args.environment_string
    gold_timesteps = args.gold_timesteps
    training_timesteps = args.training_timesteps 
    num_concepts_selected = args.num_concepts_selected
    out_folder = args.out_folder
    method = args.method 


if is_main:
        results = {}
        results['parameters'] = {'seed'      : seed,
                'environment_string'    : environment_string, 
                'training_timesteps': training_timesteps, 
                'gold_timesteps': gold_timesteps,
                'num_concepts_selected': num_concepts_selected,
                'method': method,
                'experiment': 'perfect_comparison'
        }
        print("Parameters {}".format(results['parameters']))

if is_main:
    np.random.seed(seed)
    random.seed(seed)

if is_main:
    concept_list, processed_concepts = get_concepts(environment_string,"human_selected_binary",seed)
    num_concepts_selected = min(num_concepts_selected,len(concept_list))
    ground_truth_env, ground_truth_gym_env = get_environment(environment_string, None, seed)   
    model_name = "../../results/models/env={}_training={}_seed={}.zip".format(environment_string,gold_timesteps,seed)
    if os.path.exists(model_name):
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
    
    if method == 'perfect_concepts':
        groundtruth_reward = evaluate_model(environment_string,ground_truth_gym_env,groundtruth_model,seed)
        results['ground_truth'] = {'reward': groundtruth_reward}


if is_main:    
    model_name = "../../results/q_estimates/env={}_training={}_seed={}_selection={}_source={}.pkl".format(environment_string,gold_timesteps,seed,"q_value","human_selected_binary")
    if os.path.exists(model_name):
        q_estimates = pickle.load(open(model_name,"rb"))
    else:
        q_estimates = rollout_q_estimates_td(groundtruth_model,ground_truth_gym_env,concept_list)
        pickle.dump(q_estimates,open(model_name,"wb"))

# Perfect Concepts 
if is_main and method == 'perfect_concepts':    
    subset_concept = concept_list 
    idx = list(range(len(concept_list)))

# Random
if is_main and method == 'random': 
    subset_concept, idx = random_selection(concept_list,num_concepts_selected)

# # Basic Greedy
if is_main and method == 'entropy':
    subset_concept, idx = basic_greedy_selection(concept_list,num_concepts_selected,"q_value",q_estimates,"human_selected_binary")

# # Greedy
if is_main and method == 'greedy':
    subset_concept, idx = greedy_selection(concept_list,num_concepts_selected,"q_value",q_estimates,"human_selected_binary")

# # LP
if is_main and method == 'lp':
    subset_concept, idx = lp_based_selection(ground_truth_env,concept_list,num_concepts_selected,"q_value",q_estimates,"human_selected_binary")

if is_main and method == 'lp_hybrid':
    subset_concept, idx = policy_coverage_selection_lp_hybrid(
    ground_truth_gym_env,
    concept_list,
    num_concepts_selected,
    groundtruth_model,
    q_estimates)

if is_main and method == 'lp_weighted':
    subset_concept, idx = policy_coverage_selection_lp_weighted(
    ground_truth_gym_env,
    concept_list,
    num_concepts_selected,
    groundtruth_model,
    q_estimates)


if is_main and method == 'lp_old':
    subset_concept, idx = lp_based_selection_old(ground_truth_env,concept_list,num_concepts_selected,"q_value",q_estimates,"human_selected_binary")

if is_main and method == 'policy_selection_lp':
    subset_concept, idx = policy_coverage_selection_lp(ground_truth_gym_env,concept_list,num_concepts_selected,groundtruth_model)

if is_main and method == 'policy_selection_td':
    subset_concept, idx = policy_coverage_selection_lp_advantage(ground_truth_gym_env,concept_list,num_concepts_selected,groundtruth_model)

if is_main:
    two_stage_env, two_stage_gym_env = get_environment(environment_string,subset_concept,seed,processed_concepts=processed_concepts,concept_idx=idx)
    model = train_ppo_model(two_stage_env,environment_string,policy="MlpPolicy",total_timesteps=training_timesteps,custom_name="{}_perfect_{}_{}".format(environment_string,method,seed))    
    reward = evaluate_model(environment_string,two_stage_gym_env,model,seed)
    results[method] = {'reward': reward, 'concepts': idx}

if is_main:
    save_name = secrets.token_hex(4)  
    save_path = get_save_path(out_folder,save_name)
    delete_duplicate_results(out_folder,"",results)
    json.dump(results,open('../../results/'+save_path,'w'))

if is_main:
    ground_truth_env.close()
    ground_truth_gym_env.close()
    two_stage_env.close()
    two_stage_gym_env.close()
