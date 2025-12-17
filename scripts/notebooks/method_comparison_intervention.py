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

is_jupyter = 'ipykernel' in sys.modules
is_main = __name__ == "__main__"

if is_main:
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', help='Random Seed', type=int, default=42)
    parser.add_argument('--environment_string', help='Which environment to create', type=str, default="tree")
    parser.add_argument('--training_timesteps', help='Number of training timesteps without concepts', type=int, default=10000)
    parser.add_argument('--gold_timesteps', help='Number of training timesteps without concepts', type=int, default=10000)
    parser.add_argument('--num_concepts_selected', help='Number of concepts selected by greedy or random',type=int, default=0)
    parser.add_argument('--intervention_prob', help='Intervention Accuracy',type=float, default=0.0)
    parser.add_argument('--method', help='Which selection method',type=str, default='random')
    parser.add_argument('--out_folder', help='Which folder to write results to',type=str, default="basic")

    args = parser.parse_args()

    seed = args.seed
    environment_string = args.environment_string
    gold_timesteps = args.gold_timesteps
    training_timesteps = args.training_timesteps 
    num_concepts_selected = args.num_concepts_selected
    intervention_prob = args.intervention_prob
    method = args.method 
    out_folder = args.out_folder


if is_main:
        results = {}
        results['parameters'] = {'seed'      : seed,
                'environment_string'    : environment_string, 
                'training_timesteps': training_timesteps, 
                'gold_timesteps': gold_timesteps,
                'num_concepts_selected': num_concepts_selected,
                'intervention_prob': intervention_prob,
                'method': method
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
        
        groundtruth_model = train_ppo_model(ground_truth_env,environment_string,total_timesteps=gold_timesteps,policy=policy)
        groundtruth_model.save(model_name)

if is_main:
    model_name = "../../results/models/concept_predictor_env={}_training={}_seed={}.pth".format(environment_string,100,seed)

    height = width = 84

    if environment_string == "mini_grid":
        num_frames = 1
    else:
        num_frames = 4

    if environment_string == "cart_pole":
        height = 160
        width = 240

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if os.path.exists(model_name):
        concept_predictor = ConceptPredictorCNN(len(concept_list), num_frames=num_frames,height=height,width=width).to(device)
        concept_predictor.load_state_dict(torch.load(model_name, weights_only=True))
        concept_predictor.eval()
    else:
        concept_predictor, acc_list = train_concept_predictor(ground_truth_gym_env,groundtruth_model,concept_list,list(range(len(concept_list))),environment_string,epochs=25,max_episode_length=10_000)
        torch.save(concept_predictor.state_dict(), model_name)
        concept_predictor.eval()

if is_main and (method == "multiple" or method == "policy_selection_multiple") and 'concept_accuracy' not in results:
    matching_params = get_results_matching_parameters("training","",{'environment_string': environment_string,
                'gold_timesteps': gold_timesteps, 'seed': seed})
    if len(matching_params) > 0 and 'concept_accuracy' in matching_params[0]:
        acc_list = matching_params[0]['concept_accuracy']
        results["concept_accuracy"] = acc_list
    else:
        acc_list = evaluate_concept_predictor(concept_predictor,ground_truth_gym_env,groundtruth_model,concept_list)
        results["concept_accuracy"] = acc_list.tolist()

if is_main and torch.cuda.is_available():
    concept_predictor = concept_predictor.cuda()
    concept_predictor.eval()
    
    # Warmup to allocate memory
    dummy_input = torch.zeros(8, num_frames, height, width, device='cuda')
    with torch.no_grad():
        _ = concept_predictor(dummy_input)


if is_main:    
    model_name = "../../results/q_estimates/env={}_training={}_seed={}_selection={}_source={}.pkl".format(environment_string,gold_timesteps,seed,"q_value","human_selected_binary")
    if os.path.exists(model_name):
        q_estimates = pickle.load(open(model_name,"rb"))
    else:
        q_estimates = rollout_q_estimates_td(groundtruth_model,ground_truth_gym_env,concept_list)
        pickle.dump(q_estimates,open(model_name,"wb"))

# All Concept Model 
if is_main and method == 'imperfect_concepts': 
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

# # Multiple
if is_main and method == 'multiple':
    subset_concept, idx = multiple_lp_selection(ground_truth_env,concept_list,num_concepts_selected,"q_value",q_estimates,"human_selected_binary",[acc for acc in acc_list])

if is_main and method == 'policy_selection_lp':
    subset_concept, idx = policy_coverage_selection_lp(ground_truth_gym_env,concept_list,num_concepts_selected,groundtruth_model)

if is_main and method == 'policy_selection_td':
    full_two_stage_env, full_two_stage_gym_env = get_environment(environment_string,concept_list,seed,processed_concepts=processed_concepts,concept_idx=list(range(len(concept_list))))
    
    if environment_string in ["cart_pole","mini_grid"]:
        all_concept_model = train_ppo_model(full_two_stage_env,environment_string,policy="MlpPolicy",total_timesteps=250_000,custom_name="{}_perfect_greedy_{}".format(environment_string,seed)) 
    elif environment_string == "glucose":
        all_concept_model = train_ppo_model(full_two_stage_env,environment_string,policy="MlpPolicy",total_timesteps=1_000_000,custom_name="{}_perfect_greedy_{}".format(environment_string,seed)) 
    else:
        all_concept_model = train_ppo_model(full_two_stage_env,environment_string,policy="MlpPolicy",total_timesteps=2_000_000,custom_name="{}_perfect_greedy_{}".format(environment_string,seed)) 

    subset_concept, idx = policy_coverage_selection_lp_advantage(full_two_stage_gym_env,concept_list,num_concepts_selected,all_concept_model)


if is_main and method == 'policy_selection_multiple':
    subset_concept, idx = policy_coverage_selection_exp_lp(ground_truth_gym_env,concept_list,acc_list,num_concepts_selected,groundtruth_model)

if is_main:
    two_stage_env, two_stage_gym_env = get_environment(environment_string,subset_concept,seed,fast_predictor=concept_predictor,use_processed=True,concept_idx=idx,intervention_prob=intervention_prob,processed_concepts=processed_concepts)
    model = train_ppo_model(two_stage_env,environment_string,policy="MlpPolicy",total_timesteps=training_timesteps,custom_name="{}_intervention_{}_{}".format(environment_string,method,intervention_prob))    
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
