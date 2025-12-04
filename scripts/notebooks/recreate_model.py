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
    parser.add_argument('--gold_timesteps', help='Number of training timesteps without concepts', type=int, default=10000)
    parser.add_argument("--train_ground_truth",action="store_true",help="Use ground-truth training data")
    parser.add_argument('--out_folder', help='Which folder to write results to',type=str, default="basic")

    args = parser.parse_args()

    seed = args.seed
    environment_string = args.environment_string
    gold_timesteps = args.gold_timesteps
    train_ground_truth = args.train_ground_truth
    out_folder = args.out_folder


if is_main:
        results = {}
        results['parameters'] = {'seed'      : seed,
                'gold_timesteps': gold_timesteps,
                'environment_string': environment_string,
                'experiment': 'training',
        }
        print("Parameters {}".format(results['parameters']))

if is_main:
    np.random.seed(seed)
    random.seed(seed)

if is_main:
    concept_list,processed_concepts = get_concepts(environment_string,"human_selected_binary",seed)
    ground_truth_env, ground_truth_gym_env = get_environment(environment_string, None, seed)   
    model_name = "../../results/models/env={}_training={}_seed={}.zip".format(environment_string,gold_timesteps,seed)
    if os.path.exists(model_name) and not train_ground_truth:
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
    groundtruth_reward = evaluate_model(environment_string,ground_truth_gym_env,groundtruth_model,seed)
    results['ground_truth'] = {'reward': groundtruth_reward}

# if is_main:
#     model_name = "../../results/models/concept_predictor_env={}_training={}_seed={}.pth".format(environment_string,100,seed)

#     height = width = 84

#     if environment_string == "mini_grid":
#         num_frames = 1
#     else:
#         num_frames = 4

#     if environment_string == "cart_pole":
#         height = 160
#         width = 240

#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     concept_predictor, acc_list = train_concept_predictor(ground_truth_gym_env,groundtruth_model,concept_list,list(range(len(concept_list))),environment_string,epochs=25,max_episode_length=10_000)
#     torch.save(concept_predictor.state_dict(), model_name)
#     concept_predictor.eval()
#     results["concept_accuracy"] = acc_list.tolist()
#     print("Concept Accuracy {}".format(acc_list.tolist()))

if is_main:    
    model_name = "../../results/q_estimates/env={}_training={}_seed={}_selection={}_source={}.pkl".format(environment_string,gold_timesteps,seed,"q_value","human_selected_binary")
    q_estimates = rollout_q_estimates_td(groundtruth_model,ground_truth_gym_env,concept_list)
    pickle.dump(q_estimates,open(model_name,"wb"))


if is_main:
    save_name = secrets.token_hex(4)  
    save_path = get_save_path(out_folder,save_name)
    delete_duplicate_results(out_folder,"",results)
    json.dump(results,open('../../results/'+save_path,'w'))

if is_main:
    ground_truth_env.close()
    ground_truth_gym_env.close()