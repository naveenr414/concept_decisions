import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
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
import resource

# torch.cuda.set_per_process_memory_fraction(0.5)
# torch.set_num_threads(1)
# resource.setrlimit(resource.RLIMIT_AS, (30 * 1024 * 1024 * 1024, -1))


is_jupyter = 'ipykernel' in sys.modules
is_main = __name__ == "__main__"

if is_main:
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', help='Random Seed', type=int, default=42)
    parser.add_argument('--environment_string', help='Which environment to create', type=str, default="tree")
    parser.add_argument('--gold_timesteps', help='Number of training timesteps without concepts', type=int, default=10000)

    args = parser.parse_args()

    seed = args.seed
    environment_string = args.environment_string
    gold_timesteps = args.gold_timesteps

if is_main:
    results = {}
    results['parameters'] = {'seed'      : seed,
            'environment_string'    : environment_string, 
            'gold_timesteps': gold_timesteps,
    }
    print("Parameters {}".format(results['parameters']))

if is_main:
    np.random.seed(seed)
    random.seed(seed)

# ### Basic Setup

if is_main:
    ground_truth_env, ground_truth_gym_env, additional_info = get_environment(environment_string, None, seed)   

if is_main:
    model_name = "../../../results/models/env={}_training={}_seed={}.zip".format(environment_string,gold_timesteps,seed)
    
    if "cyclic" in environment_string or "tree" in environment_string or "glucose" in environment_string:
        policy = "MlpPolicy"
    else:
        policy = "CnnPolicy"

    custom_params = {}
    
    if policy == "MlpPolicy":
        groundtruth_model = train_ppo_model(ground_truth_env,environment_string+"_raw",total_timesteps=gold_timesteps,policy=policy,override=custom_params)
    else:
        groundtruth_model = train_ppo_model(ground_truth_env,environment_string,total_timesteps=gold_timesteps,policy=policy,override=custom_params)
    groundtruth_model.save(model_name)
    groundtruth_reward = evaluate_model(environment_string,ground_truth_gym_env,additional_info,groundtruth_model,seed)
    results['ground_truth'] = {'reward':groundtruth_reward}
    print("Basic:",results['ground_truth']['reward'])