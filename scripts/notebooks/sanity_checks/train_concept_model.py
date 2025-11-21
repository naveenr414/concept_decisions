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

torch.cuda.set_per_process_memory_fraction(0.5)
torch.set_num_threads(1)
resource.setrlimit(resource.RLIMIT_AS, (30 * 1024 * 1024 * 1024, -1))


is_jupyter = 'ipykernel' in sys.modules
is_main = __name__ == "__main__"

if is_main:
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', help='Random Seed', type=int, default=42)
    parser.add_argument('--environment_string', help='Which environment to create', type=str, default="tree")
    parser.add_argument('--training_timesteps', help='Number of training timesteps without concepts', type=int, default=10000)

    args = parser.parse_args()

    seed = args.seed
    environment_string = args.environment_string
    training_timesteps = args.training_timesteps

if is_main:
        results = {}
        results['parameters'] = {'seed'      : seed,
                'environment_string'    : environment_string, 
                'training_timesteps': training_timesteps,
        }
        print("Parameters {}".format(results['parameters']))

if is_main:
    np.random.seed(seed)
    random.seed(seed)

# ### Basic Setup

if is_main:
    ground_truth_env, ground_truth_gym_env, additional_info = get_environment(environment_string, None, seed)   
    concept_list = get_concepts(environment_string,"human_selected_binary",seed)
    num_concepts_selected = len(concept_list)

if is_main:
    env, eval_env, additional_info = get_environment(environment_string,concept_list,seed)    
    custom_params = {}

    model = train_ppo_model(env,environment_string,total_timesteps=training_timesteps,policy="MlpPolicy",custom_name="{}_all_concepts".format(environment_string),override=custom_params)
    random_selection_reward = evaluate_model(environment_string,eval_env,additional_info,model,seed)
    print("Random Selection:",random_selection_reward)