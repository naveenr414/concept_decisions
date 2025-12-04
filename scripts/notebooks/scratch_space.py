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

os.environ["CUDA_LAUNCH_BLOCKING"] = "0" 
os.environ["GRB_LICENSE_FILE"] = "/usr0/home/naveenr/gurobi.lic" 
os.environ['MKL_THREADING_LAYER'] = "GNU"
import torch 
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
import time 

is_jupyter = 'ipykernel' in sys.modules
is_main = __name__ == "__main__"

if is_main:
    seed = 42
    environment_string = "cart_pole"
    gold_timesteps = 4_000_000
    training_timesteps = 100_000 
    num_concepts_selected = 3
    out_folder = "basic"
    method = "lp" 


if is_main:
    for seed in [42,43,44]:
        concept_list, processed_concepts = get_concepts(environment_string,"human_selected_binary",seed)
        num_concepts_selected = min(num_concepts_selected,len(concept_list))
        ground_truth_env, ground_truth_gym_env = get_environment(environment_string, None, seed)   
        model_name = "../../results/models/env={}_training={}_seed={}.zip".format(environment_string,gold_timesteps,seed)
        if os.path.exists(model_name):
            groundtruth_model = PPO.load(model_name)
        q_estimates_full_random = rollout_q_estimates_td(groundtruth_model,ground_truth_gym_env,concept_list)
        subset_concept, idx = lp_based_selection(ground_truth_env,concept_list,num_concepts_selected,"q_value",q_estimates_full_random,"human_selected_binary")
        print(seed,idx)
        two_stage_env, two_stage_gym_env = get_environment(environment_string,subset_concept,seed,processed_concepts=processed_concepts,concept_idx=idx)
        model = train_ppo_model(two_stage_env,environment_string,policy="MlpPolicy",total_timesteps=250_000,custom_name="{}_perfect_lp_{}".format(environment_string,seed))

