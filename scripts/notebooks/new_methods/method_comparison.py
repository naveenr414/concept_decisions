import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["GRB_LICENSE_FILE"] = "/usr0/home/naveenr/gurobi.lic"
os.environ['MKL_THREADING_LAYER'] = "GNU"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"  
os.environ["NUMEXPR_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
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
import pickle

import torch 
torch.set_num_threads(2)
torch.set_num_interop_threads(1)
is_jupyter = 'ipykernel' in sys.modules
is_main = __name__ == "__main__"

if is_main:
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', help='Random Seed', type=int, default=42)
    parser.add_argument('--environment_string', help='Which environment to create', type=str, default="tree")
    parser.add_argument('--training_timesteps', help='Number of training timesteps without concepts', type=int, default=10000)
    parser.add_argument('--gold_timesteps', help='Number of training timesteps without concepts', type=int, default=10000)
    parser.add_argument('--num_concepts_selected', help='Number of concepts selected by greedy or random',type=int, default=0)

    args = parser.parse_args()

    seed = args.seed
    environment_string = args.environment_string
    gold_timesteps = args.gold_timesteps
    training_timesteps = args.training_timesteps 
    num_concepts_selected = args.num_concepts_selected


if is_main:
        results = {}
        results['parameters'] = {'seed'      : seed,
                'environment_string'    : environment_string, 
                'training_timesteps': training_timesteps, 
                'gold_timesteps': gold_timesteps,
                'num_concepts_selected': num_concepts_selected,
        }
        print("Parameters {}".format(results['parameters']))

if is_main:
    np.random.seed(seed)
    random.seed(seed)

if is_main:
    concept_list = get_concepts(environment_string,"human_selected_binary",seed)
    num_concepts_selected = min(num_concepts_selected,len(concept_list))
    ground_truth_env, ground_truth_gym_env, additional_info = get_environment(environment_string, None, seed)   
    model_name = "../../../results/models/env={}_training={}_seed={}.zip".format(environment_string,gold_timesteps,seed)
    groundtruth_model = PPO.load(model_name)

if is_main:
    model_name = "../../../results/models/concept_predictor_env={}_training={}_seed={}.pth".format(environment_string,100,seed)

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


if is_main and torch.cuda.is_available():
    concept_predictor = concept_predictor.cuda()
    concept_predictor.eval()
    
    # Warmup to allocate memory
    dummy_input = torch.zeros(8, num_frames, height, width, device='cuda')
    with torch.no_grad():
        _ = concept_predictor(dummy_input)


if is_main:    
    model_name = "../../../results/q_estimates/env={}_training={}_seed={}_selection={}_source={}.pkl".format(environment_string,gold_timesteps,seed,"q_value","human_selected_binary")
    if os.path.exists(model_name):
        q_estimates = pickle.load(open(model_name,"rb"))
    else:
        q_estimates = rollout_q_estimates_td(groundtruth_model,ground_truth_gym_env,concept_list)
        pickle.dump(q_estimates,open(model_name,"wb"))

if is_main:    
    fqe_name = "../../../results/models/fqe_env={}_seed={}.pth".format(environment_string,seed)
    env, eval_env, additional_info = get_environment(environment_string,concept_list,seed)
    optimal_model = train_ppo_model(env,environment_string,policy="MlpPolicy",total_timesteps=training_timesteps,custom_name="{}_optimal".format(environment_string))    
    
    if os.path.exists(fqe_name):
        fqe, states, state_mean, state_std, q_values  = load_fqe(fqe_name, eval_env, optimal_model)
    else:
        fqe,states, state_mean, state_std, q_values = train_fqe(eval_env,optimal_model)  
        fqe_name = "../../../results/models/fqe_env={}_seed={}.pth".format(environment_string,seed)
        save_fqe(fqe,fqe_name)

# ## Comparing Methods

# Random
if is_main:
    subset_concept, idx = random_selection(concept_list,num_concepts_selected)
    two_stage_env, two_stage_gym_env, additional_info = get_environment(environment_string,subset_concept,seed,fast_predictor=concept_predictor,use_processed=True,concept_idx=idx)
    model = train_ppo_model(two_stage_env,environment_string,policy="MlpPolicy",total_timesteps=training_timesteps,custom_name="{}_random".format(environment_string))    
    random_two_stage_reward = evaluate_model(environment_string,two_stage_gym_env,additional_info,model,seed)

# # Basic Greedy
if is_main:
    subset_concept, idx = basic_greedy_selection(concept_list,num_concepts_selected,"q_value",q_estimates,"human_selected_binary")
    two_stage_env, two_stage_gym_env, additional_info = get_environment(environment_string,subset_concept,seed,fast_predictor=concept_predictor,use_processed=True,concept_idx=idx)
    model = train_ppo_model(two_stage_env,environment_string,policy="MlpPolicy",total_timesteps=training_timesteps,custom_name="{}_basic_greedy".format(environment_string))    
    basic_greedy_two_stage_reward = evaluate_model(environment_string,two_stage_gym_env,additional_info,model,seed)

# # Greedy
if is_main:
    subset_concept, idx = greedy_selection(concept_list,num_concepts_selected,"q_value",q_estimates,"human_selected_binary")
    two_stage_env, two_stage_gym_env, additional_info = get_environment(environment_string,subset_concept,seed,fast_predictor=concept_predictor,use_processed=True,concept_idx=idx)
    model = train_ppo_model(two_stage_env,environment_string,policy="MlpPolicy",total_timesteps=training_timesteps,custom_name="{}_greedy".format(environment_string))    
    greedy_two_stage_reward = evaluate_model(environment_string,two_stage_gym_env,additional_info,model,seed)

# # LP
if is_main:
    subset_concept, idx = lp_based_selection(ground_truth_env,concept_list,num_concepts_selected,"q_value",q_estimates,"human_selected_binary")
    two_stage_env, two_stage_gym_env, additional_info = get_environment(environment_string,subset_concept,seed,fast_predictor=concept_predictor,use_processed=True,concept_idx=idx)
    model = train_ppo_model(two_stage_env,environment_string,policy="MlpPolicy",total_timesteps=training_timesteps,custom_name="{}_lp".format(environment_string))    
    lp_two_stage_reward = evaluate_model(environment_string,two_stage_gym_env,additional_info,model,seed)

# # Multiple
if is_main:
    subset_concept, idx = multiple_lp_selection(ground_truth_env,concept_list,num_concepts_selected,"q_value",q_estimates,"human_selected_binary")
    two_stage_env, two_stage_gym_env, additional_info = get_environment(environment_string,subset_concept,seed,fast_predictor=concept_predictor,use_processed=True,concept_idx=idx)
    model = train_ppo_model(two_stage_env,environment_string,policy="MlpPolicy",total_timesteps=training_timesteps,custom_name="{}_multiple".format(environment_string))    
    multiple_two_stage_reward = evaluate_model(environment_string,two_stage_gym_env,additional_info,model,seed)

# # +
# # LP with Fitted Q
if is_main:
    original_q_estimates = []
    s = ((states*state_std)+state_mean).astype(np.int8)

    for idx in range(len(states)):
        action_0 = q_values[idx][0]
        action_1 = q_values[idx][1]

        original_q_estimates.append((s[idx],0,action_0))
        original_q_estimates.append((s[idx],1,action_1))

    subset_concept, idx = lp_based_selection(ground_truth_env,concept_list,num_concepts_selected,"q_value",original_q_estimates,"human_selected_binary")
    two_stage_env, two_stage_gym_env, additional_info = get_environment(environment_string,subset_concept,seed,fast_predictor=concept_predictor,use_processed=True,concept_idx=idx)
    model = train_ppo_model(two_stage_env,environment_string,policy="MlpPolicy",total_timesteps=training_timesteps,custom_name="{}_fitted_q".format(environment_string))    
    fitted_lp_two_stage_reward = evaluate_model(environment_string,two_stage_gym_env,additional_info,model,seed)

# # LP with Vectors + Fitted Q
if is_main:
    subset_concept, idx = lp_vector_fqe_selection(states,fqe,state_mean,state_std,concept_list,num_concepts_selected)
    two_stage_env, two_stage_gym_env, additional_info = get_environment(environment_string,subset_concept,seed,fast_predictor=concept_predictor,use_processed=True,concept_idx=idx)
    model = train_ppo_model(two_stage_env,environment_string,policy="MlpPolicy",total_timesteps=training_timesteps,custom_name="{}_fitted_q_vector".format(environment_string))    
    vector_lp_two_stage_reward = evaluate_model(environment_string,two_stage_gym_env,additional_info,model,seed)

# # Mutual Information
if is_main:
    states_norm = ((states*state_std)+state_mean).astype(np.int8)
    subset_concept, idx = select_concepts_by_mi(states_norm, q_values,num_concepts_selected,concept_list)
    two_stage_env, two_stage_gym_env, additional_info = get_environment(environment_string,subset_concept,seed,fast_predictor=concept_predictor,use_processed=True,concept_idx=idx)
    model = train_ppo_model(two_stage_env,environment_string,policy="MlpPolicy",total_timesteps=training_timesteps,custom_name="{}_mutual".format(environment_string))    
    vector_lp_two_stage_reward = evaluate_model(environment_string,two_stage_gym_env,additional_info,model,seed)

if is_main:
    ground_truth_env.close()
    ground_truth_gym_env.close()
    two_stage_env.close()
    two_stage_gym_env.close()
