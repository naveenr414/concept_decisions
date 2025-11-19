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
import secrets 
import itertools

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
    parser.add_argument('--num_samples', help='How many combinations we will subsample',type=int, default=20)
    parser.add_argument('--intervention_prob', help='Probability of intervention',type=float, default=0.5)
    parser.add_argument('--out_folder', help='Which folder to write results to',type=str, default="basic")

    args = parser.parse_args()

    seed = args.seed
    environment_string = args.environment_string
    gold_timesteps = args.gold_timesteps
    training_timesteps = args.training_timesteps 
    num_concepts_selected = args.num_concepts_selected
    num_samples = args.num_samples
    intervention_prob = args.intervention_prob
    out_folder = args.out_folder


if is_main:
        results = {}
        results['parameters'] = {'seed'      : seed,
                'environment_string'    : environment_string, 
                'training_timesteps': training_timesteps, 
                'gold_timesteps': gold_timesteps,
                'num_samples': num_samples, 
                'intervention_prob': intervention_prob,
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
    model_name = "../../results/models/env={}_training={}_seed={}.zip".format(environment_string,gold_timesteps,seed)
    if os.path.exists(model_name):
        groundtruth_model = PPO.load(model_name)
    else:
        policy = "CnnPolicy"
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

if is_main:
    unique_actions = list(set([int(i[1]) for i in q_estimates]))
    actions = np.array([i[1] for i in q_estimates])

    # Continuous
    discretized_X = np.array([i[0] for i in q_estimates])

    q_values = np.array([i[2] for i in q_estimates])

    final_vals = []
    seen = set()
    for a in unique_actions:
        relevant_idx = np.where(actions == a)[0]
        if len(relevant_idx) <= 500:
            relevant_low = relevant_high = relevant_idx
        else:
            relevant_low = np.argsort(np.abs(q_values))[:500]
            relevant_high = np.argsort(np.abs(q_values))[-500:]
        for low_idx in relevant_low:
            for high_idx in relevant_high:
                diff = abs(q_values[low_idx] - q_values[high_idx])
                # tuple of differing concept indices
                diffs = tuple(i for i, (l, h) in enumerate(zip(discretized_X[low_idx], discretized_X[high_idx])) if l != h)
                tup = (diff, diffs)
                if diffs not in seen and diffs != ():
                    seen.add(diffs)
                    final_vals.append(tup)
    final_vals = sorted(final_vals,reverse=True)

if is_main:
    def get_score(final_vals,idx):
        for j in final_vals:
            if set(j[1]).intersection(set(idx)) == set([]):
                break 
        else:
            return 0
        return j[0]

    # Compute pairs of accuracy x score
    all_pairs = []

    for combo in itertools.combinations(range(len(concept_list)), num_concepts_selected):
        average_accuracy = float(np.mean([acc_list[i] for i in combo]))
        score = float(get_score(final_vals,combo))
        all_pairs.append((average_accuracy,score))
    results['no_intervention'] = all_pairs 


    all_pairs_intervention = []

    for combo in itertools.combinations(range(len(concept_list)), num_concepts_selected):
        average_accuracy = float(np.mean([acc_list[i] for i in combo]))*(1-intervention_prob) + intervention_prob
        score = float(get_score(final_vals,combo))
        all_pairs_intervention.append((average_accuracy,score))
    results['intervention'] = all_pairs_intervention

# # LP
if is_main:
    # Select a subset of the all_pairs
    all_combos = list(itertools.combinations(range(len(concept_list)), num_concepts_selected))
    sampled = random.sample(all_combos, 20)
    sampled.append(lp_based_selection(ground_truth_env,concept_list,num_concepts_selected,"q_value",q_estimates,"human_selected_binary")[1])

    results['lp'] = {'combos': sampled, 'scores': []}

    for combo in sampled:
        average_accuracy = float(np.mean([acc_list[i] for i in combo]))
        score = float(get_score(final_vals,combo))
        idx = combo
        subset_concept = [concept_list[i] for i in idx]

        two_stage_env, two_stage_gym_env, additional_info = get_environment(environment_string,subset_concept,seed,fast_predictor=concept_predictor,use_processed=True,concept_idx=idx)
        model = train_ppo_model(two_stage_env,environment_string,policy="MlpPolicy",total_timesteps=training_timesteps,custom_name="{}_lp".format(environment_string))    
        reward_baseline = evaluate_model(environment_string,two_stage_gym_env,additional_info,model,seed)

        two_stage_env, two_stage_gym_env, additional_info = get_environment(environment_string,subset_concept,seed,fast_predictor=concept_predictor,use_processed=True,concept_idx=idx,intervention_prob=intervention_prob)
        model = train_ppo_model(two_stage_env,environment_string,policy="MlpPolicy",total_timesteps=training_timesteps,custom_name="{}_lp".format(environment_string))    
        reward_intervention = evaluate_model(environment_string,two_stage_gym_env,additional_info,model,seed)

        results['lp']['scores'].append({'reward_intervention': reward_intervention, 'reward_baseline': reward_baseline})

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
