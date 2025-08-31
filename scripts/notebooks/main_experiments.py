# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: food
#     language: python
#     name: python3
# ---

# %load_ext autoreload
# %autoreload 2

# +
# TODO: 
# 2) Get LLM Concepts + Implement that 
# 3) Get LLM Labels + Implemenet Two-Stage training with that
# 4) Create Concept Completeness Baseline
# 5) Figure out the right target abstraction

# +
from concept_abstraction.training import train_ppo_model, evaluate_model, RandomAgent, train_two_stage_ppo_model
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
import time 
from collections import Counter
import scipy
# -

is_jupyter = 'ipykernel' in sys.modules

# +
if is_jupyter: 
    # Basics 
    seed        = 42
    environment_string = "mimic"
    training_timesteps = 5_000
    num_concepts_selected = 8
    selection_function = "policy"
    # Experiment #1 & #2
    run_basic = True
    run_iterative = False
    run_two_stage = False
    # Experiment #3
    cbm_accuracy_by_concept = None   
    cbm_std_by_concept = None 
    target_abstraction = 0.05
    reward_error = 0
    # Experiment #4
    concept_source = "human_selected_binary"
    # Experiment #5
    assess_completeness=False
    out_folder = "basic"
else:
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', help='Random Seed', type=int, default=42)
    parser.add_argument('--environment_string', help='Which environment to create', type=str, default="tree")
    parser.add_argument('--training_timesteps', help='Number of training timesteps', type=int, default=10000)
    parser.add_argument('--num_concepts_selected', help='Number of concepts selected by greedy or random',type=int, default=0)
    parser.add_argument('--selection_function', help='When selecting, use q_value, policy, or transition?', type=str, default="policy")
    parser.add_argument('--cbm_accuracy_by_concept', help="What is the accuracy of AI per concept?", nargs='*', type=float, default=None)
    parser.add_argument('--cbm_std_by_concept', help="What is the error of AI per concept?", nargs='*', type=float, default=None)
    parser.add_argument('--run_two_stage', help='Run the basic comparisons?', action='store_true')
    parser.add_argument('--run_iterative', help='Run the basic comparisons?', action='store_true')
    parser.add_argument('--run_basic', help='Run the basic comparisons?', action='store_true')
    parser.add_argument('--target_abstraction', help='Value for the target abstraction with human performance', type=float, default=0.05)
    parser.add_argument('--reward_error', help="How much to perturb the reward by?", type=float, default=0)
    parser.add_argument('--concept_source', help='When selecting, use q_value, policy, or transition?', type=str, default="human_selected")
    parser.add_argument('--assess_completeness', help='Compare to the concept completeness algorithm?', action='store_true')
    parser.add_argument('--out_folder', help='Which folder', type=str, default="exploration")

    args = parser.parse_args()

    seed = args.seed
    environment_string = args.environment_string
    training_timesteps = args.training_timesteps 
    num_concepts_selected = args.num_concepts_selected
    selection_function = args.selection_function
    cbm_accuracy_by_concept = args.cbm_accuracy_by_concept
    cbm_std_by_concept = args.cbm_std_by_concept
    run_basic = args.run_basic
    run_iterative = args.run_iterative
    run_two_stage = args.run_two_stage
    target_abstraction = args.target_abstraction
    reward_error = args.reward_error
    concept_source = args.concept_source
    assess_completeness = args.assess_completeness
    out_folder = args.out_folder

save_name = secrets.token_hex(4)  
# -

results = {}
results['parameters'] = {'seed'      : seed,
        'environment_string'    : environment_string, 
        'training_timesteps': training_timesteps, 
        'selection_function': selection_function,
        'num_concepts_selected': num_concepts_selected,
        'cbm_accuracy_by_concept': cbm_accuracy_by_concept,
        'cbm_std_by_concept': cbm_std_by_concept,
        'target_abstraction': target_abstraction,
        'reward_error': reward_error, 
        'concept_source': concept_source,
        'assess_completeness': assess_completeness,
        'run_basic': run_basic,
        'run_iterative': run_iterative, 
        'run_two_stage': run_two_stage, 
}
print("Parameters {}".format(results['parameters']))

np.random.seed(seed)
random.seed(seed)

# # ### Basic Setup

# concept_list = get_concepts(environment_string,concept_source,seed)

# ground_truth_env, ground_truth_eval_env, additional_info = get_environment(environment_string,None,seed)
# if environment_string == 'mimic': 
#     concept_list = additional_info['concept_list']

# # Train the groundtruth policy
# if "cyclic" in environment_string or "tree" in environment_string or "mimic" in environment_string or "mini_grid" in environment_string:
#     policy = "MlpPolicy"
# else:
#     policy = "CnnPolicy"
# groundtruth_model = train_ppo_model(ground_truth_env,total_timesteps=training_timesteps,policy=policy)
# groundtruth_reward = evaluate_model(environment_string,ground_truth_eval_env,additional_info,groundtruth_model,seed)
# results['ground_truth'] = {'reward':groundtruth_reward}
# print(results['ground_truth']['reward'])

# # ### Basic Comparison

# if run_basic:
#     results['basic_comparison'] = {}

# if run_basic:
#     # Train a random policy
#     model = RandomAgent(ground_truth_env)
#     random_reward = evaluate_model(environment_string,ground_truth_eval_env,additional_info,model,seed)
#     results['basic_comparison']['random'] = {'reward':random_reward}
#     print(results['basic_comparison']['random']['reward'])

# if run_basic:
#     # Train a random selector
#     subset_concept = random_selection(concept_list,num_concepts_selected)
#     env, eval_env, additional_info = get_environment(environment_string,subset_concept,seed)
#     model = train_ppo_model(env,total_timesteps=training_timesteps,policy="MlpPolicy")
#     random_selection_reward = evaluate_model(environment_string,eval_env,additional_info,model,seed)
#     results['basic_comparison']['random_selection'] = {'reward':random_selection_reward}
#     print(results['basic_comparison']['random_selection']['reward'])

# if run_basic:
#     # Train a greedy selector
#     subset_concept, greedy_idx = greedy_selection(ground_truth_eval_env,concept_list,num_concepts_selected,groundtruth_model,selection_function)
#     env, eval_env, additional_info = get_environment(environment_string,subset_concept,seed)
#     model = train_ppo_model(env,total_timesteps=training_timesteps,policy="MlpPolicy")
#     greedy_selection_reward = evaluate_model(environment_string,eval_env,additional_info,model,seed)
#     results['basic_comparison']['greedy'] = {'concepts': greedy_idx, 'reward':greedy_selection_reward}
#     print(results['basic_comparison']['greedy']['reward'])


# if run_basic:
#     subset_concept, greedy_iterative_idx = greedy_iterative_selection(ground_truth_eval_env,concept_list,num_concepts_selected,groundtruth_model,selection_function)
#     env, eval_env, additional_info = get_environment(environment_string,subset_concept,seed)
#     model = train_ppo_model(env,total_timesteps=training_timesteps,policy="MlpPolicy")
#     greedy_iterative_selection_reward = evaluate_model(environment_string,eval_env,additional_info,model,seed)
#     results['basic_comparison']['greedy_iterative'] = {'concepts': greedy_iterative_idx, 'reward': greedy_iterative_selection_reward}
#     print(results['basic_comparison']['greedy_iterative']['reward'])

# if run_basic:
#     subset_concept, lp_idx = lp_based_selection(ground_truth_eval_env,concept_list,num_concepts_selected,groundtruth_model,selection_function,target_abstraction)
#     env, eval_env, additional_info = get_environment(environment_string,subset_concept,seed)
#     model = train_ppo_model(env,total_timesteps=training_timesteps,policy="MlpPolicy")
#     lp_selection_reward = evaluate_model(environment_string,eval_env,additional_info,model,seed)
#     results['basic_comparison']['lp'] = {'concepts': lp_idx, 'reward': lp_selection_reward}
#     print(results['basic_comparison']['lp']['reward'])

# # ### Imperfect Concept Predictors

# if cbm_std_by_concept is not None or cbm_accuracy_by_concept is not None:
#     results['inaccurate_comparison'] = {}


# # +
# greedy_inaccurate_reward = {}
# for modification in ["continuous","binary"]:
#     if modification == "continuous" and cbm_std_by_concept is not None:
#         modified_concept_predictors = [inaccurate_concepts_continuous(func,0,std) for func,std in zip(concept_list,cbm_std_by_concept)]
#     elif modification == "binary" and cbm_accuracy_by_concept is not None:
#         modified_concept_predictors = [inaccurate_concepts_binary(func,acc) for func,acc in zip(concept_list,cbm_accuracy_by_concept)]
#     else:
#         continue 
#     subset_concept = [modified_concept_predictors[i] for i in greedy_idx]
#     env, eval_env, additional_info = get_environment(environment_string,subset_concept,seed)
#     model = train_ppo_model(env,total_timesteps=training_timesteps,policy="MlpPolicy")
#     greedy_inaccurate_reward[modification] = {
#         'reward': evaluate_model(environment_string,eval_env,additional_info,model,seed),
#         'concepts': greedy_idx
#     }

# if greedy_inaccurate_reward != {}:
#     results['inaccurate_comparison']['greedy'] = greedy_inaccurate_reward
#     print(greedy_inaccurate_reward)

# # +
# greedy_iterative_inaccurate_reward = {}
# for modification in ["continuous","binary"]:
#     if modification == "continuous" and cbm_std_by_concept is not None:
#         modified_concept_predictors = [inaccurate_concepts_continuous(func,0,std) for func,std in zip(concept_list,cbm_std_by_concept)]
#     elif modification == "binary" and cbm_accuracy_by_concept is not None:
#         modified_concept_predictors = [inaccurate_concepts_binary(func,acc) for func,acc in zip(concept_list,cbm_accuracy_by_concept)]
#     else:
#         continue 
    
#     subset_concept = [modified_concept_predictors[i] for i in greedy_iterative_idx]
#     env, eval_env, additional_info = get_environment(environment_string,subset_concept,seed)
#     model = train_ppo_model(env,total_timesteps=training_timesteps,policy="MlpPolicy")
#     greedy_iterative_inaccurate_reward[modification] = {
#         'reward': evaluate_model(environment_string,eval_env,additional_info,model,seed), 
#         'concepts': greedy_iterative_idx
#     } 

# if greedy_iterative_inaccurate_reward != {}:
#     results['inaccurate_comparison']['greedy_iterative'] = greedy_iterative_inaccurate_reward
#     print(greedy_iterative_inaccurate_reward)

# # +
# lp_inaccurate_reward = {}
# for modification in ["continuous","binary"]:
#     if modification == "continuous" and cbm_std_by_concept is not None:
#         modified_concept_predictors = [inaccurate_concepts_continuous(func,0,std) for func,std in zip(concept_list,cbm_std_by_concept)]
#     elif modification == "binary" and cbm_accuracy_by_concept is not None:
#         modified_concept_predictors = [inaccurate_concepts_binary(func,acc) for func,acc in zip(concept_list,cbm_accuracy_by_concept)]
#     else:
#         continue 
    
#     subset_concept = [modified_concept_predictors[i] for i in lp_idx]
#     env, eval_env, additional_info = get_environment(environment_string,subset_concept,seed)
#     model = train_ppo_model(env,total_timesteps=training_timesteps,policy="MlpPolicy")
#     lp_inaccurate_reward[modification] = { 'reward': evaluate_model(environment_string,eval_env,additional_info,model,seed),
#     'concepts': lp_idx}

# if lp_inaccurate_reward != {}:
#     results['inaccurate_comparison']['lp'] = lp_inaccurate_reward
#     print(lp_inaccurate_reward)

# # +
# imperfect_lp_selection_reward = {}
# for modification in ["continuous","binary"]:
#     if modification == "continuous" and cbm_std_by_concept is not None:
#         modified_concept_predictors = [inaccurate_concepts_continuous(func,0,std) for func,std in zip(concept_list,cbm_std_by_concept)]
#     elif modification == "binary" and cbm_accuracy_by_concept is not None:
#         modified_concept_predictors = [inaccurate_concepts_binary(func,acc) for func,acc in zip(concept_list,cbm_accuracy_by_concept)]
#     else:
#         continue 
    
#     if modification == "continuous":
#         subset_concept, imperfect_idx = imperfect_lp_selection(ground_truth_eval_env,modified_concept_predictors,groundtruth_model,selection_function,target_abstraction,cbm_std_by_concept,direction='min')
#     else:
#         subset_concept, imperfect_idx = imperfect_lp_selection(ground_truth_eval_env,modified_concept_predictors,groundtruth_model,selection_function,target_abstraction,cbm_accuracy_by_concept,direction='max')
#     env, eval_env, additional_info = get_environment(environment_string,subset_concept,seed)
#     model = train_ppo_model(env,total_timesteps=training_timesteps,policy="MlpPolicy")
#     imperfect_lp_selection_reward[modification] = evaluate_model(environment_string,eval_env,additional_info,model,seed)

# if imperfect_lp_selection_reward != {}:
#     results['inaccurate_comparison']['imperfect_lp'] = imperfect_lp_selection_reward
#     print(imperfect_lp_selection_reward)
# # -

# # ### Two-Stage Concept Predictors

# if not isinstance(ground_truth_env, ConceptEnv) and run_two_stage:
#     model = train_two_stage_ppo_model(ground_truth_env,total_timesteps=training_timesteps)
#     two_stage_eval = evaluate_model(environment_string,ground_truth_eval_env,additional_info,model,seed)
#     per_state_loss = model.per_state_loss
#     results['two_stage'] = {}

# if not isinstance(ground_truth_env, ConceptEnv) and run_two_stage:
#     ground_truth_concept_env = InfoTransformWrapper(ground_truth_env,[concept_list[i] for i in greedy_idx])
#     ground_truth_eval_concept_env = InfoTransformWrapper(ground_truth_eval_env,[concept_list[i] for i in greedy_idx])
#     model = train_two_stage_ppo_model(ground_truth_concept_env,total_timesteps=training_timesteps)
#     greedy_two_stage_reward = evaluate_model(environment_string,ground_truth_eval_concept_env,additional_info,model,seed)

#     results['two_stage']['greedy'] = {
#         'reward': greedy_two_stage_reward, 
#         'concepts': greedy_idx,
#     }

#     print(greedy_two_stage_reward)

# if not isinstance(ground_truth_env, ConceptEnv) and run_two_stage:
#     ground_truth_concept_env = InfoTransformWrapper(ground_truth_env,[concept_list[i] for i in greedy_iterative_idx])
#     ground_truth_eval_concept_env = InfoTransformWrapper(ground_truth_eval_env,[concept_list[i] for i in greedy_iterative_idx])
#     model = train_two_stage_ppo_model(ground_truth_concept_env,total_timesteps=training_timesteps)
#     greedy_iterative_two_stage_reward = evaluate_model(environment_string,ground_truth_eval_concept_env,additional_info,model,seed)

#     results['two_stage']['greedy_iterative'] = {
#         'reward': greedy_iterative_two_stage_reward, 
#         'concepts': greedy_iterative_idx,
#     }

#     print(greedy_iterative_two_stage_reward)


# if not isinstance(ground_truth_env, ConceptEnv) and run_two_stage:
#     ground_truth_concept_env = InfoTransformWrapper(ground_truth_env,[concept_list[i] for i in lp_idx])
#     ground_truth_eval_concept_env = InfoTransformWrapper(ground_truth_eval_env,[concept_list[i] for i in lp_idx])
#     model = train_two_stage_ppo_model(ground_truth_concept_env,total_timesteps=training_timesteps)
#     lp_two_stage_reward = evaluate_model(environment_string,ground_truth_eval_concept_env,additional_info,model,seed)

#     results['two_stage']['lp'] = {
#         'reward': lp_two_stage_reward, 
#         'concepts': lp_idx,
#     }

#     print(lp_two_stage_reward)


# if not isinstance(ground_truth_env, ConceptEnv) and per_state_loss is not None and run_two_stage:
#     top_k = [(idx,std) for idx,std in enumerate(per_state_loss)]
#     top_k_idx = sorted(top_k,key=lambda k: k[1])[:num_concepts_selected]
    
#     ground_truth_concept_env  = InfoTransformWrapper(ground_truth_env,[concept_list[i] for i in top_k_idx])
#     ground_truth_eval_concept_env = InfoTransformWrapper(ground_truth_eval_env,[concept_list[i] for i in top_k_idx])
#     model = train_two_stage_ppo_model(ground_truth_concept_env,total_timesteps=training_timesteps)
#     top_k_two_stage_reward = evaluate_model(environment_string,ground_truth_eval_concept_env,additional_info,model,seed)

#     results['two_stage']['top_k'] = {
#         'reward': top_k_two_stage_reward, 
#         'concepts': top_k_idx,
#     }
#     print(top_k_two_stage_reward)


# # ### Iterative

# if not isinstance(ground_truth_env, ConceptEnv) and run_iterative:
#     results['iterative'] = {}
#     iterative_concepts, iterative_idx = iterative_selection(ground_truth_env,concept_list,groundtruth_model,selection_function,target_abstraction,2,training_timesteps)
#     ground_truth_concept_env  = InfoTransformWrapper(ground_truth_env,iterative_concepts)
#     ground_truth_eval_concept_env = InfoTransformWrapper(ground_truth_eval_env,iterative_concepts)
#     model = train_two_stage_ppo_model(ground_truth_concept_env,total_timesteps=training_timesteps)
#     iterative_two_stage_reward = evaluate_model(environment_string,ground_truth_eval_concept_env,additional_info,model,seed)
#     results['iterative']['iterative'] = {
#         'reward': iterative_two_stage_reward, 
#         'concepts': iterative_idx
#     }
#     print(iterative_two_stage_reward)

# if not isinstance(ground_truth_env, ConceptEnv) and run_iterative:
#     bayesian_concepts, bayesian_idx = bayesian_iterative_selection(ground_truth_env,ground_truth_eval_env,environment_string,additional_info,seed,concept_list,2,training_timesteps)
#     ground_truth_concept_env  = InfoTransformWrapper(ground_truth_env,bayesian_concepts)
#     ground_truth_eval_concept_env = InfoTransformWrapper(ground_truth_eval_env,bayesian_concepts)
#     model = train_two_stage_ppo_model(ground_truth_concept_env,total_timesteps=training_timesteps)
#     bayesian_two_stage_reward = evaluate_model(environment_string,ground_truth_eval_concept_env,additional_info,model,seed)

#     results['iterative']['bayesian'] = {
#         'reward': bayesian_two_stage_reward, 
#         'concepts': bayesian_idx
#     }
#     print(bayesian_two_stage_reward)

# # ## Ablations

# # ### Reward Perturbation

# if reward_error > 0:
#     results['reward_error'] = {}
#     perturbed_groundtruth_eval_env = RewardPerturbationWrapper(ground_truth_eval_env,reward_error)
#     subset_concept, greedy_perturbed_idx = greedy_selection(perturbed_groundtruth_eval_env,concept_list,num_concepts_selected,groundtruth_model,selection_function)
#     env, eval_env, additional_info = get_environment(environment_string,subset_concept,seed)
#     perturbed_model = train_ppo_model(env,total_timesteps=training_timesteps,policy="MlpPolicy")
#     greedy_selection_perturbed_reward = evaluate_model(environment_string,eval_env,additional_info,perturbed_model,seed)
#     results['reward_error']['greedy'] = {
#         'reward': greedy_selection_perturbed_reward,
#         'concepts': greedy_perturbed_idx
#     }
#     print(greedy_selection_perturbed_reward)

# if reward_error > 0:
#     subset_concept, greedy_perturbed_iterative_idx = greedy_iterative_selection(perturbed_groundtruth_eval_env,concept_list,num_concepts_selected,groundtruth_model,selection_function)
#     env, eval_env, additional_info = get_environment(environment_string,subset_concept,seed)
#     model = train_ppo_model(env,total_timesteps=training_timesteps,policy="MlpPolicy")
#     greedy_iterative_selection_perturbed_reward = evaluate_model(environment_string,eval_env,additional_info,model,seed)
#     results['reward_error']['greedy_iterative'] = {
#         'reward': greedy_iterative_selection_perturbed_reward,
#         'concepts': greedy_perturbed_iterative_idx
#     }
#     print(greedy_iterative_selection_perturbed_reward)

# if reward_error > 0:
#     subset_concept, lp_perturbed_idx = lp_based_selection(perturbed_groundtruth_eval_env,concept_list,num_concepts_selected,groundtruth_model,selection_function,target_abstraction)
#     env, eval_env, additional_info = get_environment(environment_string,subset_concept,seed)
#     model = train_ppo_model(env,total_timesteps=training_timesteps,policy="MlpPolicy")
#     lp_selection_perturbed_reward = evaluate_model(environment_string,eval_env,additional_info,model,seed)
#     results['reward_error']['lp'] = {
#         'reward': lp_selection_perturbed_reward,
#         'concepts': lp_perturbed_idx
#     }
#     print(lp_selection_perturbed_reward)

# # ### Comparison with Concept Completeness

# # TODO: Create a Shapley-based baseline
# if assess_completeness:
#     pass 


# ## Save Data

save_path = get_save_path(out_folder,save_name)

delete_duplicate_results(out_folder,"",results)

json.dump(results,open('../../results/'+save_path,'w'))


