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
# 1) Get LLM Concepts + Implement that 
# 2) Get LLM Labels + Implemenet Two-Stage training with that
# 3) Create Concept Completeness Baseline
# 4) Run experiments with CUB + Two-Stage Training

# +
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
# -

is_jupyter = 'ipykernel' in sys.modules
is_main = __name__ == "__main__"

if is_main:
    if is_jupyter: 
        # Basics 
        seed        = 42
        environment_string = "cart_pole"
        training_timesteps = 250000
        num_concepts_selected = 30
        selection_function = "q_value"
        # Experiment #1 & #2
        run_basic = True
        run_iterative = False
        run_two_stage = False
        # Experiment #3
        cbm_accuracy_by_concept = [0.75 for i in range(94)]
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

if is_main:
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

if is_main:
    np.random.seed(seed)
    random.seed(seed)

# ### Basic Setup

if is_main:
    concept_list = get_concepts(environment_string,concept_source,seed)
    num_concepts_selected = min(num_concepts_selected,len(concept_list))
    ground_truth_env, ground_truth_gym_env, additional_info = get_environment(environment_string, None, seed)   

if is_main:
    model_name = "../../results/models/env={}_training={}_seed={}.zip".format(environment_string,training_timesteps,seed)
    
    if os.path.exists(model_name):
        groundtruth_model = PPO.load(model_name)
    else:
        if "cyclic" in environment_string or "tree" in environment_string or "mimic" in environment_string:
            policy = "MlpPolicy"
        else:
            policy = "CnnPolicy"
        groundtruth_model = train_ppo_model(ground_truth_env,environment_string,total_timesteps=training_timesteps,policy=policy)
        groundtruth_model.save(model_name)
    groundtruth_reward = evaluate_model(environment_string,ground_truth_gym_env,additional_info,groundtruth_model,seed)
    results['ground_truth'] = {'reward':groundtruth_reward}
    print(results['ground_truth']['reward'])

### Basic Comparison

if is_main:
    model_name = "../../results/q_estimates/env={}_training={}_seed={}_selection={}_source={}.pkl".format(environment_string,training_timesteps,seed,selection_function,concept_source)
    results['basic_comparison'] = {}
    if selection_function == "q_value":
        if environment_string == "mimic":
            modified_concepts = [lambda s, concept=concept: concept(additional_info['centers'][s]) 
                        for concept in concept_list]

            q_estimates = rollout_q_estimates_td(groundtruth_model,GymnasiumWrapper(DummyVecEnv([lambda: ground_truth_gym_env])),modified_concepts,learning_rate=1e-2,mimic=True,total_timesteps=5000,final_training=0)
        else:
            q_estimates = rollout_q_estimates_td(groundtruth_model,ground_truth_gym_env,concept_list)
    elif selection_function == "policy":
        if environment_string == "mimic":
            modified_concepts = [lambda s, concept=concept: concept(additional_info['centers'][s]) 
                        for concept in concept_list]

            q_estimates = rollout_pi_estimates(groundtruth_model,GymnasiumWrapper(DummyVecEnv([lambda: ground_truth_gym_env])),modified_concepts,mimic=True)
        else:
            q_estimates = rollout_pi_estimates(groundtruth_model,ground_truth_gym_env,concept_list)
    pickle.dump(q_estimates,open(model_name,"wb"))

if is_main and run_basic:
    # Train a random policy
    if environment_string == "mimic":
        model = RandomAgent(GymnasiumWrapper(DummyVecEnv([lambda: ground_truth_gym_env])))
    else:
        model = RandomAgent(ground_truth_gym_env)
    random_reward = evaluate_model(environment_string,ground_truth_gym_env,additional_info,model,seed)
    results['basic_comparison']['random'] = {'reward':random_reward}
    print(results['basic_comparison']['random']['reward'])

if is_main and run_basic:
    # Train a random selector
    subset_concept, random_idx = random_selection(concept_list,num_concepts_selected)
    env, eval_env, additional_info = get_environment(environment_string,subset_concept,seed)
    model = train_ppo_model(env,environment_string,total_timesteps=training_timesteps,policy="MlpPolicy")
    random_selection_reward = evaluate_model(environment_string,eval_env,additional_info,model,seed)
    results['basic_comparison']['random_selection'] = {'reward':random_selection_reward, 'concepts': random_idx}
    print(results['basic_comparison']['random_selection']['reward'])

if is_main and run_basic:
    # Train a greedy selector
    subset_concept, greedy_idx = greedy_selection(concept_list,num_concepts_selected,selection_function,q_estimates,concept_source)
    env, eval_env, additional_info = get_environment(environment_string,subset_concept,seed)
    model = train_ppo_model(env,environment_string,total_timesteps=training_timesteps,policy="MlpPolicy")
    greedy_selection_reward = evaluate_model(environment_string,eval_env,additional_info,model,seed)
    results['basic_comparison']['greedy'] = {'concepts': greedy_idx, 'reward':greedy_selection_reward}
    print(results['basic_comparison']['greedy']['reward'])


if is_main and run_basic:
    subset_concept, greedy_iterative_idx = greedy_iterative_selection(concept_list,num_concepts_selected,selection_function,q_estimates,concept_source)
    env, eval_env, additional_info = get_environment(environment_string,subset_concept,seed)
    model = train_ppo_model(env,environment_string,seed,total_timesteps=training_timesteps,policy="MlpPolicy")
    greedy_iterative_selection_reward = evaluate_model(environment_string,eval_env,additional_info,model,seed)
    results['basic_comparison']['greedy_iterative'] = {'concepts': greedy_iterative_idx, 'reward': greedy_iterative_selection_reward}
    print(results['basic_comparison']['greedy_iterative']['reward'])

if is_main and run_basic:
    subset_concept, lp_idx = lp_based_selection(ground_truth_gym_env,concept_list,num_concepts_selected,selection_function,q_estimates,concept_source)
    env, eval_env, additional_info = get_environment(environment_string,subset_concept,seed)
    model = train_ppo_model(env,environment_string,total_timesteps=training_timesteps,policy="MlpPolicy")
    lp_selection_reward = evaluate_model(environment_string,eval_env,additional_info,model,seed)
    results['basic_comparison']['lp'] = {'concepts': lp_idx, 'reward': lp_selection_reward}
    print(results['basic_comparison']['lp']['reward'])

# # ### Imperfect Concept Predictors

# if is_main and (cbm_std_by_concept is not None or cbm_accuracy_by_concept is not None):
#     results['inaccurate_comparison'] = {}


# if is_main:
#     greedy_inaccurate_reward = {}
#     for modification in ["continuous","binary"]:
#         if modification == "continuous" and cbm_std_by_concept is not None:
#             modified_concept_predictors = [inaccurate_concepts_continuous(func,0,std) for func,std in zip(concept_list,cbm_std_by_concept)]
#         elif modification == "binary" and cbm_accuracy_by_concept is not None:
#             modified_concept_predictors = [inaccurate_concepts_binary(func,acc,seed) for (func,acc) in zip(concept_list,cbm_accuracy_by_concept)]
#         else:
#             continue 
#         _, greedy_idx = greedy_selection(concept_list,num_concepts_selected,selection_function,q_estimates,concept_source)
#         subset_concept = [modified_concept_predictors[i] for i in greedy_idx]
#         env, eval_env, additional_info = get_environment(environment_string,subset_concept,seed)
#         model = train_ppo_model(env,environment_string,total_timesteps=training_timesteps,policy="MlpPolicy")
#         env, eval_env, additional_info = get_environment(environment_string,[concept_list[i] for i in greedy_idx],seed)

#         greedy_inaccurate_reward[modification] = {
#             'reward': evaluate_model(environment_string,eval_env,additional_info,model,seed),
#             'concepts': greedy_idx
#         }

#     if greedy_inaccurate_reward != {}:
#         results['inaccurate_comparison']['greedy'] = greedy_inaccurate_reward
#         print(greedy_inaccurate_reward)

# if is_main:
#     greedy_iterative_inaccurate_reward = {}
#     for modification in ["continuous","binary"]:
#         if modification == "continuous" and cbm_std_by_concept is not None:
#             modified_concept_predictors = [inaccurate_concepts_continuous(func,0,std) for func,std in zip(concept_list,cbm_std_by_concept)]
#         elif modification == "binary" and cbm_accuracy_by_concept is not None:
#             modified_concept_predictors = [inaccurate_concepts_binary(func,acc,seed) for func,acc in zip(concept_list,cbm_accuracy_by_concept)]
#         else:
#             continue 
        
#         _, greedy_iterative_idx = greedy_iterative_selection(concept_list,num_concepts_selected,selection_function,q_estimates,concept_source)
#         subset_concept = [modified_concept_predictors[i] for i in greedy_iterative_idx]
#         env, eval_env, additional_info = get_environment(environment_string,subset_concept,seed)
#         model = train_ppo_model(env,environment_string,total_timesteps=training_timesteps,policy="MlpPolicy")
#         greedy_iterative_inaccurate_reward[modification] = {
#             'reward': evaluate_model(environment_string,eval_env,additional_info,model,seed), 
#             'concepts': greedy_iterative_idx
#         } 

#     if greedy_iterative_inaccurate_reward != {}:
#         results['inaccurate_comparison']['greedy_iterative'] = greedy_iterative_inaccurate_reward
#         print(greedy_iterative_inaccurate_reward)

# if is_main:
#     lp_inaccurate_reward = {}
#     for modification in ["continuous","binary"]:
#         if modification == "continuous" and cbm_std_by_concept is not None:
#             modified_concept_predictors = [inaccurate_concepts_continuous(func,0,std) for func,std in zip(concept_list,cbm_std_by_concept)]
#         elif modification == "binary" and cbm_accuracy_by_concept is not None:
#             modified_concept_predictors = [inaccurate_concepts_binary(func,acc,seed) for func,acc in zip(concept_list,cbm_accuracy_by_concept)]
#         else:
#             continue 
#         _, lp_idx = lp_based_selection(ground_truth_gym_env,concept_list,num_concepts_selected,selection_function,q_estimates,concept_source)
#         subset_concept = [modified_concept_predictors[i] for i in lp_idx]
#         env, eval_env, additional_info = get_environment(environment_string,subset_concept,seed)
#         model = train_ppo_model(env,environment_string,total_timesteps=training_timesteps,policy="MlpPolicy")
#         lp_inaccurate_reward[modification] = { 'reward': evaluate_model(environment_string,eval_env,additional_info,model,seed),
#         'concepts': lp_idx}

#     if lp_inaccurate_reward != {}:
#         results['inaccurate_comparison']['lp'] = lp_inaccurate_reward
#         print(lp_inaccurate_reward)

# if is_main:
#     imperfect_lp_selection_reward = {}
#     for modification in ["continuous","binary"]:
#         if modification == "continuous" and cbm_std_by_concept is not None:
#             modified_concept_predictors = [inaccurate_concepts_continuous(func,0,std) for func,std in zip(concept_list,cbm_std_by_concept)]
#         elif modification == "binary" and cbm_accuracy_by_concept is not None:
#             modified_concept_predictors = [inaccurate_concepts_binary(func,acc,seed) for func,acc in zip(concept_list,cbm_accuracy_by_concept)]
#         else:
#             continue 
        
#         if modification == "continuous":
#             subset_concept, imperfect_idx = imperfect_lp_selection(ground_truth_gym_env,modified_concept_predictors,groundtruth_model,selection_function,target_abstraction,num_concepts_selected,cbm_accuracy_by_concept,concept_source,environment_string,additional_info,direction='min')
#         else:
#             subset_concept, imperfect_idx = imperfect_lp_selection(ground_truth_gym_env,modified_concept_predictors,groundtruth_model,selection_function,target_abstraction,num_concepts_selected,cbm_accuracy_by_concept,concept_source,environment_string,additional_info,direction='max')
#         env, eval_env, additional_info = get_environment(environment_string,subset_concept,seed)
#         model = train_ppo_model(env,environment_string,total_timesteps=training_timesteps,policy="MlpPolicy")
#         imperfect_lp_selection_reward[modification] = evaluate_model(environment_string,eval_env,additional_info,model,seed)

#     if imperfect_lp_selection_reward != {}:
#         results['inaccurate_comparison']['imperfect_lp'] = imperfect_lp_selection_reward
#         print(imperfect_lp_selection_reward)

# # ### Two-Stage Concept Predictors

# # +
# if is_main and not isinstance(ground_truth_env, ConceptEnv) and run_two_stage:
#     greedy_two_stage = {}
#     results['two_stage'] = {}
#     greedy_concepts, greedy_idx = greedy_selection(concept_list,num_concepts_selected,selection_function,q_estimates,concept_source)

#     X,Y = get_concept_labels(ground_truth_gym_env,groundtruth_model,greedy_concepts)
#     model = train_concept_predictor(X,Y)
#     fast_predictor = FastGPUPredictor(model, "cuda")
#     acc_list, f1_list = score_concept_predictors(model,X,Y)
#     two_stage_env, two_stage_gym_env, additional_info = get_environment(environment_string, greedy_concepts, seed,fast_predictor=fast_predictor,use_processed=True)   
#     model = train_ppo_model(two_stage_env,environment_string,policy="MlpPolicy",total_timesteps=training_timesteps)    
#     greedy_two_stage_reward = evaluate_model(environment_string,two_stage_gym_env,additional_info,model,seed)

#     results['two_stage']['greedy'] = {'reward': greedy_two_stage_reward, 'concepts': greedy_idx}
#     results['two_stage']['accuracy'] = acc_list 
#     print(greedy_two_stage_reward)



# # +
# if is_main and not isinstance(ground_truth_env, ConceptEnv) and run_two_stage:
#     greedy_iterative_concepts, greedy_iterative_idx = greedy_iterative_selection(concept_list,num_concepts_selected,selection_function,q_estimates,concept_source)

#     X,Y = get_concept_labels(ground_truth_gym_env,groundtruth_model,greedy_iterative_concepts)
#     model = train_concept_predictor(X,Y)
#     fast_predictor = FastGPUPredictor(model, "cuda")
#     two_stage_env, two_stage_gym_env, additional_info = get_environment(environment_string, greedy_iterative_concepts, seed,fast_predictor=fast_predictor,use_processed=True)   
#     model = train_ppo_model(two_stage_env,environment_string,policy="MlpPolicy",total_timesteps=training_timesteps)    
#     greedy_iterative_two_stage_reward = evaluate_model(environment_string,two_stage_gym_env,additional_info,model,seed)

#     results['two_stage']['greedy_iterative'] = {'reward': greedy_iterative_two_stage_reward, 'concepts': greedy_iterative_idx}
#     print(greedy_iterative_two_stage_reward)



# # +
# if is_main and not isinstance(ground_truth_env, ConceptEnv) and run_two_stage:
#     lp_concepts, lp_idx = lp_based_selection(ground_truth_gym_env,concept_list,num_concepts_selected,selection_function,q_estimates,concept_source)

#     X,Y = get_concept_labels(ground_truth_gym_env,groundtruth_model,lp_concepts)
#     model = train_concept_predictor(X,Y)
#     fast_predictor = FastGPUPredictor(model, "cuda")
#     two_stage_env, two_stage_gym_env, additional_info = get_environment(environment_string, lp_concepts, seed,fast_predictor=fast_predictor,use_processed=True)   
#     model = train_ppo_model(two_stage_env,environment_string,policy="MlpPolicy",total_timesteps=training_timesteps)    
#     lp_two_stage_reward = evaluate_model(environment_string,two_stage_gym_env,additional_info,model,seed)

#     results['two_stage']['lp'] = {'reward': lp_two_stage_reward, 'concepts': lp_idx}
#     print(lp_two_stage_reward)



# # +
# if is_main and not isinstance(ground_truth_env, ConceptEnv) and run_two_stage:
#     acc_list = results['two_stage']['accuracy']
#     top_k_idx = np.argsort(acc_list)[-num_concepts_selected:]
#     top_k_concepts = [concept_list[i] for i in top_k_idx]
    
#     X,Y = get_concept_labels(ground_truth_gym_env,groundtruth_model,top_k_concepts)
#     model = train_concept_predictor(X,Y)
#     fast_predictor = FastGPUPredictor(model, "cuda")
#     two_stage_env, two_stage_gym_env, additional_info = get_environment(environment_string, top_k_concepts, seed,fast_predictor=fast_predictor,use_processed=True)   
#     model = train_ppo_model(two_stage_env,environment_string,policy="MlpPolicy",total_timesteps=training_timesteps)    
#     top_k_two_stage_reward = evaluate_model(environment_string,two_stage_gym_env,additional_info,model,seed)

#     results['two_stage']['top_k'] = {'reward': top_k_two_stage_reward, 'concepts': top_k_idx}
#     print(top_k_two_stage_reward)


# # -

# # if is_main and not isinstance(ground_truth_env, ConceptEnv) and run_two_stage:
# #     acc_list = results['two_stage']['accuracy']
# #     imperfect_concepts, imperfect_idx = imperfect_lp_selection(ground_truth_gym_env,modified_concept_predictors,groundtruth_model,selection_function,target_abstraction,num_concepts_selected,acc_list,concept_source,environment_string,additional_info,direction='max')
    
# #     X,Y = get_concept_labels(ground_truth_gym_env,groundtruth_model,imperfect_concepts)
# #     model = train_concept_predictor(X,Y)
# #     fast_predictor = FastGPUPredictor(model, "cuda")
# #     two_stage_env, two_stage_gym_env, additional_info = get_environment(environment_string, imperfect_concepts, seed,fast_predictor=fast_predictor,use_processed=True)   
# #     model = train_ppo_model(two_stage_env,environment_string,policy="MlpPolicy",total_timesteps=training_timesteps)    
# #     top_k_two_stage_reward = evaluate_model(environment_string,two_stage_gym_env,additional_info,model,seed)

# #     results['two_stage']['imperfect'] = {'reward': imperfect_two_stage_reward, 'concepts': imperfect_idx}
# #     print(imperfect_two_stage_reward)

# # ### Iterative

# if is_main and not isinstance(ground_truth_env, ConceptEnv) and run_iterative:
#     results['iterative'] = {}
#     iterative_concepts, iterative_idx = iterative_selection(environment_string,ground_truth_gym_env,concept_list,groundtruth_model,selection_function,target_abstraction,num_concepts_selected,training_timesteps)
#     ground_truth_concept_env  = InfoTransformWrapper(ground_truth_gym_env,iterative_concepts)
#     ground_truth_eval_concept_env = InfoTransformWrapper(ground_truth_gym_env,iterative_concepts)
#     model = train_two_stage_ppo_model(environment_string,ground_truth_concept_env,iterative_concepts,total_timesteps=training_timesteps)
#     iterative_two_stage_reward = evaluate_model(environment_string,ground_truth_eval_concept_env,additional_info,model,seed)
#     results['iterative']['iterative'] = {
#         'reward': iterative_two_stage_reward, 
#         'concepts': iterative_idx
#     }
#     print(iterative_two_stage_reward)

# if is_main and not isinstance(ground_truth_env, ConceptEnv) and run_iterative:
#     bayesian_concepts, bayesian_idx = bayesian_iterative_selection(ground_truth_gym_env,ground_truth_gym_env,environment_string,additional_info,seed,concept_list,2,training_timesteps)
#     ground_truth_concept_env  = InfoTransformWrapper(ground_truth_gym_env,bayesian_concepts)
#     ground_truth_eval_concept_env = InfoTransformWrapper(ground_truth_gym_env,bayesian_concepts)
#     model = train_two_stage_ppo_model(environment_string,ground_truth_concept_env,bayesian_concepts,total_timesteps=training_timesteps)
#     bayesian_two_stage_reward = evaluate_model(environment_string,ground_truth_eval_concept_env,additional_info,model,seed)

#     results['iterative']['bayesian'] = {
#         'reward': bayesian_two_stage_reward, 
#         'concepts': bayesian_idx
#     }
#     print(bayesian_two_stage_reward)

# # ## Ablations

# # ### Reward Perturbation

# if is_main and reward_error > 0:
#     results['reward_error'] = {}
#     perturbed_groundtruth_eval_env = RewardPerturbationWrapper(ground_truth_gym_env,reward_error)

#     if selection_function == "q_value":
#         if environment_string == "mimic":
#             q_estimates_perturbed = rollout_q_estimates_td(groundtruth_model,GymnasiumWrapper(DummyVecEnv([lambda: ground_truth_gym_env])),concept_list,learning_rate=1e-3,mimic=True,total_timesteps=5000,final_training=0)
#         else:
#             q_estimates_perturbed = rollout_q_estimates_td(groundtruth_model,ground_truth_gym_env,concept_list)
#     elif selection_function == "policy":
#         if environment_string == "mimic":
#             q_estimates_perturbed = rollout_pi_estimates(groundtruth_model,GymnasiumWrapper(DummyVecEnv([lambda: ground_truth_gym_env])),concept_list,mimic=True)
#         else:
#             q_estimates_perturbed = rollout_pi_estimates(groundtruth_model,ground_truth_gym_env,concept_list)

#     subset_concept, greedy_perturbed_idx = greedy_selection(concept_list,num_concepts_selected,selection_function,q_estimates_perturbed,concept_source)
#     env, eval_env, additional_info = get_environment(environment_string,subset_concept,seed)
#     perturbed_model = train_ppo_model(env,environment_string,total_timesteps=training_timesteps,policy="MlpPolicy")
#     greedy_selection_perturbed_reward = evaluate_model(environment_string,eval_env,additional_info,perturbed_model,seed)
#     results['reward_error']['greedy'] = {
#         'reward': greedy_selection_perturbed_reward,
#         'concepts': greedy_perturbed_idx
#     }
#     print(greedy_selection_perturbed_reward)

# if is_main and reward_error > 0:
#     subset_concept, greedy_perturbed_iterative_idx = greedy_iterative_selection(concept_list,num_concepts_selected,selection_function,q_estimates_perturbed,concept_source)
#     env, eval_env, additional_info = get_environment(environment_string,subset_concept,seed)
#     model = train_ppo_model(env,environment_string,total_timesteps=training_timesteps,policy="MlpPolicy")
#     greedy_iterative_selection_perturbed_reward = evaluate_model(environment_string,eval_env,additional_info,model,seed)
#     results['reward_error']['greedy_iterative'] = {
#         'reward': greedy_iterative_selection_perturbed_reward,
#         'concepts': greedy_perturbed_iterative_idx
#     }
#     print(greedy_iterative_selection_perturbed_reward)

# if is_main and reward_error > 0:
#     subset_concept, lp_perturbed_idx = lp_based_selection(concept_list,num_concepts_selected,selection_function,q_estimates_perturbed,concept_source)
#     env, eval_env, additional_info = get_environment(environment_string,subset_concept,seed)
#     model = train_ppo_model(env,environment_string,total_timesteps=training_timesteps,policy="MlpPolicy")
#     lp_selection_perturbed_reward = evaluate_model(environment_string,eval_env,additional_info,model,seed)
#     results['reward_error']['lp'] = {
#         'reward': lp_selection_perturbed_reward,
#         'concepts': lp_perturbed_idx
#     }
#     print(lp_selection_perturbed_reward)

# # ### Comparison with Concept Completeness

# # TODO: Create a Shapley-based baseline
# if is_main and assess_completeness:
#     pass 


# ## Save Data

if is_main:
    save_path = get_save_path(out_folder,save_name)

if is_main:
    delete_duplicate_results(out_folder,"",results)

if is_main:
    json.dump(results,open('../../results/'+save_path,'w'))
