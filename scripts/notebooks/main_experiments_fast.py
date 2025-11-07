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

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["GRB_LICENSE_FILE"] = "/usr0/home/naveenr/gurobi.lic"
os.environ['MKL_THREADING_LAYER'] = "GNU"
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

torch.cuda.set_per_process_memory_fraction(0.5)
torch.set_num_threads(1)
resource.setrlimit(resource.RLIMIT_AS, (30 * 1024 * 1024 * 1024, -1))


is_jupyter = 'ipykernel' in sys.modules
is_main = __name__ == "__main__"

if is_main:
    if is_jupyter: 
        # Basics 
        seed        = 42
        environment_string = "pong"
        gold_timesteps = 4_000_000
        training_timesteps = 4_000_000
        num_concepts_selected = 80
        selection_function = "q_value"
        # Experiment #1 & #2
        run_basic = True
        run_iterative = False 
        run_two_stage = False  
        run_imperfect=False
        run_intervention=False
        run_per_epoch=False
        # Experiment #3
        cbm_accuracy_by_concept = None 
        intervention_probability = 0
        intervention_accuracy_by_concept = None 
        cbm_std_by_concept = None 
        target_abstraction = 0.05
        reward_error = 0
        # Experiment #4
        concept_source = "human_selected_binary"
        # Experiment #5
        assess_completeness=False
        # Experiment #6
        num_iterations = 0
        selections_per_round = 0
        initial_concepts = 0
        out_folder = "llm"
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument('--seed', help='Random Seed', type=int, default=42)
        parser.add_argument('--environment_string', help='Which environment to create', type=str, default="tree")
        parser.add_argument('--training_timesteps', help='Number of training timesteps', type=int, default=10000)
        parser.add_argument('--gold_timesteps', help='Number of training timesteps without concepts', type=int, default=10000)
        parser.add_argument('--num_concepts_selected', help='Number of concepts selected by greedy or random',type=int, default=0)
        parser.add_argument('--selection_function', help='When selecting, use q_value, policy, or transition?', type=str, default="policy")
        parser.add_argument('--cbm_accuracy_by_concept', help="What is the accuracy of AI per concept?", nargs='*', type=float, default=None)
        parser.add_argument('--intervention_accuracy_by_concept', help="What is the accuracy of AI per concept?", nargs='*', type=float, default=None)
        parser.add_argument('--cbm_std_by_concept', help="What is the error of AI per concept?", nargs='*', type=float, default=None)
        parser.add_argument('--run_two_stage', help='Run the two stage?', action='store_true')
        parser.add_argument('--run_iterative', help='Run the iterative?', action='store_true')
        parser.add_argument('--run_intervention', help='Run the intervention?', action='store_true')
        parser.add_argument('--run_basic', help='Run the basic comparisons?', action='store_true')
        parser.add_argument('--run_imperfect', help='Run the imperfect comparisons?', action='store_true')
        parser.add_argument('--run_per_epoch', help='Run the imperfect comparison per epoch?', action='store_true')
        parser.add_argument('--intervention_probability', help='Value for the target abstraction with human performance', type=float, default=0.05)
        parser.add_argument('--target_abstraction', help='Value for the target abstraction with human performance', type=float, default=0.05)
        parser.add_argument('--reward_error', help="How much to perturb the reward by?", type=float, default=0)
        parser.add_argument('--concept_source', help='When selecting, use q_value, policy, or transition?', type=str, default="human_selected")
        parser.add_argument('--assess_completeness', help='Compare to the concept completeness algorithm?', action='store_true')
        parser.add_argument('--num_iterations', help='Number of iterations for iterative algorithms',type=int, default=0)
        parser.add_argument('--selections_per_round', help='Concepts to select per round',type=int, default=0)
        parser.add_argument('--initial_concepts', help='Number of starting/initial concepts',type=int, default=0)
        parser.add_argument('--out_folder', help='Which folder', type=str, default="exploration")

        args = parser.parse_args()

        seed = args.seed
        environment_string = args.environment_string
        training_timesteps = args.training_timesteps 
        gold_timesteps = args.gold_timesteps
        num_concepts_selected = args.num_concepts_selected
        selection_function = args.selection_function
        cbm_accuracy_by_concept = args.cbm_accuracy_by_concept
        cbm_std_by_concept = args.cbm_std_by_concept
        run_basic = args.run_basic
        run_iterative = args.run_iterative
        run_two_stage = args.run_two_stage
        run_imperfect = args.run_imperfect
        run_per_epoch = args.run_per_epoch
        run_intervention = args.run_intervention
        intervention_probability = args.intervention_probability
        intervention_accuracy_by_concept = args.intervention_accuracy_by_concept
        target_abstraction = args.target_abstraction
        reward_error = args.reward_error
        concept_source = args.concept_source
        assess_completeness = args.assess_completeness
        num_iterations = args.num_iterations 
        selections_per_round = args.selections_per_round
        initial_concepts = args.initial_concepts
        out_folder = args.out_folder

    save_name = secrets.token_hex(4)  

if is_main:
        results = {}
        results['parameters'] = {'seed'      : seed,
                'environment_string'    : environment_string, 
                'training_timesteps': training_timesteps, 
                'gold_timesteps': gold_timesteps,
                'selection_function': selection_function,
                'num_concepts_selected': num_concepts_selected,
                'cbm_accuracy_by_concept': cbm_accuracy_by_concept,
                'cbm_std_by_concept': cbm_std_by_concept,
                'intervention_probability': intervention_probability,
                'intervention_accuracy_by_concept': intervention_accuracy_by_concept,
                'target_abstraction': target_abstraction,
                'reward_error': reward_error, 
                'concept_source': concept_source,
                'assess_completeness': assess_completeness,
                'num_iterations': num_iterations,
                'selections_per_round': selections_per_round, 
                'initial_concepts': initial_concepts,
                'run_basic': run_basic,
                'run_iterative': run_iterative, 
                'run_two_stage': run_two_stage, 
                'run_intervention': run_intervention,
                'run_imperfect': run_imperfect, 
                'run_per_epoch': run_per_epoch,
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
    model_name = "../../results/models/env={}_training={}_seed={}.zip".format(environment_string,gold_timesteps,seed)
    
    if os.path.exists(model_name):
        print("Model exists!")
        groundtruth_model = PPO.load(model_name)
        additional_info['subset_concepts'] = get_concepts(environment_string,"human_selected",seed)
    else:
        if "cyclic" in environment_string or "tree" in environment_string or "glucose" in environment_string:
            policy = "MlpPolicy"
        else:
            policy = "CnnPolicy"
        
        if environment_string == "mimic":
            additional_info['subset_concepts'] = get_concepts(environment_string,"human_selected",seed)
            groundtruth_model = train_ppo_model(ground_truth_env,"mimic_raw",total_timesteps=gold_timesteps,policy=policy,)
        else:
            groundtruth_model = train_ppo_model(ground_truth_env,environment_string,total_timesteps=gold_timesteps,policy=policy)
        groundtruth_model.save(model_name)
    groundtruth_reward = evaluate_model(environment_string,ground_truth_gym_env,additional_info,groundtruth_model,seed)
    results['ground_truth'] = {'reward':groundtruth_reward}
    print("Basic:",results['ground_truth']['reward'])

# ### Basic Comparison

if is_main:    
    model_name = "../../results/q_estimates/env={}_training={}_seed={}_selection={}_source={}.pkl".format(environment_string,gold_timesteps,seed,selection_function,concept_source)
    results['basic_comparison'] = {}
    if os.path.exists(model_name):
        q_estimates = pickle.load(open(model_name,"rb"))
    else:
        if selection_function == "q_value":
            if environment_string == "glucose":
                q_estimates = rollout_q_estimates_td(groundtruth_model,ground_truth_gym_env,concept_list,total_timesteps=20_000)
            else:
                q_estimates = rollout_q_estimates_td(groundtruth_model,ground_truth_gym_env,concept_list)
        elif selection_function == "policy":
            q_estimates = rollout_pi_estimates(groundtruth_model,ground_truth_gym_env,concept_list)
        pickle.dump(q_estimates,open(model_name,"wb"))

if is_main and run_basic:
    subset_concept, multiple_idx = multiple_lp_selection(ground_truth_gym_env,concept_list,num_concepts_selected,selection_function,q_estimates,concept_source)
    env, eval_env, additional_info = get_environment(environment_string,subset_concept,seed)
    model = train_ppo_model(env,environment_string,total_timesteps=training_timesteps,policy="MlpPolicy",custom_name="{}_multiple".format(environment_string))
    multiple_selection_reward = evaluate_model(environment_string,eval_env,additional_info,model,seed)
    results['basic_comparison']['multiple'] = {'concepts': multiple_idx, 'reward': multiple_selection_reward}
    print("Multiple Selection:",results['basic_comparison']['multiple']['reward'])

# ### Imperfect Concept Predictors

if is_main and run_imperfect:
    results['inaccurate_comparison'] = {}

if is_main and run_imperfect:
    multiple_lp_selection_reward = {}
    modified_concept_predictors = [inaccurate_concepts_binary(func,acc,seed) for func,acc in zip(concept_list,cbm_accuracy_by_concept)]
    q_estimates_error = rollout_q_estimates_td(groundtruth_model,ground_truth_gym_env,modified_concept_predictors)
    subset_concept, multiple_idx = multiple_lp_selection(ground_truth_gym_env,modified_concept_predictors,num_concepts_selected,selection_function,q_estimates_error,concept_source)
    env, eval_env, additional_info = get_environment(environment_string,subset_concept,seed)
    model = train_ppo_model(env,environment_string,total_timesteps=training_timesteps,policy="MlpPolicy",custom_name="{}_error_multiple".format(environment_string))
    multiple_lp_selection_reward = {}
    multiple_lp_selection_reward['reward'] = evaluate_model(environment_string,eval_env,additional_info,model,seed)
    multiple_lp_selection_reward['concepts'] = multiple_idx

    if multiple_lp_selection_reward != {}:
        results['inaccurate_comparison']['multiple_lp'] = multiple_lp_selection_reward
        print(multiple_lp_selection_reward)

if is_main and run_per_epoch:
    results['inaccurate_comparison'] = {}
    multiple_lp_selection_reward = {}
    modified_concept_predictors = [inaccurate_concepts_binary(func,acc,seed) for func,acc in zip(concept_list,cbm_accuracy_by_concept)]
    subset_concept, multiple_idx = multiple_lp_selection(ground_truth_gym_env,modified_concept_predictors,num_concepts_selected,selection_function,q_estimates,concept_source)
    env, eval_env, additional_info = get_environment(environment_string,subset_concept,seed)
    
    multiple_lp_selection_reward = {}
    multiple_lp_selection_reward['concepts'] = multiple_idx
    multiple_lp_selection_reward['reward'] = []
    model = None 
    for i in range(10):
        model = train_ppo_model(env,environment_string,total_timesteps=training_timesteps//10,policy="MlpPolicy",custom_name="{}_error_multiple".format(environment_string),model=model)
        multiple_lp_selection_reward['reward'].append(evaluate_model(environment_string,eval_env,additional_info,model,seed))
        print(multiple_lp_selection_reward['reward'])
    results['inaccurate_comparison']['multiple_lp'] = multiple_lp_selection_reward
    print(multiple_lp_selection_reward)

# ### Intervention

if is_main and run_intervention and intervention_accuracy_by_concept is not None:
    results['intervention_comparison'] = {}

if is_main and run_intervention:
    multiple_lp_intervention_reward = {}
    modified_concept_predictors = [inaccurate_concepts_binary_intervention(func,acc,intervene_acc,0,seed+idx) for (func,acc,intervene_acc,idx) in zip(concept_list,cbm_accuracy_by_concept,intervention_accuracy_by_concept,list(range(len(concept_list))))]
    subset_concept, multiple_idx = multiple_lp_selection(ground_truth_gym_env,modified_concept_predictors,num_concepts_selected,selection_function,q_estimates,concept_source)
    env, eval_env, additional_info = get_environment(environment_string,subset_concept,seed)

    model = train_ppo_model(env,environment_string,total_timesteps=training_timesteps,policy="MlpPolicy",custom_name="{}_intervention_multiple".format(environment_string))
    multiple_lp_intervention_reward = {}
    modified_concept_predictors = [inaccurate_concepts_binary_intervention(func,acc,intervene_acc,intervention_probability,seed+idx) for (func,acc,intervene_acc,idx) in zip(concept_list,cbm_accuracy_by_concept,intervention_accuracy_by_concept,list(range(len(concept_list))))]
    subset_concept = [modified_concept_predictors[i] for i in multiple_idx]
    env, eval_env, additional_info = get_environment(environment_string,subset_concept,seed)
    multiple_lp_intervention_reward['reward'] = evaluate_model(environment_string,eval_env,additional_info,model,seed)
    multiple_lp_intervention_reward['concepts'] = multiple_idx

    if multiple_lp_intervention_reward != {}:
        results['intervention_comparison']['multiple_lp'] = multiple_lp_intervention_reward
        print(multiple_lp_intervention_reward)

if is_main and run_intervention and environment_string == 'glucose':
    multiple_lp_intervention_reward = {}
    modified_concept_predictors = [inaccurate_concepts_binary_intervention(func,acc,intervene_acc,intervention_probability,seed+idx) for (func,acc,intervene_acc,idx) in zip(concept_list,cbm_accuracy_by_concept,intervention_accuracy_by_concept,list(range(len(concept_list))))]
    subset_concept, multiple_idx = multiple_lp_selection(ground_truth_gym_env,modified_concept_predictors,num_concepts_selected,selection_function,q_estimates,concept_source)
    env, eval_env, additional_info = get_environment(environment_string,subset_concept,seed)

    model = train_ppo_model(env,environment_string,total_timesteps=training_timesteps,policy="MlpPolicy",custom_name="{}_intervention_multiple".format(environment_string))
    multiple_lp_intervention_reward = {}
    modified_concept_predictors = [inaccurate_concepts_binary_intervention(func,acc,intervene_acc,intervention_probability,seed+idx) for (func,acc,intervene_acc,idx) in zip(concept_list,cbm_accuracy_by_concept,intervention_accuracy_by_concept,list(range(len(concept_list))))]
    subset_concept = [modified_concept_predictors[i] for i in multiple_idx]
    env, eval_env, additional_info = get_environment(environment_string,subset_concept,seed)
    multiple_lp_intervention_reward['reward'] = evaluate_model(environment_string,eval_env,additional_info,model,seed)
    multiple_lp_intervention_reward['concepts'] = multiple_idx

    if multiple_lp_intervention_reward != {}:
        results['intervention_comparison']['multiple_lp_train'] = multiple_lp_intervention_reward
        print(multiple_lp_intervention_reward)


# ## Save Data

if is_main:
    save_path = get_save_path(out_folder,save_name)

if is_main:
    delete_duplicate_results(out_folder,"",results)

if is_main:
    json.dump(results,open('../../results/'+save_path,'w'))

if is_main:
    ground_truth_env.close()
    ground_truth_gym_env.close()
    env.close()
    eval_env.close()
