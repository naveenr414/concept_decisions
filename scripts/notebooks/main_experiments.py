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

# +
from concept_abstraction.training import *
from concept_abstraction.selection import *
from concept_abstraction.concept_bank import *
from concept_abstraction.env_utils import *
from concept_abstraction.environments import *
from concept_abstraction.utils import *
from concept_abstraction.environments import ConceptEnv
from concept_abstraction.mimic import get_mimic_eval_info
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.cluster import KMeans

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

torch.cuda.set_per_process_memory_fraction(0.5)
torch.set_num_threads(1)
resource.setrlimit(resource.RLIMIT_AS, (30 * 1024 * 1024 * 1024, -1))


is_jupyter = 'ipykernel' in sys.modules
is_main = __name__ == "__main__"

if is_main:
    if is_jupyter: 
        # Basics 
        seed        = 42
        environment_string = "mimic"
        gold_timesteps = 250_000
        training_timesteps = 250_000
        num_concepts_selected = 40
        selection_function = "q_value"
        # Experiment #1 & #2
        run_basic = True
        run_iterative = False 
        run_two_stage = False 
        run_imperfect=False
        run_intervention=False
        # Experiment #3
        cbm_accuracy_by_concept = [0.9 for i in range(24)] 
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
        num_iterations = 2
        selections_per_round = 1
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
        if "cyclic" in environment_string or "tree" in environment_string or "mimic" in environment_string:
            policy = "MlpPolicy"
        else:
            policy = "CnnPolicy"
        
        if environment_string == "mimic":
            additional_info['subset_concepts'] = get_concepts(environment_string,"human_selected",seed)
            groundtruth_model = train_ppo_model(ground_truth_env,"mimic_raw",total_timesteps=gold_timesteps,policy=policy,additional_info=additional_info)
        else:
            groundtruth_model = train_ppo_model(ground_truth_env,environment_string,total_timesteps=gold_timesteps,policy=policy)
        groundtruth_model.save(model_name)
    groundtruth_reward = evaluate_model(environment_string,ground_truth_gym_env,additional_info,groundtruth_model,seed)
    results['ground_truth'] = {'reward':groundtruth_reward}
    print("Basic:",results['ground_truth']['reward'])

# ### Basic Comparison

if is_main:    
    model_name = "../../results/q_estimates/env={}_training={}_seed={}_selection={}_source={}.pkl".format(environment_string,training_timesteps,seed,selection_function,concept_source)
    results['basic_comparison'] = {}
    if os.path.exists(model_name):
        q_estimates = pickle.load(open(model_name,"rb"))
    else:
        if selection_function == "q_value":
            if environment_string == "mimic":
                modified_concepts = [lambda s, concept=concept: concept(additional_info['centers'][s]) 
                            for concept in concept_list]

                q_estimates = rollout_q_estimates_td(groundtruth_model,GymnasiumWrapper(DummyVecEnv([lambda: ground_truth_gym_env])),modified_concepts,learning_rate=1e-2,mimic=True,total_timesteps=10000,final_training=0)
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
    print("Random:",results['basic_comparison']['random']['reward'])

if is_main and run_basic:
    # Train a random selector
    subset_concept, random_idx = random_selection(concept_list,num_concepts_selected)
    subset_concept = [concept_list[i] for i in random_idx]
    env, eval_env, additional_info = get_environment(environment_string,subset_concept,seed)    
    model = train_ppo_model(env,environment_string,total_timesteps=training_timesteps,policy="MlpPolicy",additional_info=additional_info)
    random_selection_reward = evaluate_model(environment_string,eval_env,additional_info,model,seed)
    results['basic_comparison']['random_selection'] = {'reward':random_selection_reward, 'concepts': random_idx}
    print("Random Selection:",results['basic_comparison']['random_selection']['reward'])

if is_main and run_basic:
    # Train a greedy selector
    subset_concept, greedy_idx = greedy_selection(concept_list,num_concepts_selected,selection_function,q_estimates,concept_source)
    env, eval_env, additional_info = get_environment(environment_string,subset_concept,seed)
    
    model = train_ppo_model(env,environment_string,total_timesteps=training_timesteps,policy="MlpPolicy",additional_info=additional_info)
    greedy_selection_reward = evaluate_model(environment_string,eval_env,additional_info,model,seed)
    results['basic_comparison']['greedy'] = {'concepts': greedy_idx, 'reward':greedy_selection_reward}
    print("Greedy:",results['basic_comparison']['greedy']['reward'])


if is_main and run_basic and environment_string == "mimic":
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.cluster import KMeans

    # Example: use unsupervised cluster labels as pseudo-target
    labels = KMeans(n_clusters=5).fit_predict(additional_info['centers'])
    tree = DecisionTreeClassifier(max_leaf_nodes=21)
    tree.fit(additional_info['centers'], labels)

    # Extract top 20 splits
    features = tree.tree_.feature
    thresholds = tree.tree_.threshold

    functions = []
    for idx in np.where(features >= 0)[0][:num_concepts_selected]:
        j, t = features[idx], thresholds[idx]
        functions.append(lambda x, j=j, t=t: int(x[j] > t))
    subset_concept = functions
    env, eval_env, additional_info = get_environment(environment_string,subset_concept,seed)
    
    model = train_ppo_model(env,environment_string,total_timesteps=training_timesteps,policy="MlpPolicy",additional_info=additional_info)
    decision_tree_selection_reward = evaluate_model(environment_string,eval_env,additional_info,model,seed)
    results['basic_comparison']['decision_tree'] = {'concepts': [], 'reward':decision_tree_selection_reward}
    print("Decision Tree:",results['basic_comparison']['decision_tree']['reward'])


if is_main and run_basic:
    subset_concept, lp_idx = lp_based_selection(ground_truth_gym_env,concept_list,num_concepts_selected,selection_function,q_estimates,concept_source)
    env, eval_env, additional_info = get_environment(environment_string,subset_concept,seed)
    model = train_ppo_model(env,environment_string,total_timesteps=training_timesteps,policy="MlpPolicy",additional_info=additional_info)
    lp_selection_reward = evaluate_model(environment_string,eval_env,additional_info,model,seed)
    results['basic_comparison']['lp'] = {'concepts': lp_idx, 'reward': lp_selection_reward}
    print("LP Selection:",results['basic_comparison']['lp']['reward'])

if is_main and run_basic:
    subset_concept, multiple_idx = multiple_lp_selection(ground_truth_gym_env,concept_list,num_concepts_selected,selection_function,q_estimates,concept_source)
    env, eval_env, additional_info = get_environment(environment_string,subset_concept,seed)
    model = train_ppo_model(env,environment_string,total_timesteps=training_timesteps,policy="MlpPolicy",additional_info=additional_info)
    multiple_selection_reward = evaluate_model(environment_string,eval_env,additional_info,model,seed)
    results['basic_comparison']['multiple'] = {'concepts': multiple_idx, 'reward': multiple_selection_reward}
    print("Multiple Selection:",results['basic_comparison']['multiple']['reward'])

# ### Imperfect Concept Predictors

if is_main and run_imperfect:
    results['inaccurate_comparison'] = {}


if is_main and run_imperfect:
    greedy_inaccurate_reward = {}
    for modification in ["continuous","binary"]:
        if modification == "continuous" and cbm_std_by_concept is not None:
            modified_concept_predictors = [inaccurate_concepts_continuous(func,0,std) for func,std in zip(concept_list,cbm_std_by_concept)]
        elif modification == "binary" and cbm_accuracy_by_concept is not None:
            modified_concept_predictors = [inaccurate_concepts_binary(func,acc,seed) for (func,acc) in zip(concept_list,cbm_accuracy_by_concept)]
        else:
            continue 
        subset_concept, greedy_idx = greedy_selection(modified_concept_predictors,num_concepts_selected,selection_function,q_estimates,concept_source)
        env, eval_env, additional_info = get_environment(environment_string,subset_concept,seed)
        model = train_ppo_model(env,environment_string,total_timesteps=training_timesteps,policy="MlpPolicy",additional_info=additional_info)

        greedy_inaccurate_reward[modification] = {
            'reward': evaluate_model(environment_string,eval_env,additional_info,model,seed),
            'concepts': greedy_idx
        }

    if greedy_inaccurate_reward != {}:
        results['inaccurate_comparison']['greedy'] = greedy_inaccurate_reward
        print(greedy_inaccurate_reward)

if is_main and run_imperfect:
    lp_inaccurate_reward = {}
    for modification in ["continuous","binary"]:
        if modification == "continuous" and cbm_std_by_concept is not None:
            modified_concept_predictors = [inaccurate_concepts_continuous(func,0,std) for func,std in zip(concept_list,cbm_std_by_concept)]
        elif modification == "binary" and cbm_accuracy_by_concept is not None:
            modified_concept_predictors = [inaccurate_concepts_binary(func,acc,seed) for func,acc in zip(concept_list,cbm_accuracy_by_concept)]
        else:
            continue 
        _, lp_idx = lp_based_selection(ground_truth_gym_env,modified_concept_predictors,num_concepts_selected,selection_function,q_estimates,concept_source)
        subset_concept = [modified_concept_predictors[i] for i in lp_idx]
        env, eval_env, additional_info = get_environment(environment_string,subset_concept,seed)
        model = train_ppo_model(env,environment_string,total_timesteps=training_timesteps,policy="MlpPolicy",additional_info=additional_info)
        lp_inaccurate_reward[modification] = { 'reward': evaluate_model(environment_string,eval_env,additional_info,model,seed),
        'concepts': lp_idx}

    if lp_inaccurate_reward != {}:
        results['inaccurate_comparison']['lp'] = lp_inaccurate_reward
        print(lp_inaccurate_reward)

if is_main and run_imperfect:
    multiple_lp_selection_reward = {}
    modified_concept_predictors = [inaccurate_concepts_binary(func,acc,seed) for func,acc in zip(concept_list,cbm_accuracy_by_concept)]
    subset_concept, multiple_idx = multiple_lp_selection(ground_truth_gym_env,modified_concept_predictors,num_concepts_selected,selection_function,q_estimates,concept_source)
    env, eval_env, additional_info = get_environment(environment_string,subset_concept,seed)
    model = train_ppo_model(env,environment_string,total_timesteps=training_timesteps,policy="MlpPolicy",additional_info=additional_info)
    multiple_lp_selection_reward = {}
    multiple_lp_selection_reward['reward'] = evaluate_model(environment_string,eval_env,additional_info,model,seed)
    multiple_lp_selection_reward['concepts'] = multiple_idx

    if multiple_lp_selection_reward != {}:
        results['inaccurate_comparison']['multiple_lp'] = multiple_lp_selection_reward
        print(multiple_lp_selection_reward)

if is_main and run_imperfect:
    imperfect_lp_selection_reward = {}
    for modification in ["continuous","binary"]:
        if modification == "continuous" and cbm_std_by_concept is not None:
            modified_concept_predictors = [inaccurate_concepts_continuous(func,0,std) for func,std in zip(concept_list,cbm_std_by_concept)]
        elif modification == "binary" and cbm_accuracy_by_concept is not None:
            modified_concept_predictors = [inaccurate_concepts_binary(func,acc,seed) for func,acc in zip(concept_list,cbm_accuracy_by_concept)]
        else:
            continue 
        
        if modification == "continuous":
            subset_concept, imperfect_idx = imperfect_lp_selection(ground_truth_gym_env,modified_concept_predictors,q_estimates,selection_function,target_abstraction,num_concepts_selected,cbm_accuracy_by_concept,concept_source,environment_string,additional_info,direction='min')
        else:
            subset_concept, imperfect_idx = imperfect_lp_selection(ground_truth_gym_env,modified_concept_predictors,q_estimates,selection_function,target_abstraction,num_concepts_selected,cbm_accuracy_by_concept,concept_source,environment_string,additional_info,direction='max')
        env, eval_env, additional_info = get_environment(environment_string,subset_concept,seed)
        model = train_ppo_model(env,environment_string,total_timesteps=training_timesteps,policy="MlpPolicy",additional_info=additional_info)
        imperfect_lp_selection_reward[modification] = {}
        imperfect_lp_selection_reward[modification]['reward'] = evaluate_model(environment_string,eval_env,additional_info,model,seed)
        imperfect_lp_selection_reward[modification]['concepts'] = imperfect_idx

    if imperfect_lp_selection_reward != {}:
        results['inaccurate_comparison']['imperfect_lp'] = imperfect_lp_selection_reward
        print(imperfect_lp_selection_reward)

# ### Intervention

if is_main and run_intervention and intervention_accuracy_by_concept is not None:
    results['intervention_comparison'] = {}

if is_main and run_intervention:
    greedy_intervention_reward = {}
    for modification in ["binary"]:
        if modification == "binary" and cbm_accuracy_by_concept is not None and intervention_accuracy_by_concept is not None:
            modified_concept_predictors = [inaccurate_concepts_binary_intervention(func,acc,intervene_acc,0,seed+idx) for (func,acc,intervene_acc,idx) in zip(concept_list,cbm_accuracy_by_concept,intervention_accuracy_by_concept,list(range(len(concept_list))))]
        else:
            continue 
        _, greedy_idx = greedy_selection(concept_list,num_concepts_selected,selection_function,q_estimates,concept_source)
        subset_concept = [modified_concept_predictors[i] for i in greedy_idx]
        env, eval_env, additional_info = get_environment(environment_string,subset_concept,seed)
        model = train_ppo_model(env,environment_string,total_timesteps=training_timesteps,policy="MlpPolicy",additional_info=additional_info)

        modified_concept_predictors = [inaccurate_concepts_binary_intervention(func,acc,intervene_acc,intervention_probability,seed+idx) for (func,acc,intervene_acc,idx) in zip(concept_list,cbm_accuracy_by_concept,intervention_accuracy_by_concept,list(range(len(concept_list))))]
        subset_concept = [modified_concept_predictors[i] for i in greedy_idx]
        env, eval_env, additional_info = get_environment(environment_string,subset_concept,seed)

        greedy_intervention_reward[modification] = {
            'reward': evaluate_model(environment_string,eval_env,additional_info,model,seed),
            'concepts': greedy_idx
        }

    if greedy_intervention_reward != {}:
        results['intervention_comparison']['greedy'] = greedy_intervention_reward
        print(greedy_intervention_reward)

if is_main and run_intervention:
    lp_intervention_reward = {}
    for modification in ["binary"]:
        if modification == "binary" and cbm_accuracy_by_concept is not None and intervention_accuracy_by_concept is not None:
            modified_concept_predictors = [inaccurate_concepts_binary_intervention(func,acc,intervene_acc,0,seed+idx) for (func,acc,intervene_acc,idx) in zip(concept_list,cbm_accuracy_by_concept,intervention_accuracy_by_concept,list(range(len(concept_list))))]
        else:
            continue 
        _, lp_idx = lp_based_selection(ground_truth_gym_env,concept_list,num_concepts_selected,selection_function,q_estimates,concept_source)
        subset_concept = [modified_concept_predictors[i] for i in lp_idx]
        env, eval_env, additional_info = get_environment(environment_string,subset_concept,seed)
        model = train_ppo_model(env,environment_string,total_timesteps=training_timesteps,policy="MlpPolicy",additional_info=additional_info)

        modified_concept_predictors = [inaccurate_concepts_binary_intervention(func,acc,intervene_acc,intervention_probability,seed+idx) for (func,acc,intervene_acc,idx) in zip(concept_list,cbm_accuracy_by_concept,intervention_accuracy_by_concept,list(range(len(concept_list))))]
        subset_concept = [modified_concept_predictors[i] for i in lp_idx]

        env, eval_env, additional_info = get_environment(environment_string,subset_concept,seed)

        lp_intervention_reward[modification] = {
            'reward': evaluate_model(environment_string,eval_env,additional_info,model,seed),
            'concepts': lp_idx
        }

    if lp_intervention_reward != {}:
        results['intervention_comparison']['lp'] = lp_intervention_reward
        print(lp_intervention_reward)

if is_main and run_intervention:
    multiple_lp_intervention_reward = {}
    modified_concept_predictors = [inaccurate_concepts_binary_intervention(func,acc,intervene_acc,0,seed+idx) for (func,acc,intervene_acc,idx) in zip(concept_list,cbm_accuracy_by_concept,intervention_accuracy_by_concept,list(range(len(concept_list))))]
    subset_concept, multiple_idx = multiple_lp_selection(ground_truth_gym_env,modified_concept_predictors,num_concepts_selected,selection_function,q_estimates,concept_source)
    env, eval_env, additional_info = get_environment(environment_string,subset_concept,seed)

    model = train_ppo_model(env,environment_string,total_timesteps=training_timesteps,policy="MlpPolicy",additional_info=additional_info)
    multiple_lp_intervention_reward = {}
    modified_concept_predictors = [inaccurate_concepts_binary_intervention(func,acc,intervene_acc,intervention_probability,seed+idx) for (func,acc,intervene_acc,idx) in zip(concept_list,cbm_accuracy_by_concept,intervention_accuracy_by_concept,list(range(len(concept_list))))]
    subset_concept = [modified_concept_predictors[i] for i in multiple_idx]
    env, eval_env, additional_info = get_environment(environment_string,subset_concept,seed)
    multiple_lp_intervention_reward['reward'] = evaluate_model(environment_string,eval_env,additional_info,model,seed)
    multiple_lp_intervention_reward['concepts'] = multiple_idx

    if multiple_lp_intervention_reward != {}:
        results['intervention_comparison']['multiple_lp'] = multiple_lp_intervention_reward
        print(multiple_lp_intervention_reward)

if is_main and run_intervention:
    imperfect_intervention_reward = {}
    for modification in ["binary"]:
        if modification == "binary" and cbm_accuracy_by_concept is not None and intervention_accuracy_by_concept is not None:
            modified_concept_predictors = [inaccurate_concepts_binary_intervention(func,acc,intervene_acc,0,seed+idx) for (func,acc,intervene_acc,idx) in zip(concept_list,cbm_accuracy_by_concept,intervention_accuracy_by_concept,list(range(len(concept_list))))]
        else:
            continue 
        subset_concept, imperfect_idx = imperfect_lp_selection(ground_truth_gym_env,modified_concept_predictors,q_estimates,selection_function,target_abstraction,num_concepts_selected,cbm_accuracy_by_concept,concept_source,environment_string,additional_info,direction='max')
        env, eval_env, additional_info = get_environment(environment_string,subset_concept,seed)
        model = train_ppo_model(env,environment_string,total_timesteps=training_timesteps,policy="MlpPolicy",additional_info=additional_info)
        modified_concept_predictors = [inaccurate_concepts_binary_intervention(func,acc,intervene_acc,intervention_probability,seed+idx) for (func,acc,intervene_acc,idx) in zip(concept_list,cbm_accuracy_by_concept,intervention_accuracy_by_concept,list(range(len(concept_list))))]
        subset_concept = [modified_concept_predictors[i] for i in imperfect_idx]
        env, eval_env, additional_info = get_environment(environment_string,subset_concept,seed)

        imperfect_intervention_reward[modification] = {
            'reward': evaluate_model(environment_string,eval_env,additional_info,model,seed),
            'concepts': imperfect_idx
        }

    if imperfect_intervention_reward != {}:
        results['intervention_comparison']['imperfect_lp'] = imperfect_intervention_reward
        print(imperfect_intervention_reward)

# ### Iterative

if is_main and run_iterative:
    env, eval_env, additional_info = get_environment(environment_string,concept_list,seed)
    gold_model = train_ppo_model(env,environment_string,total_timesteps=training_timesteps,policy="MlpPolicy",additional_info=additional_info)
    results['iterative'] = {}

if is_main and run_iterative:
    if cbm_accuracy_by_concept is None:
        modified_concept_predictors = concept_list 
    else:
        modified_concept_predictors = [inaccurate_concepts_binary(func,acc,seed+idx) for func,acc,idx in zip(concept_list,cbm_accuracy_by_concept,list(range(len(concept_list))))]
    rewards_iterative, concepts_iterative = iterative_selection(eval_env,gold_model,environment_string,modified_concept_predictors,num_iterations,selections_per_round,additional_info,seed,training_timesteps=training_timesteps)
    results['iterative']['iterative_selection'] = {'reward': rewards_iterative, 'concepts': concepts_iterative}
    print(rewards_iterative)

if is_main and run_iterative:
    if cbm_accuracy_by_concept is None:
        modified_concept_predictors = concept_list 
    else:
        modified_concept_predictors = [inaccurate_concepts_binary(func,acc,seed+idx) for func,acc,idx in zip(concept_list,cbm_accuracy_by_concept,list(range(len(concept_list))))]

    num_concepts_selected = num_iterations*selections_per_round
    bayesian_reward, bayesian_idx = bayesian_iterative_selection(ground_truth_gym_env,environment_string,seed,modified_concept_predictors,num_iterations+1,num_concepts_selected,additional_info,training_timesteps=training_timesteps)

    results['iterative']['bayesian'] = {
        'reward': bayesian_reward, 
        'concepts': bayesian_idx
    }
    print(bayesian_reward)

# ### Two-Stage Training

if is_main and run_two_stage:
    env, eval_env, additional_info = get_environment(environment_string,concept_list,seed)
    gold_model = train_ppo_model(env,environment_string,total_timesteps=training_timesteps,policy="MlpPolicy",additional_info=additional_info)

if is_main and not isinstance(ground_truth_env, ConceptEnv) and run_two_stage:
    concept_predictor, acc_list = train_concept_predictor(ground_truth_gym_env,gold_model,concept_list,list(range(len(concept_list))))
    results['two_stage'] = {}
    results['two_stage']['accuracy'] = acc_list 


# +
if is_main and not isinstance(ground_truth_env, ConceptEnv) and run_two_stage:
    greedy_two_stage = {}
    greedy_concepts, greedy_idx = greedy_selection(concept_list,num_concepts_selected,selection_function,q_estimates,concept_source)

    concept_predictor, _ = train_concept_predictor(ground_truth_gym_env,gold_model,concept_list,greedy_idx)
    two_stage_env, two_stage_gym_env, additional_info = get_environment(environment_string, greedy_concepts, seed,fast_predictor=concept_predictor,use_processed=True)   
    model = train_ppo_model(two_stage_env,environment_string,policy="MlpPolicy",total_timesteps=training_timesteps)    
    greedy_two_stage_reward = evaluate_model(environment_string,two_stage_gym_env,additional_info,model,seed)

    results['two_stage']['greedy'] = {'reward': greedy_two_stage_reward, 'concepts': greedy_idx}
    print(greedy_two_stage_reward)



# +
if is_main and not isinstance(ground_truth_env, ConceptEnv) and run_two_stage:
    greedy_iterative_concepts, greedy_iterative_idx = greedy_iterative_selection(concept_list,num_concepts_selected,selection_function,q_estimates,concept_source)

    concept_predictor, _ = train_concept_predictor(ground_truth_gym_env,gold_model,concept_list,greedy_iterative_idx)
    two_stage_env, two_stage_gym_env, additional_info = get_environment(environment_string, greedy_concepts, seed,fast_predictor=concept_predictor,use_processed=True)   
    model = train_ppo_model(two_stage_env,environment_string,policy="MlpPolicy",total_timesteps=training_timesteps)    
    greedy_iterative_two_stage_reward = evaluate_model(environment_string,two_stage_gym_env,additional_info,model,seed)

    results['two_stage']['greedy_iterative'] = {'reward': greedy_iterative_two_stage_reward, 'concepts': greedy_iterative_idx}
    print(greedy_iterative_two_stage_reward)



# +
if is_main and not isinstance(ground_truth_env, ConceptEnv) and run_two_stage:
    lp_concepts, lp_idx = lp_based_selection(ground_truth_gym_env,concept_list,num_concepts_selected,selection_function,q_estimates,concept_source)

    concept_predictor, _ = train_concept_predictor(ground_truth_gym_env,gold_model,concept_list,lp_idx)
    two_stage_env, two_stage_gym_env, additional_info = get_environment(environment_string, lp_concepts, seed,fast_predictor=concept_predictor,use_processed=True)   
    model = train_ppo_model(two_stage_env,environment_string,policy="MlpPolicy",total_timesteps=training_timesteps)    
    lp_two_stage_reward = evaluate_model(environment_string,two_stage_gym_env,additional_info,model,seed)

    results['two_stage']['lp'] = {'reward': lp_two_stage_reward, 'concepts': lp_idx}
    print(lp_two_stage_reward)



# +
if is_main and not isinstance(ground_truth_env, ConceptEnv) and run_two_stage:
    acc_list = results['two_stage']['accuracy']
    top_k_idx = np.argsort(acc_list)[-num_concepts_selected:]
    top_k_concepts = [concept_list[i] for i in top_k_idx]
    
    concept_predictor, _ = train_concept_predictor(ground_truth_gym_env,gold_model,concept_list,top_k_idx)
    two_stage_env, two_stage_gym_env, additional_info = get_environment(environment_string, top_k_concepts, seed,fast_predictor=concept_predictor,use_processed=True)   
    model = train_ppo_model(two_stage_env,environment_string,policy="MlpPolicy",total_timesteps=training_timesteps)    
    top_k_two_stage_reward = evaluate_model(environment_string,two_stage_gym_env,additional_info,model,seed)

    results['two_stage']['top_k'] = {'reward': top_k_two_stage_reward, 'concepts': top_k_idx}
    print(top_k_two_stage_reward)


# -

if is_main and not isinstance(ground_truth_env, ConceptEnv) and run_two_stage:
    acc_list = results['two_stage']['accuracy']
    imperfect_concepts, imperfect_idx = imperfect_lp_selection(ground_truth_gym_env,concept_list,q_estimates,selection_function,target_abstraction,num_concepts_selected,acc_list,concept_source,environment_string,additional_info,direction='max')
    
    concept_predictor, _ = train_concept_predictor(ground_truth_gym_env,gold_model,concept_list,imperfect_idx)
    two_stage_env, two_stage_gym_env, additional_info = get_environment(environment_string, imperfect_concepts, seed,fast_predictor=concept_predictor,use_processed=True)   
    model = train_ppo_model(two_stage_env,environment_string,policy="MlpPolicy",total_timesteps=training_timesteps)    
    imperfect_two_stage_reward = evaluate_model(environment_string,two_stage_gym_env,additional_info,model,seed)

    results['two_stage']['imperfect'] = {'reward': imperfect_two_stage_reward, 'concepts': imperfect_idx}
    print(imperfect_two_stage_reward)

# ## Ablations

# ### Reward Perturbation

if is_main and reward_error > 0:
    results['reward_error'] = {}
    perturbed_groundtruth_eval_env = RewardPerturbationWrapper(ground_truth_gym_env,reward_error)

    if selection_function == "q_value":
        if environment_string == "mimic":
            q_estimates_perturbed = rollout_q_estimates_td(groundtruth_model,GymnasiumWrapper(DummyVecEnv([lambda: ground_truth_gym_env])),concept_list,learning_rate=1e-3,mimic=True,total_timesteps=5000,final_training=0)
        else:
            q_estimates_perturbed = rollout_q_estimates_td(groundtruth_model,ground_truth_gym_env,concept_list)
    elif selection_function == "policy":
        if environment_string == "mimic":
            q_estimates_perturbed = rollout_pi_estimates(groundtruth_model,GymnasiumWrapper(DummyVecEnv([lambda: ground_truth_gym_env])),concept_list,mimic=True)
        else:
            q_estimates_perturbed = rollout_pi_estimates(groundtruth_model,ground_truth_gym_env,concept_list)

    subset_concept, greedy_perturbed_idx = greedy_selection(concept_list,num_concepts_selected,selection_function,q_estimates_perturbed,concept_source)
    env, eval_env, additional_info = get_environment(environment_string,subset_concept,seed)
    perturbed_model = train_ppo_model(env,environment_string,total_timesteps=training_timesteps,policy="MlpPolicy",additional_info=additional_info)
    greedy_selection_perturbed_reward = evaluate_model(environment_string,eval_env,additional_info,perturbed_model,seed)
    results['reward_error']['greedy'] = {
        'reward': greedy_selection_perturbed_reward,
        'concepts': greedy_perturbed_idx
    }
    print(greedy_selection_perturbed_reward)

if is_main and reward_error > 0:
    subset_concept, greedy_perturbed_iterative_idx = greedy_iterative_selection(concept_list,num_concepts_selected,selection_function,q_estimates_perturbed,concept_source)
    env, eval_env, additional_info = get_environment(environment_string,subset_concept,seed)
    model = train_ppo_model(env,environment_string,total_timesteps=training_timesteps,policy="MlpPolicy",additional_info=additional_info)
    greedy_iterative_selection_perturbed_reward = evaluate_model(environment_string,eval_env,additional_info,model,seed)
    results['reward_error']['greedy_iterative'] = {
        'reward': greedy_iterative_selection_perturbed_reward,
        'concepts': greedy_perturbed_iterative_idx
    }
    print(greedy_iterative_selection_perturbed_reward)

if is_main and reward_error > 0:
    subset_concept, lp_perturbed_idx = lp_based_selection(concept_list,num_concepts_selected,selection_function,q_estimates_perturbed,concept_source)
    env, eval_env, additional_info = get_environment(environment_string,subset_concept,seed)
    model = train_ppo_model(env,environment_string,total_timesteps=training_timesteps,policy="MlpPolicy",additional_info=additional_info)
    lp_selection_perturbed_reward = evaluate_model(environment_string,eval_env,additional_info,model,seed)
    results['reward_error']['lp'] = {
        'reward': lp_selection_perturbed_reward,
        'concepts': lp_perturbed_idx
    }
    print(lp_selection_perturbed_reward)

# ### Comparison with Concept Completeness

# TODO: Create a Shapley-based baseline
if is_main and assess_completeness:
    pass 


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
