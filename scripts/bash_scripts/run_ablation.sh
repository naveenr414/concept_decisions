#!/bin/bash 
: > runs/logs/error_ablation.txt
LOGFILE=../../runs/logs/error_ablation.txt

environment=food
tmux new-session -d -s concepts
tmux send-keys -t concepts ENTER 
tmux send-keys -t concepts "cd scripts/notebooks" ENTER

# declare -A training_timesteps=(
#   [cyclic_4]=10000
#   [cyclic_16]=20000
#   [tree_7]=15000
#   [tree_31]=30000
#   [cart_pole]=50000
#   [mini_grid]=50000
#   [pong]=1000000
#   [boxing]=1200000
#   [mimic]=25000
# )

declare -A training_timesteps=(
  [cyclic_4]=1
  [cyclic_16]=1
  [tree_7]=1
  [tree_31]=1
  [cart_pole]=1
  [mini_grid]=1
  [pong]=1
  [boxing]=1
  [mimic]=1
)



for seed in 42
do 
    # Reward Perturbation
    num_concepts_selected=4
    
    for reward_error in 0.1 # 0.01 0.05 0.1 0.2 
    do 
        for env in cyclic_4 tree_7 boxing # cyclic_4 cyclic_16 tree_7 tree_31 cart_pole mini_grid pong boxing mimic
        do 
            training=${training_timesteps[$env]}
            tmux send-keys -t concepts "conda activate ${environment}; python main_experiments.py --seed ${seed} --environment_string ${env} --training_timesteps ${training} --num_concepts_selected ${num_concepts_selected} --selection_function q_value --out_folder ablation --concept_source human_selected_binary --run_basic >> ${LOGFILE} 2>&1"  ENTER 
        done 
    done 

    # Policy as a selector
    num_concepts_selected=4
    for env in cyclic_4 tree_7 cart_pole pong mimic # cyclic_4 cyclic_16 tree_7 tree_31 cart_pole mini_grid pong boxing mimic
    do 
        training=${training_timesteps[$env]}
        tmux send-keys -t concepts "conda activate ${environment}; python main_experiments.py --seed ${seed} --environment_string ${env} --training_timesteps ${training} --num_concepts_selected ${num_concepts_selected} --selection_function policy --out_folder ablation --concept_source human_selected_binary --run_basic >> ${LOGFILE} 2>&1"  ENTER 
    done 

    # Training Timesteps
    num_concepts_selected=4
    for env in cyclic_4 tree_7 cart_pole pong mimic # cyclic_4 cyclic_16 tree_7 tree_31 cart_pole mini_grid pong boxing mimic
    do 
        steps=${training_timesteps[$env]}
        half_steps=$(( steps / 2 ))
        tmux send-keys -t concepts "conda activate ${environment}; python main_experiments.py --seed ${seed} --environment_string ${env} --training_timesteps ${half_steps} --num_concepts_selected ${num_concepts_selected} --selection_function q_value --out_folder ablation --concept_source human_selected_binary --run_basic >> ${LOGFILE} 2>&1"  ENTER 
    done 

    # Concept Completeness Baseline
    # TODO: Implement Concept Completeness Baseline
done 
