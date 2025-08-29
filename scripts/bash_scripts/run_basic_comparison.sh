#!/bin/bash 
: > runs/logs/error_concepts.txt
LOGFILE=../../runs/logs/error_concepts.txt

environment=food
tmux new-session -d -s concepts
tmux send-keys -t concepts ENTER 
tmux send-keys -t concepts "cd scripts/notebooks" ENTER

declare -A training_timesteps=(
  [cycle_4]=10000
  [cycle_16]=20000
  [tree_7]=15000
  [tree_31]=30000
  [cart_pole]=50000
  [pong]=1000000
  [boxing]=1200000
  [mimic]=25000
)


for seed in 42
do 

    for num_concepts_selected in 1 2 3 4 
    do 
        for env in cycle_4 cycle_16 tree_7 tree_31 cart_pole pong boxing mimic
        do 
            training=${training_timesteps[$env]}
            tmux send-keys -t concepts "conda activate ${environment}; python main_experiments.py --seed ${seed} --environment_string ${env} --training_timesteps ${training} --num_concepts_selected ${num_concepts_selected} --selection_function q_value --out_folder basic --concept_source human_selected_binary --run_basic >> ${LOGFILE} 2>&1"  ENTER 
        done 
    done 
    
    num_concepts_selected=8
    env=mimic
    tmux send-keys -t concepts "conda activate ${environment}; python main_experiments.py --seed ${seed} --environment_string ${env} --training_timesteps ${training} --num_concepts_selected ${num_concepts_selected} --selection_function q_value --out_folder basic --concept_source human_selected_binary --run_basic >> ${LOGFILE} 2>&1"  ENTER 

    # Continuous 
    for num_concepts_selected in 1 2 3 4 
    do 
        for env in cart_pole pong boxing mimic
        do 
            training=${training_timesteps[$env]}
            tmux send-keys -t concepts "conda activate ${environment}; python main_experiments.py --seed ${seed} --environment_string ${env} --training_timesteps ${training} --num_concepts_selected ${num_concepts_selected} --selection_function q_value --out_folder basic --concept_source human_selected --run_basic  >> ${LOGFILE} 2>&1"  ENTER 
        done 
    done 
done 
