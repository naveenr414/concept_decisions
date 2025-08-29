#!/bin/bash 
: > runs/logs/error_imperfect.txt
LOGFILE=../../runs/logs/error_imperfect.txt

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
    # Artificial Noise
    env=cycle_4
    num_concepts_selected=4
    training=${training_timesteps[$env]}
    for fixed_accuracy in 0.1 0.25 0.5 0.75 0.9 0.99
    do 
        cbm_accuracy=$(printf "%s " $(yes "$fixed_accuracy" | head -n 3))
        tmux send-keys -t concepts "conda activate ${environment}; python main_experiments.py --seed ${seed} --environment_string ${env} --training_timesteps ${training} --num_concepts_selected ${num_concepts_selected} --selection_function q_value --out_folder imperfect --concept_source human_selected_binary --run_basic --cbm_accuracy_by_concept ${cbm_accuracy} >> ${LOGFILE} 2>&1"  ENTER 
    done 

    env=cycle_16
    num_concepts_selected=4
    training=${training_timesteps[$env]}
    for fixed_accuracy in 0.1 0.25 0.5 0.75 0.9 0.99
    do 
        cbm_accuracy=$(printf "%s " $(yes "$fixed_accuracy" | head -n 15))
        tmux send-keys -t concepts "conda activate ${environment}; python main_experiments.py --seed ${seed} --environment_string ${env} --training_timesteps ${training} --num_concepts_selected ${num_concepts_selected} --selection_function q_value --out_folder imperfect --concept_source human_selected_binary --run_basic --cbm_accuracy_by_concept ${cbm_accuracy} >> ${LOGFILE} 2>&1"  ENTER 
    done 


    env=tree_7
    training=${training_timesteps[$env]}
    for fixed_accuracy in 0.1 0.25 0.5 0.75 0.9 0.99
    do 
        cbm_accuracy=$(printf "%s " $(yes "$fixed_accuracy" | head -n 4))
        tmux send-keys -t concepts "conda activate ${environment}; python main_experiments.py --seed ${seed} --environment_string ${env} --training_timesteps ${training} --num_concepts_selected ${num_concepts_selected} --selection_function q_value --out_folder imperfect --concept_source human_selected_binary --run_basic --cbm_accuracy_by_concept ${cbm_accuracy} >> ${LOGFILE} 2>&1"  ENTER 
    done 

    env=tree_15
    training=${training_timesteps[$env]}
    for fixed_accuracy in 0.1 0.25 0.5 0.75 0.9 0.99
    do 
        cbm_accuracy=$(printf "%s " $(yes "$fixed_accuracy" | head -n 5))
        tmux send-keys -t concepts "conda activate ${environment}; python main_experiments.py --seed ${seed} --environment_string ${env} --training_timesteps ${training} --num_concepts_selected ${num_concepts_selected} --selection_function q_value --out_folder imperfect --concept_source human_selected_binary --run_basic --cbm_accuracy_by_concept ${cbm_accuracy} >> ${LOGFILE} 2>&1"  ENTER 
    done 

    env=cart_pole
    training=${training_timesteps[$env]}
    for fixed_accuracy in 0.25 0.5 0.75
    do 
        cbm_accuracy=$(printf "%s " $(yes "$fixed_accuracy" | head -n 8))
        tmux send-keys -t concepts "conda activate ${environment}; python main_experiments.py --seed ${seed} --environment_string ${env} --training_timesteps ${training} --num_concepts_selected ${num_concepts_selected} --selection_function q_value --out_folder imperfect --concept_source human_selected_binary --run_basic --cbm_accuracy_by_concept ${cbm_accuracy} >> ${LOGFILE} 2>&1"  ENTER 
    done 

    env=pong
    training=${training_timesteps[$env]}
    for fixed_accuracy in 0.25 0.5 0.75
    do 
        cbm_accuracy=$(printf "%s " $(yes "$fixed_accuracy" | head -n 14))
        tmux send-keys -t concepts "conda activate ${environment}; python main_experiments.py --seed ${seed} --environment_string ${env} --training_timesteps ${training} --num_concepts_selected ${num_concepts_selected} --selection_function q_value --out_folder imperfect --concept_source human_selected_binary --run_basic --cbm_accuracy_by_concept ${cbm_accuracy} >> ${LOGFILE} 2>&1"  ENTER 
    done 

    env=boxing
    training=${training_timesteps[$env]}
    for fixed_accuracy in 0.25 0.5 0.75
    do 
        cbm_accuracy=$(printf "%s " $(yes "$fixed_accuracy" | head -n 16))
        tmux send-keys -t concepts "conda activate ${environment}; python main_experiments.py --seed ${seed} --environment_string ${env} --training_timesteps ${training} --num_concepts_selected ${num_concepts_selected} --selection_function q_value --out_folder imperfect --concept_source human_selected_binary --run_basic --cbm_accuracy_by_concept ${cbm_accuracy} >> ${LOGFILE} 2>&1"  ENTER 
    done 

    env=mimic
    training=${training_timesteps[$env]}
    for fixed_accuracy in 0.25 0.5 0.75
    do 
        cbm_accuracy=$(printf "%s " $(yes "$fixed_accuracy" | head -n 82))
        tmux send-keys -t concepts "conda activate ${environment}; python main_experiments.py --seed ${seed} --environment_string ${env} --training_timesteps ${training} --num_concepts_selected ${num_concepts_selected} --selection_function q_value --out_folder imperfect --concept_source human_selected_binary --run_basic --cbm_accuracy_by_concept ${cbm_accuracy} >> ${LOGFILE} 2>&1"  ENTER 
    done 

    # Two Stage
    num_concepts_selected=4
    for env in cart_pole pong boxing
    do 
        training=${training_timesteps[$env]}
        tmux send-keys -t concepts "conda activate ${environment}; python main_experiments.py --seed ${seed} --environment_string ${env} --training_timesteps ${training} --num_concepts_selected ${num_concepts_selected} --selection_function q_value --out_folder basic --concept_source human_selected_binary --run_basic --run_two_stage >> ${LOGFILE} 2>&1"  ENTER 
    done 
done 
