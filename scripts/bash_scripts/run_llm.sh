#!/bin/bash 
: > runs/logs/error_llm.txt
LOGFILE=../../runs/logs/error_llm.txt

environment=food
tmux new-session -d -s concepts
tmux send-keys -t concepts ENTER 
tmux send-keys -t concepts "cd scripts/notebooks" ENTER

for seed in 42
do 
    env=door_key
    training=10000
    for num_concepts_selected in 1 2 3 4 
    do 
        tmux send-keys -t concepts "conda activate ${environment}; python main_experiments.py --seed ${seed} --environment_string ${env} --training_timesteps ${training} --num_concepts_selected ${num_concepts_selected} --selection_function q_value --out_folder llm --concept_source llm_selected --run_basic --run_two_stage >> ${LOGFILE} 2>&1"  ENTER 
    done 
done 
