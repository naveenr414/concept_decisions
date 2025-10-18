#!/bin/bash 
: > runs/logs/error_concept_two_stage.txt
LOGFILE=../../runs/logs/error_concept_two_stage.txt

environment=food
tmux new-session -d -s concept_two_stage
tmux send-keys -t concept_two_stage ENTER 
tmux send-keys -t concept_two_stage "source ~/.bashrc" ENTER
tmux send-keys -t concept_two_stage "cd scripts/notebooks" ENTER
tmux send-keys -t concept_two_stage "export PYTHONWARNINGS='ignore'" ENTER
tmux send-keys -t concept_two_stage "export GYMNASIUM_DISABLE_WARNINGS=1" ENTER


declare -A training_timesteps=(
  [cart_pole]=1000000
  [mini_grid]=250000
  [pong]=4000000
  [boxing]=4000000
)

for seed in 42
do 
  env=mini_grid
  for num_concepts_selected in 40
  do 
    training=${training_timesteps[$env]}
    tmux send-keys -t concept_two_stage "conda activate ${environment}; python -u  main_experiments.py --seed ${seed} --environment_string ${env} --gold_timesteps 4000000 --training_timesteps ${training} --num_concepts_selected ${num_concepts_selected} --selection_function q_value --out_folder two_stage --concept_source human_selected_binary --run_two_stage >> ${LOGFILE} 2>&1"  ENTER 
  done 

  env=pong
  training=${training_timesteps[$env]}
  for num_concepts_selected in 80
  do 
    tmux send-keys -t concept_two_stage "conda activate ${environment}; python -u main_experiments.py --seed ${seed} --environment_string ${env} --gold_timesteps 4000000 --training_timesteps ${training} --num_concepts_selected ${num_concepts_selected} --selection_function q_value --out_folder two_stage --concept_source human_selected_binary --run_two_stage >> ${LOGFILE} 2>&1"  ENTER 
  done 

  env=boxing
  training=${training_timesteps[$env]}
  for num_concepts_selected in 80
  do 
    tmux send-keys -t concept_two_stage "conda activate ${environment}; python -u  main_experiments.py --seed ${seed} --environment_string ${env} --gold_timesteps 10000000 --training_timesteps ${training} --num_concepts_selected ${num_concepts_selected} --selection_function q_value --out_folder two_stage --concept_source human_selected_binary --run_two_stage >> ${LOGFILE} 2>&1"  ENTER 
  done 

  env=cart_pole
  for num_concepts_selected in 15
  do 
    training=${training_timesteps[$env]}
    tmux send-keys -t concept_two_stage "conda activate ${environment}; python -u  main_experiments.py --seed ${seed} --environment_string ${env} --gold_timesteps 4000000 --training_timesteps ${training} --num_concepts_selected ${num_concepts_selected} --selection_function q_value --out_folder two_stage --concept_source human_selected_binary --run_two_stage >> ${LOGFILE} 2>&1"  ENTER 
  done
done 
