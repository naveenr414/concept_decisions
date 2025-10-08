#!/bin/bash 
: > runs/logs/error_concepts.txt
LOGFILE=../../runs/logs/error_concepts.txt

: > runs/logs/error_glucose.txt
LOGFILE_GLUCOSE=../../runs/logs/error_glucose.txt

environment=food
tmux new-session -d -s concepts
tmux send-keys -t concepts ENTER 
tmux send-keys -t concepts "source ~/.bashrc" ENTER
tmux send-keys -t concepts "cd scripts/notebooks" ENTER
tmux send-keys -t concepts "export PYTHONWARNINGS='ignore'" ENTER
tmux send-keys -t concepts "export GYMNASIUM_DISABLE_WARNINGS=1" ENTER

tmux new-session -d -s concepts_glucose
tmux send-keys -t concepts_glucose ENTER 
tmux send-keys -t concepts_glucose "source ~/.bashrc" ENTER
tmux send-keys -t concepts_glucose "cd scripts/notebooks" ENTER
tmux send-keys -t concepts_glucose "export PYTHONWARNINGS='ignore'" ENTER
tmux send-keys -t concepts_glucose "export GYMNASIUM_DISABLE_WARNINGS=1" ENTER


declare -A training_timesteps=(
  [cyclic_4]=10000
  [cyclic_16]=10000
  [tree_7]=25000
  [tree_31]=25000
  [cart_pole]=1000000
  [mini_grid]=250000
  [glucose]=500000
  [pong]=10000000
  [boxing]=10000000
)

for seed in 42 # 43 44 45 46 47 48 49 50 51 52 53 54 55 56
do 
  # env=cyclic_4
  # for num_concepts_selected in 1 2 3
  # do 
  #   training=${training_timesteps[$env]}
  #   tmux send-keys -t concepts "conda activate ${environment}; python main_experiments.py --seed ${seed} --environment_string ${env} --training_timesteps ${training} --num_concepts_selected ${num_concepts_selected} --selection_function q_value --out_folder basic --concept_source human_selected_binary --run_basic >> ${LOGFILE} 2>&1"  ENTER 
  # done 

  # env=cyclic_16
  # for num_concepts_selected in 2 4 6 8
  # do 
  #   training=${training_timesteps[$env]}
  #   tmux send-keys -t concepts "conda activate ${environment}; python main_experiments.py --seed ${seed} --environment_string ${env} --training_timesteps ${training} --num_concepts_selected ${num_concepts_selected} --selection_function q_value --out_folder basic --concept_source human_selected_binary --run_basic >> ${LOGFILE} 2>&1"  ENTER 
  # done 

  # env=tree_7
  # for num_concepts_selected in 1 2 3 4
  # do 
  #   training=${training_timesteps[$env]}
  #   tmux send-keys -t concepts "conda activate ${environment}; python main_experiments.py --seed ${seed} --environment_string ${env} --training_timesteps ${training} --num_concepts_selected ${num_concepts_selected} --selection_function q_value --out_folder basic --concept_source human_selected_binary --run_basic >> ${LOGFILE} 2>&1"  ENTER 
  # done 

  # env=tree_31
  # for num_concepts_selected in 4 5 # 1 2 3 4 5
  # do 
  #   training=${training_timesteps[$env]}
  #   tmux send-keys -t concepts "conda activate ${environment}; python main_experiments.py --seed ${seed} --environment_string ${env} --training_timesteps ${training} --num_concepts_selected ${num_concepts_selected} --selection_function q_value --out_folder basic --concept_source human_selected_binary --run_basic >> ${LOGFILE} 2>&1"  ENTER 
  # done 

  # env=cart_pole
  # for num_concepts_selected in 5 10 15
  # do 
  #   training=${training_timesteps[$env]}
  #   tmux send-keys -t concepts "conda activate ${environment}; python main_experiments.py --seed ${seed} --environment_string ${env} --gold_timesteps 4000000 --training_timesteps ${training} --num_concepts_selected ${num_concepts_selected} --selection_function q_value --out_folder basic --concept_source human_selected_binary --run_basic >> ${LOGFILE} 2>&1"  ENTER 
  # done

  # env=mini_grid
  # for num_concepts_selected in 20 30 40
  # do 
  #   training=${training_timesteps[$env]}
  #   tmux send-keys -t concepts "conda activate ${environment}; python main_experiments.py --seed ${seed} --environment_string ${env} --gold_timesteps 4000000 --training_timesteps ${training} --num_concepts_selected ${num_concepts_selected} --selection_function q_value --out_folder basic --concept_source human_selected_binary --run_basic >> ${LOGFILE} 2>&1"  ENTER 
  # done 

  # env=mimic
  # for num_concepts_selected in 20 40 60 80
  # do 
  #   training=${training_timesteps[$env]}
  #   tmux send-keys -t concepts "conda activate ${environment}; python main_experiments.py --seed ${seed} --environment_string ${env} --gold_timesteps 250000 --training_timesteps ${training} --num_concepts_selected ${num_concepts_selected} --selection_function q_value --out_folder basic --concept_source human_selected_binary --run_basic >> ${LOGFILE} 2>&1"  ENTER 
  # done

  env=pong
  training=${training_timesteps[$env]}
  for num_concepts_selected in 20 #30 40
  do 
    tmux send-keys -t concepts "conda activate ${environment}; python main_experiments.py --seed ${seed} --environment_string ${env} --gold_timesteps 4000000 --training_timesteps ${training} --num_concepts_selected ${num_concepts_selected} --selection_function q_value --out_folder basic --concept_source human_selected_binary --run_basic >> ${LOGFILE} 2>&1"  ENTER 
  done 

  # env=boxing
  # training=${training_timesteps[$env]}
  # for num_concepts_selected in 20 #30 40
  # do 
  #   tmux send-keys -t concepts "conda activate ${environment}; python main_experiments.py --seed ${seed} --environment_string ${env} --gold_timesteps 10000000 --training_timesteps ${training} --num_concepts_selected ${num_concepts_selected} --selection_function q_value --out_folder basic --concept_source human_selected_binary --run_basic >> ${LOGFILE} 2>&1"  ENTER 
  # done 

  # env=glucose
  # training=${training_timesteps[$env]}
  # for num_concepts_selected in 40 #60 80
  # do 
  #   tmux send-keys -t concepts_glucose "conda activate ${environment}; python main_experiments.py --seed ${seed} --environment_string ${env} --gold_timesteps 500000 --training_timesteps ${training} --num_concepts_selected ${num_concepts_selected} --selection_function q_value --out_folder basic --concept_source human_selected_binary --run_basic >> ${LOGFILE_GLUCOSE} 2>&1"  ENTER 
  # done 
done 
