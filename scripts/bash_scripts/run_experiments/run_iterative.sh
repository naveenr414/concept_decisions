#!/bin/bash 
: > runs/logs/error_iterative.txt
LOGFILE=../../runs/logs/error_iterative.txt

environment=food
tmux new-session -d -s concept_iterative
tmux send-keys -t concept_iterative ENTER 
tmux send-keys -t concept_iterative "source ~/.bashrc" ENTER
tmux send-keys -t concept_iterative "cd scripts/notebooks" ENTER
tmux send-keys -t concept_iterative "export PYTHONWARNINGS='ignore'" ENTER
tmux send-keys -t concept_iterative "export GYMNASIUM_DISABLE_WARNINGS=1" ENTER

declare -A training_timesteps=(
  [cyclic_4]=10000
  [cyclic_16]=10000
  [tree_7]=25000
  [tree_31]=25000
  [cart_pole]=1000000
  [mini_grid]=250000
  [mimic]=250000
  [pong]=20000000
  [boxing]=12000000
)


for seed in 42 43 44
do 
  # env=cyclic_4
  # num_concepts_selected=1
  # num_iterations=2
  # training=${training_timesteps[$env]}
  # tmux send-keys -t concept_iterative "conda activate ${environment}; python main_experiments.py --seed ${seed} --environment_string ${env} --training_timesteps ${training} --num_iterations ${num_iterations} --selections_per_round ${num_concepts_selected} --selection_function q_value --out_folder iterative --concept_source human_selected_binary --run_iterative >> ${LOGFILE} 2>&1"  ENTER 
  # cbm_accuracy="0.75 0.75 0.75"
  # tmux send-keys -t concept_iterative "conda activate ${environment}; python main_experiments.py --seed ${seed} --environment_string ${env} --training_timesteps ${training} --num_iterations ${num_iterations} --selections_per_round ${num_concepts_selected} --selection_function q_value --out_folder iterative --concept_source human_selected_binary --cbm_accuracy_by_concept ${cbm_accuracy} --run_iterative >> ${LOGFILE} 2>&1"  ENTER 

  # env=cyclic_16
  # num_concepts_selected=1
  # num_iterations=2
  # training=${training_timesteps[$env]}
  # tmux send-keys -t concept_iterative "conda activate ${environment}; python main_experiments.py --seed ${seed} --environment_string ${env} --training_timesteps ${training} --num_iterations ${num_iterations} --selections_per_round ${num_concepts_selected} --selection_function q_value --out_folder iterative --concept_source human_selected_binary --run_iterative >> ${LOGFILE} 2>&1"  ENTER 
  # cbm_accuracy="0.75 0.75 0.75 0.75 0.75 0.75 0.75 0.75 0.75 0.75 0.75 0.75 0.75 0.75 0.75"
  # tmux send-keys -t concept_iterative "conda activate ${environment}; python main_experiments.py --seed ${seed} --environment_string ${env} --training_timesteps ${training} --num_iterations ${num_iterations} --selections_per_round ${num_concepts_selected} --selection_function q_value --out_folder iterative --concept_source human_selected_binary --cbm_accuracy_by_concept ${cbm_accuracy} --run_iterative >> ${LOGFILE} 2>&1"  ENTER 

  # env=cyclic_16
  # num_concepts_selected=2
  # num_iterations=2
  # training=${training_timesteps[$env]}
  # tmux send-keys -t concept_iterative "conda activate ${environment}; python main_experiments.py --seed ${seed} --environment_string ${env} --training_timesteps ${training} --num_iterations ${num_iterations} --selections_per_round ${num_concepts_selected} --selection_function q_value --out_folder iterative --concept_source human_selected_binary --run_iterative >> ${LOGFILE} 2>&1"  ENTER 

  # env=tree_7
  # num_concepts_selected=1
  # num_iterations=2
  # training=${training_timesteps[$env]}
  # tmux send-keys -t concept_iterative "conda activate ${environment}; python main_experiments.py --seed ${seed} --environment_string ${env} --training_timesteps ${training} --num_iterations ${num_iterations} --selections_per_round ${num_concepts_selected} --selection_function q_value --out_folder iterative --concept_source human_selected_binary --run_iterative >> ${LOGFILE} 2>&1"  ENTER 
  # cbm_accuracy="0.75 0.75 0.75 0.75"
  # tmux send-keys -t concept_iterative "conda activate ${environment}; python main_experiments.py --seed ${seed} --environment_string ${env} --training_timesteps ${training} --num_iterations ${num_iterations} --selections_per_round ${num_concepts_selected} --selection_function q_value --out_folder iterative --concept_source human_selected_binary --cbm_accuracy_by_concept ${cbm_accuracy} --run_iterative >> ${LOGFILE} 2>&1"  ENTER 

  # env=tree_31
  # num_concepts_selected=1
  # num_iterations=2
  # training=${training_timesteps[$env]}
  # tmux send-keys -t concept_iterative "conda activate ${environment}; python main_experiments.py --seed ${seed} --environment_string ${env} --training_timesteps ${training} --num_iterations ${num_iterations} --selections_per_round ${num_concepts_selected} --selection_function q_value --out_folder iterative --concept_source human_selected_binary --run_iterative >> ${LOGFILE} 2>&1"  ENTER 
  # cbm_accuracy="0.75 0.75 0.75 0.75 0.75 0.75"
  # tmux send-keys -t concept_iterative "conda activate ${environment}; python main_experiments.py --seed ${seed} --environment_string ${env} --training_timesteps ${training} --num_iterations ${num_iterations} --selections_per_round ${num_concepts_selected} --selection_function q_value --out_folder iterative --concept_source human_selected_binary --cbm_accuracy_by_concept ${cbm_accuracy} --run_iterative >> ${LOGFILE} 2>&1"  ENTER 


  env=cart_pole
  num_concepts_selected=5
  num_iterations=3
  training=${training_timesteps[$env]}
  fixed_accuracy=0.9
  cbm_accuracy=$(printf "%s " $(yes "$fixed_accuracy" | head -n 24))
  tmux send-keys -t concept_iterative "conda activate ${environment}; python main_experiments.py --seed ${seed} --environment_string ${env} --training_timesteps ${training} --gold_timesteps 4000000 --num_iterations ${num_iterations} --selections_per_round ${num_concepts_selected} --selection_function q_value --out_folder iterative --concept_source human_selected_binary --cbm_accuracy_by_concept ${cbm_accuracy} --run_iterative >> ${LOGFILE} 2>&1"  ENTER 

  env=mini_grid
  num_concepts_selected=10
  num_iterations=4
  training=${training_timesteps[$env]}
  fixed_accuracy=0.9
  cbm_accuracy=$(printf "%s " $(yes "$fixed_accuracy" | head -n 44))
  tmux send-keys -t concept_iterative "conda activate ${environment}; python main_experiments.py --seed ${seed} --environment_string ${env} --training_timesteps ${training} --gold_timesteps 4000000 --num_iterations ${num_iterations} --selections_per_round ${num_concepts_selected} --selection_function q_value --out_folder iterative --concept_source human_selected_binary --cbm_accuracy_by_concept ${cbm_accuracy} --run_iterative >> ${LOGFILE} 2>&1"  ENTER 

  # env=mimic
  # num_concepts_selected=10
  # num_iterations=4
  # training=${training_timesteps[$env]}
  # fixed_accuracy=0.75
  # cbm_accuracy=$(printf "%s " $(yes "$fixed_accuracy" | head -n 141))
  # tmux send-keys -t concept_iterative "conda activate ${environment}; python main_experiments.py --seed ${seed} --environment_string ${env} --training_timesteps ${training} --gold_timesteps 250000 --num_iterations ${num_iterations} --selections_per_round ${num_concepts_selected} --selection_function q_value --out_folder iterative --concept_source human_selected_binary --cbm_accuracy_by_concept ${cbm_accuracy} --run_iterative >> ${LOGFILE} 2>&1"  ENTER 
done 
