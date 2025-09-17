#!/bin/bash 
: > runs/logs/error_concepts.txt
LOGFILE=../../runs/logs/error_concepts.txt

environment=food
tmux new-session -d -s concepts
tmux send-keys -t concepts ENTER 
tmux send-keys -t concepts "cd scripts/notebooks" ENTER

declare -A training_timesteps=(
  [cyclic_4]=2500
  [cyclic_16]=2500
  [tree_7]=2500
  [tree_31]=50000
  [cart_pole]=250000
  [mini_grid]=12000000
  [pong]=20000000
  [boxing]=12000000
  [mimic]=250000
)

# declare -A training_timesteps=(
#   [cyclic_4]=1
#   [cyclic_16]=1
#   [tree_7]=1
#   [tree_31]=1
#   [cart_pole]=1
#   [mini_grid]=1
#   [pong]=1
#   [boxing]=1
#   [mimic]=1
# )

for seed in 42
do 

    # for num_concepts_selected in 2 # 1 2 3 4 
    # do 
    #     for env in tree_31 # cyclic_4 cyclic_16 tree_7 tree_31 cart_pole mini_grid pong boxing mimic
    #     do 
    #         training=${training_timesteps[$env]}
    #         tmux send-keys -t concepts "conda activate ${environment}; python main_experiments.py --seed ${seed} --environment_string ${env} --training_timesteps ${training} --num_concepts_selected ${num_concepts_selected} --selection_function q_value --out_folder basic --concept_source human_selected_binary --run_basic >> ${LOGFILE} 2>&1"  ENTER 
    #     done 
    # done 

    # num_concepts_selected=28
    # env=mimic
    # training=${training_timesteps[$env]}
    # tmux send-keys -t concepts "conda activate ${environment}; python main_experiments.py --seed ${seed} --environment_string ${env} --training_timesteps ${training} --num_concepts_selected ${num_concepts_selected} --selection_function q_value --out_folder basic --concept_source human_selected --run_basic >> ${LOGFILE} 2>&1"  ENTER 

    # num_concepts_selected=50
    # env=mimic
    # training=${training_timesteps[$env]}
    # tmux send-keys -t concepts "conda activate ${environment}; python main_experiments.py --seed ${seed} --environment_string ${env} --training_timesteps ${training} --num_concepts_selected ${num_concepts_selected} --selection_function q_value --out_folder basic --concept_source human_selected_binary --run_basic >> ${LOGFILE} 2>&1"  ENTER 

    # num_concepts_selected=20
    # env=mini_grid
    # training=${training_timesteps[$env]}
    # tmux send-keys -t concepts "conda activate ${environment}; python main_experiments.py --seed ${seed} --environment_string ${env} --training_timesteps ${training} --num_concepts_selected ${num_concepts_selected} --selection_function q_value --out_folder basic --concept_source human_selected_binary --run_basic >> ${LOGFILE} 2>&1"  ENTER 

    # num_concepts_selected=25
    # env=cart_pole
    # training=1500000
    # tmux send-keys -t concepts "conda activate ${environment}; python main_experiments.py --seed ${seed} --environment_string ${env} --training_timesteps ${training} --num_concepts_selected ${num_concepts_selected} --selection_function q_value --out_folder basic --concept_source human_selected_binary --run_basic >> ${LOGFILE} 2>&1"  ENTER 

    # num_concepts_selected=48
    # env=boxing
    # training=${training_timesteps[$env]}
    # tmux send-keys -t concepts "conda activate ${environment}; python main_experiments.py --seed ${seed} --environment_string ${env} --training_timesteps 3000000 --num_concepts_selected ${num_concepts_selected} --selection_function q_value --out_folder basic --concept_source human_selected_binary --run_basic >> ${LOGFILE} 2>&1"  ENTER 

    num_concepts_selected=25
    env=pong
    training=${training_timesteps[$env]}
    tmux send-keys -t concepts "conda activate ${environment}; python main_experiments.py --seed ${seed} --environment_string ${env} --training_timesteps 6000000 --num_concepts_selected ${num_concepts_selected} --selection_function q_value --out_folder basic --concept_source human_selected_binary --run_basic >> ${LOGFILE} 2>&1"  ENTER 

    num_concepts_selected=30
    env=boxing
    training=${training_timesteps[$env]}
    tmux send-keys -t concepts "conda activate ${environment}; python main_experiments.py --seed ${seed} --environment_string ${env} --training_timesteps 6000000 --num_concepts_selected ${num_concepts_selected} --selection_function q_value --out_folder basic --concept_source human_selected_binary --run_basic >> ${LOGFILE} 2>&1"  ENTER 

    # for num_concepts_selected in 8 # 1 2 3 4 
    # do 
    #     for env in pong # cart_pole pong boxing mini_grid # cyclic_4 cyclic_16 tree_7 tree_31 cart_pole mini_grid pong boxing mimic
    #     do 
    #         training=${training_timesteps[$env]}
    #         tmux send-keys -t concepts "conda activate ${environment}; python main_experiments.py --seed ${seed} --environment_string ${env} --training_timesteps ${training} --num_concepts_selected ${num_concepts_selected} --selection_function q_value --out_folder basic --concept_source human_selected_binary --run_basic >> ${LOGFILE} 2>&1"  ENTER 
    #     done 
    # done 

    # num_concepts_selected=8
    # env=mimic
    # tmux send-keys -t concepts "conda activate ${environment}; python main_experiments.py --seed ${seed} --environment_string ${env} --training_timesteps ${training} --num_concepts_selected ${num_concepts_selected} --selection_function q_value --out_folder basic --concept_source human_selected_binary --run_basic >> ${LOGFILE} 2>&1"  ENTER 

    # # Continuous 
    # for num_concepts_selected in 4 # 1 2 3 4 
    # do 
    #     for env in cart_pole pong boxing mimic
    #     do 
    #         training=${training_timesteps[$env]}
    #         tmux send-keys -t concepts "conda activate ${environment}; python main_experiments.py --seed ${seed} --environment_string ${env} --training_timesteps ${training} --num_concepts_selected ${num_concepts_selected} --selection_function q_value --out_folder basic --concept_source human_selected --run_basic  >> ${LOGFILE} 2>&1"  ENTER 
    #     done 
    # done 
done 
