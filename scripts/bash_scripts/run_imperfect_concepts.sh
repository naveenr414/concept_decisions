#!/bin/bash 
: > runs/logs/error_imperfect.txt
LOGFILE=../../runs/logs/error_imperfect.txt

environment=food
tmux new-session -d -s concept_imperfect
tmux send-keys -t concept_imperfect ENTER 
tmux send-keys -t concept_imperfect "source ~/.bashrc" ENTER
tmux send-keys -t concept_imperfect "cd scripts/notebooks" ENTER
tmux send-keys -t concept_imperfect "export PYTHONWARNINGS='ignore'" ENTER
tmux send-keys -t concept_imperfect "export GYMNASIUM_DISABLE_WARNINGS=1" ENTER

declare -A training_timesteps=(
  [cyclic_4]=10000
  [cyclic_16]=10000
  [tree_7]=10000
  [tree_31]=10000
  [cart_pole]=250000
  [mini_grid]=12000000
  [pong]=20000000
  [boxing]=12000000
  [mimic]=250000
)


for seed in 42
do 
    env=cyclic_4
    for num_concepts_selected in 1 # 1 2 3
    do 
        training=${training_timesteps[$env]}
        for fixed_accuracy in 0.5 # 0.75 0.9
        do 
            cbm_accuracy=$(printf "%s " $(yes "$fixed_accuracy" | head -n 3))
            tmux send-keys -t concept_imperfect "conda activate ${environment}; python main_experiments.py --seed ${seed} --environment_string ${env} --training_timesteps ${training} --num_concepts_selected ${num_concepts_selected} --selection_function q_value --out_folder imperfect --run_imperfect --concept_source human_selected_binary --cbm_accuracy_by_concept ${cbm_accuracy} >> ${LOGFILE} 2>&1"  ENTER 
        done 
    done 

    # env=cyclic_16
    # for num_concepts_selected in 1 2 4
    # do 
    #     training=${training_timesteps[$env]}
    #     for fixed_accuracy in 0.5 0.75 0.9
    #     do 
    #         cbm_accuracy=$(printf "%s " $(yes "$fixed_accuracy" | head -n 15))
    #         tmux send-keys -t concept_imperfect "conda activate ${environment}; python main_experiments.py --seed ${seed} --environment_string ${env} --training_timesteps ${training} --num_concepts_selected ${num_concepts_selected} --selection_function q_value --out_folder imperfect --run_imperfect --concept_source human_selected_binary --cbm_accuracy_by_concept ${cbm_accuracy} >> ${LOGFILE} 2>&1"  ENTER 
    #     done 
    # done 

    # env=tree_7
    # for num_concepts_selected in 1 2 3
    # do 
    #     training=${training_timesteps[$env]}
    #     for fixed_accuracy in 0.5 0.75 0.9
    #     do 
    #         cbm_accuracy=$(printf "%s " $(yes "$fixed_accuracy" | head -n 4))
    #         tmux send-keys -t concept_imperfect "conda activate ${environment}; python main_experiments.py --seed ${seed} --environment_string ${env} --training_timesteps ${training} --num_concepts_selected ${num_concepts_selected} --selection_function q_value --out_folder imperfect --run_imperfect --concept_source human_selected_binary --cbm_accuracy_by_concept ${cbm_accuracy} >> ${LOGFILE} 2>&1"  ENTER 
    #     done 
    # done 

    # env=tree_15
    # for num_concepts_selected in 1 2 3
    # do 
    #     training=${training_timesteps[$env]}
    #     for fixed_accuracy in 0.5 0.75 0.9
    #     do 
    #         cbm_accuracy=$(printf "%s " $(yes "$fixed_accuracy" | head -n 5))
    #         tmux send-keys -t concept_imperfect "conda activate ${environment}; python main_experiments.py --seed ${seed} --environment_string ${env} --training_timesteps ${training} --num_concepts_selected ${num_concepts_selected} --selection_function q_value --out_folder imperfect --run_imperfect --concept_source human_selected_binary --cbm_accuracy_by_concept ${cbm_accuracy} >> ${LOGFILE} 2>&1"  ENTER 
    #     done 
    # done 
done 
