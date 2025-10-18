#!/bin/bash 
: > runs/logs/error_imperfect.txt
LOGFILE=../../runs/logs/error_imperfect.txt

: > runs/logs/error_glucose.txt
LOGFILE_GLUCOSE=../../runs/logs/error_glucose.txt

environment=food
tmux new-session -d -s concept_imperfect
tmux send-keys -t concept_imperfect ENTER 
tmux send-keys -t concept_imperfect "source ~/.bashrc" ENTER
tmux send-keys -t concept_imperfect "cd scripts/notebooks" ENTER
tmux send-keys -t concept_imperfect "export PYTHONWARNINGS='ignore'" ENTER
tmux send-keys -t concept_imperfect "export GYMNASIUM_DISABLE_WARNINGS=1" ENTER

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
  [glucose]=1 #250000
  [pong]=1 #4000000
  [boxing]=4000000
)


for seed in 43
do 
    # env=cyclic_4
    # num_concepts_selected=2
    # training=${training_timesteps[$env]}
    # for fixed_accuracy in 0.5 0.75 0.9
    # do 
    #     cbm_accuracy=$(printf "%s " $(yes "$fixed_accuracy" | head -n 3))
    #     tmux send-keys -t concept_imperfect "conda activate ${environment}; python main_experiments.py --seed ${seed} --environment_string ${env} --training_timesteps ${training} --num_concepts_selected ${num_concepts_selected} --selection_function q_value --out_folder imperfect --run_imperfect --concept_source human_selected_binary --cbm_accuracy_by_concept ${cbm_accuracy} >> ${LOGFILE} 2>&1"  ENTER 
    # done 
    # fixed_accuracy=0.75
    # for num_concepts_selected in 1 2 3
    # do 
    #     cbm_accuracy=$(printf "%s " $(yes "$fixed_accuracy" | head -n 3))
    #     tmux send-keys -t concept_imperfect "conda activate ${environment}; python main_experiments.py --seed ${seed} --environment_string ${env} --training_timesteps ${training} --num_concepts_selected ${num_concepts_selected} --selection_function q_value --out_folder imperfect --run_imperfect --concept_source human_selected_binary --cbm_accuracy_by_concept ${cbm_accuracy} >> ${LOGFILE} 2>&1"  ENTER 
    # done 

    # env=cyclic_16
    # num_concepts_selected=2
    # training=${training_timesteps[$env]}
    # for fixed_accuracy in 0.5 0.75 0.9
    # do 
    #     cbm_accuracy=$(printf "%s " $(yes "$fixed_accuracy" | head -n 15))
    #     tmux send-keys -t concept_imperfect "conda activate ${environment}; python main_experiments.py --seed ${seed} --environment_string ${env} --training_timesteps ${training} --num_concepts_selected ${num_concepts_selected} --selection_function q_value --out_folder imperfect --run_imperfect --concept_source human_selected_binary --cbm_accuracy_by_concept ${cbm_accuracy} >> ${LOGFILE} 2>&1"  ENTER 
    # done 
    # fixed_accuracy=0.75
    # for num_concepts_selected in 1 2 3 4
    # do 
    #     cbm_accuracy=$(printf "%s " $(yes "$fixed_accuracy" | head -n 15))
    #     tmux send-keys -t concept_imperfect "conda activate ${environment}; python main_experiments.py --seed ${seed} --environment_string ${env} --training_timesteps ${training} --num_concepts_selected ${num_concepts_selected} --selection_function q_value --out_folder imperfect --run_imperfect --concept_source human_selected_binary --cbm_accuracy_by_concept ${cbm_accuracy} >> ${LOGFILE} 2>&1"  ENTER 
    # done 

    # env=tree_7
    # num_concepts_selected=2
    # training=${training_timesteps[$env]}
    # for fixed_accuracy in 0.5 0.75 0.9
    # do 
    #     cbm_accuracy=$(printf "%s " $(yes "$fixed_accuracy" | head -n 4))
    #     tmux send-keys -t concept_imperfect "conda activate ${environment}; python main_experiments.py --seed ${seed} --environment_string ${env} --training_timesteps ${training} --num_concepts_selected ${num_concepts_selected} --selection_function q_value --out_folder imperfect --run_imperfect --concept_source human_selected_binary --cbm_accuracy_by_concept ${cbm_accuracy} >> ${LOGFILE} 2>&1"  ENTER 
    # done 
    # fixed_accuracy=0.95
    # for num_concepts_selected in 1 2 3 4
    # do 
    #     cbm_accuracy=$(printf "%s " $(yes "$fixed_accuracy" | head -n 4))
    #     tmux send-keys -t concept_imperfect "conda activate ${environment}; python main_experiments.py --seed ${seed} --environment_string ${env} --training_timesteps ${training} --num_concepts_selected ${num_concepts_selected} --selection_function q_value --out_folder imperfect --run_imperfect --concept_source human_selected_binary --cbm_accuracy_by_concept ${cbm_accuracy} >> ${LOGFILE} 2>&1"  ENTER 
    # done 

    # env=tree_31
    # num_concepts_selected=2
    # training=${training_timesteps[$env]}
    # for fixed_accuracy in 0.95 # 0.5 0.75 0.9
    # do 
    #     cbm_accuracy=$(printf "%s " $(yes "$fixed_accuracy" | head -n 6))
    #     tmux send-keys -t concept_imperfect "conda activate ${environment}; python main_experiments.py --seed ${seed} --environment_string ${env} --training_timesteps ${training} --num_concepts_selected ${num_concepts_selected} --selection_function q_value --out_folder imperfect --run_imperfect --concept_source human_selected_binary --cbm_accuracy_by_concept ${cbm_accuracy} >> ${LOGFILE} 2>&1"  ENTER 
    # done 
    # fixed_accuracy=0.95
    # for num_concepts_selected in 1 2 3 4
    # do 
    #     cbm_accuracy=$(printf "%s " $(yes "$fixed_accuracy" | head -n 6))
    #     tmux send-keys -t concept_imperfect "conda activate ${environment}; python main_experiments.py --seed ${seed} --environment_string ${env} --training_timesteps ${training} --num_concepts_selected ${num_concepts_selected} --selection_function q_value --out_folder imperfect --run_imperfect --concept_source human_selected_binary --cbm_accuracy_by_concept ${cbm_accuracy} >> ${LOGFILE} 2>&1"  ENTER 
    # done 

    # env=cart_pole
    # for num_concepts_selected in 15 # 5 10 15
    # do 
    #     training=${training_timesteps[$env]}
    #     for fixed_accuracy in 0.5 0.75 0.9
    #     do 
    #         cbm_accuracy=$(printf "%s " $(yes "$fixed_accuracy" | head -n 24))
    #         tmux send-keys -t concept_imperfect "conda activate ${environment}; python main_experiments.py --seed ${seed} --environment_string ${env} --gold_timesteps 4000000 --training_timesteps ${training} --num_concepts_selected ${num_concepts_selected} --selection_function q_value --out_folder imperfect --run_imperfect --concept_source human_selected_binary --cbm_accuracy_by_concept ${cbm_accuracy} >> ${LOGFILE} 2>&1"  ENTER 
    #     done 
    # done 

    # env=mini_grid
    # for num_concepts_selected in 40 #20 30 40
    # do 
    #     training=${training_timesteps[$env]}
    #     for fixed_accuracy in 0.5 0.75 0.9
    #     do 
    #         cbm_accuracy=$(printf "%s " $(yes "$fixed_accuracy" | head -n 44))
    #         tmux send-keys -t concept_imperfect "conda activate ${environment}; python main_experiments.py --seed ${seed} --environment_string ${env} --gold_timesteps 4000000  --training_timesteps ${training} --num_concepts_selected ${num_concepts_selected} --selection_function q_value --out_folder imperfect --run_imperfect --concept_source human_selected_binary --cbm_accuracy_by_concept ${cbm_accuracy} >> ${LOGFILE} 2>&1"  ENTER 
    #     done 
    # done 

    # env=glucose
    # for num_concepts_selected in 4 # 20 40 60 80
    # do 
    #     training=${training_timesteps[$env]}
    #     for fixed_accuracy in 0.5 0.75 0.9
    #     do 
    #         cbm_accuracy=$(printf "%s " $(yes "$fixed_accuracy" | head -n 13))
    #         tmux send-keys -t concepts_glucose "conda activate ${environment}; python main_experiments.py --seed ${seed} --environment_string ${env} --gold_timesteps 250000 --training_timesteps ${training} --num_concepts_selected ${num_concepts_selected} --selection_function q_value --out_folder imperfect --run_imperfect --concept_source human_selected_binary --cbm_accuracy_by_concept ${cbm_accuracy} >> ${LOGFILE_GLUCOSE} 2>&1"  ENTER 
    #     done 
    # done 

    # env=pong
    # for num_concepts_selected in 80 # 20 40 60 80
    # do 
    #     training=${training_timesteps[$env]}
    #     for fixed_accuracy in 0.5 0.75 0.9
    #     do 
    #         cbm_accuracy=$(printf "%s " $(yes "$fixed_accuracy" | head -n 228))
    #         tmux send-keys -t concept_imperfect "conda activate ${environment}; python main_experiments.py --seed ${seed} --environment_string ${env} --gold_timesteps 4000000 --training_timesteps ${training} --num_concepts_selected ${num_concepts_selected} --selection_function q_value --out_folder imperfect --run_imperfect --concept_source human_selected_binary --cbm_accuracy_by_concept ${cbm_accuracy} >> ${LOGFILE} 2>&1"  ENTER 
    #     done 
    # done 

    env=boxing
    for num_concepts_selected in 80 # 20 40 60 80
    do 
        training=${training_timesteps[$env]}
        for fixed_accuracy in 0.9 # 0.5 0.75 0.9
        do 
            cbm_accuracy=$(printf "%s " $(yes "$fixed_accuracy" | head -n 190))
            tmux send-keys -t concept_imperfect "conda activate ${environment}; python main_experiments.py --seed ${seed} --environment_string ${env} --gold_timesteps 10000000 --training_timesteps ${training} --num_concepts_selected ${num_concepts_selected} --selection_function q_value --out_folder imperfect --run_imperfect --concept_source human_selected_binary --cbm_accuracy_by_concept ${cbm_accuracy} >> ${LOGFILE} 2>&1"  ENTER 
        done 
    done 
done 
