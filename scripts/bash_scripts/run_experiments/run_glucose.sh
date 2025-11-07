#!/bin/bash 
: > runs/logs/error_glucose.txt
LOGFILE_GLUCOSE=../../runs/logs/error_glucose.txt


environment=food
tmux new-session -d -s concepts_glucose
tmux send-keys -t concepts_glucose ENTER 
tmux send-keys -t concepts_glucose "source ~/.bashrc" ENTER
tmux send-keys -t concepts_glucose "cd scripts/notebooks" ENTER
tmux send-keys -t concepts_glucose "export PYTHONWARNINGS='ignore'" ENTER
tmux send-keys -t concepts_glucose "export GYMNASIUM_DISABLE_WARNINGS=1" ENTER

declare -A training_timesteps=(
  [glucose]=250000
)


for seed in 42 #43 44
do 
    env=glucose
    training=${training_timesteps[$env]}
    for num_concepts_selected in 4 6 8
    do 
        tmux send-keys -t concepts_glucose "conda activate ${environment}; python main_experiments_fast.py --seed ${seed} --environment_string ${env} --gold_timesteps 250000 --training_timesteps ${training} --num_concepts_selected ${num_concepts_selected} --selection_function q_value --out_folder glucose --concept_source human_selected_binary --run_basic >> ${LOGFILE_GLUCOSE} 2>&1"  ENTER 
    done 

    env=glucose
    for num_concepts_selected in 4 6 8 # 20 40 60 80
    do 
        training=${training_timesteps[$env]}
        for fixed_accuracy in 0.5 0.75 0.9
        do 
            cbm_accuracy=$(printf "%s " $(yes "$fixed_accuracy" | head -n 13))
            tmux send-keys -t concepts_glucose "conda activate ${environment}; python -u main_experiments_fast.py --seed ${seed} --environment_string ${env} --gold_timesteps 250000 --training_timesteps ${training} --num_concepts_selected ${num_concepts_selected} --selection_function q_value --out_folder glucose --run_imperfect --concept_source human_selected_binary --cbm_accuracy_by_concept ${cbm_accuracy} >> ${LOGFILE_GLUCOSE} 2>&1"  ENTER 
        done 
    done 

    intervention_probability=0.25
    for num_concepts_selected in 4 6 8 # 20 40 60 80
    do 
        training=${training_timesteps[$env]}
        for fixed_accuracy in 0.5 0.75 0.9
        do
            cbm_accuracy=$(printf "%s " $(yes "$fixed_accuracy" | head -n 13))
            human_acc=1
            intervention_accuracy=$(printf "%s " $(yes "$human_acc" | head -n 13))
            tmux send-keys -t concepts_glucose "conda activate ${environment}; python main_experiments_fast.py --seed ${seed} --environment_string ${env} --training_timesteps ${training} --gold_timesteps 250000 --num_concepts_selected ${num_concepts_selected} --selection_function q_value --out_folder glucose --run_intervention --concept_source human_selected_binary --cbm_accuracy_by_concept ${cbm_accuracy} --intervention_accuracy ${intervention_accuracy} --intervention_probability ${intervention_probability} >> ${LOGFILE_GLUCOSE} 2>&1"  ENTER 
        done 
    done 
done 
