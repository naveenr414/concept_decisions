#!/bin/bash 
: > runs/logs/error_intervention.txt
LOGFILE=../../runs/logs/error_intervention.txt

environment=food
tmux new-session -d -s concept_intervention
tmux send-keys -t concept_intervention ENTER 
tmux send-keys -t concept_intervention "source ~/.bashrc" ENTER
tmux send-keys -t concept_intervention "cd scripts/notebooks" ENTER
tmux send-keys -t concept_intervention "export PYTHONWARNINGS='ignore'" ENTER
tmux send-keys -t concept_intervention "export GYMNASIUM_DISABLE_WARNINGS=1" ENTER

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
    # num_concepts_selected=2
    # training=${training_timesteps[$env]}
    # intervention_accuracy="1 1 1"  
    # cbm_accuracy="0.5 0.75 0.75"
    # for intervention_probability in 0.25 0.5 0.75
    # do 
    #     tmux send-keys -t concept_intervention "conda activate ${environment}; python main_experiments.py --seed ${seed} --environment_string ${env} --training_timesteps ${training} --num_concepts_selected ${num_concepts_selected} --selection_function q_value --out_folder intervention --run_intervention --concept_source human_selected_binary --cbm_accuracy_by_concept ${cbm_accuracy} --intervention_accuracy ${intervention_accuracy} --intervention_probability ${intervention_probability} >> ${LOGFILE} 2>&1"  ENTER 
    # done 
    # cbm_accuracy="0.5 0.95 0.95"
    # for intervention_probability in 0.25 0.5 0.75
    # do 
    #     tmux send-keys -t concept_intervention "conda activate ${environment}; python main_experiments.py --seed ${seed} --environment_string ${env} --training_timesteps ${training} --num_concepts_selected ${num_concepts_selected} --selection_function q_value --out_folder intervention --run_intervention --concept_source human_selected_binary --cbm_accuracy_by_concept ${cbm_accuracy} --intervention_accuracy ${intervention_accuracy} --intervention_probability ${intervention_probability} >> ${LOGFILE} 2>&1"  ENTER 
    # done 

    # env=cyclic_16
    # num_concepts_selected=2
    # training=${training_timesteps[$env]}
    # cbm_accuracy="0.5 0.75 0.75 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5"
    # intervention_accuracy="1 1 1 1 1 1 1 1 1 1 1 1 1 1 1"
    # for intervention_probability in 0.25 0.5 0.75
    # do 
    #     tmux send-keys -t concept_intervention "conda activate ${environment}; python main_experiments.py --seed ${seed} --environment_string ${env} --training_timesteps ${training} --num_concepts_selected ${num_concepts_selected} --selection_function q_value --out_folder intervention --run_intervention --concept_source human_selected_binary --cbm_accuracy_by_concept ${cbm_accuracy} --intervention_accuracy ${intervention_accuracy} --intervention_probability ${intervention_probability} >> ${LOGFILE} 2>&1"  ENTER 
    # done 
    # cbm_accuracy="0.5 0.95 0.95 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5"
    # for intervention_probability in 0.25 0.5 0.75
    # do 
    #     tmux send-keys -t concept_intervention "conda activate ${environment}; python main_experiments.py --seed ${seed} --environment_string ${env} --training_timesteps ${training} --num_concepts_selected ${num_concepts_selected} --selection_function q_value --out_folder intervention --run_intervention --concept_source human_selected_binary --cbm_accuracy_by_concept ${cbm_accuracy} --intervention_accuracy ${intervention_accuracy} --intervention_probability ${intervention_probability} >> ${LOGFILE} 2>&1"  ENTER 
    # done 

    # env=tree_7
    # num_concepts_selected=2
    # training=${training_timesteps[$env]}
    # cbm_accuracy="0.75 0.75 0.75 0.5"
    # intervention_accuracy="1 1 1 1"
    # for intervention_probability in 0.25 0.5 0.75
    # do 
    #     tmux send-keys -t concept_intervention "conda activate ${environment}; python main_experiments.py --seed ${seed} --environment_string ${env} --training_timesteps ${training} --num_concepts_selected ${num_concepts_selected} --selection_function q_value --out_folder intervention --run_intervention --concept_source human_selected_binary --cbm_accuracy_by_concept ${cbm_accuracy} --intervention_accuracy ${intervention_accuracy} --intervention_probability ${intervention_probability} >> ${LOGFILE} 2>&1"  ENTER 
    # done 
    # cbm_accuracy="0.95 0.95 0.95 0.5"
    # for intervention_probability in 0.25 0.5 0.75
    # do 
    #     tmux send-keys -t concept_intervention "conda activate ${environment}; python main_experiments.py --seed ${seed} --environment_string ${env} --training_timesteps ${training} --num_concepts_selected ${num_concepts_selected} --selection_function q_value --out_folder intervention --run_intervention --concept_source human_selected_binary --cbm_accuracy_by_concept ${cbm_accuracy} --intervention_accuracy ${intervention_accuracy} --intervention_probability ${intervention_probability} >> ${LOGFILE} 2>&1"  ENTER 
    # done 

    # env=tree_31
    # num_concepts_selected=2
    # training=${training_timesteps[$env]}
    # cbm_accuracy="0.75 0.75 0.75 0.75 0.75 0.5"
    # intervention_accuracy="1 1 1 1 1 1"
    # for intervention_probability in 0.25 0.5 0.75
    # do 
    #     tmux send-keys -t concept_intervention "conda activate ${environment}; python main_experiments.py --seed ${seed} --environment_string ${env} --training_timesteps ${training} --num_concepts_selected ${num_concepts_selected} --selection_function q_value --out_folder intervention --run_intervention --concept_source human_selected_binary --cbm_accuracy_by_concept ${cbm_accuracy} --intervention_accuracy ${intervention_accuracy} --intervention_probability ${intervention_probability} >> ${LOGFILE} 2>&1"  ENTER 
    # done 
    # cbm_accuracy="0.95 0.95 0.95 0.95 0.95 0.5"
    # for intervention_probability in 0.25 0.5 0.75
    # do 
    #     tmux send-keys -t concept_intervention "conda activate ${environment}; python main_experiments.py --seed ${seed} --environment_string ${env} --training_timesteps ${training} --num_concepts_selected ${num_concepts_selected} --selection_function q_value --out_folder intervention --run_intervention --concept_source human_selected_binary --cbm_accuracy_by_concept ${cbm_accuracy} --intervention_accuracy ${intervention_accuracy} --intervention_probability ${intervention_probability} >> ${LOGFILE} 2>&1"  ENTER 
    # done 


    env=cart_pole
    for num_concepts_selected in 5 10 15
    do 
        training=${training_timesteps[$env]}
        for intervention_probability in 0.25 0.5 0.75
        do 
            fixed_accuracy=0.75
            cbm_accuracy=$(printf "%s " $(yes "$fixed_accuracy" | head -n 24))
            human_acc=1
            intervention_accuracy=$(printf "%s " $(yes "$human_acc" | head -n 24))
            tmux send-keys -t concept_intervention "conda activate ${environment}; python main_experiments.py --seed ${seed} --environment_string ${env} --training_timesteps ${training} --num_concepts_selected ${num_concepts_selected} --selection_function q_value --out_folder intervention --gold_timesteps 4000000 --run_intervention --concept_source human_selected_binary --cbm_accuracy_by_concept ${cbm_accuracy} --intervention_accuracy ${intervention_accuracy} --intervention_probability ${intervention_probability} >> ${LOGFILE} 2>&1"  ENTER 
        done 
    done 

    env=mini_grid
    for num_concepts_selected in 20 30 40
    do 
        training=${training_timesteps[$env]}
        for intervention_probability in 0.25 0.5 0.75
        do 
            fixed_accuracy=0.9
            cbm_accuracy=$(printf "%s " $(yes "$fixed_accuracy" | head -n 44))
            human_acc=1
            intervention_accuracy=$(printf "%s " $(yes "$human_acc" | head -n 44))
            tmux send-keys -t concept_intervention "conda activate ${environment}; python main_experiments.py --seed ${seed} --environment_string ${env} --training_timesteps ${training} --num_concepts_selected ${num_concepts_selected} --selection_function q_value --out_folder intervention --gold_timesteps 4000000 --run_intervention --concept_source human_selected_binary --cbm_accuracy_by_concept ${cbm_accuracy} --intervention_accuracy ${intervention_accuracy} --intervention_probability ${intervention_probability} >> ${LOGFILE} 2>&1"  ENTER 
        done 
    done 

    env=mimic
    for num_concepts_selected in 20 40 60 80
    do 
        training=${training_timesteps[$env]}
        for intervention_probability in 0.25 0.5 0.75
        do 
            fixed_accuracy=0.75
            cbm_accuracy=$(printf "%s " $(yes "$fixed_accuracy" | head -n 141))
            human_acc=1
            intervention_accuracy=$(printf "%s " $(yes "$human_acc" | head -n 141))
            tmux send-keys -t concept_intervention "conda activate ${environment}; python main_experiments.py --seed ${seed} --environment_string ${env} --training_timesteps ${training} --num_concepts_selected ${num_concepts_selected} --selection_function q_value --out_folder intervention --gold_timesteps 250000 --run_intervention --concept_source human_selected_binary --cbm_accuracy_by_concept ${cbm_accuracy} --intervention_accuracy ${intervention_accuracy} --intervention_probability ${intervention_probability} >> ${LOGFILE} 2>&1"  ENTER 
        done 
    done 
done 
