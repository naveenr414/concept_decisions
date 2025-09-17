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
    for num_concepts_selected in 1 # 2 3
    do 
        training=${training_timesteps[$env]}
        for fixed_accuracy in 0.5 # 0.75 0.9
        do 
            cbm_accuracy=$(printf "%s " $(yes "$fixed_accuracy" | head -n 3))
            intervention_accuracy=($(for a in "${cbm_accuracy[@]}"; do
                awk -v val="$a" 'BEGIN{printf "%.2f ", val+0.1}'
            done))            
            intervention_accuracy=$(printf "%s " $(yes "$intervention_accuracy" | head -n 3))

            tmux send-keys -t concept_intervention "conda activate ${environment}; python main_experiments.py --seed ${seed} --environment_string ${env} --training_timesteps ${training} --num_concepts_selected ${num_concepts_selected} --selection_function q_value --out_folder intervention --run_intervention --concept_source human_selected_binary --cbm_accuracy_by_concept ${cbm_accuracy} --intervention_accuracy ${intervention_accuracy} --intervention_probability 0.5 >> ${LOGFILE} 2>&1"  ENTER 
        done 
    done 

    # env=cyclic_16
    # for num_concepts_selected in 1 2 4
    # do 
    #     training=${training_timesteps[$env]}
    #     for fixed_accuracy in 0.5 0.75 0.9
    #     do 
    #         cbm_accuracy=$(printf "%s " $(yes "$fixed_accuracy" | head -n 15))
    #         intervention_accuracy=($(for a in "${cbm_accuracy[@]}"; do
    #             awk -v val="$a" 'BEGIN{printf "%.2f ", val+0.1}'
    #         done))            
    #         intervention_accuracy=$(printf "%s " $(yes "$intervention_accuracy" | head -n 15))

    #         tmux send-keys -t concept_intervention "conda activate ${environment}; python main_experiments.py --seed ${seed} --environment_string ${env} --training_timesteps ${training} --num_concepts_selected ${num_concepts_selected} --selection_function q_value --out_folder intervention --run_intervention --concept_source human_selected_binary --cbm_accuracy_by_concept ${cbm_accuracy} --intervention_accuracy ${intervention_accuracy} --intervention_probability 0.5 >> ${LOGFILE} 2>&1"  ENTER 
    #     done 
    # done 

    # env=tree_7
    # for num_concepts_selected in 1 2 3
    # do 
    #     training=${training_timesteps[$env]}
    #     for fixed_accuracy in 0.5 0.75 0.9
    #     do 
    #         cbm_accuracy=$(printf "%s " $(yes "$fixed_accuracy" | head -n 4))
    #         intervention_accuracy=($(for a in "${cbm_accuracy[@]}"; do
    #             awk -v val="$a" 'BEGIN{printf "%.2f ", val+0.1}'
    #         done))     
    #         intervention_accuracy=$(printf "%s " $(yes "$intervention_accuracy" | head -n 4))
       
    #         tmux send-keys -t concept_intervention "conda activate ${environment}; python main_experiments.py --seed ${seed} --environment_string ${env} --training_timesteps ${training} --num_concepts_selected ${num_concepts_selected} --selection_function q_value --out_folder intervention --run_intervention --concept_source human_selected_binary --cbm_accuracy_by_concept ${cbm_accuracy} --intervention_accuracy ${intervention_accuracy} --intervention_probability 0.5 >> ${LOGFILE} 2>&1"  ENTER 
    #     done 
    # done 

    # env=tree_15
    # for num_concepts_selected in 1 2 3
    # do 
    #     training=${training_timesteps[$env]}
    #     for fixed_accuracy in 0.5 0.75 0.9
    #     do 
    #         cbm_accuracy=$(printf "%s " $(yes "$fixed_accuracy" | head -n 5))
    #         intervention_accuracy=($(for a in "${cbm_accuracy[@]}"; do
    #             awk -v val="$a" 'BEGIN{printf "%.2f ", val+0.1}'
    #         done))     
    #         intervention_accuracy=$(printf "%s " $(yes "$intervention_accuracy" | head -n 5))
       
    #         tmux send-keys -t concept_intervention "conda activate ${environment}; python main_experiments.py --seed ${seed} --environment_string ${env} --training_timesteps ${training} --num_concepts_selected ${num_concepts_selected} --selection_function q_value --out_folder intervention --run_intervention --concept_source human_selected_binary --cbm_accuracy_by_concept ${cbm_accuracy} --intervention_accuracy ${intervention_accuracy} --intervention_probability 0.5 >> ${LOGFILE} 2>&1"  ENTER 
    #     done 
    # done 
done 
