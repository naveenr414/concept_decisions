#!/bin/bash 
: > runs/logs/error_intervention.txt
LOGFILE=../../runs/logs/error_intervention.txt

# : > runs/logs/error_glucose.txt
# LOGFILE_GLUCOSE=../../runs/logs/error_glucose.txt


environment=food
tmux new-session -d -s concept_intervention
tmux send-keys -t concept_intervention ENTER 
tmux send-keys -t concept_intervention "source ~/.bashrc" ENTER
tmux send-keys -t concept_intervention "cd scripts/notebooks" ENTER
tmux send-keys -t concept_intervention "export PYTHONWARNINGS='ignore'" ENTER
tmux send-keys -t concept_intervention "export GYMNASIUM_DISABLE_WARNINGS=1" ENTER

# tmux new-session -d -s concepts_glucose
# tmux send-keys -t concepts_glucose ENTER 
# tmux send-keys -t concepts_glucose "source ~/.bashrc" ENTER
# tmux send-keys -t concepts_glucose "cd scripts/notebooks" ENTER
# tmux send-keys -t concepts_glucose "export PYTHONWARNINGS='ignore'" ENTER
# tmux send-keys -t concepts_glucose "export GYMNASIUM_DISABLE_WARNINGS=1" ENTER

declare -A training_timesteps=(
  [cyclic_4]=10000
  [cyclic_16]=10000
  [tree_7]=25000
  [tree_31]=25000
  [cart_pole]=1000000
  [mini_grid]=250000
  [glucose]=250000
  [pong]=4000000
  [boxing]=4000000
)


for seed in 42 #43 44
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

    # env=cart_pole
    # for num_concepts_selected in 10 # 5 10 15
    # do 
    #     training=${training_timesteps[$env]}
    #     for intervention_probability in 0.25 0.5 0.75
    #     do 
    #         for fixed_accuracy in 0.9
    #         do 
    #             cbm_accuracy=$(printf "%s " $(yes "$fixed_accuracy" | head -n 24))
    #             human_acc=1
    #             intervention_accuracy=$(printf "%s " $(yes "$human_acc" | head -n 24))
    #             tmux send-keys -t concept_intervention "conda activate ${environment}; python main_experiments.py --seed ${seed} --environment_string ${env} --training_timesteps ${training} --num_concepts_selected ${num_concepts_selected} --selection_function q_value --out_folder intervention --gold_timesteps 4000000 --run_intervention --concept_source human_selected_binary --cbm_accuracy_by_concept ${cbm_accuracy} --intervention_accuracy ${intervention_accuracy} --intervention_probability ${intervention_probability} >> ${LOGFILE} 2>&1"  ENTER 
    #         done 
    #     done 
    # done 

    # env=mini_grid
    # for num_concepts_selected in 20
    # do 
    #     training=${training_timesteps[$env]}
    #     for intervention_probability in 0.25 0.5 0.75
    #     do 
    #         fixed_accuracy=0.9
    #         cbm_accuracy=$(printf "%s " $(yes "$fixed_accuracy" | head -n 44))
    #         human_acc=1
    #         intervention_accuracy=$(printf "%s " $(yes "$human_acc" | head -n 44))
    #         tmux send-keys -t concept_intervention "conda activate ${environment}; python main_experiments.py --seed ${seed} --environment_string ${env} --training_timesteps ${training} --num_concepts_selected ${num_concepts_selected} --selection_function q_value --out_folder intervention --gold_timesteps 4000000 --run_intervention --concept_source human_selected_binary --cbm_accuracy_by_concept ${cbm_accuracy} --intervention_accuracy ${intervention_accuracy} --intervention_probability ${intervention_probability} >> ${LOGFILE} 2>&1"  ENTER 
    #     done 
    # done 

    # Run different concepts get high/low accuracy (Figure 8)
    env=mini_grid
    for num_concepts_selected in 20
    do 
        training=${training_timesteps[$env]}
        intervention_probability=0.5
        fixed_accuracy=0.9
        cbm_accuracy=$(printf "%s " $(yes "$fixed_accuracy" | head -n 44))
        block_sizes=(5 5 4 5 5 5 5 2 2 2 2 2)
        human_acc_values=(0.5 0.75 0.9)
        total=44
        combo_id=0
        
        for ((acc_idx=0; acc_idx<${#human_acc_values[@]}; acc_idx++)); do
            human_acc=${human_acc_values[$acc_idx]}
            for ((block_idx=0; block_idx<${#block_sizes[@]}; block_idx++)); do
                values=()
                start=0
                for ((i=0; i<${#block_sizes[@]}; i++)); do
                    size=${block_sizes[i]}
                    # If this is the selected block, use human_acc, otherwise use 1.0
                    if [ $i -eq $block_idx ]; then
                        value=$human_acc
                    else
                        value=1.0
                    fi
                    for ((j=0; j<size; j++)); do
                        idx=$((start + j))
                        values[idx]=$value
                    done
                    start=$((start + size))
                done
                # Convert to space-separated string
                intervention_accuracy=$(printf "%s " "${values[@]}")
                echo "Running combo ${combo_id}: Block $((block_idx+1)) gets ${human_acc}, others get 1.0"
                combo_id=$((combo_id+1))
                # Send the job to tmux
                tmux send-keys -t concept_intervention \
                    "conda activate ${environment}; \
                    python -u main_experiments.py \
                    --seed ${seed} \
                    --environment_string ${env} \
                    --training_timesteps ${training} \
                    --num_concepts_selected ${num_concepts_selected} \
                    --selection_function q_value \
                    --out_folder intervention \
                    --gold_timesteps 4000000 \
                    --run_intervention \
                    --concept_source human_selected_binary \
                    --cbm_accuracy_by_concept ${cbm_accuracy} \
                    --intervention_accuracy ${intervention_accuracy} \
                    --intervention_probability ${intervention_probability} \
                    >> ${LOGFILE} 2>&1" ENTER
            done 
        done
    done 

    # env=glucose
    # for num_concepts_selected in 4 # 20 40 60 80
    # do 
    #     training=${training_timesteps[$env]}
    #     for intervention_probability in 0.25 0.5 0.75
    #     do 
    #         fixed_accuracy=0.75
    #         cbm_accuracy=$(printf "%s " $(yes "$fixed_accuracy" | head -n 13))
    #         human_acc=1
    #         intervention_accuracy=$(printf "%s " $(yes "$human_acc" | head -n 13))
    #         tmux send-keys -t concepts_glucose "conda activate ${environment}; python main_experiments.py --seed ${seed} --environment_string ${env} --training_timesteps ${training} --gold_timesteps 250000 --num_concepts_selected ${num_concepts_selected} --selection_function q_value --out_folder intervention --run_intervention --concept_source human_selected_binary --cbm_accuracy_by_concept ${cbm_accuracy} --intervention_accuracy ${intervention_accuracy} --intervention_probability ${intervention_probability} >> ${LOGFILE_GLUCOSE} 2>&1"  ENTER 
    #     done 
    # done 

    # env=pong
    # for num_concepts_selected in 80 # 20 40 60 80
    # do 
    #     training=${training_timesteps[$env]}
    #     for intervention_probability in 0.25 0.5 0.75
    #     do 
    #         for fixed_accuracy in 0.9
    #         do 
    #             cbm_accuracy=$(printf "%s " $(yes "$fixed_accuracy" | head -n 228))
    #             human_acc=1
    #             intervention_accuracy=$(printf "%s " $(yes "$human_acc" | head -n 228))
    #             tmux send-keys -t concept_intervention "conda activate ${environment}; python main_experiments.py --seed ${seed} --environment_string ${env} --training_timesteps ${training} --gold_timesteps 4000000 --num_concepts_selected ${num_concepts_selected} --selection_function q_value --out_folder intervention --run_intervention --concept_source human_selected_binary --cbm_accuracy_by_concept ${cbm_accuracy} --intervention_accuracy ${intervention_accuracy} --intervention_probability ${intervention_probability} >> ${LOGFILE} 2>&1"  ENTER 
    #         done 
    #     done 
    # done 

    # env=boxing
    # for num_concepts_selected in 80 # 20 40 60 80
    # do 
    #     training=${training_timesteps[$env]}
    #     for intervention_probability in 0.25 0.5 0.75
    #     do 
    #         for fixed_accuracy in 0.9
    #         do 
    #             cbm_accuracy=$(printf "%s " $(yes "$fixed_accuracy" | head -n 190))
    #             human_acc=1
    #             intervention_accuracy=$(printf "%s " $(yes "$human_acc" | head -n 190))
    #             tmux send-keys -t concept_intervention "conda activate ${environment}; python main_experiments.py --seed ${seed} --gold_timesteps 10000000 --environment_string ${env} --training_timesteps ${training} --num_concepts_selected ${num_concepts_selected} --selection_function q_value --out_folder intervention  --run_intervention --concept_source human_selected_binary --cbm_accuracy_by_concept ${cbm_accuracy} --intervention_accuracy ${intervention_accuracy} --intervention_probability ${intervention_probability} >> ${LOGFILE} 2>&1"  ENTER 
    #         done 
    #     done 
    # done 
done 
