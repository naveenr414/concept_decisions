#!/bin/bash 
: > runs/logs/error_cart_pole.txt
LOGFILE=../../runs/logs/error_cart_pole.txt

environment=food
tmux new-session -d -s cart_pole
tmux send-keys -t cart_pole ENTER 
tmux send-keys -t cart_pole "cd scripts/notebooks" ENTER

for seed in 42
do 
    # Impact of Size
    env=cart_pole
    selected_function=policy
    # for training_timesteps in 5000 10000 15000 20000
    # do 
    #     tmux send-keys -t cart_pole "echo 'Running cart pole with training timesteps ${training_timesteps}' >> ${LOGFILE} 2>&1"  ENTER 
    #     tmux send-keys -t cart_pole "conda activate ${environment}; python real_world_experiments.py --seed ${seed} --environment_string ${env} --training_timesteps ${training_timesteps} --num_concepts_selected 4 --selection_function policy --out_folder cart_pole >> ${LOGFILE} 2>&1"  ENTER 
    # done 
    # env=cart_pole_binary
    # for training_timesteps in 5000 10000 15000 20000
    # do 
    #     tmux send-keys -t cart_pole "echo 'Running cart pole binary with training timesteps ${training_timesteps}' >> ${LOGFILE} 2>&1"  ENTER 
    #     tmux send-keys -t cart_pole "conda activate ${environment}; python real_world_experiments.py --seed ${seed} --environment_string ${env} --training_timesteps ${training_timesteps} --num_concepts_selected 4 --selection_function policy --out_folder cart_pole >> ${LOGFILE} 2>&1"  ENTER 
    # done 

    # # Impact of concept selection method
    training_timesteps=10000
    # for selection_function in policy q_value transition
    # do
    #     for env in cart_pole cart_pole_binary cart_pole_llm cart_pole_post_hoc
    #     do 
    #         tmux send-keys -t cart_pole "echo 'Running cart pole ${env} with selection function ${selected_function}' >> ${LOGFILE} 2>&1"  ENTER 
    #         tmux send-keys -t cart_pole "conda activate ${environment}; python real_world_experiments.py --seed ${seed} --environment_string ${env} --training_timesteps ${training_timesteps} --num_concepts_selected 4 --selection_function ${selection_function} --out_folder cart_pole >> ${LOGFILE} 2>&1"  ENTER 
    #     done 
    # done 

    env=cart_pole_binary
    selection_function=policy
    target_abstraction=0.05
    # # Error
    for fixed_accuracy in 0.1 0.25 0.5 0.75 0.9 0.99
    do 
        tmux send-keys -t cart_pole "echo 'Running cart pole with accuracy ${fixed_accuracy}' >> ${LOGFILE} 2>&1"  ENTER 
        cbm_accuracy=$(printf "%s " $(yes "$fixed_accuracy" | head -n 16))
        tmux send-keys -t cart_pole "conda activate ${environment}; python real_world_experiments.py --seed ${seed} --environment_string ${env} --training_timesteps ${training_timesteps} --num_concepts_selected 0 --target_abstraction ${target_abstraction} --selection_function ${selection_function} --cbm_accuracy_by_concept ${cbm_accuracy} --out_folder cart_pole >> ${LOGFILE} 2>&1"  ENTER 
    done 

    fixed_accuracy=0.75
    selection_function=q_value
    cbm_accuracy=$(printf "%s " $(yes "$fixed_accuracy" | head -n 16))
    for target_abstraction in 0.01 0.05 0.1 0.2
    do 
        tmux send-keys -t cart_pole "conda activate ${environment}; python real_world_experiments.py --seed ${seed} --environment_string ${env} --training_timesteps ${training_timesteps} --num_concepts_selected 0 --target_abstraction ${target_abstraction} --selection_function ${selection_function} --cbm_accuracy_by_concept ${cbm_accuracy} --out_folder cart_pole >> ${LOGFILE} 2>&1"  ENTER 
    done 

    selection_function=policy
    human_accuracy_by_concept=(0.9 0.9 0.9 0.9 0.9 0.9 0.9 0.9 0.9 0.9 0.9 0.9 0.9 0.9 0.9 0.9)
    human_accuracy_by_concept="${human_accuracy_by_concept[*]}"
    cbm_accuracy=(0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5)
    cbm_accuracy="${cbm_accuracy[*]}"
    target_abstraction=0.05
    for human_rel in 0.1 0.25 0.5 0.75 0.9 
    do 
        tmux send-keys -t cart_pole "echo 'Running human reliance ${human_rel}' >> ${LOGFILE} 2>&1"  ENTER 
        human_reliance_by_concept=(${human_rel} ${human_rel} ${human_rel} ${human_rel} ${human_rel} ${human_rel} ${human_rel} ${human_rel} ${human_rel} ${human_rel} ${human_rel} ${human_rel} ${human_rel} ${human_rel} ${human_rel} ${human_rel})
        human_reliance_by_concept="${human_reliance_by_concept[*]}"
 
        tmux send-keys -t cart_pole "conda activate ${environment}; python real_world_experiments.py --seed ${seed} --environment_string ${env} --training_timesteps ${training_timesteps} --num_concepts_selected 0 --selection_function ${selection_function} --cbm_accuracy_by_concept ${cbm_accuracy} --human_accuracy_by_concept ${human_accuracy_by_concept} --human_reliance_by_concept ${human_reliance_by_concept} --target_abstraction ${target_abstraction} --out_folder cart_pole >> ${LOGFILE} 2>&1"  ENTER 
    done 

    # # Reward Perturbation
    # tmux send-keys -t cart_pole "echo 'Running cart pole with reward perturbation' >> ${LOGFILE} 2>&1"  ENTER 
    # num_concepts_selected=4
    # selection_function=policy
    # for reward_error in 0.1 0.25 0.5 0.75 1
    # do 
    #     for env in cart_pole cart_pole_binary cart_pole_llm cart_pole_post_hoc
    #     do 
    #         tmux send-keys -t cart_pole "echo 'Running cart pole ${env} with selection function ${selected_function}' >> ${LOGFILE} 2>&1"  ENTER 
    #         tmux send-keys -t cart_pole "conda activate ${environment}; python real_world_experiments.py --seed ${seed} --environment_string ${env} --training_timesteps ${training_timesteps} --num_concepts_selected 4 --selection_function ${selection_function} --reward_error ${reward_error} --out_folder cart_pole >> ${LOGFILE} 2>&1"  ENTER 
    #     done 
    # done 
done 
