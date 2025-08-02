#!/bin/bash 
: > runs/logs/error_tree.txt
: > runs/logs/error_cycle.txt
LOGFILE_TREE=../../runs/logs/error_tree.txt
LOGFILE_CYCLE=../../runs/logs/error_cycle.txt

#### Cycle Environment

environment=food
tmux new-session -d -s match_cycle
tmux send-keys -t match_cycle ENTER 
tmux send-keys -t match_cycle "cd scripts/notebooks" ENTER

tmux new-session -d -s match_tree
tmux send-keys -t match_tree ENTER 
tmux send-keys -t match_tree "cd scripts/notebooks" ENTER


for seed in 42
do 
    env=cycle 
    LOGFILE=${LOGFILE_CYCLE}
    # Impact of Size
    # for nodes in 4 8 16
    # do 
    #     tmux send-keys -t match_${env} "echo 'Running cycle with nodes ${nodes}' >> ${LOGFILE} 2>&1"  ENTER 
    #     tmux send-keys -t match_${env} "conda activate ${environment}; python synthetic_environments.py --seed ${seed} --environment_string ${env} --environment_nodes ${nodes} --show-baseline --num_concepts_selected 4 --out_folder synthetic >> ${LOGFILE} 2>&1"  ENTER 
    # done 
    # nodes=16
    # for concepts_selected in 8 12 16
    # do 
    #     tmux send-keys -t match_${env} "conda activate ${environment}; python synthetic_environments.py --seed ${seed} --environment_string ${env} --environment_nodes ${nodes} --show-baseline --num_concepts_selected ${concepts_selected} --out_folder synthetic >> ${LOGFILE} 2>&1"  ENTER 
    # done 

    # # Error
    # for nodes in 4 8 16
    # do 
    #     for fixed_accuracy in 0.1 0.25 0.5 0.75 0.9 0.99
    #     do 
    #         tmux send-keys -t match_${env} "echo 'Running cycle with nodes ${nodes} accuracy ${fixed_accuracy}' >> ${LOGFILE_CYCLE} 2>&1"  ENTER 
    #         cbm_accuracy=$(printf "%s " $(yes "$fixed_accuracy" | head -n $((nodes-1))))
    #         tmux send-keys -t match_${env} "conda activate ${environment}; python synthetic_environments.py --seed ${seed} --environment_string ${env} --environment_nodes ${nodes} --cbm_accuracy_by_concept ${cbm_accuracy} --out_folder synthetic >> ${LOGFILE} 2>&1" ENTER
    #     done 
    # done 

    # Error & Intervention
    tmux send-keys -t match_${env} "echo 'Running cycle with error & intervention' >> ${LOGFILE} 2>&1"  ENTER 
    nodes=4 
    human_accuracy_by_concept=(0.9 0.9 0.9)
    human_accuracy_by_concept="${human_accuracy_by_concept[*]}"
    human_reliance_by_concept=(0.75 0.75 0.75)
    human_reliance_by_concept="${human_reliance_by_concept[*]}"
    target_abstraction=0.1
    cbm_accuracy=(0.5 1 1)
    cbm_accuracy="${cbm_accuracy[*]}"
    tmux send-keys -t match_${env} "conda activate ${environment}; python synthetic_environments.py --seed ${seed} --environment_string ${env} --environment_nodes ${nodes} --cbm_accuracy_by_concept ${cbm_accuracy} --human_accuracy_by_concept ${human_accuracy_by_concept} --human_reliance_by_concept ${human_reliance_by_concept} --target_abstraction ${target_abstraction} --out_folder synthetic >> ${LOGFILE} 2>&1" ENTER
    cbm_accuracy=(1 0.5 1)
    cbm_accuracy="${cbm_accuracy[*]}"
    tmux send-keys -t match_${env} "conda activate ${environment}; python synthetic_environments.py --seed ${seed} --environment_string ${env} --environment_nodes ${nodes} --cbm_accuracy_by_concept ${cbm_accuracy} --human_accuracy_by_concept ${human_accuracy_by_concept} --human_reliance_by_concept ${human_reliance_by_concept} --target_abstraction ${target_abstraction} --out_folder synthetic >> ${LOGFILE} 2>&1" ENTER
    cbm_accuracy=(1 1 0.5) 
    cbm_accuracy="${cbm_accuracy[*]}"
    tmux send-keys -t match_${env} "conda activate ${environment}; python synthetic_environments.py --seed ${seed} --environment_string ${env} --environment_nodes ${nodes} --cbm_accuracy_by_concept ${cbm_accuracy} --human_accuracy_by_concept ${human_accuracy_by_concept} --human_reliance_by_concept ${human_reliance_by_concept} --target_abstraction ${target_abstraction} --out_folder synthetic >> ${LOGFILE} 2>&1" ENTER

    # cbm_accuracy=(0.5 0.5 0.5)
    # cbm_accuracy="${cbm_accuracy[*]}"
    # for human_rel in 0.1 0.25 0.5 0.75 0.9 
    # do 
    #     human_reliance_by_concept=(${human_rel} ${human_rel} ${human_rel})
    #     human_reliance_by_concept="${human_reliance_by_concept[*]}"
    #     tmux send-keys -t match_${env} "conda activate ${environment}; python synthetic_environments.py --seed ${seed} --environment_string ${env} --environment_nodes ${nodes} --cbm_accuracy_by_concept ${cbm_accuracy} --human_accuracy_by_concept ${human_accuracy_by_concept} --human_reliance_by_concept ${human_reliance_by_concept} --target_abstraction ${target_abstraction} --out_folder synthetic >> ${LOGFILE} 2>&1" ENTER
    # done 

    # Reward Perturbation
    # nodes=4
    # tmux send-keys -t match_${env} "echo 'Running cycle with reward perturbation' >> ${LOGFILE} 2>&1"  ENTER 
    # num_concepts_selected=3
    # for reward_error in 0.1 0.25 0.5 0.75 1 1.25 1.5 1.75 2
    # do 
    #     tmux send-keys -t match_${env} "conda activate ${environment}; python synthetic_environments.py --seed ${seed} --environment_string ${env} --environment_nodes ${nodes} --show-baseline --num_concepts_selected ${num_concepts_selected} --reward_error ${reward_error} --out_folder synthetic >> ${LOGFILE} 2>&1" ENTER
    # done 
    # for transition_error in 0.1 0.25 0.5 0.75 1 1.25 1.5 1.75 2
    # do 
    #     tmux send-keys -t match_${env} "conda activate ${environment}; python synthetic_environments.py --seed ${seed} --environment_string ${env} --environment_nodes ${nodes} --show-baseline --num_concepts_selected ${num_concepts_selected} --transition_error ${transition_error} --out_folder synthetic >> ${LOGFILE} 2>&1" ENTER
    # done 

    # #### Tree Environment

    env=tree
    LOGFILE=${LOGFILE_TREE}

    # Impact of Size
    # for nodes in 3 7 15 31 63 127
    # do 
    #     tmux send-keys -t match_${env} "echo 'Running trees with ${nodes}' >> ${LOGFILE} 2>&1"  ENTER 
    #     tmux send-keys -t match_${env} "conda activate ${environment}; python synthetic_environments.py --seed ${seed} --environment_string ${env} --environment_nodes ${nodes} --show-baseline --num_concepts_selected 4 --out_folder synthetic >> ${LOGFILE} 2>&1" ENTER
    # done 
    # # Error & Intervention
    # nodes=15
    # for fixed_accuracy in 0.1 0.25 0.5 0.6 0.75 0.8 0.9 0.99
    # do 
    #     tmux send-keys -t match_${env} "echo 'Running trees with 15 nodes, fixed accuracy ${fixed_accuracy}' >> ${LOGFILE} 2>&1"  ENTER 
    #     cbm_accuracy=$(yes "$fixed_accuracy" | head -n 5 | paste -sd " ")
    #     tmux send-keys -t match_${env} "conda activate ${environment}; python synthetic_environments.py --seed ${seed} --environment_string ${env} --environment_nodes ${nodes} --cbm_accuracy_by_concept ${cbm_accuracy} --out_folder synthetic >> ${LOGFILE} 2>&1" ENTER
    # done 
    # nodes=127
    # for fixed_accuracy in 0.1 0.25 0.5 0.6 0.75 0.8 0.9 0.99
    # do 
    #     tmux send-keys -t match_${env} "echo 'Running trees with 127 nodes, fixed accuracy ${fixed_accuracy}' >> ${LOGFILE} 2>&1"  ENTER 
    #     cbm_accuracy=$(yes "$fixed_accuracy" | head -n 8 | paste -sd " ")
    #     tmux send-keys -t match_${env} "conda activate ${environment}; python synthetic_environments.py --seed ${seed} --environment_string ${env} --environment_nodes ${nodes} --cbm_accuracy_by_concept ${cbm_accuracy} --out_folder synthetic >> ${LOGFILE} 2>&1" ENTER
    # done 

    tmux send-keys -t match_${env} "echo 'Running tree with error & intervention' >> ${LOGFILE} 2>&1"  ENTER 
    nodes=15
    human_accuracy_by_concept=(0.9 0.9 0.9 0.9 0.9)
    human_accuracy_by_concept="${human_accuracy_by_concept[*]}"
    human_reliance_by_concept=(0.75 0.75 0.75 0.75 0.75)
    human_reliance_by_concept="${human_reliance_by_concept[*]}"
    target_abstraction=0.1
    cbm_accuracy=(0.5 1 1 1 1)
    cbm_accuracy="${cbm_accuracy[*]}"
    tmux send-keys -t match_${env} "conda activate ${environment}; python synthetic_environments.py --seed ${seed} --environment_string ${env} --environment_nodes ${nodes} --cbm_accuracy_by_concept ${cbm_accuracy} --human_accuracy_by_concept ${human_accuracy_by_concept} --human_reliance_by_concept ${human_reliance_by_concept} --target_abstraction ${target_abstraction} --out_folder synthetic >> ${LOGFILE} 2>&1" ENTER
    cbm_accuracy=(1 0.5 1 1 1)
    cbm_accuracy="${cbm_accuracy[*]}"
    tmux send-keys -t match_${env} "conda activate ${environment}; python synthetic_environments.py --seed ${seed} --environment_string ${env} --environment_nodes ${nodes} --cbm_accuracy_by_concept ${cbm_accuracy} --human_accuracy_by_concept ${human_accuracy_by_concept} --human_reliance_by_concept ${human_reliance_by_concept} --target_abstraction ${target_abstraction} --out_folder synthetic >> ${LOGFILE} 2>&1" ENTER
    cbm_accuracy=(1 1 0.5 1 1)
    cbm_accuracy="${cbm_accuracy[*]}"
    tmux send-keys -t match_${env} "conda activate ${environment}; python synthetic_environments.py --seed ${seed} --environment_string ${env} --environment_nodes ${nodes} --cbm_accuracy_by_concept ${cbm_accuracy} --human_accuracy_by_concept ${human_accuracy_by_concept} --human_reliance_by_concept ${human_reliance_by_concept} --target_abstraction ${target_abstraction} --out_folder synthetic >> ${LOGFILE} 2>&1" ENTER
    cbm_accuracy=(1 1 1 0.5 1)
    cbm_accuracy="${cbm_accuracy[*]}"
    tmux send-keys -t match_${env} "conda activate ${environment}; python synthetic_environments.py --seed ${seed} --environment_string ${env} --environment_nodes ${nodes} --cbm_accuracy_by_concept ${cbm_accuracy} --human_accuracy_by_concept ${human_accuracy_by_concept} --human_reliance_by_concept ${human_reliance_by_concept} --target_abstraction ${target_abstraction} --out_folder synthetic >> ${LOGFILE} 2>&1" ENTER
    cbm_accuracy=(1 1 1 1 0.5)
    cbm_accuracy="${cbm_accuracy[*]}"
    tmux send-keys -t match_${env} "conda activate ${environment}; python synthetic_environments.py --seed ${seed} --environment_string ${env} --environment_nodes ${nodes} --cbm_accuracy_by_concept ${cbm_accuracy} --human_accuracy_by_concept ${human_accuracy_by_concept} --human_reliance_by_concept ${human_reliance_by_concept} --target_abstraction ${target_abstraction} --out_folder synthetic >> ${LOGFILE} 2>&1" ENTER

    # cbm_accuracy=(0.5 0.5 0.5 0.5 0.5)
    # cbm_accuracy="${cbm_accuracy[*]}"
    # for human_rel in 0.1 0.25 0.5 0.75 0.9 
    # do 
    #     human_reliance_by_concept=(${human_rel} ${human_rel} ${human_rel} ${human_rel} ${human_rel})
    #     human_reliance_by_concept="${human_reliance_by_concept[*]}"
    #     tmux send-keys -t match_${env} "conda activate ${environment}; python synthetic_environments.py --seed ${seed} --environment_string ${env} --environment_nodes ${nodes} --cbm_accuracy_by_concept ${cbm_accuracy} --human_accuracy_by_concept ${human_accuracy_by_concept} --human_reliance_by_concept ${human_reliance_by_concept} --target_abstraction ${target_abstraction} --out_folder synthetic >> ${LOGFILE} 2>&1" ENTER
    # done 

    # # Reward Perturbation
    # nodes=15
    # tmux send-keys -t match_${env} "echo 'Running tree with reward perturbation' >> ${LOGFILE} 2>&1"  ENTER 
    # num_concepts_selected=4
    # for reward_error in 0.1 0.25 0.5 0.75 1 1.25 1.5 1.75 2
    # do 
    #     tmux send-keys -t match_${env} "conda activate ${environment}; python synthetic_environments.py --seed ${seed} --environment_string ${env} --environment_nodes ${nodes} --num_concepts_selected ${num_concepts_selected} --reward_error ${reward_error} --out_folder synthetic >> ${LOGFILE} 2>&1" ENTER
    # done 
    # for transition_error in 0.1 0.25 0.5 0.75 1 1.25 1.5 1.75 2
    # do 
    #     tmux send-keys -t match_${env} "conda activate ${environment}; python synthetic_environments.py --seed ${seed} --environment_string ${env} --environment_nodes ${nodes} --num_concepts_selected ${num_concepts_selected} --transition_error ${transition_error} --out_folder synthetic >> ${LOGFILE} 2>&1" ENTER
    # done 

    # nodes=127
    # tmux send-keys -t match_${env} "echo 'Running tree with reward perturbation' >> ${LOGFILE} 2>&1"  ENTER 
    # num_concepts_selected=4
    # for reward_error in 0.1 0.25 0.5 0.75 1 1.25 1.5 1.75 2
    # do 
    #     tmux send-keys -t match_${env} "conda activate ${environment}; python synthetic_environments.py --seed ${seed} --environment_string ${env} --environment_nodes ${nodes} --show-baseline --num_concepts_selected ${num_concepts_selected} --reward_error ${reward_error} --out_folder synthetic >> ${LOGFILE} 2>&1" ENTER
    # done 
    # for transition_error in 0.1 0.25 0.5 0.75 1 1.25 1.5 1.75 2
    # do 
    #     tmux send-keys -t match_${env} "conda activate ${environment}; python synthetic_environments.py --seed ${seed} --environment_string ${env} --environment_nodes ${nodes} --show-baseline --num_concepts_selected ${num_concepts_selected} --transition_error ${transition_error} --out_folder synthetic >> ${LOGFILE} 2>&1" ENTER
    # done 
done 
