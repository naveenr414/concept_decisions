#!/bin/bash 
: > scripts/notebooks/error_tree.txt
: > scripts/notebooks/error_cycle.txt
LOGFILE_TREE=error_tree.txt
LOGFILE_CYCLE=error_cycle.txt
seed=42

#### Cycle Environment

environment=food
env=cycle 
tmux new-session -d -s match_${env}
tmux send-keys -t match_${env} ENTER 
tmux send-keys -t match_${env} "cd scripts/notebooks" ENTER

LOGFILE=${LOGFILE_CYCLE}
# # Impact of Size
for nodes in 4 8 16 32 64
do 
    tmux send-keys -t match_${env} "echo 'Running cycle with nodes ${nodes}' >> ${LOGFILE} 2>&1"  ENTER 
    tmux send-keys -t match_${env} "conda activate ${environment}; python synthetic_environments.py --seed ${seed} --environment_string ${env} --environment_nodes ${nodes} --show-baseline --num_concepts_selected 4 --out_folder synthetic >> ${LOGFILE} 2>&1"  ENTER 
done 

# # Error
for nodes in 4 8 
do 
    for fixed_accuracy in 0.25 0.5 0.75 0.9 0.99
    do 
        tmux send-keys -t match_${env} "echo 'Running cycle with nodes ${nodes} accuracy ${fixed_accuracy}' >> ${LOGFILE_CYCLE} 2>&1"  ENTER 
        cbm_accuracy=$(printf "%s " $(yes "$fixed_accuracy" | head -n $((nodes-1))))
        tmux send-keys -t match_${env} "conda activate ${environment}; python synthetic_environments.py --seed ${seed} --environment_string ${env} --environment_nodes ${nodes} --cbm_accuracy_by_concept ${cbm_accuracy} --out_folder synthetic >> ${LOGFILE} 2>&1" ENTER
    done 
done 

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
tmux send-keys -t match_${env} "conda activate ${environment}; python synthetic_environments.py --seed ${seed} --environment_string ${env} --environment_nodes ${nodes} --cbm_accuracy_by_concept ${cbm_accuracy} --human_accuracy_by_concept ${human_accuracy_by_concept} --human_reliance_by_concept ${human_reliance_by_concept} --target_abstraction ${target_abstraction} --out_folder synthetic >> ${LOGFILE} 2>&1" ENTER
cbm_accuracy=(1 1 0.5) 
tmux send-keys -t match_${env} "conda activate ${environment}; python synthetic_environments.py --seed ${seed} --environment_string ${env} --environment_nodes ${nodes} --cbm_accuracy_by_concept ${cbm_accuracy} --human_accuracy_by_concept ${human_accuracy_by_concept} --human_reliance_by_concept ${human_reliance_by_concept} --target_abstraction ${target_abstraction} --out_folder synthetic >> ${LOGFILE} 2>&1" ENTER

# Reward Perturbation
nodes=4
tmux send-keys -t match_${env} "echo 'Running cycle with reward perturbation' >> ${LOGFILE} 2>&1"  ENTER 
num_concepts_selected=3
for reward_error in 0.1 0.25 0.5 0.75 1
do 
    tmux send-keys -t match_${env} "conda activate ${environment}; python synthetic_environments.py --seed ${seed} --environment_string ${env} --environment_nodes ${nodes} --num_concepts_selected ${num_concepts_selected} --reward_error ${reward_error} --out_folder synthetic >> ${LOGFILE} 2>&1" ENTER
done 
for transition_error in 0.1 0.5 1
do 
    tmux send-keys -t match_${env} "conda activate ${environment}; python synthetic_environments.py --seed ${seed} --environment_string ${env} --environment_nodes ${nodes} --num_concepts_selected ${num_concepts_selected} --transition_error ${transition_error} --out_folder synthetic >> ${LOGFILE} 2>&1" ENTER
done 

#### Tree Environment

env=tree
LOGFILE=${LOGFILE_TREE}
tmux new-session -d -s match_${env}
tmux send-keys -t match_${env} ENTER 
tmux send-keys -t match_${env} "cd scripts/notebooks" ENTER

# Impact of Size
for nodes in 3 7 15 31 63 127
do 
    tmux send-keys -t match_${env} "echo 'Running trees with ${nodes}' >> ${LOGFILE} 2>&1"  ENTER 
    tmux send-keys -t match_${env} "conda activate ${environment}; python synthetic_environments.py --seed ${seed} --environment_string ${env} --environment_nodes ${nodes} --show-baseline --num_concepts_selected 4 --out_folder synthetic >> ${LOGFILE} 2>&1" ENTER
done 
# Error & Intervention
nodes=15
for fixed_accuracy in 0.25 0.5 0.75 0.9 0.99
do 
    tmux send-keys -t match_${env} "echo 'Running trees with 15 nodes, fixed accuracy ${fixed_accuracy}' >> ${LOGFILE} 2>&1"  ENTER 
    cbm_accuracy=$(yes "$fixed_accuracy" | head -n 5 | paste -sd " ")
    tmux send-keys -t match_${env} "conda activate ${environment}; python synthetic_environments.py --seed ${seed} --environment_string ${env} --environment_nodes ${nodes} --cbm_accuracy_by_concept ${cbm_accuracy} --out_folder synthetic >> ${LOGFILE} 2>&1" ENTER
done 
nodes=127
for fixed_accuracy in 0.25 0.5 0.75 0.9 0.99
do 
    tmux send-keys -t match_${env} "echo 'Running trees with 127 nodes, fixed accuracy ${fixed_accuracy}' >> ${LOGFILE} 2>&1"  ENTER 
    cbm_accuracy=$(yes "$fixed_accuracy" | head -n 8 | paste -sd " ")
    tmux send-keys -t match_${env} "conda activate ${environment}; python synthetic_environments.py --seed ${seed} --environment_string ${env} --environment_nodes ${nodes} --cbm_accuracy_by_concept ${cbm_accuracy} --out_folder synthetic >> ${LOGFILE} 2>&1" ENTER
done 

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
tmux send-keys -t match_${env} "conda activate ${environment}; python synthetic_environments.py --seed ${seed} --environment_string ${env} --environment_nodes ${nodes} --cbm_accuracy_by_concept ${cbm_accuracy} --human_accuracy_by_concept ${human_accuracy_by_concept} --human_reliance_by_concept ${human_reliance_by_concept} --target_abstraction ${target_abstraction} --out_folder synthetic >> ${LOGFILE} 2>&1" ENTER
cbm_accuracy=(1 1 0.5 1 1)
tmux send-keys -t match_${env} "conda activate ${environment}; python synthetic_environments.py --seed ${seed} --environment_string ${env} --environment_nodes ${nodes} --cbm_accuracy_by_concept ${cbm_accuracy} --human_accuracy_by_concept ${human_accuracy_by_concept} --human_reliance_by_concept ${human_reliance_by_concept} --target_abstraction ${target_abstraction} --out_folder synthetic >> ${LOGFILE} 2>&1" ENTER
cbm_accuracy=(1 1 0.5 1 1)
tmux send-keys -t match_${env} "conda activate ${environment}; python synthetic_environments.py --seed ${seed} --environment_string ${env} --environment_nodes ${nodes} --cbm_accuracy_by_concept ${cbm_accuracy} --human_accuracy_by_concept ${human_accuracy_by_concept} --human_reliance_by_concept ${human_reliance_by_concept} --target_abstraction ${target_abstraction} --out_folder synthetic >> ${LOGFILE} 2>&1" ENTER
cbm_accuracy=(1 1 1 1 0.5)
tmux send-keys -t match_${env} "conda activate ${environment}; python synthetic_environments.py --seed ${seed} --environment_string ${env} --environment_nodes ${nodes} --cbm_accuracy_by_concept ${cbm_accuracy} --human_accuracy_by_concept ${human_accuracy_by_concept} --human_reliance_by_concept ${human_reliance_by_concept} --target_abstraction ${target_abstraction} --out_folder synthetic >> ${LOGFILE} 2>&1" ENTER

# Reward Perturbation
nodes=15
tmux send-keys -t match_${env} "echo 'Running tree with reward perturbation' >> ${LOGFILE} 2>&1"  ENTER 
num_concepts_selected=4
for reward_error in 0.1 0.25 0.5 0.75 1
do 
    tmux send-keys -t match_${env} "conda activate ${environment}; python synthetic_environments.py --seed ${seed} --environment_string ${env} --environment_nodes ${nodes} --num_concepts_selected ${num_concepts_selected} --reward_error ${reward_error} --out_folder synthetic >> ${LOGFILE} 2>&1" ENTER
done 
for transition_error in 0.1 0.25 0.5 0.75 1
do 
    tmux send-keys -t match_${env} "conda activate ${environment}; python synthetic_environments.py --seed ${seed} --environment_string ${env} --environment_nodes ${nodes} --num_concepts_selected ${num_concepts_selected} --transition_error ${transition_error} --out_folder synthetic >> ${LOGFILE} 2>&1" ENTER
done 

