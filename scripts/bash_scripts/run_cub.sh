#!/bin/bash 
: > runs/logs/error_cub.txt
LOGFILE=../../runs/logs/error_cub.txt

environment=food
tmux new-session -d -s concepts
tmux send-keys -t concepts ENTER 
tmux send-keys -t concepts "cd scripts/notebooks" ENTER

for seed in 42
do 
    num_concepts_selected=2
    tmux send-keys -t concepts "conda activate ${environment}; python supervised_learning.py --seed ${seed} --num_concepts_selected ${num_concepts_selected} --out_folder cub >> ${LOGFILE} 2>&1"  ENTER 
done 
