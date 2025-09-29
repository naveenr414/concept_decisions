#!/bin/bash 
: > runs/logs/error_cub.txt
LOGFILE=../../runs/logs/error_cub.txt

environment=food
tmux new-session -d -s concepts_cub
tmux send-keys -t concepts_cub ENTER 
tmux send-keys -t concepts_cub "cd scripts/notebooks" ENTER

for seed in 42 43 44
do 
    num_concepts_selected=40
    tmux send-keys -t concepts_cub "conda activate ${environment}; python supervised_learning.py --seed ${seed} --num_concepts_selected ${num_concepts_selected} --out_folder cub >> ${LOGFILE} 2>&1"  ENTER 
done 
