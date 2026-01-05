#!/bin/bash 
: > runs/logs/error_cub.txt
LOGFILE=../../runs/logs/error_cub.txt

environment=food
for seed in 42 43 44 
do 
    tmux new-session -d -s concepts_cub_${seed}
    tmux send-keys -t concepts_cub_${seed} ENTER 
    tmux send-keys -t concepts_cub_${seed} "source ~/.bashrc" ENTER
    tmux send-keys -t concepts_cub_${seed} "cd scripts/notebooks" ENTER
    tmux send-keys -t concepts_cub_${seed} "export PYTHONWARNINGS='ignore'" ENTER
    tmux send-keys -t concepts_cub_${seed} "export GYMNASIUM_DISABLE_WARNINGS=1" ENTER
done 

for seed in 42 43 44
do 
    num_concepts_selected=311
    tmux send-keys -t concepts_cub_${seed} "conda activate ${environment}; python -u supervised_learning.py --seed ${seed} --num_concepts_selected ${num_concepts_selected} --out_folder cub >> ${LOGFILE} 2>&1"  ENTER 
done 
