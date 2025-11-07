#!/bin/bash 
: > runs/logs/error_cub.txt
LOGFILE=../../runs/logs/error_cub.txt

environment=food
tmux new-session -d -s concepts_cub
tmux send-keys -t concepts_cub ENTER 
tmux send-keys -t concepts_cub "source ~/.bashrc" ENTER
tmux send-keys -t concepts_cub "cd scripts/notebooks" ENTER
tmux send-keys -t concepts_cub "export PYTHONWARNINGS='ignore'" ENTER
tmux send-keys -t concepts_cub "export GYMNASIUM_DISABLE_WARNINGS=1" ENTER

for seed in 42 43 44
do 
    num_concepts_selected=311
    epochs=50

    tmux send-keys -t concepts_cub "conda activate ${environment}; python -u supervised_learning.py --seed ${seed} --num_concepts_selected ${num_concepts_selected} --epochs ${epochs} --out_folder cub >> ${LOGFILE} 2>&1"  ENTER 
done 
