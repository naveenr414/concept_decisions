#!/bin/bash 
: > runs/logs/error_concepts.txt
LOGFILE=../../runs/logs/error_concepts.txt

: > runs/logs/error_glucose.txt
LOGFILE_GLUCOSE=../../runs/logs/error_glucose.txt

environment=food

for session in concepts_atari concepts_glucose 
do 
    tmux new-session -d -s ${session}
    tmux send-keys -t ${session} ENTER 
    tmux send-keys -t ${session} "source ~/.bashrc" ENTER
    tmux send-keys -t ${session} "cd scripts/notebooks" ENTER
    tmux send-keys -t ${session} "export PYTHONWARNINGS='ignore'" ENTER
    tmux send-keys -t ${session} "export GYMNASIUM_DISABLE_WARNINGS=1" ENTER
done 

declare -A training_timesteps=(
  [cyclic_4]=10000
  [cyclic_16]=10000
  [tree_7]=25000
  [tree_31]=25000
  [cart_pole]=1000000
  [mini_grid]=250000
  [glucose]=250000
  [pong]=4000000
  [boxing]=10000000
)

for seed in 42
do 
  for lr in 0.0006
  do 
    env=glucose
    training=${training_timesteps[$env]}
    tmux send-keys -t concepts_glucose "conda activate ${environment}; python train_base_model.py --seed ${seed} --environment_string ${env} --gold_timesteps ${training} --learning_rate ${lr} >> ${LOGFILE_GLUCOSE} 2>&1"  ENTER 
  done 
done 
