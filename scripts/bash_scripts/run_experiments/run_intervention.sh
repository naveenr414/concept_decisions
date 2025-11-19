#!/bin/bash

# Create log files for each environment
# : > runs/logs/error_intervention_mini_grid.txt
: > runs/logs/error_intervention_cart_pole.txt
# : > runs/logs/error_intervention_pong.txt
# : > runs/logs/error_intervention_boxing.txt

# LOGFILE_MINI_GRID=../../runs/logs/error_intervention_mini_grid.txt
LOGFILE_CART_POLE=../../runs/logs/error_intervention_cart_pole.txt
# LOGFILE_PONG=../../runs/logs/error_intervention_pong.txt
# LOGFILE_BOXING=../../runs/logs/error_intervention_boxing.txt

environment=food

# Define timesteps and concepts for each environment
declare -A gold_timesteps=(
  [mini_grid]=1000000
  [cart_pole]=4000000
  [pong]=15000000
  [boxing]=15000000
)

# declare -A gold_timesteps=(
#   [mini_grid]=10
#   [cart_pole]=10
#   [pong]=10
#   [boxing]=10
# )

declare -A training_timesteps=(
  [mini_grid]=1000000
  [cart_pole]=4000000
  [pong]=15000000
  [boxing]=15000000
)

# declare -A training_timesteps=(
#   [mini_grid]=10
#   [cart_pole]=10
#   [pong]=10
#   [boxing]=10
# )

declare -A num_concepts=(
  [mini_grid]=11
  [cart_pole]=3
  [pong]=57
  [boxing]=48
)

# Function to create and setup tmux session
setup_tmux_session() {
  local session_name=$1
  tmux new-session -d -s ${session_name}
  tmux send-keys -t ${session_name} ENTER 
  tmux send-keys -t ${session_name} "source ~/.bashrc" ENTER
  tmux send-keys -t ${session_name} "cd scripts/notebooks" ENTER
  tmux send-keys -t ${session_name} "export PYTHONWARNINGS='ignore'" ENTER
  tmux send-keys -t ${session_name} "export GYMNASIUM_DISABLE_WARNINGS=1" ENTER
}

# Create tmux sessions for each environment
# setup_tmux_session "intervention_mini_grid"
setup_tmux_session "intervention_cart_pole"
# setup_tmux_session "intervention_pong"
# setup_tmux_session "intervention_boxing"

# Run experiments
for seed in 42
do 
#   for intervention_prob in 0.25 0.5 0.75
#   do 
    # Mini Grid
    # env=mini_grid
    # tmux send-keys -t intervention_mini_grid "conda activate ${environment}; python -u intervention_comparison.py --seed ${seed} --environment_string ${env} --training_timesteps ${training_timesteps[$env]} --gold_timesteps ${gold_timesteps[$env]} --num_concepts_selected ${num_concepts[$env]} --out_folder intervention --intervention_prob ${intervention_prob} >> ${LOGFILE_MINI_GRID} 2>&1" ENTER

    # Cart Pole
    # env=cart_pole
    # tmux send-keys -t intervention_cart_pole "conda activate ${environment}; python -u intervention_comparison.py --seed ${seed} --environment_string ${env} --training_timesteps ${training_timesteps[$env]} --gold_timesteps ${gold_timesteps[$env]} --num_concepts_selected ${num_concepts[$env]} --out_folder intervention --intervention_prob ${intervention_prob} >> ${LOGFILE_CART_POLE} 2>&1" ENTER

    # # Pong
    # env=pong
    # tmux send-keys -t intervention_pong "conda activate ${environment}; python -u intervention_comparison.py --seed ${seed} --environment_string ${env} --training_timesteps ${training_timesteps[$env]} --gold_timesteps ${gold_timesteps[$env]} --num_concepts_selected ${num_concepts[$env]} --out_folder intervention  --intervention_prob ${intervention_prob} >> ${LOGFILE_PONG} 2>&1" ENTER

    # # Boxing
    # env=boxing
    # tmux send-keys -t intervention_boxing "conda activate ${environment}; python -u intervention_comparison.py --seed ${seed} --environment_string ${env} --training_timesteps ${training_timesteps[$env]} --gold_timesteps ${gold_timesteps[$env]} --num_concepts_selected ${num_concepts[$env]} --out_folder intervention --intervention_prob ${intervention_prob} >> ${LOGFILE_BOXING} 2>&1" ENTER
   #done 

  # Cart Pole
  env=cart_pole
  tmux send-keys -t intervention_cart_pole "conda activate ${environment}; python -u pareto_curve.py --seed ${seed} --environment_string ${env} --training_timesteps ${training_timesteps[$env]} --gold_timesteps ${gold_timesteps[$env]} --num_concepts_selected ${num_concepts[$env]} --out_folder pareto --intervention_prob 0.5 --num_samples 10 >> ${LOGFILE_CART_POLE} 2>&1" ENTER
done
