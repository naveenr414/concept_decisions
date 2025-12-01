#!/bin/bash

sessions=(
  ablations
)


# Create log files for each environment
for s in "${sessions[@]}"; do
    : > "runs/logs/error_${s}.txt"
done

environment=food

# Define timesteps and concepts for each environment
declare -A gold_timesteps=(
  [mini_grid]=1000000
  [cart_pole]=4000000
  [pong]=15000000
  [boxing]=30000000
  [glucose]=4000000
)

# declare -A gold_timesteps=(
#   [mini_grid]=10
#   [cart_pole]=10
#   [pong]=10
#   [boxing]=10
# )

declare -A training_timesteps=(
  [mini_grid]=250000
  [cart_pole]=4000000
  [pong]=15000000
  [boxing]=15000000
  [glucose]=4000000
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
  [glucose]=10
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
for s in "${sessions[@]}"; do
    setup_tmux_session "$s"
done

# Run experiments
for seed in 42
do 
  env=mini_grid 
  for g in 100000 250000 500000
  do 
    for method in greedy lp multiple 
    do 
      tmux send-keys -t ablations "conda activate ${environment}; python -u method_comparison_imperfect.py --seed ${seed} --environment_string ${env} --training_timesteps ${training_timesteps[$env]} --gold_timesteps ${g} --num_concepts_selected ${num_concepts[$env]} --method ${method} --out_folder ablations >> ../../runs/logs/error_ablations.txt 2>&1" ENTER
    done 
  done 

  method=completeness
  for env in cart_pole mini_grid
  do 
    tmux send-keys -t ablations "conda activate ${environment}; python -u method_comparison_imperfect.py --seed ${seed} --environment_string ${env} --training_timesteps ${training_timesteps[$env]} --gold_timesteps ${gold_timesteps[$env]} --num_concepts_selected ${num_concepts[$env]} --method ${method} --out_folder basic >> ../../runs/logs/error_ablations.txt 2>&1" ENTER
  done 

  method=lp_policy
  for env in cart_pole mini_grid
  do 
    tmux send-keys -t ablations "conda activate ${environment}; python -u method_comparison_imperfect.py --seed ${seed} --environment_string ${env} --training_timesteps ${training_timesteps[$env]} --gold_timesteps ${gold_timesteps[$env]} --num_concepts_selected ${num_concepts[$env]} --method ${method} --out_folder basic >> ../../runs/logs/error_ablations.txt 2>&1" ENTER
  done 
done
