#!/bin/bash

sessions=(
  imperfect_cart_pole_pong_imperfect_concepts
  imperfect_cart_pole_pong_random
  imperfect_cart_pole_pong_entropy
  imperfect_cart_pole_pong_greedy
  imperfect_cart_pole_pong_lp
  imperfect_cart_pole_pong_multiple

  imperfect_mini_grid_boxing_imperfect_concepts
  imperfect_mini_grid_boxing_random
  imperfect_mini_grid_boxing_entropy
  imperfect_mini_grid_boxing_greedy
  imperfect_mini_grid_boxing_lp
  imperfect_mini_grid_boxing_multiple
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
  [boxing]=15000000
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
  for method in imperfect_concepts random entropy greedy lp multiple
  do 
    for env in cart_pole pong 
    do 
      tmux send-keys -t imperfect_cart_pole_pong_${method} "conda activate ${environment}; python -u method_comparison_intervention.py --seed ${seed} --environment_string ${env} --training_timesteps ${training_timesteps[$env]} --gold_timesteps ${gold_timesteps[$env]} --num_concepts_selected ${num_concepts[$env]} --method ${method} --intervention_prob 0.5 --out_folder intervention >> ../../runs/logs/error_imperfect_cart_pole_pong_${method}.txt 2>&1" ENTER
    done 
  done 

  for method in imperfect_concepts random entropy greedy lp multiple 
  do 
    for env in boxing # mini_grid  
    do 
      tmux send-keys -t imperfect_mini_grid_boxing_${method} "conda activate ${environment}; python -u method_comparison_intervention.py --seed ${seed} --environment_string ${env} --training_timesteps ${training_timesteps[$env]} --gold_timesteps ${gold_timesteps[$env]} --num_concepts_selected ${num_concepts[$env]} --method ${method} --intervention_prob 0.5 --out_folder intervention >> ../../runs/logs/error_imperfect_mini_grid_boxing_${method}.txt 2>&1" ENTER
    done 
  done 

  env=cart_pole
  for intervention_prob in 0.25 0.75
  do
    for method in imperfect_concepts random entropy greedy lp multiple
    do 
      tmux send-keys -t imperfect_cart_pole_pong_${method} "conda activate ${environment}; python -u method_comparison_intervention.py --seed ${seed} --environment_string ${env} --training_timesteps ${training_timesteps[$env]} --gold_timesteps ${gold_timesteps[$env]} --num_concepts_selected ${num_concepts[$env]} --method ${method} --intervention_prob ${intervention_prob} --out_folder intervention >> ../../runs/logs/error_imperfect_cart_pole_pong_${method}.txt 2>&1" ENTER
    done 
  done 
  
  # env=cart_pole
  # tmux send-keys -t imperfect_cart_pole_pong_lp "conda activate ${environment}; python -u pareto_curve.py --seed ${seed} --environment_string ${env} --training_timesteps ${training_timesteps[$env]} --gold_timesteps ${gold_timesteps[$env]} --num_concepts_selected ${num_concepts[$env]} --intervention_prob 0.5 --out_folder intervention >> ../../runs/logs/error_imperfect_cart_pole_pong_lp.txt 2>&1" ENTER
done
