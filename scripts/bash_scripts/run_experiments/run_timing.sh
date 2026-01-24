#!/bin/bash

# Seeds to run in parallel
SEEDS=(42 43 44)

sessions=(
  #entropy
  #greedy
  timing_rho_075_cart_pole
  timing_rho_075_mini_grid
  timing_rho_075_pong
  timing_rho_075_boxing
  
  timing_multiple_log_cart_pole
  timing_multiple_log_mini_grid
  timing_multiple_log_pong
  timing_multiple_log_boxing
)

environment=food

declare -A gold_timesteps=(
  [mini_grid]=1000000
  [cart_pole]=4000000
  [pong]=15000000
  [boxing]=30000000
  [glucose]=4000000
)

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
  local gpu=$2
  tmux new-session -d -s ${session_name}
  tmux send-keys -t ${session_name} ENTER 
  tmux send-keys -t ${session_name} "source ~/.bashrc" ENTER
  tmux send-keys -t ${session_name} "cd scripts/notebooks" ENTER
  tmux send-keys -t ${session_name} "export PYTHONWARNINGS='ignore'" ENTER
  tmux send-keys -t ${session_name} "export GYMNASIUM_DISABLE_WARNINGS=1" ENTER
  tmux send-keys -t ${session_name} "export CUDA_VISIBLE_DEVICES=${gpu}" ENTER
}

# # Create tmux sessions for each experiment × seed, assigning GPU
for idx in "${!SEEDS[@]}"; do
  seed=${SEEDS[$idx]}
  gpu=${GPU_MAP[$idx]}
  for s in "${sessions[@]}"; do
      session_name="${s}_${seed}"
      setup_tmux_session "$session_name" "$gpu"
      for shift in 0 3 
      do 
        true_seed=$(( SEEDS[idx] + shift ))
        session_name="${s}_${true_seed}"
        : > "runs/logs/error_${session_name}.txt"
      done 
  done
done

# Run experiments
for shift in 0 3 
do 
  for idx in "${!SEEDS[@]}"; do
    seed=${SEEDS[$idx]}
    true_seed=$(( SEEDS[idx] + shift ))

    for method in  rho_075 multiple_log # entropy greedy
    do 
      for env in cart_pole mini_grid pong boxing
      do 
        session="timing_${method}_${env}_${seed}"
        tmux send-keys -t "$session" \
          "conda activate ${environment}; python -u get_runtimes.py \
          --seed ${true_seed} \
          --environment_string ${env} \
          --gold_timesteps ${gold_timesteps[$env]} \
          --num_concepts_selected ${num_concepts[$env]} \
          --method ${method} \
          --out_folder timing >> ../../runs/logs/error_${method}_${true_seed}.txt 2>&1" ENTER
      done 
    done 
  done
done 