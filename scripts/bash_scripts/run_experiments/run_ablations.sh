#!/bin/bash

# Seeds to run in parallel
SEEDS=(42 43 44)

# Map seeds to GPUs (manually, can also extend dynamically)
# Make sure you have enough GPUs for the number of seeds you run in parallel
GPU_MAP=(0 1 2)

sessions=(
  ablations

  # perfect_mini_grid_relaxed
  # perfect_mini_grid_rho_0
  # perfect_mini_grid_rho_05
  # perfect_mini_grid_rho_075
  # perfect_mini_grid_mi

  # perfect_cart_pole_relaxed
  # perfect_cart_pole_rho_0
  # perfect_cart_pole_rho_05
  # perfect_cart_pole_rho_075
  # perfect_cart_pole_mi

  # imperfect_mini_grid_relaxed
  # imperfect_mini_grid_rho_0
  # imperfect_mini_grid_rho_05
  # imperfect_mini_grid_rho_075
  # imperfect_mini_grid_mi

  # imperfect_cart_pole_relaxed
  # imperfect_cart_pole_rho_0
  # imperfect_cart_pole_rho_05
  # imperfect_cart_pole_rho_075
  # imperfect_cart_pole_mi
)

environment=food

# Define timesteps and concepts for each environment
declare -A gold_timesteps=(
  [mini_grid]=1000000
  [cart_pole]=4000000
  [pong]=15000000
  [boxing]=30000000
  [glucose]=4000000
)

declare -A training_timesteps=(
  [mini_grid]=500000
  [cart_pole]=4000000
  [pong]=15000000
  [boxing]=15000000
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

# Create tmux sessions for each experiment × seed, assigning GPU
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
    gpu=${GPU_MAP[$idx]}
    true_seed=$(( SEEDS[idx] + shift ))
    
    env=mini_grid 
    for g in 100000 250000 500000
    do 
      for method in rho_075 multiple_log # entropy greedy lp_hybrid multiple_log
      do 
        session="ablations_${seed}"
        tmux send-keys -t "$session" \
          "conda activate ${environment}; CUDA_VISIBLE_DEVICES=$gpu python -u method_comparison_imperfect.py \
          --seed ${true_seed} \
          --environment_string ${env} \
          --training_timesteps ${training_timesteps[$env]} \
          --gold_timesteps ${g} \
          --num_concepts_selected ${num_concepts[$env]} \
          --method ${method} \
          --out_folder ablations >> ../../runs/logs/error_ablations_${true_seed}.txt 2>&1" ENTER
      done 
    done 

    # for method in rho_0 # relaxed rho_05 rho_075 mi 
    # do 
    #   for env in cart_pole mini_grid
    #   do 
    #     session="perfect_${env}_${method}_${seed}"
    #     tmux send-keys -t "$session" \
    #       "conda activate ${environment}; CUDA_VISIBLE_DEVICES=$gpu python -u method_comparison_perfect.py \
    #       --seed ${true_seed} \
    #       --environment_string ${env} \
    #       --training_timesteps ${training_timesteps[$env]} \
    #       --gold_timesteps ${gold_timesteps[$env]} \
    #       --num_concepts_selected ${num_concepts[$env]} \
    #       --method ${method} \
    #       --out_folder basic >> ../../runs/logs/error_perfect_${env}_${method}_${true_seed}.txt 2>&1" ENTER
    #   done
    # done

    # for method in rho_075 # relaxed rho_0 rho_05 rho_075 mi 
    # do 
    #   for env in mini_grid # cart_pole mini_grid
    #   do 
    #     session="imperfect_${env}_${method}_${seed}"
    #     tmux send-keys -t "$session" \
    #       "conda activate ${environment}; CUDA_VISIBLE_DEVICES=$gpu python -u method_comparison_imperfect.py \
    #       --seed ${true_seed} \
    #       --environment_string ${env} \
    #       --training_timesteps ${training_timesteps[$env]} \
    #       --gold_timesteps ${gold_timesteps[$env]} \
    #       --num_concepts_selected ${num_concepts[$env]} \
    #       --method ${method} \
    #       --out_folder basic >> ../../runs/logs/error_imperfect_${env}_${method}_${true_seed}.txt 2>&1" ENTER
    #   done
    # done

  #   for method in rho_0 # relaxed rho_05 rho_075 mi 
  #   do 
  #     for env in cart_pole # mini_grid
  #     do 
  #       session="imperfect_${env}_${method}_${seed}"
  #       tmux send-keys -t "$session" \
  #         "conda activate ${environment}; CUDA_VISIBLE_DEVICES=$gpu python -u method_comparison_imperfect.py \
  #         --seed ${true_seed} \
  #         --environment_string ${env} \
  #         --training_timesteps ${training_timesteps[$env]} \
  #         --gold_timesteps ${gold_timesteps[$env]} \
  #         --num_concepts_selected ${num_concepts[$env]} \
  #         --method ${method} \
  #         --out_folder basic >> ../../runs/logs/error_imperfect_${env}_${method}_${true_seed}.txt 2>&1" ENTER
  #     done
  #   done
  done 
done