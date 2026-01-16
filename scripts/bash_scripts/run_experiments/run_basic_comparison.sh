#!/bin/bash

# Seeds to run in parallel
SEEDS=(42 43 44)

# Map seeds to GPUs (manually, can also extend dynamically)
# Make sure you have enough GPUs for the number of seeds you run in parallel
GPU_MAP=(0 2 3)

sessions=(
  # perfect_mini_grid_perfect_concepts
  # perfect_mini_grid_random
  # perfect_mini_grid_entropy
  # perfect_mini_grid_greedy
  # perfect_mini_grid_lp_hybrid

  # perfect_cart_pole_perfect_concepts
  # perfect_cart_pole_random
  # perfect_cart_pole_entropy
  # perfect_cart_pole_greedy
  # perfect_cart_pole_lp_hybrid

  # perfect_pong_perfect_concepts
  # perfect_pong_random
  # perfect_pong_entropy
  # perfect_pong_greedy
  # perfect_pong_lp_hybrid

  # perfect_boxing_perfect_concepts
  # perfect_boxing_random
  # perfect_boxing_entropy
  # perfect_boxing_greedy
  # perfect_boxing_lp_hybrid

  # perfect_glucose_perfect_concepts
  # perfect_glucose_random
  # perfect_glucose_entropy
  # perfect_glucose_greedy
  # perfect_glucose_lp_hybrid

  # imperfect_cart_pole_pong_imperfect_concepts
  # imperfect_cart_pole_pong_random
  # imperfect_cart_pole_pong_entropy
  # imperfect_cart_pole_pong_greedy
  # imperfect_cart_pole_pong_lp_hybrid
  imperfect_cart_pole_pong_multiple_log

  # imperfect_mini_grid_boxing_imperfect_concepts
  # imperfect_mini_grid_boxing_random
  # imperfect_mini_grid_boxing_entropy
  # imperfect_mini_grid_boxing_greedy
  # imperfect_mini_grid_boxing_lp_hybrid
  # imperfect_mini_grid_boxing_multiple_log
)

environment=food

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
    gpu=${GPU_MAP[$idx]}
    true_seed=$(( SEEDS[idx] + shift ))

    # # # --- Perfect methods ---
    # for method in lp_hybrid # perfect_concepts random entropy greedy lp_hybrid
    # do 
    #   for env in cart_pole mini_grid pong boxing glucose
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

    # # # # # # --- Imperfect CartPole/Pong ---
    for method in multiple_log # imperfect_concepts random entropy greedy lp_hybrid multiple_log 
    do 
      for env in cart_pole # pong  
      do 
        session="imperfect_cart_pole_pong_${method}_${seed}"
        tmux send-keys -t "$session" \
          "conda activate ${environment}; CUDA_VISIBLE_DEVICES=$gpu python -u method_comparison_imperfect.py \
          --seed ${true_seed} \
          --environment_string ${env} \
          --training_timesteps ${training_timesteps[$env]} \
          --gold_timesteps ${gold_timesteps[$env]} \
          --num_concepts_selected ${num_concepts[$env]} \
          --method ${method} \
          --out_folder basic >> ../../runs/logs/error_imperfect_cart_pole_pong_${method}_${true_seed}.txt 2>&1" ENTER
      done
    done


  #   # # --- Imperfect MiniGrid/Boxing ---
  #   for method in lp_hybrid multiple_log # imperfect_concepts random entropy greedy lp_hybrid multiple_log 
  #   do 
  #     for env in mini_grid boxing   
  #     do 
  #       session="imperfect_mini_grid_boxing_${method}_${seed}"
  #       tmux send-keys -t "$session" \
  #         "conda activate ${environment}; CUDA_VISIBLE_DEVICES=$gpu python -u method_comparison_imperfect.py \
  #         --seed ${true_seed} \
  #         --environment_string ${env} \
  #         --training_timesteps ${training_timesteps[$env]} \
  #         --gold_timesteps ${gold_timesteps[$env]} \
  #         --num_concepts_selected ${num_concepts[$env]} \
  #         --method ${method} \
  #         --out_folder basic >> ../../runs/logs/error_imperfect_mini_grid_boxing_${method}_${true_seed}.txt 2>&1" ENTER
  #     done
  #   done
  done
done 