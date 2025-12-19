#!/bin/bash

# Seeds to run in parallel
SEEDS=(42 43 44)

# Map seeds to GPUs (manually, can also extend dynamically)
# Make sure you have enough GPUs for the number of seeds you run in parallel
GPU_MAP=(0 2 3)

sessions=(
  base_mini_grid
  base_cart_pole
  base_pong
  base_boxing
  base_glucose
)

environment=food

declare -A gold_timesteps=(
  [mini_grid]=1000000
  [cart_pole]=4000000
  [pong]=15000000
  [boxing]=30000000
  [glucose]=4000000
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
      : > "runs/logs/error_${session_name}.txt"
  done
done

# Run experiments
for idx in "${!SEEDS[@]}"
do
    seed=${SEEDS[$idx]}
    gpu=${GPU_MAP[$idx]}

    for env in boxing # cart_pole mini_grid boxing pong glucose 
    do 
    session="base_${env}_${seed}"
    tmux send-keys -t "$session" \
        "conda activate ${environment}; CUDA_VISIBLE_DEVICES=$gpu python -u recreate_model.py \
        --seed ${seed} \
        --environment_string ${env} \
        --gold_timesteps ${gold_timesteps[$env]} \
        --out_folder training >> ../../runs/logs/error_base_${env}_${seed}.txt 2>&1" ENTER
    done
done
