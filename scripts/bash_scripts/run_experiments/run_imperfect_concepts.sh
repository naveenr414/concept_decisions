#!/bin/bash

# Seeds to run
SEEDS=(42 43 44)

# Assign GPUs to sessions
GPU_MAP=(0 2 3)  # 3 GPUs

# Define tmux groups (session “types”)
SESSION_GROUPS_PONG_BOXING=(perfect_pong_boxing)
SESSION_GROUPS_MINI_CART_GLUC=(perfect_mini_grid_cart_pole_glucose)

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

# Function to create and setup tmux session with GPU
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

# # Create tmux sessions and log files for each seed × session group
for idx in "${!SEEDS[@]}"; do
  seed=${SEEDS[$idx]}
  gpu=${GPU_MAP[$idx]}  # assign GPU for this seed

  # Pong/Boxing sessions
  for i in {1..3}; do
    session_name="${SESSION_GROUPS_PONG_BOXING[0]}_${i}_seed${seed}"
    setup_tmux_session "$session_name" "$gpu"
    : > "runs/logs/error_${session_name}.txt"
  done

  # MiniGrid/CartPole/Glucose sessions
  for i in {1..3}; do
    session_name="${SESSION_GROUPS_MINI_CART_GLUC[0]}_${i}_seed${seed}"
    setup_tmux_session "$session_name" "$gpu"
    : > "runs/logs/error_${session_name}.txt"
  done
done

# Run experiments for each seed
for seed in "${SEEDS[@]}"; do
  # MiniGrid / CartPole / Glucose
  for env in mini_grid cart_pole glucose
  do
    for concept_accuracy in 0.75 0.85 0.95; do
      for concept_fraction in 0.25 0.5 1; do
        case "$concept_accuracy" in
          0.75) idx=1 ;;
          0.85) idx=2 ;;
          0.95) idx=3 ;;
        esac
        tmux_target="perfect_mini_grid_cart_pole_glucose_${idx}_seed${seed}"

        num_concepts_selected=$(printf "%.0f" $(echo "${num_concepts[$env]} * ${concept_fraction}" | bc))

        tmux send-keys -t "$tmux_target" \
          "conda activate ${environment}; python -u only_multiple.py \
          --seed ${seed} \
          --environment_string ${env} \
          --training_timesteps ${training_timesteps[$env]} \
          --gold_timesteps ${gold_timesteps[$env]} \
          --num_concepts_selected ${num_concepts_selected} \
          --concept_accuracy ${concept_accuracy} \
          --out_folder imperfect >> ../../runs/logs/error_${tmux_target}.txt 2>&1" ENTER
      done
    done
  done

  # Pong / Boxing
  for env in pong boxing
  do 
    for concept_accuracy in 0.75 0.85 0.95; do
      for concept_fraction in 0.25 0.5 1; do
        case "$concept_accuracy" in
          0.75) idx=1 ;;
          0.85) idx=2 ;;
          0.95) idx=3 ;;
        esac
        tmux_target="perfect_pong_boxing_${idx}_seed${seed}"

        num_concepts_selected=$(printf "%.0f" $(echo "${num_concepts[$env]} * ${concept_fraction}" | bc))

        tmux send-keys -t "$tmux_target" \
          "conda activate ${environment}; python -u only_multiple.py \
          --seed ${seed} \
          --environment_string ${env} \
          --training_timesteps ${training_timesteps[$env]} \
          --gold_timesteps ${gold_timesteps[$env]} \
          --num_concepts_selected ${num_concepts_selected} \
          --concept_accuracy ${concept_accuracy} \
          --out_folder imperfect >> ../../runs/logs/error_${tmux_target}.txt 2>&1" ENTER
      done
    done
  done

  env=mini_grid
  for concept_accuracy in 0.75 0.85 0.95; do
    for training_ts in 250000 500000 750000 1000000; do

      case "$concept_accuracy" in
        0.75) idx=1 ;;
        0.85) idx=2 ;;
        0.95) idx=3 ;;
      esac

      tmux_target="perfect_mini_grid_cart_pole_glucose_${idx}_seed${seed}"

      tmux send-keys -t "$tmux_target" \
        "conda activate ${environment}; python -u only_multiple.py \
        --seed ${seed} \
        --environment_string ${env} \
        --training_timesteps ${training_ts} \
        --gold_timesteps ${gold_timesteps[$env]} \
        --num_concepts_selected ${num_concepts[$env]} \
        --concept_accuracy ${concept_accuracy} \
        --out_folder imperfect >> ../../runs/logs/error_${tmux_target}.txt 2>&1" ENTER

    done
  done



  env=mini_grid
  for concept_fraction in 0.25 0.5 1; do

    num_concepts_selected=$(printf "%.0f" $(echo "${num_concepts[$env]} * ${concept_fraction}" | bc))

    for training_ts in 250000 500000 750000 1000000; do

      case "$concept_fraction" in
        0.25) idx=1 ;;
        0.5) idx=2 ;;
        1) idx=3 ;;
      esac

      tmux_target="perfect_mini_grid_cart_pole_glucose_${idx}_seed${seed}"

      tmux send-keys -t "$tmux_target" \
        "conda activate ${environment}; python -u only_multiple.py \
        --seed ${seed} \
        --environment_string ${env} \
        --training_timesteps ${training_ts} \
        --gold_timesteps ${gold_timesteps[$env]} \
        --num_concepts_selected ${num_concepts_selected} \
        --concept_accuracy 0.95 \
        --out_folder imperfect >> ../../runs/logs/error_${tmux_target}.txt 2>&1" ENTER

    done
  done
  
done
