#!/bin/bash


SEEDS=(42 43 44)          # Add as many seeds as you want
GPU_MAP=(0 2 3)           # GPU assignment for each seed index

METHODS=(lp_hybrid)

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

for i in "${!SEEDS[@]}"; do
  seed=${SEEDS[$i]}
  gpu=${GPU_MAP[$i]}

  # cart_pole + pong sessions
  for m in "${METHODS[@]}"; do
    session_name="intervention_cart_pole_pong_${m}_${seed}"
    setup_tmux_session "$session_name" "$gpu"
    : > "runs/logs/error_${session_name}.txt"
  done

  # mini_grid + boxing sessions
  for m in "${METHODS[@]}"; do
    session_name="intervention_mini_grid_boxing_${m}_${seed}"
    setup_tmux_session "$session_name" "$gpu"
    : > "runs/logs/error_${session_name}.txt"
  done

  # pareto LP session (cart_pole only)
  # session_name="intervention_cart_pole_pong_lp_pareto_${seed}"
  # setup_tmux_session "$session_name" "$gpu"
  # : > "runs/logs/error_${session_name}.txt"
done


for seed in "${SEEDS[@]}"; do

  # for method in "${METHODS[@]}"; do
  #   for env in cart_pole 
  #   do

  #     tmux_target="intervention_cart_pole_pong_${method}_${seed}"

  #     tmux send-keys -t "$tmux_target" \
  #       "conda activate ${environment}; python -u method_comparison_intervention.py \
  #       --seed ${seed} \
  #       --environment_string ${env} \
  #       --training_timesteps ${training_timesteps[$env]} \
  #       --gold_timesteps ${gold_timesteps[$env]} \
  #       --num_concepts_selected ${num_concepts[$env]} \
  #       --method ${method} \
  #       --intervention_prob 0.5 \
  #       --out_folder intervention >> ../../runs/logs/error_${tmux_target}.txt 2>&1" ENTER
  #   done
  # done

  for method in "${METHODS[@]}"; do
    for env in boxing  
    do 

      tmux_target="intervention_mini_grid_boxing_${method}_${seed}"

      tmux send-keys -t "$tmux_target" \
        "conda activate ${environment}; python -u method_comparison_intervention.py \
        --seed ${seed} \
        --environment_string ${env} \
        --training_timesteps ${training_timesteps[$env]} \
        --gold_timesteps ${gold_timesteps[$env]} \
        --num_concepts_selected ${num_concepts[$env]} \
        --method ${method} \
        --intervention_prob 0.5 \
        --out_folder intervention >> ../../runs/logs/error_${tmux_target}.txt 2>&1" ENTER
    done
  done


  # env=cart_pole
  # for intervention_prob in 0.25 0.75; do
  #   for method in "${METHODS[@]}"; do

  #     tmux_target="intervention_cart_pole_pong_${method}_${seed}"

  #     tmux send-keys -t "$tmux_target" \
  #       "conda activate ${environment}; python -u method_comparison_intervention.py \
  #       --seed ${seed} \
  #       --environment_string ${env} \
  #       --training_timesteps ${training_timesteps[$env]} \
  #       --gold_timesteps ${gold_timesteps[$env]} \
  #       --num_concepts_selected ${num_concepts[$env]} \
  #       --method ${method} \
  #       --intervention_prob ${intervention_prob} \
  #       --out_folder intervention >> ../../runs/logs/error_${tmux_target}.txt 2>&1" ENTER
  #   done
  # done
done
