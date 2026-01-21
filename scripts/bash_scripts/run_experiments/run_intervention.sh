#!/bin/bash


SEEDS=(42 43 44)          # Add as many seeds as you want
GPU_MAP=(0 2 3)           # GPU assignment for each seed index

METHODS=(random entropy greedy rho_075 multiple_log)

environment=food

declare -A gold_timesteps=(
  [mini_grid]=1000000
  [cart_pole]=4000000
  [pong]=15000000
  [boxing]=30000000
  [glucose]=4000000
)

declare -A training_timesteps=(
  [mini_grid]=250000
  [cart_pole]=4000000
  [pong]=4000000
  [boxing]=2000000
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

  for env in mini_grid_pong # cart_pole_boxing 
  do 
    # cart_pole + pong sessions
    for m in "${METHODS[@]}"; do
      session_name="intervention_${env}_${m}_${seed}"
      setup_tmux_session "$session_name" "$gpu"
      for shift in 0 3 
      do 
        true_seed=$(( SEEDS[idx] + shift ))
        session_name="${s}_${true_seed}"
        : > "runs/logs/error_${session_name}.txt"
      done 
    done
  done 
done


for idx in "${!SEEDS[@]}"; do
  for shift in 0 3
  do 
    seed=${SEEDS[$idx]}
    gpu=${GPU_MAP[$idx]}  # assign GPU for this seed
    true_seed=$(( SEEDS[idx] + shift ))
    echo ${true_seed}

    # for method in "${METHODS[@]}"; do
    #   for env in cart_pole 
    #   do
    #     tmux_target="intervention_cart_pole_boxing_${method}_${seed}"

    #     tmux send-keys -t "$tmux_target" \
    #       "conda activate ${environment}; python -u method_comparison_intervention.py \
    #       --seed ${true_seed} \
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
      for env in pong   
      do 

        tmux_target="intervention_mini_grid_pong_${method}_${seed}"

        tmux send-keys -t "$tmux_target" \
          "conda activate ${environment}; python -u method_comparison_intervention.py \
          --seed ${true_seed} \
          --environment_string ${env} \
          --training_timesteps ${training_timesteps[$env]} \
          --gold_timesteps ${gold_timesteps[$env]} \
          --num_concepts_selected ${num_concepts[$env]} \
          --method ${method} \
          --predictor_epochs 1 \
          --intervention_prob 0.0 \
          --out_folder intervention >> ../../runs/logs/error_${tmux_target}.txt 2>&1" ENTER

        tmux send-keys -t "$tmux_target" \
          "conda activate ${environment}; python -u method_comparison_intervention.py \
          --seed ${true_seed} \
          --environment_string ${env} \
          --training_timesteps ${training_timesteps[$env]} \
          --gold_timesteps ${gold_timesteps[$env]} \
          --num_concepts_selected ${num_concepts[$env]} \
          --method ${method} \
          --predictor_epochs 1 \
          --intervention_prob 0.5 \
          --out_folder intervention >> ../../runs/logs/error_${tmux_target}.txt 2>&1" ENTER
      done
    done

    for method in "${METHODS[@]}"; do
      for env in mini_grid  
      do 

        tmux_target="intervention_mini_grid_pong_${method}_${seed}"

        tmux send-keys -t "$tmux_target" \
          "conda activate ${environment}; python -u method_comparison_intervention.py \
          --seed ${true_seed} \
          --environment_string ${env} \
          --training_timesteps ${training_timesteps[$env]} \
          --gold_timesteps ${gold_timesteps[$env]} \
          --num_concepts_selected ${num_concepts[$env]} \
          --method ${method} \
          --predictor_epochs 1 \
          --intervention_prob 0.0 \
          --out_folder intervention >> ../../runs/logs/error_${tmux_target}.txt 2>&1" ENTER

        tmux send-keys -t "$tmux_target" \
          "conda activate ${environment}; python -u method_comparison_intervention.py \
          --seed ${true_seed} \
          --environment_string ${env} \
          --training_timesteps ${training_timesteps[$env]} \
          --gold_timesteps ${gold_timesteps[$env]} \
          --num_concepts_selected ${num_concepts[$env]} \
          --method ${method} \
          --predictor_epochs 1 \
          --intervention_prob 0.5 \
          --out_folder intervention >> ../../runs/logs/error_${tmux_target}.txt 2>&1" ENTER
      done
    done

    # for method in "${METHODS[@]}"; do
    #   for env in boxing  
    #   do 

    #     tmux_target="intervention_cart_pole_boxing_${method}_${seed}"

    #     tmux send-keys -t "$tmux_target" \
    #       "conda activate ${environment}; python -u method_comparison_intervention.py \
    #       --seed ${true_seed} \
    #       --environment_string ${env} \
    #       --training_timesteps ${training_timesteps[$env]} \
    #       --gold_timesteps ${gold_timesteps[$env]} \
    #       --num_concepts_selected ${num_concepts[$env]} \
    #       --method ${method} \
    #       --predictor_epochs 1 \
    #       --intervention_prob 0.0 \
    #       --out_folder intervention >> ../../runs/logs/error_${tmux_target}.txt 2>&1" ENTER

    #     tmux send-keys -t "$tmux_target" \
    #       "conda activate ${environment}; python -u method_comparison_intervention.py \
    #       --seed ${true_seed} \
    #       --environment_string ${env} \
    #       --training_timesteps ${training_timesteps[$env]} \
    #       --gold_timesteps ${gold_timesteps[$env]} \
    #       --num_concepts_selected ${num_concepts[$env]} \
    #       --method ${method} \
    #       --predictor_epochs 1 \
    #       --intervention_prob 0.5 \
    #       --out_folder intervention >> ../../runs/logs/error_${tmux_target}.txt 2>&1" ENTER
    #   done
    # done

    env=cart_pole
    # for intervention_prob in 0.25
    # do 
    #   for method in "${METHODS[@]}"; do

    #     tmux_target="intervention_cart_pole_boxing_${method}_${seed}"

    #     tmux send-keys -t "$tmux_target" \
    #       "conda activate ${environment}; python -u method_comparison_intervention.py \
    #       --seed ${true_seed} \
    #       --environment_string ${env} \
    #       --training_timesteps ${training_timesteps[$env]} \
    #       --gold_timesteps ${gold_timesteps[$env]} \
    #       --num_concepts_selected ${num_concepts[$env]} \
    #       --method ${method} \
    #       --intervention_prob ${intervention_prob} \
    #       --out_folder intervention >> ../../runs/logs/error_${tmux_target}.txt 2>&1" ENTER
    #   done
    # done

    for intervention_prob in 0.75
    do 
      for method in "${METHODS[@]}"; do

      tmux_target="intervention_mini_grid_pong_${method}_${seed}"

      tmux send-keys -t "$tmux_target" \
        "conda activate ${environment}; python -u method_comparison_intervention.py \
        --seed ${true_seed} \
        --environment_string ${env} \
        --training_timesteps ${training_timesteps[$env]} \
        --gold_timesteps ${gold_timesteps[$env]} \
        --num_concepts_selected ${num_concepts[$env]} \
        --method ${method} \
        --intervention_prob ${intervention_prob} \
        --out_folder intervention >> ../../runs/logs/error_${tmux_target}.txt 2>&1" ENTER
      done
    done
  done
done 
