#!/bin/bash
# Create log files for each environment
sessions=(
  perfect_perfect_concepts
  perfect_random
  perfect_entropy
  perfect_greedy
  perfect_lp
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

declare -A training_timesteps=(
  [mini_grid]=250000
  [cart_pole]=250000 #4000000
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

  for env in mini_grid cart_pole pong boxing glucose 
  do 
    for concept_accuracy in 0.75 0.85 0.95
    do
      for concept_fraction in 0.25 0.5 1
      do
        case "$concept_accuracy" in
          0.75) tmux_target="perfect_perfect_concepts" ;;
          0.85) tmux_target="perfect_random" ;;
          0.95) tmux_target="perfect_entropy" ;;
          *) echo "Unknown concept_accuracy $concept_accuracy"; exit 1 ;;
        esac

        num_concepts_selected=$(printf "%.0f" \
          $(echo "${num_concepts[$env]} * ${concept_fraction}" | bc))

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

  # env=mini_grid
  # for concept_accuracy in 0.75 0.85 0.95
  # do
  #   for training_ts in 250000 500000 750000 1000000
  #   do
  #     tmux send-keys -t perfect_greedy "conda activate ${environment}; python -u only_multiple.py --seed ${seed} --environment_string ${env} --training_timesteps ${training_ts} --gold_timesteps ${gold_timesteps[$env]} --num_concepts_selected ${num_concepts[$env]} --concept_accuracy ${concept_accuracy} --out_folder imperfect >> ../../runs/logs/error_perfect_greedy.txt 2>&1" ENTER
  #   done
  # done

  # # # Mini Grid - vary num_concepts_selected and training_timesteps (fix concept_accuracy=0.95)
  # for concept_fraction in 0.25 0.5 1
  # do
  #   num_concepts_selected=$(printf "%.0f" $(echo "${num_concepts[$env]} * ${concept_fraction}" | bc))
  #   for training_ts in 250000 500000 750000 1000000
  #   do
  #     tmux send-keys -t perfect_lp "conda activate ${environment}; python -u only_multiple.py --seed ${seed} --environment_string ${env} --training_timesteps ${training_ts} --gold_timesteps ${gold_timesteps[$env]} --num_concepts_selected ${num_concepts_selected} --concept_accuracy 0.95 --out_folder imperfect >> ../../runs/logs/error_perfect_lp.txt 2>&1" ENTER
  #   done
  # done
done