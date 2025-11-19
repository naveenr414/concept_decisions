#!/bin/bash
# Create log files for each environment
: > runs/logs/error_imperfect_mini_grid.txt
: > runs/logs/error_imperfect_cart_pole.txt
: > runs/logs/error_imperfect_pong.txt
: > runs/logs/error_imperfect_boxing.txt

LOGFILE_MINI_GRID=../../runs/logs/error_imperfect_mini_grid.txt
LOGFILE_CART_POLE=../../runs/logs/error_imperfect_cart_pole.txt
LOGFILE_PONG=../../runs/logs/error_imperfect_pong.txt
LOGFILE_BOXING=../../runs/logs/error_imperfect_boxing.txt

environment=food

# Define timesteps and concepts for each environment
declare -A gold_timesteps=(
  [mini_grid]=1000000
  [cart_pole]=4000000
  [pong]=15000000
  [boxing]=15000000
)

declare -A training_timesteps=(
  [mini_grid]=1000000
  [cart_pole]=4000000
  [pong]=15000000
  [boxing]=15000000
)

declare -A num_concepts=(
  [mini_grid]=11
  [cart_pole]=3
  [pong]=57
  [boxing]=48
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
setup_tmux_session "imperfect_mini_grid"
# setup_tmux_session "imperfect_cart_pole"
# setup_tmux_session "imperfect_pong"
# setup_tmux_session "imperfect_boxing"

# Run experiments
for seed in 42
do 
  # Mini Grid - vary concept_accuracy and num_concepts_selected
  env=mini_grid
  for concept_accuracy in 0.75 0.85 0.975
  do
    for concept_fraction in 0.25 0.5 1
    do
      num_concepts_selected=$(printf "%.0f" $(echo "${num_concepts[$env]} * ${concept_fraction}" | bc))
      tmux send-keys -t imperfect_mini_grid "conda activate ${environment}; python -u only_lp.py --seed ${seed} --environment_string ${env} --training_timesteps ${training_timesteps[$env]} --gold_timesteps ${gold_timesteps[$env]} --num_concepts_selected ${num_concepts_selected} --concept_accuracy ${concept_accuracy} --out_folder imperfect >> ${LOGFILE_MINI_GRID} 2>&1" ENTER
    done
  done

  # Mini Grid - vary concept_accuracy and training_timesteps (fix num_concepts_selected)
  for concept_accuracy in 0.75 0.85 0.975
  do
    for training_ts in 250000 500000 750000 1000000
    do
      tmux send-keys -t imperfect_mini_grid "conda activate ${environment}; python -u only_lp.py --seed ${seed} --environment_string ${env} --training_timesteps ${training_ts} --gold_timesteps ${gold_timesteps[$env]} --num_concepts_selected ${num_concepts[$env]} --concept_accuracy ${concept_accuracy} --out_folder imperfect >> ${LOGFILE_MINI_GRID} 2>&1" ENTER
    done
  done

  # Mini Grid - vary num_concepts_selected and training_timesteps (fix concept_accuracy=0.95)
  for concept_fraction in 0.25 0.5 1
  do
    num_concepts_selected=$(printf "%.0f" $(echo "${num_concepts[$env]} * ${concept_fraction}" | bc))
    for training_ts in 250000 500000 750000 1000000
    do
      tmux send-keys -t imperfect_mini_grid "conda activate ${environment}; python -u only_lp.py --seed ${seed} --environment_string ${env} --training_timesteps ${training_ts} --gold_timesteps ${gold_timesteps[$env]} --num_concepts_selected ${num_concepts_selected} --concept_accuracy 0.95 --out_folder imperfect >> ${LOGFILE_MINI_GRID} 2>&1" ENTER
    done
  done

  # # Cart Pole - vary concept_accuracy and num_concepts_selected
  # env=cart_pole
  # for concept_accuracy in 0.75 0.85 0.975
  # do
  #   for concept_fraction in 0.25 0.5 1
  #   do
  #     num_concepts_selected=$(printf "%.0f" $(echo "${num_concepts[$env]} * ${concept_fraction}" | bc))
  #     tmux send-keys -t imperfect_cart_pole "conda activate ${environment}; python -u only_lp.py --seed ${seed} --environment_string ${env} --training_timesteps ${training_timesteps[$env]} --gold_timesteps ${gold_timesteps[$env]} --num_concepts_selected ${num_concepts_selected} --concept_accuracy ${concept_accuracy} --out_folder imperfect >> ${LOGFILE_CART_POLE} 2>&1" ENTER
  #   done
  # done

  # # Pong - vary concept_accuracy and num_concepts_selected
  # env=pong
  # for concept_accuracy in 0.75 0.85 0.975
  # do
  #   for concept_fraction in 0.25 0.5 1
  #   do
  #     num_concepts_selected=$(printf "%.0f" $(echo "${num_concepts[$env]} * ${concept_fraction}" | bc))
  #     tmux send-keys -t imperfect_pong "conda activate ${environment}; python -u only_lp.py --seed ${seed} --environment_string ${env} --training_timesteps ${training_timesteps[$env]} --gold_timesteps ${gold_timesteps[$env]} --num_concepts_selected ${num_concepts_selected} --concept_accuracy ${concept_accuracy} --out_folder imperfect >> ${LOGFILE_PONG} 2>&1" ENTER
  #   done
  # done

  # # Boxing - vary concept_accuracy and num_concepts_selected
  # env=boxing
  # for concept_accuracy in 0.75 0.85 0.975
  # do
  #   for concept_fraction in 0.25 0.5 1
  #   do
  #     num_concepts_selected=$(printf "%.0f" $(echo "${num_concepts[$env]} * ${concept_fraction}" | bc))
  #     tmux send-keys -t imperfect_boxing "conda activate ${environment}; python -u only_lp.py --seed ${seed} --environment_string ${env} --training_timesteps ${training_timesteps[$env]} --gold_timesteps ${gold_timesteps[$env]} --num_concepts_selected ${num_concepts_selected} --concept_accuracy ${concept_accuracy} --out_folder imperfect >> ${LOGFILE_BOXING} 2>&1" ENTER
  #   done
  # done
done