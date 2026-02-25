#!/bin/bash
# reproduce_results.sh
#
# Reproduces all experimental results from:
#   "Selecting Decision-Relevant Concepts in Reinforcement Learning"
#
# STRUCTURE:
#   Step 1 — train_prerequisites.py
#             Trains base policies (pi*), Q-estimates, and concept predictors
#             for all environments × seeds. Safe to run once; skips cached work.
#
#   Step 2 — run_experiment.py (one call per config)
#             Runs the actual concept selection experiments. These are
#             independent of each other and can be parallelized:
#
#             Terminal 1: bash reproduce_results.sh --configs main_perfect main_imperfect
#             Terminal 2: bash reproduce_results.sh --configs intervention accuracy_sweep
#             Terminal 3: bash reproduce_results.sh --configs ablations timing cub
#
#             (Step 1 must complete before any Step 2 terminal is started.)
#
# Requirements:
#   - conda environment installed (see environment.yaml)
#   - Gurobi license at path set in GRB_LICENSE_FILE
#   - GPU recommended but not required
#
# Usage:
#   bash reproduce_results.sh                          # run everything
#   bash reproduce_results.sh --dry_run                # print commands only
#   bash reproduce_results.sh --resume                 # skip completed jobs
#   bash reproduce_results.sh --skip_prereqs           # skip step 1 (already done)
#   bash reproduce_results.sh --configs main_perfect   # run one config only

set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
CONDA_ENV="concept_abstraction"
GRB_LICENSE_FILE="${GRB_LICENSE_FILE:-/usr0/home/naveenr/gurobi.lic}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)/scripts"
NOTEBOOKS_DIR="${SCRIPT_DIR}/notebooks"

export GRB_LICENSE_FILE
export PYTHONWARNINGS=ignore
export GYMNASIUM_DISABLE_WARNINGS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
mkdir -p results/{imperfect,models,q_estimates,timing,cub,intervention,ablations,basic,training}
# ── Argument parsing ──────────────────────────────────────────────────────────
DRY_RUN=""
RESUME=""
SKIP_PREREQS=0
CONFIGS=(main_perfect main_imperfect intervention accuracy_sweep ablations timing cub)

while [[ $# -gt 0 ]]; do
  case $1 in
    --dry_run)      DRY_RUN="--dry_run"; shift ;;
    --resume)       RESUME="--resume";   shift ;;
    --skip_prereqs) SKIP_PREREQS=1;      shift ;;
    --configs)      shift; CONFIGS=("$@"); break ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

conda activate "${CONDA_ENV}" 2>/dev/null || source activate "${CONDA_ENV}"

# ── Step 1: Prerequisites ─────────────────────────────────────────────────────
if [[ $SKIP_PREREQS -eq 0 ]]; then
  echo ""
  echo "══════════════════════════════════════════"
  echo "  Step 1: Training prerequisites"
  echo "  (base policies, Q-estimates, concept predictors)"
  echo "══════════════════════════════════════════"
  python "${NOTEBOOKS_DIR}/train_prerequisites.py" ${DRY_RUN}
fi

# ── Step 2: Experiments ───────────────────────────────────────────────────────
for config_name in "${CONFIGS[@]}"; do
  config_path="${SCRIPT_DIR}/configs/${config_name}.yaml"
  if [[ ! -f "$config_path" ]]; then
    echo "Config not found: $config_path"
    exit 1
  fi

  echo ""
  echo "══════════════════════════════════════════"
  echo "  Step 2: Running ${config_name}"
  echo "══════════════════════════════════════════"
  python "${SCRIPT_DIR}/run_experiment.py" \
    --config "${config_path}" \
    ${DRY_RUN} ${RESUME}
done

echo ""
echo "All experiments complete. Results written to results/."
echo "Next: open plot_results.ipynb to generate figures."