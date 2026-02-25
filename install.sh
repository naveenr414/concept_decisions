#!/bin/bash
set -e

echo "Creating environment..."
conda env create -f environment.yaml

# Activate environment for the remaining steps
source $(conda info --base)/etc/profile.d/conda.sh
conda activate concept-selection

echo "Installing patched/special packages..."
# Fix for the simglucose bug you encountered
pip install simglucose --no-deps 

echo "Setup complete. Remember to set your GRB_LICENSE_FILE path!"