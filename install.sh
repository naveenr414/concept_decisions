#!/bin/bash
set -e

echo "Creating environment..."
conda env create -f environment.yaml

source $(conda info --base)/etc/profile.d/conda.sh
conda activate concept-selection

echo "Installing patched/special packages..."
pip install simglucose --no-deps 
pip install -e .

echo "Setup complete. Remember to set your GRB_LICENSE_FILE path!"
echo "Run conda activate concept-selection to activate the environment"