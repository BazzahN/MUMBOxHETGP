#!/bin/bash

# Quick start script for hetGPy experiments
# This demonstrates how to generate data, fit models, and save results

set -e  # Exit on error

echo "======================================"
echo "hetGPy Experiment Quick Start"
echo "======================================"
echo ""

# Configuration
CONFIG_FILE="configs/exp_k25_n2.yml"
NUM_MACROS=5

# Check if config exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    echo "Please create the config file first."
    exit 1
fi

echo "Configuration: $CONFIG_FILE"
echo "Number of data replicates: $NUM_MACROS"
echo ""

# Step 1: Generate Data
# echo "Step 1: Generating synthetic experimental data..."
# echo "Command: python hetgp_input_generation.py --config $CONFIG_FILE --n_macros $NUM_MACROS"
# python hetgp_input_generation.py --config "$CONFIG_FILE" --n_macros "$NUM_MACROS"
# echo "✓ Data generation complete!"
# echo ""

# Step 2: Fit Models
echo "Step 2: Fitting heteroscedastic GP models..."
EXP_NAME=$(basename "$CONFIG_FILE" .yml)

for macro_idx in $(seq 0 $((NUM_MACROS - 1))); do
    echo "  Fitting macro replicate $macro_idx..."
    conda run -n hetGP_env python -u hetgp_exp_script.py --config "$CONFIG_FILE" --macro "$macro_idx"
done
echo "✓ Model fitting complete!"
echo ""

# Step 3: Display results summary
echo "======================================"
echo "Results Summary"
echo "======================================"
echo "Experiment: $EXP_NAME"
echo "Data directory: ${EXP_NAME}/Input/"
echo "Results directory: ${EXP_NAME}/Data/"
echo ""
echo "Generated files:"
echo "  - Input data: train_x_m*.npy, train_y_m*.npy (standardized)"
echo "  - Test data: test_x.npy, test_y.npy, test_sigma2.npy"
echo "  - Predictions: pred_f_m*.npy, pred_sigma2_f_m*.npy, pred_sigma2_eps_m*.npy"
echo "  - Predictions (original scale): pred_*_unstd_m*.npy"
echo "  - Hyperparameters: hyperparameters_m*.json"
echo ""
echo "Next steps:"
echo "  1. Open data_fit_hetgpy.ipynb for visualization and analysis"
echo "  2. Or run: jupyter notebook data_fit_hetgpy.ipynb"
echo ""
echo "======================================"
echo "✓ Quick start complete!"
echo "======================================"
