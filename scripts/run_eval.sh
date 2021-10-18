#!/usr/bin/bash

source activate FACIL

# Method to evaluate
method="$1"
num_tasks=10
num_workers=1

# Defining output folder names
bic_models="./data/bic_models_${method}_s${num_tasks}"
logits_dir="./data/logits_${method}_s${num_tasks}"

## Evaluating the bias-corrected models
# =============================================================================

printf "\n\n"
echo "Evaluating the models..."
echo "===================================================="

PYTHONPATH=./src/ \
python -u ./src/calibrate_bic.py \
        --batch-size 128 \
        --calib-method forward-all \
        --datasets cif100 \
        --gpu 0 \
        --logits-dir $logits_dir \
        --bic-models-dir $bic_models \
        --num-tasks $num_tasks \
        --num-workers $num_workers \
        --seed 0
