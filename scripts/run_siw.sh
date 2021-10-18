#!/usr/bin/bash

source activate FACIL

# Defining reference and target datasets to use
dset_ref="$1"
dset_target="$2"
num_tasks="$3"

# Number of worker loaders
num_workers=8

# Number of epochs
num_epochs_start=300 # -> First (non-incremental) state
num_epochs=70        # -> Subsequent states

# Defining output folder names
str_ref="siw_${dset_ref}_s${num_tasks}"
str_target="siw_${dset_target}_s${num_tasks}"

dir_ref="./output/${str_ref}"
dir_target="./output/${str_target}"

# Creating output folders
mkdir -p "${dir_ref}/models/"
mkdir -p "${dir_target}/models/"

mkdir -p "${dir_ref}/bic_models/"
mkdir -p "${dir_target}/bic_models/"

mkdir -p "${dir_ref}/logits/"
mkdir -p "${dir_target}/logits/"



## 1) Training the reference and target models
# =============================================================================


# -> Reference model training
printf "\n\n"
echo "Training the reference model..."
echo "===================================================="

PYTHONPATH=./src/ \
python3 -u ./src/basic_step.py \
        --batch-size 128 \
        --datasets $dset_ref \
        --gpu 0 \
        --lr 0.1 \
        --lr_decay 0.1 \
        --models-dir $dir_ref/models/ \
        --momentum 0.9 \
        --nepochs $num_epochs_start \
        --num-tasks $num_tasks \
        --num-workers $num_workers \
        --patience 60 \
        --seed 0 \
        --weight-decay 0.0005

PYTHONPATH=./src/ \
python3 -u ./src/finetuning.py \
        --batch-size 128 \
        --datasets $dset_ref \
        --gpu 0 \
        --lr 0.1 \
        --lr_decay 0.1 \
        --models-dir $dir_ref/models/ \
        --momentum 0.9 \
        --nepochs $num_epochs \
        --num-tasks $num_tasks \
        --num-workers $num_workers \
        --patience 15 \
        --seed 0 \
        --weight-decay 0.0005

# -> Target model training
printf "\n\n"
echo "Training the target model..."
echo "===================================================="

PYTHONPATH=./src/ \
python3 -u ./src/basic_step.py \
        --batch-size 128 \
        --datasets $dset_target \
        --gpu 0 \
        --lr 0.1 \
        --lr_decay 0.1 \
        --models-dir $dir_target/models/ \
        --momentum 0.9 \
        --nepochs $num_epochs_start \
        --num-tasks $num_tasks \
        --num-workers $num_workers \
        --patience 60 \
        --seed 0 \
        --weight-decay 0.0005

PYTHONPATH=./src/ \
python3 -u ./src/finetuning.py \
        --batch-size 128 \
        --datasets $dset_target \
        --gpu 0 \
        --lr 0.1 \
        --lr_decay 0.1 \
        --models-dir $dir_target/models/ \
        --momentum 0.9 \
        --nepochs $num_epochs \
        --num-tasks $num_tasks \
        --num-workers $num_workers \
        --patience 15 \
        --seed 0 \
        --weight-decay 0.0005

printf "\n\n"
echo "Reference and target models trained!"


## 2) Extracting logits for reference and target models
# =============================================================================


printf "\n\n"
echo "Extracting reference logits..."
echo "===================================================="

PYTHONPATH=./src/ \
python -u ./src/extract_logits.py \
        --batch-size 128 \
        --datasets $dset_ref \
        --gpu 0 \
        --logits-outdir $dir_ref/logits/ \
        --models-dir $dir_ref/models/ \
        --models-base-name task \
        --model-type siw \
        --num-tasks $num_tasks \
        --num-workers $num_workers \
        --seed 0


printf "\n\n"
echo "Extracting target logits..."
echo "===================================================="

PYTHONPATH=./src/ \
python -u ./src/extract_logits.py \
        --batch-size 128 \
        --datasets $dset_target \
        --gpu 0 \
        --logits-outdir $dir_target/logits/ \
        --models-dir $dir_target/models/ \
        --models-base-name task \
        --model-type siw \
        --num-tasks $num_tasks \
        --num-workers $num_workers \
        --seed 0

printf "\n\n"
echo "Reference and target logits extracted!"


## 3) Calibrating adBiC layer on the reference dataset
# =============================================================================


printf "\n\n"
echo "Calibrating reference adBiC layer..."
echo "===================================================="

PYTHONPATH=./src/ \
python -u ./src/calibrate_bic.py \
        --batch-size 128 \
        --calib-method adaptive \
        --datasets $dset_ref \
        --gpu 0 \
        --logits-dir $dir_ref \
        --bic-models-outdir $dir_ref/bic_models \
        --nepochs 200 \
        --num-tasks $num_tasks \
        --num-workers $num_workers \
        --seed 0


## 4) Transferring adBiC parameters to target datasets
# =============================================================================


printf "\n\n"
echo "Transferring adBiC parameters to target model..."
echo "===================================================="

PYTHONPATH=./src/ \
python -u ./src/calibrate_bic.py \
        --batch-size 128 \
        --calib-method forward-all \
        --datasets $dset_target \
        --gpu 0 \
        --logits-dir $dir_target \
        --bic-models-dir $dir_ref/bic_models \
        --num-tasks $num_tasks \
        --num-workers $num_workers \
        --seed 0


# Clearing empty folders
find ./output/ -type d -empty -delete
