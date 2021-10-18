#!/usr/bin/bash

source activate iCaRL

# Defining reference and target datasets to use
dset_ref="$1"
dset_target="$2"
num_tasks="$3"

# Number of worker loaders
num_workers=8

# Number of epochs
num_epochs=70

# Defining output folder names
str_ref="lwf_${dset_ref}_s${num_tasks}"
str_target="lwf_${dset_target}_s${num_tasks}"

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

PYTHONPATH=./src/lwf \
python -u ./src/lwf/lwf.py \
        --batch-size 128 \
        --datasets $dset_ref \
        --lr 1.0 \
        --lr-factor 5.0 \
        --lr-strat 20 30 40 50 \
        --models-dir $dir_ref/models/ \
        --nepochs $num_epochs \
        --num-tasks $num_tasks \
        --seed 0 \
        --weight-decay 1e-05


# -> Target model training
printf "\n\n"
echo "Training the target model..."
echo "===================================================="

PYTHONPATH=./src/lwf \
python -u ./src/lwf/lwf.py \
        --batch-size 128 \
        --datasets $dset_target \
        --lr 1.0 \
        --lr-factor 5.0 \
        --lr-strat 20 30 40 50 \
        --models-dir $dir_target/models/ \
        --nepochs $num_epochs \
        --num-tasks $num_tasks \
        --seed 0 \
        --weight-decay 1e-05

printf "\n\n"
echo "Reference and target models trained!"



## 2) Extracting logits for reference and target models
# =============================================================================

printf "\n\n"
echo "Extracting reference logits..."
echo "===================================================="

PYTHONPATH=./src/ \
python -u ./src/extract_logits_lwf.py \
        --batch-size 128 \
        --datasets $dset_ref \
        --gpu 0 \
        --logits-outdir $dir_ref/logits/ \
        --models-dir $dir_ref/models/ \
        --num-tasks $num_tasks


printf "\n\n"
echo "Extracting target logits..."
echo "===================================================="

PYTHONPATH=./src/ \
python -u ./src/extract_logits_lwf.py \
        --batch-size 128 \
        --datasets $dset_target \
        --gpu 0 \
        --logits-outdir $dir_target/logits/ \
        --models-dir $dir_target/models/ \
        --num-tasks $num_tasks

printf "\n\n"
echo "Reference and target logits extracted!"


source activate FACIL

# Converting logits from Numpy to Torch format

PYTHONPATH=./src/ \
python -u ./src/convert_logits.py \
        --logits-dir $dir_ref/logits/ \
        --num-tasks $num_tasks

PYTHONPATH=./src/ \
python -u ./src/convert_logits.py \
        --logits-dir $dir_target/logits/ \
        --num-tasks $num_tasks


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
