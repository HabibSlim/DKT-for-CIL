#!/usr/bin/bash

source activate FACIL

# Defining reference and target datasets to use
dset_ref="mock_ref"
dset_target="mock_target"
num_tasks="2"

str_ref="lucir_${dset_ref}_s${num_tasks}"
str_target="lucir_${dset_target}_s${num_tasks}"

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

PYTHONPATH=./src/lucir \
python -u ./src/lucir/lucir.py \
        --adapt-lambda True \
        --batch-size 128 \
        --datasets $dset_ref \
        --lambd 5.0 \
        --less-forget True \
        --lr 0.1 \
        --lr-factor 0.1 \
        --lr-strat 80 120 \
        --momentum 0.9 \
        --models-dir $dir_ref/models/ \
        --nepochs 1 \
        --num-classes 5 \
        --num-classes-total 10 \
        --num-tasks $num_tasks \
        --num-workers 8 \
        --seed 0 \
        --weight-decay 0.0001


# -> Target model training
printf "\n\n"
echo "Training the target model..."
echo "===================================================="
PYTHONPATH=./src/lucir \
python -u ./src/lucir/lucir.py \
        --adapt-lambda True \
        --batch-size 128 \
        --datasets $dset_target \
        --lambd 5.0 \
        --less-forget True \
        --lr 0.1 \
        --lr-factor 0.1 \
        --lr-strat 80 120 \
        --momentum 0.9 \
        --models-dir $dir_target/models/ \
        --nepochs 1 \
        --num-classes 5 \
        --num-classes-total 10 \
        --num-tasks $num_tasks \
        --num-workers 8 \
        --seed 0 \
        --weight-decay 0.0001 

printf "\n\n"
echo "Reference and target models trained!"



## 2) Extracting logits for reference and target models
# =============================================================================

printf "\n\n"
echo "Extracting reference logits..."
echo "===================================================="

PYTHONPATH=./src/ \
python -u ./src/extract_logits_lucir.py \
        --batch-size 128 \
        --datasets $dset_ref \
        --gpu 0 \
        --logits-outdir $dir_ref/logits/ \
        --models-dir $dir_ref/models/ \
        --num-tasks $num_tasks \
        --num-workers 8 \
        --seed 0


printf "\n\n"
echo "Extracting target logits..."
echo "===================================================="

PYTHONPATH=./src/ \
python -u ./src/extract_logits_lucir.py \
        --batch-size 128 \
        --datasets $dset_target \
        --gpu 0 \
        --logits-outdir $dir_target/logits/ \
        --models-dir $dir_target/models/ \
        --num-tasks $num_tasks \
        --num-workers 8 \
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
        --nepochs 1 \
        --num-tasks $num_tasks \
        --num-workers 8 \
        --seed 1337



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
        --num-workers 8 \
        --seed 0


echo "Debugging script done!"

# Clearing empty folders
find ./output/ -type d -empty -delete
