#!/bin/bash

# Common parameters for CouplingNSF architecture and training methods
COMMON_ARGS="--submit --batch-size 1024 --learning-rate 1e-4 --max-patience 250 --num-epochs 2000 --n-neurons 256 --n-blocks-per-transform 3 --num-bins 8 --N-samples-training 200000"

# Arrays for iteration
POPULATIONS=("uniform" "gaussian" "double_gaussian")
SOURCES=("bns" "nsbh")
EOS_SAMPLES=("radio" "radio_chiEFT" "radio_NICER")

echo "Submitting flow training jobs..."
for pop in "${POPULATIONS[@]}"; do
    for src in "${SOURCES[@]}"; do
        for eos in "${EOS_SAMPLES[@]}"; do
            echo "Submitting: $pop $src $eos"
            python train_NF_prior.py --population-type $pop --source-type $src --eos-samples-name $eos $COMMON_ARGS
        done
    done
done
