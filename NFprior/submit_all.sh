#!/bin/bash

# Common parameters for CouplingNSF architecture and training methods
COMMON_ARGS="--submit --batch-size 1024 --learning-rate 1e-4 --max-patience 100 --num-epochs 2000 --n-neurons 128 --n-blocks-per-transform 2 --num-bins 4 --N-samples-training 200000 --batch-norm-within-blocks --linear-transform permutation"

# Arrays for iteration
POPULATIONS=("uniform" "gaussian" "double_gaussian")
SOURCES=("bns" "nsbh")
EOS_SAMPLES=("radio" "radio_chiEFT" "radio_chiEFT_NICER")

echo "Submitting flow training jobs..."
for pop in "${POPULATIONS[@]}"; do
    for src in "${SOURCES[@]}"; do
        for eos in "${EOS_SAMPLES[@]}"; do
            echo "Submitting: $pop $src $eos"
            python train_NF_prior.py --population-type $pop --source-type $src --eos-samples-name $eos $COMMON_ARGS
        done
    done
done