#!/bin/bash

# Common parameters for CouplingNSF (existing flows)
COMMON_ARGS_NSF="--batch-size 1024 --learning-rate 5e-4 --max-patience 250 --num-epochs 2000 --n-neurons 256 --n-blocks-per-transform 6 --num-bins 16 --N-samples-training 50000"

# Conservative parameters for Masked Autoregressive Flows (to prevent overfitting)
COMMON_ARGS_MAF="--glasflow-type MaskedPiecewiseRationalQuadraticAutoregressiveFlow --batch-size 256 --learning-rate 5e-4 --max-patience 200 --num-epochs 1500 --n-neurons 128 --n-blocks-per-transform 3 --num-bins 8 --n-transforms 3 --N-samples-training 50000"

# Arrays for iteration
POPULATIONS=("uniform" "gaussian" "double_gaussian")
SOURCES=("bns" "nsbh")
EOS_SAMPLES=("radio" "radio_chiEFT" "radio_chiEFT_NICER")

echo "Submitting Masked Autoregressive Flow training jobs..."
for pop in "${POPULATIONS[@]}"; do
    for src in "${SOURCES[@]}"; do
        for eos in "${EOS_SAMPLES[@]}"; do
            echo "Submitting: $pop $src $eos (MAF)"
            python train_NF_prior.py --submit --population-type $pop --source-type $src --eos-samples-name $eos $COMMON_ARGS_MAF
        done
    done
done