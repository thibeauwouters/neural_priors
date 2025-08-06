#!/bin/bash

# Improved hyperparameters for better flowjax BNAF performance:
# --nn-depth 4 --nn-block-dim 24 --flow-layers 4 --learning-rate 1e-3 --max-patience 250 --num-epochs 2000

python train_NF_prior.py --submit --use-flowjax --population-type uniform --source-type bns --eos-samples-name radio --nn-depth 4 --nn-block-dim 24 --flow-layers 4 --learning-rate 1e-3 --max-patience 250 --num-epochs 2000
python train_NF_prior.py --submit --use-flowjax --population-type uniform --source-type bns --eos-samples-name radio_chiEFT --nn-depth 4 --nn-block-dim 24 --flow-layers 4 --learning-rate 1e-3 --max-patience 250 --num-epochs 2000
python train_NF_prior.py --submit --use-flowjax --population-type uniform --source-type bns --eos-samples-name radio_chiEFT_NICER --nn-depth 4 --nn-block-dim 24 --flow-layers 4 --learning-rate 1e-3 --max-patience 250 --num-epochs 2000
python train_NF_prior.py --submit --use-flowjax --population-type gaussian --source-type bns --eos-samples-name radio --nn-depth 4 --nn-block-dim 24 --flow-layers 4 --learning-rate 1e-3 --max-patience 250 --num-epochs 2000
python train_NF_prior.py --submit --use-flowjax --population-type gaussian --source-type bns --eos-samples-name radio_chiEFT --nn-depth 4 --nn-block-dim 24 --flow-layers 4 --learning-rate 1e-3 --max-patience 250 --num-epochs 2000
python train_NF_prior.py --submit --use-flowjax --population-type gaussian --source-type bns --eos-samples-name radio_chiEFT_NICER --nn-depth 4 --nn-block-dim 24 --flow-layers 4 --learning-rate 1e-3 --max-patience 250 --num-epochs 2000
python train_NF_prior.py --submit --use-flowjax --population-type double_gaussian --source-type bns --eos-samples-name radio --nn-depth 4 --nn-block-dim 24 --flow-layers 4 --learning-rate 1e-3 --max-patience 250 --num-epochs 2000
python train_NF_prior.py --submit --use-flowjax --population-type double_gaussian --source-type bns --eos-samples-name radio_chiEFT --nn-depth 4 --nn-block-dim 24 --flow-layers 4 --learning-rate 1e-3 --max-patience 250 --num-epochs 2000
python train_NF_prior.py --submit --use-flowjax --population-type double_gaussian --source-type bns --eos-samples-name radio_chiEFT_NICER --nn-depth 4 --nn-block-dim 24 --flow-layers 4 --learning-rate 1e-3 --max-patience 250 --num-epochs 2000
python train_NF_prior.py --submit --use-flowjax --population-type uniform --source-type nsbh --eos-samples-name radio --nn-depth 4 --nn-block-dim 24 --flow-layers 4 --learning-rate 1e-3 --max-patience 250 --num-epochs 2000
python train_NF_prior.py --submit --use-flowjax --population-type uniform --source-type nsbh --eos-samples-name radio_chiEFT --nn-depth 4 --nn-block-dim 24 --flow-layers 4 --learning-rate 1e-3 --max-patience 250 --num-epochs 2000
python train_NF_prior.py --submit --use-flowjax --population-type uniform --source-type nsbh --eos-samples-name radio_chiEFT_NICER --nn-depth 4 --nn-block-dim 24 --flow-layers 4 --learning-rate 1e-3 --max-patience 250 --num-epochs 2000
python train_NF_prior.py --submit --use-flowjax --population-type gaussian --source-type nsbh --eos-samples-name radio --nn-depth 4 --nn-block-dim 24 --flow-layers 4 --learning-rate 1e-3 --max-patience 250 --num-epochs 2000
python train_NF_prior.py --submit --use-flowjax --population-type gaussian --source-type nsbh --eos-samples-name radio_chiEFT --nn-depth 4 --nn-block-dim 24 --flow-layers 4 --learning-rate 1e-3 --max-patience 250 --num-epochs 2000
python train_NF_prior.py --submit --use-flowjax --population-type gaussian --source-type nsbh --eos-samples-name radio_chiEFT_NICER --nn-depth 4 --nn-block-dim 24 --flow-layers 4 --learning-rate 1e-3 --max-patience 250 --num-epochs 2000
python train_NF_prior.py --submit --use-flowjax --population-type double_gaussian --source-type nsbh --eos-samples-name radio --nn-depth 4 --nn-block-dim 24 --flow-layers 4 --learning-rate 1e-3 --max-patience 250 --num-epochs 2000
python train_NF_prior.py --submit --use-flowjax --population-type double_gaussian --source-type nsbh --eos-samples-name radio_chiEFT --nn-depth 4 --nn-block-dim 24 --flow-layers 4 --learning-rate 1e-3 --max-patience 250 --num-epochs 2000
python train_NF_prior.py --submit --use-flowjax --population-type double_gaussian --source-type nsbh --eos-samples-name radio_chiEFT_NICER --nn-depth 4 --nn-block-dim 24 --flow-layers 4 --learning-rate 1e-3 --max-patience 250 --num-epochs 2000


# ### GW170817 ###
# python train_NF_prior.py --submit --use-flowjax --population-type GW170817 --source-type bns --eos-samples-name radio
# python train_NF_prior.py --submit --use-flowjax --population-type GW170817 --source-type bns --eos-samples-name radio_chiEFT
# python train_NF_prior.py --submit --use-flowjax --population-type GW170817 --source-type bns --eos-samples-name radio_chiEFT_NICER
# python train_NF_prior.py --submit --use-flowjax --population-type GW170817 --source-type nsbh --eos-samples-name radio
# python train_NF_prior.py --submit --use-flowjax --population-type GW170817 --source-type nsbh --eos-samples-name radio_chiEFT
# python train_NF_prior.py --submit --use-flowjax --population-type GW170817 --source-type nsbh --eos-samples-name radio_chiEFT_NICER

# ### GW190425 ###
# python train_NF_prior.py --submit --use-flowjax --population-type GW190425 --source-type bns --eos-samples-name radio
# python train_NF_prior.py --submit --use-flowjax --population-type GW190425 --source-type bns --eos-samples-name radio_chiEFT
# python train_NF_prior.py --submit --use-flowjax --population-type GW190425 --source-type bns --eos-samples-name radio_chiEFT_NICER
# python train_NF_prior.py --submit --use-flowjax --population-type GW190425 --source-type nsbh --eos-samples-name radio
# python train_NF_prior.py --submit --use-flowjax --population-type GW190425 --source-type nsbh --eos-samples-name radio_chiEFT
# python train_NF_prior.py --submit --use-flowjax --population-type GW190425 --source-type nsbh --eos-samples-name radio_chiEFT_NICER

# ### GW230529 ###
# python train_NF_prior.py --submit --use-flowjax --population-type GW230529 --source-type bns --eos-samples-name radio
# python train_NF_prior.py --submit --use-flowjax --population-type GW230529 --source-type bns --eos-samples-name radio_chiEFT
# python train_NF_prior.py --submit --use-flowjax --population-type GW230529 --source-type bns --eos-samples-name radio_chiEFT_NICER
# python train_NF_prior.py --submit --use-flowjax --population-type GW230529 --source-type nsbh --eos-samples-name radio
# python train_NF_prior.py --submit --use-flowjax --population-type GW230529 --source-type nsbh --eos-samples-name radio_chiEFT
# python train_NF_prior.py --submit --use-flowjax --population-type GW230529 --source-type nsbh --eos-samples-name radio_chiEFT_NICER