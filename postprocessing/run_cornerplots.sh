#!/bin/bash

# Example usage script for cornerplots.py
# This script demonstrates how to generate corner plots for different scenarios

# Set common parameters
BASE_DIR="../GW_runs/"

GW_EVENT="GW190425"
echo "Generating corner plots for $GW_EVENT..."
python cornerplots.py --gw-event $GW_EVENT --population-type uniform --plot-all-params --plot-bns --plot-hauke --convert-lambdas
python cornerplots.py --gw-event $GW_EVENT --population-type uniform --plot-all-params --plot-bns --plot-hauke --no-convert-lambdas
python cornerplots.py --gw-event $GW_EVENT --population-type gaussian --plot-all-params --plot-bns --plot-hauke --no-convert-lambdas
python cornerplots.py --gw-event $GW_EVENT --population-type gaussian --plot-all-params --plot-bns --plot-hauke --convert-lambdas
python cornerplots.py --gw-event $GW_EVENT --population-type double_gaussian --plot-all-params --plot-bns --plot-hauke --no-convert-lambdas
python cornerplots.py --gw-event $GW_EVENT --population-type double_gaussian --plot-all-params --plot-bns --plot-hauke --convert-lambdas

GW_EVENT="GW170817"
echo "Generating corner plots for $GW_EVENT..."
python cornerplots.py --gw-event $GW_EVENT --population-type uniform --plot-all-params --plot-bns --plot-hauke --convert-lambdas
python cornerplots.py --gw-event $GW_EVENT --population-type uniform --plot-all-params --plot-bns --plot-hauke --no-convert-lambdas
python cornerplots.py --gw-event $GW_EVENT --population-type gaussian --plot-all-params --plot-bns --plot-hauke --no-convert-lambdas
python cornerplots.py --gw-event $GW_EVENT --population-type gaussian --plot-all-params --plot-bns --plot-hauke --convert-lambdas
python cornerplots.py --gw-event $GW_EVENT --population-type double_gaussian --plot-all-params --plot-bns --plot-hauke --no-convert-lambdas
python cornerplots.py --gw-event $GW_EVENT --population-type double_gaussian --plot-all-params --plot-bns --plot-hauke --convert-lambdas