#!/bin/bash

# Example usage script for cornerplots.py
# This script demonstrates how to generate corner plots for different scenarios

# Set common parameters
GW_EVENT="GW190425"
BASE_DIR="../GW_runs/"

echo "Generating corner plots for $GW_EVENT..."

# Generate corner plots for each population type
python cornerplots.py --gw-event $GW_EVENT --population-type uniform --plot-bns --plot-hauke --convert-lambdas
python cornerplots.py --gw-event $GW_EVENT --population-type uniform --plot-bns --plot-hauke --no-convert-lambdas
python cornerplots.py --gw-event $GW_EVENT --population-type gaussian --plot-bns --plot-hauke --no-convert-lambdas
python cornerplots.py --gw-event $GW_EVENT --population-type gaussian --plot-bns --plot-hauke --convert-lambdas
python cornerplots.py --gw-event $GW_EVENT --population-type double_gaussian --plot-bns --plot-hauke --no-convert-lambdas
python cornerplots.py --gw-event $GW_EVENT --population-type double_gaussian --plot-bns --plot-hauke --convert-lambdas