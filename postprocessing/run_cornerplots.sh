#!/bin/bash

# Example usage script for cornerplots.py
# This script demonstrates how to generate corner plots for different scenarios

# Set common parameters
GW_EVENT="GW170817"
BASE_DIR="../GW_runs/"

echo "Generating corner plots for $GW_EVENT..."

# Generate corner plots for each population type
echo "=== Uniform population ==="
python cornerplots.py --gw-event $GW_EVENT --population-type uniform --plot-default --convert-lambdas
python cornerplots.py --gw-event $GW_EVENT --population-type uniform --plot-default --no-convert-lambdas

echo "=== Gaussian population ==="
python cornerplots.py --gw-event $GW_EVENT --population-type gaussian --plot-default --convert-lambdas
python cornerplots.py --gw-event $GW_EVENT --population-type gaussian --plot-default --no-convert-lambdas

echo "=== Double Gaussian population ==="
python cornerplots.py --gw-event $GW_EVENT --population-type double_gaussian --plot-default --convert-lambdas
python cornerplots.py --gw-event $GW_EVENT --population-type double_gaussian --plot-default --no-convert-lambdas

# Example with all parameters plotted
echo "=== Generating detailed plots with all parameters ==="
python cornerplots.py --gw-event $GW_EVENT --population-type uniform --plot-default --plot-all-params --convert-lambdas

# Example with external comparison data (if available)
echo "=== Generating plots with external comparisons ==="
python cornerplots.py --gw-event $GW_EVENT --population-type uniform --plot-default --plot-hauke --plot-adrian --convert-lambdas

echo "Corner plot generation complete!"