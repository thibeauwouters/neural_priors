#!/bin/bash

# Example usage script for get_bayes_factors.py
# This script demonstrates how to calculate Bayes factors for different scenarios

# Set common parameters
GW_EVENT="GW190425"
BASE_DIR="../GW_runs/"

echo "Calculating Bayes factors for $GW_EVENT..."

# Calculate Bayes factors for each population type
echo "=== Uniform population ==="
python get_bayes_factors.py --gw-event $GW_EVENT --population-type uniform --base-dir $BASE_DIR