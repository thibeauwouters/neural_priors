#!/bin/bash

# Example usage script for get_bayes_factors.py
# This script demonstrates how to calculate Bayes factors for different scenarios

# Set common parameters
GW_EVENT="GW170817"
BASE_DIR="../GW_runs/"

echo "Calculating Bayes factors for $GW_EVENT..."

# Calculate Bayes factors for each population type
echo "=== Uniform population ==="
python get_bayes_factors.py --gw-event $GW_EVENT --population-type uniform --base-dir $BASE_DIR

echo "=== Gaussian population ==="
python get_bayes_factors.py --gw-event $GW_EVENT --population-type gaussian --base-dir $BASE_DIR

echo "=== Double Gaussian population ==="
python get_bayes_factors.py --gw-event $GW_EVENT --population-type double_gaussian --base-dir $BASE_DIR

# Example with different EOS samples (if available)
echo "=== Example with different EOS samples ==="
# python get_bayes_factors.py --gw-event $GW_EVENT --population-type uniform --eos-samples-name other_radio --base-dir $BASE_DIR

echo "Bayes factor calculation complete!"
echo ""
echo "Results saved to:"
echo "  $BASE_DIR/$GW_EVENT/uniform/bayes_factors.txt"
echo "  $BASE_DIR/$GW_EVENT/gaussian/bayes_factors.txt"
echo "  $BASE_DIR/$GW_EVENT/double_gaussian/bayes_factors.txt"