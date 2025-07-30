#!/bin/bash

# Example usage script for get_bayes_factors.py
# This script demonstrates how to calculate Bayes factors for different scenarios

# Set common parameters
GW_EVENT="GW170817"
BASE_DIR="../GW_runs/"

echo "Calculating Bayes factors for $GW_EVENT..."
python get_bayes_factors.py --gw-event $GW_EVENT --population-type uniform --base-dir $BASE_DIR