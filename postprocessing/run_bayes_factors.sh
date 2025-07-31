#!/bin/bash

# Example usage script for get_bayes_factors.py
# This script demonstrates how to calculate Bayes factors for different scenarios

# Set common parameters
BASE_DIR="../GW_runs/"

GW_EVENT="GW170817"
echo "Calculating Bayes factors for $GW_EVENT..."
python get_bayes_factors.py --gw-event $GW_EVENT --population-type gaussian --base-dir $BASE_DIR
python get_bayes_factors.py --gw-event $GW_EVENT --population-type gaussian --base-dir $BASE_DIR
python get_bayes_factors.py --gw-event $GW_EVENT --population-type double_gaussian --base-dir $BASE_DIR

GW_EVENT="GW190425"
echo "Calculating Bayes factors for $GW_EVENT..."
python get_bayes_factors.py --gw-event $GW_EVENT --population-type gaussian --base-dir $BASE_DIR
python get_bayes_factors.py --gw-event $GW_EVENT --population-type gaussian --base-dir $BASE_DIR
python get_bayes_factors.py --gw-event $GW_EVENT --population-type double_gaussian --base-dir $BASE_DIR