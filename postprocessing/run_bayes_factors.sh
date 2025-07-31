#!/bin/bash

# Example usage script for get_bayes_factors.py
# This script demonstrates how to calculate Bayes factors for different scenarios
# using both legacy mode and new flexible comparison modes

# Set common parameters
BASE_DIR="../GW_runs/"

echo "========================================"
echo "NEW FLEXIBLE COMPARISON MODE EXAMPLES"
echo "========================================"

GW_EVENT="GW170817"

echo ""
echo "SOURCE TYPE COMPARISONS (fix population & EOS, compare across sources)"
echo "----------------------------------------------------------------------"
echo "Calculating Bayes factors for $GW_EVENT source type comparisons..."
python get_bayes_factors.py --gw-event $GW_EVENT --comparison-mode source --population-type uniform --eos-samples-name radio --base-dir $BASE_DIR
python get_bayes_factors.py --gw-event $GW_EVENT --comparison-mode source --population-type gaussian --eos-samples-name radio --base-dir $BASE_DIR
python get_bayes_factors.py --gw-event $GW_EVENT --comparison-mode source --population-type double_gaussian --eos-samples-name radio --base-dir $BASE_DIR
python get_bayes_factors.py --gw-event $GW_EVENT --comparison-mode source --population-type uniform --eos-samples-name radio_chiEFT --base-dir $BASE_DIR

echo ""
echo "POPULATION TYPE COMPARISONS (fix source & EOS, compare across populations)"
echo "--------------------------------------------------------------------------"
echo "Calculating Bayes factors for $GW_EVENT population type comparisons..."
python get_bayes_factors.py --gw-event $GW_EVENT --comparison-mode population --source-type bns --eos-samples-name radio --base-dir $BASE_DIR
python get_bayes_factors.py --gw-event $GW_EVENT --comparison-mode population --source-type nsbh --eos-samples-name radio --base-dir $BASE_DIR
python get_bayes_factors.py --gw-event $GW_EVENT --comparison-mode population --source-type default --eos-samples-name radio --base-dir $BASE_DIR
python get_bayes_factors.py --gw-event $GW_EVENT --comparison-mode population --source-type bns --eos-samples-name radio_chiEFT --base-dir $BASE_DIR

echo ""
echo "EOS CONSTRAINT COMPARISONS (fix population & source, compare across EOS)"
echo "------------------------------------------------------------------------"
echo "Calculating Bayes factors for $GW_EVENT EOS constraint comparisons..."
python get_bayes_factors.py --gw-event $GW_EVENT --comparison-mode eos --population-type uniform --source-type bns --base-dir $BASE_DIR
python get_bayes_factors.py --gw-event $GW_EVENT --comparison-mode eos --population-type gaussian --source-type bns --base-dir $BASE_DIR
python get_bayes_factors.py --gw-event $GW_EVENT --comparison-mode eos --population-type double_gaussian --source-type bns --base-dir $BASE_DIR
python get_bayes_factors.py --gw-event $GW_EVENT --comparison-mode eos --population-type uniform --source-type nsbh --base-dir $BASE_DIR

echo ""
echo "GW190425 EXAMPLES"
echo "-----------------"
GW_EVENT="GW190425"

echo "Calculating Bayes factors for $GW_EVENT source type comparisons..."
python get_bayes_factors.py --gw-event $GW_EVENT --comparison-mode source --population-type uniform --eos-samples-name radio --base-dir $BASE_DIR
python get_bayes_factors.py --gw-event $GW_EVENT --comparison-mode source --population-type gaussian --eos-samples-name radio --base-dir $BASE_DIR
python get_bayes_factors.py --gw-event $GW_EVENT --comparison-mode source --population-type double_gaussian --eos-samples-name radio --base-dir $BASE_DIR

echo "Calculating Bayes factors for $GW_EVENT population type comparisons..."
python get_bayes_factors.py --gw-event $GW_EVENT --comparison-mode population --source-type bns --eos-samples-name radio --base-dir $BASE_DIR
python get_bayes_factors.py --gw-event $GW_EVENT --comparison-mode population --source-type nsbh --eos-samples-name radio --base-dir $BASE_DIR

echo "Calculating Bayes factors for $GW_EVENT EOS constraint comparisons..."
python get_bayes_factors.py --gw-event $GW_EVENT --comparison-mode eos --population-type uniform --source-type bns --base-dir $BASE_DIR
python get_bayes_factors.py --gw-event $GW_EVENT --comparison-mode eos --population-type gaussian --source-type bns --base-dir $BASE_DIR

echo ""
echo "GW230529 EXAMPLES"
echo "-----------------"
GW_EVENT="GW230529"

echo "Calculating Bayes factors for $GW_EVENT source type comparisons..."
python get_bayes_factors.py --gw-event $GW_EVENT --comparison-mode source --population-type uniform --eos-samples-name radio --base-dir $BASE_DIR

echo "Calculating Bayes factors for $GW_EVENT population type comparisons..."
python get_bayes_factors.py --gw-event $GW_EVENT --comparison-mode population --source-type bns --eos-samples-name radio --base-dir $BASE_DIR

echo ""
echo "========================================"
echo "COMPREHENSIVE ANALYSIS EXAMPLE"
echo "========================================"
echo "Running all comparison modes for GW170817 with uniform population and radio EOS..."

# All source comparisons for uniform/radio
python get_bayes_factors.py --gw-event GW170817 --comparison-mode source --population-type uniform --eos-samples-name radio --base-dir $BASE_DIR

# All population comparisons for bns/radio  
python get_bayes_factors.py --gw-event GW170817 --comparison-mode population --source-type bns --eos-samples-name radio --base-dir $BASE_DIR

# All EOS comparisons for uniform/bns
python get_bayes_factors.py --gw-event GW170817 --comparison-mode eos --population-type uniform --source-type bns --base-dir $BASE_DIR

echo ""
echo "Bayes factor calculation complete!"