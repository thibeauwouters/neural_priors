#!/bin/bash

# Example usage script for cornerplots.py
# This script demonstrates how to generate corner plots for different scenarios
# using both legacy mode and new flexible comparison modes

# Set common parameters
BASE_DIR="../GW_runs/"

### TODO: remove me
# echo "========================================"
# echo "LEGACY MODE EXAMPLES"
# echo "========================================"

# GW_EVENT="GW190425"
# echo "Generating legacy corner plots for $GW_EVENT..."
# python cornerplots.py --gw-event $GW_EVENT --population-type uniform --plot-all-params --plot-bns --plot-nsbh --convert-lambdas
# python cornerplots.py --gw-event $GW_EVENT --population-type uniform --plot-all-params --plot-bns --plot-nsbh --no-convert-lambdas

# GW_EVENT="GW170817"
# echo "Generating legacy corner plots for $GW_EVENT..."
# python cornerplots.py --gw-event $GW_EVENT --population-type uniform --plot-all-params --plot-bns --plot-nsbh --plot-hauke --convert-lambdas
# python cornerplots.py --gw-event $GW_EVENT --population-type uniform --plot-all-params --plot-bns --plot-nsbh --plot-hauke --no-convert-lambdas

GW_EVENT="GW170817"

echo ""
echo "SOURCE TYPE COMPARISONS (fix population & EOS, compare across sources)"
echo "----------------------------------------------------------------------"
python cornerplots.py --gw-event $GW_EVENT --comparison-mode source --population-type uniform --eos-samples-name radio --convert-lambdas
python cornerplots.py --gw-event $GW_EVENT --comparison-mode source --population-type gaussian --eos-samples-name radio --convert-lambdas
python cornerplots.py --gw-event $GW_EVENT --comparison-mode source --population-type uniform --eos-samples-name radio_chiEFT --convert-lambdas

echo ""
echo "POPULATION TYPE COMPARISONS (fix source & EOS, compare across populations)"
echo "--------------------------------------------------------------------------"
python cornerplots.py --gw-event $GW_EVENT --comparison-mode population --source-type bns --eos-samples-name radio --convert-lambdas
python cornerplots.py --gw-event $GW_EVENT --comparison-mode population --source-type nsbh --eos-samples-name radio --convert-lambdas
python cornerplots.py --gw-event $GW_EVENT --comparison-mode population --source-type bns --eos-samples-name radio_chiEFT --convert-lambdas

echo ""
echo "EOS CONSTRAINT COMPARISONS (fix population & source, compare across EOS)"
echo "------------------------------------------------------------------------"
python cornerplots.py --gw-event $GW_EVENT --comparison-mode eos --population-type uniform --source-type bns --convert-lambdas
python cornerplots.py --gw-event $GW_EVENT --comparison-mode eos --population-type gaussian --source-type bns --convert-lambdas
python cornerplots.py --gw-event $GW_EVENT --comparison-mode eos --population-type uniform --source-type nsbh --convert-lambdas

echo ""
echo "ALL PARAMETERS PLOTS"
echo "--------------------"
python cornerplots.py --gw-event $GW_EVENT --comparison-mode source --population-type uniform --eos-samples-name radio --plot-all-params --convert-lambdas
python cornerplots.py --gw-event $GW_EVENT --comparison-mode population --source-type bns --eos-samples-name radio --plot-all-params --convert-lambdas
python cornerplots.py --gw-event $GW_EVENT --comparison-mode eos --population-type uniform --source-type bns --plot-all-params --convert-lambdas

echo ""
echo "GW190425 EXAMPLES"
echo "-----------------"
GW_EVENT="GW190425"
python cornerplots.py --gw-event $GW_EVENT --comparison-mode source --population-type uniform --eos-samples-name radio --convert-lambdas
python cornerplots.py --gw-event $GW_EVENT --comparison-mode population --source-type bns --eos-samples-name radio --convert-lambdas
python cornerplots.py --gw-event $GW_EVENT --comparison-mode eos --population-type uniform --source-type bns --convert-lambdas

echo ""
echo "Corner plot generation complete!"