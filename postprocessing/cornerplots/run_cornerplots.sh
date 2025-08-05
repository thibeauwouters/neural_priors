#!/bin/bash
# Example usage script for cornerplots.py

conda activate eos_source_classification

# Set common parameters
BASE_DIR="../GW_runs/"

for GW_EVENT in GW170817 GW190425; do
  for EOS in radio radio_chiEFT radio_chiEFT_NICER; do
    echo "Generating usual corner plots for $GW_EVENT with EOS $EOS..."
    python cornerplots.py --gw-event $GW_EVENT --population-type $GW_EVENT --eos-samples-name $EOS --plot-bns --plot-nsbh --plot-hauke --convert-lambdas
    python cornerplots.py --gw-event $GW_EVENT --population-type $GW_EVENT --eos-samples-name $EOS --plot-bns --plot-nsbh --plot-hauke --no-convert-lambdas
  done
done

for GW_EVENT in GW170817 GW190425; do

  python cornerplots.py --gw-event $GW_EVENT --comparison-mode source --population-type uniform --eos-samples-name radio --convert-lambdas
  python cornerplots.py --gw-event $GW_EVENT --comparison-mode source --population-type gaussian --eos-samples-name radio --convert-lambdas
  python cornerplots.py --gw-event $GW_EVENT --comparison-mode source --population-type uniform --eos-samples-name radio_chiEFT --convert-lambdas

  python cornerplots.py --gw-event $GW_EVENT --comparison-mode population --source-type bns --eos-samples-name radio --convert-lambdas
  python cornerplots.py --gw-event $GW_EVENT --comparison-mode population --source-type nsbh --eos-samples-name radio --convert-lambdas
  python cornerplots.py --gw-event $GW_EVENT --comparison-mode population --source-type bns --eos-samples-name radio_chiEFT --convert-lambdas

  python cornerplots.py --gw-event $GW_EVENT --comparison-mode eos --population-type uniform --source-type bns --convert-lambdas
  python cornerplots.py --gw-event $GW_EVENT --comparison-mode eos --population-type gaussian --source-type bns --convert-lambdas
  python cornerplots.py --gw-event $GW_EVENT --comparison-mode eos --population-type uniform --source-type nsbh --convert-lambdas
done

echo ""
echo "Corner plot generation complete!"