#!/bin/bash
# Example usage script for cornerplots.py using new comparison-only approach

conda activate eos_source_classification

# Generate corner plots for each GW event using the new comparison mode approach
for GW_EVENT in GW170817 GW190425; do
  echo "Generating comparison corner plots for $GW_EVENT..."
  
  # Source comparison plots (fix population + eos, vary source)
  for POPULATION in uniform gaussian double_gaussian $GW_EVENT; do
    for EOS in radio radio_chiEFT radio_chiEFT_NICER; do
      echo "  Source comparison: population=$POPULATION, eos=$EOS"
      python cornerplots.py --gw-event $GW_EVENT --comparison-mode source --population-type $POPULATION --eos-samples-name $EOS --convert-lambdas
      python cornerplots.py --gw-event $GW_EVENT --comparison-mode source --population-type $POPULATION --eos-samples-name $EOS --no-convert-lambdas
    done
  done
  
  # Population comparison plots (fix source + eos, vary population)
  for SOURCE in bns nsbh; do
    for EOS in radio radio_chiEFT radio_chiEFT_NICER; do
      echo "  Population comparison: source=$SOURCE, eos=$EOS"
      python cornerplots.py --gw-event $GW_EVENT --comparison-mode population --source-type $SOURCE --eos-samples-name $EOS --convert-lambdas
      python cornerplots.py --gw-event $GW_EVENT --comparison-mode population --source-type $SOURCE --eos-samples-name $EOS --no-convert-lambdas
    done
  done
  
  # EOS comparison plots (fix population + source, vary eos)
  for POPULATION in uniform gaussian double_gaussian $GW_EVENT; do
    for SOURCE in bns nsbh; do
      echo "  EOS comparison: population=$POPULATION, source=$SOURCE"
      python cornerplots.py --gw-event $GW_EVENT --comparison-mode eos --population-type $POPULATION --source-type $SOURCE --convert-lambdas
      python cornerplots.py --gw-event $GW_EVENT --comparison-mode eos --population-type $POPULATION --source-type $SOURCE --no-convert-lambdas
    done
  done
done

echo ""
echo "Corner plot generation complete!"