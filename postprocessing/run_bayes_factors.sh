#!/bin/bash

# Example usage script for get_bayes_factors.py
# This script demonstrates how to calculate Bayes factors for different scenarios
# using flexible comparison modes with comprehensive configuration options

# Set common parameters
BASE_DIR="../GW_runs/"

# Define arrays for batch processing
GW_EVENTS=("GW170817" "GW190425" "GW230529")
POPULATION_TYPES=("uniform" "gaussian" "double_gaussian" "GW170817" "GW190425" "GW230529")
SOURCE_TYPES=("bns" "nsbh" "default")
EOS_SAMPLES=("radio" "radio_chiEFT" "radio_chiEFT_NICER")

echo "========================================"
echo "FLEXIBLE BAYES FACTOR ANALYSIS"
echo "========================================"

# Function to run a single analysis
run_analysis() {
    local gw_event=$1
    local comparison_mode=$2
    local population_type=$3
    local source_type=$4
    local eos_samples=$5
    
    echo "Running: $gw_event | $comparison_mode | pop=$population_type | src=$source_type | eos=$eos_samples"
    python get_bayes_factors.py \
        --gw-event "$gw_event" \
        --comparison-mode "$comparison_mode" \
        --population-type "$population_type" \
        --source-type "$source_type" \
        --eos-samples-name "$eos_samples" \
        --base-dir "$BASE_DIR"
}

# Default single event examples
GW_EVENT="GW170817"

echo ""
echo "SOURCE TYPE COMPARISONS (fix population & EOS, compare across sources)"
echo "----------------------------------------------------------------------"
echo "Calculating Bayes factors for $GW_EVENT source type comparisons..."
# For source comparisons, we fix population & EOS, but the source_type parameter is ignored
# since the comparison mode will automatically find all available source types
run_analysis $GW_EVENT source uniform bns radio
run_analysis $GW_EVENT source gaussian bns radio
run_analysis $GW_EVENT source double_gaussian bns radio
run_analysis $GW_EVENT source uniform bns radio_chiEFT

echo ""
echo "POPULATION TYPE COMPARISONS (fix source & EOS, compare across populations)"
echo "--------------------------------------------------------------------------"
echo "Calculating Bayes factors for $GW_EVENT population type comparisons..."
run_analysis $GW_EVENT population uniform bns radio
run_analysis $GW_EVENT population uniform nsbh radio
run_analysis $GW_EVENT population uniform default radio
run_analysis $GW_EVENT population uniform bns radio_chiEFT

echo ""
echo "EOS CONSTRAINT COMPARISONS (fix population & source, compare across EOS)"
echo "------------------------------------------------------------------------"
echo "Calculating Bayes factors for $GW_EVENT EOS constraint comparisons..."
run_analysis $GW_EVENT eos uniform bns radio
run_analysis $GW_EVENT eos gaussian bns radio
run_analysis $GW_EVENT eos double_gaussian bns radio
run_analysis $GW_EVENT eos uniform nsbh radio

echo ""
echo "========================================"
echo "BATCH PROCESSING EXAMPLES"
echo "========================================"

# Option 1: Run all source comparisons for specific fixed parameters
echo "BATCH: All source comparisons for uniform population and radio EOS..."
for gw_event in "${GW_EVENTS[@]}"; do
    run_analysis "$gw_event" source uniform bns radio
done

echo ""
echo "BATCH: All population comparisons for BNS source and radio EOS..."
for gw_event in "${GW_EVENTS[@]}"; do
    run_analysis "$gw_event" population uniform bns radio
done

echo ""
echo "BATCH: All EOS comparisons for uniform population and BNS source..."
for gw_event in "${GW_EVENTS[@]}"; do
    run_analysis "$gw_event" eos uniform bns radio
done

echo ""
echo "========================================"
echo "COMPREHENSIVE ANALYSIS EXAMPLE"
echo "========================================"
echo "Running all comparison modes for GW170817 with standard configurations..."

# All source comparisons for uniform/radio
run_analysis GW170817 source uniform bns radio

# All population comparisons for bns/radio  
run_analysis GW170817 population uniform bns radio

# All EOS comparisons for uniform/bns
run_analysis GW170817 eos uniform bns radio

echo ""
echo "========================================"
echo "CUSTOM BATCH PROCESSING"
echo "========================================"
echo "Uncomment and modify the loops below for comprehensive analysis:"
echo ""
echo "# Full combinatorial analysis (WARNING: many combinations!)"
echo "# for gw_event in \"\${GW_EVENTS[@]}\"; do"
echo "#     for pop_type in \"\${POPULATION_TYPES[@]}\"; do"
echo "#         for eos_samples in \"\${EOS_SAMPLES[@]}\"; do"
echo "#             run_analysis \"\$gw_event\" source \"\$pop_type\" bns \"\$eos_samples\""
echo "#         done"
echo "#     done"
echo "# done"

echo ""
echo "Bayes factor calculation complete!"