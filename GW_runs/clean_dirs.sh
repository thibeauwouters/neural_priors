#!/bin/bash

# Define variables
GW_event="GW230529"
population_types=("uniform" "gaussian" "double_gaussian" $GW_event)
eos_list=("radio" "radio_chiEFT" "radio_chiEFT_NICER")

echo "⚠️  WARNING: This will delete all files in:"
for population_type in "${population_types[@]}"; do
    for eos in "${eos_list[@]}"; do
        if [[ "$eos" == "default" ]]; then
            echo "  ${GW_event}/${population_type}/default/*"
        else
            echo "  ${GW_event}/${population_type}/bns/${eos}/*"
            echo "  ${GW_event}/${population_type}/nsbh/${eos}/*"
        fi
    done
done

echo "Are you sure you want to proceed? (y/n)"
read -r confirm
if [[ "$confirm" != "y" ]]; then
    echo "Operation cancelled."
    exit 1
fi

# Perform cleanup
for population_type in "${population_types[@]}"; do
    for eos in "${eos_list[@]}"; do
        targets=()
        if [[ "$eos" == "default" ]]; then
            targets+=("${GW_event}/${population_type}/default")
        else
            targets+=(
                "${GW_event}/${population_type}/bns/${eos}"
                "${GW_event}/${population_type}/nsbh/${eos}"
            )
        fi

        for dir in "${targets[@]}"; do
            if [[ -d "$dir" ]]; then
                echo "Clearing contents of $dir"
                rm -f "$dir"/*
            else
                echo "Directory $dir does not exist. Skipping."
            fi
        done
    done
done

echo "✅ Cleanup complete."
