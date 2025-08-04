#!/bin/bash

# Create base directory
GW_event="GW230529"

# Top-level subdirs (population types)
top_levels=("uniform" "gaussian" "double_gaussian" "$GW_event")

# Middle-level subdirs
mid_subdirs=("default" "bns" "nsbh")

# Lowest-level subdirs (only for bns and nsbh)
nested_subdirs=("radio" "radio_chiEFT" "radio_chiEFT_NICER")

# Loop through top-levels
for top in "${top_levels[@]}"; do
    # Create 'default' directly
    mkdir -p "${GW_event}/${top}/default/radio"

    # Create bns and nsbh substructure
    for mid in "bns" "nsbh"; do
        for nested in "${nested_subdirs[@]}"; do
            mkdir -p "${GW_event}/${top}/${mid}/${nested}"
        done
    done
done

echo "Directory structure created successfully."