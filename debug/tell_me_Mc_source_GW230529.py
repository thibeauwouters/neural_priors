#!/usr/bin/env python3
"""
Simple script to print the median source-frame chirp mass from GW230529 NSBH default inference.
"""

import numpy as np
import os

# Path to the default NSBH results for GW230529
script_dir = os.path.dirname(os.path.abspath(__file__))
result_path = os.path.join(script_dir, "..", "final_results", "GW230529", "nsbh", "default", "samples.npz")

# Check if file exists
if not os.path.exists(result_path):
    print(f"Error: Result file not found at {result_path}")
    print("\nSearching for alternative locations...")

    # Try alternative path in GW_runs
    alt_path = os.path.join(script_dir, "..", "GW_runs", "final_results", "GW230529", "nsbh", "default", "samples.npz")
    if os.path.exists(alt_path):
        result_path = alt_path
        print(f"Found at: {alt_path}")
    else:
        print("Could not find result file. Please check the path.")
        exit(1)

# Load the data
print(f"Loading data from: {result_path}")
data = np.load(result_path)

# Print available parameters for debugging
print("\nAvailable parameters:")
print(list(data.keys()))

# Extract chirp_mass_source
if 'chirp_mass_source' in data:
    chirp_mass_source = data['chirp_mass_source']
    median_Mc_source = np.median(chirp_mass_source)

    print(f"\n{'='*60}")
    print(f"GW230529 NSBH Default Inference")
    print(f"{'='*60}")
    print(f"Median source-frame chirp mass: {median_Mc_source:.4f} M☉")
    print(f"{'='*60}")
else:
    print("\nWarning: 'chirp_mass_source' not found in data.")
    print("Attempting to compute from available parameters...")

    # Try to compute from chirp_mass and redshift if available
    if 'chirp_mass' in data and 'redshift' in data:
        chirp_mass = data['chirp_mass']
        redshift = data['redshift']
        chirp_mass_source = chirp_mass / (1 + redshift)
        median_Mc_source = np.median(chirp_mass_source)

        print(f"\n{'='*60}")
        print(f"GW230529 NSBH Default Inference (computed)")
        print(f"{'='*60}")
        print(f"Median source-frame chirp mass: {median_Mc_source:.4f} M☉")
        print(f"{'='*60}")
    elif 'chirp_mass' in data and 'luminosity_distance' in data:
        # Compute redshift from luminosity distance
        from bilby.gw.conversion import luminosity_distance_to_redshift
        chirp_mass = data['chirp_mass']
        luminosity_distance = data['luminosity_distance']
        redshift = luminosity_distance_to_redshift(luminosity_distance)
        chirp_mass_source = chirp_mass / (1 + redshift)
        median_Mc_source = np.median(chirp_mass_source)

        print(f"\n{'='*60}")
        print(f"GW230529 NSBH Default Inference (computed from dL)")
        print(f"{'='*60}")
        print(f"Median source-frame chirp mass: {median_Mc_source:.4f} M☉")
        print(f"{'='*60}")
    else:
        print("Cannot compute chirp_mass_source from available parameters.")
        print("Available parameters:", list(data.keys()))
