#!/usr/bin/env python3
"""
Script to create and save a fast cosmology interpolator for d_L -> z conversion.

This script creates a high-resolution grid of luminosity distances and corresponding
redshifts using Planck15 cosmology, then saves the interpolation data for fast
lookup during corner plot generation.

Usage:
    python make_cosmology_interpolator.py
    
Output:
    cosmology_interpolator.npz - Contains d_L grid and z values for interpolation
"""

import numpy as np
from astropy.cosmology import Planck15
import astropy.units as u

def create_cosmology_interpolator():
    """
    Create a high-resolution cosmology interpolator for d_L -> z conversion.
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: (d_L_grid, z_grid) for interpolation
    """
    # Define redshift range - focus on typical GW detection range (up to ~1000 Mpc)
    # Most GW events are at z < 0.3, with very dense sampling for z < 0.1
    z_min = 0.001  # Avoid z=0 issues
    z_max = 0.5    # Covers up to ~2500 Mpc, well beyond typical GW range
    
    # Create fine-grained redshift grid with high density at low z
    z_very_low = np.linspace(z_min, 0.05, 2000)   # Very high resolution for z < 0.05 (~200 Mpc)
    z_low = np.linspace(0.05, 0.15, 2000)         # High resolution for 0.05 < z < 0.15 (~650 Mpc)  
    z_mid = np.linspace(0.15, 0.3, 1500)          # Medium resolution for 0.15 < z < 0.3 (~1300 Mpc)
    z_high = np.linspace(0.3, z_max, 500)         # Lower resolution for z > 0.3
    
    # Combine and remove duplicates
    z_grid = np.unique(np.concatenate([z_very_low, z_low, z_mid, z_high]))
    
    print(f"Computing luminosity distances for {len(z_grid)} redshift points...")
    print(f"Redshift range: {z_grid.min():.4f} to {z_grid.max():.4f}")
    
    # Calculate luminosity distances using Planck15 cosmology
    d_L_grid = Planck15.luminosity_distance(z_grid).to(u.Mpc).value
    
    print(f"Luminosity distance range: {d_L_grid.min():.1f} to {d_L_grid.max():.1f} Mpc")
    
    return d_L_grid, z_grid

def save_interpolator(d_L_grid, z_grid, filename="cosmology_interpolator.npz"):
    """
    Save the interpolation grids to a compressed numpy file.
    
    Args:
        d_L_grid (np.ndarray): Luminosity distance grid in Mpc
        z_grid (np.ndarray): Corresponding redshift grid
        filename (str): Output filename
    """
    print(f"Saving interpolator to {filename}...")
    
    # Save with metadata
    np.savez_compressed(
        filename,
        d_L_grid=d_L_grid,
        z_grid=z_grid,
        cosmology="Planck15",
        d_L_min=d_L_grid.min(),
        d_L_max=d_L_grid.max(),
        z_min=z_grid.min(),
        z_max=z_grid.max(),
        n_points=len(z_grid),
        description="Cosmology interpolator for fast d_L -> z conversion"
    )
    
    # Verify the saved file
    data = np.load(filename)
    print(f"Verification: loaded {len(data['d_L_grid'])} grid points")
    data.close()

def test_interpolator(d_L_grid, z_grid):
    """
    Test the interpolator accuracy against known bilby values.
    
    Args:
        d_L_grid (np.ndarray): Luminosity distance grid
        z_grid (np.ndarray): Redshift grid
    """
    from scipy.interpolate import interp1d
    from bilby.gw.conversion import luminosity_distance_to_redshift
    
    print("Testing interpolator accuracy...")
    
    # Create interpolator
    interp_func = interp1d(d_L_grid, z_grid, kind='cubic', 
                          bounds_error=False, fill_value='extrapolate')
    
    # Test on some representative distances
    test_distances = np.array([50, 100, 200, 500, 1000, 2000])  # Mpc
    
    # Get exact values from bilby
    z_exact = luminosity_distance_to_redshift(test_distances)
    
    # Get interpolated values
    z_interp = interp_func(test_distances)
    
    # Calculate relative errors
    rel_errors = np.abs((z_interp - z_exact) / z_exact) * 100
    
    print(f"{'Distance (Mpc)':<15} {'z_exact':<10} {'z_interp':<10} {'Error (%)':<10}")
    print("-" * 50)
    for i, d_L in enumerate(test_distances):
        print(f"{d_L:<15.0f} {z_exact[i]:<10.6f} {z_interp[i]:<10.6f} {rel_errors[i]:<10.4f}")
    
    print(f"\nMax relative error: {rel_errors.max():.4f}%")
    print(f"Mean relative error: {rel_errors.mean():.4f}%")
    
    if rel_errors.max() < 0.1:  # Less than 0.1% error
        print("✓ Interpolator accuracy is excellent!")
    elif rel_errors.max() < 1.0:  # Less than 1% error
        print("✓ Interpolator accuracy is good for corner plotting.")
    else:
        print("⚠ Interpolator may need higher resolution.")

def main():
    """Main execution function."""
    print("Creating Planck15 cosmology interpolator...")
    print("=" * 50)
    
    # Create the interpolation grids
    d_L_grid, z_grid = create_cosmology_interpolator()
    
    # Test accuracy
    test_interpolator(d_L_grid, z_grid)
    
    # Save to file
    save_interpolator(d_L_grid, z_grid)
    
    print("=" * 50)
    print("Interpolator creation complete!")
    print("\nTo use in corner plotting:")
    print("1. The interpolator will be automatically loaded when fast_plotting=True")
    print("2. Use --no-fast-plotting to fall back to exact calculations if needed")

if __name__ == "__main__":
    main()