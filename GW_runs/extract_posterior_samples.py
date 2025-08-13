#!/usr/bin/env python3
"""
Script to extract posterior samples from GW run results and save as NPZ files.

This script iterates through all subdirectories under a GW event directory,
finds JSON result files, extracts posterior samples, and saves them as NPZ files.
"""

import os
import json
import numpy as np
import argparse
from pathlib import Path
import bilby

import logging
logging.getLogger('bilby').setLevel(logging.WARNING)


def extract_posterior_from_json(json_path: str) -> dict:
    """
    Extract posterior samples from a bilby result JSON file.
    
    Args:
        json_path: Path to the JSON result file
        
    Returns:
        Dictionary containing posterior samples with parameter names as keys
    """
    try:
        # Load the bilby result
        result = bilby.result.read_in_result(json_path)
        
        # Extract posterior samples as dictionary
        posterior_dict = {}
        for param in result.posterior.columns:
            posterior_dict[param] = np.array(result.posterior[param])
            
        return posterior_dict
        
    except Exception as e:
        print(f"Error loading result from {json_path}: {e}")
        return None


def find_json_files_recursive(directory: Path) -> list:
    """
    Recursively find all JSON files in directory structure.
    
    Args:
        directory: Directory to search recursively
        
    Returns:
        List of JSON file paths
    """
    json_files = []
    
    def search_directory(current_dir: Path):
        """Recursive helper function to search directories."""
        try:
            for item in current_dir.iterdir():
                if item.is_file() and item.suffix.lower() == '.json':
                    # Skip reference_parameters.json and other config files
                    if 'result' in item.name.lower():
                        json_files.append(item)
                elif item.is_dir():
                    # Continue searching subdirectories
                    search_directory(item)
        except PermissionError:
            print(f"Permission denied accessing {current_dir}")
        except Exception as e:
            print(f"Error accessing {current_dir}: {e}")
    
    search_directory(directory)
    return json_files


def create_output_path(json_path: Path, base_input_dir: Path, base_output_dir: Path, gw_event: str) -> Path:
    """
    Create output NPZ path maintaining the same directory structure.
    
    Args:
        json_path: Path to the JSON file
        base_input_dir: Base input directory (e.g., ./GW_runs/GW170817)
        base_output_dir: Base output directory (e.g., ./final_results)
        gw_event: GW event name (e.g., GW170817)
        
    Returns:
        Path for the output NPZ file
    """
    # Get relative path from base input dir
    try:
        relative_path = json_path.relative_to(base_input_dir)
    except ValueError:
        # If json_path is not under base_input_dir, use the full path structure
        relative_path = json_path
    
    # Create output path with GW event name included, replacing JSON with NPZ
    output_path = base_output_dir / gw_event / relative_path.with_suffix('.npz')
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Extract posterior samples from GW run results")
    parser.add_argument('gw_event', type=str,
                        help='GW event name (e.g., GW170817)')
    parser.add_argument('--input-dir', type=str, default='.',
                        help='Base input directory containing GW runs (default: current directory)')
    parser.add_argument('--output-dir', type=str, default='../final_results',
                        help='Base output directory for NPZ files (default: ../final_results)')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing NPZ files')
    
    args = parser.parse_args()
    
    # Construct paths
    base_input_dir = Path(args.input_dir) / args.gw_event
    base_output_dir = Path(args.output_dir)
    
    # Check if GW event directory exists
    if not base_input_dir.exists():
        print(f"GW event directory does not exist: {base_input_dir}")
        return
    
    # Find all JSON result files
    print(f"Searching for JSON result files in: {base_input_dir}")
    json_files = find_json_files_recursive(base_input_dir)
    
    if not json_files:
        print("No JSON result files found.")
        return
        
    print(f"Found {len(json_files)} JSON result files to process")
    
    # Process each file
    processed = 0
    skipped = 0
    errors = 0
    
    for json_path in json_files:
        # Create output path
        output_path = create_output_path(json_path, base_input_dir, base_output_dir, args.gw_event)
        
        # Check if NPZ already exists
        if output_path.exists() and not args.overwrite:
            print(f"Skipping {output_path} (already exists, use --overwrite to replace)")
            skipped += 1
            continue
        
        print(f"Processing: {json_path}")
        
        # Extract posterior samples
        posterior_dict = extract_posterior_from_json(str(json_path))
        
        if posterior_dict is None:
            errors += 1
            continue
        
        # Create output directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as NPZ
        try:
            np.savez(str(output_path), **posterior_dict)
            print(f"  Saved {len(posterior_dict)} parameters with {len(next(iter(posterior_dict.values())))} samples to: {output_path}")
            processed += 1
            
        except Exception as e:
            print(f"  Error saving NPZ file: {e}")
            errors += 1
    
    print(f"\nSummary:")
    print(f"  Processed: {processed}")
    print(f"  Skipped: {skipped}")
    print(f"  Errors: {errors}")


if __name__ == "__main__":
    main()