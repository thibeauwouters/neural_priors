#!/usr/bin/env python3
"""
Script to collect all Bayes factors from GW parameter estimation runs.

This script scans the GW_runs directory structure and extracts Bayes factors
from all result JSON files, creating a nested structure matching the directory
layout and saving to all_bayes_factors.json.
"""

import os
import json
from typing import Dict, Any


def collect_all_bayes_factors(base_dir: str = "../GW_runs/") -> Dict[str, Any]:
    """
    Collect all Bayes factors from the GW runs directory structure.
    
    Args:
        base_dir (str): Base directory path (should be ../GW_runs/)
        
    Returns:
        Dict with nested structure matching directory layout
    """
    all_bayes_factors = {}
    
    if not os.path.exists(base_dir):
        print(f"Base directory {base_dir} does not exist")
        return all_bayes_factors
    
    # Fixed options from cornerplots.py
    gw_events = ["GW170817", "GW190425", "GW230529"]
    population_types = ["uniform", "gaussian", "double_gaussian", "GW170817", "GW190425", "GW230529"]
    source_types = ["bns", "nsbh", "default"]
    eos_samples_names = ["radio", "radio_chiEFT", "radio_chiEFT_NICER"]
    
    for gw_event in gw_events:
        event_path = os.path.join(base_dir, gw_event)
        if not os.path.exists(event_path):
            print(f"Event directory {event_path} does not exist, skipping")
            continue
            
        print(f"Processing event: {gw_event}")
        all_bayes_factors[gw_event] = {}
        
        for population_type in population_types:
            pop_path = os.path.join(event_path, population_type)
            if not os.path.exists(pop_path):
                continue
                
            print(f"  Population: {population_type}")
            all_bayes_factors[gw_event][population_type] = {}
            
            for source_type in source_types:
                source_path = os.path.join(pop_path, source_type)
                if not os.path.exists(source_path):
                    continue
                    
                print(f"    Source: {source_type}")
                all_bayes_factors[gw_event][population_type][source_type] = {}
                
                for eos_type in eos_samples_names:
                    eos_path = os.path.join(source_path, eos_type)
                    if not os.path.exists(eos_path):
                        continue
                        
                    print(f"      EOS: {eos_type}")
                    
                    # Look for result file using the pattern from utils.py
                    # Handle special case: default runs in gaussian/double_gaussian are stored in uniform/radio/
                    if source_type == "default" and population_type in ["gaussian", "double_gaussian"]:
                        actual_population = "uniform"
                        actual_eos = "radio"
                        result_file = os.path.join(base_dir, gw_event, actual_population, source_type, 
                                                 actual_eos, f"{source_type}_result.json")
                    else:
                        result_file = os.path.join(eos_path, f"{source_type}_result.json")
                    
                    if os.path.exists(result_file):
                        try:
                            with open(result_file, 'r') as f:
                                result_data = json.load(f)
                                bayes_factor = result_data.get("log_bayes_factor", 0.0)
                                all_bayes_factors[gw_event][population_type][source_type][eos_type] = bayes_factor
                                print(f"        Found Bayes factor: {bayes_factor}")
                        except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
                            print(f"        Error reading {result_file}: {e}. Setting to 0.0")
                            all_bayes_factors[gw_event][population_type][source_type][eos_type] = 0.0
                    else:
                        print(f"        Result file not found: {result_file}. Setting to 0.0")
                        all_bayes_factors[gw_event][population_type][source_type][eos_type] = 0.0
    
    return all_bayes_factors


def main():
    """Main function to collect all Bayes factors and save to JSON file."""
    base_dir = "../GW_runs/"
    output_file = "./all_bayes_factors.json"
    
    print(f"Scanning directory structure in: {base_dir}")
    bayes_factors = collect_all_bayes_factors(base_dir)
    
    print(f"\nSaving results to: {output_file}")
    with open(output_file, 'w') as f:
        json.dump(bayes_factors, f, indent=2)
    
    print("Done!")
    print(f"\nSummary:")
    total_runs = 0
    for event, populations in bayes_factors.items():
        for pop, sources in populations.items():
            for source, eos_types in sources.items():
                total_runs += len(eos_types)
    
    print(f"Events processed: {len(bayes_factors)}")
    print(f"Total runs found: {total_runs}")


if __name__ == "__main__":
    main()