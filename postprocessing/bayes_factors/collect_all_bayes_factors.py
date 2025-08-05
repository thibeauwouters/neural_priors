#!/usr/bin/env python3
"""
Script to collect all Bayes factors from GW parameter estimation runs.

This script scans the GW_runs directory structure and extracts Bayes factors
from all result JSON files, creating a nested structure matching the directory
layout and saving to all_bayes_factors.json. It also generates a LaTeX table
showing the structure.
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


def generate_latex_table(bayes_factors: Dict[str, Any]) -> str:
    """
    Generate LaTeX table code for the Bayes factors data.
    
    Args:
        bayes_factors: Nested dictionary with Bayes factors data
        
    Returns:
        String containing LaTeX table code
    """
    latex_lines = []
    
    # Table header
    latex_lines.append("\\begin{table}[htbp]")
    latex_lines.append("\\centering")
    latex_lines.append("\\caption{Bayes Factors for Gravitational Wave Events}")
    latex_lines.append("\\label{tab:bayes_factors}")
    latex_lines.append("\\begin{tabular}{|l|l|l|l|l|}")
    latex_lines.append("\\hline")
    latex_lines.append("\\textbf{Population} & \\textbf{Source} & \\textbf{EOS Constraints} & \\textbf{Event} & \\textbf{Log Bayes Factor} \\\\")
    latex_lines.append("\\hline")
    
    # Fixed options for consistent ordering
    gw_events = ["GW170817", "GW190425", "GW230529"]
    population_types = ["uniform", "gaussian", "double_gaussian", "GW170817", "GW190425", "GW230529"]
    source_types = ["bns", "nsbh", "default"]
    eos_samples_names = ["radio", "radio_chiEFT", "radio_chiEFT_NICER"]
    
    # Generate table rows
    for pop_type in population_types:
        pop_first_row = True
        pop_row_count = 0
        
        # Count total rows for this population to use multirow
        for event in gw_events:
            if event in bayes_factors and pop_type in bayes_factors[event]:
                for source_type in source_types:
                    if source_type in bayes_factors[event][pop_type]:
                        for eos_type in eos_samples_names:
                            if eos_type in bayes_factors[event][pop_type][source_type]:
                                bf_val = bayes_factors[event][pop_type][source_type][eos_type]
                                if bf_val != 0.0:  # Only count non-zero entries
                                    pop_row_count += 1
        
        if pop_row_count == 0:
            continue
            
        for event in gw_events:
            if event not in bayes_factors or pop_type not in bayes_factors[event]:
                continue
                
            for source_type in source_types:
                if source_type not in bayes_factors[event][pop_type]:
                    continue
                    
                source_first_row = True
                source_row_count = sum(1 for eos_type in eos_samples_names 
                                     if eos_type in bayes_factors[event][pop_type][source_type] 
                                     and bayes_factors[event][pop_type][source_type][eos_type] != 0.0)
                
                if source_row_count == 0:
                    continue
                    
                for eos_type in eos_samples_names:
                    if eos_type not in bayes_factors[event][pop_type][source_type]:
                        continue
                        
                    bf_val = bayes_factors[event][pop_type][source_type][eos_type]
                    if bf_val == 0.0:  # Skip zero entries (missing data)
                        continue
                    
                    # Format the row
                    pop_cell = f"\\multirow{{{pop_row_count}}}{{*}}{{{pop_type}}}" if pop_first_row else ""
                    source_cell = f"\\multirow{{{source_row_count}}}{{*}}{{{source_type.upper()}}}" if source_first_row else ""
                    eos_cell = eos_type.replace("_", "\\_")
                    event_cell = event
                    bf_cell = f"{bf_val:.2f}"
                    
                    latex_lines.append(f"{pop_cell} & {source_cell} & {eos_cell} & {event_cell} & {bf_cell} \\\\")
                    
                    if not (pop_first_row and source_first_row):
                        latex_lines.append("\\cline{3-5}")
                    
                    pop_first_row = False
                    source_first_row = False
                
                if not pop_first_row:
                    latex_lines.append("\\cline{2-5}")
        
        latex_lines.append("\\hline")
    
    # Close table
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\end{table}")
    
    return "\n".join(latex_lines)


def main():
    """Main function to collect all Bayes factors and save to JSON file and LaTeX table."""
    base_dir = "../../GW_runs/"
    output_dir = "."
    json_file = os.path.join(output_dir, "all_bayes_factors.json")
    latex_file = os.path.join(output_dir, "bayes_factors_table.tex")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Scanning directory structure in: {base_dir}")
    bayes_factors = collect_all_bayes_factors(base_dir)
    
    print(f"\nSaving JSON results to: {json_file}")
    with open(json_file, 'w') as f:
        json.dump(bayes_factors, f, indent=2)
    
    print(f"Generating LaTeX table and saving to: {latex_file}")
    latex_table = generate_latex_table(bayes_factors)
    with open(latex_file, 'w') as f:
        f.write(latex_table)
    
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