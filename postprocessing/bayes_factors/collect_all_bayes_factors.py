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
import argparse
import shutil
from typing import Dict, Any

# Fixed options from cornerplots.py
GW_EVENTS = ["GW170817", "GW190425", "GW230529"]
POPULATION_TYPES = ["uniform", "gaussian", "double_gaussian", "GW170817", "GW190425", "GW230529"]
SOURCE_TYPES = ["bns", "nsbh"]
EOS_SAMPLES_NAMES = ["radio", "radio_chiEFT", "radio_chiEFT_NICER"]


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
    
    for gw_event in GW_EVENTS:
        event_path = os.path.join(base_dir, gw_event)
        if not os.path.exists(event_path):
            print(f"Event directory {event_path} does not exist, skipping")
            continue
            
        print(f"Processing event: {gw_event}")
        all_bayes_factors[gw_event] = {}
        
        for population_type in POPULATION_TYPES:
            pop_path = os.path.join(event_path, population_type)
            if not os.path.exists(pop_path):
                continue
                
            print(f"  Population: {population_type}")
            all_bayes_factors[gw_event][population_type] = {}
            
            for source_type in SOURCE_TYPES:
                source_path = os.path.join(pop_path, source_type)
                if not os.path.exists(source_path):
                    continue
                    
                print(f"    Source: {source_type}")
                all_bayes_factors[gw_event][population_type][source_type] = {}
                
                for eos_type in EOS_SAMPLES_NAMES:
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
    
    # Translation dict for human-readable names
    population_translations = {
        "uniform": "Uniform",
        "gaussian": "Gaussian", 
        "double_gaussian": "Double Gaussian",
        "gw_event": "GW-event"
    }
    
    # EOS incremental formatting
    eos_translations = {
        "radio": "radio",
        "radio_chiEFT": "+\\chiEFT",
        "radio_chiEFT_NICER": "+NICER"
    }
    
    # Group population types for display
    population_types_display = ["uniform", "gaussian", "double_gaussian", "gw_event"]
    gw_event_populations = ["GW170817", "GW190425", "GW230529"]
    
    # Table header - wider format with events as columns
    latex_lines.append("\\begin{table*}[htbp]")
    latex_lines.append("\\centering")
    latex_lines.append("\\caption{Bayes Factors for Gravitational Wave Events}")
    latex_lines.append("\\label{tab:bayes_factors}")
    
    # Create column specification: Population | Source | EOS | GW170817 | GW190425 | GW230529
    latex_lines.append("\\begin{tabular}{|l|l|l|c|c|c|}")
    latex_lines.append("\\hline")
    latex_lines.append("\\textbf{Population} & \\textbf{Source} & \\textbf{EOS Constraints} & \\textbf{GW170817} & \\textbf{GW190425} & \\textbf{GW230529} \\\\")
    latex_lines.append("\\hline")
    
    # Generate table rows
    for pop_type_display in population_types_display:
        pop_first_row = True
        pop_row_count = 0
        
        # Determine which population types to check for this display type
        if pop_type_display == "gw_event":
            pop_types_to_check = gw_event_populations
        else:
            pop_types_to_check = [pop_type_display]
        
        # Count total rows for this population to use multirow
        for source_type in SOURCE_TYPES:
            for eos_type in EOS_SAMPLES_NAMES:
                # Check if this combination has any data across events
                has_data = any(
                    event in bayes_factors and 
                    any(pop_type in bayes_factors[event] and 
                        source_type in bayes_factors[event][pop_type] and 
                        eos_type in bayes_factors[event][pop_type][source_type] and
                        bayes_factors[event][pop_type][source_type][eos_type] != 0.0
                        for pop_type in pop_types_to_check)
                    for event in GW_EVENTS
                )
                if has_data:
                    pop_row_count += 1
        
        if pop_row_count == 0:
            continue
            
        for source_type in SOURCE_TYPES:
            source_first_row = True
            source_row_count = 0
            
            # Count rows for this source type
            for eos_type in EOS_SAMPLES_NAMES:
                has_data = any(
                    event in bayes_factors and 
                    any(pop_type in bayes_factors[event] and 
                        source_type in bayes_factors[event][pop_type] and 
                        eos_type in bayes_factors[event][pop_type][source_type] and
                        bayes_factors[event][pop_type][source_type][eos_type] != 0.0
                        for pop_type in pop_types_to_check)
                    for event in GW_EVENTS
                )
                if has_data:
                    source_row_count += 1
            
            if source_row_count == 0:
                continue
                
            for eos_type in EOS_SAMPLES_NAMES:
                # Check if this combination has any data across events
                has_data = any(
                    event in bayes_factors and 
                    any(pop_type in bayes_factors[event] and 
                        source_type in bayes_factors[event][pop_type] and 
                        eos_type in bayes_factors[event][pop_type][source_type] and
                        bayes_factors[event][pop_type][source_type][eos_type] != 0.0
                        for pop_type in pop_types_to_check)
                    for event in GW_EVENTS
                )
                
                if not has_data:
                    continue
                
                # Format the row
                pop_cell = f"\\multirow{{{pop_row_count}}}{{*}}{{{population_translations[pop_type_display]}}}" if pop_first_row else ""
                source_cell = f"\\multirow{{{source_row_count}}}{{*}}{{{source_type.upper()}}}" if source_first_row else ""
                eos_cell = eos_translations[eos_type]
                
                # Get Bayes factors for each event
                event_cells = []
                for event in GW_EVENTS:
                    # For gw_event population, use the event name as the key
                    # For other populations, use the display type as the key
                    if pop_type_display == "gw_event":
                        pop_key = event  # Use event name (GW170817, etc.) as the population key
                    else:
                        pop_key = pop_type_display
                    
                    if (event in bayes_factors and 
                        pop_key in bayes_factors[event] and 
                        source_type in bayes_factors[event][pop_key] and 
                        eos_type in bayes_factors[event][pop_key][source_type]):
                        bf_val = bayes_factors[event][pop_key][source_type][eos_type]
                        if bf_val != 0.0:
                            event_cells.append(f"{bf_val:.2f}")
                        else:
                            event_cells.append("--")
                    else:
                        event_cells.append("--")
                
                latex_lines.append(f"{pop_cell} & {source_cell} & {eos_cell} & {' & '.join(event_cells)} \\\\")
                
                if not (pop_first_row and source_first_row):
                    latex_lines.append("\\cline{3-6}")
                
                pop_first_row = False
                source_first_row = False
            
            if not pop_first_row:
                latex_lines.append("\\cline{2-6}")
        
        latex_lines.append("\\hline")
    
    # Close table
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\end{table*}")
    
    return "\n".join(latex_lines)


def main():
    """Main function to collect all Bayes factors and save to JSON file and LaTeX table."""
    parser = argparse.ArgumentParser(description="Collect Bayes factors from GW parameter estimation runs")
    parser.add_argument('--get-JSON', action='store_true', default=False,
                        help='Generate JSON file with all Bayes factors (default: False)')
    parser.add_argument('--make-table', action='store_true', default=True,
                        help='Generate LaTeX table (default: True)')
    parser.add_argument('--no-make-table', dest='make_table', action='store_false',
                        help='Do not generate LaTeX table')
    
    args = parser.parse_args()
    
    base_dir = "../../GW_runs/"
    output_dir = "."
    json_file = os.path.join(output_dir, "all_bayes_factors.json")
    latex_file = os.path.join(output_dir, "bayes_factors_table.tex")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    if args.get_JSON:
        print(f"Scanning directory structure in: {base_dir}")
        bayes_factors = collect_all_bayes_factors(base_dir)
        print(f"Saving JSON results to: {json_file}")
        with open(json_file, 'w') as f:
            json.dump(bayes_factors, f, indent=2)
    
    if args.make_table:
        print(f"Loading Bayes factors from: {json_file}")
        if not os.path.exists(json_file):
            print(f"Error: JSON file {json_file} not found. Run with --get-JSON first.")
            return
        
        with open(json_file, 'r') as f:
            bayes_factors = json.load(f)
        
        print(f"Generating LaTeX table and saving to: {latex_file}")
        latex_table = generate_latex_table(bayes_factors)
        with open(latex_file, 'w') as f:
            f.write(latex_table)
        
        paper_dir = "/Users/Woute029/Documents/Code/projects/eos_source_classification/paper"
        if os.path.exists(paper_dir):
            paper_latex_file = os.path.join(paper_dir, "bayes_factors_table.tex")
            shutil.copy2(latex_file, paper_latex_file)
            print(f"LaTeX table copied to paper directory: {paper_latex_file}")
    
if __name__ == "__main__":
    main()