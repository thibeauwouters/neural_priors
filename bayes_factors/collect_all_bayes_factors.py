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
import numpy as np
import joblib
from typing import Dict, Any

# Fixed options from cornerplots.py
GW_EVENTS = ["GW170817", "GW190425", "GW230529"]
POPULATION_TYPES = ["uniform", "gaussian", "double_gaussian", "GW170817", "GW190425", "GW230529"]
SOURCE_TYPES = ["bns", "nsbh"]
EOS_SAMPLES_NAMES = ["radio", "radio_chiEFT", "radio_chiEFT_NICER"]


def load_scaler_jacobian_correction(model_path: str) -> float:
    """
    Load the MinMaxScaler from a model directory and compute the Jacobian correction.
    
    Args:
        model_path (str): Path to the model directory containing scaler.gz
        
    Returns:
        float: Jacobian correction factor: -sum(log(data_max_ - data_min_))
    """
    scaler_path = os.path.join(model_path, "scaler.gz")
    
    if not os.path.exists(scaler_path):
        print(f"Warning: No scaler found at {scaler_path}")
        return 0.0
    
    try:
        scaler = joblib.load(scaler_path)
        
        if not hasattr(scaler, 'data_min_') or not hasattr(scaler, 'data_max_'):
            print(f"Warning: Scaler at {scaler_path} does not have expected attributes")
            return 0.0
        
        # Compute Jacobian correction: -sum(log(data_max_ - data_min_))
        data_range = scaler.data_max_ - scaler.data_min_
        jacobian_correction = -np.sum(np.log(data_range))
        
        return jacobian_correction
        
    except Exception as e:
        print(f"Error loading scaler from {scaler_path}: {e}")
        return 0.0


def get_model_path_for_run(event: str, population: str, source: str, eos: str, nf_base_dir: str = "../NFprior/models/") -> str:
    """
    Get the model path for a specific run configuration.
    
    Args:
        event (str): GW event name
        population (str): Population type
        source (str): Source type (bns/nsbh)
        eos (str): EOS type
        nf_base_dir (str): Base directory for NF models
        
    Returns:
        str: Path to the model directory
    """
    return os.path.join(nf_base_dir, population, source, eos)


def collect_all_bayes_factors(base_dir: str = "../GW_runs/", apply_jacobian_correction: bool = False, nf_base_dir: str = "../NFprior/models/") -> Dict[str, Any]:
    """
    Collect all Bayes factors from the GW runs directory structure.
    
    Args:
        base_dir (str): Base directory path (should be ../GW_runs/)
        apply_jacobian_correction (bool): Whether to apply MinMaxScaler Jacobian correction
        nf_base_dir (str): Base directory for NF models
        
    Returns:
        Dict with nested structure matching directory layout plus log_evidence_err values
    """
    all_bayes_factors = {}
    log_evidence_errors = []
    jacobian_corrections = {}
    
    if apply_jacobian_correction:
        print("\n=== APPLYING JACOBIAN CORRECTION FOR MINMAXSCALER BUG ===")
        print("This corrects for the missing Jacobian determinant in NFDist._ln_prob")
        print("Correction formula: log_evidence_corrected = log_evidence_raw + (-sum(log(data_max_ - data_min_)))")
        print("="*70)
    
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
                                
                                # Load and store Jacobian correction for this specific run
                                model_path = get_model_path_for_run(gw_event, population_type, source_type, eos_type, nf_base_dir)
                                jacobian_key = f"{population_type}_{source_type}_{eos_type}"
                                
                                if jacobian_key not in jacobian_corrections:
                                    jacobian_correction = load_scaler_jacobian_correction(model_path)
                                    jacobian_corrections[jacobian_key] = jacobian_correction
                                    if jacobian_correction != 0.0:
                                        print(f"        Jacobian correction for {jacobian_key}: {jacobian_correction:.6f}")
                                else:
                                    jacobian_correction = jacobian_corrections[jacobian_key]
                                
                                # Store the Jacobian correction for this run in the nested structure
                                jacobian_correction_key = f"{source_type}_jacobian_correction"
                                if jacobian_correction_key not in all_bayes_factors[gw_event][population_type]:
                                    all_bayes_factors[gw_event][population_type][jacobian_correction_key] = {}
                                all_bayes_factors[gw_event][population_type][jacobian_correction_key][eos_type] = jacobian_correction
                                
                                # Apply Jacobian correction if requested
                                if apply_jacobian_correction and bayes_factor != 0.0:
                                    # Apply correction to log evidence (Bayes factor is difference of log evidences)
                                    # Since different models have different corrections, we need to track them separately
                                    corrected_bayes_factor = bayes_factor + jacobian_correction
                                    print(f"        Raw Bayes factor: {bayes_factor:.6f}, Corrected: {corrected_bayes_factor:.6f} (correction: {jacobian_correction:.6f})")
                                    bayes_factor = corrected_bayes_factor
                                
                                all_bayes_factors[gw_event][population_type][source_type][eos_type] = bayes_factor
                                
                                # Collect log_evidence_err if available
                                log_evidence_err = result_data.get("log_evidence_err")
                                if log_evidence_err is not None:
                                    log_evidence_errors.append(log_evidence_err)
                                
                                if not apply_jacobian_correction:
                                    print(f"        Found Bayes factor: {bayes_factor}")
                        except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
                            print(f"        Error reading {result_file}: {e}. Setting to 0.0")
                            all_bayes_factors[gw_event][population_type][source_type][eos_type] = 0.0
                    else:
                        print(f"        Result file not found: {result_file}. Setting to 0.0")
                        all_bayes_factors[gw_event][population_type][source_type][eos_type] = 0.0
    
    # Add log_evidence_errors and jacobian_corrections to the output dictionary
    all_bayes_factors["log_evidence_errors"] = log_evidence_errors
    all_bayes_factors["jacobian_corrections"] = jacobian_corrections
    print(f"Collected {len(log_evidence_errors)} log_evidence_err values")
    
    # if apply_jacobian_correction:
    #     print(f"\nApplied Jacobian corrections for {len(jacobian_corrections)} model configurations:")
    #     for key, correction in jacobian_corrections.items():
    #         print(f"  {key}: {correction:.6f}")
    
    return all_bayes_factors


def get_jeffreys_color(log10_bf: float) -> str:
    """Get color name for Jeffrey's scale interpretation of log10 Bayes factor."""
    abs_bf = abs(log10_bf)
    if abs_bf < 0.5:
        return "jeffreysred1"  # barely worth mentioning - lightest red
    elif abs_bf < 1.0:
        return "jeffreysred2"  # substantial - light red
    elif abs_bf < 1.5:
        return "jeffreysred3"  # strong - medium red
    elif abs_bf < 2.0:
        return "jeffreysred4"  # very strong - dark red
    else:
        return "jeffreysred5"  # decisive - darkest red


def generate_latex_table(bayes_factors: Dict[str, Any],
                         include_gw_event: bool = False,
                         source_first: bool = False,
                         apply_jacobian_correction: bool = True) -> str:
    """
    Generate LaTeX table code for the Bayes factors data.
    
    Args:
        bayes_factors: Nested dictionary with Bayes factors data
        include_gw_event: Include GW-event specific population priors
        source_first: Organize table with source types first, then population types
        
    Returns:
        String containing LaTeX table code
    """
    latex_lines = []
    
    # First, collect all Bayes factor values for each event to find column maxima
    event_max_values = {event: float('-inf') for event in GW_EVENTS}
    
    # Scan all data to find maximum values per event column
    population_types_display = ["uniform", "gaussian", "double_gaussian", "gw_event"]
    gw_event_populations = ["GW170817", "GW190425", "GW230529"]
    
    for pop_type_display in population_types_display:
        if pop_type_display == "gw_event" and include_gw_event:
            pop_types_to_check = gw_event_populations
        elif pop_type_display == "gw_event":
            continue  # Skip if gw_event not included
        else:
            pop_types_to_check = [pop_type_display]
            
        for source_type in SOURCE_TYPES:
            for eos_type in EOS_SAMPLES_NAMES:
                for event in GW_EVENTS:
                    if pop_type_display == "gw_event" and include_gw_event:
                        pop_key = event
                    elif pop_type_display == "gw_event":
                        continue  # Skip if gw_event not included
                    else:
                        pop_key = pop_type_display
                    
                    if (event in bayes_factors and 
                        pop_key in bayes_factors[event] and 
                        source_type in bayes_factors[event][pop_key] and 
                        eos_type in bayes_factors[event][pop_key][source_type] and
                        not source_type.endswith('_jacobian_correction')):
                        bf_val = bayes_factors[event][pop_key][source_type][eos_type]
                        if bf_val != 0.0:
                            event_max_values[event] = max(event_max_values[event], bf_val)
    
    # Choose table layout based on source_first flag
    if source_first:
        return generate_source_first_table(bayes_factors, event_max_values, include_gw_event, 
                                          population_types_display, gw_event_populations)
    else:
        return generate_population_first_table(bayes_factors, event_max_values, include_gw_event,
                                              population_types_display, gw_event_populations)


def generate_population_first_table(bayes_factors, event_max_values, include_gw_event, 
                                   population_types_display, gw_event_populations):
    """Generate table with population first, then source, then EOS."""
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
        "radio": "Radio",
        "radio_chiEFT": "+\\chiEFT",
        "radio_chiEFT_NICER": "+NICER"
    }
    
    # Group population types for display
    if include_gw_event:
        population_types_display = ["uniform", "gaussian", "double_gaussian", "gw_event"]
    else:
        population_types_display = ["uniform", "gaussian", "double_gaussian"]
    
    # Create column specification: Population | Source | EOS | GW170817 | GW190425 | GW230529
    latex_lines.append("\\begin{tabular}{|l|l|l|c|c|c|}")
    latex_lines.append("\\hline")
    latex_lines.append("\\textbf{Population} & \\textbf{Source} & \\textbf{EOS Constraints} & \\textbf{GW170817} & \\textbf{GW190425} & \\textbf{GW230529} \\\\")
    latex_lines.append("\\hline\\hline")
    
    # Generate table rows
    for pop_type_display in population_types_display:
        pop_first_row = True
        pop_row_count = 0
        
        # Determine which population types to check for this display type
        if pop_type_display == "gw_event" and include_gw_event:
            pop_types_to_check = gw_event_populations
        elif pop_type_display == "gw_event":
            continue  # Skip if gw_event not included
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
                    if pop_type_display == "gw_event" and include_gw_event:
                        pop_key = event  # Use event name (GW170817, etc.) as the population key
                    elif pop_type_display == "gw_event":
                        continue  # Skip if gw_event not included
                    else:
                        pop_key = pop_type_display
                    
                    if (event in bayes_factors and 
                        pop_key in bayes_factors[event] and 
                        source_type in bayes_factors[event][pop_key] and 
                        eos_type in bayes_factors[event][pop_key][source_type]):
                        bf_val = bayes_factors[event][pop_key][source_type][eos_type]
                        if bf_val != 0.0:
                            # Show difference from maximum value in this column
                            max_val = event_max_values[event]
                            if max_val != float('-inf'):
                                diff = bf_val - max_val
                                if diff == 0.0:
                                    event_cells.append(f"\\textbf{{ref.}}")  # Show actual value for maximum
                                else:
                                    # Add colored background based on Jeffrey's scale
                                    color = get_jeffreys_color(diff)
                                    event_cells.append(f"\\cellcolor{{{color}}}${diff:+.2f}$")  # Show difference with full cell coloring
                            else:
                                event_cells.append(f"${bf_val:.2f}$")
                        else:
                            event_cells.append("--")
                    else:
                        event_cells.append("--")
                
                latex_lines.append(f"{pop_cell} & {source_cell} & {eos_cell} & {' & '.join(event_cells)} \\\\")
                
                # Add line between radio and +chiEFT for BNS
                if source_type == "bns" and eos_type == "radio":
                    latex_lines.append("\\cline{3-6}")
                elif not (pop_first_row and source_first_row):
                    latex_lines.append("\\cline{3-6}")
                
                pop_first_row = False
                source_first_row = False
            
            if not pop_first_row:
                latex_lines.append("\\cline{2-6}")
        
        latex_lines.append("\\hline\\hline")
    
    # Replace the last double hline with single hline
    if latex_lines and latex_lines[-1] == "\\hline\\hline":
        latex_lines[-1] = "\\hline"
    
    # Close table
    latex_lines.append("\\end{tabular}")
    
    return "\n".join(latex_lines)


def generate_source_first_table(bayes_factors, event_max_values, include_gw_event,
                               population_types_display, gw_event_populations):
    """Generate table with source first, then population, then EOS."""
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
        "radio": "Radio",
        "radio_chiEFT": "+\\chiEFT",
        "radio_chiEFT_NICER": "+NICER"
    }
    
    # Group population types for display
    if include_gw_event:
        population_types_display = ["uniform", "gaussian", "double_gaussian", "gw_event"]
    else:
        population_types_display = ["uniform", "gaussian", "double_gaussian"]
    
    # Create column specification: Source | Population | EOS | GW170817 | GW190425 | GW230529
    latex_lines.append("\\begin{tabular}{|l|l|l|c|c|c|}")
    latex_lines.append("\\hline")
    latex_lines.append("\\textbf{Source} & \\textbf{Population} & \\textbf{EOS Constraints} & \\textbf{GW170817} & \\textbf{GW190425} & \\textbf{GW230529} \\\\")
    latex_lines.append("\\hline\\hline")
    
    # Generate table rows - source first
    for source_type in SOURCE_TYPES:
        source_first_row = True
        source_row_count = 0
        
        # Count total rows for this source type to use multirow
        for pop_type_display in population_types_display:
            if pop_type_display == "gw_event" and include_gw_event:
                pop_types_to_check = gw_event_populations
            elif pop_type_display == "gw_event":
                continue  # Skip if gw_event not included
            else:
                pop_types_to_check = [pop_type_display]
                
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
                    source_row_count += 1
        
        if source_row_count == 0:
            continue
            
        for pop_type_display in population_types_display:
            pop_first_row = True
            pop_row_count = 0
            
            # Determine which population types to check for this display type
            if pop_type_display == "gw_event" and include_gw_event:
                pop_types_to_check = gw_event_populations
            elif pop_type_display == "gw_event":
                continue  # Skip if gw_event not included
            else:
                pop_types_to_check = [pop_type_display]
            
            # Count rows for this population type
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
                    pop_row_count += 1
            
            if pop_row_count == 0:
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
                
                # Format the row - note order change: source, population, eos
                source_cell = f"\\multirow{{{source_row_count}}}{{*}}{{{source_type.upper()}}}" if source_first_row else ""
                pop_cell = f"\\multirow{{{pop_row_count}}}{{*}}{{{population_translations[pop_type_display]}}}" if pop_first_row else ""
                eos_cell = eos_translations[eos_type]
                
                # Get Bayes factors for each event
                event_cells = []
                for event in GW_EVENTS:
                    # For gw_event population, use the event name as the key
                    # For other populations, use the display type as the key
                    if pop_type_display == "gw_event" and include_gw_event:
                        pop_key = event  # Use event name (GW170817, etc.) as the population key
                    elif pop_type_display == "gw_event":
                        continue  # Skip if gw_event not included
                    else:
                        pop_key = pop_type_display
                    
                    if (event in bayes_factors and 
                        pop_key in bayes_factors[event] and 
                        source_type in bayes_factors[event][pop_key] and 
                        eos_type in bayes_factors[event][pop_key][source_type]):
                        bf_val = bayes_factors[event][pop_key][source_type][eos_type]
                        if bf_val != 0.0:
                            # Show difference from maximum value in this column
                            max_val = event_max_values[event]
                            if max_val != float('-inf'):
                                diff = bf_val - max_val
                                if diff == 0.0:
                                    event_cells.append(f"\\textbf{{ref.}}")  # Show actual value for maximum
                                else:
                                    # Add colored background based on Jeffrey's scale
                                    color = get_jeffreys_color(diff)
                                    event_cells.append(f"\\cellcolor{{{color}}}${diff:+.2f}$")  # Show difference with full cell coloring
                            else:
                                event_cells.append(f"${bf_val:.2f}$")
                        else:
                            event_cells.append("--")
                    else:
                        event_cells.append("--")
                
                latex_lines.append(f"{source_cell} & {pop_cell} & {eos_cell} & {' & '.join(event_cells)} \\\\")
                
                # Add line between radio and +chiEFT
                if eos_type == "radio":
                    latex_lines.append("\\cline{3-6}")
                elif not (source_first_row and pop_first_row):
                    latex_lines.append("\\cline{3-6}")
                
                source_first_row = False
                pop_first_row = False
            
            if not source_first_row:
                latex_lines.append("\\cline{2-6}")
        
        latex_lines.append("\\hline\\hline")
    
    # Replace the last double hline with single hline
    if latex_lines and latex_lines[-1] == "\\hline\\hline":
        latex_lines[-1] = "\\hline"
    
    # Close table
    latex_lines.append("\\end{tabular}")
    
    return "\n".join(latex_lines)


def convert_bayes_factors_to_log10(bayes_factors: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert all Bayes factors and log evidence errors from natural log to log10.
    
    Args:
        bayes_factors: Nested dictionary with Bayes factors data and log_evidence_errors
        
    Returns:
        Dict with converted Bayes factors and log evidence errors
    """
    converted = {}
    
    for event in bayes_factors:
        # Handle the log_evidence_errors entry separately
        if event == "log_evidence_errors":
            # Convert log evidence errors from ln to log10
            converted[event] = [err / np.log(10) for err in bayes_factors[event]]
            continue
        
        if event == "jacobian_corrections":
            # TODO: not sure, skip for now
            continue
            
        converted[event] = {}
        for population in bayes_factors[event]:
            converted[event][population] = {}
            for source in bayes_factors[event][population]:
                converted[event][population][source] = {}
                for eos in bayes_factors[event][population][source]:
                    bf_val = bayes_factors[event][population][source][eos]
                    if bf_val != 0.0:
                        # Convert from ln to log10: log10(x) = ln(x) / ln(10)
                        converted[event][population][source][eos] = bf_val / np.log(10)
                    else:
                        converted[event][population][source][eos] = bf_val
    
    return converted


def main():
    """Main function to collect all Bayes factors and save to JSON file and LaTeX table."""
    parser = argparse.ArgumentParser(description="Collect Bayes factors from GW parameter estimation runs")
    parser.add_argument('--get-JSON', action='store_true', default=False,
                        help='Generate JSON file with all Bayes factors (default: False)')
    parser.add_argument('--make-table', action='store_true', default=True,
                        help='Generate LaTeX table (default: True)')
    parser.add_argument('--no-make-table', dest='make_table', action='store_false',
                        help='Do not generate LaTeX table')
    parser.add_argument('--convert-to-log10', action='store_true', default=True,
                        help='Convert Bayes factors from ln to log10 (default: True)')
    parser.add_argument('--include-gw-event', action='store_true', default=False,
                        help='Include GW-event specific population priors in table (default: False)')
    parser.add_argument('--source-first', action='store_true', default=True,
                        help='Organize table with source types first, then population types (default: False)')
    parser.add_argument('--apply-jacobian-correction', action='store_true', default=True,
                        help='Apply MinMaxScaler Jacobian correction to fix NFDist normalization bug (default: False)')
    parser.add_argument('--nf-base-dir', default='../NFprior/models/',
                        help='Base directory for NF models (default: ../NFprior/models/)')
    
    args = parser.parse_args()
    
    base_dir = "../GW_runs/"
    output_dir = "."
    json_file = os.path.join(output_dir, "all_bayes_factors.json")
    latex_file = os.path.join(output_dir, "bayes_factors_table.tex")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    if args.get_JSON:
        print(f"Scanning directory structure in: {base_dir}")
        if args.apply_jacobian_correction:
            print(f"NF models directory: {args.nf_base_dir}")
        bayes_factors = collect_all_bayes_factors(base_dir, apply_jacobian_correction=args.apply_jacobian_correction, nf_base_dir=args.nf_base_dir)
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
        
        # Print mean log evidence error before conversion
        if "log_evidence_errors" in bayes_factors and bayes_factors["log_evidence_errors"]:
            mean_ln_err = np.mean(bayes_factors["log_evidence_errors"])
            print(f"Mean log evidence error (ln): {mean_ln_err:.4f}")
        
        # Print Jacobian correction summary if available
        if "jacobian_corrections" in bayes_factors and bayes_factors["jacobian_corrections"]:
            corrections = list(bayes_factors["jacobian_corrections"].values())
            if corrections:
                print(f"Jacobian correction range: [{min(corrections):.6f}, {max(corrections):.6f}]")
                print(f"Mean Jacobian correction: {np.mean(corrections):.6f}")
        
        # Convert to log10 if requested
        if args.convert_to_log10:
            print("WARNING: Converting Bayes factors from ln to log10")
            bayes_factors = convert_bayes_factors_to_log10(bayes_factors)
            
            # Print mean log evidence error after conversion
            if "log_evidence_errors" in bayes_factors and bayes_factors["log_evidence_errors"]:
                mean_log10_err = np.mean(bayes_factors["log_evidence_errors"])
                print(f"Mean log evidence error (log10): {mean_log10_err:.4f}")
                
        print(f"Generating LaTeX table and saving to: {latex_file}. Source_first = {args.source_first}")
        latex_table = generate_latex_table(bayes_factors,
                                           include_gw_event=args.include_gw_event,
                                           source_first=args.source_first,
                                           apply_jacobian_correction=args.apply_jacobian_correction)
        with open(latex_file, 'w') as f:
            f.write(latex_table)
        
        paper_dir = "/Users/Woute029/Documents/Code/projects/eos_source_classification/paper"
        if os.path.exists(paper_dir):
            paper_latex_file = os.path.join(paper_dir, "bayes_factors_table.tex")
            shutil.copy2(latex_file, paper_latex_file)
            print(f"LaTeX table copied to paper directory: {paper_latex_file}")
    
if __name__ == "__main__":
    main()