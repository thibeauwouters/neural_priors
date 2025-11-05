#!/usr/bin/env python3
"""
Efficient script to collect Bayes factors from GW parameter estimation runs.

This script uses a source-first approach to organize data and generate LaTeX tables.
The JSON structure follows: source -> population -> eos -> {event: value} format
for streamlined table generation.
"""

import os
import h5py
import json
import argparse
import shutil
import numpy as np
from typing import Dict, Any

# Configuration constants
GW_EVENTS = ["GW170817", "GW190425", "GW230529"]
POPULATION_TYPES = ["uniform", "gaussian", "double_gaussian", "GW170817", "GW190425", "GW230529"]
SOURCE_TYPES = ["bns", "nsbh"]
EOS_SAMPLES_NAMES = ["radio", "radio_chiEFT", "radio_NICER", "radio_GW170817", "radio_chiEFT_NICER"]

# Display name mappings
POPULATION_DISPLAY = {
    "uniform": "Uniform",
    "gaussian": "Gaussian", 
    "double_gaussian": "Double Gaussian"
}

EOS_DISPLAY = {
    "radio": "Heavy PSRs",
    "radio_chiEFT": "+$\\chi_{\\rm{EFT}}$",
    "radio_NICER": "+NICER",
    "radio_GW170817": "+GW170817",
    "radio_chiEFT_NICER": "+$\\chi_{\\rm{EFT}}$+NICER"
}

def collect_bayes_factors_source_first(base_dir: str = "../GW_runs/", ignore_gw170817_eos: bool = False) -> Dict[str, Any]:
    """
    Collect Bayes factors organized in source-first structure.
    Directory structure: {event}/{source}/{population}/{eos} with special case {event}/{source}/default/
    Files are now HDF5 format instead of JSON.
    
    Returns:
        Dict with structure: source -> population -> eos -> {event: bayes_factor}
        Plus 'log_evidence_errors' key with list of errors
    """
    data = {"log_evidence_errors": []}
    
    if not os.path.exists(base_dir):
        print(f"Base directory {base_dir} does not exist")
        return data
    
    # Filter EOS samples if ignoring GW170817 constraints
    eos_to_process = [eos for eos in EOS_SAMPLES_NAMES if not (ignore_gw170817_eos and "GW170817" in eos)]
    
    # Initialize source-first structure
    for source in SOURCE_TYPES:
        data[source] = {}
        for pop in POPULATION_TYPES:
            data[source][pop] = {}
            for eos in eos_to_process:
                data[source][pop][eos] = {}
    
    # Scan directory structure: {event}/{source}/{population}/{eos}
    for event in GW_EVENTS:
        event_path = os.path.join(base_dir, event)
        if not os.path.exists(event_path):
            print(f"Event directory {event_path} does not exist, skipping")
            continue
            
        print(f"Processing event: {event}")
        
        for source in SOURCE_TYPES:
            source_path = os.path.join(event_path, source)
            if not os.path.exists(source_path):
                print(f"  Source directory {source_path} does not exist, skipping")
                continue
                
            print(f"  Processing source: {source}")
            
            # Check for default directory first
            default_path = os.path.join(source_path, "default")
            if os.path.exists(default_path):
                print(f"    Processing default configuration")
                # For default, we'll use the first population type as placeholder
                default_pop = "uniform"  # or whatever makes sense for your default
                
                # Look for HDF5 result files in default directory
                for filename in os.listdir(default_path):
                    if filename.endswith("_result.h5") or filename.endswith(".h5"):
                        result_file = os.path.join(default_path, filename)
                        try:
                            with h5py.File(result_file, 'r') as f:
                                # Debug: print available keys in the HDF5 file
                                print(f"      Available keys in {filename}: {list(f.keys())}")
                                
                                # Try common HDF5 key names for Bayes factor
                                bf_val = 0.0
                                found_key = None
                                for key in ['log_bayes_factor', 'logZ', 'log_evidence', 'log_bf', 'ln_bayes_factor']:
                                    if key in f:
                                        bf_val = float(f[key][()])
                                        found_key = key
                                        break
                                
                                if found_key:
                                    print(f"      Found Bayes factor in key '{found_key}': {bf_val:.2f}")
                                else:
                                    print(f"      No Bayes factor key found in {filename}")
                                
                                # Store in default population and first EOS
                                default_eos = eos_to_process[0] if eos_to_process else "radio"
                                data[source][default_pop][default_eos][event] = bf_val
                                
                                # Collect log evidence errors
                                log_err = None
                                for err_key in ['log_evidence_err', 'logZ_err', 'log_evidence_error', 'ln_evidence_err']:
                                    if err_key in f:
                                        log_err = float(f[err_key][()])
                                        break
                                
                                if log_err is not None:
                                    data["log_evidence_errors"].append(log_err)
                                
                        except (OSError, KeyError, ValueError) as e:
                            print(f"      Error reading {result_file}: {e}")
            
            # Process structured population/eos directories
            for pop in POPULATION_TYPES:
                pop_path = os.path.join(source_path, pop)
                if not os.path.exists(pop_path):
                    continue
                    
                print(f"    Processing population: {pop}")
                
                for eos in eos_to_process:
                    eos_path = os.path.join(pop_path, eos)
                    if not os.path.exists(eos_path):
                        continue
                        
                    print(f"      Processing EOS: {eos}")
                    
                    # Look for HDF5 result files in this directory
                    eos_path = os.path.join(eos_path, "outdir/result/")
                    for filename in os.listdir(eos_path):
                        if filename.endswith("_result.hdf5"):
                            result_file = os.path.join(eos_path, filename)
                            try:
                                with h5py.File(result_file, 'r') as f:
                                    # Debug: print available keys in the HDF5 file
                                    print(f"        Available keys in {filename}: {list(f.keys())}")
                                    
                                    # Try common HDF5 key names for Bayes factor
                                    bf_val = 0.0
                                    found_key = None
                                    for key in ['log_bayes_factor', 'logZ', 'log_evidence', 'log_bf', 'ln_bayes_factor']:
                                        if key in f:
                                            bf_val = float(f[key][()])
                                            found_key = key
                                            break
                                    
                                    if found_key:
                                        print(f"        Found Bayes factor in key '{found_key}': {bf_val:.2f}")
                                    else:
                                        print(f"        No Bayes factor key found in {filename}")
                                    
                                    data[source][pop][eos][event] = bf_val
                                    
                                    # Collect log evidence errors
                                    log_err = None
                                    for err_key in ['log_evidence_err', 'logZ_err', 'log_evidence_error', 'ln_evidence_err']:
                                        if err_key in f:
                                            log_err = float(f[err_key][()])
                                            break
                                    
                                    if log_err is not None:
                                        data["log_evidence_errors"].append(log_err)
                                    
                            except (OSError, KeyError, ValueError) as e:
                                print(f"        Error reading {result_file}: {e}")
    
    print(f"Collected {len(data['log_evidence_errors'])} log evidence errors")
    return data

def get_jeffreys_color(log10_bf: float) -> str:
    """Get LaTeX color for Jeffrey's scale interpretation."""
    abs_bf = abs(log10_bf)
    if abs_bf < 0.5:
        return "jeffreysred1"
    elif abs_bf < 1.0:
        return "jeffreysred2"
    elif abs_bf < 1.5:
        return "jeffreysred3"
    elif abs_bf < 2.0:
        return "jeffreysred4"
    else:
        return "jeffreysred5"


def find_column_maxima(data: Dict[str, Any], ignore_gw170817_eos: bool = False) -> Dict[str, float]:
    """Find maximum Bayes factor value for each event column."""
    maxima = {event: float('-inf') for event in GW_EVENTS}
    eos_to_process = [eos for eos in EOS_SAMPLES_NAMES if not (ignore_gw170817_eos and "GW170817" in eos)]
    
    for source in SOURCE_TYPES:
        for pop in POPULATION_TYPES:
            for eos in eos_to_process:
                if pop in data[source] and eos in data[source][pop]:
                    for event in GW_EVENTS:
                        if event in data[source][pop][eos]:
                            val = data[source][pop][eos][event]
                            if isinstance(val, (int, float)) and val != 0.0:
                                maxima[event] = max(maxima[event], val)
    
    return maxima


def generate_latex_table(data: Dict[str, Any], replace_nsbh_zeros: bool = True, ignore_gw170817_eos: bool = False) -> str:
    """Generate LaTeX table with source-first organization."""
    lines = []
    maxima = find_column_maxima(data, ignore_gw170817_eos)
    eos_to_process = [eos for eos in EOS_SAMPLES_NAMES if not (ignore_gw170817_eos and "GW170817" in eos)]
    
    # Table header
    lines.extend([
        "\\begin{tabular}{|l|l|l|c|c|c|}",
        "\\hline",
        "\\textbf{Source} & \\textbf{Population} & \\textbf{EOS Constraints} & \\textbf{GW170817} & \\textbf{GW190425} & \\textbf{GW230529} \\\\",
        "\\hline\\hline"
    ])
    
    # Generate table rows
    for source in SOURCE_TYPES:
        source_first = True
        
        # Count rows for this source (for multirow)
        source_rows = sum(1 for pop in ["uniform", "gaussian", "double_gaussian"] 
                         for eos in eos_to_process 
                         if has_data(data, source, pop, eos))
        
        for pop in ["uniform", "gaussian", "double_gaussian"]:
            pop_first = True
            
            # Count rows for this population
            pop_rows = sum(1 for eos in eos_to_process 
                          if has_data(data, source, pop, eos))
            
            for eos in eos_to_process:
                if not has_data(data, source, pop, eos):
                    continue
                
                # Format cells
                source_cell = f"\\multirow{{{source_rows}}}{{*}}{{{source.upper()}}}" if source_first else ""
                pop_cell = f"\\multirow{{{pop_rows}}}{{*}}{{{POPULATION_DISPLAY[pop]}}}" if pop_first else ""
                eos_cell = EOS_DISPLAY[eos]
                
                # Event value cells
                event_cells = []
                for event in GW_EVENTS:
                    val = data[source][pop][eos].get(event, 0.0)
                    
                    if replace_nsbh_zeros and event == "GW170817" and source == "nsbh" and val == 0.0:
                        event_cells.append("\\cellcolor{jeffreysred5}$<-200$")
                    elif val != 0.0:
                        max_val = maxima[event]
                        if max_val != float('-inf'):
                            diff = val - max_val
                            if abs(diff) < 1e-10:  # Essentially zero difference
                                event_cells.append("\\textbf{ref.}")
                            else:
                                color = get_jeffreys_color(diff)
                                event_cells.append(f"\\cellcolor{{{color}}}${diff:+.2f}$")
                        else:
                            event_cells.append(f"${val:.2f}$")
                    else:
                        event_cells.append("--")
                
                lines.append(f"{source_cell} & {pop_cell} & {eos_cell} & {' & '.join(event_cells)} \\\\")
                
                # Add appropriate lines
                if eos == "radio":
                    lines.append("\\cline{3-6}")
                elif eos == "radio_chiEFT":
                    lines.append("\\cline{3-6}")
                
                source_first = False
                pop_first = False
            
            if not source_first:  # Only add line if we've written rows
                lines.append("\\cline{2-6}")
        
        lines.append("\\hline\\hline")
    
    # Clean up last line
    if lines and lines[-1] == "\\hline\\hline":
        lines[-1] = "\\hline"
    
    lines.append("\\end{tabular}")
    return "\n".join(lines)


def has_data(data: Dict[str, Any], source: str, pop: str, eos: str) -> bool:
    """Check if combination has any non-zero data across events."""
    if pop not in data[source] or eos not in data[source][pop]:
        return False
    
    for event in GW_EVENTS:
        val = data[source][pop][eos].get(event, 0.0)
        if val != 0.0:
            return True
    return False


def convert_to_log10(data: Dict[str, Any], ignore_gw170817_eos: bool = False) -> Dict[str, Any]:
    """Convert Bayes factors from natural log to log10."""
    converted = {"log_evidence_errors": [err / np.log(10) for err in data["log_evidence_errors"]]}
    eos_to_process = [eos for eos in EOS_SAMPLES_NAMES if not (ignore_gw170817_eos and "GW170817" in eos)]
    
    for source in SOURCE_TYPES:
        converted[source] = {}
        for pop in POPULATION_TYPES:
            converted[source][pop] = {}
            for eos in eos_to_process:
                converted[source][pop][eos] = {}
                for event in GW_EVENTS:
                    val = data[source][pop][eos].get(event, 0.0)
                    if isinstance(val, str):  # Keep strings like "<-200"
                        converted[source][pop][eos][event] = val
                    elif val != 0.0:
                        converted[source][pop][eos][event] = val / np.log(10)
                    else:
                        converted[source][pop][eos][event] = val
    
    return converted


def main():
    """Main function to collect Bayes factors and generate LaTeX table."""
    parser = argparse.ArgumentParser(description="Collect Bayes factors (source-first approach)")
    parser.add_argument('--get-JSON', action='store_true', 
                        help='Generate JSON file with Bayes factors')
    parser.add_argument('--make-table', action='store_true', default=True,
                        help='Generate LaTeX table (default: True)')
    parser.add_argument('--no-make-table', dest='make_table', action='store_false',
                        help='Skip LaTeX table generation')
    parser.add_argument('--convert-to-log10', action='store_true', default=True,
                        help='Convert from ln to log10 (default: True)')
    parser.add_argument('--replace-nsbh-zeros', action='store_true', default=True,
                        help='Replace GW170817 NSBH zeros with "<-200" (default: True)')
    parser.add_argument('--ignore-GW170817', action='store_true', default=True,
                        help='Ignore EOS constraints that include GW170817 data')
    
    args = parser.parse_args()
    
    # base_dir = "../GW_runs/" # This is on Nikhef, for the relative binning runs (first few runs)
    base_dir = "/work/wouters/neural_priors_paper_runs/" # This is on Jarvis, multibanding runs (newer runs)
    json_file = "all_bayes_factors.json"
    latex_file = "bayes_factors_table.tex"
    
    if args.get_JSON:
        print(f"Collecting Bayes factors from: {base_dir}")
        ignore_gw170817_flag = getattr(args, 'ignore_GW170817', False)
        if ignore_gw170817_flag:
            print("Ignoring EOS constraints that include GW170817 data")
        data = collect_bayes_factors_source_first(base_dir, ignore_gw170817_flag)
        
        print(f"Saving JSON to: {json_file}")
        with open(json_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    if args.make_table:
        if not os.path.exists(json_file):
            print(f"JSON file {json_file} not found. Run with --get-JSON first.")
            return
        
        print(f"Loading data from: {json_file}")
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Print log evidence statistics
        if data["log_evidence_errors"]:
            mean_err = np.mean(data["log_evidence_errors"])
            print(f"Mean log evidence error (ln): {mean_err:.4f}")
        
        if args.convert_to_log10:
            print("Converting to log10...")
            data = convert_to_log10(data, getattr(args, 'ignore_GW170817', False))
            
            if data["log_evidence_errors"]:
                mean_err = np.mean(data["log_evidence_errors"])
                print(f"Mean log evidence error (log10): {mean_err:.4f}")
        
        print(f"Generating LaTeX table: {latex_file}")
        table = generate_latex_table(data, args.replace_nsbh_zeros, getattr(args, 'ignore_GW170817', False))
        
        with open(latex_file, 'w') as f:
            f.write(table)
        
        # Copy to paper directory if it exists
        paper_dir = "/Users/Woute029/Documents/Code/projects/eos_source_classification/paper"
        if os.path.exists(paper_dir):
            paper_file = os.path.join(paper_dir, "bayes_factors_table.tex")
            shutil.copy2(latex_file, paper_file)
            print(f"Table copied to: {paper_file}")


if __name__ == "__main__":
    main()