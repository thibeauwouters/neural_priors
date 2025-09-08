#!/usr/bin/env python3
"""
Efficient script to collect parameter summaries with 90% credible intervals from GW parameter estimation runs.

This script uses a source-first approach to organize data and generate LaTeX tables.
The JSON structure follows: source -> population -> eos -> event -> parameter -> {median, low, high, low_diff, high_diff}
for streamlined table generation.
"""

import os
import h5py
import json
import argparse
import shutil
import numpy as np
import arviz as az
from typing import Dict, Any

# Configuration constants
GW_EVENTS = ["GW170817", "GW190425", "GW230529"]
POPULATION_TYPES = ["uniform", "gaussian", "double_gaussian", "GW170817", "GW190425", "GW230529"]
SOURCE_TYPES = ["bns", "nsbh"]
EOS_SAMPLES_NAMES = ["radio", "radio_chiEFT", "radio_NICER", "radio_GW170817", "radio_chiEFT_NICER"]

# Parameters of interest - guaranteed to exist in HDF5 files
MONEY_PARAMETERS = [
    "mass_1_source", "mass_2_source", "mass_ratio", 
    "lambda_1", "lambda_2", "lambda_tilde", "delta_lambda_tilde",
    "luminosity_distance", "chi_eff", "chi_p"
]

# Display name mappings
POPULATION_DISPLAY = {
    "uniform": "U",
    "gaussian": "G", 
    "double_gaussian": "DG",
    "default": "Default"
}

EOS_DISPLAY = {
    "radio": "Radio",
    "radio_chiEFT": "+\\chiEFT",
    "radio_NICER": "+NICER",
    "radio_GW170817": "+GW170817",
    "radio_chiEFT_NICER": "+\\chiEFT+NICER"
}

PARAMETER_DISPLAY = {
    "mass_1_source": "$m_1^{\\mathrm{src}}$ [$M_\\odot$]",
    "mass_2_source": "$m_2^{\\mathrm{src}}$ [$M_\\odot$]",
    "mass_ratio": "$q$",
    "lambda_1": "$\\Lambda_1$",
    "lambda_2": "$\\Lambda_2$",
    "lambda_tilde": "$\\tilde{\\Lambda}$",
    "delta_lambda_tilde": "$\\delta\\tilde{\\Lambda}$",
    "luminosity_distance": "$d_L$ [Mpc]",
    "chi_eff": "$\\chi_{\\mathrm{eff}}$",
    "chi_p": "$\\chi_p$"
}


def compute_parameter_summary(samples: np.ndarray, hdi_prob: float = 0.9) -> Dict[str, float]:
    """
    Compute parameter summary statistics using arviz.
    
    Args:
        samples: 1D array of parameter samples
        hdi_prob: Probability for highest density interval (default 0.9 for 90%)
    
    Returns:
        Dict with median, low, high, low_diff, high_diff
    """
    if len(samples) == 0:
        return {"median": 0.0, "low": 0.0, "high": 0.0, "low_diff": 0.0, "high_diff": 0.0}
    
    # Compute median
    median = np.median(samples)
    
    # Compute 90% HDI using arviz
    try:
        hdi_bounds = az.hdi(samples, hdi_prob=hdi_prob)
        low_bound = float(hdi_bounds[0])
        high_bound = float(hdi_bounds[1])
    except Exception as e:
        print(f"    Warning: Could not compute HDI, using percentiles: {e}")
        # Fallback to percentiles
        alpha = 1 - hdi_prob
        low_bound = np.percentile(samples, 100 * alpha / 2)
        high_bound = np.percentile(samples, 100 * (1 - alpha / 2))
    
    # Compute differences for LaTeX formatting
    low_diff = median - low_bound
    high_diff = high_bound - median
    
    return {
        "median": float(median),
        "low": float(low_bound),
        "high": float(high_bound),
        "low_diff": float(low_diff),
        "high_diff": float(high_diff)
    }


def collect_parameter_summaries_source_first(base_dir: str = "../GW_runs/", ignore_gw170817_eos: bool = False) -> Dict[str, Any]:
    """
    Collect parameter summaries organized in source-first structure.
    Directory structure: {event}/{source}/{population}/{eos} with special case {event}/{source}/default/
    Files are HDF5 format.
    
    Returns:
        Dict with structure: source -> population -> eos -> event -> parameter -> {median, low, high, low_diff, high_diff}
        Plus 'processing_errors' key with list of errors
    """
    data = {"processing_errors": []}
    
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
                for event in GW_EVENTS:
                    data[source][pop][eos][event] = {}
                    for param in MONEY_PARAMETERS:
                        data[source][pop][eos][event][param] = {}
    
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
                                print(f"      Processing {filename}")
                                
                                # Look for posterior samples
                                posterior_key = None
                                for key in ['posterior_samples', 'posterior', 'samples']:
                                    if key in f:
                                        posterior_key = key
                                        break
                                
                                if posterior_key is None:
                                    print(f"      No posterior samples found in {filename}")
                                    continue
                                
                                posterior_data = f[posterior_key]
                                
                                # Extract each parameter
                                default_eos = eos_to_process[0] if eos_to_process else "radio"
                                for param in MONEY_PARAMETERS:
                                    samples = posterior_data[param][:]
                                    summary = compute_parameter_summary(samples)
                                    data[source][default_pop][default_eos][event][param] = summary
                                    print(f"        {param}: {summary['median']:.3f} +{summary['high_diff']:.3f} -{summary['low_diff']:.3f}")
                                
                        except (OSError, KeyError, ValueError) as e:
                            error_msg = f"Error reading {result_file}: {e}"
                            print(f"      {error_msg}")
                            data["processing_errors"].append(error_msg)
            
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
                    result_dir = os.path.join(eos_path, "outdir/result/")
                    if not os.path.exists(result_dir):
                        continue
                        
                    for filename in os.listdir(result_dir):
                        if filename.endswith("_result.hdf5"):
                            result_file = os.path.join(result_dir, filename)
                            try:
                                with h5py.File(result_file, 'r') as f:
                                    print(f"        Processing {filename}")
                                    
                                    # Look for posterior samples
                                    posterior_key = None
                                    for key in ['posterior_samples', 'posterior', 'samples']:
                                        if key in f:
                                            posterior_key = key
                                            break
                                    
                                    if posterior_key is None:
                                        print(f"        No posterior samples found in {filename}")
                                        continue
                                    
                                    posterior_data = f[posterior_key]
                                    
                                    # Extract each parameter
                                    for param in MONEY_PARAMETERS:
                                        samples = posterior_data[param][:]
                                        summary = compute_parameter_summary(samples)
                                        data[source][pop][eos][event][param] = summary
                                        print(f"          {param}: {summary['median']:.3f} +{summary['high_diff']:.3f} -{summary['low_diff']:.3f}")
                                
                            except (OSError, KeyError, ValueError) as e:
                                error_msg = f"Error reading {result_file}: {e}"
                                print(f"        {error_msg}")
                                data["processing_errors"].append(error_msg)
    
    print(f"Processing completed with {len(data['processing_errors'])} errors")
    return data


def has_parameter_data(data: Dict[str, Any], source: str, pop: str, eos: str, event: str, param: str) -> bool:
    """Check if combination has parameter data."""
    try:
        return (pop in data[source] and 
                eos in data[source][pop] and 
                event in data[source][pop][eos] and
                param in data[source][pop][eos][event] and
                data[source][pop][eos][event][param].get("median", 0.0) != 0.0)
    except (KeyError, TypeError):
        return False


def generate_latex_parameter_table(data: Dict[str, Any], ignore_gw170817_eos: bool = False) -> str:
    """Generate LaTeX table with parameter summaries for specific event-source combinations."""
    lines = []
    eos_to_process = [eos for eos in EOS_SAMPLES_NAMES if not (ignore_gw170817_eos and "GW170817" in eos)]
    
    # Define specific event-source combinations
    event_source_combinations = {
        "GW170817": "bns",
        "GW190425": "bns", 
        "GW230529": "nsbh"
    }
    
    # Populations to include (now including default)
    populations_to_include = ["uniform", "gaussian", "double_gaussian", "default"]
    
    # Table header - parameters as columns, configurations as rows (no source column)
    param_headers = " & ".join([PARAMETER_DISPLAY[param] for param in MONEY_PARAMETERS])
    lines.extend([
        "\\begin{tabular}{|l|l|l|" + "c|" * len(MONEY_PARAMETERS) + "}",
        "\\hline",
        f"\\textbf{{Event}} & \\textbf{{Pop}} & \\textbf{{EOS}} & {param_headers} \\\\",
        "\\hline\\hline"
    ])
    
    # Generate table rows for specific event-source combinations
    for event, source in event_source_combinations.items():
        if event not in GW_EVENTS:
            continue
            
        event_first = True
        
        # Count rows for this event (for multirow)
        event_rows = sum(1 for pop in populations_to_include
                        for eos in eos_to_process
                        if any(has_parameter_data(data, source, pop, eos, event, param) 
                              for param in MONEY_PARAMETERS))
        
        if event_rows == 0:
            continue
            
        for pop in populations_to_include:
            pop_first = True
            
            # Count rows for this population
            pop_rows = sum(1 for eos in eos_to_process
                          if any(has_parameter_data(data, source, pop, eos, event, param) 
                                for param in MONEY_PARAMETERS))
            
            if pop_rows == 0:
                continue
            
            for eos in eos_to_process:
                # Check if this configuration has any data
                if not any(has_parameter_data(data, source, pop, eos, event, param) 
                          for param in MONEY_PARAMETERS):
                    continue
                
                # Format cells (no source column)
                event_cell = f"\\multirow{{{event_rows}}}{{*}}{{{event}}}" if event_first else ""
                pop_cell = f"\\multirow{{{pop_rows}}}{{*}}{{{POPULATION_DISPLAY[pop]}}}" if pop_first else ""
                eos_cell = EOS_DISPLAY[eos]
                
                # Parameter value cells
                param_cells = []
                for param in MONEY_PARAMETERS:
                    if has_parameter_data(data, source, pop, eos, event, param):
                        summary = data[source][pop][eos][event][param]
                        median = summary["median"]
                        low_diff = summary["low_diff"]
                        high_diff = summary["high_diff"]
                        
                        # Format based on parameter type
                        if param in ["mass_1_source", "mass_2_source", "luminosity_distance"]:
                            param_cells.append(f"${median:.2f}_{{-{low_diff:.2f}}}^{{+{high_diff:.2f}}}$")
                        elif param in ["lambda_1", "lambda_2", "lambda_tilde", "delta_lambda_tilde"]:
                            param_cells.append(f"${median:.0f}_{{-{low_diff:.0f}}}^{{+{high_diff:.0f}}}$")
                        else:  # chi_eff, chi_p, mass_ratio
                            param_cells.append(f"${median:.3f}_{{-{low_diff:.3f}}}^{{+{high_diff:.3f}}}$")
                    else:
                        param_cells.append("--")
                
                lines.append(f"{event_cell} & {pop_cell} & {eos_cell} & {' & '.join(param_cells)} \\\\")
                
                event_first = False
                pop_first = False
            
            if not pop_first:  # Only add line if we've written rows
                lines.append("\\cline{2-" + str(3 + len(MONEY_PARAMETERS)) + "}")
        
        if not event_first:  # Only add line if we've written rows
            lines.append("\\hline\\hline")
    
    # Clean up last line
    if lines and lines[-1] == "\\hline\\hline":
        lines[-1] = "\\hline"
    
    lines.append("\\end{tabular}")
    return "\n".join(lines)


def main():
    """Main function to collect parameter summaries and generate LaTeX table."""
    parser = argparse.ArgumentParser(description="Collect parameter summaries with 90% credible intervals")
    parser.add_argument('--get-JSON', action='store_true', 
                        help='Generate JSON file with parameter summaries')
    parser.add_argument('--make-table', action='store_true', default=True,
                        help='Generate LaTeX table (default: True)')
    parser.add_argument('--no-make-table', dest='make_table', action='store_false',
                        help='Skip LaTeX table generation')
    parser.add_argument('--ignore-GW170817', action='store_true', default=True,
                        help='Ignore EOS constraints that include GW170817 data')
    
    args = parser.parse_args()
    
    # base_dir = "../GW_runs/" # This is on Nikhef, for the relative binning runs
    base_dir = "/work/wouters/neural_priors_paper_runs/" # This is on Nikhef, for the relative binning runs
    json_file = "parameter_summaries.json"
    latex_file = "parameter_summaries_table.tex"
    
    if args.get_JSON:
        print(f"Collecting parameter summaries from: {base_dir}")
        ignore_gw170817_flag = getattr(args, 'ignore_GW170817', False)
        if ignore_gw170817_flag:
            print("Ignoring EOS constraints that include GW170817 data")
        data = collect_parameter_summaries_source_first(base_dir, ignore_gw170817_flag)
        
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
        
        # Print processing statistics
        if data["processing_errors"]:
            print(f"Found {len(data['processing_errors'])} processing errors")
            for error in data["processing_errors"][:5]:  # Show first 5 errors
                print(f"  {error}")
        
        print(f"Generating LaTeX table: {latex_file}")
        table = generate_latex_parameter_table(data, getattr(args, 'ignore_GW170817', False))
        
        with open(latex_file, 'w') as f:
            f.write(table)
        
        # Copy to paper directory if it exists
        paper_dir = "/Users/Woute029/Documents/Code/projects/eos_source_classification/paper"
        if os.path.exists(paper_dir):
            paper_file = os.path.join(paper_dir, "parameter_summaries_table.tex")
            shutil.copy2(latex_file, paper_file)
            print(f"Table copied to: {paper_file}")


if __name__ == "__main__":
    main()