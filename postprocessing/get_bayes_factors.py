import os
import json
import numpy as np
import argparse
from utils import get_bayes_factor_data, get_output_directory

def get_bayes_factors(gw_event: str, 
                     comparison_mode: str = "source",
                     population_type: str = "uniform", 
                     source_type: str = "bns",
                     eos_samples_name: str = "radio", 
                     base_dir: str = "../GW_runs/"):
    """
    Calculate Bayes factors comparing different dimensions.
    
    Args:
        gw_event (str): GW event name (e.g., GW170817)
        comparison_mode (str): What to compare - 'source', 'population', or 'eos'
        population_type (str): Population type (uniform, gaussian, double_gaussian)
        source_type (str): Source type (bns, nsbh, default)
        eos_samples_name (str): EOS samples name (default: radio)
        base_dir (str): Base directory path
    """
    
    # Set up fixed parameters based on comparison mode
    if comparison_mode == "source":
        fixed_params = {"population_type": population_type, "eos_samples_name": eos_samples_name}
    elif comparison_mode == "population":
        fixed_params = {"source_type": source_type, "eos_samples_name": eos_samples_name}
    elif comparison_mode == "eos":
        fixed_params = {"population_type": population_type, "source_type": source_type}
    else:
        raise ValueError(f"Invalid comparison mode: {comparison_mode}")
    
    # Load Bayes factor data for all groups in the comparison
    bf_dict = get_bayes_factor_data(gw_event, base_dir, comparison_mode, fixed_params)
    
    if not bf_dict:
        print(f"No data found for {comparison_mode} comparison")
        return {}, {}
    
    # Calculate relative Bayes factors
    relative_bf = {}
    group_names = list(bf_dict.keys())
    
    # Calculate pairwise comparisons
    for i, group1 in enumerate(group_names):
        for j, group2 in enumerate(group_names):
            if i < j:  # Avoid duplicate comparisons
                relative_bf[f"{group1}_vs_{group2}"] = bf_dict[group1] - bf_dict[group2]
    
    # Save results
    output_dir = get_output_directory(base_dir, gw_event, comparison_mode, fixed_params)
    os.makedirs(output_dir, exist_ok=True)
    
    # Create filename based on comparison mode
    if comparison_mode == "source":
        filename = f"bayes_factors_{comparison_mode}_{population_type}_{eos_samples_name}.txt"
    elif comparison_mode == "population":
        filename = f"bayes_factors_{comparison_mode}_{source_type}_{eos_samples_name}.txt"
    elif comparison_mode == "eos":
        filename = f"bayes_factors_{comparison_mode}_{population_type}_{source_type}.txt"
    
    output_file = os.path.join(output_dir, filename)
    
    with open(output_file, "w") as f:
        # Create header based on comparison mode
        if comparison_mode == "source":
            header = f"Bayes Factor Analysis for {gw_event} - comparing source types\n" + \
                    f"Fixed: population={population_type}, eos={eos_samples_name}\n"
        elif comparison_mode == "population":
            header = f"Bayes Factor Analysis for {gw_event} - comparing population types\n" + \
                    f"Fixed: source={source_type}, eos={eos_samples_name}\n"
        elif comparison_mode == "eos":
            header = f"Bayes Factor Analysis for {gw_event} - comparing EOS constraints\n" + \
                    f"Fixed: population={population_type}, source={source_type}\n"
        
        f.write(header)
        f.write("=" * 70 + "\n\n")
        
        f.write("Absolute log Bayes factors:\n")
        f.write(f"{'Group':<20}{'Log Bayes Factor':>20}\n")
        f.write("-" * 40 + "\n")
        for group_name, ln_bf in bf_dict.items():
            f.write(f"{group_name:<20}{ln_bf:>20.6f}\n")
        
        f.write("\nRelative log Bayes factors:\n")
        f.write(f"{'Comparison':<20}{'ln(BF)':>12}{'log10(BF)':>12}{'Preference':>20}\n")
        f.write("-" * 64 + "\n")
        
        for comparison, ln_bf_rel in relative_bf.items():
            if ln_bf_rel > 0:
                preference = comparison.split('_vs_')[0].upper()
            elif ln_bf_rel < 0:
                preference = comparison.split('_vs_')[1].upper()
            else:
                preference = "EQUAL"
            
            # Convert ln(BF) to log10(BF) for Jeffrey's scale
            log10_bf_rel = ln_bf_rel / np.log(10)
            abs_log10_bf_rel = abs(log10_bf_rel)
            
            # Jeffrey's scale interpretation
            if abs_log10_bf_rel < 0.5:
                strength = "barely worth mentioning"
            elif abs_log10_bf_rel < 1.0:
                strength = "substantial"
            elif abs_log10_bf_rel < 1.5:
                strength = "strong"
            elif abs_log10_bf_rel < 2.0:
                strength = "very strong"
            else:
                strength = "decisive"
            
            f.write(f"{comparison:<20}{ln_bf_rel:>12.6f}{log10_bf_rel:>12.6f}    {preference} ({strength})\n")
    
    print(f"\nBayes factor analysis saved to: {output_file}")
    return bf_dict, relative_bf

def main():
    parser = argparse.ArgumentParser(description="Calculate Bayes factors for GW parameter estimation results")
    parser.add_argument('--gw-event', type=str, required=True,
                        help='GW event name (e.g., GW170817)')
    parser.add_argument('--comparison-mode', type=str, default='source',
                        choices=['source', 'population', 'eos'],
                        help='What to compare across (default: source)')
    parser.add_argument('--population-type', type=str, default='uniform',
                        choices=['uniform', 'gaussian', 'double_gaussian', 'GW170817', 'GW190425', 'GW230529'],
                        help='Population type for the analysis (default: uniform)')
    parser.add_argument('--source-type', type=str, default='bns',
                        choices=['bns', 'nsbh', 'default'],
                        help='Source type for the analysis (default: bns)')
    parser.add_argument('--eos-samples-name', type=str, default='radio',
                        choices=['radio', 'radio_chiEFT', 'radio_chiEFT_NICER'],
                        help='EOS samples name (default: radio)')
    parser.add_argument('--base-dir', type=str, default='../GW_runs/',
                        help='Base directory path (default: ../GW_runs/)')
    
    args = parser.parse_args()
    
    get_bayes_factors(
        gw_event=args.gw_event,
        comparison_mode=args.comparison_mode,
        population_type=args.population_type,
        source_type=args.source_type,
        eos_samples_name=args.eos_samples_name,
        base_dir=args.base_dir
    )
    
if __name__ == "__main__":
    main()