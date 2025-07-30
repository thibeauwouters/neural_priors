import os
import json
import numpy as np
import argparse

def get_bayes_factors(gw_event: str, population_type: str, eos_samples_name: str = "radio", base_dir: str = "../GW_runs/"):
    """
    Calculate Bayes factors comparing different priors within a population type.
    
    Args:
        gw_event (str): GW event name (e.g., GW170817)
        population_type (str): Population type (uniform, gaussian, double_gaussian)
        eos_samples_name (str): EOS samples name (default: radio)
        base_dir (str): Base directory path
    """
    
    # For default runs in gaussian/double_gaussian, they are stored in uniform/radio/
    if population_type in ["gaussian", "double_gaussian"]:
        default_population_type = "uniform"
        default_eos_samples_name = "radio"
    else:
        default_population_type = population_type
        default_eos_samples_name = eos_samples_name
    
    # Define the priors to compare
    priors = {
        "default": (default_population_type, "default", default_eos_samples_name),
        "bns": (population_type, "bns", eos_samples_name), 
        "nsbh": (population_type, "nsbh", eos_samples_name)
    }
    
    # Store all Bayes factors
    bf_dict = {}
    
    for prior_name, (pop_type, prior_type, eos_name) in priors.items():
        # Construct path to results file
        results_path = os.path.join(base_dir, gw_event, pop_type, prior_type, eos_name, f"{prior_type}_result.json")
        
        if not os.path.exists(results_path):
            print(f"Results file not found for {prior_name}: {results_path}. Setting Bayes factor to 0.0.")
            ln_bf = 0.0
        else:
            try:
                with open(results_path, "r") as f:
                    result = json.load(f)
                    ln_bf = result["log_bayes_factor"]
            except (FileNotFoundError, KeyError) as e:
                print(f"Error loading Bayes factor for {prior_name}: {e}. Setting to 0.0.")
                ln_bf = 0.0
                
        bf_dict[prior_name] = ln_bf
    
    # Calculate relative Bayes factors (using default as reference)
    default_ln_bf = bf_dict["default"]
    relative_bf = {}
    for prior_name, ln_bf in bf_dict.items():
        if prior_name != "default":
            relative_bf[f"{prior_name}_vs_default"] = ln_bf - default_ln_bf
    
    # Also calculate BNS vs NSBH
    relative_bf["bns_vs_nsbh"] = bf_dict["bns"] - bf_dict["nsbh"]
    
    # Save results
    output_dir = os.path.join(base_dir, gw_event, population_type)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "bayes_factors.txt")
    
    with open(output_file, "w") as f:
        f.write(f"Bayes Factor Analysis for {gw_event} - {population_type} population\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("Absolute log Bayes factors:\n")
        f.write(f"{'Prior':<20}{'Log Bayes Factor':>20}\n")
        f.write("-" * 40 + "\n")
        for prior_name, ln_bf in bf_dict.items():
            f.write(f"{prior_name:<20}{ln_bf:>20.6f}\n")
        
        f.write("\nRelative log Bayes factors:\n")
        f.write(f"{'Comparison':<20}{'Log BF':>15}{'Preference':>20}\n")
        f.write("-" * 55 + "\n")
        
        for comparison, ln_bf_rel in relative_bf.items():
            if ln_bf_rel > 0:
                preference = comparison.split('_vs_')[0].upper()
            elif ln_bf_rel < 0:
                preference = comparison.split('_vs_')[1].upper()
            else:
                preference = "EQUAL"
            
            # Convert ln(BF) to log10(BF) for Jeffrey's scale
            log10_bf_rel = abs(ln_bf_rel) / np.log(10)
            
            # Jeffrey's scale interpretation
            if log10_bf_rel < 0.5:
                strength = "barely worth mentioning"
            elif log10_bf_rel < 1.0:
                strength = "substantial"
            elif log10_bf_rel < 1.5:
                strength = "strong"
            elif log10_bf_rel < 2.0:
                strength = "very strong"
            else:
                strength = "decisive"
            
            f.write(f"{comparison:<20}{ln_bf_rel:>15.6f}    {preference} ({strength})\n")
    
    print(f"\nBayes factor analysis saved to: {output_file}")
    return bf_dict, relative_bf

def main():
    parser = argparse.ArgumentParser(description="Calculate Bayes factors for GW parameter estimation results")
    parser.add_argument('--gw-event', type=str, required=True,
                        help='GW event name (e.g., GW170817)')
    parser.add_argument('--population-type', type=str, required=True,
                        choices=['uniform', 'gaussian', 'double_gaussian'],
                        help='Population type for the analysis')
    parser.add_argument('--eos-samples-name', type=str, default='radio',
                        help='EOS samples name (default: radio)')
    parser.add_argument('--base-dir', type=str, default='../GW_runs/',
                        help='Base directory path (default: ../GW_runs/)')
    
    args = parser.parse_args()
    
    get_bayes_factors(
        gw_event=args.gw_event,
        population_type=args.population_type,
        eos_samples_name=args.eos_samples_name,
        base_dir=args.base_dir
    )
    
if __name__ == "__main__":
    main()