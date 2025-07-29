import os
import json
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--run-dir',
                    type = str,
                    required = True,
                    help = "Directory containing all the runs as subdirs, and we will save the classification summary there as well.")

def main():
    args = parser.parse_args()
    run_dir = args.run_dir
    if not os.path.exists(run_dir):
        raise FileNotFoundError(f"The specified run directory does not exist: {run_dir}")
    
    output_file = os.path.join(run_dir, "bayes_factors.txt")
    subdirs = [d for d in os.listdir(run_dir) if os.path.isdir(os.path.join(run_dir, d))]
    subdirs = [d for d in subdirs if "dag" not in d and "figures" not in d]
    
    print(f"Found {len(subdirs)} subdirectories in {run_dir}: {subdirs}.")
    
    # Store all the rundir's Bayes factors in a dictionary
    bf_dict = {}
    for subdir in subdirs:
        # Locate results file
        full_dir = os.path.join(run_dir, subdir)
        results_filename = os.path.join(full_dir, f"{subdir}_result.json")
        
        if not os.path.exists(results_filename):
            print(f"Results file not found for {subdir}: {results_filename}. Skipping this directory.")
            continue
        
        # Open them and load the Bayes factor
        try:
            with open(results_filename, "r") as f:
                result = json.load(f)
                ln_bf = result["log_bayes_factor"]
        except FileNotFoundError:
            print(f"File not found: {results_filename}. Setting its Bayes factor to 0.0.")
            ln_bf = 0.0
            
        # Store it
        bf_dict[subdir] = ln_bf
        
    # Sort all the Bayes factors from highest to lowest
    sort_idx = np.argsort(list(bf_dict.values()))[::-1]
    sorted_bf = {k: bf_dict[k] for k in np.array(list(bf_dict.keys()))[sort_idx]}
    
    # Save to the txt: one column is subdir, the other is the Bayes factor
    with open(output_file, "w") as f:
        f.write(f"{'Subdirectory':<30}{'Log Bayes Factor':>20}\n")
        f.write("-" * 50 + "\n")
        for subdir, ln_bf in sorted_bf.items():
            f.write(f"{subdir:<30}{ln_bf:>20.6f}\n")

            
    print("Saved Bayes factors to:", output_file)
    
if __name__ == "__main__":
    main()