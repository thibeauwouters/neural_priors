import os
import json
import numpy as np

GW_event_list = ["GW170817", "GW190425", "GW230529"]

# Save the text to a file as well:
output_filename = "./bayes_factors.txt"

# Empty the file at the start
with open(output_filename, "w") as f:
    f.write("Classification Summary of GW Events:\n\n")
    
for GW_event in GW_event_list:
    base_path = f"../GW_runs/final_results/{GW_event}"
    
    bns_results_filename = os.path.join(base_path, "bns/bns_result.json")
    default_results_filename = os.path.join(base_path, "default/default_result.json")
    nsbh_results_filename = os.path.join(base_path, "nsbh/nsbh_result.json")
    
    # Open them and load the Bayes factorr
    try:
        with open(bns_results_filename, "r") as f:
            bns_result = json.load(f)
            bf_bns = bns_result["log_bayes_factor"]
    except FileNotFoundError:
        print(f"File not found: {bns_results_filename}. Setting its Bayes factor to 0.0.")
        bf_bns = 0.0
    
    try:
        with open(default_results_filename, "r") as f:
            default_result = json.load(f)
            bf_default = default_result["log_bayes_factor"]
    except FileNotFoundError:
        print(f"File not found: {default_results_filename}. Setting its Bayes factor to 0.0.")
        bf_default = 0.0
        
    try:
        with open(nsbh_results_filename, "r") as f:
            nsbh_result = json.load(f)
            bf_nsbh = nsbh_result["log_bayes_factor"]
    except FileNotFoundError:
        print(f"File not found: {nsbh_results_filename}. Setting its Bayes factor to 0.0.")
        bf_nsbh = 0.0
    
    # Print the Bayes factors
    print(f"Checking source classification for {GW_event}:")
    print(f"   Bayes factor for BNS: {bf_bns}")
    print(f"   Bayes factor for default: {bf_default}")
    print(f"   Bayes factor for NSBH: {bf_nsbh}")
    
    highest_bf = max(bf_bns, bf_default, bf_nsbh)
    if highest_bf == bf_bns:
        diff = bf_bns - max(bf_default, bf_nsbh)
        diff_10 = diff / np.log(10)
        print(f"{GW_event} is classified as BNS (diff ln BF = {diff:.2f}, diff log10 BF = {diff_10:.2f})\n\n")
    elif highest_bf == bf_default:
        diff = bf_default - max(bf_bns, bf_nsbh)
        diff_10 = diff / np.log(10)
        print(f"{GW_event} is classified as default (diff ln BF = {diff:.2f}, diff log10 BF = {diff_10:.2f})\n\n")
    elif highest_bf == bf_nsbh:
        diff = bf_nsbh - max(bf_bns, bf_default)
        diff_10 = diff / np.log(10)
        print(f"{GW_event} is classified as NSBH (diff ln BF = {diff:.2f}, diff log10 BF = {diff_10:.2f})\n\n")
        
    with open(output_filename, "a") as f:
        f.write(f"Checking source classification for {GW_event}:\n")
        f.write(f"   Bayes factor for BNS: {bf_bns}\n")
        f.write(f"   Bayes factor for default: {bf_default}\n")
        f.write(f"   Bayes factor for NSBH: {bf_nsbh}\n")
        
        if highest_bf == bf_bns:
            f.write(f"{GW_event} is classified as BNS (diff ln BF = {diff:.2f})\n\n")
        elif highest_bf == bf_default:
            f.write(f"{GW_event} is classified as default (diff ln BF = {diff:.2f})\n\n")
        elif highest_bf == bf_nsbh:
            f.write(f"{GW_event} is classified as NSBH (diff ln BF = {diff:.2f})\n\n")