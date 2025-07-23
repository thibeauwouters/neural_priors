import os
import json
import numpy as np

GW_event_list = ["GW170817", "GW190425", "GW230529"]
for GW_event in GW_event_list:
    base_path = f"../GW_runs/{GW_event}"
    
    bns_results_filename = os.path.join(base_path, "bns/bns_result.json")
    default_results_filename = os.path.join(base_path, "default/default_result.json")
    nsbh_results_filename = os.path.join(base_path, "nsbh/nsbh_result.json")
    
    # Open them and load the Bayes factorr
    try:
        with open(bns_results_filename, "r") as f:
            bns_result = json.load(f)
            bf_bns = bns_result["log_bayes_factor"]
    except FileNotFoundError:
        bf_bns = 0.0
    
    try:
        with open(default_results_filename, "r") as f:
            default_result = json.load(f)
            bf_default = default_result["log_bayes_factor"]
    except FileNotFoundError:
        bf_default = 0.0
        
    try:
        with open(nsbh_results_filename, "r") as f:
            nsbh_result = json.load(f)
            bf_nsbh = nsbh_result["log_bayes_factor"]
    except FileNotFoundError:
        bf_nsbh = 0.0
    
    # Print the Bayes factors
    print(f"Checking source classification for {GW_event}:")
    print(f"   Bayes factor for BNS: {bf_bns}")
    print(f"   Bayes factor for default: {bf_default}")
    print(f"   Bayes factor for NSBH: {bf_nsbh}\n")
    
    highest_bf = max(bf_bns, bf_default, bf_nsbh)
    if highest_bf == bf_bns:
        diff = bf_bns - max(bf_default, bf_nsbh)
        print(f"{GW_event} is classified as BNS (diff ln BF = {diff:.2f})")
    elif highest_bf == bf_default:
        diff = bf_default - max(bf_bns, bf_nsbh)
        print(f"{GW_event} is classified as default (diff ln BF = {diff:.2f})")
    elif highest_bf == bf_nsbh:
        diff = bf_nsbh - max(bf_bns, bf_default)
        print(f"{GW_event} is classified as NSBH (diff ln BF = {diff:.2f})")