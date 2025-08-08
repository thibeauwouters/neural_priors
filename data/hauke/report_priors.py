"""
Let's see which priors were used by Hauke for these runs
"""
import numpy as np
import h5py
import json

# Setting to filter priors - only show luminosity_distance prior
FILTER_LUMINOSITY_DISTANCE_ONLY = True

GW_event_list = ["GW170817",
                 "GW170817+EM", 
                 "GW190425"
                 ]
filename_list = ["GW170817/GW170817_result.hdf5",
                 "GW170817/GW170817+EM_result.hdf5",
                 "GW190425/GW190425_result.hdf5"]

for GW_event, filename in zip(GW_event_list, filename_list):

    print(f"Processing {GW_event} data... \n\n\n")
    
    with h5py.File(filename, 'r') as f:
        # Print all root level object names (aka keys) 
        priors_raw = f['priors'][()]
        priors_str = priors_raw.decode() if isinstance(priors_raw, bytes) else str(priors_raw)
        priors_dict = json.loads(priors_str)

        for param, info in priors_dict.items():
            if param.startswith("__"):
                continue  # skip metadata
            
            # Filter for luminosity_distance only if setting is enabled
            if FILTER_LUMINOSITY_DISTANCE_ONLY and param != "luminosity_distance":
                continue
                
            print(f"Parameter: {param}")
            print(f"  Prior type: {info.get('__name__', 'N/A')}")
            print(f"  Module: {info.get('__module__', 'N/A')}")
            kwargs = info.get("kwargs", {})
            for k, v in kwargs.items():
                print(f"    {k}: {v}")
            print()