import numpy as np
import h5py

GW_event_list = ["GW170817", "GW190425"]
filename_list = ["GW170817/GW170817_Pade_0_result.hdf5",
                 "GW190425/GW190425_nrt3_0_result.hdf5"]

for GW_event, filename in zip(GW_event_list, filename_list):

    print(f"Processing {GW_event} data...")
    
    with h5py.File(filename, 'r') as f:
        # Print all root level object names (aka keys) 
        posterior = f['posterior']
        
        chirp_mass = posterior['chirp_mass'][:]
        mass_ratio = posterior['mass_ratio'][:]
        geocent_time = posterior['geocent_time'][:]
        lambda_1 = posterior['lambda_1'][:]
        lambda_2 = posterior['lambda_2'][:]
        
        # Save it
        np.savez(f"./{GW_event}/{GW_event}_result.npz", chirp_mass=chirp_mass, mass_ratio=mass_ratio, geocent_time=geocent_time, lambda_1=lambda_1, lambda_2=lambda_2)