import numpy as np
import h5py
hauke_gw170817_posterior_filename = "../data/hauke/GW170817_result.hdf5"

# Open the file
with h5py.File(hauke_gw170817_posterior_filename, 'r') as f:
    # Print all root level object names (aka keys) 
    posterior = f['posterior']
    
    chirp_mass = posterior['chirp_mass'][:]
    mass_ratio = posterior['mass_ratio'][:]
    lambda_1 = posterior['lambda_1'][:]
    lambda_2 = posterior['lambda_2'][:]
    
    # Save it
    np.savez(hauke_gw170817_posterior_filename.replace('.hdf5', '.npz'), chirp_mass=chirp_mass, mass_ratio=mass_ratio, lambda_1=lambda_1, lambda_2=lambda_2)