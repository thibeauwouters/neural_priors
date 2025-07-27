import numpy as np
import h5py
from bilby.gw.conversion import lambda_1_lambda_2_to_lambda_tilde, lambda_1_lambda_2_to_delta_lambda_tilde
from bilby.gw.conversion import chirp_mass_and_mass_ratio_to_component_masses

GW_event_list = ["GW170817", 
                 "GW190425"
                 ]

filename_list = ["./GW170817/GW170817_result.hdf5",
                 "./GW190425/GW190425_result.hdf5"
                 ]

keys_to_extract = ['chirp_mass', 'mass_ratio', 'luminosity_distance', 'geocent_time', 
                   'a_1', 'a_2', 'tilt_1', 'tilt_2', 'phi_12', 'phi_jl', 
                   'dec', 'ra', 'theta_jn', 'psi', 'phase', 'lambda_1', 'lambda_2']

for GW_event, filename in zip(GW_event_list, filename_list):
    with h5py.File(filename, 'r') as f:
        # Load the data
        posterior = f['posterior']
        data = {key: posterior[key][:] for key in keys_to_extract}
        
        # Do the conversions for lambda_tilde and delta_lambda_tilde
        mass_1, mass_2 = chirp_mass_and_mass_ratio_to_component_masses(data["chirp_mass"], data["mass_ratio"])
        lambda_tilde = lambda_1_lambda_2_to_lambda_tilde(data["lambda_1"], data["lambda_2"], mass_1, mass_2)
        delta_lambda_tilde = lambda_1_lambda_2_to_delta_lambda_tilde(data["lambda_1"], data["lambda_2"], mass_1, mass_2)
        
        data["lambda_tilde"] = lambda_tilde
        data["delta_lambda_tilde"] = delta_lambda_tilde
        
        # Save it
        save_name = f"./{GW_event}/{GW_event}_result.npz"
        np.savez(save_name, **data)
        print(f"Data for {GW_event} saved to {save_name}")