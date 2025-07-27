'''
Script to extract injection parameters that are taken from a `default` analysis of an event for most parameters.
The tidal deformabilities, on the other hand, are extracted from a given EOS file, which is expected to be in a specific format.

# TODO: we should also allow this to create "random" injections, but first, focus on creating the event-specific injections.
'''

# Put on some flags since MPI seems to mess up with the NF priors
import os
import shutil
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["MKL_DYNAMIC"] = "FALSE"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import argparse
import json

from bilby.gw.conversion import lambda_1_lambda_2_to_lambda_tilde, lambda_1_lambda_2_to_delta_lambda_tilde
from bilby.gw.conversion import chirp_mass_and_mass_ratio_to_component_masses
from bilby.gw.conversion import luminosity_distance_to_redshift # FIXME: what cosmology?

#############
### UTILS ###
#############

def load_eos(eos_name: str = 'jester') -> tuple[np.array, np.array]:
    """
    Load the EOS from a given file.
    Note: should adhere to a specific format, therefore, likely requires a small step to put another EOS in this format.
    Use it to extract the mass and tidal deformabilities arrays, to create GW injections

    Args:
        eos_filename (str): Filename of the EOS file to load. Default is '../data/eos/jester.npz'.
    Returns:
        tuple: A tuple containing two numpy arrays: mass and tidal deformabilities.
    """
    
    eos_filename = f'../data/eos/{eos_name}.npz'
    data = np.load(eos_filename)
    if "masses_EOS" not in data or "Lambdas_EOS" not in data:
        raise ValueError(f"Expected keys 'masses_EOS' and 'Lambdas_EOS' not found in {eos_filename}.")
    masses_EOS, Lambdas_eos = data['masses_EOS'], data['Lambdas_EOS']
    return masses_EOS, Lambdas_eos

################
### ARGPARSE ###
################

parser = argparse.ArgumentParser()
parser.add_argument('--GW-event',
                    type=str,
                    default='GW170817',
                    help = "GW event from which to copy the setup, such as the prior files.")
parser.add_argument('--eos-name',
                    type=str,
                    default='jester',
                    help="Name of the EOS to be loaded, that also has to correspond with the filename of the EOS file in ../data/eos/ directory.")
parser.add_argument('--source-type',
                    type=str,
                    default='bns',
                    help="What kind of source to inject, e.g. bns, nsbh.") # TODO: add nsbh_primary as well

def main(args):
    
    # Locate the base directory of the default GW analysis of the real event
    # TODO: we might want to update this path?
    results_filename = f"../GW_runs/final_results/{args.GW_event}/default/default_result.json"
    if not os.path.exists(results_filename):
        raise FileNotFoundError(f"Results file {results_filename} does not exist. Please check the GW event name or the path building.")
    
    print(f"Loading results from {results_filename} for the {args.GW_event} event.")
    with open(results_filename, 'r') as f:
        # Load posterior and its probability distribution
        results = json.load(f)
        posterior = results["posterior"]['content']
        log_likelihood = np.array(posterior["log_likelihood"][:])
        # log_prior = np.array(posterior["log_prior"][:])
        # log_posterior = log_prior + log_likelihood
        
        # Compute median for each parameter separately
        keys_to_skip = ['geocent_time', 'lambda_1', 'lambda_2', 'log_likelihood', 'log_prior']
        median_params = {key: np.median(posterior[key]) for key in posterior.keys() if key not in keys_to_skip}
        
    # Add the geocent time from the reference parameters, which is identical to the one usually used in config.ini files
    original_ref_params_filename = f"../GW_runs/{args.GW_event}/reference_parameters.json"
    with open(original_ref_params_filename, 'r') as f:
        ref_params = json.load(f)
        median_params['geocent_time'] = ref_params['geocent_time']
        
    # Convert to source-frame component masses
    z = luminosity_distance_to_redshift(median_params['luminosity_distance'])
    m1, m2 = chirp_mass_and_mass_ratio_to_component_masses(
        median_params['chirp_mass'],
        median_params['mass_ratio'])
    median_params['mass_1'] = m1
    median_params['mass_2'] = m2
    m1_src = m1 / (1 + z)
    m2_src = m2 / (1 + z)
    print(f"Source-frame component masses: m1 = {m1_src:.2f} M_sun, m2 = {m2_src:.2f} M_sun")
    
    # Compute the tidal deformabilities, first, load the EOS
    masses_EOS, Lambdas_eos = load_eos(args.eos_name)
    lambda_1 = 0.0
    lambda_2 = 0.0
    
    if args.source_type == 'bns':
        lambda_1 = np.interp(m1_src, masses_EOS, Lambdas_eos)
        lambda_2 = np.interp(m2_src, masses_EOS, Lambdas_eos)
    
    elif args.source_type == 'nsbh':
        # For NSBH, we assume the primary is a BH and the secondary is the NS
        lambda_2 = np.interp(m2_src, masses_EOS, Lambdas_eos)
    else:
        raise ValueError(f"Unsupported source type: {args.source_type}. Supported types are 'bns' and 'nsbh'.")
    
    median_params['lambda_1'] = lambda_1
    median_params['lambda_2'] = lambda_2
        
    print(f"Interpolated tidal deformabilities: lambda_1 = {lambda_1:.2f}, lambda_2 = {lambda_2:.2f}")
            
    print(f"Extracted the following parameters from the {args.GW_event} event:")
    for key, value in median_params.items():
        print(f"    {key}: {value}")
        
    # Save to a directory with a name uniquely built from the most important settings put here
    dir_name = f"./{args.GW_event}_{args.source_type}_{args.eos_name}"
    os.makedirs(dir_name, exist_ok=True)
    
    # Put the parameters into a JSON file
    output_filename = os.path.join(dir_name, 'injection_parameters.json')
    with open(output_filename, 'w') as f:
        json.dump(median_params, f, indent=4)
        print(f"Saved injection parameters to {output_filename}")
        
    # Finally, copy the prior.prior file from the original event's prior directory
    prior_filename = f"../GW_runs/{args.GW_event}/prior.prior"
    if not os.path.exists(prior_filename):
        raise FileNotFoundError(f"Prior file {prior_filename} does not exist. Please check the GW event name or the path building.")
    output_prior_filename = os.path.join(dir_name, 'prior.prior')
    
    # Copy with shutil or something 
    shutil.copy(prior_filename, output_prior_filename)
    print(f"Copied prior file to {output_prior_filename}")
    
if __name__ == "__main__":
    args = parser.parse_args()
    main(args)