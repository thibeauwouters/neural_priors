import numpy as np
import matplotlib.pyplot as plt

import os
import utils
import arviz
import numpy as np

from plot_priors import *

from utils import (
    load_comparison_data, construct_result_path, load_posterior_data, 
    load_cosmology_interpolator, setup_matplotlib_style, PARAMETER_LATEX_LABELS, 
    DEFAULT_CORNER_KWARGS, VERBOSE, DEFAULT_COLOR, BNS_COLOR, NSBH_COLOR, 
    HAUKE_COLOR, HAUKE_EM_COLOR, ADRIAN_COLOR, load_hauke_data, load_adrian_data
)

from bilby.gw.conversion import lambda_1_lambda_2_to_lambda_tilde, lambda_1_lambda_2_to_delta_lambda_tilde
from bilby.gw.conversion import chirp_mass_and_mass_ratio_to_component_masses
from bilby.gw.conversion import luminosity_distance_to_redshift

def plot_log_priors():
    """Plot histogram of log_prior values from posterior samples."""
    base_dir = "../final_results/"
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for pop in utils.POPULATION_NAMES:
        path = construct_result_path(base_dir, "GW170817", pop, "BNS", "radio")
        samples = load_posterior_data(path, fast_mode=True)
        
        if 'log_prior' in samples:
            ax.hist(samples['log_prior'], bins=50, alpha=0.7, label=f'{pop}', density=True)
    
    ax.set_xlabel('log_prior')
    ax.set_ylabel('Density')
    ax.set_title('Distribution of log_prior values')
    ax.legend()
    plt.savefig("./figures/GW170817/log_prior_distribution.pdf", bbox_inches='tight')
    plt.close()

def plot_lambda_tilde_comparisons():
    base_dir = "../final_results/"
    for pop in utils.POPULATION_NAMES:
        path = construct_result_path(base_dir, "GW170817", pop, "BNS", "radio")
        samples = load_posterior_data(path, fast_mode=True)
        print(samples.keys())
        
    

def main():
    plot_log_priors()
    plot_lambda_tilde_comparisons()
    
if __name__ == "__main__":
    main()