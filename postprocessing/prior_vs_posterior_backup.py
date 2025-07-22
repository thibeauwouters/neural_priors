import os
import json
import numpy as np
import matplotlib.pyplot as plt
import corner

import sys
import bilby 
import numpy as np
from bilby.core.prior.analytical import Uniform, Sine, Cosine
from bilby.gw.prior import UniformComovingVolume
# Imports moved inline where needed
import argparse
import json
import pickle
import matplotlib.pyplot as plt
from bilby.core.utils import logger
import warnings
import signal
import time
from contextlib import contextmanager

params = {"axes.grid": True,
        "text.usetex" : True,
        "font.family" : "serif",
        "ytick.color" : "black",
        "xtick.color" : "black",
        "axes.labelcolor" : "black",
        "axes.edgecolor" : "black",
        "font.serif" : ["Computer Modern Serif"],
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "axes.labelsize": 16,
        "legend.fontsize": 16,
        "legend.title_fontsize": 16,
        "figure.titlesize": 16}

plt.rcParams.update(params)

# Improved corner kwargs
default_corner_kwargs = dict(bins=40, 
                        smooth=1., 
                        show_titles=False,
                        label_kwargs=dict(fontsize=16),
                        title_kwargs=dict(fontsize=16), 
                        # quantiles=[],
                        # levels=[0.9],
                        plot_density=True, 
                        plot_datapoints=False, 
                        fill_contours=True,
                        max_n_ticks=4, 
                        min_n_ticks=3,
                        truth_color = "red",
                        save=False)

GW_event = "GW230529"
base_path = f"../GW_runs/{GW_event}"

# Load result files to investigate structure
bns_results_filename = os.path.join(base_path, "bns/bns_result.json")
default_results_filename = os.path.join(base_path, "default/default_result.json")
nsbh_results_filename = os.path.join(base_path, "nsbh/nsbh_result.json")

# Load posterior samples
with open(bns_results_filename, "r") as f:
    bns_result = json.load(f)
    bns_posterior = bns_result['posterior']['content']

with open(default_results_filename, "r") as f:
    default_result = json.load(f)
    default_posterior = default_result['posterior']['content']

with open(nsbh_results_filename, "r") as f:
    nsbh_result = json.load(f)
    nsbh_posterior = nsbh_result['posterior']['content']

# Define parameters to plot (excluding log_likelihood and log_prior)
params_to_plot = ['chirp_mass', 'mass_ratio', 'luminosity_distance', 'geocent_time', 'lambda_1', 'lambda_2']

# Create data arrays for each run
bns_data = []
default_data = []
nsbh_data = []
labels = []

for param in params_to_plot:
    if param in bns_posterior and param in default_posterior and param in nsbh_posterior:
        bns_data.append(bns_posterior[param])
        default_data.append(default_posterior[param])
        nsbh_data.append(nsbh_posterior[param])
        labels.append(param)

# Convert to numpy arrays and transpose to get (n_samples, n_params)
bns_samples = np.array(bns_data).T
default_samples = np.array(default_data).T  
nsbh_samples = np.array(nsbh_data).T

# Create range dictionary to handle constant parameters
ranges = []
for i, param in enumerate(labels):
    # Get all values for this parameter across all runs
    all_vals = np.concatenate([bns_samples[:, i], default_samples[:, i], nsbh_samples[:, i]])
    
    if param == 'lambda_1' and np.std(nsbh_samples[:, i]) < 1e-10:
        # For constant lambda_1 in NSBH, use range from BNS and Default only
        non_zero_vals = np.concatenate([bns_samples[:, i], default_samples[:, i]])
        param_range = (np.min(non_zero_vals), np.max(non_zero_vals))
    else:
        # Use full range for all other parameters
        param_range = (np.min(all_vals), np.max(all_vals))
    
    ranges.append(param_range)

# Create corner plot with three overlaid distributions
corner_kwargs_with_range = default_corner_kwargs.copy()
corner_kwargs_with_range['range'] = ranges

DEFAULT_COLOR = 'blue'
BNS_COLOR = 'red'
NSBH_COLOR = 'green'

# Create three different corner kwargs with different colors
default_kwargs = corner_kwargs_with_range.copy()
default_kwargs.update({'color': DEFAULT_COLOR, 'alpha': 0.7, 'hist_kwargs': {'alpha': 0.7, 'color': DEFAULT_COLOR}})

bns_kwargs = corner_kwargs_with_range.copy()  
bns_kwargs.update({'color': BNS_COLOR, 'alpha': 0.7, 'hist_kwargs': {'alpha': 0.7, 'color': BNS_COLOR}})

nsbh_kwargs = corner_kwargs_with_range.copy()
nsbh_kwargs.update({'color': NSBH_COLOR, 'alpha': 0.7, 'hist_kwargs': {'alpha': 0.7, 'color': NSBH_COLOR}})

# Create the overlaid corner plot
fig = corner.corner(default_samples, labels=labels, **default_kwargs)
corner.corner(bns_samples, labels=labels, fig=fig, **bns_kwargs)
corner.corner(nsbh_samples, labels=labels, fig=fig, **nsbh_kwargs)

# Now get the prior!

# Get event configuration
reference_parameters_filename = os.path.join("../GW_runs/GW230529/reference_parameters.json")
prior_filename = os.path.join("../GW_runs/GW230529/prior.prior")

with open(reference_parameters_filename, 'r') as f:
    reference_parameters = json.load(f)
ifo_list = ['L1']
duration =  128.0
minimum_frequency = 20.0

# Use geocent_time from reference parameters if available, otherwise from config
if 'geocent_time' in reference_parameters:
    geocent_time = reference_parameters['geocent_time']
    del reference_parameters['geocent_time']  # Remove to avoid confusion later
    logger.info(f"Using geocent_time from reference parameters: {geocent_time}")
else:
    raise ValueError(f"geocent_time not found")

# Execute prior file
local_vars = {}
# This is to easily and flexibly import the necessary classes below, also for NFs
safe_globals = {
    '__builtins__': {
        'abs': abs, 'min': min, 'max': max, 'round': round,
        'int': int, 'float': float, 'str': str, 'bool': bool,
    },
    'np': np,
    'bilby': bilby,
    'Uniform': Uniform,
    'Sine': Sine, 
    'Cosine': Cosine,
    'UniformComovingVolume': UniformComovingVolume
}

with open(prior_filename, 'r') as f:
    prior_code = f.read()
exec(prior_code, safe_globals, local_vars)

# Use ConditionalPriorDict for NSBH
priors = bilby.core.prior.ConditionalPriorDict()

# Path to NSBH NF model
nf_model_path = '../NFprior/models/conditional_nsbh/model.pt'

# Add all non-lambda priors from the prior file
# For NSBH, we exclude lambda_1 (BH has no tidal deformability) and lambda_2 (will be conditional)
lambda_params = ['lambda_1', 'lambda_2']
for var_name, var_value in local_vars.items():
    if var_name not in lambda_params:
        priors[var_name] = var_value

# Add fixed lambda_1 = 0 for black hole (no tidal deformability)
from bilby.core.prior.analytical import DeltaFunction
priors['lambda_1'] = DeltaFunction(peak=0.0, name='lambda_1', latex_label='$\\Lambda_1$')

# Add conditional NF prior for lambda_2 (neutron star)
from bilby.core.prior.dict import NFConditionalPrior

priors['lambda_2'] = NFConditionalPrior(
    nf_model_path=nf_model_path,
    target_param='lambda_2',
    minimum=0.0,
    maximum=10000.0,
    latex_label='$\\Lambda_2$'
)
    
# Need to do a few conversions on these parameters 
chirp_mass = reference_parameters["chirp_mass"]
symmetric_mass_ratio = reference_parameters["symmetric_mass_ratio"]
mass_ratio = bilby.gw.conversion.symmetric_mass_ratio_to_mass_ratio(symmetric_mass_ratio)
mass_1, mass_2 = bilby.gw.conversion.chirp_mass_and_mass_ratio_to_component_masses(chirp_mass, mass_ratio)
    
# Now save the conversions
reference_parameters["mass_ratio"] = mass_ratio
reference_parameters["mass_1"] = mass_1 # TODO: source frame or detector frame? To check!
reference_parameters["mass_2"] = mass_2


# Add legend
fs = 26
plt.text(0.95, 0.95, 'Default', fontsize=fs, color=DEFAULT_COLOR, ha='center', va='center', transform=plt.gcf().transFigure)
plt.text(0.95, 0.85, 'BNS', fontsize=fs, color=BNS_COLOR, ha='center', va='center', transform=plt.gcf().transFigure)
plt.text(0.95, 0.75, 'NSBH', fontsize=fs, color=NSBH_COLOR, ha='center', va='center', transform=plt.gcf().transFigure)
plt.savefig(f'./figures/prior_vs_posterior_GW230529.pdf', dpi=300, bbox_inches='tight')
plt.close()