import os
import json
import numpy as np
import matplotlib.pyplot as plt
import corner
import torch

from bilby.core.prior.dict import NFConditionalPrior

params = {"axes.grid": False,
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

DEFAULT_COLOR = 'blue'
BNS_COLOR = 'red'
NSBH_COLOR = 'green'
PRIOR_COLOR = "gray"

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

for param in params_to_plot:
    if param in bns_posterior and param in default_posterior and param in nsbh_posterior:
        bns_data.append(bns_posterior[param])
        default_data.append(default_posterior[param])
        nsbh_data.append(nsbh_posterior[param])

# Convert to numpy arrays and transpose to get (n_samples, n_params)
bns_samples = np.array(bns_data).T
default_samples = np.array(default_data).T  
nsbh_samples = np.array(nsbh_data).T

# Create range dictionary to handle constant parameters
ranges = []
for i, param in enumerate(params_to_plot):
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

nf_model_path = '../NFprior/models/conditional_nsbh/model.pt'

# priors = bilby.core.prior.ConditionalPriorDict()
nf_prior = NFConditionalPrior(
    nf_model_path=nf_model_path,
    target_param='lambda_2',
    minimum=0.0,
    maximum=10000.0,
    latex_label='$\\Lambda_2$'
)

# Now get the transformed m1, m2 samples

# Get the NSBH posterior samples on Mc, q, dL
chirp_mass = np.array(bns_posterior["chirp_mass"])
mass_ratio = np.array(bns_posterior["mass_ratio"])
dL = np.array(bns_posterior["luminosity_distance"])

lambda_2_posterior = np.array(bns_posterior["lambda_2"])

m1, m2 = nf_prior._convert_to_source_masses(chirp_mass, mass_ratio, dL)

# Get Lambda
batch_size = len(chirp_mass)
with torch.inference_mode():
    cond = torch.tensor(m2.reshape(-1, 1), dtype=torch.float32)
    lambda_2_prior = nf_prior.nf.sample(batch_size, conditional=cond).cpu().numpy()
    lambda_2_prior = np.clip(lambda_2_prior, 0.0, 10_000)
    lambda_2_prior = lambda_2_prior.flatten()
    
    lambda_2_prior = np.exp(lambda_2_prior)
    
# Corner the m1, m2 samples
samples = np.array([m2, lambda_2_posterior]).T
labels = [r"$m_2$ [M$_\odot$]",
          r"$\Lambda_2$",
          ]

hist_kwargs = {"density": True,
               "color": NSBH_COLOR,
               }
default_corner_kwargs["color"] = NSBH_COLOR
default_corner_kwargs["hist_kwargs"] = hist_kwargs
fig = corner.corner(samples, labels=labels, **default_corner_kwargs)

samples = np.array([m2, lambda_2_prior]).T
hist_kwargs = {"density": True,
               "color": PRIOR_COLOR,
               }
default_corner_kwargs["color"] = PRIOR_COLOR
default_corner_kwargs["hist_kwargs"] = hist_kwargs
corner.corner(samples, labels=labels, fig=fig, **default_corner_kwargs)

# Add legend
fs = 26
plt.text(0.85, 0.85, 'Prior', fontsize=fs, color=PRIOR_COLOR, ha='center', va='center', transform=plt.gcf().transFigure)
plt.text(0.85, 0.75, 'NSBH', fontsize=fs, color=NSBH_COLOR, ha='center', va='center', transform=plt.gcf().transFigure)
plt.savefig(f'./figures/prior_vs_posterior_GW230529.pdf', dpi=300, bbox_inches='tight')
plt.close()