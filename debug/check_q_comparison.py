"""
Tim asked me why the mass ratio tapers so quickly for GW170817. I am checking whether that is already there for just a Gaussian prior on masses without the EOS stuff there. Doing a quick plot here to verify that. 
"""

import numpy as np
import matplotlib.pyplot as plt

import h5py

params = {"axes.grid": False,
        "text.usetex" : False,
        "font.family" : "serif",
        "ytick.color" : "black",
        "xtick.color" : "black",
        "axes.labelcolor" : "black",
        "axes.edgecolor" : "black",
        # "font.serif" : ["Computer Modern Serif"],
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
                        color="blue",
                        # quantiles=[],
                        # levels=[0.9],
                        plot_density=True, 
                        plot_datapoints=False, 
                        fill_contours=True,
                        max_n_ticks=3, 
                        min_n_ticks=2,
                        truth_color="red",
                        save=False)

paths_dict = {"Neural priors": "/work/wouters/neural_priors_paper_runs/GW170817/bns/gaussian/radio/outdir/final_result/pe_data0_1187008882-43_analysis_H1L1V1_result.hdf5",
              "Gaussian": "/work/wouters/neural_priors_paper_runs/test/gaussian_agnostic/outdir/final_result/pe_data0_1187008882-43_analysis_H1L1V1_result.hdf5"
              }

posteriors_dict = {key: {} for key in paths_dict.keys()}
runtime_dict = {}
keys_to_fetch = ["mass_ratio"]

for key, path in paths_dict.items():
    print(f"Loading {key} from {path}")
    with h5py.File(path, "r") as f:
        posterior = f["posterior"]
        
        for param_key in keys_to_fetch:
            posteriors_dict[key][param_key] = posterior[param_key][:]
            
        # Also get the runtime:
        sampling_time = f["sampling_time"][()]
        
        # Convert seconds to hours/mins/seconds
        hrs = sampling_time // 3600
        mins = (sampling_time % 3600) // 60
        secs = sampling_time % 60
        
        print(f"    Sampling time for {key}: {int(hrs)} hrs, {int(mins)} mins, {int(secs)} secs")
        
        runtime_dict[key] = sampling_time
            
# Plot a histogram comparing only mass_ratio for now:
plt.figure(figsize=(6,5))
for key in paths_dict.keys():
    plt.hist(posteriors_dict[key]["mass_ratio"], bins=50, density=True, histtype="step", label=key)
plt.xlabel("Mass ratio q")
plt.ylabel("Probability density")
plt.legend()
plt.title("Comparison of mass ratio posteriors for GW170817")
plt.tight_layout()
plt.savefig("./figures/debug_q.pdf", bbox_inches="tight")
plt.close()

# Compare
efficiency_over_neural_priors = runtime_dict["Neural priors"] / runtime_dict["Gaussian"]
print(f"Sampling with Gaussian priors is {efficiency_over_neural_priors:.2f} times faster than with Neural priors.")