import h5py
import os
import numpy as np
import matplotlib.pyplot as plt
import corner

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
                        max_n_ticks=4, 
                        min_n_ticks=3,
                        truth_color = "red",
                        save=False)

filename = "/work/wouters/neural_priors_paper_runs/test/gaussian_agnostic/outdir/final_result/pe_data0_1187008882-43_analysis_H1L1V1_result.hdf5"

# params_to_plot = ["chirp_mass", "mass_ratio", "a_1", "a_2", "tilt_1", "tilt_2", "luminosity_distance", "theta_jn", "lambda_1", "lambda_2"]
params_to_plot = ["chirp_mass", "mass_ratio", "chi_eff", "chi_p", "lambda_tilde", "delta_lambda_tilde"]

with h5py.File(filename, "r") as f:
    posterior = f["posterior"]
    samples = np.array([posterior[k] for k in params_to_plot]).T
    
print(np.shape(samples))

corner.corner(samples, labels=params_to_plot, **default_corner_kwargs)
plt.savefig("./figures/ cornerplot.pdf", bbox_inches="tight")
plt.close()