import os
import numpy as np
import matplotlib.pyplot as plt
import corner
from bilby.gw.conversion import lambda_1_lambda_2_to_lambda_tilde, lambda_1_lambda_2_to_delta_lambda_tilde

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

training_data_path = "../NFPrior/models/conditional_bns/training_data.npz"
data = np.load(training_data_path)
m1, m2, lambda_1, lambda_2 = data["m1"], data["m2"], data["lambda_1"], data["lambda_2"]

if any(m2 > m1):
    raise ValueError("m2 should not be greater than m1 in the training data.")

if any(lambda_1 > lambda_2):
    raise ValueError("lambda_1 should not be greater than lambda_2 in the training data.")

lambda_tilde = lambda_1_lambda_2_to_lambda_tilde(lambda_1, lambda_2, m1, m2)
delta_lambda_tilde = lambda_1_lambda_2_to_delta_lambda_tilde(lambda_1, lambda_2, m1, m2)

print("np.min(delta_lambda_tilde), np.max(delta_lambda_tilde)")
print(np.min(delta_lambda_tilde), np.max(delta_lambda_tilde))

# Cornerplot
samples = np.array([m1, m2,lambda_tilde, delta_lambda_tilde]).T
corner.corner(samples, 
               labels=[r"$m_1$ [M$_\odot$]", r"$m_2$ [M$_\odot$]", r"$\tilde{\Lambda}$", r"$\delta \tilde{\Lambda}$"],
               **default_corner_kwargs)

plt.savefig("./figures/lambda_tilde_bns.pdf", bbox_inches="tight")
plt.close()