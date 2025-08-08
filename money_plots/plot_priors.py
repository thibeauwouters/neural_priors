import os

import numpy as np
import matplotlib.pyplot as plt
import corner

from utils import *

from bilby.gw.conversion import component_masses_to_chirp_mass
from bilby.gw.conversion import lambda_1_lambda_2_to_lambda_tilde, lambda_1_lambda_2_to_delta_lambda_tilde

BLUE_COLORS = [JAX_LIGHT_BLUE, JAX_DARK_BLUE_TINT1, JAX_DARK_BLUE_TINT2]
GREEN_COLORS = [JAX_LIGHT_GREEN, JAX_DARK_GREEN_TINT1, JAX_DARK_GREEN_TINT2]
PURPLE_COLORS = [JAX_LIGHT_PURPLE, JAX_DARK_PURPLE_TINT1, JAX_DARK_PURPLE_TINT2]

COLORS = [JAX_LIGHT_BLUE, JAX_LIGHT_GREEN, JAX_LIGHT_PURPLE]

COLORS_LIST_DICT = {"uniform": COLORS,
                    "gaussian": COLORS,
                    "double_gaussian": COLORS}

LABELS_DICT = {"BNS": [r"$m_1$ [M$_\odot$]", r"$m_2$ [M$_\odot$]", r"$\Lambda_1$", r"$\Lambda_2$"],
               "NSBH": [r"$m_1$ [M$_\odot$]", r"$m_2$ [M$_\odot$]", r"$\Lambda_2$"]
               }

TEX_TRANSLATION_DICT = {"m1": r"$m_1$ [M$_\odot$]",
                        "m2": r"$m_2$ [M$_\odot$]",
                        "chirp_mass_source": r"$\mathcal{M}_c^{\rm{src}}$ [M$_\odot$]",
                        "mass_ratio": r"$q$",
                        "lambda_1": r"$\Lambda_1$",
                        "lambda_2": r"$\Lambda_2$",
                        "lambda_tilde": r"$\tilde{\Lambda}$",
                        "delta_lambda_tilde": r"$\delta \tilde{\Lambda}$"
                        }

fs = 24
params = {"axes.grid": False,
        "text.usetex" : True,
        "font.family" : "serif",
        "ytick.color" : "black",
        "xtick.color" : "black",
        "axes.labelcolor" : "black",
        "axes.edgecolor" : "black",
        "font.serif" : ["Computer Modern Serif"],
        "xtick.labelsize": fs,
        "ytick.labelsize": fs,
        "axes.labelsize": fs,
        "legend.fontsize": fs,
        "legend.title_fontsize": fs,
        "figure.titlesize": fs}

plt.rcParams.update(params)

# Improved corner kwargs
default_corner_kwargs = dict(bins=40, 
                        smooth=1., 
                        show_titles=False,
                        label_kwargs=dict(fontsize=32),
                        title_kwargs=dict(fontsize=46), 
                        # quantiles=[],
                        # levels=[0.393, 0.675, 0.95],
                        plot_density=False, 
                        plot_datapoints=False, 
                        fill_contours=False,
                        max_n_ticks=3, 
                        min_n_ticks=2,
                        save=False)

POPULATION_NAMES = ["uniform", "gaussian", "double_gaussian"]
SOURCE_TYPES = ["BNS", "NSBH"]
EOS_SAMPLES_NAMES = ["radio", "radio_chiEFT", "radio_chiEFT_NICER"]
EOS_SAMPLES_NAMES_DICT = {"radio": r"Radio",
                         "radio_chiEFT": r"+$\chi_{\rm{EFT}}$",
                         "radio_chiEFT_NICER": r"+$\chi_{\rm{EFT}}$+NICER"}

def make_conversions(data: np.ndarray) -> None:
    """
    Add the lambda_tilde parameter to the data if it is not already present.

    Args:
        data (np.ndarray): Array of shape (n_samples, n_parameters) containing the samples.
    """
    m1, m2 = data["m1"], data["m2"]
    lambda_1, lambda_2 = data["lambda_1"], data["lambda_2"]
    
    lambda_tilde = lambda_1_lambda_2_to_lambda_tilde(lambda_1, lambda_2, m1, m2)
    delta_lambda_tilde = lambda_1_lambda_2_to_delta_lambda_tilde(lambda_1, lambda_2, m1, m2)
    
    data["lambda_tilde"] = lambda_tilde
    data["delta_lambda_tilde"] = delta_lambda_tilde
    
    if "chirp_mass_source" not in list(data.keys()):
        chirp_mass_source = component_masses_to_chirp_mass(m1, m2)
        data["chirp_mass_source"] = chirp_mass_source
        
    if "mass_ratio" not in list(data.keys()):
        mass_ratio = m2 / m1
        data["mass_ratio"] = mass_ratio
    
    return data

def get_ranges(data: np.ndarray,
               quantile: float=0.01) -> list:
    """
    Get the ranges of the data for each parameter using the quantiles to make it flexible but not dominated by outliers.

    Args:
        data (np.ndarray): Array of shape (n_samples, n_parameters) containing the samples.

    Returns:
        tuple: A tuple containing the minimum and maximum values for each parameter.
    """
    ranges = []
    for i in range(data.shape[1]):
        min_val = np.quantile(data[:, i], quantile)
        max_val = np.quantile(data[:, i], 1-quantile)
        ranges.append((min_val, max_val))
    return ranges

def single_plot(pop: str,
                source_type: str,
                convert_masses: bool=True,
                convert_to_lambda_tilde: bool=True,
                add_legend: bool=False,
                add_title: bool=False) -> None:
    
    colors = COLORS_LIST_DICT[pop]
    
    # Determine which parameters to fetch for the plotting
    if convert_masses:
        mass_keys = ["chirp_mass_source", "mass_ratio"]
    else:
        mass_keys = ["m1", "m2"]
    if convert_to_lambda_tilde:
        lambda_keys = ["lambda_tilde", "delta_lambda_tilde"]
    else:
        if source_type == "BNS":
            lambda_keys = ["lambda_1", "lambda_2"]
        elif source_type == "NSBH":
            lambda_keys = ["lambda_2"]
    keys = mass_keys + lambda_keys
    labels = [TEX_TRANSLATION_DICT[key] for key in keys]
    
    for i, eos_samples_name in enumerate(EOS_SAMPLES_NAMES):
        # Load the corner plot data
        path = get_training_data_path(pop, source_type, eos_samples_name)
        data = dict(np.load(path))
        data = make_conversions(data)
        
        # Fetch the relevant data for the corner plot
        corner_data = [data[k] for k in keys]
        corner_data = np.array(corner_data).T
            
        # Determine ranges dynamically
        ranges = get_ranges(corner_data)
        
        # Create the corner plots
        if i == 0:
            fig = corner.corner(corner_data, 
                        labels=labels,
                        range=ranges,
                        color=colors[i],
                        **default_corner_kwargs)
        else:
            corner.corner(corner_data, 
                          labels=labels,
                          range=ranges,
                          color=colors[i],
                          fig=fig,
                          **default_corner_kwargs)
            
    # Add title if requested
    if add_title:
        title_map = {"uniform": "Uniform", "gaussian": "Gaussian", "double_gaussian": "Double Gaussian"}
        plt.suptitle(title_map[pop], fontsize=32)
    
    # Put text there
    if add_legend:
        x = 0.55
        y = 0.85
        dy = 0.05
        for i, eos_samples_name in enumerate(EOS_SAMPLES_NAMES):
            plt.text(x, y - i * dy, 
                    EOS_SAMPLES_NAMES_DICT[eos_samples_name], 
                    color=colors[i],
                    fontsize=32,
                    transform=plt.gcf().transFigure)
    
    save_name = f"./figures/priors/{source_type}_{pop}_priors.pdf"
    plt.savefig(save_name, bbox_inches="tight")
    if os.path.exists("../../paper/Figures"):
        print(f"Also saving to the Overleaf paper repo")
        save_name = save_name.replace("./figures/priors/", "../../paper/Figures/")
        plt.savefig(save_name, bbox_inches="tight")
    plt.close()

def main():
    for pop in POPULATION_NAMES:
        for source_type in SOURCE_TYPES:
            print(f"Plotting priors for {pop} {source_type}...")
            single_plot(pop,
                        source_type,
                        add_legend=pop=="uniform", # only add the legend on the "uniform" population plot
                        add_title=False)
            
    
if __name__ == "__main__":
    main()