"""
Quick and dirty script to plot distance-iota posteriors from bilby samples to check up on them -- might delete later on
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import corner 

from utils import DEFAULT_CORNER_KWARGS

path_default = "../final_results/GW190425/uniform/bns/radio/bns_result.npz"
path_bns = "../final_results/GW190425/uniform/default/radio/default_result.npz"

def main():
    
    data_default = np.load(path_default)
    samples_default = np.array([data_default['luminosity_distance'], data_default['theta_jn']]).T
    
    data_bns = np.load(path_bns)
    samples_bns = np.array([data_bns['luminosity_distance'], data_bns['theta_jn']]).T
    
    DEFAULT_CORNER_KWARGS["density"] = True
    fig = corner.corner(samples_default, labels=["$d_L$", "$\\iota$"], color='blue', label="Informed", **DEFAULT_CORNER_KWARGS)
    corner.corner(samples_bns, labels=["$d_L$", "$\\iota$"], color='gray', label="Informed", fig=fig, **DEFAULT_CORNER_KWARGS)
    
    plt.savefig("./figures/distance_iota_posterior.png", dpi=300)
    plt.close()
    
    
if __name__ == "__main__":
    main()