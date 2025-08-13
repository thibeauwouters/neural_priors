import numpy as np
import matplotlib.pyplot as plt

import os
import json
import arviz
import numpy as np
import matplotlib.pyplot as plt
import corner
import argparse

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

def main():
    
    exit()
    
if __name__ == "__main__":
    main()