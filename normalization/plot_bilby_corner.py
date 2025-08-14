#!/usr/bin/env python
"""
Create corner plot from bilby result file.

Usage:
    python plot_bilby_corner.py
"""

import sys
import os
import bilby
import corner
import matplotlib.pyplot as plt
import numpy as np

# Add bilby path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../bilby'))

# Load bilby result
result_path = "../NFprior/models/uniform/bns/radio/normalization_validation/nf_normalization_validation_result.json"
result = bilby.core.result.Result.from_json(result_path)

# Parameters to plot
params = ['chirp_mass_source', 'mass_ratio', 'lambda_1', 'lambda_2']
param_labels = [r'$\mathcal{M}_{\rm src}$ [M$_\odot$]', r'$q$', r'$\Lambda_1$', r'$\Lambda_2$']

# Extract samples
samples = result.posterior[params].values

# Create corner plot
fig = corner.corner(
    samples, 
    labels=param_labels,
    quantiles=[0.16, 0.5, 0.84],
    show_titles=True,
    title_kwargs={"fontsize": 12}
)

# Add title
fig.suptitle('NF Normalization Validation Samples', fontsize=14, y=0.98)

# Save plot
output_path = "nf_samples_corner.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Corner plot saved to: {output_path}")

plt.show()