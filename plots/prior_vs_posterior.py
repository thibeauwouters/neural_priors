"""
Prior vs Posterior comparison plots for gravitational wave parameter estimation.

This script creates corner plots comparing prior distributions (from normalizing flows)
with posterior distributions from specific GW events. Designed to be modular for
easy extension to different events, populations, and EOS samples.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import corner
from typing import List, Dict, Optional, Tuple

# Import utilities from existing modules
import utils
from plot_priors import get_training_data_path, make_conversions, get_ranges
from money_plots import (
    load_posterior_data, construct_result_path, setup_matplotlib_style,
    DEFAULT_CORNER_KWARGS, EOS_COLORS, EOS_SAMPLES_NAMES_DICT,
    convert_lambdas_with_verbose, filter_constant_parameters,
    calculate_corner_plot_ranges, handle_nsbh_lambda_plotting, 
    PARAMETER_LATEX_LABELS
)

# Setup matplotlib style
setup_matplotlib_style()

# Colors for prior (lighter) vs posterior (darker)
PRIOR_ALPHA = 0.3
POSTERIOR_ALPHA = 0.8

def map_prior_to_posterior_params(params: List[str]) -> List[str]:
    """
    Map posterior parameter names to their corresponding prior parameter names.
    
    The prior training data uses slightly different parameter names than 
    the processed posterior data.
    
    Args:
        params (List[str]): List of posterior parameter names
        
    Returns:
        List[str]: List of corresponding prior parameter names
    """
    param_mapping = {
        "chirp_mass": "chirp_mass_source",  # Prior has source-frame, posterior has detector-frame
        # All other parameters should match
    }
    
    return [param_mapping.get(param, param) for param in params]

def load_prior_samples(population: str,
                      source_type: str,
                      eos_samples_name: str,
                      params_to_plot: List[str],
                      n_samples: int = 10000) -> np.ndarray:
    """
    Load prior samples from training data.
    
    Args:
        population (str): Population name (e.g., "gaussian", "uniform", "double_gaussian")
        source_type (str): Source type ("bns", "nsbh")
        eos_samples_name (str): EOS samples name (e.g., "radio", "radio_chiEFT", "radio_NICER")
        params_to_plot (List[str]): Parameters to extract
        n_samples (int): Number of samples to use
        
    Returns:
        np.ndarray: Prior samples array with shape (n_samples, n_params)
    """
    path = get_training_data_path(population, source_type, eos_samples_name)
    data = dict(np.load(path))
    data = make_conversions(data)
    
    # Map posterior parameter names to prior parameter names
    prior_param_names = map_prior_to_posterior_params(params_to_plot)
    
    # Extract parameter samples using the mapped names
    prior_samples = []
    for i, prior_param in enumerate(prior_param_names):
        if prior_param in data:
            samples = np.array(data[prior_param])
            # Randomly subsample to n_samples
            if len(samples) > n_samples:
                idx = np.random.choice(len(samples), n_samples, replace=False)
                samples = samples[idx]
            prior_samples.append(samples)
        else:
            original_param = params_to_plot[i]
            raise KeyError(f"Parameter '{original_param}' (mapped to '{prior_param}') not found in prior data. Available keys: {list(data.keys())}")
    
    return np.column_stack(prior_samples)

def plot_prior_vs_posterior(gw_event: str,
                           population: str,
                           source_type: str,
                           eos_samples_name: str,
                           params_to_plot: List[str] = None,
                           base_path: str = "../final_results/",
                           output_dir: str = "./figures/prior_vs_posterior/",
                           ranges: List[tuple] = None,
                           prior_n_samples: int = 10000,
                           add_legend: bool = True,
                           verbose: bool = False) -> str:
    """
    Create corner plot comparing prior and posterior distributions.
    
    Args:
        gw_event (str): GW event name (e.g., 'GW170817', 'GW190425', 'GW230529')
        population (str): Population type ('uniform', 'gaussian', 'double_gaussian')
        source_type (str): Source type ('bns', 'nsbh')
        eos_samples_name (str): EOS samples name (e.g., 'radio', 'radio_chiEFT', 'radio_NICER')
        params_to_plot (List[str]): Parameters to plot (default: standard set)
        base_path (str): Base path to result files
        output_dir (str): Output directory for figures
        ranges (List[tuple]): Custom ranges for each parameter
        prior_n_samples (int): Number of prior samples to use
        add_legend (bool): Whether to add legend to plot
        verbose (bool): Print verbose information
        
    Returns:
        str: Path to saved figure
    """
    
    # Set default parameters if not provided
    if params_to_plot is None:
        if source_type.lower() == "bns":
            params_to_plot = ["chirp_mass", "mass_ratio", "lambda_tilde", "delta_lambda_tilde"]
        elif source_type.lower() == "nsbh":
            params_to_plot = ["chirp_mass", "mass_ratio", "lambda_2"]
    
    print(f"Creating prior vs posterior plot for {gw_event}")
    print(f"Population: {population}, Source: {source_type}, EOS: {eos_samples_name}")
    print(f"Parameters: {params_to_plot}")
    
    # Load prior samples
    print("Loading prior samples...")
    prior_samples = load_prior_samples(population, source_type, eos_samples_name, 
                                     params_to_plot, n_samples=prior_n_samples)
    if verbose:
        print(f"Loaded prior samples with shape: {prior_samples.shape}")
    
    # Load posterior samples
    print("Loading posterior samples...")
    result_path = construct_result_path(base_path, gw_event, population, source_type, eos_samples_name)
    
    if not os.path.exists(result_path):
        raise FileNotFoundError(f"Result file not found: {result_path}")
    
    posterior = load_posterior_data(result_path, fast_mode=True)
    if posterior is None:
        raise ValueError(f"Could not load posterior data from: {result_path}")
    
    # Convert lambdas to tilde parameters if needed
    if any('lambda_tilde' in p or 'delta_lambda_tilde' in p for p in params_to_plot):
        posterior_dict = {eos_samples_name: posterior}
        posterior_dict, params_to_plot = convert_lambdas_with_verbose(
            posterior_dict, params_to_plot, verbose=verbose
        )
        posterior = posterior_dict[eos_samples_name]
    
    # Extract posterior samples
    posterior_samples_list = []
    for param in params_to_plot:
        if param in posterior:
            posterior_samples_list.append(np.array(posterior[param]))
        else:
            raise KeyError(f"Parameter '{param}' not found in posterior data")
    
    posterior_samples = np.column_stack(posterior_samples_list)
    if verbose:
        print(f"Extracted posterior samples with shape: {posterior_samples.shape}")
    
    # Apply source-specific processing
    samples_dict = {"prior": prior_samples, "posterior": posterior_samples}
    
    # Compute percentage of samples with negative delta lambda tilde for BNS
    if source_type.lower() == 'bns' and 'delta_lambda_tilde' in params_to_plot:
        delta_lambda_idx = params_to_plot.index('delta_lambda_tilde')
        
        print(f"\n=== Negative Delta Lambda Tilde Statistics ===")
        for dataset_name, samples in samples_dict.items():
            delta_lambda_samples = samples[:, delta_lambda_idx]
            negative_count = np.sum(delta_lambda_samples < 0)
            total_count = len(delta_lambda_samples)
            percentage = (negative_count / total_count) * 100
            print(f"{dataset_name}: {negative_count}/{total_count} ({percentage:.1f}%) negative delta_lambda_tilde")
    elif source_type.lower() == 'nsbh':
        samples_dict = handle_nsbh_lambda_plotting(samples_dict, params_to_plot, params_to_plot)
    
    prior_samples = samples_dict["prior"]
    posterior_samples = samples_dict["posterior"]
    
    # Filter constant parameters
    labels, latex_labels, filtered_samples_dict = filter_constant_parameters(
        params_to_plot,
        [PARAMETER_LATEX_LABELS.get(p, p) for p in params_to_plot],
        {"prior": prior_samples, "posterior": posterior_samples}
    )
    
    prior_samples = filtered_samples_dict["prior"]
    posterior_samples = filtered_samples_dict["posterior"]
    
    # Handle ranges
    if ranges is None:
        # Calculate ranges from prior data using quantiles
        ranges = []
        for i in range(prior_samples.shape[1]):
            param_samples = prior_samples[:, i]
            min_val = np.quantile(param_samples, 0.01)
            max_val = np.quantile(param_samples, 0.99)
            ranges.append((min_val, max_val))
        
        if verbose:
            print(f"Calculated ranges from prior quantiles (0.01, 0.99): {ranges}")
    else:
        if len(ranges) != len(params_to_plot):
            raise ValueError(f"ranges must have same length as params_to_plot ({len(params_to_plot)}), got {len(ranges)}")
        # Filter ranges to match filtered parameters
        if len(labels) != len(params_to_plot):
            filtered_ranges = []
            for i, param in enumerate(params_to_plot):
                if param in labels:
                    filtered_ranges.append(ranges[i])
            ranges = filtered_ranges
        if verbose:
            print(f"Using custom ranges: {ranges}")
    
    # Set colors
    prior_color = 'lightblue'
    posterior_color = 'blue'
    
    # Create corner plot
    # Plot prior first (lighter, in background)
    prior_kwargs = DEFAULT_CORNER_KWARGS.copy()
    prior_kwargs.update({
        'color': prior_color,
        'labels': latex_labels,
        'range': ranges,
    })
    
    fig = corner.corner(prior_samples, **prior_kwargs)
    
    # Overlay posterior (darker, in foreground)
    posterior_kwargs = DEFAULT_CORNER_KWARGS.copy()
    posterior_kwargs.update({
        'color': posterior_color,
        'labels': latex_labels,
        'range': ranges,
    })
    
    corner.corner(posterior_samples, fig=fig, **posterior_kwargs)
    
    # Add legend
    if add_legend:
        legend_x = 0.65
        legend_y = 0.95
        legend_dy = 0.06
        legend_fontsize = 32
        
        # EOS label
        eos_label = EOS_SAMPLES_NAMES_DICT.get(eos_samples_name, eos_samples_name)
        plt.text(legend_x, legend_y, eos_label, 
                fontsize=legend_fontsize, color='black', ha='left', va='top',
                transform=fig.transFigure, weight='bold')
        
        # Prior/Posterior labels
        plt.text(legend_x, legend_y - legend_dy, f"Prior ({population})", 
                fontsize=legend_fontsize-4, color=prior_color, ha='left', va='top',
                transform=fig.transFigure, alpha=0.7, style='italic')
        
        plt.text(legend_x, legend_y - 2*legend_dy, f"Posterior ({gw_event})", 
                fontsize=legend_fontsize-4, color=posterior_color, ha='left', va='top',
                transform=fig.transFigure, weight='bold')
    
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Save figure
    output_filename = f"{gw_event}_{population}_{source_type}_{eos_samples_name}_prior_vs_posterior.pdf"
    output_path = os.path.join(output_dir, output_filename)
    
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Saved figure to: {output_path}")
    plt.close()
    
    return output_path

def plot_gw170817_gaussian_radio_chieft_bns() -> str:
    """
    Convenience function for GW170817 BNS Gaussian+radio_chiEFT prior vs posterior.
    """
    return plot_prior_vs_posterior(
        gw_event="GW170817",
        population="gaussian",
        source_type="bns",
        eos_samples_name="radio_chiEFT",
        params_to_plot=["chirp_mass", "mass_ratio", "lambda_tilde", "delta_lambda_tilde"],
        ranges=None,
        verbose=True
    )

def plot_gw190425_uniform_radio_bns() -> str:
    """
    Convenience function for GW190425 BNS Uniform+radio prior vs posterior.
    """
    return plot_prior_vs_posterior(
        gw_event="GW190425",
        population="uniform",
        source_type="bns", 
        eos_samples_name="radio",
        params_to_plot=["chirp_mass", "mass_ratio", "lambda_tilde", "delta_lambda_tilde"],
        verbose=True
    )

def plot_gw230529_gaussian_radio_nicer_nsbh() -> str:
    """
    Convenience function for GW230529 NSBH Gaussian+radio_NICER prior vs posterior.
    """
    return plot_prior_vs_posterior(
        gw_event="GW230529",
        population="gaussian",
        source_type="nsbh",
        eos_samples_name="radio_NICER",
        params_to_plot=["chirp_mass", "mass_ratio", "lambda_2"],
        ranges=None,
        verbose=True
    )

def main():
    """
    Run example prior vs posterior comparisons.
    """
    print("Creating GW170817 Gaussian+radio_chiEFT BNS prior vs posterior plot...")
    plot_gw170817_gaussian_radio_chieft_bns()
    
    print("\nCreating GW190425 Uniform+radio BNS prior vs posterior plot...")
    plot_gw190425_uniform_radio_bns() 
    
    print("\nCreating GW230529 Gaussian+radio_NICER NSBH prior vs posterior plot...")
    plot_gw230529_gaussian_radio_nicer_nsbh()

if __name__ == "__main__":
    main()