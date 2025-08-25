"""
Publication-ready corner plots for gravitational wave parameter estimation results.

This script provides hardcoded configurations for creating publication-quality
corner plots for specific GW events and parameter combinations. Unlike the batch
approach in cornerplots.py, this focuses on creating carefully configured plots
for paper figures.

Author: Claude Code
Date: 2025-08-20
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import corner
from typing import List

from utils import (
    construct_result_path, load_posterior_data, setup_matplotlib_style,
    PARAMETER_LATEX_LABELS, DEFAULT_CORNER_KWARGS, EOS_COLORS, EOS_SAMPLES_NAMES_DICT,
    convert_lambdas_with_verbose, filter_constant_parameters,
    calculate_corner_plot_ranges, handle_nsbh_lambda_plotting, prevent_bns_leakage
)

# Setup matplotlib style
setup_matplotlib_style()


def ensure_output_directory(output_path: str) -> None:
    """
    Ensure the output directory exists.
    
    Args:
        output_path (str): Path to the output file
    """
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created output directory: {output_dir}")


def plot_corner_fixed_population_varying_eos(gw_event: str,
                                            population: str = "gaussian",
                                            source_type: str = "bns",
                                            base_path: str = "../final_results/",
                                            output_dir: str = "./figures/money_plots/",
                                            eos_samples: List[str] = None,
                                            params_to_plot: List[str] = None,
                                            ranges: List[tuple] = None,
                                            normalization_keys: List[str] = None,
                                            legend_x: float = 0.85,
                                            legend_y: float = 0.95,
                                            legend_dy: float = 0.08,
                                            legend_fontsize: int = 14,
                                            verbose: bool = False) -> str:
    """
    Create corner plot for a GW event with fixed population type and varying EOS samples.
    
    This function compares different EOS constraints while keeping the population
    prior fixed. It's designed for studying the impact of different nuclear physics
    constraints on parameter estimation.
    
    Args:
        gw_event (str): GW event name (e.g., 'GW170817', 'GW190425', 'GW230529')
        population (str): Population type ('uniform', 'gaussian', 'double_gaussian')
        source_type (str): Source type ('bns', 'nsbh', 'default')
        base_path (str): Base path to result files
        output_dir (str): Output directory for figures
        eos_samples (List[str]): EOS samples to include (default: radio, radio_chiEFT, radio_NICER)
        params_to_plot (List[str]): Parameters to plot (default: standard set)
        ranges (List[tuple]): Custom ranges for each parameter as list of (min, max) tuples.
            Must have same length as params_to_plot. If None, ranges are automatically
            calculated from the data. Example: [(1.0, 1.6), (0.5, 1.0), (0, 800), (-500, 500)]
        normalization_keys (List[str]): Dataset keys for histogram normalization dummy dataset.
            Must have same length as params_to_plot. Each key specifies which dataset
            to use for that parameter when creating the normalization dummy dataset.
            Example: ["radio", "radio_chiEFT", "radio", "radio"] uses radio for params 1&3,
            radio_chiEFT for param 2. If None, no normalization dummy dataset is created.
        legend_x (float): X position of legend (0-1 scale)
        legend_y (float): Y position of legend (0-1 scale)
        legend_dy (float): Vertical spacing between legend entries
        legend_fontsize (int): Font size for legend text
        verbose (bool): Print verbose information
        
    Returns:
        str: Path to saved figure
    """
    
    # Set default parameters
    if eos_samples is None:
        eos_samples = ["radio", "radio_chiEFT", "radio_NICER"]
    
    if params_to_plot is None:
        params_to_plot = ["chirp_mass", "mass_ratio", 
                         "lambda_tilde", "delta_lambda_tilde"]
    
    print(f"Creating {gw_event} corner plot for population: {population}, source: {source_type}")
    print(f"EOS samples: {eos_samples}")
    print(f"Parameters: {params_to_plot}")
    
    # Load posterior data for each EOS sample
    posteriors_dict = {}
    
    for eos_name in eos_samples:
        result_path = construct_result_path(base_path, gw_event, population, 
                                          source_type, eos_name)
        print(f"Loading {eos_name} from: {result_path}")
        
        if not os.path.exists(result_path):
            print(f"Warning: Result file not found: {result_path}")
            continue
            
        posterior = load_posterior_data(result_path, fast_mode=True)
        if posterior is not None:
            posteriors_dict[eos_name] = posterior
            if verbose:
                print(f"Loaded {eos_name} with {len(posterior['chirp_mass'])} samples")
        else:
            print(f"Warning: Could not load data for {eos_name}")
    
    if not posteriors_dict:
        raise ValueError("No posterior data could be loaded!")
    
    # Convert lambdas to tilde parameters if needed
    if any('lambda_tilde' in p or 'delta_lambda_tilde' in p for p in params_to_plot):
        posteriors_dict, params_to_plot = convert_lambdas_with_verbose(
            posteriors_dict, params_to_plot, verbose=verbose
        )
    
    # Create samples dictionary for corner plotting
    samples_dict = {}
    for eos_name, posterior in posteriors_dict.items():
        # Extract parameter samples
        samples_list = []
        for param in params_to_plot:
            if param in posterior:
                samples_list.append(np.array(posterior[param]))
            else:
                print(f"Warning: Parameter '{param}' not found in {eos_name} posterior")
        
        if samples_list:
            samples_dict[eos_name] = np.column_stack(samples_list)
    
    if not samples_dict:
        raise ValueError("No valid samples could be extracted!")
    
    # Apply BNS leakage prevention if needed
    if source_type.lower() == 'bns':
        samples_dict = prevent_bns_leakage(samples_dict, params_to_plot, params_to_plot)
    
    # Handle NSBH lambda plotting if needed
    if source_type.lower() == 'nsbh':
        samples_dict = handle_nsbh_lambda_plotting(samples_dict, params_to_plot, params_to_plot)
    
    # Validate and create normalization dummy dataset if requested
    dummy_dataset = None
    if normalization_keys is not None:
        # Validate normalization_keys
        n_params = len(params_to_plot)
        if len(normalization_keys) != n_params:
            raise ValueError(f"normalization_keys must have length {n_params} (number of parameters), got {len(normalization_keys)}")
        
        # Check that all keys exist in loaded datasets
        loaded_keys = set(samples_dict.keys())
        for key in normalization_keys:
            if key not in loaded_keys:
                raise ValueError(f"normalization_keys contains '{key}' which was not loaded. Available keys: {loaded_keys}")
        
        # Find minimum number of samples across all datasets
        min_samples = min(len(dataset) for dataset in samples_dict.values())
        if verbose:
            sample_counts = {key: len(dataset) for key, dataset in samples_dict.items()}
            print(f"Sample counts by dataset: {sample_counts}")
            print(f"Using minimum sample count for normalization: {min_samples}")
        
        # Create dummy dataset by selecting specified dataset for each parameter
        # All datasets are downsampled to min_samples to ensure consistent sizes
        dummy_columns = []
        for param_idx, dataset_key in enumerate(normalization_keys):
            dataset = samples_dict[dataset_key]
            # Downsample to min_samples by taking first min_samples rows
            downsampled_dataset = dataset[:min_samples]
            dummy_columns.append(downsampled_dataset[:, param_idx])
        
        dummy_dataset = np.column_stack(dummy_columns)
        if verbose:
            print(f"Created normalization dummy dataset using keys: {normalization_keys}")
            print(f"Dummy dataset shape: {dummy_dataset.shape}")
    
    # Filter constant parameters
    labels, latex_labels, samples_dict = filter_constant_parameters(
        params_to_plot, 
        [PARAMETER_LATEX_LABELS.get(p, p) for p in params_to_plot],
        samples_dict
    )
    
    # Handle ranges: use custom if provided, otherwise calculate automatically
    if ranges is not None:
        # Validate custom ranges
        if len(ranges) != len(params_to_plot):
            raise ValueError(f"ranges must have same length as params_to_plot ({len(params_to_plot)}), got {len(ranges)}")
        
        # Filter ranges to match filtered parameters (in case some were removed as constant)
        if len(labels) != len(params_to_plot):
            # Some parameters were filtered out, need to adjust ranges accordingly
            filtered_ranges = []
            for i, param in enumerate(params_to_plot):
                if param in labels:
                    param_idx = labels.index(param)
                    filtered_ranges.append(ranges[i])
            ranges = filtered_ranges
        
        if verbose:
            print(f"Using custom ranges: {ranges}")
    else:
        # Calculate ranges automatically
        ranges = calculate_corner_plot_ranges(labels, samples_dict)
        if verbose:
            print(f"Calculated automatic ranges: {ranges}")
    
    # Create corner plot
    fig = None
    group_colors = {}
    
    for i, (eos_name, samples) in enumerate(samples_dict.items()):
        color = EOS_COLORS.get(eos_name, f"C{i}")  # Fallback to default colors if needed
        group_colors[eos_name] = color
        
        # Corner plot kwargs
        kwargs = DEFAULT_CORNER_KWARGS.copy()
        kwargs.update({
            'color': color,
            'labels': latex_labels,
            'range': ranges,
            'hist_kwargs': {'color': color, 'density': True},
        })
        
        if fig is None:
            # First plot - create the figure
            fig = corner.corner(samples, **kwargs)
        else:
            # Overlay subsequent plots
            corner.corner(samples, fig=fig, **kwargs)
    
    # Plot invisible dummy dataset LAST for histogram normalization
    # Only plot histograms (1D) to avoid white space in 2D contours
    if dummy_dataset is not None:
        if verbose:
            print(f"Plotting invisible dummy dataset for histogram normalization")
        
        # Create corner kwargs that only plots 1D histograms
        invisible_kwargs = DEFAULT_CORNER_KWARGS.copy()
        invisible_kwargs.update({
            'labels': latex_labels,
            'range': ranges,
            'hist_kwargs': {'alpha': 0, 'density': True},
            'plot_density': False,     # Disable 2D density plots
            'plot_contours': False,    # Disable 2D contours  
            'plot_datapoints': False,  # Disable scatter points
            'no_fill_contours': True,  # Disable filled contours
            'color': 'black'
        })
        
        corner.corner(dummy_dataset, fig=fig, **invisible_kwargs)
    
    # Add legend
    # Create custom legend entries
    legend_entries = []
    legend_colors = []
    
    for eos_name in samples_dict.keys():
        legend_entries.append(EOS_SAMPLES_NAMES_DICT.get(eos_name, eos_name))
        legend_colors.append(EOS_COLORS.get(eos_name, 'black'))
    
    # Add legend with configurable positioning
    for i, (label, color) in enumerate(zip(legend_entries, legend_colors)):
        plt.text(legend_x, legend_y - i * legend_dy, label, 
                fontsize=legend_fontsize, color=color, ha='left', va='top',
                transform=fig.transFigure, weight='bold')
    
    # Save figure
    output_filename = f"{gw_event}_corner_{population}_{source_type}.pdf"
    output_path = os.path.join(output_dir, output_filename)
    ensure_output_directory(output_path)
    
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Saved figure to: {output_path}")
    plt.close()
    
    return output_path


def plot_gw170817_corner() -> str:
    """
    Convenience function for creating GW170817 corner plot.
    """
    # Set hardcoded defaults for GW170817
    gw170817_defaults = {
        'legend_x': 0.75,
        'legend_y': 0.95,
        'legend_dy': 0.08,
        'legend_fontsize': 16,
        'normalization_keys': ["radio", "radio_chiEFT", "radio_NICER", "radio"],
        'ranges': [[1.197435, 1.1977],
                   [0.88, 1.0],
                   [251, 1000],
                   [0.0, 75.0]]
    }
    
    path = plot_corner_fixed_population_varying_eos(
        gw_event="GW170817",
        population="gaussian",
        source_type="bns",  # GW170817 is primarily BNS
        **gw170817_defaults
    )
    
    # For double Gaussian the delta_lambda_tilde key should be changed for normalization
    gw170817_defaults['normalization_keys'] = ["radio", "radio_chiEFT", "radio_NICER", "radio_chiEFT"]
    path = plot_corner_fixed_population_varying_eos(
        gw_event="GW170817",
        population="double_gaussian",
        source_type="bns",  # GW170817 is primarily BNS
        **gw170817_defaults
    )
    
    return path


def plot_gw190425_corner(population: str = "gaussian",
                        source_type: str = "bns",
                        **kwargs) -> str:
    """
    Convenience function for creating GW190425 corner plot.
    
    Args:
        population (str): Population type
        source_type (str): Source type ('bns' or 'nsbh')
        **kwargs: Additional arguments passed to plot_corner_fixed_population_varying_eos
        
    Returns:
        str: Path to saved figure
    """
    return plot_corner_fixed_population_varying_eos(
        gw_event="GW190425",
        population=population,
        source_type=source_type,
        **kwargs
    )


def plot_gw230529_corner(population: str = "gaussian",
                        source_type: str = "nsbh",
                        **kwargs) -> str:
    """
    Convenience function for creating GW230529 corner plot.
    
    Args:
        population (str): Population type
        source_type (str): Source type ('bns' or 'nsbh')
        **kwargs: Additional arguments passed to plot_corner_fixed_population_varying_eos
        
    Returns:
        str: Path to saved figure
    """
    return plot_corner_fixed_population_varying_eos(
        gw_event="GW230529",
        population=population,
        source_type=source_type,
        **kwargs
    )


def main():
    """
    Main function for creating money plots.
    Demonstrates usage of the plotting functions.
    """
    
    print("=== Money Plots - Fixed Population, Varying EOS ===")
    
    # Create GW170817 corner plot with Gaussian population
    output_path = plot_gw170817_corner()
    
    print(f"\nSuccessfully created GW170817 corner plot: {output_path}")
    
    # Example: Create plot with custom ranges
    # custom_ranges = [(1.1, 1.6), (0.4, 1.0), (0, 800), (-500, 500)]  # chirp_mass, mass_ratio, lambda_tilde, delta_lambda_tilde
    # output_path_custom = plot_gw170817_corner(
    #     population="gaussian",
    #     ranges=custom_ranges,
    #     verbose=True
    # )
    
    # Example: Create plots for other events/configurations  
    # output_path_gw190425 = plot_gw190425_corner(
    #     population="gaussian",
    #     source_type="bns",
    #     verbose=True
    # )
    
    # output_path_gw230529 = plot_gw230529_corner(
    #     population="gaussian", 
    #     source_type="nsbh",
    #     verbose=True
    # )


if __name__ == "__main__":
    main()