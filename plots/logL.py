"""
Log-likelihood distribution analysis for gravitational wave parameter estimation results.

This script provides functions for creating publication-quality histograms of log-likelihood
distributions across different EOS constraints and population priors.

Now includes support for:
- log_likelihood distributions
- log_prior distributions
- Network SNR (combined detector SNR)
- Individual detector SNRs (H1, L1, V1)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Dict, Tuple

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from plots.utils import (
    construct_result_path, load_posterior_data,
    EOS_COLORS, EOS_SAMPLES_NAMES_DICT
)

# Font size
fs_ticks = 16
fs_labels = 18
fs_legend = 14
fs_title = 20

# Matplotlib style parameters
params = {
    "axes.grid": False,
    "text.usetex": True,
    "font.family": "serif",
    "ytick.color": "black",
    "xtick.color": "black",
    "axes.labelcolor": "black",
    "axes.edgecolor": "black",
    "font.serif": ["Computer Modern Serif"],
    "xtick.labelsize": fs_ticks,
    "ytick.labelsize": fs_ticks,
    "axes.labelsize": fs_labels,
}

plt.rcParams.update(params)

# Default run color constants
DEFAULT_RUN_COLOR = 'dimgray'
DEFAULT_RUN_LABEL = 'Uninformed'

# Detector list
DETECTORS = ['H1', 'L1', 'V1']

# Quantity configuration: maps quantity names to their properties
QUANTITY_CONFIG = {
    'log_likelihood': {
        'key': 'log_likelihood',
        'label': r'$\log \mathcal{L}$',
        'ylabel_density': r'Probability Density',
        'ylabel_count': r'Count',
        'output_dir': 'logL'
    },
    'log_prior': {
        'key': 'log_prior',
        'label': r'$\log \pi$',
        'ylabel_density': r'Probability Density',
        'ylabel_count': r'Count',
        'output_dir': 'logPrior'
    },
    'network_snr': {
        'key': None,  # Computed from detector SNRs
        'label': r'Network SNR',
        'ylabel_density': r'Probability Density',
        'ylabel_count': r'Count',
        'output_dir': 'networkSNR'
    }
}

# Add individual detector SNR configurations
for det in DETECTORS:
    QUANTITY_CONFIG[f'{det}_snr'] = {
        'key': f'{det}_optimal_snr',
        'label': f'{det} SNR',
        'ylabel_density': r'Probability Density',
        'ylabel_count': r'Count',
        'output_dir': f'{det}_SNR'
    }


def compute_network_snr(posterior: Dict) -> np.ndarray:
    """
    Compute network SNR from individual detector matched filter SNRs.

    Network SNR = sqrt(sum of |matched_filter_snr|^2 for all detectors)

    Args:
        posterior (Dict): Posterior samples dictionary

    Returns:
        np.ndarray: Network SNR values
    """
    snr_squared_sum = np.zeros(len(posterior[list(posterior.keys())[0]]))

    for det in DETECTORS:
        key = f'{det}_matched_filter_snr'
        if key in posterior:
            # Matched filter SNR is complex, take absolute value
            snr_values = np.abs(posterior[key])
            snr_squared_sum += snr_values**2

    return np.sqrt(snr_squared_sum)


def extract_quantity(posterior: Dict, quantity: str) -> Optional[np.ndarray]:
    """
    Extract a quantity from posterior samples, computing it if necessary.

    Args:
        posterior (Dict): Posterior samples dictionary
        quantity (str): Quantity name (e.g., 'log_likelihood', 'network_snr', 'H1_snr')

    Returns:
        Optional[np.ndarray]: Extracted quantity values, or None if not available
    """
    if quantity not in QUANTITY_CONFIG:
        raise ValueError(f"Unknown quantity: {quantity}")

    config = QUANTITY_CONFIG[quantity]

    # Handle network SNR specially
    if quantity == 'network_snr':
        return compute_network_snr(posterior)

    # For regular quantities, check if key exists
    key = config['key']
    if key in posterior:
        return np.array(posterior[key])

    return None


def plot_quantity_histogram(gw_event: str,
                            quantity: str = 'log_likelihood',
                            population: str = "gaussian",
                            source_type: str = "bns",
                            base_path: str = "../final_results/",
                            output_dir: str = "./figures/",
                            eos_samples: List[str] = None,
                            include_default: bool = True,
                            bins: int = 50,
                            alpha: float = 0.6,
                            normalize: bool = True,
                            show_stats: bool = True,
                            verbose: bool = False) -> str:
    """
    Create histogram of any quantity for a GW event with fixed population
    type and varying EOS samples.

    Args:
        gw_event (str): GW event name (e.g., 'GW170817', 'GW190425', 'GW230529')
        quantity (str): Quantity to plot ('log_likelihood', 'log_prior', 'network_snr', 'H1_snr', etc.)
        population (str): Population type ('uniform', 'gaussian', 'double_gaussian')
        source_type (str): Source type ('bns', 'nsbh', 'default')
        base_path (str): Base path to result files
        output_dir (str): Base output directory for figures
        eos_samples (List[str]): EOS samples to include (default: radio, radio_chiEFT, radio_NICER)
        include_default (bool): Include default uniform prior run
        bins (int): Number of bins for histogram
        alpha (float): Transparency for histogram bars (0-1)
        normalize (bool): Normalize histograms to density
        show_stats (bool): Print summary statistics
        verbose (bool): Print verbose information

    Returns:
        str: Path to saved figure
    """

    if quantity not in QUANTITY_CONFIG:
        raise ValueError(f"Unknown quantity: {quantity}. Available: {list(QUANTITY_CONFIG.keys())}")

    config = QUANTITY_CONFIG[quantity]

    # Set default parameters
    if eos_samples is None:
        eos_samples = ["radio", "radio_chiEFT", "radio_NICER"]

    print(f"Creating {quantity} histogram for {gw_event}")
    print(f"Population: {population}, Source: {source_type}")
    print(f"EOS samples: {eos_samples}")

    # Load quantity data for each EOS sample
    quantity_dict = {}

    for eos_name in eos_samples:
        result_path = construct_result_path(base_path, gw_event, population,
                                          source_type, eos_name)
        print(f"Loading {eos_name} from: {result_path}")

        if not os.path.exists(result_path):
            print(f"Warning: Result file not found: {result_path}")
            continue

        posterior = load_posterior_data(result_path, fast_mode=True)
        if posterior is not None:
            quantity_data = extract_quantity(posterior, quantity)
            if quantity_data is not None:
                quantity_dict[eos_name] = quantity_data
                if verbose:
                    print(f"Loaded {eos_name} with {len(quantity_dict[eos_name])} {quantity} values")
                    print(f"  Mean: {np.mean(quantity_dict[eos_name]):.2f}, Std: {np.std(quantity_dict[eos_name]):.2f}")
                    print(f"  Min: {np.min(quantity_dict[eos_name]):.2f}, Max: {np.max(quantity_dict[eos_name]):.2f}")
            else:
                print(f"Warning: Could not extract {quantity} for {eos_name}")
        else:
            print(f"Warning: Could not load posterior for {eos_name}")

    if not quantity_dict:
        raise ValueError(f"No {quantity} data could be loaded!")

    # Load default run if requested
    default_quantity = None
    if include_default:
        default_path = construct_result_path(base_path, gw_event, source_type,
                                           "default", "radio")
        print(f"Loading default run from: {default_path}")

        if os.path.exists(default_path):
            default_posterior = load_posterior_data(default_path, fast_mode=True)
            if default_posterior is not None:
                default_quantity = extract_quantity(default_posterior, quantity)
                if default_quantity is not None and verbose:
                    print(f"Loaded default run with {len(default_quantity)} {quantity} values")
                    print(f"  Mean: {np.mean(default_quantity):.2f}, Std: {np.std(default_quantity):.2f}")
                    print(f"  Min: {np.min(default_quantity):.2f}, Max: {np.max(default_quantity):.2f}")
                elif default_quantity is None:
                    print(f"Warning: Could not extract {quantity} from default run")
        else:
            print(f"Warning: Default run file not found: {default_path}")

    # Print summary statistics if requested
    if show_stats:
        print(f"\n=== {quantity} Statistics for {gw_event} ({population} {source_type}) ===")

        stats_data = []
        for eos_name, data in quantity_dict.items():
            stats_data.append({
                'name': eos_name,
                'mean': np.mean(data),
                'std': np.std(data),
                'median': np.median(data),
                'min': np.min(data),
                'max': np.max(data)
            })

        if default_quantity is not None:
            stats_data.append({
                'name': 'default',
                'mean': np.mean(default_quantity),
                'std': np.std(default_quantity),
                'median': np.median(default_quantity),
                'min': np.min(default_quantity),
                'max': np.max(default_quantity)
            })

        # Sort by mean (highest first)
        stats_data.sort(key=lambda x: x['mean'], reverse=True)

        print(f"{'Dataset':<20} {'Mean':<10} {'Median':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
        print("-" * 70)
        for stat in stats_data:
            print(f"{stat['name']:<20} {stat['mean']:<10.2f} {stat['median']:<10.2f} "
                  f"{stat['std']:<10.2f} {stat['min']:<10.2f} {stat['max']:<10.2f}")

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Determine global range for histograms
    all_data = np.concatenate([data for data in quantity_dict.values()])
    if default_quantity is not None:
        all_data = np.concatenate([all_data, default_quantity])

    data_min = np.min(all_data)
    data_max = np.max(all_data)
    data_range = (data_min, data_max)

    # Plot default run first (so it appears behind)
    if default_quantity is not None:
        ax.hist(default_quantity, bins=bins, range=data_range, alpha=1.0,
                color=DEFAULT_RUN_COLOR, label=DEFAULT_RUN_LABEL,
                density=normalize, histtype='step', linewidth=2.0)

    # Plot EOS runs
    for eos_name, data in quantity_dict.items():
        color = EOS_COLORS.get(eos_name, 'black')
        label = EOS_SAMPLES_NAMES_DICT.get(eos_name, eos_name)

        ax.hist(data, bins=bins, range=data_range, alpha=1.0,
                color=color, label=label, density=normalize,
                histtype='step', linewidth=2.0)

    # Formatting
    ax.set_xlabel(config['label'], fontsize=fs_labels)
    if normalize:
        ax.set_ylabel(config['ylabel_density'], fontsize=fs_labels)
    else:
        ax.set_ylabel(config['ylabel_count'], fontsize=fs_labels)

    # Add title
    title = f"{gw_event} - {population.replace('_', ' ').title()} Population"
    ax.set_title(title, fontsize=fs_title, weight='bold', pad=15)

    # Add legend
    ax.legend(fontsize=fs_legend, loc='best', framealpha=0.9)

    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    # Ensure output directory exists
    full_output_dir = os.path.join(output_dir, config['output_dir'])
    os.makedirs(full_output_dir, exist_ok=True)

    # Save figure
    output_filename = f"{gw_event}_{quantity}_{population}_{source_type}.pdf"
    output_path = os.path.join(full_output_dir, output_filename)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"\nSaved figure to: {output_path}")
    plt.close()

    return output_path


def plot_log_likelihood_histogram(gw_event: str,
                                   population: str = "gaussian",
                                   source_type: str = "bns",
                                   base_path: str = "../final_results/",
                                   output_dir: str = "./figures/",
                                   eos_samples: List[str] = None,
                                   include_default: bool = True,
                                   bins: int = 50,
                                   alpha: float = 0.6,
                                   normalize: bool = True,
                                   show_stats: bool = True,
                                   verbose: bool = False) -> str:
    """
    Create histogram of log-likelihood values for a GW event with fixed population
    type and varying EOS samples.

    DEPRECATED: This function now wraps plot_quantity_histogram for backwards compatibility.
    Use plot_quantity_histogram with quantity='log_likelihood' instead.

    Args:
        gw_event (str): GW event name (e.g., 'GW170817', 'GW190425', 'GW230529')
        population (str): Population type ('uniform', 'gaussian', 'double_gaussian')
        source_type (str): Source type ('bns', 'nsbh', 'default')
        base_path (str): Base path to result files
        output_dir (str): Output directory for figures
        eos_samples (List[str]): EOS samples to include (default: radio, radio_chiEFT, radio_NICER)
        include_default (bool): Include default uniform prior run
        bins (int): Number of bins for histogram
        alpha (float): Transparency for histogram bars (0-1)
        normalize (bool): Normalize histograms to density
        show_stats (bool): Print summary statistics
        verbose (bool): Print verbose information

    Returns:
        str: Path to saved figure
    """
    return plot_quantity_histogram(
        gw_event=gw_event,
        quantity='log_likelihood',
        population=population,
        source_type=source_type,
        base_path=base_path,
        output_dir=output_dir,
        eos_samples=eos_samples,
        include_default=include_default,
        bins=bins,
        alpha=alpha,
        normalize=normalize,
        show_stats=show_stats,
        verbose=verbose
    )


def plot_log_likelihood_cdf(gw_event: str,
                            population: str = "gaussian",
                            source_type: str = "bns",
                            base_path: str = "../final_results/",
                            output_dir: str = "./figures/logL/",
                            eos_samples: List[str] = None,
                            include_default: bool = True,
                            show_stats: bool = False,
                            verbose: bool = False) -> str:
    """
    Create cumulative distribution function (CDF) plot of log-likelihood values
    for a GW event with fixed population type and varying EOS samples.

    This function plots smooth CDF curves to show the cumulative probability
    distribution of log-likelihood values, making it easier to compare the
    relative quality of fit across different EOS constraints.

    Args:
        gw_event (str): GW event name (e.g., 'GW170817', 'GW190425', 'GW230529')
        population (str): Population type ('uniform', 'gaussian', 'double_gaussian')
        source_type (str): Source type ('bns', 'nsbh', 'default')
        base_path (str): Base path to result files
        output_dir (str): Output directory for figures
        eos_samples (List[str]): EOS samples to include (default: radio, radio_chiEFT, radio_NICER)
        include_default (bool): Include default uniform prior run
        show_stats (bool): Print summary statistics
        verbose (bool): Print verbose information

    Returns:
        str: Path to saved figure
    """

    # Set default parameters
    if eos_samples is None:
        eos_samples = ["radio", "radio_chiEFT", "radio_NICER"]

    print(f"Creating log-likelihood CDF for {gw_event}")
    print(f"Population: {population}, Source: {source_type}")
    print(f"EOS samples: {eos_samples}")

    # Load log-likelihood data for each EOS sample
    logL_dict = {}

    for eos_name in eos_samples:
        result_path = construct_result_path(base_path, gw_event, population,
                                          source_type, eos_name)

        if verbose:
            print(f"Loading {eos_name} from: {result_path}")

        if not os.path.exists(result_path):
            print(f"Warning: Result file not found: {result_path}")
            continue

        posterior = load_posterior_data(result_path, fast_mode=True)
        if posterior is not None and 'log_likelihood' in posterior:
            logL_dict[eos_name] = np.array(posterior['log_likelihood'])
            if verbose:
                print(f"Loaded {eos_name} with {len(logL_dict[eos_name])} log-likelihood values")
        else:
            print(f"Warning: Could not load log_likelihood for {eos_name}")

    if not logL_dict:
        raise ValueError("No log-likelihood data could be loaded!")

    # Load default run if requested
    default_logL = None
    if include_default:
        default_path = construct_result_path(base_path, gw_event, source_type,
                                           "default", "radio")

        if verbose:
            print(f"Loading default run from: {default_path}")

        if os.path.exists(default_path):
            default_posterior = load_posterior_data(default_path, fast_mode=True)
            if default_posterior is not None and 'log_likelihood' in default_posterior:
                default_logL = np.array(default_posterior['log_likelihood'])
                if verbose:
                    print(f"Loaded default run with {len(default_logL)} log-likelihood values")
            else:
                print("Warning: Could not load log_likelihood from default run")
        else:
            print(f"Warning: Default run file not found: {default_path}")

    # Print summary statistics if requested
    if show_stats:
        print(f"\n=== Log-Likelihood Statistics for {gw_event} ({population} {source_type}) ===")
        stats_data = []
        for eos_name, logL in logL_dict.items():
            stats_data.append({
                'name': eos_name,
                'mean': np.mean(logL),
                'median': np.median(logL)
            })

        if default_logL is not None:
            stats_data.append({
                'name': 'default',
                'mean': np.mean(default_logL),
                'median': np.median(default_logL)
            })

        stats_data.sort(key=lambda x: x['mean'], reverse=True)
        for stat in stats_data:
            print(f"{stat['name']:<20} Mean: {stat['mean']:.2f}, Median: {stat['median']:.2f}")

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot default run first (so it appears behind)
    if default_logL is not None:
        sorted_logL = np.sort(default_logL)
        cdf = np.arange(1, len(sorted_logL) + 1) / len(sorted_logL)
        ax.plot(sorted_logL, cdf, color=DEFAULT_RUN_COLOR,
                label=DEFAULT_RUN_LABEL, linewidth=2.5, alpha=0.9)

    # Plot EOS runs
    for eos_name, logL in logL_dict.items():
        color = EOS_COLORS.get(eos_name, 'black')
        label = EOS_SAMPLES_NAMES_DICT.get(eos_name, eos_name)

        # Compute empirical CDF
        sorted_logL = np.sort(logL)
        cdf = np.arange(1, len(sorted_logL) + 1) / len(sorted_logL)

        ax.plot(sorted_logL, cdf, color=color, label=label,
                linewidth=2.5, alpha=0.9)

    # Formatting
    ax.set_xlabel(r'$\log \mathcal{L}$', fontsize=fs_labels)
    ax.set_ylabel(r'Cumulative Probability', fontsize=fs_labels)

    # Add title (use hyphen instead of em dash for LaTeX compatibility)
    title = f"{gw_event} - {population.replace('_', ' ').title()} Population"
    ax.set_title(title, fontsize=fs_title, weight='bold', pad=15)

    # Add legend
    ax.legend(fontsize=fs_legend, loc='best', framealpha=0.9)

    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    # Set y-axis limits to [0, 1]
    ax.set_ylim([0, 1])

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save figure
    output_filename = f"{gw_event}_logL_cdf_{population}_{source_type}.pdf"
    output_path = os.path.join(output_dir, output_filename)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"Saved CDF figure to: {output_path}")
    plt.close()

    return output_path


def plot_all_events_logL_histograms(eos_samples: List[str] = None,
                                     include_default: bool = True,
                                     verbose: bool = False):
    """
    Convenience function to create log-likelihood histograms for all main events.

    Args:
        eos_samples (List[str]): EOS samples to include
        include_default (bool): Include default uninformed prior run
        verbose (bool): Print verbose information
    """

    if eos_samples is None:
        eos_samples = ["radio", "radio_chiEFT", "radio_NICER"]

    # GW170817 - BNS
    print("\n" + "="*60)
    print("GW170817 - BNS")
    print("="*60)

    for population in ["gaussian", "double_gaussian"]:
        try:
            plot_log_likelihood_histogram(
                gw_event="GW170817",
                population=population,
                source_type="bns",
                eos_samples=eos_samples,
                include_default=include_default,
                verbose=verbose
            )
        except Exception as e:
            print(f"Failed to create plot for GW170817 {population}: {e}")

    # GW190425 - BNS
    print("\n" + "="*60)
    print("GW190425 - BNS")
    print("="*60)

    for population in ["uniform", "double_gaussian"]:
        try:
            plot_log_likelihood_histogram(
                gw_event="GW190425",
                population=population,
                source_type="bns",
                eos_samples=eos_samples,
                include_default=include_default,
                verbose=verbose
            )
        except Exception as e:
            print(f"Failed to create plot for GW190425 {population}: {e}")

    # GW230529 - NSBH
    print("\n" + "="*60)
    print("GW230529 - NSBH")
    print("="*60)

    for population in ["gaussian", "uniform", "double_gaussian"]:
        try:
            plot_log_likelihood_histogram(
                gw_event="GW230529",
                population=population,
                source_type="nsbh",
                eos_samples=eos_samples,
                include_default=include_default,
                verbose=verbose
            )
        except Exception as e:
            print(f"Failed to create plot for GW230529 {population}: {e}")


def plot_all_events_logL_cdfs(eos_samples: List[str] = None,
                              include_default: bool = True,
                              verbose: bool = False):
    """
    Convenience function to create log-likelihood CDFs for all main events.

    Args:
        eos_samples (List[str]): EOS samples to include
        include_default (bool): Include default uninformed prior run
        verbose (bool): Print verbose information
    """

    if eos_samples is None:
        eos_samples = ["radio", "radio_chiEFT", "radio_NICER"]

    # GW170817 - BNS
    print("\n" + "="*60)
    print("GW170817 - BNS CDFs")
    print("="*60)

    for population in ["gaussian", "double_gaussian"]:
        try:
            plot_log_likelihood_cdf(
                gw_event="GW170817",
                population=population,
                source_type="bns",
                eos_samples=eos_samples,
                include_default=include_default,
                verbose=verbose
            )
        except Exception as e:
            print(f"Failed to create CDF plot for GW170817 {population}: {e}")

    # GW190425 - BNS
    print("\n" + "="*60)
    print("GW190425 - BNS CDFs")
    print("="*60)

    for population in ["uniform", "double_gaussian"]:
        try:
            plot_log_likelihood_cdf(
                gw_event="GW190425",
                population=population,
                source_type="bns",
                eos_samples=eos_samples,
                include_default=include_default,
                verbose=verbose
            )
        except Exception as e:
            print(f"Failed to create CDF plot for GW190425 {population}: {e}")

    # GW230529 - NSBH
    print("\n" + "="*60)
    print("GW230529 - NSBH CDFs")
    print("="*60)

    for population in ["gaussian", "uniform", "double_gaussian"]:
        try:
            plot_log_likelihood_cdf(
                gw_event="GW230529",
                population=population,
                source_type="nsbh",
                eos_samples=eos_samples,
                include_default=include_default,
                verbose=verbose
            )
        except Exception as e:
            print(f"Failed to create CDF plot for GW230529 {population}: {e}")


def debug_gw170817_bns_gaussian_chiEFT_threshold(
        base_path: str = "../final_results/",
        output_dir: str = "./figures/logL/debug/",
        verbose: bool = True) -> str:
    """
    Debug analysis for GW170817 BNS Gaussian population.

    This function:
    1. Loads the chiEFT posterior and finds the maximum log-likelihood value
    2. Loads the default/agnostic posterior
    3. Filters default posterior to keep only samples where log_likelihood > max(chiEFT log_likelihood)
    4. Creates scatter plots for chirp_mass, mass_ratio, and lambda_tilde

    Args:
        base_path (str): Base path to result files
        output_dir (str): Output directory for debug figures
        verbose (bool): Print verbose information

    Returns:
        str: Path to output directory containing saved figures
    """

    print("="*80)
    print("DEBUG: GW170817 BNS Gaussian - chiEFT threshold analysis")
    print("="*80)

    # Load chiEFT posterior
    chiEFT_path = construct_result_path(base_path, "GW170817", "gaussian",
                                       "bns", "radio_chiEFT")
    print(f"\nLoading chiEFT posterior from: {chiEFT_path}")

    if not os.path.exists(chiEFT_path):
        raise FileNotFoundError(f"chiEFT result file not found: {chiEFT_path}")

    chiEFT_posterior = load_posterior_data(chiEFT_path, fast_mode=False)
    if chiEFT_posterior is None or 'log_likelihood' not in chiEFT_posterior:
        raise ValueError("Could not load log_likelihood from chiEFT posterior")

    chiEFT_logL = np.array(chiEFT_posterior['log_likelihood'])
    max_chiEFT_logL = np.max(chiEFT_logL)

    if verbose:
        print(f"chiEFT posterior statistics:")
        print(f"  Number of samples: {len(chiEFT_logL)}")
        print(f"  Mean log_likelihood: {np.mean(chiEFT_logL):.2f}")
        print(f"  Max log_likelihood: {max_chiEFT_logL:.2f}")
        print(f"  Min log_likelihood: {np.min(chiEFT_logL):.2f}")

    # Load default posterior
    default_path = construct_result_path(base_path, "GW170817", "bns",
                                        "default", "radio")
    print(f"\nLoading default posterior from: {default_path}")

    if not os.path.exists(default_path):
        raise FileNotFoundError(f"Default result file not found: {default_path}")

    default_posterior = load_posterior_data(default_path, fast_mode=False)
    if default_posterior is None or 'log_likelihood' not in default_posterior:
        raise ValueError("Could not load log_likelihood from default posterior")

    default_logL = np.array(default_posterior['log_likelihood'])

    if verbose:
        print(f"Default posterior statistics:")
        print(f"  Number of samples: {len(default_logL)}")
        print(f"  Mean log_likelihood: {np.mean(default_logL):.2f}")
        print(f"  Max log_likelihood: {np.max(default_logL):.2f}")
        print(f"  Min log_likelihood: {np.min(default_logL):.2f}")

    # Filter default posterior based on chiEFT max log_likelihood
    print(f"\nFiltering default posterior where log_likelihood > {max_chiEFT_logL:.2f}")
    mask = default_logL > max_chiEFT_logL
    n_filtered = np.sum(mask)

    if verbose:
        print(f"  Samples passing threshold: {n_filtered} / {len(default_logL)} "
              f"({100*n_filtered/len(default_logL):.2f}%)")

    if n_filtered == 0:
        print("WARNING: No samples pass the threshold! Exiting.")
        return output_dir

    # Extract required parameters from filtered samples
    required_params = ['chirp_mass', 'mass_ratio', 'lambda_tilde']
    filtered_data = {}

    for param in required_params:
        if param not in default_posterior:
            raise ValueError(f"Required parameter '{param}' not found in default posterior")
        filtered_data[param] = np.array(default_posterior[param])[mask]

        if verbose:
            print(f"\n{param} statistics (filtered):")
            print(f"  Mean: {np.mean(filtered_data[param]):.4f}")
            print(f"  Median: {np.median(filtered_data[param]):.4f}")
            print(f"  Std: {np.std(filtered_data[param]):.4f}")
            print(f"  Range: [{np.min(filtered_data[param]):.4f}, {np.max(filtered_data[param]):.4f}]")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Create scatter plots
    param_labels = {
        'chirp_mass': r'$\mathcal{M}_c \, [M_\odot]$',
        'mass_ratio': r'$q$',
        'lambda_tilde': r'$\tilde{\Lambda}$'
    }

    print(f"\nCreating scatter plots...")

    # Plot 1: chirp_mass vs mass_ratio
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(filtered_data['chirp_mass'], filtered_data['mass_ratio'],
               alpha=0.5, s=10, c='steelblue', edgecolors='none')
    ax.set_xlabel(param_labels['chirp_mass'], fontsize=fs_labels)
    ax.set_ylabel(param_labels['mass_ratio'], fontsize=fs_labels)
    ax.set_title(f'GW170817 BNS Default (logL > {max_chiEFT_logL:.1f})',
                fontsize=fs_title, weight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    plt.tight_layout()

    output_path_1 = os.path.join(output_dir, "chirp_mass_vs_mass_ratio.pdf")
    plt.savefig(output_path_1, bbox_inches='tight', dpi=300)
    print(f"  Saved: {output_path_1}")
    plt.close()

    # Plot 2: chirp_mass vs lambda_tilde
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(filtered_data['chirp_mass'], filtered_data['lambda_tilde'],
               alpha=0.5, s=10, c='steelblue', edgecolors='none')
    ax.set_xlabel(param_labels['chirp_mass'], fontsize=fs_labels)
    ax.set_ylabel(param_labels['lambda_tilde'], fontsize=fs_labels)
    ax.set_title(f'GW170817 BNS Default (logL > {max_chiEFT_logL:.1f})',
                fontsize=fs_title, weight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    plt.tight_layout()

    output_path_2 = os.path.join(output_dir, "chirp_mass_vs_lambda_tilde.pdf")
    plt.savefig(output_path_2, bbox_inches='tight', dpi=300)
    print(f"  Saved: {output_path_2}")
    plt.close()

    # Plot 3: mass_ratio vs lambda_tilde
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(filtered_data['mass_ratio'], filtered_data['lambda_tilde'],
               alpha=0.5, s=10, c='steelblue', edgecolors='none')
    ax.set_xlabel(param_labels['mass_ratio'], fontsize=fs_labels)
    ax.set_ylabel(param_labels['lambda_tilde'], fontsize=fs_labels)
    ax.set_title(f'GW170817 BNS Default (logL > {max_chiEFT_logL:.1f})',
                fontsize=fs_title, weight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    plt.tight_layout()

    output_path_3 = os.path.join(output_dir, "mass_ratio_vs_lambda_tilde.pdf")
    plt.savefig(output_path_3, bbox_inches='tight', dpi=300)
    print(f"  Saved: {output_path_3}")
    plt.close()

    print(f"\n{'='*80}")
    print(f"Debug analysis complete! All plots saved to: {output_dir}")
    print(f"{'='*80}")

    return output_dir


def plot_all_quantities_for_event(gw_event: str,
                                   population: str,
                                   source_type: str,
                                   quantities: List[str] = None,
                                   eos_samples: List[str] = None,
                                   include_default: bool = True,
                                   verbose: bool = False):
    """
    Convenience function to plot multiple quantities for a single event.

    Args:
        gw_event (str): GW event name
        population (str): Population type
        source_type (str): Source type
        quantities (List[str]): List of quantities to plot (default: log_likelihood, log_prior, network_snr, detector SNRs)
        eos_samples (List[str]): EOS samples to include
        include_default (bool): Include default run
        verbose (bool): Print verbose information
    """
    if quantities is None:
        # Default: plot all available quantities
        quantities = ['log_likelihood', 'log_prior', 'network_snr', 'H1_snr', 'L1_snr', 'V1_snr']

    if eos_samples is None:
        eos_samples = ["radio", "radio_chiEFT", "radio_NICER"]

    for quantity in quantities:
        print(f"\n{'-'*60}")
        print(f"Plotting {quantity} for {gw_event} {population} {source_type}")
        print(f"{'-'*60}")
        try:
            plot_quantity_histogram(
                gw_event=gw_event,
                quantity=quantity,
                population=population,
                source_type=source_type,
                eos_samples=eos_samples,
                include_default=include_default,
                verbose=verbose
            )
        except Exception as e:
            print(f"Failed to create {quantity} plot: {e}")


def main():
    """Main function to generate all log-likelihood histograms and CDFs."""
    print("Generating log-likelihood histograms for all events...\n")
    plot_all_events_logL_histograms(verbose=True)
    print("\nAll log-likelihood histograms generated successfully!")

    print("\n" + "="*80)
    print("Generating log-likelihood CDFs for all events...\n")
    plot_all_events_logL_cdfs(verbose=True)
    print("\nAll log-likelihood CDFs generated successfully!")

    print("\n" + "="*80)
    print("Generating all quantities (log_prior, SNRs) for all events...\n")

    # GW170817 - BNS
    for population in ["gaussian", "double_gaussian"]:
        plot_all_quantities_for_event("GW170817", population, "bns", verbose=True)

    # GW190425 - BNS
    for population in ["uniform", "double_gaussian"]:
        plot_all_quantities_for_event("GW190425", population, "bns", verbose=True)

    # GW230529 - NSBH
    for population in ["gaussian", "uniform", "double_gaussian"]:
        plot_all_quantities_for_event("GW230529", population, "nsbh", verbose=True)

    print("\nAll quantity histograms generated successfully!")

    print("\n" + "="*80)
    print("Running debug analysis for GW170817 BNS Gaussian...\n")
    debug_gw170817_bns_gaussian_chiEFT_threshold(verbose=True)
    print("\nDebug analysis completed!")


if __name__ == "__main__":
    main()
