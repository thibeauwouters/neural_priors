"""
Create prior vs posterior comparison for tidal deformability parameters.

This script generates "prior" samples by:
1. Sampling posterior masses (m1_source, m2_source)
2. Sampling EOS mass-Lambda curves
3. Computing Lambda_1, Lambda_2 at those masses using the EOS curves
4. Converting to lambda_tilde and delta_lambda_tilde

Then creates KDE plots comparing the "prior" (computed from EOS) vs "posterior"
(from parameter estimation) for different GW events.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from scipy.interpolate import interp1d
from scipy.stats import gaussian_kde

from bilby.gw.conversion import lambda_1_lambda_2_to_lambda_tilde, lambda_1_lambda_2_to_delta_lambda_tilde
from utils import (
    construct_result_path, load_posterior_data,
    EOS_COLORS, EOS_SAMPLES_NAMES_DICT
)

# Font sizes (from logL.py)
from logL import fs_ticks, fs_labels, fs_legend, fs_title, title_pad

# Matplotlib style parameters (from logL.py)
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
    "text.latex.preamble": r"\usepackage{amsmath}",
}

plt.rcParams.update(params)


def load_eos_curves(eos_name: str, eos_data_dir: str = "../data/eos") -> Tuple[np.ndarray, np.ndarray]:
    """
    Load EOS mass-Lambda curves.

    Args:
        eos_name (str): Name of EOS (e.g., 'radio', 'radio_chiEFT', 'radio_NICER')
        eos_data_dir (str): Base directory containing EOS data

    Returns:
        Tuple[np.ndarray, np.ndarray]: masses_EOS (n_curves, n_points), Lambdas_EOS (n_curves, n_points)
    """
    eos_path = os.path.join(eos_data_dir, eos_name, "eos_samples.npz")

    if not os.path.exists(eos_path):
        raise FileNotFoundError(f"EOS data not found: {eos_path}")

    print(f"Loading EOS curves from: {eos_path}")
    eos_data = np.load(eos_path)

    # Load mass and Lambda curves
    # Format: (n_curves, n_points) where each curve is a different EOS realization
    masses_EOS = eos_data['masses_EOS']
    Lambdas_EOS = eos_data['Lambdas_EOS']

    print(f"Loaded {len(masses_EOS)} EOS curves with {masses_EOS.shape[1]} points each")

    return masses_EOS, Lambdas_EOS


def interpolate_lambda_from_mass(mass: float, masses_curve: np.ndarray,
                                 lambdas_curve: np.ndarray) -> float:
    """
    Interpolate Lambda value at a given mass using an EOS curve.

    Args:
        mass (float): Mass value to interpolate at (solar masses)
        masses_curve (np.ndarray): Mass values for this EOS curve
        lambdas_curve (np.ndarray): Lambda values for this EOS curve

    Returns:
        float: Interpolated Lambda value
    """
    # Create interpolator (linear interpolation, extrapolate if needed)
    interpolator = interp1d(masses_curve, lambdas_curve,
                           kind='linear', bounds_error=False,
                           fill_value=(lambdas_curve[0], lambdas_curve[-1]))

    return float(interpolator(mass))


def create_prior_samples(posterior: Dict[str, np.ndarray],
                        masses_EOS: np.ndarray,
                        Lambdas_EOS: np.ndarray,
                        n_samples: int = 10000,
                        verbose: bool = False) -> Dict[str, np.ndarray]:
    """
    Create "prior" samples by sampling posterior masses and computing Lambdas from EOS curves.

    Only keeps physical samples where:
    - m1 >= m2 (mass ordering)
    - lambda_1 >= 0 and lambda_2 >= 0 (non-negative tidal deformabilities)

    Args:
        posterior (Dict): Posterior samples containing mass_1_source, mass_2_source
        masses_EOS (np.ndarray): EOS mass curves (n_curves, n_points)
        Lambdas_EOS (np.ndarray): EOS Lambda curves (n_curves, n_points)
        n_samples (int): Number of valid samples to generate
        verbose (bool): Print progress information

    Returns:
        Dict: Dictionary containing mass_1_source, mass_2_source, lambda_1, lambda_2,
              lambda_tilde, delta_lambda_tilde arrays
    """
    n_curves = len(masses_EOS)
    n_posterior_samples = len(posterior['mass_1_source'])

    if verbose:
        print(f"Creating {n_samples} prior samples...")
        print(f"Posterior has {n_posterior_samples} samples")
        print(f"EOS has {n_curves} curves")

    # Initialize arrays
    m1_samples = np.zeros(n_samples)
    m2_samples = np.zeros(n_samples)
    lambda_1_samples = np.zeros(n_samples)
    lambda_2_samples = np.zeros(n_samples)

    # Sample and compute - continue until we have n_samples valid samples
    good_samples = 0
    total_attempts = 0

    while good_samples < n_samples:
        # Sample a random posterior index to get masses
        posterior_idx = np.random.randint(0, n_posterior_samples)
        m1 = posterior['mass_1_source'][posterior_idx]
        m2 = posterior['mass_2_source'][posterior_idx]

        # Check mass ordering
        if m1 < m2:
            total_attempts += 1
            continue

        # Sample a random EOS curve
        eos_idx = np.random.randint(0, n_curves)
        masses_curve = masses_EOS[eos_idx]
        lambdas_curve = Lambdas_EOS[eos_idx]

        # Interpolate Lambda values at the sampled masses
        lambda_1 = interpolate_lambda_from_mass(m1, masses_curve, lambdas_curve)
        lambda_2 = interpolate_lambda_from_mass(m2, masses_curve, lambdas_curve)

        # Check for non-negative lambdas
        if lambda_1 < 0 or lambda_2 < 0:
            total_attempts += 1
            continue

        # Check for NaN or Inf
        if not (np.isfinite(lambda_1) and np.isfinite(lambda_2)):
            total_attempts += 1
            continue

        # Check lambda ordering (lambda_2 > lambda_1 for m2 < m1)
        if lambda_2 <= lambda_1:
            total_attempts += 1
            continue

        # Store valid sample
        m1_samples[good_samples] = m1
        m2_samples[good_samples] = m2
        lambda_1_samples[good_samples] = lambda_1
        lambda_2_samples[good_samples] = lambda_2

        good_samples += 1
        total_attempts += 1

    # Convert to tilde parameters
    lambda_tilde_samples = lambda_1_lambda_2_to_lambda_tilde(
        lambda_1_samples, lambda_2_samples, m1_samples, m2_samples
    )

    delta_lambda_tilde_samples = lambda_1_lambda_2_to_delta_lambda_tilde(
        lambda_1_samples, lambda_2_samples, m1_samples, m2_samples
    )

    if verbose:
        acceptance_rate = 100 * good_samples / total_attempts
        rejected = total_attempts - good_samples
        print(f"Generated {good_samples} valid samples from {total_attempts} attempts")
        print(f"Rejected {rejected} unphysical samples (acceptance rate: {acceptance_rate:.1f}%)")
        print(f"Lambda_tilde range: [{np.min(lambda_tilde_samples):.1f}, {np.max(lambda_tilde_samples):.1f}]")
        print(f"Delta_lambda_tilde range: [{np.min(delta_lambda_tilde_samples):.1f}, {np.max(delta_lambda_tilde_samples):.1f}]")

    return {
        'mass_1_source': m1_samples,
        'mass_2_source': m2_samples,
        'lambda_1': lambda_1_samples,
        'lambda_2': lambda_2_samples,
        'lambda_tilde': lambda_tilde_samples,
        'delta_lambda_tilde': delta_lambda_tilde_samples
    }




def plot_combined_prior_vs_posterior(event_configs: List[Dict],
                                     posterior_dict_all: Dict[str, Dict[str, Dict[str, np.ndarray]]],
                                     prior_dict_all: Dict[str, Dict[str, Dict[str, np.ndarray]]],
                                     output_dir: str = "./figures/prior_vs_posterior",
                                     output_filename: str = "combined_prior_vs_posterior.pdf",
                                     quantile_range: Tuple[float, float] = (0.0001, 0.9975)):
    """
    Create combined KDE comparison plot with 3 vertical panels (one per event).

    Each panel shows prior vs posterior for lambda_tilde (or lambda_2 for NSBH)
    with 3 curves for the 3 different EOSs.

    Args:
        event_configs (List[Dict]): List of event configurations
        posterior_dict_all (Dict): Nested dict: event -> EOS -> parameter data
        prior_dict_all (Dict): Nested dict: event -> EOS -> parameter data
        output_dir (str): Output directory for figures
        output_filename (str): Output filename
        quantile_range (Tuple[float, float]): Quantile range for x-axis limits
    """
    # Create figure with 3 vertical panels
    fig, axes = plt.subplots(3, 1, figsize=(8, 12))

    # Collect legend handles and labels for top panel
    from matplotlib.patches import Rectangle
    from matplotlib.lines import Line2D

    # Plot each event in a separate panel
    for idx, config in enumerate(event_configs):
        ax = axes[idx]

        gw_event = config['gw_event']
        population = config['population']
        source_type = config['source_type']
        eos_samples = config['eos_samples']

        # Determine parameter to plot based on source type
        if source_type.lower() == 'nsbh':
            param = 'lambda_2'
            param_label = r'$\Lambda_2$'
        else:
            param = 'lambda_tilde'
            param_label = r'$\tilde{\Lambda}$'

        # Get data for this event
        posterior_dict = posterior_dict_all[gw_event]
        prior_dict = prior_dict_all[gw_event]

        # Determine x-axis limits using quantiles from prior data
        all_prior_data = []
        for eos_name in eos_samples:
            if eos_name in prior_dict and param in prior_dict[eos_name]:
                all_prior_data.append(prior_dict[eos_name][param])

        if all_prior_data:
            all_prior_combined = np.concatenate(all_prior_data)
            x_min = np.quantile(all_prior_combined, quantile_range[0])
            x_max = np.quantile(all_prior_combined, quantile_range[1])
            x_min = max(0, x_min)  # Don't go below 0 for Lambda parameters
        else:
            x_min, x_max = 0, 1000  # Fallback

        # Plot each EOS
        for eos_name in eos_samples:
            if eos_name not in posterior_dict or eos_name not in prior_dict:
                print(f"Warning: Missing data for {gw_event} {eos_name}")
                continue

            color = EOS_COLORS.get(eos_name, 'black')

            # Get posterior and prior data
            posterior_data = posterior_dict[eos_name][param]
            prior_data = prior_dict[eos_name][param]

            # Create KDEs
            posterior_kde = gaussian_kde(posterior_data)
            prior_kde = gaussian_kde(prior_data)

            # Create x grid based on determined limits
            x_grid = np.linspace(x_min, x_max, 500)

            # Evaluate KDEs
            posterior_kde_vals = posterior_kde(x_grid)
            prior_kde_vals = prior_kde(x_grid)

            # Plot posterior (solid line with fill)
            ax.fill_between(x_grid, 0, posterior_kde_vals,
                           color=color, alpha=0.3)
            ax.plot(x_grid, posterior_kde_vals, color=color, linewidth=2.5,
                   linestyle='-')

            # Plot prior (dashed line with lighter fill)
            ax.fill_between(x_grid, 0, prior_kde_vals,
                           color=color, alpha=0.15)
            ax.plot(x_grid, prior_kde_vals, color=color, linewidth=2.5,
                   linestyle='--')

        # Formatting
        ax.set_xlabel(param_label, fontsize=fs_labels)
        ax.set_ylabel('Probability Density', fontsize=fs_labels)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(bottom=0)

        # Add title for each event
        title = f"{gw_event} - {population.replace('_', ' ').title()} population"
        ax.set_title(title, fontsize=fs_title, weight='bold', pad=title_pad)

        # Add custom legend only to top subplot
        if idx == 0:
            # Create legend handles
            legend_handles = []
            legend_labels = []

            # Add EOS color squares
            for eos_name in eos_samples:
                color = EOS_COLORS.get(eos_name, 'black')
                label = EOS_SAMPLES_NAMES_DICT.get(eos_name, eos_name)
                legend_handles.append(Rectangle((0, 0), 1, 1, fc=color, alpha=1.0, edgecolor='none'))
                legend_labels.append(label)

            # Add posterior and prior line styles
            legend_handles.append(Line2D([0], [0], color='black', linewidth=2.5, linestyle='-'))
            legend_labels.append('Posterior')
            legend_handles.append(Line2D([0], [0], color='black', linewidth=2.5, linestyle='--'))
            legend_labels.append('Prior')

            ax.legend(legend_handles, legend_labels, fontsize=fs_legend,
                     loc='upper right', framealpha=0.9)

    plt.tight_layout()

    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, output_filename)
    plt.savefig(filepath, bbox_inches='tight', dpi=300)
    print(f"\nSaved combined figure to: {filepath}")

    plt.close()


def generate_prior_samples_for_event(gw_event: str,
                                     population: str,
                                     source_type: str,
                                     eos_samples: List[str],
                                     base_path: str = "../final_results/",
                                     eos_data_dir: str = "../data/eos",
                                     n_samples: int = 10000,
                                     verbose: bool = True) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Generate prior samples for all EOS samples for a given event configuration.

    Args:
        gw_event (str): GW event name
        population (str): Population type
        source_type (str): Source type
        eos_samples (List[str]): List of EOS names
        base_path (str): Base path to PE results
        eos_data_dir (str): Directory containing EOS data
        n_samples (int): Number of prior samples to generate
        verbose (bool): Print progress information

    Returns:
        Dict[str, Dict[str, np.ndarray]]: Prior samples for each EOS
    """
    print(f"\nGenerating prior samples for {gw_event} ({population} {source_type})...")

    prior_dict = {}

    for eos_name in eos_samples:
        # Load posterior samples
        result_path = construct_result_path(base_path, gw_event, population,
                                          source_type, eos_name)

        if not os.path.exists(result_path):
            print(f"  Warning: Result file not found: {result_path}")
            continue

        posterior = load_posterior_data(result_path, fast_mode=True)

        if posterior is None:
            print(f"  Warning: Could not load posterior for {eos_name}")
            continue

        # Check for required parameters
        if 'mass_1_source' not in posterior or 'mass_2_source' not in posterior:
            print(f"  Warning: mass_1_source or mass_2_source not in posterior for {eos_name}")
            continue

        # Load EOS curves
        try:
            masses_EOS, Lambdas_EOS = load_eos_curves(eos_name, eos_data_dir)
        except FileNotFoundError as e:
            print(f"  Warning: {e}")
            continue

        # Generate prior samples
        prior_samples = create_prior_samples(posterior, masses_EOS, Lambdas_EOS,
                                            n_samples=n_samples, verbose=verbose)

        prior_dict[eos_name] = prior_samples
        print(f"  Generated {n_samples} prior samples for {eos_name}")

    return prior_dict


def load_posterior_data_for_event(gw_event: str,
                                  population: str,
                                  source_type: str,
                                  eos_samples: List[str],
                                  base_path: str = "../final_results/") -> Dict[str, Dict]:
    """
    Load posterior data for a single event.

    Args:
        gw_event (str): GW event name
        population (str): Population type
        source_type (str): Source type
        eos_samples (List[str]): List of EOS names
        base_path (str): Base path to PE results

    Returns:
        Dict: posterior_dict for each EOS
    """
    print(f"\nLoading posterior data for {gw_event} ({population} {source_type})...")

    # Load posterior samples
    posterior_dict = {}
    for eos_name in eos_samples:
        result_path = construct_result_path(base_path, gw_event, population,
                                          source_type, eos_name)

        if not os.path.exists(result_path):
            print(f"  Warning: Result file not found: {result_path}")
            continue

        posterior = load_posterior_data(result_path, fast_mode=True)

        if posterior is None:
            print(f"  Warning: Could not load posterior for {eos_name}")
            continue

        # Ensure we have tilde parameters for BNS
        if source_type.lower() == 'bns':
            if 'lambda_tilde' not in posterior:
                if 'lambda_1' in posterior and 'lambda_2' in posterior:
                    from bilby.gw.conversion import chirp_mass_and_mass_ratio_to_component_masses
                    m1, m2 = chirp_mass_and_mass_ratio_to_component_masses(
                        posterior['chirp_mass'], posterior['mass_ratio'])
                    posterior['lambda_tilde'] = lambda_1_lambda_2_to_lambda_tilde(
                        posterior['lambda_1'], posterior['lambda_2'], m1, m2)
                    posterior['delta_lambda_tilde'] = lambda_1_lambda_2_to_delta_lambda_tilde(
                        posterior['lambda_1'], posterior['lambda_2'], m1, m2)

        posterior_dict[eos_name] = posterior
        print(f"  Loaded posterior for {eos_name}")

    return posterior_dict


def main():
    """Main function to generate prior samples and create combined KDE plot."""

    # Configuration for specific events
    events_config = [
        {
            'gw_event': 'GW170817',
            'population': 'gaussian',
            'source_type': 'bns',
            'eos_samples': ['radio', 'radio_chiEFT', 'radio_NICER']
        },
        {
            'gw_event': 'GW190425',
            'population': 'uniform',
            'source_type': 'bns',
            'eos_samples': ['radio', 'radio_chiEFT', 'radio_NICER']
        },
        {
            'gw_event': 'GW230529',
            'population': 'gaussian',
            'source_type': 'nsbh',
            'eos_samples': ['radio', 'radio_chiEFT', 'radio_NICER']
        }
    ]

    # Number of prior samples to generate
    n_samples = 10000

    # Step 1: Load posterior data and generate prior samples
    print("\n" + "="*60)
    print("STEP 1: Loading Posterior Data and Generating Prior Samples")
    print("="*60)

    posterior_dict_all = {}
    prior_dict_all = {}

    for config in events_config:
        # Load posterior data
        posterior_dict = load_posterior_data_for_event(
            gw_event=config['gw_event'],
            population=config['population'],
            source_type=config['source_type'],
            eos_samples=config['eos_samples']
        )

        # Generate prior samples on-the-fly
        prior_dict = generate_prior_samples_for_event(
            gw_event=config['gw_event'],
            population=config['population'],
            source_type=config['source_type'],
            eos_samples=config['eos_samples'],
            n_samples=n_samples,
            verbose=False
        )

        posterior_dict_all[config['gw_event']] = posterior_dict
        prior_dict_all[config['gw_event']] = prior_dict

    # Step 2: Create combined KDE comparison plot
    print("\n" + "="*60)
    print("STEP 2: Creating Combined KDE Comparison Plot")
    print("="*60)

    plot_combined_prior_vs_posterior(
        event_configs=events_config,
        posterior_dict_all=posterior_dict_all,
        prior_dict_all=prior_dict_all
    )

    print("\n" + "="*60)
    print("DONE!")
    print("="*60)


if __name__ == "__main__":
    main()
