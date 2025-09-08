"""
Comparison plots for mass_1_source vs mass_2_source between default and best evidence runs.

This script creates scatter plots comparing the default (agnostic prior) run
with the highest evidence run from the Bayes factor analysis.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import corner
from typing import Dict, Tuple, List

from utils import (
    construct_result_path, load_posterior_data, setup_matplotlib_style,
    EOS_COLORS
)

# Setup matplotlib style
setup_matplotlib_style()

# Color constants
DEFAULT_RUN_PLOT_COLOR = 'lightgray'

def load_bayes_factors(bayes_factors_path: str = "../bayes_factors/all_bayes_factors.json") -> Dict:
    """Load Bayes factors from JSON file."""
    with open(bayes_factors_path, 'r') as f:
        return json.load(f)

def find_highest_evidence_run(bayes_factors: Dict, gw_event: str) -> Tuple[str, str, str, float]:
    """
    Find the run with highest evidence (Bayes factor) for a given GW event.
    
    Args:
        bayes_factors (Dict): Loaded Bayes factors data
        gw_event (str): GW event name
        
    Returns:
        Tuple[str, str, str, float]: (source_type, population, eos_name, max_bf_value)
    """
    max_bf = -np.inf
    best_run = None
    
    for source_type in ['bns', 'nsbh']:
        if source_type in bayes_factors:
            for population in bayes_factors[source_type]:
                if isinstance(bayes_factors[source_type][population], dict):
                    for eos_name in bayes_factors[source_type][population]:
                        if isinstance(bayes_factors[source_type][population][eos_name], dict):
                            if gw_event in bayes_factors[source_type][population][eos_name]:
                                bf_value = bayes_factors[source_type][population][eos_name][gw_event]
                                if bf_value > max_bf:
                                    max_bf = bf_value
                                    best_run = (source_type, population, eos_name, bf_value)
    
    if best_run is None:
        raise ValueError(f"No Bayes factors found for {gw_event}")
    
    return best_run

def plot_mass_comparison_single(events: List[str] = None,
                              base_path: str = "../final_results/",
                              output_dir: str = "./figures/mass_comparison/",
                              bayes_factors_path: str = "../bayes_factors/all_bayes_factors.json",
                              figsize: Tuple[int, int] = (10, 10)) -> str:
    """
    Create single mass_1_source vs mass_2_source comparison plot with all events.
    
    Args:
        events (List[str]): List of GW events to include
        base_path (str): Base path to result files
        output_dir (str): Output directory for figures
        bayes_factors_path (str): Path to Bayes factors JSON file
        figsize (Tuple[int, int]): Figure size
        
    Returns:
        str: Path to saved figure
    """
    if events is None:
        events = ['GW170817', 'GW190425', 'GW230529']
    
    print(f"Creating combined mass comparison plot for events: {events}")
    
    # Load Bayes factors
    bayes_factors = load_bayes_factors(bayes_factors_path)
    
    # Create figure with marginal histograms
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 2, width_ratios=[3, 1], height_ratios=[1, 3], 
                          hspace=0.05, wspace=0.05)
    
    ax_main = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)
    
    # Color scheme for events
    event_colors = ['#FF69B4', '#9ACD32', '#FF4500']  # Pink, YellowGreen, OrangeRed
    
    all_m1_data = []
    all_m2_data = []
    legend_entries = []
    
    # Process each event
    for i, gw_event in enumerate(events):
        try:
            # Find best evidence run
            best_source_type, best_population, best_eos, best_bf = find_highest_evidence_run(bayes_factors, gw_event)
            print(f"{gw_event}: Best evidence run: {best_source_type}/{best_population}/{best_eos} (ln(BF) = {best_bf:.2f})")
            
            # Load default run data (agnostic prior)
            default_path = construct_result_path(base_path, gw_event, best_source_type, "default", "radio")
            
            if not os.path.exists(default_path):
                print(f"Warning: Default run file not found for {gw_event}: {default_path}")
                continue
            
            default_posterior = load_posterior_data(default_path, fast_mode=True)
            if default_posterior is None:
                print(f"Warning: Could not load default run data for {gw_event}")
                continue
            
            # Load best evidence run data
            best_path = construct_result_path(base_path, gw_event, best_population, best_source_type, best_eos)
            
            if not os.path.exists(best_path):
                print(f"Warning: Best evidence run file not found for {gw_event}: {best_path}")
                continue
            
            best_posterior = load_posterior_data(best_path, fast_mode=True)
            if best_posterior is None:
                print(f"Warning: Could not load best evidence run data for {gw_event}")
                continue
            
            # Check if mass_1_source and mass_2_source are available
            required_keys = ['mass_1_source', 'mass_2_source']
            if not all(key in default_posterior for key in required_keys):
                print(f"Warning: Required mass parameters not found in default posterior for {gw_event}")
                continue
            if not all(key in best_posterior for key in required_keys):
                print(f"Warning: Required mass parameters not found in best evidence posterior for {gw_event}")
                continue
            
            event_color = event_colors[i % len(event_colors)]
            
            # Plot contours for default run (light version of event color)
            default_samples = np.column_stack([default_posterior['mass_1_source'], 
                                             default_posterior['mass_2_source']])
            
            corner.hist2d(default_samples[:, 0], default_samples[:, 1], 
                         ax=ax_main, 
                         color=event_color,
                         alpha=0.3,
                         plot_datapoints=False,
                         plot_density=False,
                         fill_contours=True,
                         levels=[0.5, 0.9])
            
            # Plot contours for best evidence run (full color)
            best_samples = np.column_stack([best_posterior['mass_1_source'], 
                                          best_posterior['mass_2_source']])
            
            corner.hist2d(best_samples[:, 0], best_samples[:, 1], 
                         ax=ax_main,
                         color=event_color,
                         alpha=0.8,
                         plot_datapoints=False,
                         plot_density=False,
                         fill_contours=True,
                         levels=[0.5, 0.9])
            
            # Store data for marginal plots
            all_m1_data.extend([default_posterior['mass_1_source'], best_posterior['mass_1_source']])
            all_m2_data.extend([default_posterior['mass_2_source'], best_posterior['mass_2_source']])
            
            # Add to legend
            legend_entries.append((gw_event, event_color))
            
            # Plot marginal histograms
            ax_top.hist(default_posterior['mass_1_source'], bins=30, alpha=0.3, 
                       color=event_color, density=True, histtype='stepfilled')
            ax_top.hist(best_posterior['mass_1_source'], bins=30, alpha=0.8, 
                       color=event_color, density=True, histtype='step', linewidth=2)
            
            ax_right.hist(default_posterior['mass_2_source'], bins=30, alpha=0.3, 
                         color=event_color, density=True, histtype='stepfilled', 
                         orientation='horizontal')
            ax_right.hist(best_posterior['mass_2_source'], bins=30, alpha=0.8, 
                         color=event_color, density=True, histtype='step', 
                         linewidth=2, orientation='horizontal')
            
        except Exception as e:
            print(f"Error processing {gw_event}: {e}")
            continue
    
    # Set labels and formatting for main plot
    ax_main.set_xlabel(r'$m_1^{\rm src}$ [$M_{\odot}$]', fontsize=16)
    ax_main.set_ylabel(r'$m_2^{\rm src}$ [$M_{\odot}$]', fontsize=16)
    
    # Add equal mass ratio lines
    if all_m1_data and all_m2_data:
        all_m1 = np.concatenate(all_m1_data)
        all_m2 = np.concatenate(all_m2_data)
        
        mass_min = min(np.min(all_m1), np.min(all_m2))
        mass_max = max(np.max(all_m1), np.max(all_m2))
        
        # Mass ratio lines removed as requested
        
        ax_main.set_xlim(mass_min * 0.95, mass_max * 1.05)
        ax_main.set_ylim(mass_min * 0.95, mass_max * 1.05)
    
    # Format marginal plots
    ax_top.tick_params(labelbottom=False)
    ax_right.tick_params(labelleft=False)
    ax_top.set_ylabel('Density', fontsize=12)
    ax_right.set_xlabel('Density', fontsize=12)
    
    # Legend removed as requested
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    output_filename = "all_events_mass_comparison_contours.pdf"
    output_path = os.path.join(output_dir, output_filename)
    
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"Saved figure to: {output_path}")
    plt.close()
    
    return output_path

def plot_mass_corner_individual(gw_event: str,
                               base_path: str = "../final_results/",
                               output_dir: str = "./figures/mass_comparison/",
                               bayes_factors_path: str = "../bayes_factors/all_bayes_factors.json",
                               figsize: Tuple[int, int] = (8, 8)) -> str:
    """
    Create individual corner plot for mass_1_source vs mass_2_source comparison.
    
    Args:
        gw_event (str): GW event name
        base_path (str): Base path to result files
        output_dir (str): Output directory for figures
        bayes_factors_path (str): Path to Bayes factors JSON file
        figsize (Tuple[int, int]): Figure size
        
    Returns:
        str: Path to saved figure
    """
    print(f"Creating corner plot for {gw_event}")
    
    # Load Bayes factors and find best run
    bayes_factors = load_bayes_factors(bayes_factors_path)
    best_source_type, best_population, best_eos, best_bf = find_highest_evidence_run(bayes_factors, gw_event)
    
    print(f"Best evidence run: {best_source_type}/{best_population}/{best_eos} (ln(BF) = {best_bf:.2f})")
    
    # Load default run data (agnostic prior)
    default_path = construct_result_path(base_path, gw_event, best_source_type, "default", "radio")
    print(f"Loading default run from: {default_path}")
    
    if not os.path.exists(default_path):
        raise FileNotFoundError(f"Default run file not found: {default_path}")
    
    default_posterior = load_posterior_data(default_path, fast_mode=True)
    if default_posterior is None:
        raise ValueError("Could not load default run data")
    
    # Load best evidence run data
    best_path = construct_result_path(base_path, gw_event, best_population, best_source_type, best_eos)
    print(f"Loading best evidence run from: {best_path}")
    
    if not os.path.exists(best_path):
        raise FileNotFoundError(f"Best evidence run file not found: {best_path}")
    
    best_posterior = load_posterior_data(best_path, fast_mode=True)
    if best_posterior is None:
        raise ValueError("Could not load best evidence run data")
    
    # Check if mass_1_source and mass_2_source are available
    for key in ['mass_1_source', 'mass_2_source']:
        if key not in default_posterior:
            raise ValueError(f"Parameter '{key}' not found in default posterior")
        if key not in best_posterior:
            raise ValueError(f"Parameter '{key}' not found in best evidence posterior")
    
    # Prepare samples for corner plot
    default_samples = np.column_stack([default_posterior['mass_1_source'], 
                                     default_posterior['mass_2_source']])
    
    best_samples = np.column_stack([best_posterior['mass_1_source'], 
                                  best_posterior['mass_2_source']])
    
    # Parameter labels
    labels = [r'$m_1^{\rm src}$ [$M_{\odot}$]', r'$m_2^{\rm src}$ [$M_{\odot}$]']
    
    # Calculate ranges
    all_samples = np.vstack([default_samples, best_samples])
    ranges = [(np.min(all_samples[:, i]) * 0.95, np.max(all_samples[:, i]) * 1.05) 
             for i in range(2)]
    
    # Corner plot configuration
    corner_kwargs = {
        'labels': labels,
        'range': ranges,
        'bins': 40,
        'smooth': 1.0,
        'show_titles': False,
        'plot_density': True,
        'plot_datapoints': False,
        'fill_contours': True,
        'max_n_ticks': 4,
        'min_n_ticks': 3,
        'label_kwargs': {'fontsize': 16},
        'title_kwargs': {'fontsize': 16}
    }
    
    # Create corner plot - default first (background)
    fig = corner.corner(default_samples, 
                       color=DEFAULT_RUN_PLOT_COLOR,
                       hist_kwargs={'color': 'gray', 'density': True, 'alpha': 0.7},
                       **corner_kwargs)
    
    # Overlay best evidence run
    best_color = EOS_COLORS.get(best_eos, 'red')
    corner.corner(best_samples, 
                 fig=fig,
                 color=best_color,
                 hist_kwargs={'color': best_color, 'density': True, 'alpha': 0.8},
                 **corner_kwargs)
    
    # Mass ratio lines removed as requested
    
    # Legend and title removed as requested
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    output_filename = f"{gw_event}_mass_corner.pdf"
    output_path = os.path.join(output_dir, output_filename)
    
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"Saved figure to: {output_path}")
    plt.close()
    
    return output_path

def plot_all_individual_mass_corners() -> List[str]:
    """
    Create individual corner plots for all available GW events.
    
    Returns:
        List[str]: Paths to saved figures
    """
    events = ['GW170817', 'GW190425', 'GW230529']
    output_paths = []
    
    for event in events:
        try:
            path = plot_mass_corner_individual(event)
            output_paths.append(path)
        except Exception as e:
            print(f"Error creating corner plot for {event}: {e}")
    
    return output_paths

def main():
    """Main function to create mass comparison plots."""
    print("Creating combined mass comparison plot...")
    combined_path = plot_mass_comparison_single()
    print(f"Created combined plot: {combined_path}")
    
    print("\nCreating individual corner plots...")
    individual_paths = plot_all_individual_mass_corners()
    print(f"Created {len(individual_paths)} individual corner plots:")
    for path in individual_paths:
        print(f"  - {path}")

if __name__ == "__main__":
    main()