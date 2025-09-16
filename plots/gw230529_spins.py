"""
Spin analysis for GW230529 NSBH event.

This script creates KDE plots of spin_1z (dimensionless spin of the primary object)
for GW230529 NSBH event across different population priors.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import corner
from scipy.stats import gaussian_kde
from typing import Dict, List, Optional

from utils import (
    construct_result_path, load_posterior_data, setup_matplotlib_style,
    EOS_COLORS, DEFAULT_CORNER_KWARGS, EOS_SAMPLES_NAMES_DICT
)

# Setup matplotlib style
setup_matplotlib_style()

def load_spin_data_for_populations(populations: List[str], eos_samples: List[str], 
                                   base_path: str, gw_event: str, source_type: str,
                                   spin_params: List[str]) -> Dict:
    """
    Load spin data for all populations and EOS samples.
    
    Args:
        populations (List[str]): Population types to include
        eos_samples (List[str]): EOS samples to include
        base_path (str): Base path to result files
        gw_event (str): GW event name
        source_type (str): Source type (bns/nsbh)
        spin_params (List[str]): Spin parameters to load
        
    Returns:
        Dict: Nested dictionary with population -> dataset -> parameter -> samples
    """
    all_data = {}
    
    for population in populations:
        print(f"\nProcessing population: {population}")
        pop_data = {}
        
        # Load default run first (if available)
        default_path = construct_result_path(base_path, gw_event, source_type, "default", "radio")
        if os.path.exists(default_path):
            default_posterior = load_posterior_data(default_path, fast_mode=True)
            
            if default_posterior is not None:
                default_spin_data = {}
                for param in spin_params:
                    if param in default_posterior:
                        default_spin_data[param] = np.array(default_posterior[param])
                
                if default_spin_data:
                    pop_data['Uninformed'] = {
                        'data': default_spin_data,
                        'color': 'gray',
                        'linewidth': 2,
                        'alpha': 1.0,
                        'linestyle': '--'
                    }
                    sample_count = len(list(default_spin_data.values())[0])
                    print(f"  Default: {sample_count} samples")
        else:
            raise FileNotFoundError(f"Default result file not found: {default_path}")
        
        # Load EOS samples
        for eos_name in eos_samples:
            result_path = construct_result_path(base_path, gw_event, population, source_type, eos_name)
            
            if not os.path.exists(result_path):
                print(f"  Warning: {eos_name} result file not found: {result_path}")
                continue
            
            posterior = load_posterior_data(result_path, fast_mode=True)
            if posterior is None:
                print(f"  Warning: Could not load {eos_name} posterior")
                continue
            
            eos_spin_data = {}
            missing_params = []
            for param in spin_params:
                if param in posterior:
                    eos_spin_data[param] = np.array(posterior[param])
                else:
                    missing_params.append(param)
            
            if missing_params:
                print(f"  Warning: {', '.join(missing_params)} not found in {eos_name} posterior")
                continue
            
            if eos_spin_data:
                color = EOS_COLORS.get(eos_name, 'black')
                display_name = eos_name.replace('_', ' ')
                pop_data[display_name] = {
                    'data': eos_spin_data,
                    'color': color,
                    'linewidth': 2.5,
                    'alpha': 0.9,
                    'linestyle': '-'
                }
                sample_count = len(list(eos_spin_data.values())[0])
                print(f"  {eos_name}: {sample_count} samples")
        
        all_data[population] = pop_data
    
    return all_data

def plot_gw230529_chi1z_kde(populations: List[str] = None,
                           eos_samples: List[str] = None,
                           base_path: str = "../final_results/",
                           output_dir: str = "./figures/spin_analysis/",
                           figsize: tuple = (10, 6)) -> str:
    """
    Plot KDE of spin_1z for GW230529 NSBH across different populations.
    
    Args:
        populations (List[str]): Population types to include
        eos_samples (List[str]): EOS samples to include
        base_path (str): Base path to result files
        output_dir (str): Output directory for figures
        figsize (tuple): Figure size
        
    Returns:
        str: Path to saved figure
    """
    # Formatting constants
    LEGEND_FONTSIZE = 14
    XLABEL_FONTSIZE = 18
    YLABEL_FONTSIZE = 18
    TITLE_FONTSIZE = 20
    TITLE_PAD = 10
    
    if populations is None:
        populations = ["uniform", "gaussian", "double_gaussian"]
    
    if eos_samples is None:
        eos_samples = ["radio", "radio_chiEFT", "radio_NICER"]
    
    gw_event = "GW230529"
    source_type = "nsbh"
    
    print(f"Creating spin_1z KDE plot for {gw_event} {source_type.upper()}")
    print(f"Populations: {populations}")
    print(f"EOS samples: {eos_samples}")
    
    # Load spin data
    all_data = load_spin_data_for_populations(
        populations, eos_samples, base_path, gw_event, source_type, ['spin_1z']
    )
    
    fig, axes = plt.subplots(1, len(populations), figsize=figsize, sharey=True)
    if len(populations) == 1:
        axes = [axes]
    
    # Define common spin_1z range for all plots
    xmin, xmax = -0.41, 0.1
    chi1z_range = np.linspace(xmin, xmax, 200)
    
    # First pass: calculate all KDE values to determine maximum height
    all_kde_data = []
    
    for pop_idx, population in enumerate(populations):
        pop_kde_data = []
        pop_data = all_data[population]
        
        # Debug: check what pop_data actually is
        if not isinstance(pop_data, dict):
            print(f"ERROR: pop_data for {population} is not a dict, it's {type(pop_data)}: {pop_data}")
            continue
        
        for dataset_name, dataset_info in pop_data.items():
            chi1z_samples = dataset_info['data']['spin_1z']
            
            # Create KDE with adjusted bandwidth
            kde = gaussian_kde(chi1z_samples)
            kde.set_bandwidth(kde.factor * 1.0)
            kde_values = kde(chi1z_range)
            
            # Get proper display name for legend
            if dataset_name == 'Uninformed':
                display_label = "Uninformed"
            else:
                # Map display name back to EOS name for proper lookup
                eos_name = dataset_name.replace(' ', '_')
                display_label = EOS_SAMPLES_NAMES_DICT.get(eos_name, dataset_name)
            
            pop_kde_data.append({
                'values': kde_values,
                'color': dataset_info['color'],
                'linewidth': dataset_info['linewidth'],
                'alpha': dataset_info['alpha'],
                'label': display_label,
                'linestyle': dataset_info['linestyle']
            })
        
        all_kde_data.append(pop_kde_data)
    
    # Calculate maximum KDE height across all plots
    max_kde_height = 0
    for pop_data in all_kde_data:
        for kde_info in pop_data:
            max_kde_height = max(max_kde_height, np.max(kde_info['values']))
    
    # Add some padding to the maximum height
    y_max = max_kde_height * 1.01
    
    # Second pass: create the plots
    for pop_idx, population in enumerate(populations):
        ax = axes[pop_idx]
        pop_kde_data = all_kde_data[pop_idx]
        
        # Plot all KDE curves for this population
        for kde_info in pop_kde_data:
            ax.plot(chi1z_range, kde_info['values'],
                   color=kde_info['color'], 
                   linewidth=kde_info['linewidth'],
                   alpha=kde_info['alpha'],
                   label=kde_info['label'],
                   linestyle=kde_info['linestyle'])
        
        # Format subplot
        ax.set_xlabel(r'$\chi_{1z}$', fontsize=XLABEL_FONTSIZE)
        if pop_idx == 0:
            ax.set_ylabel('Probability Density', fontsize=YLABEL_FONTSIZE)
        
        ax.set_title(population.replace('_', ' ').title(), fontsize=TITLE_FONTSIZE, pad=TITLE_PAD)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(0, y_max)
        
        # Add vertical line at zero
        ax.axvline(0, color='black', linestyle=':', alpha=0.5, linewidth=1)
        
        # Legend only on first subplot
        if pop_idx == 0:
            ax.legend(loc='upper left', fontsize=LEGEND_FONTSIZE)
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    output_filename = f"{gw_event}_{source_type}_chi1z_kde.pdf"
    output_path = os.path.join(output_dir, output_filename)
    
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"\nSaved figure to: {output_path}")
    plt.close()
    
    return output_path

def plot_chi1z_statistics_comparison(populations: List[str] = None,
                                   eos_samples: List[str] = None,
                                   base_path: str = "../final_results/",
                                   output_dir: str = "./figures/spin_analysis/") -> None:
    """
    Print statistical comparison of spin_1z distributions.
    
    Args:
        populations (List[str]): Population types to include
        eos_samples (List[str]): EOS samples to include
        base_path (str): Base path to result files
        output_dir (str): Output directory for figures
    """
    if populations is None:
        populations = ["uniform", "gaussian", "double_gaussian"]
    
    if eos_samples is None:
        eos_samples = ["radio", "radio_chiEFT", "radio_NICER"]
    
    gw_event = "GW230529"
    source_type = "nsbh"
    
    print(f"\n=== spin_1z Statistics for {gw_event} {source_type.upper()} ===")
    
    for population in populations:
        print(f"\n{population.replace('_', ' ').title()} Population:")
        print(f"{'Dataset':<15} {'Mean':<8} {'Std':<8} {'Median':<8} {'IQR':<12} {'Samples':<8}")
        print("-" * 70)
        
        # Default run
        default_path = construct_result_path(base_path, gw_event, source_type, "default", "radio")
        if os.path.exists(default_path):
            default_posterior = load_posterior_data(default_path, fast_mode=True)
            if default_posterior is not None and 'spin_1z' in default_posterior:
                chi1z = np.array(default_posterior['spin_1z'])
                mean_val = np.mean(chi1z)
                std_val = np.std(chi1z)
                median_val = np.median(chi1z)
                q25, q75 = np.percentile(chi1z, [25, 75])
                iqr = q75 - q25
                print(f"{'Uninformed':<15} {mean_val:<8.3f} {std_val:<8.3f} {median_val:<8.3f} {iqr:<12.3f} {len(chi1z):<8d}")
        
        # EOS samples
        for eos_name in eos_samples:
            result_path = construct_result_path(base_path, gw_event, population, source_type, eos_name)
            
            if not os.path.exists(result_path):
                continue
            
            posterior = load_posterior_data(result_path, fast_mode=True)
            if posterior is None or 'spin_1z' not in posterior:
                continue
            
            chi1z = np.array(posterior['spin_1z'])
            mean_val = np.mean(chi1z)
            std_val = np.std(chi1z)
            median_val = np.median(chi1z)
            q25, q75 = np.percentile(chi1z, [25, 75])
            iqr = q75 - q25
            
            display_name = eos_name.replace('_', ' ')
            print(f"{display_name:<15} {mean_val:<8.3f} {std_val:<8.3f} {median_val:<8.3f} {iqr:<12.3f} {len(chi1z):<8d}")

def plot_gw230529_chi_2d(populations: List[str] = None,
                        eos_samples: List[str] = None,
                        base_path: str = "../final_results/",
                        output_dir: str = "./figures/spin_analysis/") -> List[str]:
    """
    Create corner plots of spin_1z vs spin_2z for GW230529 NSBH, one per population.
    
    Args:
        populations (List[str]): Population types to include
        eos_samples (List[str]): EOS samples to include
        base_path (str): Base path to result files
        output_dir (str): Output directory for figures
        
    Returns:
        List[str]: Paths to saved figures
    """
    if populations is None:
        populations = ["uniform", "gaussian", "double_gaussian"]
    
    if eos_samples is None:
        eos_samples = ["radio", "radio_chiEFT", "radio_NICER"]
    
    gw_event = "GW230529"
    source_type = "nsbh"
    
    print(f"Creating 2D spin corner plots for {gw_event} {source_type.upper()}")
    print(f"Populations: {populations}")
    print(f"EOS samples: {eos_samples}")
    
    # Load spin data for both spin components
    all_data = load_spin_data_for_populations(
        populations, eos_samples, base_path, gw_event, source_type, ['spin_1z', 'spin_2z']
    )
    
    output_paths = []
    
    # Create separate corner plot for each population
    for population in populations:
        print(f"\nCreating corner plot for population: {population}")
        pop_data = all_data[population]
        
        # Prepare data for corner plot
        samples_dict = {}
        default_samples = None
        
        for dataset_name, dataset_info in pop_data.items():
            spin_data = dataset_info['data']
            
            # Check if both spin components are available
            if 'spin_1z' not in spin_data or 'spin_2z' not in spin_data:
                print(f"  Warning: Missing spin data for {dataset_name} in {population}")
                continue
            
            # Stack spin components into 2D array
            samples = np.column_stack([
                spin_data['spin_1z'],
                spin_data['spin_2z']
            ])
            
            if dataset_name == 'Uninformed':
                default_samples = samples
            else:
                # Map display name back to EOS name for color lookup
                eos_name = dataset_name.replace(' ', '_')
                samples_dict[eos_name] = samples
        
        if not samples_dict:
            print(f"  Warning: No valid spin data for population {population}")
            continue
        
        # Define parameter labels and ranges
        params = ['spin_1z', 'spin_2z']
        latex_labels = [r'$\chi_{1z}$', r'$\chi_{2z}$']
        ranges = [(-0.5, 0.1), (-0.05, 0.05)]
        
        # Create corner plot
        fig = None
        
        # Plot default samples first (if available) as background
        if default_samples is not None:
            print(f"  Plotting uninformed prior samples (background)")
            
            default_kwargs = DEFAULT_CORNER_KWARGS.copy()
            default_kwargs.update({
                'color': 'lightgray',
                'labels': latex_labels,
                'range': ranges,
                'hist_kwargs': {'color': 'gray', 'density': True, 'alpha': 0.7},
            })
            
            fig = corner.corner(default_samples, **default_kwargs)
        
        # Plot EOS samples on top
        for eos_name, samples in samples_dict.items():
            color = EOS_COLORS.get(eos_name, 'black')
            display_name = EOS_SAMPLES_NAMES_DICT.get(eos_name, eos_name.replace('_', ' '))
            
            print(f"  Plotting {display_name} samples")
            
            kwargs = DEFAULT_CORNER_KWARGS.copy()
            kwargs.update({
                'color': color,
                'labels': latex_labels,
                'range': ranges,
                'hist_kwargs': {'color': color, 'density': True, 'alpha': 0.8},
            })
            
            if fig is None:
                fig = corner.corner(samples, **kwargs)
            else:
                corner.corner(samples, fig=fig, **kwargs)
        
        # Add legend
        # Create custom legend entries
        legend_entries = []
        legend_colors = []
        
        for eos_name in samples_dict.keys():
            base_label = EOS_SAMPLES_NAMES_DICT.get(eos_name, eos_name)
            legend_entries.append(base_label)
            legend_colors.append(EOS_COLORS.get(eos_name, 'black'))
        
        # Add default run to legend if included
        if default_samples is not None:
            legend_entries.append("Uninformed")
            legend_colors.append('gray')
        
        # Add legend with configurable positioning
        legend_x = 0.65
        legend_y = 0.95
        legend_dy = 0.08
        legend_fontsize = 14
        
        for i, (label, color) in enumerate(zip(legend_entries, legend_colors)):
            plt.text(legend_x, legend_y - i * legend_dy, label, 
                    fontsize=legend_fontsize, color=color, ha='left', va='top',
                    transform=fig.transFigure, weight='bold')
        
        # Add title
        title = f"{population.replace('_', ' ').title()} Population"
        plt.suptitle(title, fontsize=16, weight='bold', y=0.98)
        
        # Save figure
        os.makedirs(output_dir, exist_ok=True)
        output_filename = f"{gw_event}_{source_type}_chi_2d_{population}.pdf"
        output_path = os.path.join(output_dir, output_filename)
        
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        print(f"  Saved corner plot to: {output_path}")
        plt.close()
        
        output_paths.append(output_path)
    
    return output_paths

def main():
    """Main function to create spin analysis plots."""
    print("Creating GW230529 NSBH spin_1z KDE plots...")
    output_path_1d = plot_gw230529_chi1z_kde()
    print(f"Created 1D KDE plot: {output_path_1d}")
    
    print("\nCreating GW230529 NSBH 2D spin corner plots...")
    output_paths_2d = plot_gw230529_chi_2d()
    print(f"Created 2D corner plots: {output_paths_2d}")
    
    print("\nGenerating statistical comparison...")
    plot_chi1z_statistics_comparison()

if __name__ == "__main__":
    main()