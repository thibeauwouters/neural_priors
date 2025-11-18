"""
Publication-ready corner plots for gravitational wave parameter estimation results.

This script provides hardcoded configurations for creating publication-quality
corner plots for specific GW events and parameter combinations. Unlike the batch
approach in cornerplots.py, this focuses on creating carefully configured plots
for paper figures.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import corner
from typing import List, Dict, Optional

from utils import (
    construct_result_path, load_posterior_data, setup_matplotlib_style,
    PARAMETER_LATEX_LABELS, DEFAULT_CORNER_KWARGS, EOS_COLORS, EOS_SAMPLES_NAMES_DICT,
    convert_lambdas_with_verbose, filter_constant_parameters,
    calculate_corner_plot_ranges, handle_nsbh_lambda_plotting
)

# DEBUG plot specific parameter labels
DEBUG_PARAMETER_LATEX_LABELS = {
    'chirp_mass_source': r'$\mathcal{M}_c^{\rm{src}}$ [$M_{\odot}$]',
    'luminosity_distance': r'$d_L$ [Mpc]',
}

# Font size
fs_ticks = 24
fs_labels = 28
labelpad = 0.175

# Matplotlib style parameters
params = {"axes.grid": False,
        "text.usetex" : True,
        "font.family" : "serif",
        "ytick.color" : "black",
        "xtick.color" : "black",
        "axes.labelcolor" : "black",
        "axes.edgecolor" : "black",
        "font.serif" : ["Computer Modern Serif"],
        "xtick.labelsize": fs_ticks,
        "ytick.labelsize": fs_ticks,
        "axes.labelsize": fs_labels,
        }

plt.rcParams.update(params)

# Update default corner kwargs with new tick settings and increased fontsize/labelpad
DEFAULT_CORNER_KWARGS = DEFAULT_CORNER_KWARGS.copy()
DEFAULT_CORNER_KWARGS.update({
    'max_n_ticks': 3,
    'min_n_ticks': 2,
    'label_kwargs': dict(fontsize=fs_labels),
    'tick_kwargs': dict(labelsize=fs_ticks),
    'labelpad': labelpad,
})

# Presentation version of corner kwargs with larger labels and ticks
PRESENTATION_CORNER_KWARGS = DEFAULT_CORNER_KWARGS.copy()
PRESENTATION_CORNER_KWARGS.update({
    'max_n_ticks': 3,
    'min_n_ticks': 2,
})

# Default run color constants
DEFAULT_RUN_PLOT_COLOR = 'lightgray'
DEFAULT_RUN_LEGEND_COLOR = 'dimgray'

# Legend formatting constants
LEGEND_FONTSIZE = 36
LEGEND_X = 0.65
LEGEND_Y = 0.95
LEGEND_DY = 0.06  # Half of original 0.08

# Title formatting constants
TITLE_FONTSIZE = 34
TITLE_LABELPAD = 50
ADD_TITLE = False


def load_bayes_factors(bayes_factors_path: str = "../bayes_factors/all_bayes_factors.json") -> Dict:
    """
    Load Bayes factors from JSON file.
    
    Args:
        bayes_factors_path (str): Path to the Bayes factors JSON file
        
    Returns:
        Dict: Bayes factors data structure
    """
    with open(bayes_factors_path, 'r') as f:
        return json.load(f)


def get_bayes_factors_for_event(bayes_factors: Dict, 
                               gw_event: str, 
                               population: str, 
                               source_type: str, 
                               eos_samples: List[str]) -> Dict[str, float]:
    """
    Extract Bayes factors for a specific event configuration.
    
    Args:
        bayes_factors (Dict): Loaded Bayes factors data
        gw_event (str): GW event name
        population (str): Population type
        source_type (str): Source type
        eos_samples (List[str]): List of EOS sample names
        
    Returns:
        Dict[str, float]: Dictionary mapping EOS name to Bayes factor
    """
    bf_dict = {}
    
    if source_type not in bayes_factors:
        print(f"Warning: Source type '{source_type}' not found in Bayes factors")
        return bf_dict
        
    if population not in bayes_factors[source_type]:
        print(f"Warning: Population '{population}' not found for source type '{source_type}'")
        return bf_dict
    
    for eos_name in eos_samples:
        if eos_name in bayes_factors[source_type][population]:
            if gw_event in bayes_factors[source_type][population][eos_name]:
                bf_dict[eos_name] = bayes_factors[source_type][population][eos_name][gw_event]
            else:
                print(f"Warning: Event '{gw_event}' not found for {eos_name}")
        else:
            print(f"Warning: EOS '{eos_name}' not found for {population} {source_type}")
    
    return bf_dict


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
                                            legend_x: float = LEGEND_X,
                                            legend_y: float = LEGEND_Y,
                                            legend_dy: float = LEGEND_DY,
                                            legend_fontsize: int = LEGEND_FONTSIZE,
                                            title_fontsize: int = TITLE_FONTSIZE,
                                            bayes_factors_path: str = "../bayes_factors/all_bayes_factors.json",
                                            include_default: bool = True,
                                            remove_chieff: bool = False,
                                            remove_spin1z: bool = False,
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
        title_fontsize (int): Font size for plot title
        bayes_factors_path (str): Path to Bayes factors JSON file
        include_default (bool): Include default uniform prior run (plotted in light gray)
        remove_chieff (bool): Remove chi_eff parameter from plotting if True
        remove_spin1z (bool): Remove spin_1z parameter from plotting if True
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
    
    # Remove chi_eff if requested
    if remove_chieff and "chi_eff" in params_to_plot:
        params_to_plot = [p for p in params_to_plot if p != "chi_eff"]
        if verbose:
            print("Removed chi_eff from parameters to plot")
    
    # Remove spin_1z if requested
    if remove_spin1z and "spin_1z" in params_to_plot:
        params_to_plot = [p for p in params_to_plot if p != "spin_1z"]
        if verbose:
            print("Removed spin_1z from parameters to plot")
    
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
    
    # Load default run if requested
    default_posterior = None
    if include_default:
        # For default runs, pass the actual source type as population_type and "default" as source_type
        default_path = construct_result_path(base_path, gw_event, source_type, 
                                           "default", "radio")
        print(f"Loading default run from: {default_path}")
        
        if os.path.exists(default_path):
            default_posterior = load_posterior_data(default_path, fast_mode=True)
            if default_posterior is not None:
                if verbose:
                    print(f"Loaded default run with {len(default_posterior['chirp_mass'])} samples")
            else:
                print("Warning: Could not load default run data")
        else:
            print(f"Warning: Default run file not found: {default_path}")
    
    # Convert lambdas to tilde parameters if needed
    if any('lambda_tilde' in p or 'delta_lambda_tilde' in p for p in params_to_plot):
        posteriors_dict, params_to_plot = convert_lambdas_with_verbose(
            posteriors_dict, params_to_plot, verbose=verbose
        )
        
        # Also convert default posterior if it exists
        if default_posterior is not None:
            default_dict = {"default": default_posterior}
            default_dict, _ = convert_lambdas_with_verbose(
                default_dict, params_to_plot, verbose=verbose
            )
            default_posterior = default_dict["default"]
    
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
    
    # Add default samples if available
    default_samples = None
    if default_posterior is not None:
        samples_list = []
        for param in params_to_plot:
            if param in default_posterior:
                samples_list.append(np.array(default_posterior[param]))
            else:
                print(f"Warning: Parameter '{param}' not found in default posterior")
        
        if samples_list:
            default_samples = np.column_stack(samples_list)
            if verbose:
                print(f"Created default samples array with shape: {default_samples.shape}")
    
    # Compute percentage of samples with negative delta lambda tilde for BNS
    if source_type.lower() == 'bns' and 'delta_lambda_tilde' in params_to_plot:
        delta_lambda_idx = params_to_plot.index('delta_lambda_tilde')
        
        print(f"\n=== Negative Delta Lambda Tilde Statistics ===")
        for eos_name, samples in samples_dict.items():
            delta_lambda_samples = samples[:, delta_lambda_idx]
            negative_count = np.sum(delta_lambda_samples < 0)
            total_count = len(delta_lambda_samples)
            percentage = (negative_count / total_count) * 100
            print(f"{eos_name}: {negative_count}/{total_count} ({percentage:.1f}%) negative delta_lambda_tilde")
        
        # Also compute for default samples
        if default_samples is not None:
            delta_lambda_samples = default_samples[:, delta_lambda_idx]
            negative_count = np.sum(delta_lambda_samples < 0)
            total_count = len(delta_lambda_samples)
            percentage = (negative_count / total_count) * 100
            print(f"default: {negative_count}/{total_count} ({percentage:.1f}%) negative delta_lambda_tilde")
    
    # Handle NSBH lambda plotting if needed
    if source_type.lower() == 'nsbh':
        samples_dict = handle_nsbh_lambda_plotting(samples_dict, params_to_plot, params_to_plot)
        
        # Also apply to default samples
        if default_samples is not None:
            default_dict_temp = {"default": default_samples}
            default_dict_temp = handle_nsbh_lambda_plotting(default_dict_temp, params_to_plot, params_to_plot)
            default_samples = default_dict_temp["default"]
    
    # Load Bayes factors and calculate zorder
    try:
        bayes_factors = load_bayes_factors(bayes_factors_path)
        bf_dict = get_bayes_factors_for_event(bayes_factors, gw_event, population, source_type, eos_samples)
        
        if bf_dict:
            print(f"\n=== Bayes Factors for {gw_event} ({population} {source_type}) ===")
            sorted_bf = sorted(bf_dict.items(), key=lambda x: x[1], reverse=True)
            for eos_name, bf_value in sorted_bf:
                print(f"{eos_name}: ln(BF) = {bf_value:.2f}")
            
            # Find highest evidence EOS
            highest_evidence_eos = sorted_bf[0][0] if sorted_bf else None
            print(f"Highest evidence: {highest_evidence_eos} *")
            
            # Create zorder mapping (higher BF = higher zorder, so it's plotted on top)
            zorder_dict = {}
            for i, (eos_name, _) in enumerate(sorted_bf):
                zorder_dict[eos_name] = len(sorted_bf) - i  # Reverse order so highest BF gets highest zorder
        else:
            print(f"Warning: No Bayes factors found for {gw_event} {population} {source_type}")
            zorder_dict = {}
            highest_evidence_eos = None
    except Exception as e:
        print(f"Warning: Could not load Bayes factors: {e}")
        zorder_dict = {}
        highest_evidence_eos = None
    
    # Validate and create normalization dummy dataset if requested
    dummy_dataset = None
    if normalization_keys is not None:
        # Validate normalization_keys
        n_params = len(params_to_plot)
        if len(normalization_keys) != n_params:
            raise ValueError(f"normalization_keys must have length {n_params} (number of parameters), got {len(normalization_keys)}")
        
        # Check that all keys exist in loaded datasets or are "default"
        loaded_keys = set(samples_dict.keys())
        available_keys = loaded_keys.copy()
        if default_samples is not None:
            available_keys.add("default")
        
        for key in normalization_keys:
            if key not in available_keys:
                raise ValueError(f"normalization_keys contains '{key}' which was not loaded. Available keys: {available_keys}")
        
        # Create extended samples dict that includes default if available
        extended_samples_dict = samples_dict.copy()
        if default_samples is not None:
            extended_samples_dict["default"] = default_samples
        
        # Find minimum number of samples across all datasets
        min_samples = min(len(dataset) for dataset in extended_samples_dict.values())
        if verbose:
            sample_counts = {key: len(dataset) for key, dataset in extended_samples_dict.items()}
            print(f"Sample counts by dataset: {sample_counts}")
            print(f"Using minimum sample count for normalization: {min_samples}")
        
        # Create dummy dataset by selecting specified dataset for each parameter
        # All datasets are downsampled to min_samples to ensure consistent sizes
        dummy_columns = []
        for param_idx, dataset_key in enumerate(normalization_keys):
            dataset = extended_samples_dict[dataset_key]
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
    
    # Plot default samples FIRST (if available) to ensure they appear behind everything
    if default_samples is not None:
        if verbose:
            print("Plotting default run samples in light gray (first layer)")
        
        default_kwargs = DEFAULT_CORNER_KWARGS.copy()
        default_kwargs.update({
            'color': DEFAULT_RUN_PLOT_COLOR,  # Light color for 2D contours
            'labels': latex_labels,
            'range': ranges,
            'levels': [0.68, 0.95],
            'hist_kwargs': {'color': DEFAULT_RUN_LEGEND_COLOR, 'density': True},  # Dark color for 1D histograms
        })
        
        fig = corner.corner(default_samples, **default_kwargs)
    
    # Sort samples_dict by zorder (lowest first so highest evidence is plotted last/on top)
    if zorder_dict:
        sorted_samples = sorted(samples_dict.items(), 
                              key=lambda x: zorder_dict.get(x[0], 0))
    else:
        sorted_samples = list(samples_dict.items())
    
    for i, (eos_name, samples) in enumerate(sorted_samples):
        color = EOS_COLORS.get(eos_name, f"C{i}")  # Fallback to default colors if needed
        group_colors[eos_name] = color
        
        # Get zorder for this EOS (higher BF = higher zorder = plotted on top)
        zorder_value = zorder_dict.get(eos_name, 1)
        
        # Corner plot kwargs
        kwargs = DEFAULT_CORNER_KWARGS.copy()
        kwargs.update({
            'color': color,
            'labels': latex_labels,
            'range': ranges,
            'levels': [0.68, 0.95],
            'hist_kwargs': {'color': color, 'density': True, 'zorder': zorder_value},
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

    # Disable offset notation on all axes to prevent tick label overlap
    # This forces matplotlib to show full tick values instead of using offset format
    for ax in fig.get_axes():
        try:
            ax.ticklabel_format(useOffset=False, style='plain')
        except AttributeError:
            # Some axes use NullFormatter which doesn't support ticklabel_format
            pass

    # Add legend
    # Create custom legend entries
    legend_entries = []
    legend_colors = []
    
    for eos_name in samples_dict.keys():
        base_label = EOS_SAMPLES_NAMES_DICT.get(eos_name, eos_name)
        # Add superscript asterisk to highest evidence EOS
        if highest_evidence_eos and eos_name == highest_evidence_eos:
            base_label += r"$^{*}$"
        legend_entries.append(base_label)
        legend_colors.append(EOS_COLORS.get(eos_name, 'black'))
    
    # Add default run to legend if included
    if default_samples is not None:
        legend_entries.append("Uninformed")
        legend_colors.append(DEFAULT_RUN_LEGEND_COLOR)
    
    # Add title
    if ADD_TITLE:
        title = f"{gw_event} - {population.replace('_', ' ').title()} population"
        plt.suptitle(title, fontsize=title_fontsize, weight='bold', y=0.98 + TITLE_LABELPAD/1000)
    
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

    # Copy to paper directory for specific plots
    copy_to_paper = False
    if gw_event == "GW170817" and source_type == "bns" and population == "gaussian":
        copy_to_paper = True
    elif gw_event == "GW190425" and source_type == "bns" and population == "uniform":
        copy_to_paper = True
    elif gw_event == "GW230529" and source_type == "nsbh" and population == "gaussian":
        copy_to_paper = True

    if copy_to_paper and os.path.exists("../../paper/Figures"):
        print(f"Also saving to the Overleaf paper repo")
        save_name_paper = output_path.replace("./figures/money_plots/", "../../paper/Figures/")
        plt.savefig(save_name_paper, bbox_inches='tight')

    plt.close()

    return output_path


def plot_corner_fixed_population_varying_eos_PRESENTATION(gw_event: str,
                                            population: str = "gaussian",
                                            source_type: str = "bns",
                                            base_path: str = "../final_results/",
                                            output_dir: str = "./figures/money_plots/",
                                            eos_samples: List[str] = None,
                                            params_to_plot: List[str] = None,
                                            ranges: List[tuple] = None,
                                            normalization_keys: List[str] = None,
                                            legend_x: float = LEGEND_X,
                                            legend_y: float = LEGEND_Y,
                                            legend_dy: float = LEGEND_DY,
                                            legend_fontsize: int = LEGEND_FONTSIZE,
                                            title_fontsize: int = TITLE_FONTSIZE,
                                            bayes_factors_path: str = "../bayes_factors/all_bayes_factors.json",
                                            include_default: bool = True,
                                            remove_chieff: bool = False,
                                            remove_spin1z: bool = False,
                                            verbose: bool = False) -> str:
    """
    Create corner plot for a GW event with fixed population type and varying EOS samples - PRESENTATION VERSION.

    This is identical to plot_corner_fixed_population_varying_eos but uses larger labels and labelpad=0.02
    for presentation purposes. Output filename will have "_PRESENTATION" suffix.

    Args: Same as plot_corner_fixed_population_varying_eos
    Returns: str: Path to saved figure
    """

    # Set default parameters
    if eos_samples is None:
        eos_samples = ["radio", "radio_chiEFT", "radio_NICER"]

    if params_to_plot is None:
        params_to_plot = ["chirp_mass", "mass_ratio",
                         "lambda_tilde", "delta_lambda_tilde"]

    # Remove chi_eff if requested
    if remove_chieff and "chi_eff" in params_to_plot:
        params_to_plot = [p for p in params_to_plot if p != "chi_eff"]
        if verbose:
            print("Removed chi_eff from parameters to plot")

    # Remove spin_1z if requested
    if remove_spin1z and "spin_1z" in params_to_plot:
        params_to_plot = [p for p in params_to_plot if p != "spin_1z"]
        if verbose:
            print("Removed spin_1z from parameters to plot")

    print(f"Creating {gw_event} PRESENTATION corner plot for population: {population}, source: {source_type}")
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

    # Load default run if requested
    default_posterior = None
    if include_default:
        # For default runs, pass the actual source type as population_type and "default" as source_type
        default_path = construct_result_path(base_path, gw_event, source_type,
                                           "default", "radio")
        print(f"Loading default run from: {default_path}")

        if os.path.exists(default_path):
            default_posterior = load_posterior_data(default_path, fast_mode=True)
            if default_posterior is not None:
                if verbose:
                    print(f"Loaded default run with {len(default_posterior['chirp_mass'])} samples")
            else:
                print("Warning: Could not load default run data")
        else:
            print(f"Warning: Default run file not found: {default_path}")

    # Convert lambdas to tilde parameters if needed
    if any('lambda_tilde' in p or 'delta_lambda_tilde' in p for p in params_to_plot):
        posteriors_dict, params_to_plot = convert_lambdas_with_verbose(
            posteriors_dict, params_to_plot, verbose=verbose
        )

        # Also convert default posterior if it exists
        if default_posterior is not None:
            default_dict = {"default": default_posterior}
            default_dict, _ = convert_lambdas_with_verbose(
                default_dict, params_to_plot, verbose=verbose
            )
            default_posterior = default_dict["default"]

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

    # Add default samples if available
    default_samples = None
    if default_posterior is not None:
        samples_list = []
        for param in params_to_plot:
            if param in default_posterior:
                samples_list.append(np.array(default_posterior[param]))
            else:
                print(f"Warning: Parameter '{param}' not found in default posterior")

        if samples_list:
            default_samples = np.column_stack(samples_list)
            if verbose:
                print(f"Created default samples array with shape: {default_samples.shape}")

    # Compute percentage of samples with negative delta lambda tilde for BNS
    if source_type.lower() == 'bns' and 'delta_lambda_tilde' in params_to_plot:
        delta_lambda_idx = params_to_plot.index('delta_lambda_tilde')

        print(f"\n=== Negative Delta Lambda Tilde Statistics ===")
        for eos_name, samples in samples_dict.items():
            delta_lambda_samples = samples[:, delta_lambda_idx]
            negative_count = np.sum(delta_lambda_samples < 0)
            total_count = len(delta_lambda_samples)
            percentage = (negative_count / total_count) * 100
            print(f"{eos_name}: {negative_count}/{total_count} ({percentage:.1f}%) negative delta_lambda_tilde")

        # Also compute for default samples
        if default_samples is not None:
            delta_lambda_samples = default_samples[:, delta_lambda_idx]
            negative_count = np.sum(delta_lambda_samples < 0)
            total_count = len(delta_lambda_samples)
            percentage = (negative_count / total_count) * 100
            print(f"default: {negative_count}/{total_count} ({percentage:.1f}%) negative delta_lambda_tilde")

    # Handle NSBH lambda plotting if needed
    if source_type.lower() == 'nsbh':
        samples_dict = handle_nsbh_lambda_plotting(samples_dict, params_to_plot, params_to_plot)

        # Also apply to default samples
        if default_samples is not None:
            default_dict_temp = {"default": default_samples}
            default_dict_temp = handle_nsbh_lambda_plotting(default_dict_temp, params_to_plot, params_to_plot)
            default_samples = default_dict_temp["default"]

    # Load Bayes factors and calculate zorder
    try:
        bayes_factors = load_bayes_factors(bayes_factors_path)
        bf_dict = get_bayes_factors_for_event(bayes_factors, gw_event, population, source_type, eos_samples)

        if bf_dict:
            print(f"\n=== Bayes Factors for {gw_event} ({population} {source_type}) ===")
            sorted_bf = sorted(bf_dict.items(), key=lambda x: x[1], reverse=True)
            for eos_name, bf_value in sorted_bf:
                print(f"{eos_name}: ln(BF) = {bf_value:.2f}")

            # Find highest evidence EOS
            highest_evidence_eos = sorted_bf[0][0] if sorted_bf else None
            print(f"Highest evidence: {highest_evidence_eos} *")

            # Create zorder mapping (higher BF = higher zorder, so it's plotted on top)
            zorder_dict = {}
            for i, (eos_name, _) in enumerate(sorted_bf):
                zorder_dict[eos_name] = len(sorted_bf) - i  # Reverse order so highest BF gets highest zorder
        else:
            print(f"Warning: No Bayes factors found for {gw_event} {population} {source_type}")
            zorder_dict = {}
            highest_evidence_eos = None
    except Exception as e:
        print(f"Warning: Could not load Bayes factors: {e}")
        zorder_dict = {}
        highest_evidence_eos = None

    # Validate and create normalization dummy dataset if requested
    dummy_dataset = None
    if normalization_keys is not None:
        # Validate normalization_keys
        n_params = len(params_to_plot)
        if len(normalization_keys) != n_params:
            raise ValueError(f"normalization_keys must have length {n_params} (number of parameters), got {len(normalization_keys)}")

        # Check that all keys exist in loaded datasets or are "default"
        loaded_keys = set(samples_dict.keys())
        available_keys = loaded_keys.copy()
        if default_samples is not None:
            available_keys.add("default")

        for key in normalization_keys:
            if key not in available_keys:
                raise ValueError(f"normalization_keys contains '{key}' which was not loaded. Available keys: {available_keys}")

        # Create extended samples dict that includes default if available
        extended_samples_dict = samples_dict.copy()
        if default_samples is not None:
            extended_samples_dict["default"] = default_samples

        # Find minimum number of samples across all datasets
        min_samples = min(len(dataset) for dataset in extended_samples_dict.values())
        if verbose:
            sample_counts = {key: len(dataset) for key, dataset in extended_samples_dict.items()}
            print(f"Sample counts by dataset: {sample_counts}")
            print(f"Using minimum sample count for normalization: {min_samples}")

        # Create dummy dataset by selecting specified dataset for each parameter
        # All datasets are downsampled to min_samples to ensure consistent sizes
        dummy_columns = []
        for param_idx, dataset_key in enumerate(normalization_keys):
            dataset = extended_samples_dict[dataset_key]
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

    # Plot default samples FIRST (if available) to ensure they appear behind everything
    if default_samples is not None:
        if verbose:
            print("Plotting default run samples in light gray (first layer)")

        default_kwargs = PRESENTATION_CORNER_KWARGS.copy()  # Use presentation version
        default_kwargs.update({
            'color': DEFAULT_RUN_PLOT_COLOR,  # Light color for 2D contours
            'labels': latex_labels,
            'range': ranges,
            'levels': [0.68, 0.95],
            'hist_kwargs': {'color': DEFAULT_RUN_LEGEND_COLOR, 'density': True},  # Dark color for 1D histograms
        })

        fig = corner.corner(default_samples, **default_kwargs)

    # Sort samples_dict by zorder (lowest first so highest evidence is plotted last/on top)
    if zorder_dict:
        sorted_samples = sorted(samples_dict.items(),
                              key=lambda x: zorder_dict.get(x[0], 0))
    else:
        sorted_samples = list(samples_dict.items())

    for i, (eos_name, samples) in enumerate(sorted_samples):
        color = EOS_COLORS.get(eos_name, f"C{i}")  # Fallback to default colors if needed
        group_colors[eos_name] = color

        # Get zorder for this EOS (higher BF = higher zorder = plotted on top)
        zorder_value = zorder_dict.get(eos_name, 1)

        # Corner plot kwargs - use presentation version
        kwargs = PRESENTATION_CORNER_KWARGS.copy()
        kwargs.update({
            'color': color,
            'labels': latex_labels,
            'range': ranges,
            'levels': [0.68, 0.95],
            'hist_kwargs': {'color': color, 'density': True, 'zorder': zorder_value},
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

        # Create corner kwargs that only plots 1D histograms - use presentation version
        invisible_kwargs = PRESENTATION_CORNER_KWARGS.copy()
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

    # Disable offset notation on all axes to prevent tick label overlap
    # This forces matplotlib to show full tick values instead of using offset format
    for ax in fig.get_axes():
        try:
            ax.ticklabel_format(useOffset=False, style='plain')
        except AttributeError:
            # Some axes use NullFormatter which doesn't support ticklabel_format
            pass

    # Add legend
    # Create custom legend entries
    legend_entries = []
    legend_colors = []

    for eos_name in samples_dict.keys():
        base_label = EOS_SAMPLES_NAMES_DICT.get(eos_name, eos_name)
        # Add superscript asterisk to highest evidence EOS
        if highest_evidence_eos and eos_name == highest_evidence_eos:
            base_label += r"$^{*}$"
        legend_entries.append(base_label)
        legend_colors.append(EOS_COLORS.get(eos_name, 'black'))

    # Add default run to legend if included
    if default_samples is not None:
        legend_entries.append("Uninformed")
        legend_colors.append(DEFAULT_RUN_LEGEND_COLOR)

    # Add title
    if ADD_TITLE:
        title = f"{gw_event} - {population.replace('_', ' ').title()} population"
        plt.suptitle(title, fontsize=title_fontsize, weight='bold', y=0.98 + TITLE_LABELPAD/1000)

    # Add legend with configurable positioning
    for i, (label, color) in enumerate(zip(legend_entries, legend_colors)):
        plt.text(legend_x, legend_y - i * legend_dy, label,
                fontsize=legend_fontsize, color=color, ha='left', va='top',
                transform=fig.transFigure, weight='bold')

    # Save figure with PRESENTATION suffix
    output_filename = f"{gw_event}_corner_{population}_{source_type}_PRESENTATION.pdf"
    output_path = os.path.join(output_dir, output_filename)
    ensure_output_directory(output_path)

    plt.savefig(output_path, bbox_inches='tight')
    print(f"Saved PRESENTATION figure to: {output_path}")

    # Note: PRESENTATION figures are not copied to the paper directory
    plt.close()

    return output_path


def plot_gw170817_corner(remove_chieff: bool = False) -> str:
    """
    Convenience function for creating GW170817 corner plot.
    
    Args:
        remove_chieff (bool): Remove chi_eff parameter from plotting if True
    """
    # Set hardcoded defaults for GW170817
    gw170817_defaults = {
        'normalization_keys': ["radio_chiEFT",
                               "radio_NICER",
                               "radio_chiEFT",
                               "radio_chiEFT",
                               "radio_chiEFT"
                               ],
        'ranges': [[1.19735, 1.19775],
                   [0.80, 1.0],
                   [50, 1050],
                   [0.0, 75.0],
                   [20.0, 50.0],
                   ]
    }
    params_to_plot = ["chirp_mass",
                      "mass_ratio",
                      "lambda_tilde",
                      "delta_lambda_tilde",
                      "luminosity_distance"]

    # Gaussian
    path = plot_corner_fixed_population_varying_eos(
        gw_event="GW170817",
        population="gaussian",
        source_type="bns",
        params_to_plot=params_to_plot,
        remove_chieff=remove_chieff,
        **gw170817_defaults
    )

    # Double Gaussian
    gw170817_defaults['normalization_keys'] = ["radio_chiEFT",
                                               "radio_chiEFT",
                                               "radio_chiEFT",
                                               "radio_chiEFT",
                                               "radio_chiEFT",
                                               ]
    path = plot_corner_fixed_population_varying_eos(
        gw_event="GW170817",
        population="double_gaussian",
        source_type="bns",
        params_to_plot=params_to_plot,
        remove_chieff=remove_chieff,
        **gw170817_defaults
    )

    # # Uniform
    # gw170817_defaults['ranges'] = None
    # gw170817_defaults['normalization_keys'] = ["radio_chiEFT",
    #                                            "radio_chiEFT",
    #                                            "radio_chiEFT",
    #                                            "radio_chiEFT",
    #                                            "radio_chiEFT",
    #                                            ]
    # gw170817_defaults['ranges'] =  [[1.197435, 1.19775],
    #                                 [0.6, 1.0],
    #                                 [150, 850],
    #                                 [0.0, 200.0],
    #                                 [20.0, 50.0],
    #                                 ]
    # path = plot_corner_fixed_population_varying_eos(
    #     gw_event="GW170817",
    #     population="uniform",
    #     source_type="bns",
    #     params_to_plot=params_to_plot,
    #     remove_chieff=remove_chieff,
    #     **gw170817_defaults
    # )

    # Also create presentation versions for all populations
    print("\n=== Creating GW170817 PRESENTATION versions ===")

    # Gaussian PRESENTATION
    gw170817_defaults_pres = {
        'normalization_keys': ["radio_chiEFT",
                               "radio_chiEFT",
                               "radio_chiEFT",
                               "radio_chiEFT",
                               "radio_chiEFT"
                               ],
        'ranges': [[1.1974, 1.1977],
                   [0.85, 1.0],
                   [100, 1000],
                   [0.0, 75.0],
                   [20.0, 50.0],
                   ]
    }
    plot_corner_fixed_population_varying_eos_PRESENTATION(
        gw_event="GW170817",
        population="gaussian",
        source_type="bns",
        params_to_plot=params_to_plot,
        remove_chieff=remove_chieff,
        **gw170817_defaults_pres
    )

    # Double Gaussian PRESENTATION
    gw170817_defaults_pres['normalization_keys'] = ["radio_chiEFT",
                                                   "radio_chiEFT",
                                                   "radio_chiEFT",
                                                   "radio_chiEFT",
                                                   "radio_chiEFT",
                                                   ]
    plot_corner_fixed_population_varying_eos_PRESENTATION(
        gw_event="GW170817",
        population="double_gaussian",
        source_type="bns",
        params_to_plot=params_to_plot,
        remove_chieff=remove_chieff,
        **gw170817_defaults_pres
    )

    # Uniform PRESENTATION
    gw170817_defaults_pres['normalization_keys'] = ["radio_chiEFT",
                                                   "radio_chiEFT",
                                                   "radio_chiEFT",
                                                   "radio_chiEFT",
                                                   "radio_chiEFT",
                                                   ]
    gw170817_defaults_pres['ranges'] =  [[1.197435, 1.19775],
                                        [0.6, 1.0],
                                        [150, 850],
                                        [0.0, 200.0],
                                        [20.0, 50.0],
                                        ]
    plot_corner_fixed_population_varying_eos_PRESENTATION(
        gw_event="GW170817",
        population="uniform",
        source_type="bns",
        params_to_plot=params_to_plot,
        remove_chieff=remove_chieff,
        **gw170817_defaults_pres
    )

    return path


def plot_gw190425_corner(remove_chieff: bool = False) -> str:
    """
    Convenience function for creating GW190425 corner plot.
    
    Args:
        remove_chieff (bool): Remove chi_eff parameter from plotting if True
    """
    
    params_to_plot = ["chirp_mass",
                      "mass_ratio",
                      "lambda_tilde",
                      "delta_lambda_tilde",
                      "luminosity_distance"
                      ]
    
    # Set hardcoded defaults for GW190425
    gw190425_defaults = {
        'normalization_keys': ["radio",
                               "radio_chiEFT",
                               "radio_NICER",
                               "radio",
                               "radio",
                               ],
    }
    
    # Gaussian
    
    # # NOTE: for GW190425 this is a very poor fit, not plotting now
    # gw190425_defaults["ranges"] = None
    # path = plot_corner_fixed_population_varying_eos(
    #     gw_event="GW190425",
    #     population="gaussian",
    #     source_type="bns",
    #     **gw190425_defaults
    # )
    
    # Double Gaussian
    gw190425_defaults['normalization_keys'] = ["radio_chiEFT",
                                               "radio_NICER",
                                               "radio_NICER",
                                               "radio_NICER",
                                               "radio_NICER",
                                               ]
    gw190425_defaults["ranges"] = [[1.4862, 1.4873],
                                   [0.64, 1.0],
                                   [50, 400],
                                   [0, 100],
                                   [0, 300],
                                   ]
    path = plot_corner_fixed_population_varying_eos(
        gw_event="GW190425",
        population="double_gaussian",
        source_type="bns",
        params_to_plot=params_to_plot,
        remove_chieff=remove_chieff,
        **gw190425_defaults
    )
    
    # Uniform
    gw190425_defaults['normalization_keys'] = ["radio_NICER",
                                               "radio_chiEFT",
                                               "radio_chiEFT",
                                               "radio_chiEFT",
                                               "radio_NICER",
                                               ]
    gw190425_defaults["ranges"] = [[1.4862, 1.4873],
                                   [0.65, 1.0],
                                   [50, 400],
                                   [0, 100],
                                   [30, 300],
                                   ]
    path = plot_corner_fixed_population_varying_eos(
        gw_event="GW190425",
        population="uniform",
        source_type="bns",
        params_to_plot=params_to_plot,
        remove_chieff=remove_chieff,
        **gw190425_defaults
    )

    # Also create presentation versions
    print("\n=== Creating GW190425 PRESENTATION versions ===")

    # Double Gaussian PRESENTATION
    gw190425_defaults_pres = {
        'normalization_keys': ["radio_NICER",
                               "radio_NICER",
                               "radio_NICER",
                               "radio_NICER",
                               "radio_NICER",
                               ],
        "ranges": [[1.4862, 1.4873],
                   [0.64, 1.0],
                   [50, 400],
                   [0, 100],
                   [0, 300],
                   ]
    }
    plot_corner_fixed_population_varying_eos_PRESENTATION(
        gw_event="GW190425",
        population="double_gaussian",
        source_type="bns",
        params_to_plot=params_to_plot,
        remove_chieff=remove_chieff,
        **gw190425_defaults_pres
    )

    # Uniform PRESENTATION
    gw190425_defaults_pres['normalization_keys'] = ["radio_NICER",
                                                   "radio_NICER",
                                                   "radio_NICER",
                                                   "radio_NICER",
                                                   "radio_NICER",
                                                   ]
    gw190425_defaults_pres["ranges"] = [[1.4862, 1.4873],
                                       [0.65, 1.0],
                                       [50, 400],
                                       [0, 100],
                                       [30, 300],
                                       ]
    plot_corner_fixed_population_varying_eos_PRESENTATION(
        gw_event="GW190425",
        population="uniform",
        source_type="bns",
        params_to_plot=params_to_plot,
        remove_chieff=remove_chieff,
        **gw190425_defaults_pres
    )

    return path

def plot_gw230529_corner(remove_chieff: bool = False, remove_spin1z: bool = False) -> str:
    """
    Convenience function for creating GW230529 corner plot.
    
    Args:
        remove_chieff (bool): Remove chi_eff parameter from plotting if True
        remove_spin1z (bool): Remove spin_1z parameter from plotting if True
    """
    
    # Set hardcoded defaults for GW230529
    gw230529_defaults = {
        'normalization_keys': ["radio_NICER",
                               "radio_NICER",
                               "radio_NICER",
                               "radio_chiEFT",
                               "radio_chiEFT"]
    }
    params_to_plot = ["chirp_mass", "mass_ratio", "spin_1z", "lambda_2", "luminosity_distance"]
    
    # Gaussian
    gw230529_defaults["ranges"] = [[2.0245, 2.0289],
                                   [0.25, 0.5],
                                   [-0.3, 0.1],
                                   [0.0, 1800],
                                   [50.0, 400],
                                   ]
    path = plot_corner_fixed_population_varying_eos(
        gw_event="GW230529",
        population="gaussian",
        source_type="nsbh",
        params_to_plot=params_to_plot,
        remove_chieff=remove_chieff,
        remove_spin1z=remove_spin1z,
        **gw230529_defaults
    )
    
    # Double Gaussian
    gw230529_defaults["ranges"] = [[2.0245, 2.0289],
                                   [0.275, 0.5],
                                   [-0.3, 0.1],
                                   [0.0, 1800],
                                   [50.0, 400],
                                   ]
    gw230529_defaults['normalization_keys'] = ["radio",
                                               "radio_NICER",
                                               "radio_NICER",
                                               "radio_NICER",
                                               "radio_NICER"]
    path = plot_corner_fixed_population_varying_eos(
        gw_event="GW230529",
        population="double_gaussian",
        source_type="nsbh",
        params_to_plot=params_to_plot,
        remove_chieff=remove_chieff,
        remove_spin1z=remove_spin1z,
        **gw230529_defaults
    )
    
    # Uniform
    gw230529_defaults["ranges"] = [[2.0245, 2.0289],
                                   [0.275, 0.5],
                                   [-0.3, 0.1],
                                   [0.0, 1800],
                                   [50.0, 400],
                                   ]
    gw230529_defaults['normalization_keys'] = ["radio_NICER",
                                               "radio_chiEFT",
                                               "radio_NICER",
                                               "radio_chiEFT",
                                               "radio_NICER"]
    path = plot_corner_fixed_population_varying_eos(
        gw_event="GW230529",
        population="uniform",
        source_type="nsbh",
        params_to_plot=params_to_plot,
        remove_chieff=remove_chieff,
        remove_spin1z=remove_spin1z,
        **gw230529_defaults
    )

    # Also create presentation versions
    print("\n=== Creating GW230529 PRESENTATION versions ===")

    # Gaussian PRESENTATION
    gw230529_defaults_pres = {
        'normalization_keys': ["default",
                               "radio_NICER",
                               "radio_NICER",
                               "radio",
                               "radio_chiEFT"],
        "ranges": [[2.0245, 2.0289],
                   [0.25, 0.5],
                   [-0.3, 0.1],
                   [0.0, 1800],
                   [50.0, 400],
                   ]
    }
    plot_corner_fixed_population_varying_eos_PRESENTATION(
        gw_event="GW230529",
        population="gaussian",
        source_type="nsbh",
        params_to_plot=params_to_plot,
        remove_chieff=remove_chieff,
        remove_spin1z=remove_spin1z,
        **gw230529_defaults_pres
    )

    # Double Gaussian PRESENTATION
    gw230529_defaults_pres["ranges"] = [[2.0245, 2.0289],
                                       [0.275, 0.5],
                                       [-0.3, 0.1],
                                       [0.0, 1800],
                                       [50.0, 400],
                                       ]
    gw230529_defaults_pres['normalization_keys'] = ["radio",
                                                   "radio_NICER",
                                                   "radio_NICER",
                                                   "radio_NICER",
                                                   "radio_NICER"]
    plot_corner_fixed_population_varying_eos_PRESENTATION(
        gw_event="GW230529",
        population="double_gaussian",
        source_type="nsbh",
        params_to_plot=params_to_plot,
        remove_chieff=remove_chieff,
        remove_spin1z=remove_spin1z,
        **gw230529_defaults_pres
    )

    # Uniform PRESENTATION
    gw230529_defaults_pres["ranges"] = [[2.0245, 2.0289],
                                       [0.275, 0.5],
                                       [-0.3, 0.1],
                                       [0.0, 1800],
                                       [50.0, 400],
                                       ]
    gw230529_defaults_pres['normalization_keys'] = ["radio_NICER",
                                                   "radio_chiEFT",
                                                   "radio_NICER",
                                                   "radio_chiEFT",
                                                   "radio_NICER"]
    plot_corner_fixed_population_varying_eos_PRESENTATION(
        gw_event="GW230529",
        population="uniform",
        source_type="nsbh",
        params_to_plot=params_to_plot,
        remove_chieff=remove_chieff,
        remove_spin1z=remove_spin1z,
        **gw230529_defaults_pres
    )

    return path


def plot_debug_corner(gw_event: str,
                     population: str = "gaussian",
                     source_type: str = "bns",
                     base_path: str = "../final_results/",
                     output_dir: str = "./figures/money_plots/",
                     eos_samples: List[str] = None,
                     params_to_plot: List[str] = None,
                     normalization_keys: List[str] = None,
                     include_default: bool = True,
                     suffix: str = "DEBUG",
                     verbose: bool = False) -> str:
    """
    Create corner plot with custom parameters and suffix.

    By default creates debug corner plot with chirp_mass, chirp_mass_source, lambda_tilde,
    luminosity_distance, and redshift. Can be customized for other parameter sets.

    Args:
        gw_event (str): GW event name
        population (str): Population type
        source_type (str): Source type
        base_path (str): Base path to result files
        output_dir (str): Output directory for figures
        eos_samples (List[str]): EOS samples to include
        params_to_plot (List[str]): Parameters to plot (default: debug parameter set)
        normalization_keys (List[str]): Normalization keys (default: radio_chiEFT for all)
        include_default (bool): Include default uninformed prior run
        suffix (str): Suffix for output filename (default: "DEBUG")
        verbose (bool): Print verbose information

    Returns:
        str: Path to saved figure
    """

    # Set default parameters
    if eos_samples is None:
        eos_samples = ["radio", "radio_chiEFT", "radio_NICER"]

    # Parameters to plot - default to debug set if not specified
    if params_to_plot is None:
        params_to_plot = ["chirp_mass", "chirp_mass_source", "luminosity_distance"]

    # Use radio_chiEFT for all normalization indices if not specified
    if normalization_keys is None:
        normalization_keys = ["radio_chiEFT"] * len(params_to_plot)

    print(f"\n=== Creating {suffix} corner plot for {gw_event} ===")
    print(f"Population: {population}, Source: {source_type}")
    print(f"EOS samples: {eos_samples}")
    print(f"Parameters: {params_to_plot}")
    print(f"Normalization keys: {normalization_keys}")

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

    # Load default run if requested
    default_posterior = None
    if include_default:
        # For default runs, pass the actual source type as population_type and "default" as source_type
        default_path = construct_result_path(base_path, gw_event, source_type,
                                           "default", "radio")
        print(f"Loading default run from: {default_path}")

        if os.path.exists(default_path):
            default_posterior = load_posterior_data(default_path, fast_mode=True)
            if default_posterior is not None:
                if verbose:
                    print(f"Loaded default run with {len(default_posterior['chirp_mass'])} samples")
            else:
                print("Warning: Could not load default run data")
        else:
            print(f"Warning: Default run file not found: {default_path}")

    # Convert lambdas to tilde parameters if needed
    if 'lambda_tilde' in params_to_plot or 'delta_lambda_tilde' in params_to_plot:
        posteriors_dict, params_to_plot = convert_lambdas_with_verbose(
            posteriors_dict, params_to_plot, verbose=verbose
        )

        # Also convert default posterior if it exists
        if default_posterior is not None:
            default_dict = {"default": default_posterior}
            default_dict, _ = convert_lambdas_with_verbose(
                default_dict, params_to_plot, verbose=verbose
            )
            default_posterior = default_dict["default"]

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
                print(f"Available parameters: {list(posterior.keys())[:20]}")

        if samples_list and len(samples_list) == len(params_to_plot):
            samples_dict[eos_name] = np.column_stack(samples_list)
        else:
            print(f"Warning: Could not extract all parameters for {eos_name}")

    if not samples_dict:
        raise ValueError("No valid samples could be extracted!")

    # Add default samples if available
    default_samples = None
    if default_posterior is not None:
        samples_list = []
        for param in params_to_plot:
            if param in default_posterior:
                samples_list.append(np.array(default_posterior[param]))
            else:
                print(f"Warning: Parameter '{param}' not found in default posterior")

        if samples_list and len(samples_list) == len(params_to_plot):
            default_samples = np.column_stack(samples_list)
            if verbose:
                print(f"Created default samples array with shape: {default_samples.shape}")

    # Validate and create normalization dummy dataset
    dummy_dataset = None
    if normalization_keys is not None:
        # Validate normalization_keys
        n_params = len(params_to_plot)
        if len(normalization_keys) != n_params:
            raise ValueError(f"normalization_keys must have length {n_params} (number of parameters), got {len(normalization_keys)}")

        # Check that all keys exist in loaded datasets or are "default"
        loaded_keys = set(samples_dict.keys())
        available_keys = loaded_keys.copy()
        if default_samples is not None:
            available_keys.add("default")

        for key in normalization_keys:
            if key not in available_keys:
                raise ValueError(f"normalization_keys contains '{key}' which was not loaded. Available keys: {available_keys}")

        # Create extended samples dict that includes default if available
        extended_samples_dict = samples_dict.copy()
        if default_samples is not None:
            extended_samples_dict["default"] = default_samples

        # Find minimum number of samples across all datasets
        min_samples = min(len(dataset) for dataset in extended_samples_dict.values())
        if verbose:
            sample_counts = {key: len(dataset) for key, dataset in extended_samples_dict.items()}
            print(f"Sample counts by dataset: {sample_counts}")
            print(f"Using minimum sample count for normalization: {min_samples}")

        # Create dummy dataset by selecting specified dataset for each parameter
        # All datasets are downsampled to min_samples to ensure consistent sizes
        dummy_columns = []
        for param_idx, dataset_key in enumerate(normalization_keys):
            dataset = extended_samples_dict[dataset_key]
            # Downsample to min_samples by taking first min_samples rows
            downsampled_dataset = dataset[:min_samples]
            dummy_columns.append(downsampled_dataset[:, param_idx])

        dummy_dataset = np.column_stack(dummy_columns)
        if verbose:
            print(f"Created normalization dummy dataset using keys: {normalization_keys}")
            print(f"Dummy dataset shape: {dummy_dataset.shape}")

    # Filter constant parameters
    # Merge PARAMETER_LATEX_LABELS with DEBUG_PARAMETER_LATEX_LABELS (DEBUG takes priority)
    merged_labels = {**PARAMETER_LATEX_LABELS, **DEBUG_PARAMETER_LATEX_LABELS}
    labels, latex_labels, samples_dict = filter_constant_parameters(
        params_to_plot,
        [merged_labels.get(p, p) for p in params_to_plot],
        samples_dict
    )

    # Calculate ranges automatically
    ranges = calculate_corner_plot_ranges(labels, samples_dict)
    print(f"\nAutomatic ranges: {ranges}")

    # Create corner plot
    fig = None
    group_colors = {}

    # Plot default samples FIRST (if available) to ensure they appear behind everything
    if default_samples is not None:
        if verbose:
            print("Plotting default run samples in light gray (first layer)")

        default_kwargs = DEFAULT_CORNER_KWARGS.copy()
        default_kwargs.update({
            'color': DEFAULT_RUN_PLOT_COLOR,  # Light color for 2D contours
            'labels': latex_labels,
            'range': ranges,
            'levels': [0.68, 0.95],
            'hist_kwargs': {'color': DEFAULT_RUN_LEGEND_COLOR, 'density': True},  # Dark color for 1D histograms
        })

        fig = corner.corner(default_samples, **default_kwargs)

    for i, (eos_name, samples) in enumerate(samples_dict.items()):
        color = EOS_COLORS.get(eos_name, f"C{i}")
        group_colors[eos_name] = color

        # Corner plot kwargs
        kwargs = DEFAULT_CORNER_KWARGS.copy()
        kwargs.update({
            'color': color,
            'labels': latex_labels,
            'range': ranges,
            'levels': [0.68, 0.95],
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

    # Disable offset notation on all axes to prevent tick label overlap
    # This forces matplotlib to show full tick values instead of using offset format
    for ax in fig.get_axes():
        try:
            ax.ticklabel_format(useOffset=False, style='plain')
        except AttributeError:
            # Some axes use NullFormatter which doesn't support ticklabel_format
            pass

    # Add legend
    legend_entries = []
    legend_colors = []

    for eos_name in samples_dict.keys():
        base_label = EOS_SAMPLES_NAMES_DICT.get(eos_name, eos_name)
        legend_entries.append(base_label)
        legend_colors.append(EOS_COLORS.get(eos_name, 'black'))

    # Add default run to legend if included
    if default_samples is not None:
        legend_entries.append("Uninformed")
        legend_colors.append(DEFAULT_RUN_LEGEND_COLOR)

    # Add legend
    for i, (label, color) in enumerate(zip(legend_entries, legend_colors)):
        plt.text(LEGEND_X, LEGEND_Y - i * LEGEND_DY, label,
                fontsize=LEGEND_FONTSIZE, color=color, ha='left', va='top',
                transform=fig.transFigure, weight='bold')

    # Save figure with suffix in filename
    output_filename = f"{gw_event}_corner_{population}_{source_type}_{suffix}.pdf"
    output_path = os.path.join(output_dir, output_filename)
    ensure_output_directory(output_path)

    plt.savefig(output_path, bbox_inches='tight')
    print(f"\nSaved {suffix} figure to: {output_path}")
    plt.close()

    return output_path


def main():
    plot_gw170817_corner()
    plot_gw190425_corner()
    plot_gw230529_corner()

    # Debug plots
    print("\n" + "="*60)
    print("Creating DEBUG plots")
    print("="*60)

    # GW170817 debug plots
    for population in ["gaussian", "uniform", "double_gaussian"]:
        try:
            plot_debug_corner("GW170817", population=population, source_type="bns")
        except Exception as e:
            print(f"Failed to create DEBUG plot for GW190425 {population}: {e}")
            
    # GW190425 debug plots
    for population in ["gaussian", "uniform", "double_gaussian"]:
        try:
            plot_debug_corner("GW190425", population=population, source_type="bns")
        except Exception as e:
            print(f"Failed to create DEBUG plot for GW190425 {population}: {e}")

    # GW230529 debug plots
    for population in ["gaussian", "uniform", "double_gaussian"]:
        try:
            plot_debug_corner("GW230529", population=population, source_type="nsbh")
        except Exception as e:
            print(f"Failed to create DEBUG plot for GW230529 {population}: {e}")

    # Full plots with all parameters
    print("\n" + "="*60)
    print("Creating FULL plots")
    print("="*60)

    # Define full parameter set
    full_params_bns = [
        "chirp_mass", "mass_ratio", "a_1", "a_2",
        "tilt_1", "tilt_2", "phi_jl", "phi_12",
        "lambda_tilde", "delta_lambda_tilde",
        "luminosity_distance", "phase", "theta_jn", "psi",
        "ra", "dec"
    ]

    full_params_nsbh = [
        "chirp_mass", "mass_ratio", "a_1", "a_2",
        "tilt_1", "tilt_2", "phi_jl", "phi_12",
        "lambda_2",  # NSBH uses lambda_2 instead of tilde parameters
        "luminosity_distance", "phase", "theta_jn", "psi",
        "ra", "dec"
    ]

    # # GW170817 full plots
    # for population in ["gaussian", "uniform", "double_gaussian"]:
    #     try:
    #         plot_debug_corner(
    #             "GW170817",
    #             population=population,
    #             source_type="bns",
    #             params_to_plot=full_params_bns,
    #             suffix="FULL"
    #         )
    #     except Exception as e:
    #         print(f"Failed to create FULL plot for GW170817 {population}: {e}")

    # # GW190425 full plots
    # for population in ["gaussian", "uniform", "double_gaussian"]:
    #     try:
    #         plot_debug_corner(
    #             "GW190425",
    #             population=population,
    #             source_type="bns",
    #             params_to_plot=full_params_bns,
    #             suffix="FULL"
    #         )
    #     except Exception as e:
    #         print(f"Failed to create FULL plot for GW190425 {population}: {e}")

    # # GW230529 full plots
    # for population in ["gaussian", "uniform", "double_gaussian"]:
    #     try:
    #         plot_debug_corner(
    #             "GW230529",
    #             population=population,
    #             source_type="nsbh",
    #             params_to_plot=full_params_nsbh,
    #             suffix="FULL"
    #         )
    #     except Exception as e:
    #         print(f"Failed to create FULL plot for GW230529 {population}: {e}")


if __name__ == "__main__":
    main()