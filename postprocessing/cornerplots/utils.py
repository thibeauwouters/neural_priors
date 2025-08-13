"""
Utility functions for postprocessing GW parameter estimation results.

This module provides common functionality for loading data, constructing paths,
and managing configurations across different comparison modes.
"""

import os
import json
import time
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from bilby.gw.conversion import luminosity_distance_to_redshift
from scipy.interpolate import interp1d

# Global cosmology interpolator - loaded once and reused
_COSMOLOGY_INTERPOLATOR = None

def load_cosmology_interpolator(interpolator_path: str = None) -> Optional[interp1d]:
    """
    Load the pre-computed cosmology interpolator for fast d_L -> z conversion.
    
    Args:
        interpolator_path (str, optional): Path to interpolator file.
                                          If None, looks in current directory.
    
    Returns:
        scipy.interpolate.interp1d: Interpolation function or None if not found
    """
    global _COSMOLOGY_INTERPOLATOR
    
    if _COSMOLOGY_INTERPOLATOR is not None:
        return _COSMOLOGY_INTERPOLATOR
    
    if interpolator_path is None:
        # Look for interpolator in same directory as this file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        interpolator_path = os.path.join(current_dir, "cosmology_interpolator.npz")
    
    if not os.path.exists(interpolator_path):
        print(f"Warning: Cosmology interpolator not found at {interpolator_path}")
        print("Run 'python make_cosmology_interpolator.py' to create it.")
        return None
    
    try:
        print(f"Loading cosmology interpolator from {interpolator_path}")
        data = np.load(interpolator_path)
        
        # Create interpolation function
        _COSMOLOGY_INTERPOLATOR = interp1d(
            data['d_L_grid'], data['z_grid'], 
            kind='cubic', bounds_error=False, fill_value='extrapolate'
        )
        
        print(f"Loaded interpolator with {len(data['d_L_grid'])} points")
        print(f"Range: {data['d_L_min']:.1f} - {data['d_L_max']:.1f} Mpc")
        data.close()
        
        return _COSMOLOGY_INTERPOLATOR
        
    except Exception as e:
        print(f"Error loading cosmology interpolator: {e}")
        return None


def luminosity_distance_to_redshift_fast(d_L: np.ndarray, 
                                         interpolator: interp1d = None) -> np.ndarray:
    """
    Fast conversion from luminosity distance to redshift using interpolation.
    
    Args:
        d_L (np.ndarray): Luminosity distances in Mpc
        interpolator (interp1d, optional): Pre-loaded interpolator function
    
    Returns:
        np.ndarray: Corresponding redshifts
    """
    if interpolator is None:
        interpolator = load_cosmology_interpolator()
    
    if interpolator is None:
        print("Falling back to exact bilby calculation")
        return luminosity_distance_to_redshift(d_L)
    
    return interpolator(d_L)


def get_available_options(base_dir: str, gw_event: str) -> Dict[str, List[str]]:
    """
    Scan the directory structure to find available options for each dimension.
    
    Args:
        base_dir (str): Base directory path
        gw_event (str): GW event name
        
    Returns:
        Dict with keys 'populations', 'sources', 'eos_constraints' containing available options
    """
    event_dir = os.path.join(base_dir, gw_event)
    if not os.path.exists(event_dir):
        return {'populations': [], 'sources': [], 'eos_constraints': []}
    
    populations = []
    sources = set()
    eos_constraints = set()
    
    for pop_type in os.listdir(event_dir):
        pop_path = os.path.join(event_dir, pop_type)
        if os.path.isdir(pop_path) and pop_type not in ['figures', 'final_results']:
            populations.append(pop_type)
            
            for source_type in os.listdir(pop_path):
                source_path = os.path.join(pop_path, source_type)
                if os.path.isdir(source_path):
                    sources.add(source_type)
                    
                    for eos_name in os.listdir(source_path):
                        eos_path = os.path.join(source_path, eos_name)
                        if os.path.isdir(eos_path):
                            eos_constraints.add(eos_name)
    
    return {
        'populations': populations,
        'sources': list(sources),
        'eos_constraints': list(eos_constraints)
    }


def construct_result_path(base_dir: str, gw_event: str, population_type: str, 
                         source_type: str, eos_samples_name: str) -> str:
    """
    Construct path to result file with proper handling of special cases.
    
    Args:
        base_dir (str): Base directory path
        gw_event (str): GW event name  
        population_type (str): Population type
        source_type (str): Source type (bns, nsbh, default, etc.)
        eos_samples_name (str): EOS samples name
        
    Returns:
        str: Path to result file
    """
    # Handle special case: default runs in gaussian/double_gaussian are stored in uniform/radio/
    if source_type == "default" and population_type in ["gaussian", "double_gaussian"]:
        actual_population = "uniform"
        actual_eos = "radio"
    else:
        actual_population = population_type
        actual_eos = eos_samples_name
    
    return os.path.join(base_dir, gw_event, actual_population, source_type, 
                       actual_eos, f"{source_type}_result.npz")


def load_posterior_data(result_path: str, fast_mode: bool = False) -> Optional[Dict[str, Any]]:
    """
    Load posterior data from result file (NPZ format).
    
    Args:
        result_path (str): Path to result file
        fast_mode (bool): If True, use fast interpolation for cosmology calculations
        
    Returns:
        Dict containing posterior data or None if file doesn't exist
    """
    if not os.path.exists(result_path):
        return None
        
    try:
        # Load NPZ file and convert to dictionary
        npz_data = np.load(result_path)
        posterior = {key: npz_data[key] for key in npz_data.files}
        convert_chirp_mass(posterior, fast_mode=fast_mode)
        return posterior
    except (FileNotFoundError, KeyError, OSError, ValueError):
        return None


def convert_chirp_mass(posterior: Dict[str, Any], fast_mode: bool = False) -> None:
    """
    Convert source-frame chirp mass into detector-frame chirp mass.
    Modifies posterior dict in-place.
    
    Args:
        posterior (Dict): Posterior data dictionary
        fast_mode (bool): If True, use fast interpolation for d_L -> z conversion
    """
    if 'chirp_mass_source' in posterior and 'luminosity_distance' in posterior:
        d_L = np.array(posterior['luminosity_distance'])
        
        if fast_mode:
            z = luminosity_distance_to_redshift_fast(d_L)
        else:
            z = luminosity_distance_to_redshift(d_L)
            
        chirp_mass_source = np.array(posterior['chirp_mass_source'])
        posterior['chirp_mass'] = chirp_mass_source * (1 + z)


def get_comparison_groups(comparison_mode: str, gw_event: str, base_dir: str,
                         fixed_params: Dict[str, str]) -> Dict[str, Tuple[str, str, str]]:
    """
    Get the groups to compare based on comparison mode and fixed parameters.
    
    Args:
        comparison_mode (str): 'source', 'population', or 'eos'
        gw_event (str): GW event name
        base_dir (str): Base directory path
        fixed_params (Dict): Fixed parameters (keys vary by comparison mode)
        
    Returns:
        Dict mapping group names to (population_type, source_type, eos_samples_name) tuples
    """
    available = get_available_options(base_dir, gw_event)
    
    if comparison_mode == 'source':
        # Compare across source types, fix population and eos
        population_type = fixed_params['population_type']
        eos_samples_name = fixed_params['eos_samples_name']
        
        groups = {}
        for source_type in available['sources']:
            result_path = construct_result_path(base_dir, gw_event, population_type, 
                                              source_type, eos_samples_name)
            if os.path.exists(result_path):
                groups[source_type] = (population_type, source_type, eos_samples_name)
        
    elif comparison_mode == 'population':
        # Compare across population types, fix source and eos
        source_type = fixed_params['source_type']
        eos_samples_name = fixed_params['eos_samples_name']
        
        groups = {}
        for population_type in available['populations']:
            result_path = construct_result_path(base_dir, gw_event, population_type,
                                              source_type, eos_samples_name)
            if os.path.exists(result_path):
                groups[population_type] = (population_type, source_type, eos_samples_name)
                
    elif comparison_mode == 'eos':
        # Compare across EOS constraints, fix population and source
        population_type = fixed_params['population_type']
        source_type = fixed_params['source_type']
        
        groups = {}
        for eos_samples_name in available['eos_constraints']:
            result_path = construct_result_path(base_dir, gw_event, population_type,
                                              source_type, eos_samples_name)
            if os.path.exists(result_path):
                groups[eos_samples_name] = (population_type, source_type, eos_samples_name)
    
    else:
        raise ValueError(f"Invalid comparison mode: {comparison_mode}. Must be 'source', 'population', or 'eos'")
    
    return groups


def load_comparison_data(gw_event: str, base_dir: str, comparison_mode: str,
                        fixed_params: Dict[str, str], fast_mode: bool = False) -> Dict[str, Dict[str, Any]]:
    """
    Load posterior data for all groups in a comparison.
    
    Args:
        gw_event (str): GW event name
        base_dir (str): Base directory path
        comparison_mode (str): 'source', 'population', or 'eos'
        fixed_params (Dict): Fixed parameters for the comparison
        fast_mode (bool): If True, use fast interpolation for cosmology calculations
        
    Returns:
        Dict mapping group names to posterior data
    """
    groups = get_comparison_groups(comparison_mode, gw_event, base_dir, fixed_params)
    posteriors = {}
    
    for group_name, (pop_type, source_type, eos_name) in groups.items():
        result_path = construct_result_path(base_dir, gw_event, pop_type, source_type, eos_name)
        posterior = load_posterior_data(result_path, fast_mode=fast_mode)
        if posterior is not None:
            posteriors[group_name] = posterior
        else:
            print(f"Warning: Could not load data for {group_name} from {result_path}")
    
    return posteriors


def get_bayes_factor_data(gw_event: str, base_dir: str, comparison_mode: str,
                         fixed_params: Dict[str, str]) -> Dict[str, float]:
    """
    Load Bayes factor data for all groups in a comparison.
    
    Args:
        gw_event (str): GW event name
        base_dir (str): Base directory path
        comparison_mode (str): 'source', 'population', or 'eos'
        fixed_params (Dict): Fixed parameters for the comparison
        
    Returns:
        Dict mapping group names to log Bayes factors
    """
    groups = get_comparison_groups(comparison_mode, gw_event, base_dir, fixed_params)
    bf_dict = {}
    
    for group_name, (pop_type, source_type, eos_name) in groups.items():
        result_path = construct_result_path(base_dir, gw_event, pop_type, source_type, eos_name)
        
        if not os.path.exists(result_path):
            print(f"Results file not found for {group_name}: {result_path}. Setting Bayes factor to 0.0.")
            bf_dict[group_name] = 0.0
        else:
            try:
                with open(result_path, "r") as f:
                    result = json.load(f)
                    bf_dict[group_name] = result["log_bayes_factor"]
            except (FileNotFoundError, KeyError) as e:
                print(f"Error loading Bayes factor for {group_name}: {e}. Setting to 0.0.")
                bf_dict[group_name] = 0.0
    
    return bf_dict


def get_output_directory(base_dir: str, gw_event: str, comparison_mode: str, 
                        fixed_params: Dict[str, str]) -> str:
    """
    Get appropriate output directory for results based on comparison mode.
    
    Args:
        base_dir (str): Base directory path
        gw_event (str): GW event name
        comparison_mode (str): 'source', 'population', or 'eos'
        fixed_params (Dict): Fixed parameters for the comparison
        
    Returns:
        str: Output directory path
    """
    if comparison_mode == 'source':
        # For source comparison, use the population directory
        return os.path.join(base_dir, gw_event, fixed_params['population_type'])
    elif comparison_mode == 'population':
        # For population comparison, create a comparison directory
        return os.path.join(base_dir, gw_event, "population_comparison", 
                           f"{fixed_params['source_type']}_{fixed_params['eos_samples_name']}")
    elif comparison_mode == 'eos':
        # For EOS comparison, create a comparison directory
        return os.path.join(base_dir, gw_event, "eos_comparison",
                           f"{fixed_params['population_type']}_{fixed_params['source_type']}")
    else:
        raise ValueError(f"Invalid comparison mode: {comparison_mode}")


# Plotting configuration and constants
import matplotlib.pyplot as plt

# Global verbose flag
VERBOSE = False

# Matplotlib style configuration
def setup_matplotlib_style():
    """Setup matplotlib style parameters for consistent plotting."""
    params = {
        "axes.grid": False,
        "text.usetex": True,
        "font.family": "serif",
        "ytick.color": "black",
        "xtick.color": "black",
        "axes.labelcolor": "black",
        "axes.edgecolor": "black",
        "font.serif": ["Computer Modern Serif"],
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "axes.labelsize": 16,
        "legend.fontsize": 16,
        "legend.title_fontsize": 16,
        "figure.titlesize": 16
    }
    plt.rcParams.update(params)

# LaTeX parameter labels dictionary
PARAMETER_LATEX_LABELS = {
    "chirp_mass": r"$\mathcal{M}_c$ [$M_{\odot}$]",
    "mass_ratio": r"$q$",
    "luminosity_distance": r"$d_L$ [Mpc]",
    "geocent_time": r"$t_c$ [s]",
    "a_1": r"$a_1$",
    "a_2": r"$a_2$",
    "tilt_1": r"$\theta_{1}$",
    "tilt_2": r"$\theta_{2}$",
    "phi_12": r"$\phi_{12}$",
    "phi_jl": r"$\phi_{JL}$",
    "dec": r"$\delta$",
    "ra": r"$\alpha$",
    "theta_jn": r"$\theta_{JN}$",
    "psi": r"$\psi$",
    "phase": r"$\phi$",
    "lambda_1": r"$\Lambda_1$",
    "lambda_2": r"$\Lambda_2$",
    "lambda_tilde": r"$\tilde{\Lambda}$",
    "delta_lambda_tilde": r"$\delta \tilde{\Lambda}$"
}

# Default corner plot kwargs
DEFAULT_CORNER_KWARGS = dict(
    bins=40, 
    smooth=1., 
    show_titles=False,
    label_kwargs=dict(fontsize=16),
    title_kwargs=dict(fontsize=16), 
    plot_density=True, 
    plot_datapoints=False, 
    fill_contours=True,
    max_n_ticks=4, 
    min_n_ticks=3,
    truth_color="black",
    save=False
)

# Color constants for different plot groups
DEFAULT_COLOR = 'blue'
BNS_COLOR = 'green'
NSBH_COLOR = 'red'
HAUKE_COLOR = 'orange'
HAUKE_EM_COLOR = 'purple'
ADRIAN_COLOR = 'black'


def apply_parameter_dummy_replacement(data: Dict[str, np.ndarray], 
                                     params_to_replace: List[str]) -> Dict[str, np.ndarray]:
    """
    Apply dummy replacement (small jitter) to specified parameters.
    
    This is used when external data (Hauke, Adrian) has fixed parameters that need
    small jitter to avoid corner plot complaints.
    
    Args:
        data (Dict): Dictionary containing parameter data
        params_to_replace (List[str]): List of parameter names to apply jitter to
        
    Returns:
        Dict: Modified data dictionary with jittered parameters
    """
    modified_data = data.copy()
    
    for key in params_to_replace:
        if key in modified_data:
            print(f"Replacing {key} with dummy values...")
            original_values = modified_data[key]
            jitter_std = 0.01
            modified_data[key] = np.random.normal(original_values, jitter_std, size=len(original_values))
    
    return modified_data


def load_hauke_data(gw_event: str, params_to_plot: List[str], 
                   use_em_data: bool = False) -> Optional[np.ndarray]:
    """
    Load Hauke's posterior data for the specified GW event.
    
    Args:
        gw_event (str): GW event name (e.g., 'GW170817', 'GW190425')
        params_to_plot (List[str]): List of parameter names to extract
        use_em_data (bool): If True, load GW+EM analysis data (only for GW170817)
        
    Returns:
        np.ndarray: Sample array with shape (n_samples, n_params) or None if not available
    """
    if gw_event not in ["GW170817", "GW190425"]:
        print(f"Hauke's data not available for {gw_event}")
        return None
    
    # Special case: GW+EM data only available for GW170817
    if use_em_data and gw_event != "GW170817":
        print(f"Hauke's GW+EM data not available for {gw_event}")
        return None
    
    # Construct filename
    if use_em_data:
        filename = f"../../data/hauke/{gw_event}/{gw_event}+EM_result.npz"
        print(f"Loading Hauke's GW+EM data from {filename}")
    else:
        filename = f"../../data/hauke/{gw_event}/{gw_event}_result.npz"
        print(f"Loading Hauke's data from {filename}")
    
    if not os.path.exists(filename):
        print(f"Hauke's data file not found: {filename}")
        return None
    
    try:
        # Load data
        hauke_data = np.load(filename)
        hauke_dict = {k: hauke_data[k] for k in params_to_plot if k in hauke_data}
        
        # Apply parameter dummy replacement for fixed parameters
        if gw_event == "GW170817":
            params_to_dummy_replace = ["geocent_time", "phase", "ra", "dec"]
        else:
            params_to_dummy_replace = ["geocent_time", "phase"]
            
        hauke_dict = apply_parameter_dummy_replacement(hauke_dict, params_to_dummy_replace)
        
        # Convert to sample array
        samples = np.array([hauke_dict[param] for param in params_to_plot if param in hauke_dict]).T
        
        print(f"Successfully loaded Hauke's data with shape {samples.shape}")
        return samples
        
    except Exception as e:
        print(f"Error loading Hauke's data: {e}")
        return None


def load_adrian_data(gw_event: str, params_to_plot: List[str]) -> Optional[np.ndarray]:
    """
    Load Adrian's posterior data for the specified GW event.
    
    Args:
        gw_event (str): GW event name (e.g., 'GW170817', 'GW190425')
        params_to_plot (List[str]): List of parameter names to extract
        
    Returns:
        np.ndarray: Sample array with shape (n_samples, n_params) or None if not available
    """
    if gw_event not in ["GW170817", "GW190425"]:
        print(f"Adrian's data not available for {gw_event}")
        return None
    
    filename = f"../../data/adrian/{gw_event}/{gw_event}_result.npz"
    print(f"Loading Adrian's data from {filename}")
    
    if not os.path.exists(filename):
        print(f"Adrian's data file not found: {filename}")
        return None
    
    try:
        # Load data
        adrian_data = np.load(filename)
        adrian_dict = {k: adrian_data[k] for k in params_to_plot if k in adrian_data}
        
        # Apply parameter dummy replacement for fixed parameters
        if gw_event == "GW170817":
            params_to_dummy_replace = ["geocent_time", "phase", "ra", "dec"]
        else:
            params_to_dummy_replace = ["geocent_time", "phase"]
            
        adrian_dict = apply_parameter_dummy_replacement(adrian_dict, params_to_dummy_replace)
        
        # Convert to sample array
        samples = np.array([adrian_dict[param] for param in params_to_plot if param in adrian_dict]).T
        
        print(f"Successfully loaded Adrian's data with shape {samples.shape}")
        return samples
        
    except Exception as e:
        print(f"Error loading Adrian's data: {e}")
        return None