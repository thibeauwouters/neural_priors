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
from bilby.gw.conversion import lambda_1_lambda_2_to_lambda_tilde, lambda_1_lambda_2_to_delta_lambda_tilde
from bilby.gw.conversion import chirp_mass_and_mass_ratio_to_component_masses
from scipy.interpolate import interp1d

# Global cosmology interpolator - loaded once and reused
_COSMOLOGY_INTERPOLATOR = None

JAX_LIGHT_BLUE = "#5e97f6"
JAX_DARK_BLUE = "#2a56c6"
JAX_DARK_BLUE_TINT1 = "#7f9add"
JAX_DARK_BLUE_TINT2 = "#aabbe8"

JAX_LIGHT_GREEN = "#26a69a"
JAX_DARK_GREEN = "#00695c"
JAX_DARK_GREEN_TINT1 = "#4d968d"
JAX_DARK_GREEN_TINT2 = "#80b4ae"

JAX_PINK = "#ea80fc"
JAX_LIGHT_PURPLE = "#9c27b0"
JAX_DARK_PURPLE = "#6a1c9a"
JAX_DARK_PURPLE_TINT1 = "#9760b8"
JAX_DARK_PURPLE_TINT2 = "#b58ecd"

BLUE_COLORS = [JAX_LIGHT_BLUE, JAX_DARK_BLUE_TINT1, JAX_DARK_BLUE_TINT2]
GREEN_COLORS = [JAX_LIGHT_GREEN, JAX_DARK_GREEN_TINT1, JAX_DARK_GREEN_TINT2]
PURPLE_COLORS = [JAX_LIGHT_PURPLE, JAX_DARK_PURPLE_TINT1, JAX_DARK_PURPLE_TINT2]

COLORS = [JAX_LIGHT_BLUE, JAX_LIGHT_GREEN, JAX_LIGHT_PURPLE]

COLORS_LIST_DICT = {"uniform": COLORS,
                    "gaussian": COLORS,
                    "double_gaussian": COLORS}

LABELS_DICT = {"BNS": [r"$m_1$ [M$_\odot$]", r"$m_2$ [M$_\odot$]", r"$\Lambda_1$", r"$\Lambda_2$"],
               "NSBH": [r"$m_1$ [M$_\odot$]", r"$m_2$ [M$_\odot$]", r"$\Lambda_2$"]
               }

TEX_TRANSLATION_DICT = {"m1": r"$m_1$ [M$_\odot$]",
                        "m2": r"$m_2$ [M$_\odot$]",
                        "chirp_mass_source": r"$\mathcal{M}_c^{\rm{src}}$ [M$_\odot$]",
                        "mass_ratio": r"$q$",
                        "lambda_1": r"$\Lambda_1$",
                        "lambda_2": r"$\Lambda_2$",
                        "lambda_tilde": r"$\tilde{\Lambda}$",
                        "delta_lambda_tilde": r"$\delta \tilde{\Lambda}$"
                        }

POPULATION_NAMES = ["uniform", "gaussian", "double_gaussian"]
SOURCE_TYPES = ["BNS", "NSBH"]
EOS_SAMPLES_NAMES = ["radio",
                     "radio_chiEFT",
                     "radio_NICER",
                     "radio_chiEFT_NICER"]
EOS_SAMPLES_NAMES_DICT = {"radio": r"Radio",
                          "radio_chiEFT": r"+$\chi_{\rm{EFT}}$",
                          "radio_chiEFT_NICER": r"+$\chi_{\rm{EFT}}$+NICER",
                          "radio_NICER": r"+NICER"
                          }

# Colorblind-friendly colors for EOS samples (using colors that work for deuteranopia/protanopia)
EOS_COLORS = {
    "radio": "#0372b1",           # Blue - easily distinguishable  
    "radio_chiEFT": "#de8f05",    # Orange - high contrast with blue
    "radio_NICER": "#cb79bc",     # Pink/purple - distinct from both blue and orange
    "radio_chiEFT_NICER": "#029E73"  # Green - for potential future use
}

POPULATION_NAMES_DICT = {
    "uniform": r"Uniform",
    "gaussian": r"Gaussian",
    "double_gaussian": r"Double Gaussian"
}

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
    # Handle special case: default runs are stored under {event}/{actual_source_type}/default/
    if source_type == "default":
        # Default runs are stored under the actual source type (bns/nsbh) not "default" 
        # We need to determine the actual source type from context or pass it explicitly
        # For now, assume BNS default if not specified otherwise
        actual_source_type = population_type if population_type in ["bns", "nsbh"] else "bns"
        full_path = os.path.join(base_dir, gw_event, actual_source_type, "default", "samples.npz")
    else:
        full_path = os.path.join(base_dir, gw_event, source_type, population_type, eos_samples_name, "samples.npz")
    return full_path


def load_posterior_data(result_path: str, fast_mode: bool = True) -> Optional[Dict[str, Any]]:
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
        npz_data = np.load(result_path, allow_pickle=True)
        posterior = {key: npz_data[key] for key in npz_data.files}
        convert_chirp_mass(posterior, fast_mode=fast_mode)
        return posterior
    except Exception as e:
        raise RuntimeError(f"Error loading posterior data from {result_path}: {e}")


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
        
        # Also compute source-frame component masses
        if 'mass_ratio' in posterior:
            mass_1_source, mass_2_source = chirp_mass_and_mass_ratio_to_component_masses(
                chirp_mass_source, np.array(posterior['mass_ratio']))
            posterior['mass_1_source'] = mass_1_source
            posterior['mass_2_source'] = mass_2_source


def convert_lambdas(posterior: Dict[str, Any]) -> None:
    """
    Convert lambda_1 and lambda_2 to lambda_tilde and delta_lambda_tilde.
    Modifies posterior dict in-place.
    
    Args:
        posterior (Dict): Posterior data dictionary containing lambda_1, lambda_2, 
                         chirp_mass, and mass_ratio
    """
    if all(key in posterior for key in ['lambda_1', 'lambda_2', 'chirp_mass', 'mass_ratio']):
        # Compute component masses
        mass_1, mass_2 = chirp_mass_and_mass_ratio_to_component_masses(
            np.array(posterior['chirp_mass']),
            np.array(posterior['mass_ratio'])
        )
        
        # Convert to tilde parameters
        lambda_tilde = lambda_1_lambda_2_to_lambda_tilde(
            np.array(posterior['lambda_1']),
            np.array(posterior['lambda_2']),
            mass_1,
            mass_2
        )
        
        delta_lambda_tilde = lambda_1_lambda_2_to_delta_lambda_tilde(
            np.array(posterior['lambda_1']),
            np.array(posterior['lambda_2']),
            mass_1,
            mass_2
        )
        
        # Add to posterior dict
        posterior['lambda_tilde'] = lambda_tilde
        posterior['delta_lambda_tilde'] = delta_lambda_tilde


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
    
    print(f"Getting the comparison data")
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
    "chi_eff": r"$\chi_{\rm{eff}}$",
    "chi_p": r"$\\chi_{p}$",
    "spin_1z": r"$\chi_{1z}$",
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
DEFAULT_COLOR = 'gray'
BNS_COLOR = 'green'
NSBH_COLOR = 'red'
HAUKE_COLOR = 'purple'
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
        filename = f"../data/hauke/{gw_event}/{gw_event}+EM_result.npz"
        print(f"Loading Hauke's GW+EM data from {filename}")
    else:
        filename = f"../data/hauke/{gw_event}/{gw_event}_result.npz"
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
    
    filename = f"../data/adrian/{gw_event}/{gw_event}_result.npz"
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


def convert_lambdas_with_verbose(posteriors_dict: Dict[str, Dict[str, Any]], 
                                params_to_plot: List[str],
                                verbose: bool = False) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    """
    Convert lambda_1 and lambda_2 to lambda_tilde and delta_lambda_tilde for multiple posteriors.
    Enhanced version with verbose output and quantile calculations.
    
    Args:
        posteriors_dict (Dict): Dictionary of posterior data
        params_to_plot (List[str]): List of parameter names
        verbose (bool): If True, print quantile information
        
    Returns:
        Tuple[Dict, List]: Modified posteriors dict and updated params_to_plot list
    """
    import arviz
    
    updated_posteriors = {}
    updated_params = params_to_plot.copy()
    
    for key, posterior in posteriors_dict.items():
        updated_posterior = posterior.copy()
        
        if 'lambda_1' in posterior and 'lambda_2' in posterior:
            mass_1, mass_2 = chirp_mass_and_mass_ratio_to_component_masses(
                np.array(posterior['chirp_mass']),
                np.array(posterior['mass_ratio']))
            
            lambda_tilde = lambda_1_lambda_2_to_lambda_tilde(
                np.array(posterior['lambda_1']),
                np.array(posterior['lambda_2']),
                mass_1, mass_2)
            
            delta_lambda_tilde = lambda_1_lambda_2_to_delta_lambda_tilde(
                np.array(posterior['lambda_1']),
                np.array(posterior['lambda_2']),
                mass_1, mass_2)
            
            updated_posterior['lambda_tilde'] = np.array(lambda_tilde)
            updated_posterior['delta_lambda_tilde'] = np.array(delta_lambda_tilde)
            
            # Remove old lambdas
            if 'lambda_1' in updated_posterior:
                del updated_posterior['lambda_1']
            if 'lambda_2' in updated_posterior:
                del updated_posterior['lambda_2']
            
            # Print quantiles if verbose
            if verbose:
                print(f"\n{key} posterior quantiles:")
                
                # Lambda tilde
                med = np.median(updated_posterior['lambda_tilde'])
                low, high = arviz.hdi(np.array(updated_posterior['lambda_tilde']), hdi_prob=0.95)
                low = med - low
                high = high - med
                med, low, high = np.round(med, 2), np.round(low, 2), np.round(high, 2)
                print(f"   lambda_tilde: {med}-{low}+{high}")
                
                # Delta lambda tilde
                med = np.median(updated_posterior['delta_lambda_tilde'])
                low, high = arviz.hdi(np.array(updated_posterior['delta_lambda_tilde']), hdi_prob=0.95)
                low = med - low
                high = high - med
                med, low, high = np.round(med, 2), np.round(low, 2), np.round(high, 2)
                print(f"   delta_lambda_tilde: {med}-{low}+{high}")
        else:
            if 'lambda_1' in posterior or 'lambda_2' in posterior:
                raise ValueError("lambda_1 and lambda_2 must both be present in the posterior to convert to lambda_tilde and delta_lambda_tilde.")
        
        updated_posteriors[key] = updated_posterior
    
    # Update params_to_plot list
    if 'lambda_tilde' not in updated_params and any('lambda_1' in p or 'lambda_2' in p for p in posteriors_dict.values()):
        updated_params.append('lambda_tilde')
        updated_params.append('delta_lambda_tilde')
    
    if 'lambda_1' in updated_params:
        updated_params.remove('lambda_1')
    if 'lambda_2' in updated_params:
        updated_params.remove('lambda_2')
        
    return updated_posteriors, updated_params


def filter_constant_parameters(labels: List[str], latex_labels: List[str], 
                              samples_dict: Dict[str, np.ndarray],
                              threshold: float = 1e-10) -> Tuple[List[str], List[str], Dict[str, np.ndarray]]:
    """
    Filter out parameters that are constant across all datasets.
    
    Args:
        labels (List[str]): Parameter names
        latex_labels (List[str]): LaTeX parameter labels
        samples_dict (Dict): Dictionary of sample arrays
        threshold (float): Threshold for considering parameter constant
        
    Returns:
        Tuple[List, List, Dict]: Filtered labels, latex_labels, and samples_dict
    """
    params_to_remove = []
    
    for i, param in enumerate(labels):
        # Get all values for this parameter across all runs
        all_vals_list = []
        for samples in samples_dict.values():
            if i < samples.shape[1]:
                all_vals_list.append(samples[:, i])
        
        if all_vals_list:
            all_vals = np.concatenate(all_vals_list)
            # If parameter has no dynamic range, mark it for removal
            if np.std(all_vals) < threshold:
                params_to_remove.append(i)
                print(f"Removing constant parameter '{param}' from plot (std={np.std(all_vals):.2e})")
    
    # Remove constant parameters from labels and samples
    filtered_labels = labels.copy()
    filtered_latex_labels = latex_labels.copy()
    filtered_samples_dict = {}
    
    if params_to_remove:
        # Remove from labels (in reverse order to preserve indices)
        for i in reversed(params_to_remove):
            del filtered_labels[i]
            del filtered_latex_labels[i]
        
        # Remove from samples
        for key, samples in samples_dict.items():
            # Remove columns corresponding to constant parameters
            mask = np.ones(samples.shape[1], dtype=bool)
            mask[params_to_remove] = False
            filtered_samples_dict[key] = samples[:, mask]
    else:
        filtered_samples_dict = samples_dict.copy()
    
    return filtered_labels, filtered_latex_labels, filtered_samples_dict


def calculate_corner_plot_ranges(labels: List[str], samples_dict: Dict[str, np.ndarray]) -> List[Tuple[float, float]]:
    """
    Calculate ranges for corner plot parameters with special handling for NSBH lambda_1.
    
    Args:
        labels (List[str]): Parameter names
        samples_dict (Dict): Dictionary of sample arrays
        
    Returns:
        List[Tuple]: List of (min, max) tuples for each parameter
    """
    ranges = []
    
    for i, param in enumerate(labels):
        # Get all values for this parameter across all runs
        all_vals_list = []
        for samples in samples_dict.values():
            if i < samples.shape[1]:
                all_vals_list.append(samples[:, i])
        
        if not all_vals_list:
            continue
            
        all_vals = np.concatenate(all_vals_list)
        
        # Handle special case for lambda_1 in NSBH runs
        if param == 'lambda_1' and 'nsbh' in samples_dict:
            nsbh_samples = samples_dict['nsbh']
            if i < nsbh_samples.shape[1] and np.std(nsbh_samples[:, i]) < 1e-10:
                # For constant lambda_1 in NSBH, use range from other runs only
                non_zero_vals_list = []
                for key, samples in samples_dict.items():
                    if key != 'nsbh' and i < samples.shape[1]:
                        non_zero_vals_list.append(samples[:, i])
                
                if non_zero_vals_list:
                    non_zero_vals = np.concatenate(non_zero_vals_list)
                    param_range = (np.min(non_zero_vals), np.max(non_zero_vals))
                else:
                    param_range = (np.min(all_vals), np.max(all_vals))
            else:
                param_range = (np.min(all_vals), np.max(all_vals))
        else:
            param_range = (np.min(all_vals), np.max(all_vals))
        
        ranges.append(param_range)
        
    return ranges


def handle_nsbh_lambda_plotting(samples_dict: Dict[str, np.ndarray], labels: List[str],
                               params_to_plot: List[str], jitter: float = 1e-10) -> Dict[str, np.ndarray]:
    """
    Apply jitter to NSBH lambda_1 values for corner plotting to avoid constant parameter issues.
    
    Args:
        samples_dict (Dict): Dictionary of sample arrays
        labels (List[str]): Parameter names corresponding to sample columns
        params_to_plot (List[str]): Original parameter list to check for lambda_1
        jitter (float): Standard deviation of jitter noise
        
    Returns:
        Dict: Modified samples_dict with jittered lambda_1 for NSBH
    """
    modified_samples_dict = {}
    
    for key, samples in samples_dict.items():
        modified_samples = samples.copy()
        
        # Apply jitter for NSBH lambda_1 plotting
        if 'lambda_1' in params_to_plot and 'nsbh' in key.lower():
            try:
                lambda_1_index = labels.index('lambda_1')
                if lambda_1_index < modified_samples.shape[1]:
                    # Turn into very small jitter around zero instead of NaN to make plot pass
                    modified_samples[:, lambda_1_index] = np.random.normal(
                        0, jitter, size=modified_samples[:, lambda_1_index].shape
                    )
            except ValueError:
                # lambda_1 not in labels
                pass
                
        modified_samples_dict[key] = modified_samples
    
    return modified_samples_dict




def setup_corner_plot_styling(group_names: List[str], use_density: bool = True) -> Tuple[Dict[str, str], Dict[str, Dict]]:
    """
    Setup color assignment and corner plot kwargs for different groups.
    
    Args:
        group_names (List[str]): Names of groups to assign colors to
        use_density (bool): Whether to use density for histograms
        
    Returns:
        Tuple[Dict, Dict]: group_colors and group_kwargs dictionaries
    """
    # Define colors for different groups
    colors = ['blue', 'green', 'red', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    group_colors = {}
    
    # Assign colors to groups
    for i, group_name in enumerate(group_names):
        if group_name in ['default', 'bns', 'nsbh']:
            # Use predefined colors for known groups
            if group_name == 'default':
                group_colors[group_name] = DEFAULT_COLOR
            elif group_name == 'bns':
                group_colors[group_name] = BNS_COLOR
            elif group_name == 'nsbh':
                group_colors[group_name] = NSBH_COLOR
        else:
            # Use rotating colors for other groups
            group_colors[group_name] = colors[i % len(colors)]
    
    # Create corner kwargs for each group
    group_kwargs = {}
    for group_name, color in group_colors.items():
        kwargs = DEFAULT_CORNER_KWARGS.copy()
        kwargs.update({'color': color, 'hist_kwargs': {'color': color, 'density': use_density}})
        group_kwargs[group_name] = kwargs
        
    return group_colors, group_kwargs


def add_corner_plot_legend(fig, group_colors: Dict[str, str], plot_all_params: bool = False):
    """
    Add legend to corner plot with proper positioning and formatting.
    
    Args:
        fig: Matplotlib figure object
        group_colors (Dict): Dictionary mapping group names to colors
        plot_all_params (bool): Whether all parameters are plotted (affects font size)
    """
    import matplotlib.pyplot as plt
    
    # Set font size based on number of parameters
    if plot_all_params:
        fs = 64
    else:
        fs = 34
        
    x = 0.85
    y = 0.85
    dy = 0.1
    
    for group_name, color in group_colors.items():
        label = group_name.upper() if group_name in ['bns', 'nsbh', 'default'] else group_name
        plt.text(x, y, label, fontsize=fs, color=color, ha='center', va='center', 
                transform=fig.transFigure)
        y -= dy


def load_injection_truths(base_path: str, params_to_plot: List[str]) -> Optional[List[float]]:
    """
    Load injection truth values for corner plot if this is an injection run.
    
    Args:
        base_path (str): Base path to look for injection parameters
        params_to_plot (List[str]): List of parameter names
        
    Returns:
        List[float] or None: Truth values for parameters or None if not injection run
    """
    injection_run = "injection" in base_path
    
    if not injection_run:
        return None
        
    truths_filename = os.path.join(base_path, "injection_parameters.json")
    
    if not os.path.exists(truths_filename):
        print(f"Warning: Injection parameters file not found: {truths_filename}")
        return None
        
    try:
        with open(truths_filename, "r") as f:
            truths_dict = json.load(f)
        
        truths = []
        for param in params_to_plot:
            if param in truths_dict:
                truths.append(truths_dict[param])
            else:
                print(f"Warning: Parameter '{param}' not found in injection truths")
                truths.append(None)
                
        return truths
        
    except Exception as e:
        print(f"Error loading injection truths: {e}")
        return None