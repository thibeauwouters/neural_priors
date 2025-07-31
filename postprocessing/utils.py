"""
Utility functions for postprocessing GW parameter estimation results.

This module provides common functionality for loading data, constructing paths,
and managing configurations across different comparison modes.
"""

import os
import json
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from bilby.gw.conversion import luminosity_distance_to_redshift


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
                       actual_eos, f"{source_type}_result.json")


def load_posterior_data(result_path: str) -> Optional[Dict[str, Any]]:
    """
    Load posterior data from result file.
    
    Args:
        result_path (str): Path to result file
        
    Returns:
        Dict containing posterior data or None if file doesn't exist
    """
    if not os.path.exists(result_path):
        return None
        
    try:
        with open(result_path, "r") as f:
            result = json.load(f)
            posterior = result['posterior']['content']
            convert_chirp_mass(posterior)
            return posterior
    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        return None


def convert_chirp_mass(posterior: Dict[str, Any]) -> None:
    """
    Convert source-frame chirp mass into detector-frame chirp mass.
    Modifies posterior dict in-place.
    
    Args:
        posterior (Dict): Posterior data dictionary
    """
    if 'chirp_mass_source' in posterior and 'luminosity_distance' in posterior:
        d_L = np.array(posterior['luminosity_distance'])
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
                        fixed_params: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
    """
    Load posterior data for all groups in a comparison.
    
    Args:
        gw_event (str): GW event name
        base_dir (str): Base directory path
        comparison_mode (str): 'source', 'population', or 'eos'
        fixed_params (Dict): Fixed parameters for the comparison
        
    Returns:
        Dict mapping group names to posterior data
    """
    groups = get_comparison_groups(comparison_mode, gw_event, base_dir, fixed_params)
    posteriors = {}
    
    for group_name, (pop_type, source_type, eos_name) in groups.items():
        result_path = construct_result_path(base_dir, gw_event, pop_type, source_type, eos_name)
        posterior = load_posterior_data(result_path)
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