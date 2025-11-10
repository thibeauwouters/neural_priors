"""
Script to compute and display credible intervals for source-frame chirp mass and redshift.

This script provides a simple interface to load posterior samples from gravitational
wave parameter estimation runs and compute credible intervals for the source-frame
chirp mass parameter (chirp_mass_source) and redshift.
"""

import os
import numpy as np
import arviz as az
from typing import Optional, Tuple

from utils import construct_result_path, load_posterior_data


def get_chirp_mass_source_interval(
    gw_event: str,
    source_type: str,
    population: Optional[str] = None,
    eos_name: Optional[str] = None,
    base_path: str = "../final_results/",
    credible_interval: float = 0.90,
    nb_digits: int = 4,
    verbose: bool = True
) -> Optional[Tuple[float, float, float]]:
    """
    Load posterior samples and compute credible interval for source-frame chirp mass.

    Args:
        gw_event (str): GW event name (e.g., 'GW170817', 'GW190425', 'GW230529')
        source_type (str): Source type ('bns', 'nsbh', 'default')
        population (Optional[str]): Population type ('uniform', 'gaussian', 'double_gaussian').
            Set to None or 'default' for uninformed runs.
        eos_name (Optional[str]): EOS sample name (e.g., 'radio', 'radio_chiEFT', 'radio_NICER').
            Set to None for default runs (will use 'radio' internally).
        base_path (str): Base path to result files
        credible_interval (float): Credible interval percentage (default: 0.90 for 90%)
        nb_digits (int): Number of decimal places for rounding (default: 4)
        verbose (bool): Print detailed information

    Returns:
        Optional[Tuple[float, float, float]]: (median, lower_bound, upper_bound) of credible interval,
            or None if data could not be loaded

    Examples:
        # Uninformed/default run
        >>> get_chirp_mass_source_interval("GW170817", "bns")

        # Informed run with specific EOS
        >>> get_chirp_mass_source_interval("GW170817", "bns", "gaussian", "radio_chiEFT")

        # 95% credible interval with 3 decimal places
        >>> get_chirp_mass_source_interval("GW170817", "bns", "gaussian", "radio_chiEFT",
        ...                                credible_interval=0.95, nb_digits=3)
    """

    # Handle default/uninformed runs
    if population is None or population == "default":
        # For default runs, pass the actual source type as population_type and "default" as source_type
        result_path = construct_result_path(base_path, gw_event, source_type,
                                           "default", "radio")
        pop_str = "default"
        eos_str = "N/A"
    else:
        # Handle informed runs
        if eos_name is None:
            raise ValueError("eos_name must be specified for informed runs (non-default)")

        result_path = construct_result_path(base_path, gw_event, population,
                                           source_type, eos_name)
        pop_str = population
        eos_str = eos_name

    # Check if file exists
    if not os.path.exists(result_path):
        if verbose:
            print(f"ERROR: Result file not found: {result_path}")
        return None

    # Load posterior data
    posterior = load_posterior_data(result_path, fast_mode=True)
    if posterior is None:
        if verbose:
            print(f"ERROR: Could not load posterior data from {result_path}")
        return None

    # Check if chirp_mass_source exists in posterior
    if "chirp_mass_source" not in posterior:
        if verbose:
            print(f"ERROR: 'chirp_mass_source' not found in posterior samples")
        return None

    # Extract chirp_mass_source samples
    chirp_mass_source_samples = np.array(posterior["chirp_mass_source"])

    # Compute median and credible interval
    median = np.median(chirp_mass_source_samples)
    hdi_interval = az.hdi(chirp_mass_source_samples, hdi_prob=credible_interval)
    lower_bound = float(hdi_interval[0])
    upper_bound = float(hdi_interval[1])

    # Compute errors from median
    lower_error = median - lower_bound
    upper_error = upper_bound - median

    # Print results in one line with fixed-width formatting for alignment
    # Use f-string formatting to preserve trailing zeros
    interval_percent = int(credible_interval * 100)
    print(f"Event={gw_event:<10s}, source={source_type:<6s}, pop={pop_str:<16s}, eos={eos_str:<13s}: "
          f"M_c^source ({interval_percent}% HDI) = {median:.{nb_digits}f}_{{-{lower_error:.{nb_digits}f}}}^{{+{upper_error:.{nb_digits}f}}}\\,\\Msun")

    return (median, lower_bound, upper_bound)


def get_parameter_interval(
    gw_event: str,
    source_type: str,
    parameter_name: str,
    population: Optional[str] = None,
    eos_name: Optional[str] = None,
    base_path: str = "../final_results/",
    credible_interval: float = 0.90,
    nb_digits: int = 4,
    verbose: bool = True,
    parameter_label: Optional[str] = None
) -> Optional[Tuple[float, float, float]]:
    """
    Load posterior samples and compute credible interval for any parameter.

    Args:
        gw_event (str): GW event name (e.g., 'GW170817', 'GW190425', 'GW230529')
        source_type (str): Source type ('bns', 'nsbh', 'default')
        parameter_name (str): Name of parameter to extract from posterior
        population (Optional[str]): Population type ('uniform', 'gaussian', 'double_gaussian').
            Set to None or 'default' for uninformed runs.
        eos_name (Optional[str]): EOS sample name (e.g., 'radio', 'radio_chiEFT', 'radio_NICER').
            Set to None for default runs (will use 'radio' internally).
        base_path (str): Base path to result files
        credible_interval (float): Credible interval percentage (default: 0.90 for 90%)
        nb_digits (int): Number of decimal places for rounding (default: 4)
        verbose (bool): Print detailed information
        parameter_label (Optional[str]): LaTeX label for parameter (e.g., 'z' for redshift)

    Returns:
        Optional[Tuple[float, float, float]]: (median, lower_bound, upper_bound) of credible interval,
            or None if data could not be loaded

    Examples:
        # Get redshift for informed run
        >>> get_parameter_interval("GW170817", "bns", "redshift", "gaussian", "radio_chiEFT")

        # Get any parameter with custom label
        >>> get_parameter_interval("GW170817", "bns", "luminosity_distance",
        ...                       parameter_label="d_L")
    """

    # Handle default/uninformed runs
    if population is None or population == "default":
        result_path = construct_result_path(base_path, gw_event, source_type,
                                           "default", "radio")
        pop_str = "default"
        eos_str = "N/A"
    else:
        # Handle informed runs
        if eos_name is None:
            raise ValueError("eos_name must be specified for informed runs (non-default)")

        result_path = construct_result_path(base_path, gw_event, population,
                                           source_type, eos_name)
        pop_str = population
        eos_str = eos_name

    # Check if file exists
    if not os.path.exists(result_path):
        if verbose:
            print(f"ERROR: Result file not found: {result_path}")
        return None

    # Load posterior data
    posterior = load_posterior_data(result_path, fast_mode=True)
    if posterior is None:
        if verbose:
            print(f"ERROR: Could not load posterior data from {result_path}")
        return None

    # Check if parameter exists in posterior
    if parameter_name not in posterior:
        if verbose:
            print(f"ERROR: '{parameter_name}' not found in posterior samples")
        return None

    # Extract parameter samples
    parameter_samples = np.array(posterior[parameter_name])

    # Compute median and credible interval
    median = np.median(parameter_samples)
    hdi_interval = az.hdi(parameter_samples, hdi_prob=credible_interval)
    lower_bound = float(hdi_interval[0])
    upper_bound = float(hdi_interval[1])

    # Compute errors from median
    lower_error = median - lower_bound
    upper_error = upper_bound - median

    # Use provided label or fall back to parameter name
    if parameter_label is None:
        parameter_label = parameter_name

    # Print results in one line with fixed-width formatting for alignment
    interval_percent = int(credible_interval * 100)
    print(f"Event={gw_event:<10s}, source={source_type:<6s}, pop={pop_str:<16s}, eos={eos_str:<13s}: "
          f"{parameter_label} ({interval_percent}% HDI) = {median:.{nb_digits}f}_{{-{lower_error:.{nb_digits}f}}}^{{+{upper_error:.{nb_digits}f}}}")

    return (median, lower_bound, upper_bound)


def get_redshift_interval(
    gw_event: str,
    source_type: str,
    population: Optional[str] = None,
    eos_name: Optional[str] = None,
    base_path: str = "../final_results/",
    credible_interval: float = 0.90,
    nb_digits: int = 4,
    verbose: bool = True
) -> Optional[Tuple[float, float, float]]:
    """
    Load posterior samples and compute credible interval for redshift.

    This is a convenience wrapper around get_parameter_interval for the redshift parameter.

    Args:
        gw_event (str): GW event name (e.g., 'GW170817', 'GW190425', 'GW230529')
        source_type (str): Source type ('bns', 'nsbh', 'default')
        population (Optional[str]): Population type ('uniform', 'gaussian', 'double_gaussian').
            Set to None or 'default' for uninformed runs.
        eos_name (Optional[str]): EOS sample name (e.g., 'radio', 'radio_chiEFT', 'radio_NICER').
            Set to None for default runs (will use 'radio' internally).
        base_path (str): Base path to result files
        credible_interval (float): Credible interval percentage (default: 0.90 for 90%)
        nb_digits (int): Number of decimal places for rounding (default: 4)
        verbose (bool): Print detailed information

    Returns:
        Optional[Tuple[float, float, float]]: (median, lower_bound, upper_bound) of credible interval,
            or None if data could not be loaded

    Examples:
        # Uninformed/default run
        >>> get_redshift_interval("GW170817", "bns")

        # Informed run with specific EOS
        >>> get_redshift_interval("GW170817", "bns", "gaussian", "radio_chiEFT")
    """
    return get_parameter_interval(
        gw_event=gw_event,
        source_type=source_type,
        parameter_name="redshift",
        population=population,
        eos_name=eos_name,
        base_path=base_path,
        credible_interval=credible_interval,
        nb_digits=nb_digits,
        verbose=verbose,
        parameter_label="z"
    )


def compare_chirp_mass_source_intervals(
    gw_event: str,
    source_type: str,
    populations: list = None,
    eos_samples: list = None,
    base_path: str = "../final_results/",
    credible_interval: float = 0.90,
    nb_digits: int = 4,
    include_default: bool = True
) -> dict:
    """
    Compare chirp_mass_source credible intervals across multiple runs.

    Args:
        gw_event (str): GW event name
        source_type (str): Source type ('bns', 'nsbh')
        populations (list): List of population types to compare (default: ['uniform', 'gaussian', 'double_gaussian'])
        eos_samples (list): List of EOS samples to compare (default: ['radio', 'radio_chiEFT', 'radio_NICER'])
        base_path (str): Base path to result files
        credible_interval (float): Credible interval percentage
        nb_digits (int): Number of decimal places for rounding (default: 4)
        include_default (bool): Include uninformed/default run in comparison

    Returns:
        dict: Dictionary mapping run configurations to (median, lower, upper) tuples
    """

    if populations is None:
        populations = ['uniform', 'gaussian', 'double_gaussian']

    if eos_samples is None:
        eos_samples = ['radio', 'radio_chiEFT', 'radio_NICER']

    results = {}

    # Include default run if requested
    if include_default:
        result = get_chirp_mass_source_interval(
            gw_event, source_type, population=None, eos_name=None,
            base_path=base_path, credible_interval=credible_interval,
            nb_digits=nb_digits, verbose=False
        )
        if result is not None:
            results['default'] = result

    # Loop through populations and EOS samples
    for population in populations:
        for eos_name in eos_samples:
            run_key = f"{population}_{eos_name}"
            result = get_chirp_mass_source_interval(
                gw_event, source_type, population, eos_name,
                base_path=base_path, credible_interval=credible_interval,
                nb_digits=nb_digits, verbose=False
            )
            if result is not None:
                results[run_key] = result

    return results


def main():
    """
    Example usage demonstrating the functions.
    """

    # GW170817 - Default model
    print("\n# GW170817 - Default (uninformed) model:")
    get_chirp_mass_source_interval("GW170817", "bns", nb_digits=4)
    get_redshift_interval("GW170817", "bns", nb_digits=4)

    # GW170817 - Mc_source and redshift
    print("\n# GW170817 - Gaussian population, radio_chiEFT:")
    get_chirp_mass_source_interval("GW170817", "bns", "gaussian", "radio_chiEFT", nb_digits=4)
    get_redshift_interval("GW170817", "bns", "gaussian", "radio_chiEFT", nb_digits=4)

    # GW190425
    print("\n# GW190425:")
    get_chirp_mass_source_interval("GW190425", "bns", nb_digits=4)
    get_chirp_mass_source_interval("GW190425", "bns", "uniform", "radio_NICER", nb_digits=4)

    # GW230529
    print("\n# GW230529:")
    compare_chirp_mass_source_intervals("GW230529", "nsbh", nb_digits=3)


if __name__ == "__main__":
    main()
