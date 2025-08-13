"""
File to create huge batches of cornerplots with various assumptions and parameters.
This is the main entry point for generating corner plots for GW parameter estimation results.
It allows for flexible comparison modes, population types, source types, and EOS samples.
It also supports plotting Hauke's and Adrian's results for comparison.

NOTE: this is not to produce the final corner plots for the paper, but rather to generate
a large number of corner plots for different assumptions and parameters, which can then be used
to analyze the impact of different choices on the results and decide what to show in the paper.
"""

import os
import json
import arviz
import numpy as np
import matplotlib.pyplot as plt
import corner
import argparse

from bilby.gw.conversion import lambda_1_lambda_2_to_lambda_tilde, lambda_1_lambda_2_to_delta_lambda_tilde
from bilby.gw.conversion import chirp_mass_and_mass_ratio_to_component_masses
from utils import (
    load_comparison_data, construct_result_path, load_posterior_data, load_cosmology_interpolator,
    setup_matplotlib_style, PARAMETER_LATEX_LABELS, DEFAULT_CORNER_KWARGS, VERBOSE,
    DEFAULT_COLOR, BNS_COLOR, NSBH_COLOR, HAUKE_COLOR, HAUKE_EM_COLOR, ADRIAN_COLOR,
    load_hauke_data, load_adrian_data
)

# Setup matplotlib style using utils function
setup_matplotlib_style()

    

def create_corner_plot(GW_event: str,
                       comparison_mode: str = "source",
                       population_type: str = "uniform",
                       source_type: str = "bns", 
                       eos_samples_name: str = "radio",
                       base_path: str = "../final_results/",
                       plot_all_params: bool = False,
                       plot_default: bool = False,
                       convert_lambdas: bool = True,
                       plot_hauke: bool = False,
                       plot_hauke_EM: bool = False,
                       plot_adrian: bool = False,
                       prevent_bns_leakage: bool = False,
                       fast_plotting: bool = True):
    """
    Create a corner plot comparing posterior samples across different dimensions.
    
    Args:
        GW_event (str): The name of the GW event to create the corner plot for.
        comparison_mode (str): What to compare - 'source', 'population', or 'eos'.
        population_type (str): Population type (uniform, gaussian, double_gaussian).
        source_type (str): Source type (bns, nsbh, default).
        eos_samples_name (str): EOS samples name (default: radio).
        base_path (str): Base path to the GW runs directory.
        plot_all_params (bool): If True, plot all parameters. If False, plot only a subset of parameters.
        plot_default (bool): If True, include the Default run in the corner plot.
        convert_lambdas (bool): If True, convert lambda_1 and lambda_2 to lambda_tilde and delta_lambda_tilde.
        plot_hauke (bool): If True, include Hauke's data in the corner plot.
        plot_hauke_EM (bool): If True, include Hauke's data in the corner plot that analyzed the GW+EM dataset.
        plot_adrian (bool): If True, include Adrian's data in the corner plot.
        prevent_bns_leakage (bool): If True, prevent BNS leakage by masking samples with negative delta_lambda_tilde.
        fast_plotting (bool): If True, use fast cosmology interpolator for speed.
    """
    
    injection_run = "injection" in base_path
    
    # Set up fixed parameters based on comparison mode
    if comparison_mode == "source":
        fixed_params = {"population_type": population_type, "eos_samples_name": eos_samples_name}
    elif comparison_mode == "population":
        fixed_params = {"source_type": source_type, "eos_samples_name": eos_samples_name}
    elif comparison_mode == "eos":
        fixed_params = {"population_type": population_type, "source_type": source_type}
    else:
        raise ValueError(f"Invalid comparison mode: {comparison_mode}")
        
    # Load comparison data
    posteriors_dict = load_comparison_data(GW_event, base_path, comparison_mode, fixed_params, fast_mode=fast_plotting)
    
    # Add default run if requested
    if plot_default:
        default_path = construct_result_path(base_path, GW_event, population_type, "default", eos_samples_name)
        default_posterior = load_posterior_data(default_path, fast_mode=fast_plotting)
        if default_posterior:
            posteriors_dict["default"] = default_posterior
        
    if len(posteriors_dict) == 0:
        print(f"WARNING: Skipping plotting: No posteriors found for GW event {GW_event} with comparison mode {comparison_mode} and fixed parameters {fixed_params}.")
        return
        
    print(f"Loaded {len(posteriors_dict)} posterior datasets: {list(posteriors_dict.keys())}")

    # Define parameters to plot (excluding log_likelihood and log_prior)
    if plot_all_params:
        params_to_plot = ['chirp_mass', 'mass_ratio', 'luminosity_distance', 'geocent_time', 
                        'a_1', 'a_2', 'tilt_1', 'tilt_2', 'phi_12', 'phi_jl', 
                        'dec', 'ra', 'theta_jn', 'psi', 'phase', 'lambda_1', 'lambda_2']
    else:
        params_to_plot = ['chirp_mass', 'mass_ratio', 'luminosity_distance', 'geocent_time', 'lambda_1', 'lambda_2']
        
    # Remove lambda_1 for NSBH-only runs (it's constant at 0)
    if source_type == 'nsbh' and 'lambda_1' in params_to_plot:
        params_to_plot.remove('lambda_1')
        print("Removed lambda_1 from parameters to plot (NSBH source type)")
        
    # If we convert lambdas, then we need to convert them to lambda_tilde and delta_lambda_tilde
    if convert_lambdas:
        
        # Convert lambda_1 and lambda_2 to lambda_tilde and delta_lambda_tilde
        for key, posterior in posteriors_dict.items():
            
            mass_1, mass_2 = chirp_mass_and_mass_ratio_to_component_masses(
                np.array(posterior['chirp_mass']),
                np.array(posterior['mass_ratio']))
            
            if 'lambda_1' in posterior and 'lambda_2' in posterior:
                lambda_tilde = lambda_1_lambda_2_to_lambda_tilde(np.array(posterior['lambda_1']),
                                                                 np.array(posterior['lambda_2']),
                                                                 mass_1,
                                                                 mass_2)
                delta_lambda_tilde = lambda_1_lambda_2_to_delta_lambda_tilde(np.array(posterior['lambda_1']),
                                                                             np.array(posterior['lambda_2']),
                                                                             mass_1,
                                                                             mass_2)
                
                posterior['lambda_tilde'] = np.array(lambda_tilde)
                posterior['delta_lambda_tilde'] = np.array(delta_lambda_tilde)
                
                # Remove old lambdas
                del posterior['lambda_1']
                del posterior['lambda_2']
                
                # Save again
                posteriors_dict[key] = posterior
                
                # Print the 95% quantiles for lambda_tilde and delta_lambda_tilde
                if VERBOSE:
                    print(f"\n{key} posterior quantiles:")
                    med = np.median(posterior['lambda_tilde'])
                    low, high = arviz.hdi(np.array(posterior['lambda_tilde']), hdi_prob=0.95)
                    low = med - low
                    high = high - med
                    
                    # Round to 2 digits
                    med = np.round(med, 2)
                    low = np.round(low, 2)
                    high = np.round(high, 2)
                    print(f"   lambda_tilde: {med}-{low}+{high}")
                    
                    med = np.median(posterior['delta_lambda_tilde'])
                    low, high = arviz.hdi(np.array(posterior['delta_lambda_tilde']), hdi_prob=0.95)
                    low = med - low
                    high = high - med
                    
                    # Round to 2 digits
                    med = np.round(med, 2)
                    low = np.round(low, 2)
                    high = np.round(high, 2)
                    print(f"   delta_lambda_tilde: {med}-{low}+{high}")
            
            else:
                raise ValueError("lambda_1 and lambda_2 must be present in the posterior to convert to lambda_tilde and delta_lambda_tilde.")
            
        # Add new labels
        params_to_plot.append('lambda_tilde')
        params_to_plot.append('delta_lambda_tilde')
        
        if 'lambda_1' in params_to_plot:
            del params_to_plot[params_to_plot.index('lambda_1')]
        if 'lambda_2' in params_to_plot:
            del params_to_plot[params_to_plot.index('lambda_2')]
        
    # Create data arrays for each posterior
    labels = []
    latex_labels = []
    samples_dict = {}

    # Get reference posterior to check which parameters exist
    if posteriors_dict:
        ref_posterior = list(posteriors_dict.values())[0]
    else:
        ref_posterior = None

    for param in params_to_plot:
        if ref_posterior is not None and param in ref_posterior:
            labels.append(param)
            latex_labels.append(PARAMETER_LATEX_LABELS.get(param, param))

    # Create data arrays for each posterior
    for key, posterior in posteriors_dict.items():
        data = []
        for param in labels:
            if param in posterior:
                data.append(posterior[param])
        if data:
            samples_dict[key] = np.array(data).T
    
    # Filter out parameters that are constant across all datasets
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
            if np.std(all_vals) < 1e-10:
                params_to_remove.append(i)
                print(f"Removing constant parameter '{param}' from plot (std={np.std(all_vals):.2e})")
    
    # Remove constant parameters from labels and samples
    if params_to_remove:
        # Remove from labels (in reverse order to preserve indices)
        for i in reversed(params_to_remove):
            del labels[i]
            del latex_labels[i]
        
        # Remove from samples
        for key, samples in samples_dict.items():
            # Remove columns corresponding to constant parameters
            mask = np.ones(samples.shape[1], dtype=bool)
            mask[params_to_remove] = False
            samples_dict[key] = samples[:, mask]
    
    # For the BNS samples, in case we have lambda tilde, mask to only have positive delta lambda tilde
    if 'delta_lambda_tilde' in params_to_plot and prevent_bns_leakage and 'bns' in samples_dict:
        delta_lambda_tilde_index = labels.index('delta_lambda_tilde')
        bns_samples = samples_dict['bns']
        samples_dict['bns'] = bns_samples[bns_samples[:, delta_lambda_tilde_index] > 0]
    
    # Create range dictionary to handle constant parameters
    ranges = []
    for i, param in enumerate(labels):
        # Get all values for this parameter across all runs
        all_vals_list = []
        for samples in samples_dict.values():
            if i < samples.shape[1]:
                all_vals_list.append(samples[:, i])
        
        if all_vals_list:
            all_vals = np.concatenate(all_vals_list)
        else:
            continue
        
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
        
    # Make lambda_1 NaN for NSBH samples for plotting
    if 'lambda_1' in params_to_plot and 'nsbh' in samples_dict:
        lambda_1_index = labels.index('lambda_1')
        nsbh_samples = samples_dict['nsbh']
        if lambda_1_index < nsbh_samples.shape[1]:
            # Turn into very small jitter around zero instead of NaN to make plot pass
            jitter = 1e-10
            nsbh_samples[:, lambda_1_index] = np.random.normal(0, jitter, size=nsbh_samples[:, lambda_1_index].shape)
            samples_dict['nsbh'] = nsbh_samples
        
    # Create corner plot with overlaid distributions
    corner_kwargs_with_range = DEFAULT_CORNER_KWARGS.copy()
    use_density = True

    # Define colors for different groups
    colors = ['blue', 'green', 'red', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    group_colors = {}
    
    # Assign colors to groups
    for i, group_name in enumerate(samples_dict.keys()):
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
        kwargs = corner_kwargs_with_range.copy()
        kwargs.update({'color': color, 'hist_kwargs': {'color': color, 'density': use_density}})
        group_kwargs[group_name] = kwargs

    # If this was an injection, locate the truth values
    if injection_run:
        truths_filename = os.path.join(base_path, "injection_parameters.json")
        with open(truths_filename, "r") as f:
            truths_dict = json.load(f)
        truths = [truths_dict[k] for k in params_to_plot]
    else:
        truths = None

    # Create the overlaid corner plot - start with the first available dataset
    fig = None
    
    for group_name, samples in samples_dict.items():
        if fig is None:
            fig = corner.corner(samples, labels=latex_labels, truths=truths, **group_kwargs[group_name])
        else:
            corner.corner(samples, labels=latex_labels, fig=fig, **group_kwargs[group_name])
        
    # Add legend
    if plot_all_params:
        fs = 64
    else:
        fs = 34
        
    x = 0.85
    y = 0.85
    dy = 0.1
    
    for group_name, color in group_colors.items():
        label = group_name.upper() if group_name in ['bns', 'nsbh', 'default'] else group_name
        plt.text(x, y, label, fontsize=fs, color=color, ha='center', va='center', transform=plt.gcf().transFigure)
        y -= dy
    
    # External data loading for comparison (Hauke and Adrian)
    
    if plot_hauke:
        hauke_samples = load_hauke_data(GW_event, params_to_plot, use_em_data=False)
        if hauke_samples is not None:
            hauke_kwargs = corner_kwargs_with_range.copy()
            hauke_kwargs.update({'color': HAUKE_COLOR, 'hist_kwargs': {'color': HAUKE_COLOR, 'density': use_density}})
            corner.corner(hauke_samples, labels=latex_labels, fig=fig, **hauke_kwargs)
            y -= dy
            plt.text(x, y, 'Hauke', fontsize=fs, color=HAUKE_COLOR, ha='center', va='center', transform=plt.gcf().transFigure)
        
    if plot_hauke_EM:
        hauke_em_samples = load_hauke_data(GW_event, params_to_plot, use_em_data=True)
        if hauke_em_samples is not None:
            hauke_kwargs = corner_kwargs_with_range.copy()
            hauke_kwargs.update({'color': HAUKE_EM_COLOR, 'hist_kwargs': {'color': HAUKE_EM_COLOR, 'density': use_density}})
            corner.corner(hauke_em_samples, labels=latex_labels, fig=fig, **hauke_kwargs)
            y -= dy
            plt.text(x, y, 'Hauke (GW+EM)', fontsize=fs, color=HAUKE_EM_COLOR, ha='center', va='center', transform=plt.gcf().transFigure)
        
    if plot_adrian:
        adrian_samples = load_adrian_data(GW_event, params_to_plot)
        if adrian_samples is not None:
            adrian_kwargs = corner_kwargs_with_range.copy()
            adrian_kwargs.update({'color': ADRIAN_COLOR, 'hist_kwargs': {'color': ADRIAN_COLOR, 'density': use_density}})
            corner.corner(adrian_samples, labels=latex_labels, fig=fig, **adrian_kwargs)
            y -= dy
            plt.text(x, y, 'Adrian', fontsize=fs, color=ADRIAN_COLOR, ha='center', va='center', transform=plt.gcf().transFigure)
    
    # Save the figure using new directory structure
    # Create subdirectory name based on fixed parameters
    if comparison_mode == "source":
        # Fixed: population + eos, varying: source
        subdir_name = f"population_{fixed_params['population_type']}_eos_{fixed_params['eos_samples_name']}"
        varying_param = "source"
    elif comparison_mode == "population":
        # Fixed: source + eos, varying: population  
        subdir_name = f"source_{fixed_params['source_type']}_eos_{fixed_params['eos_samples_name']}"
        varying_param = "population"
    elif comparison_mode == "eos":
        # Fixed: population + source, varying: eos
        subdir_name = f"population_{fixed_params['population_type']}_source_{fixed_params['source_type']}"
        varying_param = "eos"
    
    output_dir = os.path.join(base_path, GW_event, "figures", subdir_name)
    
    # Create filename: corner_{varying_param}.pdf (with modifiers)
    save_name = f'corner_{varying_param}' + \
            ('_all' if plot_all_params else '') + \
            ('_default' if plot_default else '') + \
            ('_hauke' if (plot_hauke and GW_event in ["GW170817", "GW190425"]) else '') + \
            ('_haukeEM' if (plot_hauke_EM and GW_event == "GW170817") else '') + \
            ('_adrian' if (plot_adrian and GW_event in ["GW170817", "GW190425"]) else '') + \
            ('_tilde' if convert_lambdas else '') + \
            ('_fast' if fast_plotting else '') + '.pdf'
    
    os.makedirs(output_dir, exist_ok=True)
    full_save_path = os.path.join(output_dir, save_name)
    full_save_path = os.path.abspath(full_save_path)
            
    print(f"\nSaving corner plot to {full_save_path}\n")
    plt.savefig(full_save_path, bbox_inches='tight')
    plt.close()
    
                            
                
def run_all_corner_plots(gw_event: str, args):
    """
    Run all corner plots for the given GW event using the new comparison mode approach.
    This replicates the functionality previously in run_cornerplots.sh.
    """
    base_dir = "../final_results/"
    
    print(f"Generating ALL comparison corner plots for {gw_event}...")
    
    # Source comparison plots (fix population + eos, vary source)
    populations = ["uniform", "gaussian", "double_gaussian", gw_event]
    eos_types = ["radio", "radio_chiEFT", "radio_chiEFT_NICER"]
    
    for population in populations:
        for eos in eos_types:
            print(f"  Source comparison: population={population}, eos={eos}")
            # With lambda conversion
            create_corner_plot(
                GW_event=gw_event,
                comparison_mode="source",
                population_type=population,
                eos_samples_name=eos,
                base_path=base_dir,
                convert_lambdas=args["convert_lambdas"],
                plot_hauke=args["plot_hauke"]
            )
            # Without lambda conversion
            create_corner_plot(
                GW_event=gw_event,
                comparison_mode="source",
                population_type=population,
                eos_samples_name=eos,
                base_path=base_dir,
                convert_lambdas=args["convert_lambdas"],
                plot_hauke=args["plot_hauke"]
            )
    
    # Population comparison plots (fix source + eos, vary population)
    sources = ["bns", "nsbh"]
    
    for source in sources:
        for eos in eos_types:
            print(f"  Population comparison: source={source}, eos={eos}")
            # With lambda conversion
            create_corner_plot(
                GW_event=gw_event,
                comparison_mode="population",
                source_type=source,
                eos_samples_name=eos,
                base_path=base_dir,
                convert_lambdas=True
            )
            # Without lambda conversion
            create_corner_plot(
                GW_event=gw_event,
                comparison_mode="population",
                source_type=source,
                eos_samples_name=eos,
                base_path=base_dir,
                convert_lambdas=False
            )
    
    # EOS comparison plots (fix population + source, vary eos)
    for population in populations:
        for source in sources:
            print(f"  EOS comparison: population={population}, source={source}")
            # With lambda conversion
            create_corner_plot(
                GW_event=gw_event,
                comparison_mode="eos",
                population_type=population,
                source_type=source,
                base_path=base_dir,
                convert_lambdas=True
            )
            # Without lambda conversion
            create_corner_plot(
                GW_event=gw_event,
                comparison_mode="eos",
                population_type=population,
                source_type=source,
                base_path=base_dir,
                convert_lambdas=False
            )
    
    print("")
    print("Corner plot generation complete!")


def main():
    # Load cosmology interpolator once at startup for fast plotting
    print("Initializing cosmology interpolator for fast plotting...")
    load_cosmology_interpolator()
    print("")
    
    parser = argparse.ArgumentParser(description="Create corner plots for GW parameter estimation results")
    parser.add_argument('--gw-event', type=str,
                        help='GW event name (e.g., GW170817)')
    parser.add_argument('--comparison-mode', type=str, default='source',
                        choices=['source', 'population', 'eos'],
                        help='What to compare across (default: source)')
    parser.add_argument('--population-type', type=str, default='uniform',
                        choices=['uniform', 'gaussian', 'double_gaussian', 'GW170817', 'GW190425', 'GW230529'],
                        help='Population type for the analysis (default: uniform)')
    parser.add_argument('--source-type', type=str, default='bns',
                        choices=['bns', 'nsbh', 'default'],
                        help='Source type for the analysis (default: bns)')
    parser.add_argument('--eos-samples-name', type=str, default='radio',
                        choices=['radio', 'radio_chiEFT', 'radio_chiEFT_NICER'],
                        help='EOS samples name (default: radio)')
    parser.add_argument('--base-dir', type=str, default='../final_results/',
                        help='Base directory path (default: ../final_results/)')
    parser.add_argument('--plot-all-params', action='store_true',
                        help='Plot all parameters instead of subset')
    parser.add_argument('--plot-default', action='store_true',
                        help='Include default prior in plot')
    parser.add_argument('--convert-lambdas', action='store_true', default=True,
                        help='Convert lambda_1,lambda_2 to lambda_tilde,delta_lambda_tilde')
    parser.add_argument('--no-convert-lambdas', dest='convert_lambdas', action='store_false',
                        help='Do not convert lambdas')
    parser.add_argument('--plot-hauke', action='store_true',
                        help='Include Hauke\'s results in plot')
    parser.add_argument('--plot-hauke-em', action='store_true',
                        help='Include Hauke\'s GW+EM results in plot')
    parser.add_argument('--plot-adrian', action='store_true',
                        help='Include Adrian\'s results in plot')
    parser.add_argument('--prevent-bns-leakage', action='store_true',
                        help='Prevent BNS leakage by masking negative delta_lambda_tilde')
    parser.add_argument('--fast-plotting', action='store_true', default=True,
                        help='Enable fast plotting mode (use cosmology interpolator for speed)')
    parser.add_argument('--no-fast-plotting', dest='fast_plotting', action='store_false',
                        help='Disable fast plotting mode (use exact cosmology calculations)')
    parser.add_argument('--run-all', action='store_true',
                        help='Run all corner plots (equivalent to the old bash script)')
    
    args = parser.parse_args()
    
    if args.run_all:
        run_all_corner_plots(args.gw_event, args.__dict__)
    elif args.gw_event:
        create_corner_plot(
            GW_event=args.gw_event,
            comparison_mode=args.comparison_mode,
            population_type=args.population_type,
            source_type=args.source_type,
            eos_samples_name=args.eos_samples_name,
            base_path=args.base_dir,
            plot_all_params=args.plot_all_params,
            plot_default=args.plot_default,
            convert_lambdas=args.convert_lambdas,
            plot_hauke=args.plot_hauke,
            plot_hauke_EM=args.plot_hauke_em,
            plot_adrian=args.plot_adrian,
            prevent_bns_leakage=args.prevent_bns_leakage,
            fast_plotting=args.fast_plotting
        )
    else:
        # If no specific arguments provided, run all corner plots
        print("No specific GW event provided. Running ALL corner plots...")
        run_all_corner_plots("GW170817", args.__dict__)
        run_all_corner_plots("GW190425", args.__dict__)
        run_all_corner_plots("GW230529", args.__dict__)
            
if __name__ == "__main__":
    main()