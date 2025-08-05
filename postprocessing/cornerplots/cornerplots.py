import os
import json
import arviz
import numpy as np
import matplotlib.pyplot as plt
import corner
import argparse

from bilby.gw.conversion import lambda_1_lambda_2_to_lambda_tilde, lambda_1_lambda_2_to_delta_lambda_tilde
from bilby.gw.conversion import chirp_mass_and_mass_ratio_to_component_masses
from bilby.gw.conversion import luminosity_distance_to_redshift
from utils import load_comparison_data, construct_result_path, load_posterior_data

params = {"axes.grid": True,
        # "text.usetex" : True,
        "font.family" : "serif",
        "ytick.color" : "black",
        "xtick.color" : "black",
        "axes.labelcolor" : "black",
        "axes.edgecolor" : "black",
        # "font.serif" : ["Computer Modern Serif"],
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "axes.labelsize": 16,
        "legend.fontsize": 16,
        "legend.title_fontsize": 16,
        "figure.titlesize": 16}

plt.rcParams.update(params)

# Improved corner kwargs
default_corner_kwargs = dict(bins=40, 
                        smooth=1., 
                        show_titles=False,
                        label_kwargs=dict(fontsize=16),
                        title_kwargs=dict(fontsize=16), 
                        # quantiles=[],
                        # levels=[0.9],
                        plot_density=True, 
                        plot_datapoints=False, 
                        fill_contours=True,
                        max_n_ticks=4, 
                        min_n_ticks=3,
                        truth_color = "black",
                        save=False)

DEFAULT_COLOR = 'blue'
BNS_COLOR = 'green'
NSBH_COLOR = 'red'
HAUKE_COLOR = 'orange'
HAUKE_EM_COLOR = 'purple'
ADRIAN_COLOR = 'black'

LaTeX_dict = {"chirp_mass": r"$\mathcal{M}_c$ [$M_{\odot}$]",
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

def convert_chirp_mass(posterior: dict):
    """
    Convert source-frame chirp mass into detector-frame chirp mass.
    """
    d_L = np.array(posterior['luminosity_distance'])
    z = luminosity_distance_to_redshift(d_L)
    chirp_mass_source = np.array(posterior['chirp_mass_source'])
    posterior['chirp_mass'] = chirp_mass_source * (1 + z)
    return
    

def create_corner_plot(GW_event: str,
                       comparison_mode: str = "source",
                       population_type: str = "uniform",
                       source_type: str = "bns", 
                       eos_samples_name: str = "radio",
                       base_path: str = "../GW_runs/",
                       plot_all_params: bool = False,
                       plot_default: bool = False,
                       convert_lambdas: bool = True,
                       plot_hauke: bool = False,
                       plot_hauke_EM: bool = False,
                       plot_adrian: bool = False,
                       prevent_bns_leakage: bool = False):
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
    posteriors_dict = load_comparison_data(GW_event, base_path, comparison_mode, fixed_params)
    
    # Add default run if requested
    if plot_default:
        default_path = construct_result_path(base_path, GW_event, population_type, "default", eos_samples_name)
        default_posterior = load_posterior_data(default_path)
        if default_posterior:
            posteriors_dict["default"] = default_posterior
        
    print(f"Loaded {len(posteriors_dict)} posterior datasets: {list(posteriors_dict.keys())}")

    # Define parameters to plot (excluding log_likelihood and log_prior)
    if plot_all_params:
        params_to_plot = ['chirp_mass', 'mass_ratio', 'luminosity_distance', 'geocent_time', 
                        'a_1', 'a_2', 'tilt_1', 'tilt_2', 'phi_12', 'phi_jl', 
                        'dec', 'ra', 'theta_jn', 'psi', 'phase', 'lambda_1', 'lambda_2']
    else:
        params_to_plot = ['chirp_mass', 'mass_ratio', 'geocent_time', 'lambda_1', 'lambda_2']
        
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
        
        del params_to_plot[params_to_plot.index('lambda_1')]
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
            latex_labels.append(LaTeX_dict.get(param, param))

    # Create data arrays for each posterior
    for key, posterior in posteriors_dict.items():
        data = []
        for param in labels:
            if param in posterior:
                data.append(posterior[param])
        if data:
            samples_dict[key] = np.array(data).T
    
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
    corner_kwargs_with_range = default_corner_kwargs.copy()
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
    
    # Hauke and Adrian's runs did not sample some params and had them fixed -- jitter a bit to avoid corner complaints
    if GW_event == "GW170817":
        params_to_dummy_replace = ["geocent_time", "phase", "ra", "dec"]
    else:
        params_to_dummy_replace = ["geocent_time", "phase"]
        
    if GW_event in ["GW170817", "GW190425"] and plot_hauke:
        
        hauke_posterior_filename = f"../../data/hauke/{GW_event}/{GW_event}_result.npz"
        print(f"Loading Hauke's data and adding it to the corner plot, filename: {hauke_posterior_filename}")
        hauke_data = np.load(hauke_posterior_filename)
        hauke_data = {k: hauke_data[k] for k in params_to_plot} # convert to dict
        
        for key in params_to_dummy_replace:
            if key in hauke_data:
                print(f"Replacing {key} with dummy values for Hauke...")
                hauke_data[key] = np.random.normal(hauke_data[key], 0.01, size=len(hauke_data["chirp_mass"]))
        
        # Fetch the samples we want to plot
        hauke_samples = np.array([hauke_data[param] for param in params_to_plot]).T
        
        # Make the cornerplot:
        hauke_kwargs = corner_kwargs_with_range.copy()
        hauke_kwargs.update({'color': HAUKE_COLOR, 'hist_kwargs': {'color': HAUKE_COLOR, 'density': use_density}})
        corner.corner(hauke_samples, labels=latex_labels, fig=fig, **hauke_kwargs)
        y -= dy
        plt.text(x, y, 'Hauke', fontsize=fs, color=HAUKE_COLOR, ha='center', va='center', transform=plt.gcf().transFigure)
        
    if GW_event in ["GW170817"] and plot_hauke_EM:
        
        hauke_posterior_filename = f"../../data/hauke/{GW_event}/{GW_event}+EM_result.npz"
        print(f"Loading Hauke's data and adding it to the corner plot, filename: {hauke_posterior_filename}")
        hauke_data = np.load(hauke_posterior_filename)
        hauke_data = {k: hauke_data[k] for k in params_to_plot} # convert to dict
        
        for key in params_to_dummy_replace:
            if key in hauke_data:
                print(f"Replacing {key} with dummy values for Hauke...")
                hauke_data[key] = np.random.normal(hauke_data[key], 0.01, size=len(hauke_data["chirp_mass"]))
        
        # Fetch the samples we want to plot
        hauke_samples = np.array([hauke_data[param] for param in params_to_plot]).T
        
        # Make the cornerplot:
        hauke_kwargs = corner_kwargs_with_range.copy()
        hauke_kwargs.update({'color': HAUKE_EM_COLOR, 'hist_kwargs': {'color': HAUKE_EM_COLOR, 'density': use_density}})
        corner.corner(hauke_samples, labels=latex_labels, fig=fig, **hauke_kwargs)
        y -= dy
        plt.text(x, y, 'Hauke (GW+EM)', fontsize=fs, color=HAUKE_EM_COLOR, ha='center', va='center', transform=plt.gcf().transFigure)
        
    if GW_event in ["GW170817", "GW190425"] and plot_adrian:
        
        print(f"Loading Adrian's data and adding it to the corner plot...")
        adrian_posterior_filename = f"../../data/adrian/{GW_event}/{GW_event}_result.npz"
        adrian_data = np.load(adrian_posterior_filename)
        adrian_data = {k: adrian_data[k] for k in params_to_plot}
        
        for key in params_to_dummy_replace:
            if key in adrian_data:
                print(f"Replacing {key} with dummy values for Adrian...")
                adrian_data[key] = np.random.normal(adrian_data[key], 0.01, size=len(adrian_data["chirp_mass"]))
        
        # Fetch the samples we want to plot
        adrian_samples = np.array([adrian_data[param] for param in params_to_plot]).T
            
        # Make the cornerplot:
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
            ('_tilde' if convert_lambdas else '') + '.pdf'
    
    os.makedirs(output_dir, exist_ok=True)
    full_save_path = os.path.join(output_dir, save_name)
            
    print(f"\nSaving corner plot to {full_save_path}\n")
    plt.savefig(full_save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
                            
                
def run_all_corner_plots():
    """
    Run all corner plots for GW170817 and GW190425 using the new comparison mode approach.
    This replicates the functionality previously in run_cornerplots.sh.
    """
    base_dir = "../GW_runs/"
    gw_events = ["GW170817", "GW190425"]
    
    for gw_event in gw_events:
        print(f"Generating comparison corner plots for {gw_event}...")
        
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
                    convert_lambdas=True,
                    plot_hauke=True
                )
                # Without lambda conversion
                create_corner_plot(
                    GW_event=gw_event,
                    comparison_mode="source",
                    population_type=population,
                    eos_samples_name=eos,
                    base_path=base_dir,
                    convert_lambdas=False,
                    plot_hauke=True
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
    parser.add_argument('--base-dir', type=str, default='../GW_runs/',
                        help='Base directory path (default: ../GW_runs/)')
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
    parser.add_argument('--run-all', action='store_true',
                        help='Run all corner plots (equivalent to the old bash script)')
    parser.add_argument('--batch-mode', action='store_true',
                        help='Run in batch mode (old behavior for testing)')
    
    args = parser.parse_args()
    
    if args.run_all:
        run_all_corner_plots()
    elif args.batch_mode:
        print("Batch mode is deprecated. Use individual calls with comparison modes instead.")
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
            prevent_bns_leakage=args.prevent_bns_leakage
        )
    else:
        # If no specific arguments provided, run all corner plots
        print("No specific GW event provided. Running all corner plots...")
        run_all_corner_plots()
            
if __name__ == "__main__":
    main()