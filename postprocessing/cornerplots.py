import os
import json
import arviz
import numpy as np
import matplotlib.pyplot as plt
import corner

from bilby.gw.conversion import lambda_1_lambda_2_to_lambda_tilde, lambda_1_lambda_2_to_delta_lambda_tilde
from bilby.gw.conversion import chirp_mass_and_mass_ratio_to_component_masses

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

def create_corner_plot(GW_event: str,
                       base_path: str = "../GW_runs/final_results",
                       plot_all_params: bool = False,
                       plot_default: bool = False,
                       convert_lambdas: bool = True,
                       plot_hauke: bool = False,
                       plot_hauke_EM: bool = False,
                       plot_adrian: bool = False,
                       prevent_bns_leakage: bool = False):
    """
    Create a corner plot comparing posterior samples from BNS, Default, and NSBH runs for a given GW event.
    
    Args:
        GW_event (str): The name of the GW event to create the corner plot for.
        plot_all_params (bool): If True, plot all parameters. If False, plot only a subset of parameters.
        plot_default (bool): If True, include the Default run in the corner plot.
        convert_lambdas (bool): If True, convert lambda_1 and lambda_2 to lambda_tilde and delta_lambda_tilde.
        plot_hauke (bool): If True, include Hauke's data in the corner plot.
        plot_hauke_EM (bool): If True, include Hauke's data in the corner plot that analyzed the GW+EM dataset.
        plot_adrian (bool): If True, include Adrian's data in the corner plot.
        prevent_bns_leakage (bool): If True, prevent BNS leakage by masking samples with negative delta_lambda_tilde.
    """
    
    # Check whether this is for real events or from the injection runs
    if "injection" in base_path:
        injection_run = True
    else:
        injection_run = False
    
    # Load result files
    bns_results_filename = os.path.join(base_path, "bns/bns_result.json")
    default_results_filename = os.path.join(base_path, "default/default_result.json")
    nsbh_results_filename = os.path.join(base_path, "nsbh/nsbh_result.json")

    # Load posterior samples
    with open(bns_results_filename, "r") as f:
        bns_result = json.load(f)
        bns_posterior = bns_result['posterior']['content']

    with open(nsbh_results_filename, "r") as f:
        nsbh_result = json.load(f)
        nsbh_posterior = nsbh_result['posterior']['content']
        
    if plot_default:
        with open(default_results_filename, "r") as f:
            default_result = json.load(f)
            default_posterior = default_result['posterior']['content']
            
        posteriors_dict = {"bns": bns_posterior,
                           "default": default_posterior,
                           "nsbh": nsbh_posterior}
    else:
        posteriors_dict = {"bns": bns_posterior,
                           "nsbh": nsbh_posterior}

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
        
    # Create data arrays for each run
    bns_data = []
    default_data = []
    nsbh_data = []
    labels = []
    latex_labels = []

    for param in params_to_plot:
        if param in bns_posterior: # TODO: remove this line?
            bns_data.append(bns_posterior[param])
            nsbh_data.append(nsbh_posterior[param])
            if plot_default:
                default_data.append(default_posterior[param])
            labels.append(param)
            latex_labels.append(LaTeX_dict.get(param, param))

    # Convert to numpy arrays and transpose to get (n_samples, n_params)
    bns_samples = np.array(bns_data).T
    default_samples = np.array(default_data).T  
    nsbh_samples = np.array(nsbh_data).T
    
    # For the BNS samples, in case we have lambda tilde, mask to only have positive delta lambda tilde
    if 'delta_lambda_tilde' in params_to_plot and prevent_bns_leakage:
        delta_lambda_tilde_index = labels.index('delta_lambda_tilde')
        bns_samples = bns_samples[bns_samples[:, delta_lambda_tilde_index] > 0]
    
    # Create range dictionary to handle constant parameters
    ranges = []
    for i, param in enumerate(labels):
        # Get all values for this parameter across all runs
        if plot_default:
            all_vals = np.concatenate([bns_samples[:, i], default_samples[:, i], nsbh_samples[:, i]])
        else:
            all_vals = np.concatenate([bns_samples[:, i], nsbh_samples[:, i]])
        
        if param == 'lambda_1' and np.std(nsbh_samples[:, i]) < 1e-10:
            # For constant lambda_1 in NSBH, use range from BNS and Default only
            if plot_default:
                non_zero_vals = np.concatenate([bns_samples[:, i], default_samples[:, i]])
            else:
                non_zero_vals = [bns_samples[:, i]]
            param_range = (np.min(non_zero_vals), np.max(non_zero_vals))
        else:
            # Use full range for all other parameters
            param_range = (np.min(all_vals), np.max(all_vals))
        
        ranges.append(param_range)
        
    # Make lambda_1 NaN for NSBH samples for plotting
    if 'lambda_1' in params_to_plot:
        lambda_1_index = labels.index('lambda_1')
        # print("Setting those values to NaN")
        # # Set lambda_1 values in NSBH samples to NaN
        # nsbh_samples[:, lambda_1_index]   = np.ones_like(nsbh_samples[:, lambda_1_index]) * np.nan
        
        # Turn into very small jitter around zero instead of NaN to make plot pass
        jitter = 1e-10
        nsbh_samples[:, lambda_1_index] = np.random.normal(0, jitter, size=nsbh_samples[:, lambda_1_index].shape)
        
    # Create corner plot with three overlaid distributions
    corner_kwargs_with_range = default_corner_kwargs.copy()
    # corner_kwargs_with_range['range'] = ranges # FIXME: this breaks the scaling of the plots
    use_density = True

    # Create three different corner kwargs with different colors
    default_kwargs = corner_kwargs_with_range.copy()
    default_kwargs.update({'color': DEFAULT_COLOR, 'hist_kwargs': {'color': DEFAULT_COLOR, 'density': use_density}})

    bns_kwargs = corner_kwargs_with_range.copy()  
    bns_kwargs.update({'color': BNS_COLOR, 'hist_kwargs': {'color': BNS_COLOR, 'density': use_density}})

    nsbh_kwargs = corner_kwargs_with_range.copy()
    nsbh_kwargs.update({'color': NSBH_COLOR, 'hist_kwargs': {'color': NSBH_COLOR, 'density': use_density}})

    # If this was an injection, locate the truth values
    if injection_run:
        truths_filename = os.path.join(base_path, "injection_parameters.json")
        with open(truths_filename, "r") as f:
            truths_dict = json.load(f)
        truths = [truths_dict[k] for k in params_to_plot]
    else:
        truths = None

    # Create the overlaid corner plot
    fig = corner.corner(bns_samples, labels=latex_labels, truths=truths, **bns_kwargs)
    corner.corner(nsbh_samples, labels=latex_labels, fig=fig, **nsbh_kwargs)
    if plot_default:
        corner.corner(default_samples, labels=latex_labels, fig=fig, **default_kwargs)
        
    # Add legend
    if plot_all_params:
        fs = 64
    else:
        fs = 34
        
    x = 0.85
    y = 0.85
    dy = 0.1
    if plot_default:
        plt.text(x, y, 'Default', fontsize=fs, color=DEFAULT_COLOR, ha='center', va='center', transform=plt.gcf().transFigure)
    y -= dy
    plt.text(x, y, 'BNS', fontsize=fs, color=BNS_COLOR, ha='center', va='center', transform=plt.gcf().transFigure)
    y -= dy
    plt.text(x, y, 'NSBH', fontsize=fs, color=NSBH_COLOR, ha='center', va='center', transform=plt.gcf().transFigure)
    
    # Hauke and Adrian's runs did not sample some params and had them fixed -- jitter a bit to avoid corner complaints
    if GW_event == "GW170817":
        params_to_dummy_replace = ["geocent_time", "phase", "ra", "dec"]
    else:
        params_to_dummy_replace = ["geocent_time", "phase"]
        
    if GW_event in ["GW170817", "GW190425"] and plot_hauke:
        
        hauke_posterior_filename = f"../data/hauke/{GW_event}/{GW_event}_result.npz"
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
        
        hauke_posterior_filename = f"../data/hauke/{GW_event}/{GW_event}+EM_result.npz"
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
        adrian_posterior_filename = f"../data/adrian/{GW_event}/{GW_event}_result.npz"
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
    
    # Save the figure
    output_dir = os.path.join(base_path, "figures")
    os.makedirs(output_dir, exist_ok=True)
    save_name = 'corner' + \
            ('_all' if plot_all_params else '') + \
            ('_default' if plot_default else '') + \
            ('_hauke' if (plot_hauke and GW_event in ["GW170817", "GW190425"]) else '') + \
            ('_haukeEM' if (plot_hauke_EM and GW_event == "GW170817") else '') + \
            ('_adrian' if (plot_adrian and GW_event in ["GW170817", "GW190425"]) else '') + \
            ('_tilde' if convert_lambdas else '') + '.pdf'
    save_name = os.path.join(output_dir, f"{save_name}")
            
    print(f"\nSaving corner plot to {save_name}\n")
    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    plt.close()
    
def real_events():
    
    GW_event_list = ["GW170817",
                    #  "GW190425",
                    #  "GW230529",
                    ]
    
    # for plot_all_params in [True, False]:
    for plot_default in [True, False]:
        for plot_adrian in [False]:
            for plot_hauke in [False]:
                for plot_hauke_EM in [False]:
                    for convert_lambdas in [True, False]:
                        for GW_event in GW_event_list:  
                            settings = dict(plot_all_params=False,
                                            plot_default=plot_default,
                                            convert_lambdas=convert_lambdas,
                                            plot_hauke=plot_hauke,
                                            plot_hauke_EM=plot_hauke_EM,
                                            plot_adrian=plot_adrian,
                                            base_path=os.path.join("../GW_runs/", GW_event)
                                            )
                            print(f"Creating corner plot for {GW_event} with settings: {settings}")
                            create_corner_plot(GW_event, **settings)
                            
def injections():
    
    base_path_list = [#"../injections/GW170817_bns_jester/",
                      #"../injections/GW170817_nsbh_jester/"
                      "../injections/GW190425_design_bns_jester/",
                      "../injections/GW190425_design_nsbh_jester/"
                    ]
    
    # for plot_all_params in [True, False]:
    for plot_default in [True, False]:
        for convert_lambdas in [True, False]:
            for base_path in base_path_list:
                
                # Get GW event id from this path
                GW_event = base_path_list[0].split("/")[-2].split("_")[0]
                
                # Build settings
                settings = dict(plot_all_params=False,
                                plot_default=plot_default,
                                convert_lambdas=convert_lambdas,
                                base_path=base_path
                                )
                print(f"Creating corner plot for {base_path} with settings: {settings}")
                create_corner_plot(GW_event, **settings)
                
def main():
    # real_events()
    injections()
            
if __name__ == "__main__":
    main()