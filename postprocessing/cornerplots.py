import os
import json
import arviz
import numpy as np
import matplotlib.pyplot as plt
import corner

from bilby.gw.conversion import lambda_1_lambda_2_to_lambda_tilde, lambda_1_lambda_2_to_delta_lambda_tilde
from bilby.gw.conversion import chirp_mass_and_mass_ratio_to_component_masses

params = {"axes.grid": True,
        "text.usetex" : False,
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
                        truth_color = "red",
                        save=False)

DEFAULT_COLOR = 'blue'
BNS_COLOR = 'green'
NSBH_COLOR = 'red'

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
              "delta_lambda_tilde": r"$\Delta \tilde{\Lambda}$"
              }

def create_corner_plot(GW_event: str,
                       plot_all_params: bool = False,
                       plot_default: bool = False,
                       convert_lambdas: bool = True):
    """
    Create a corner plot comparing posterior samples from BNS, Default, and NSBH runs for a given GW event.
    """
    
    # Load result files to investigate structure
    base_path = f"../GW_runs/{GW_event}"
    if GW_event == "GW170817":
        base_path = "/data/gravwav/twouters/projects/eos_source_classification/eos_source_classification_gitlab/GW_runs/GW170817/"
        print("Using base path for GW170817:", base_path)
    
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
        
        print("params_to_plot")
        print(params_to_plot)

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
    
    print(f"BNS samples shape: {bns_samples.shape}")
    print(f"Default samples shape: {default_samples.shape}")
    print(f"NSBH samples shape: {nsbh_samples.shape}")

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
        
    print("ranges")
    print(ranges)

    # Create corner plot with three overlaid distributions
    corner_kwargs_with_range = default_corner_kwargs.copy()
    # corner_kwargs_with_range['range'] = ranges
    use_density = True

    # Create three different corner kwargs with different colors
    default_kwargs = corner_kwargs_with_range.copy()
    default_kwargs.update({'color': DEFAULT_COLOR, 'hist_kwargs': {'color': DEFAULT_COLOR, 'density': use_density}})

    bns_kwargs = corner_kwargs_with_range.copy()  
    bns_kwargs.update({'color': BNS_COLOR, 'hist_kwargs': {'color': BNS_COLOR, 'density': use_density}})

    nsbh_kwargs = corner_kwargs_with_range.copy()
    nsbh_kwargs.update({'color': NSBH_COLOR, 'hist_kwargs': {'color': NSBH_COLOR, 'density': use_density}})

    print("latex_labels")
    print(latex_labels)

    # # Concatenate all samples to create a dummy cornerplot
    # if plot_default:
    #     dummy_samples = np.concatenate([default_samples, bns_samples, nsbh_samples], axis=0)
    # else:
    #     dummy_samples = np.concatenate([bns_samples, nsbh_samples], axis=0)
        
    # # Create a dummy corner plot to set the figure size, make sure this plot is not visible
    # dummy_corner_kwargs = corner_kwargs_with_range.copy()
    # dummy_corner_kwargs["plot_density"] = False
    # dummy_corner_kwargs["fill_contours"] = False
    # dummy_corner_kwargs["contourf_kwargs"] = {"alpha": 0.0}
    # dummy_corner_kwargs["contour_kwargs"] = {"alpha": 0.0}
    # dummy_corner_kwargs["hist_kwargs"] = {"alpha": 0.0, "density": use_density, "color": "white"}
    
    # # Finally, add dummy samples to scale appropriately
    # fig = corner.corner(dummy_samples, labels=latex_labels, **dummy_corner_kwargs)
    
    # Create the overlaid corner plot
    fig = corner.corner(bns_samples, labels=latex_labels, **bns_kwargs)
    weights = np.ones(len(nsbh_samples))*len(bns_samples)/len(nsbh_samples)  
    corner.corner(nsbh_samples, labels=latex_labels, fig=fig, weights=weights, **nsbh_kwargs)
    if plot_default:
        weights = np.ones(len(default_samples))*len(bns_samples)/len(default_samples)  
        corner.corner(default_samples, labels=latex_labels, fig=fig, weights=weights, **default_kwargs)

    # Add legend
    if plot_all_params:
        fs = 26
    else:
        fs = 26
        
    if plot_default:
        plt.text(0.95, 0.95, 'Default', fontsize=fs, color=DEFAULT_COLOR, ha='center', va='center', transform=plt.gcf().transFigure)
    plt.text(0.95, 0.85, 'BNS', fontsize=fs, color=BNS_COLOR, ha='center', va='center', transform=plt.gcf().transFigure)
    plt.text(0.95, 0.75, 'NSBH', fontsize=fs, color=NSBH_COLOR, ha='center', va='center', transform=plt.gcf().transFigure)
    if not plot_all_params:
        save_name = f'./figures/{GW_event}.pdf'
    else:
        save_name = f'./figures/{GW_event}_all.pdf'
    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    plt.close()
    
def main():
    GW_event_list = ["GW170817"
                    #  "GW190425",
                    #  "GW230529",
                     ]
    
    for GW_event in GW_event_list:
        print(f"Creating corner plot for {GW_event}...")
        create_corner_plot(GW_event,
                           plot_all_params=False,
                           plot_default=True,
                           convert_lambdas=True)
            
if __name__ == "__main__":
    main()
