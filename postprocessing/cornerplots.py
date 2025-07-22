import os
import json
import numpy as np
import matplotlib.pyplot as plt
import corner

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
BNS_COLOR = 'red'
NSBH_COLOR = 'green'

def create_corner_plot(GW_event: str,
                       plot_all_params: bool = False):
    """
    Create a corner plot comparing posterior samples from BNS, Default, and NSBH runs for a given GW event.
    """
    
    # Load result files to investigate structure
    base_path = f"../GW_runs/{GW_event}"
    bns_results_filename = os.path.join(base_path, "bns/bns_result.json")
    default_results_filename = os.path.join(base_path, "default/default_result.json")
    nsbh_results_filename = os.path.join(base_path, "nsbh/nsbh_result.json")

    # Load posterior samples
    with open(bns_results_filename, "r") as f:
        bns_result = json.load(f)
        bns_posterior = bns_result['posterior']['content']

    with open(default_results_filename, "r") as f:
        default_result = json.load(f)
        default_posterior = default_result['posterior']['content']

    with open(nsbh_results_filename, "r") as f:
        nsbh_result = json.load(f)
        nsbh_posterior = nsbh_result['posterior']['content']

    # Define parameters to plot (excluding log_likelihood and log_prior)
    if plot_all_params:
        params_to_plot = ['chirp_mass', 'mass_ratio', 'luminosity_distance', 'geocent_time', 
                        'a_1', 'a_2', 'tilt_1', 'tilt_2', 'phi_12', 'phi_jl', 
                        'dec', 'ra', 'theta_jn', 'psi', 'phase', 'lambda_1', 'lambda_2']
    else:
        params_to_plot = ['chirp_mass', 'mass_ratio', 'luminosity_distance', 'geocent_time', 
                        'a_1', 'a_2', 'lambda_1', 'lambda_2']

    # Create data arrays for each run
    bns_data = []
    default_data = []
    nsbh_data = []
    labels = []

    for param in params_to_plot:
        if param in bns_posterior and param in default_posterior and param in nsbh_posterior:
            bns_data.append(bns_posterior[param])
            default_data.append(default_posterior[param])
            nsbh_data.append(nsbh_posterior[param])
            labels.append(param)

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
        all_vals = np.concatenate([bns_samples[:, i], default_samples[:, i], nsbh_samples[:, i]])
        
        if param == 'lambda_1' and np.std(nsbh_samples[:, i]) < 1e-10:
            # For constant lambda_1 in NSBH, use range from BNS and Default only
            non_zero_vals = np.concatenate([bns_samples[:, i], default_samples[:, i]])
            param_range = (np.min(non_zero_vals), np.max(non_zero_vals))
        else:
            # Use full range for all other parameters
            param_range = (np.min(all_vals), np.max(all_vals))
        
        ranges.append(param_range)

    # Create corner plot with three overlaid distributions
    corner_kwargs_with_range = default_corner_kwargs.copy()
    corner_kwargs_with_range['range'] = ranges

    # Create three different corner kwargs with different colors
    default_kwargs = corner_kwargs_with_range.copy()
    default_kwargs.update({'color': DEFAULT_COLOR, 'alpha': 0.7, 'hist_kwargs': {'alpha': 0.7, 'color': DEFAULT_COLOR}})

    bns_kwargs = corner_kwargs_with_range.copy()  
    bns_kwargs.update({'color': BNS_COLOR, 'alpha': 0.7, 'hist_kwargs': {'alpha': 0.7, 'color': BNS_COLOR}})

    nsbh_kwargs = corner_kwargs_with_range.copy()
    nsbh_kwargs.update({'color': NSBH_COLOR, 'alpha': 0.7, 'hist_kwargs': {'alpha': 0.7, 'color': NSBH_COLOR}})

    print(labels)

    # Create the overlaid corner plot
    fig = corner.corner(default_samples, labels=labels, **default_kwargs)
    corner.corner(bns_samples, labels=labels, fig=fig, **bns_kwargs)
    corner.corner(nsbh_samples, labels=labels, fig=fig, **nsbh_kwargs)

    # Add legend
    if plot_all_params:
        fs = 26
    else:
        fs = 26
        

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
    # Example usage
    GW_event_list = ["GW170817", "GW190425", "GW230529"]
    for GW_event in GW_event_list:
        print(f"Creating corner plot for {GW_event}...")
        try:
            create_corner_plot(GW_event, plot_all_params=False)
        except Exception as e:
            print(f"Failed for {GW_event}: {e}")
        print(f"Corner plot for {GW_event} saved.")
        
if __name__ == "__main__":
    main()
