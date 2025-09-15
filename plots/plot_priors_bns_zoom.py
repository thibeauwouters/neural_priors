import os
import numpy as np
import matplotlib.pyplot as plt
import corner
import utils
from bilby.gw.conversion import component_masses_to_chirp_mass
from bilby.gw.conversion import lambda_1_lambda_2_to_lambda_tilde, lambda_1_lambda_2_to_delta_lambda_tilde

fs = 24
params = {"axes.grid": False,
        "text.usetex" : True,
        "font.family" : "serif",
        "ytick.color" : "black",
        "xtick.color" : "black",
        "axes.labelcolor" : "black",
        "axes.edgecolor" : "black",
        "font.serif" : ["Computer Modern Serif"],
        "xtick.labelsize": fs,
        "ytick.labelsize": fs,
        "axes.labelsize": fs,
        "legend.fontsize": fs,
        "legend.title_fontsize": fs,
        "figure.titlesize": fs}

plt.rcParams.update(params)

# Improved corner kwargs
default_corner_kwargs = dict(bins=40, 
                        smooth=1., 
                        show_titles=False,
                        label_kwargs=dict(fontsize=32),
                        title_kwargs=dict(fontsize=46), 
                        # quantiles=[],
                        # levels=[0.393, 0.675, 0.95],
                        plot_density=False, 
                        plot_datapoints=False, 
                        fill_contours=True,
                        max_n_ticks=3, 
                        min_n_ticks=2,
                        save=False,
                        labelpad=0.05,
                        )




def get_training_data_path(population_name: str,
                           source_type: str,
                           eos_samples_name: str,
                           ) -> str:
    """
    Get the path to the training data file based on the population name, source type, and EOS samples name.

    Args:
        population_name (str): Name of the population (e.g., "uniform", "gaussian", "double_gaussian").
        source_type (str): Name of the source type (e.g., "bns", "nsbh").
        eos_samples_name (str): Name of the EOS samples (e.g., "radio", "radio_chiEFT", "radio_chiEFT_NICER").

    Raises:
        FileNotFoundError: In case the training data file does not exist.

    Returns:
        str: Path to the training data file.
    """
    path = os.path.join(os.path.dirname(__file__), f"../NFprior/models/{population_name}/{source_type}/{eos_samples_name}/training_data.npz")
    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Training data file does not exist: {path}")
    return path

def make_conversions(data: np.ndarray) -> None:
    """
    Add the lambda_tilde parameter to the data if it is not already present.

    Args:
        data (np.ndarray): Array of shape (n_samples, n_parameters) containing the samples.
    """
    m1, m2 = data["m1"], data["m2"]
    lambda_1, lambda_2 = data["lambda_1"], data["lambda_2"]
    
    lambda_tilde = lambda_1_lambda_2_to_lambda_tilde(lambda_1, lambda_2, m1, m2)
    delta_lambda_tilde = lambda_1_lambda_2_to_delta_lambda_tilde(lambda_1, lambda_2, m1, m2)
    
    data["lambda_tilde"] = lambda_tilde
    data["delta_lambda_tilde"] = delta_lambda_tilde
    
    if "chirp_mass_source" not in list(data.keys()):
        chirp_mass_source = component_masses_to_chirp_mass(m1, m2)
        data["chirp_mass_source"] = chirp_mass_source
        
    if "mass_ratio" not in list(data.keys()):
        mass_ratio = m2 / m1
        data["mass_ratio"] = mass_ratio
    
    return data

def get_ranges(data: np.ndarray,
               lower_quantile: float=0.01,
               upper_quantile: float=0.99) -> list:
    """
    Get the ranges of the data for each parameter using the quantiles to make it flexible but not dominated by outliers.

    Args:
        data (np.ndarray): Array of shape (n_samples, n_parameters) containing the samples.
        lower_quantile (float): Lower quantile for range (default: 0.01).
        upper_quantile (float): Upper quantile for range (default: 0.99).

    Returns:
        list: A list containing the (min, max) values for each parameter.
    """
    ranges = []
    for i in range(data.shape[1]):
        min_val = np.quantile(data[:, i], lower_quantile)
        max_val = np.quantile(data[:, i], upper_quantile)
        ranges.append((min_val, max_val))
    return ranges

def single_plot_bns_zoom(pop: str,
                         convert_masses: bool=True,
                         convert_to_lambda_tilde: bool=True,
                         add_legend: bool=False,
                         add_title: bool=False,
                         ranges: list[tuple] = None,
                         normalization_indices: list = None) -> None:
    """
    Create a corner plot for BNS prior distributions with q > 0.9 zoom.
    
    Args:
        pop (str): Population name (e.g., "uniform", "gaussian", "double_gaussian").
        convert_masses (bool): If True, use chirp mass and mass ratio. If False, use component masses.
        convert_to_lambda_tilde (bool): If True, convert to lambda_tilde and delta_lambda_tilde.
        add_legend (bool): If True, add legend to the plot.
        add_title (bool): If True, add title to the plot.
        ranges (list[tuple], optional): Range for each parameter in the form of a list of tuples (min, max).
        normalization_indices (list, optional): List of dataset indices for histogram normalization.
    """
    
    source_type = "bns"  # Fixed to BNS only
    
    if convert_to_lambda_tilde and source_type == "nsbh":
        print(f"Not doing lambda tilde conversion for NSBH")
        convert_to_lambda_tilde = False
        
    
    # Use single EOS color
    eos_colors = [utils.EOS_COLORS["radio_chiEFT"]]
    
    # Determine which parameters to fetch for the plotting
    if convert_masses:
        mass_keys = ["chirp_mass_source", "mass_ratio"]
    else:
        mass_keys = ["m1", "m2"]
    if convert_to_lambda_tilde:
        lambda_keys = ["lambda_tilde", "delta_lambda_tilde"]
    else:
        lambda_keys = ["lambda_1", "lambda_2"]
    
    keys = mass_keys + lambda_keys
    labels = [utils.TEX_TRANSLATION_DICT[key] for key in keys]
    
    # Plot only one EOS for cleaner visualization
    eos_samples_name_list = ["radio_chiEFT"]
    
    # ==============================================
    # LOOP 1: DATA LOADING AND FILTERING
    # ==============================================
    datasets = {}  # Dictionary to store all loaded datasets
    
    for i, eos_samples_name in enumerate(eos_samples_name_list):
        # Load the corner plot data
        path = get_training_data_path(pop, source_type, eos_samples_name)
        data = dict(np.load(path))
        data = make_conversions(data)
        
        # Apply physical constraints and mass ratio filter
        initial_count = len(data["mass_ratio"])
        
        # Remove samples where lambda_1 > lambda_2 (BNS ordering constraint)
        lambda_constraint_mask = data["lambda_1"] <= data["lambda_2"]
        
        # Remove samples where q > 1 (mass ordering constraint: m2 <= m1)
        mass_constraint_mask = data["mass_ratio"] <= 1.0
        
        # Remove samples with negative delta_lambda_tilde
        delta_lambda_constraint_mask = data["delta_lambda_tilde"] >= 0.0
        
        # Apply mass ratio filter for BNS zoom (q > 0.95)
        mass_ratio_mask = data["mass_ratio"] > 0.95
        
        # Combine all masks
        combined_mask = lambda_constraint_mask & mass_constraint_mask & delta_lambda_constraint_mask & mass_ratio_mask
        
        print(f"Filtering {eos_samples_name}:")
        print(f"  Initial samples: {initial_count}")
        print(f"  After lambda_1 <= lambda_2: {np.sum(lambda_constraint_mask)}")
        print(f"  After q <= 1: {np.sum(mass_constraint_mask & lambda_constraint_mask)}")
        print(f"  After delta_lambda_tilde >= 0: {np.sum(mass_constraint_mask & lambda_constraint_mask & delta_lambda_constraint_mask)}")
        print(f"  After q > 0.95: {np.sum(combined_mask)}")
        
        # Filter all data arrays
        for key in data.keys():
            data[key] = data[key][combined_mask]
        
        # Fetch the relevant data for the corner plot
        corner_data = [data[k] for k in keys]
        corner_data = np.array(corner_data).T
        
        # Store in datasets dictionary
        datasets[eos_samples_name] = corner_data
            
        # Determine ranges based on first dataset only using quantiles
        if ranges is None:
            print(f"Computing ranges for filtered data using 0.01-0.99 quantiles")
            ranges = get_ranges(corner_data, lower_quantile=0.01, upper_quantile=0.99)
            print(f"Computed quantile ranges: {ranges}")
            
        # Debug: Print actual data ranges for this dataset
        if i == 0:
            print(f"Actual data ranges for {eos_samples_name}:")
            for j, key in enumerate(keys):
                print(f"  {key}: [{corner_data[:, j].min():.3f}, {corner_data[:, j].max():.3f}] (full range)")
                print(f"  {key}: [{ranges[j][0]:.3f}, {ranges[j][1]:.3f}] (0.01-0.99 quantiles)")
    
    # No dummy dataset needed for single EOS plot
            
    # ==============================================
    # LOOP 2: PLOTTING
    # ==============================================
    fig = None
    
    # Plot single dataset
    corner_data = datasets[eos_samples_name_list[0]]
    fig = corner.corner(corner_data, 
                labels=labels,
                range=ranges,
                color=eos_colors[0],
                **default_corner_kwargs)
            
    # Add title if requested
    if add_title:
        title_map = {"uniform": "Uniform",
                     "gaussian": "Gaussian",
                     "double_gaussian": "Double Gaussian"}
        plt.suptitle(f"{title_map[pop]} (BNS, q > 0.9)", fontsize=32)
    
    # Add EOS label if requested
    if add_legend:
        x = 0.55
        y = 0.85
        plt.text(x, y, 
                utils.EOS_SAMPLES_NAMES_DICT[eos_samples_name_list[0]], 
                color=eos_colors[0],
                fontsize=32,
                transform=plt.gcf().transFigure)
    
    # Save the figure
    if convert_masses:
        save_name = f"./figures/priors/bns_{pop}_priors_chirp_zoom.pdf"
    else:
        save_name = f"./figures/priors/bns_{pop}_priors_component_zoom.pdf"
        
    if convert_to_lambda_tilde:
        save_name = save_name.replace(".pdf", "_tilde.pdf")
        
    print(f"Saving figure to {save_name}")
    plt.savefig(save_name, bbox_inches="tight")
    if os.path.exists("../../paper/Figures"):
        print(f"Also saving to the Overleaf paper repo")
        save_name = save_name.replace("./figures/priors/", "../../paper/Figures/")
        plt.savefig(save_name, bbox_inches="tight")
    plt.close()

def main():
    
    convert_to_lambda_tilde = True
    
    for pop in utils.POPULATION_NAMES:
        print(f"Plotting BNS zoom priors for {pop}...")
        
        # Let ranges be computed automatically from data using quantiles
        ranges = None
        
        single_plot_bns_zoom(pop,
                            add_legend=pop=="uniform", # only add the legend on the "uniform" population plot
                            add_title=False,
                            convert_masses=True,
                            convert_to_lambda_tilde=convert_to_lambda_tilde,
                            ranges=ranges,
                            normalization_indices=None  # Not needed for single EOS
                            )
        
    
if __name__ == "__main__":
    main()