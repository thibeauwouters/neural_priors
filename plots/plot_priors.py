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
                        fill_contours=False,
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
               quantile: float=0.02) -> list:
    """
    Get the ranges of the data for each parameter using the quantiles to make it flexible but not dominated by outliers.

    Args:
        data (np.ndarray): Array of shape (n_samples, n_parameters) containing the samples.

    Returns:
        tuple: A tuple containing the minimum and maximum values for each parameter.
    """
    ranges = []
    for i in range(data.shape[1]):
        min_val = np.quantile(data[:, i], quantile)
        max_val = np.quantile(data[:, i], 1-quantile)
        ranges.append((min_val, max_val))
    return ranges

def single_plot(pop: str,
                source_type: str,
                convert_masses: bool=True,
                convert_to_lambda_tilde: bool=True,
                add_legend: bool=False,
                add_title: bool=False,
                ranges: list[tuple] = None,
                normalization_indices: list = None) -> None:
    """
    Create a corner plot for prior distributions across different EOS samples.
    
    Args:
        pop (str): Population name (e.g., "uniform", "gaussian", "double_gaussian").
        source_type (str): Source type ("bns" or "nsbh").
        convert_masses (bool): If True, use chirp mass and mass ratio. If False, use component masses.
        convert_to_lambda_tilde (bool): If True, convert to lambda_tilde and delta_lambda_tilde.
        add_legend (bool): If True, add legend to the plot.
        add_title (bool): If True, add title to the plot.
        ranges (list[tuple], optional): Range for each parameter in the form of a list of tuples (min, max).
        normalization_indices (list, optional): List of dataset indices for histogram normalization.
            Must have the same length as the number of parameters. Each index specifies which
            dataset (0=radio_chiEFT, 1=radio, 2=radio_NICER) to use for that parameter when
            creating the normalization dummy dataset. If None, all datasets are concatenated
            for normalization. This fixes histogram scaling issues in corner plots.
            
    Example:
        # Use radio_chiEFT for first param, radio for second param, radio_NICER for third param
        single_plot("uniform", "bns", normalization_indices=[0, 1, 2])
    """
    
    if convert_to_lambda_tilde and source_type == "nsbh":
        print(f"Not doing lambda tilde conversion for NSBH")
        convert_to_lambda_tilde = False
        
    
    # Use consistent EOS colors (colorblind-friendly) as in money_plots.py
    eos_colors = [utils.EOS_COLORS["radio"], 
                  utils.EOS_COLORS["radio_chiEFT"], 
                  utils.EOS_COLORS["radio_NICER"]]
    
    # Determine which parameters to fetch for the plotting
    if convert_masses:
        mass_keys = ["chirp_mass_source", "mass_ratio"]
    else:
        mass_keys = ["m1", "m2"]
    if convert_to_lambda_tilde:
        lambda_keys = ["lambda_tilde", "delta_lambda_tilde"]
    else:
        if source_type == "bns":
            lambda_keys = ["lambda_1", "lambda_2"]
        elif source_type == "nsbh":
            lambda_keys = ["lambda_2"]
    
    keys = mass_keys + lambda_keys
    labels = [utils.TEX_TRANSLATION_DICT[key] for key in keys]
    
    eos_samples_name_list = ["radio",
                             "radio_chiEFT",
                             "radio_NICER"
                             ]
    
    # ==============================================
    # LOOP 1: DATA LOADING
    # ==============================================
    datasets = {}  # Dictionary to store all loaded datasets
    
    for i, eos_samples_name in enumerate(eos_samples_name_list):
        # Load the corner plot data
        path = get_training_data_path(pop, source_type, eos_samples_name)
        data = dict(np.load(path))
        data = make_conversions(data)
        
        # Fetch the relevant data for the corner plot
        corner_data = [data[k] for k in keys]
        corner_data = np.array(corner_data).T
        
        # Store in datasets dictionary
        datasets[eos_samples_name] = corner_data
            
        # Determine ranges based on first dataset only
        if ranges is None:
            print(f"Computing ranges")
            ranges = get_ranges(corner_data)
    
    # ==============================================
    # DUMMY DATASET CONSTRUCTION FOR NORMALIZATION
    # ==============================================
    dummy_dataset = None
    if len(datasets) > 1 and normalization_indices is not None:
        # Validate normalization_indices
        n_params = len(keys)
        if len(normalization_indices) != n_params:
            raise ValueError(f"normalization_indices must have length {n_params} (number of parameters), got {len(normalization_indices)}")
        
        # Check that all indices are valid
        n_datasets = len(eos_samples_name_list)
        for idx in normalization_indices:
            if not (0 <= idx < n_datasets):
                raise ValueError(f"normalization_indices must contain values between 0 and {n_datasets-1}, got {idx}")
        
        # Create dummy dataset by selecting specified dataset for each parameter
        dummy_columns = []
        for param_idx, dataset_idx in enumerate(normalization_indices):
            dataset_name = eos_samples_name_list[dataset_idx]
            dataset = datasets[dataset_name]
            dummy_columns.append(dataset[:, param_idx])
        
        dummy_dataset = np.column_stack(dummy_columns)
        print(f"Created dummy dataset using indices {normalization_indices}")
            
    # ==============================================
    # LOOP 2: PLOTTING
    # ==============================================
    fig = None
    
    # Plot actual datasets first
    for i, eos_samples_name in enumerate(eos_samples_name_list):
        corner_data = datasets[eos_samples_name]
        
        # Create the corner plots
        if i == 0:
            # First plot
            fig = corner.corner(corner_data, 
                        labels=labels,
                        range=ranges,
                        color=eos_colors[i],
                        **default_corner_kwargs)
        else:
            # Plot on existing figure
            corner.corner(corner_data, 
                          labels=labels,
                          range=ranges,
                          color=eos_colors[i],
                          fig=fig,
                          **default_corner_kwargs)
    
    # Plot invisible dummy dataset LAST for histogram normalization
    if dummy_dataset is not None:
        # Create invisible corner kwargs (zero alpha for all visual elements)
        invisible_kwargs = default_corner_kwargs.copy()
        invisible_kwargs.update({
            'hist_kwargs': {'alpha': 0},
            'contour_kwargs': {'alpha': 0},
            'contourf_kwargs': {'alpha': 0},
            'color': 'black'  # Color doesn't matter since alpha=0
        })
        
        print(f"Plotting invisible dummy dataset LAST for histogram normalization")
        corner.corner(dummy_dataset, 
                      labels=labels,
                      range=ranges,
                      fig=fig,
                      **invisible_kwargs)
            
    # Add title if requested
    if add_title:
        title_map = {"uniform": "Uniform",
                     "gaussian": "Gaussian",
                     "double_gaussian": "Double Gaussian"}
        plt.suptitle(title_map[pop], fontsize=32)
    
    # Put texts for the legend
    dy = 0.05 if source_type == "bns" else 0.08
    if add_legend:
        x = 0.55
        y = 0.85
        for i, eos_samples_name in enumerate(eos_samples_name_list):
            plt.text(x, y - i*dy, 
                    utils.EOS_SAMPLES_NAMES_DICT[eos_samples_name], 
                    color=eos_colors[i],
                    fontsize=32,
                    transform=plt.gcf().transFigure)
    
    # Save the figure
    if convert_masses:
        save_name = f"./figures/priors/{source_type}_{pop}_priors_chirp.pdf"
    else:
        save_name = f"./figures/priors/{source_type}_{pop}_priors_component.pdf"
        
    if convert_to_lambda_tilde:
        save_name = save_name.replace(".pdf", "_tilde.pdf")
        
    print(f"Saving figure to {save_name}")
    plt.savefig(save_name, bbox_inches="tight")
    if os.path.exists("../../paper/Figures"):
        print(f"Also saving to the Overleaf paper repo")
        save_name = save_name.replace("./figures/priors/", "../../paper/Figures/")
        plt.savefig(save_name, bbox_inches="tight")
    plt.close()

def scatter_plot_mass_ratio_vs_lambda(pop: str,
                                      source_type: str) -> None:
    """
    Create individual 2D scatter plots of mass ratio vs tidal parameter for each EOS sample.
    For BNS: mass ratio vs delta lambda tilde
    For NSBH: mass ratio vs lambda_2 (NS tidal deformability)
    
    Args:
        pop (str): Population name (e.g., "uniform", "gaussian", "double_gaussian").
        source_type (str): Source type ("bns" or "nsbh").
    """
        
    # Use consistent EOS colors (colorblind-friendly)
    eos_colors = [utils.EOS_COLORS["radio"], 
                  utils.EOS_COLORS["radio_chiEFT"], 
                  utils.EOS_COLORS["radio_NICER"]]
    
    # eos_samples_name_list = ["radio", "radio_chiEFT", "radio_NICER"]
    eos_samples_name_list = ["radio"]
    
    # Determine parameter name
    param_name = "delta_lambda_tilde"
    
    # Create individual plots for each EOS
    for i, eos_samples_name in enumerate(eos_samples_name_list):
        # Create figure for this EOS
        plt.figure(figsize=(8, 6))
        
        # Load the data
        path = get_training_data_path(pop, source_type, eos_samples_name)
        data = dict(np.load(path))
        data = make_conversions(data)
        
        # Extract mass ratio and appropriate tidal parameter
        mass_ratio = data["mass_ratio"]
        tidal_param = data["delta_lambda_tilde"]
        
        # Create scatter plot with low alpha to show density
        plt.scatter(mass_ratio, tidal_param, 
                   color=eos_colors[i], 
                   alpha=0.005, 
                   s=5)
        
        # Set labels
        plt.xlabel(utils.TEX_TRANSLATION_DICT["mass_ratio"], fontsize=fs)
        plt.ylabel(utils.TEX_TRANSLATION_DICT[param_name], fontsize=fs)
        
        # Add title with EOS name
        plt.title(utils.EOS_SAMPLES_NAMES_DICT[eos_samples_name], fontsize=fs)
        
        # Set reasonable limits based on source type
        if source_type == "bns":
            plt.xlim(0.4, 1.0)
            if pop == "uniform":
                plt.ylim(-500, 500)
            elif pop == "gaussian":
                plt.xlim(0.99, 1.0)
                plt.ylim(0.0, 20.0)
            else:
                plt.ylim(None, None)  # Let matplotlib choose
        else:  # nsbh
            plt.xlim(0.2, 1.0)
            if pop == "uniform":
                plt.ylim(0, 2000)
            else:
                plt.ylim(None, None)  # Let matplotlib choose
        
        plt.tight_layout()
        
        # Save the figure for this EOS
        os.makedirs("./figures/priors/scatterplots", exist_ok=True)
        save_name = f"./figures/priors/scatterplots/{source_type}_{pop}_{eos_samples_name}_mass_ratio_vs_{param_name}_scatter.pdf"
        print(f"Saving scatter plot to {save_name}")
        plt.savefig(save_name, bbox_inches="tight")
        if os.path.exists("../../paper/Figures"):
            print(f"Also saving to the Overleaf paper repo")
            save_name_paper = save_name.replace("./figures/priors/scatterplots/", "../../paper/Figures/")
            plt.savefig(save_name_paper, bbox_inches="tight")
        plt.close()

def main():
    
    convert_to_lambda_tilde = True
    
    for pop in utils.POPULATION_NAMES:
        for source_type in ["bns", "nsbh"]:
            print(f"Plotting priors for {pop} {source_type}...")
            
            if pop == "uniform" and source_type == "bns" and not convert_to_lambda_tilde:
                ranges = [[0.9, 2.1],
                          [0.425, 1.0],
                          [0.0, 700.0],
                          [0.0, 4_500.0],
                ]
            elif pop == "uniform" and source_type == "bns" and convert_to_lambda_tilde:
                ranges = [[0.9, 2.1],
                          [0.425, 1.0],
                          [0.0, 1000.0],
                          [0.0, 400.0],
                ]
            elif pop == "uniform" and source_type == "nsbh" and not convert_to_lambda_tilde:
                ranges = [[1.25, 3.0],
                          [0.20, 1.0],
                          [0.0, 2_000.0],
                ]
            elif pop == "uniform" and source_type == "nsbh" and convert_to_lambda_tilde:
                ranges = [[1.25, 3.0],
                          [0.20, 1.0],
                          [0.0, 1000.0],
                        #   [0.0, 400.0],
                ]
            else:
                ranges = None
            
            single_plot(pop,
                        source_type,
                        add_legend=pop=="uniform", # only add the legend on the "uniform" population plot
                        add_title=False,
                        convert_masses=True,
                        convert_to_lambda_tilde=convert_to_lambda_tilde,
                        ranges=ranges,
                        normalization_indices=[1, 1, 1, 1] if source_type == "bns" else [1, 1, 1]
                        )
            
            # Create scatter plot for mass ratio vs tidal parameter
            scatter_plot_mass_ratio_vs_lambda(pop, source_type)
            
    
if __name__ == "__main__":
    main()