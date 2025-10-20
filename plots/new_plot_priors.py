import os
import numpy as np
import matplotlib.pyplot as plt
import corner
import utils
from bilby.gw.conversion import component_masses_to_chirp_mass
from bilby.gw.conversion import lambda_1_lambda_2_to_lambda_tilde, lambda_1_lambda_2_to_delta_lambda_tilde

# Font size
fs = 32

# Matplotlib style parameters
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
        "figure.titlesize": fs
        }

plt.rcParams.update(params)

# Corner plot styling constants
LINEWIDTH = 2.5      # Thickness of lines in corner plots (1D histograms and 2D contours)
LABELPAD = 0.05      # Padding for axis labels in corner plots
SMOOTH = 1.5         # KDE smoothing factor
MIN_N_TICKS = 2      # Minimum number of ticks on axes (passed to corner)
MAX_N_TICKS = 3      # Maximum number of ticks on axes (passed to corner)

# Base corner plot kwargs used across all corner plots
BASE_CORNER_KWARGS = {
    'bins': 40,
    'smooth': SMOOTH,
    'plot_datapoints': False,
    'plot_density': False,
    'min_n_ticks': MIN_N_TICKS,
    'max_n_ticks': MAX_N_TICKS,
    'labelpad': LABELPAD,
}


def get_training_data_path(population_name: str,
                           source_type: str,
                           eos_samples_name: str,
                           ) -> str:
    """
    Get the path to the training data file based on the population name, source type, and EOS samples name.

    Args:
        population_name (str): Name of the population (e.g., "uniform", "gaussian", "double_gaussian").
        source_type (str): Name of the source type (e.g., "bns", "nsbh").
        eos_samples_name (str): Name of the EOS samples (e.g., "radio", "radio_chiEFT", "radio_NICER").

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


def combined_plot(source_type: str,
                  convert_masses: bool = True,
                  convert_to_lambda_tilde: bool = True,
                  levels: list = None,
                  filled_eos: str = "radio",
                  ranges_dict: dict = None,
                  normalization_indices_dict: dict = None) -> None:
    """
    Create a combined figure with three corner plots side-by-side for different populations.

    Args:
        source_type (str): Source type ("bns" or "nsbh").
        convert_masses (bool): If True, use chirp mass and mass ratio. If False, use component masses.
        convert_to_lambda_tilde (bool): If True, convert to lambda_tilde and delta_lambda_tilde.
        levels (list): Confidence levels for contours (e.g., [0.68, 0.95]).
        filled_eos (str): Which EOS to plot with filled contours (default: "radio").
        ranges_dict (dict): Dictionary mapping population names to range lists.
        normalization_indices_dict (dict): Dictionary mapping population names to normalization indices lists.
            Each list specifies which dataset (0=radio, 1=radio_chiEFT, 2=radio_NICER) to use for
            each parameter when creating the normalization dummy dataset. If None, no normalization.
    """

    if convert_to_lambda_tilde and source_type == "nsbh":
        print(f"Not doing lambda tilde conversion for NSBH")
        convert_to_lambda_tilde = False

    if levels is None:
        levels = [0.68, 0.95]

    # Use consistent EOS colors (colorblind-friendly)
    eos_colors = {
        "radio": utils.EOS_COLORS["radio"],
        "radio_chiEFT": utils.EOS_COLORS["radio_chiEFT"],
        "radio_NICER": utils.EOS_COLORS["radio_NICER"]
    }

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

    eos_samples_name_list = ["radio", "radio_chiEFT", "radio_NICER"]
    populations = ["uniform", "gaussian", "double_gaussian"]

    # Create the main figure with three subfigures side-by-side
    fig = plt.figure(figsize=(42, 14))
    subfigs = fig.subfigures(1, 3, wspace=0.05)

    # Iterate over each population
    for pop_idx, pop in enumerate(populations):
        print(f"\nPlotting {pop} population...")

        # Determine ranges for this population
        if ranges_dict is not None and pop in ranges_dict:
            ranges = ranges_dict[pop]
        else:
            ranges = None

        # Determine normalization indices for this population
        normalization_indices = None
        if normalization_indices_dict is not None and pop in normalization_indices_dict:
            normalization_indices = normalization_indices_dict[pop]

        # ==============================================
        # LOOP 1: DATA LOADING
        # ==============================================
        datasets = {}  # Dictionary to store all loaded datasets

        for eos_samples_name in eos_samples_name_list:
            # Load the corner plot data
            path = get_training_data_path(pop, source_type, eos_samples_name)
            data = dict(np.load(path))
            data = make_conversions(data)

            # Fetch the relevant data for the corner plot
            corner_data = [data[k] for k in keys]
            corner_data = np.array(corner_data).T

            # Store in datasets dictionary
            datasets[eos_samples_name] = corner_data

            # Determine ranges based on first dataset only if not provided
            if ranges is None:
                print(f"Computing ranges for {pop}")
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
            print(f"Created dummy dataset for {pop} using indices {normalization_indices}")

        # ==============================================
        # LOOP 2: PLOTTING
        # ==============================================

        # Step 1: Plot the filled contours for the specified EOS first (VERY HIGH zorder)
        print(f"Plotting {filled_eos} with filled contours...")
        filled_color = eos_colors[filled_eos]

        corner_fig = corner.corner(
            datasets[filled_eos],
            labels=labels,
            range=ranges,
            color=filled_color,
            fig=subfigs[pop_idx],
            fill_contours=True,
            plot_contours=True,
            levels=levels,
            hist_kwargs={'linewidth': LINEWIDTH, 'zorder': 1000},
            contour_kwargs={'linewidths': LINEWIDTH, 'zorder': 1000},
            contourf_kwargs={'zorder': 999},  # Filled regions very high
            **BASE_CORNER_KWARGS
        )

        # Step 2: Overlay line contours for other EOS datasets (EVEN HIGHER zorder)
        for eos_samples_name in eos_samples_name_list:
            if eos_samples_name == filled_eos:
                continue  # Already plotted with fill

            print(f"Plotting {eos_samples_name} with line contours...")
            line_color = eos_colors[eos_samples_name]

            corner_fig = corner.corner(
                datasets[eos_samples_name],
                labels=labels,
                range=ranges,
                color=line_color,
                fig=corner_fig,  # Use the existing figure
                fill_contours=False,  # No fill for overlay
                plot_contours=True,
                levels=levels,
                hist_kwargs={'linewidth': LINEWIDTH, 'zorder': 1001},  # Above filled contours
                contour_kwargs={'linewidths': LINEWIDTH, 'zorder': 1001},  # Above filled contours
                **BASE_CORNER_KWARGS
            )

        # Step 3: Plot invisible dummy dataset LAST for histogram normalization
        if dummy_dataset is not None:
            print(f"Plotting invisible dummy dataset for {pop} for histogram normalization")
            corner_fig = corner.corner(
                dummy_dataset,
                labels=labels,
                range=ranges,
                fig=corner_fig,
                plot_contours=False,
                fill_contours=False,
                hist_kwargs={'alpha': 0},  # Invisible histograms
                **BASE_CORNER_KWARGS
            )

        # # Add title for each population (REMOVED)
        # title_map = {
        #     "uniform": "Uniform",
        #     "gaussian": "Gaussian",
        #     "double_gaussian": "Double Gaussian"
        # }
        # subfigs[pop_idx].suptitle(title_map[pop], fontsize=40, y=0.98)

    # Add legend in the top-right area of the overall figure
    legend_x = 0.875
    legend_y = 0.875
    dy = 0.075
    legend_fontsize = 50

    for i, eos_samples_name in enumerate(eos_samples_name_list):
        fig.text(
            legend_x,
            legend_y - i * dy,
            utils.EOS_SAMPLES_NAMES_DICT[eos_samples_name],
            color=eos_colors[eos_samples_name],
            fontsize=legend_fontsize,
            transform=fig.transFigure,
            verticalalignment='top',
            horizontalalignment='left'
        )

    # Save the combined figure
    os.makedirs("./figures/priors/combined", exist_ok=True)

    if convert_masses:
        save_name = f"./figures/priors/combined/{source_type}_all_populations_chirp.pdf"
    else:
        save_name = f"./figures/priors/combined/{source_type}_all_populations_component.pdf"

    if convert_to_lambda_tilde:
        save_name = save_name.replace(".pdf", "_tilde.pdf")

    print(f"\nSaving combined figure to {save_name}")
    plt.savefig(save_name, bbox_inches="tight", dpi=150)

    if os.path.exists("../../paper/Figures"):
        print(f"Also saving to the Overleaf paper repo")
        save_name_paper = save_name.replace("./figures/priors/combined/", "../../paper/Figures/")
        plt.savefig(save_name_paper, bbox_inches="tight", dpi=150)

    plt.close()


def main():

    convert_to_lambda_tilde = True

    for source_type in ["bns", "nsbh"]:
        print(f"\n{'='*60}")
        print(f"Creating combined plot for {source_type}")
        print(f"{'='*60}")

        # Define ranges for each population
        ranges_dict = {}

        if source_type == "bns":
            if convert_to_lambda_tilde:
                ranges_dict = {
                    "uniform": [[0.8, 2.2], [0.40, 1.0], [0.0, 1000.0], [0.0, 300.0]],
                    "gaussian": [[0.99, 1.35], [0.75, 1.0], [0.0, 1800.0], [0.0, 180.0]],
                    "double_gaussian": [[1.00, 1.75], [0.50, 1.0], [0.0, 1700.0], [0.0, 180.0]],
                }
            else:
                # No ranges for the m1-m2 version
                ranges_dict = None
        else:  # nsbh
            if convert_to_lambda_tilde:
                ranges_dict = {
                    "uniform": [[1.25, 3.0], [0.20, 1.0], [0.0, 1000.0]],
                    "gaussian": None,
                    "double_gaussian": None,
                }
            else:
                # No ranges for the m1-m2 version
                ranges_dict = None

        # Define normalization indices for each population
        # Index mapping: 0=radio, 1=radio_chiEFT, 2=radio_NICER
        normalization_indices_dict = {}

        if source_type == "bns":
            # Use radio_chiEFT for all 4 parameters (chirp_mass, mass_ratio, lambda_tilde, delta_lambda_tilde)
            normalization_indices_dict = {
                "uniform": [1, 1, 1, 1],
                "gaussian": [1, 1, 1, 1],
                "double_gaussian": [1, 1, 1, 1],
            }
        else:  # nsbh
            # Use radio_chiEFT for all 3 parameters (chirp_mass, mass_ratio, lambda_2)
            normalization_indices_dict = {
                "uniform": [1, 1, 1],
                "gaussian": [1, 1, 1],
                "double_gaussian": [1, 1, 1],
            }

        combined_plot(
            source_type=source_type,
            convert_masses=True,
            convert_to_lambda_tilde=convert_to_lambda_tilde,
            levels=[0.68, 0.95],
            filled_eos="radio",
            ranges_dict=ranges_dict,
            normalization_indices_dict=normalization_indices_dict
        )


if __name__ == "__main__":
    main()
