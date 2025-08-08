import os

JAX_LIGHT_BLUE = "#5e97f6"
JAX_DARK_BLUE = "#2a56c6"
JAX_DARK_BLUE_TINT1 = "#7f9add"
JAX_DARK_BLUE_TINT2 = "#aabbe8"

JAX_LIGHT_GREEN = "#26a69a"
JAX_DARK_GREEN = "#00695c"
JAX_DARK_GREEN_TINT1 = "#4d968d"
JAX_DARK_GREEN_TINT2 = "#80b4ae"


JAX_PINK = "#ea80fc"
JAX_LIGHT_PURPLE = "#9c27b0"
JAX_DARK_PURPLE = "#6a1c9a"
JAX_DARK_PURPLE_TINT1 = "#9760b8"
JAX_DARK_PURPLE_TINT2 = "#b58ecd"


def get_training_data_path(population_name: str,
                           source_type: str,
                           eos_samples_name: str,
                           ) -> str:
    """
    Get the path to the training data file based on the population name, source type, and EOS samples name.

    Args:
        population_name (str): Name of the population (e.g., "uniform", "gaussian", "double_gaussian").
        source_type (str): Name of the source type (e.g., "BNS", "NSBH").
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