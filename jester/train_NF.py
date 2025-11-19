"""
Train a normalizing flow on a 4D GW marginal posterior mass_1_source, mass_2_source, lambda_1, lambda_2 and save the model
"""

import os
import sys
import matplotlib.pyplot as plt
import corner
import numpy as np
import copy

import jax
import jax.numpy as jnp
import equinox as eqx
from flowjax.flows import block_neural_autoregressive_flow
from flowjax.train import fit_to_data
from flowjax.distributions import Normal

# Configure JAX
jax.config.update("jax_enable_x64", True)

print("GPU found?")
print(jax.devices())

# Plot styling
params = {
    "axes.grid": True,
    "text.usetex": False,
    "font.family": "serif",
    "ytick.color": "black",
    "xtick.color": "black",
    "axes.labelcolor": "black",
    "axes.edgecolor": "black",
    "font.serif": ["Computer Modern Serif"],
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "axes.labelsize": 16,
    "legend.fontsize": 16,
    "legend.title_fontsize": 16,
    "figure.titlesize": 16
}
plt.rcParams.update(params)

# Corner plot defaults
default_corner_kwargs = dict(
    bins=40,
    smooth=1.0,
    show_titles=False,
    label_kwargs=dict(fontsize=16),
    title_kwargs=dict(fontsize=16),
    color="blue",
    # levels=[0.68, 0.9, 0.997],
    plot_density=False,
    plot_datapoints=False,
    fill_contours=False,
    max_n_ticks=4,
    min_n_ticks=3,
    truth_color="red",
    density=True,
    save=False
)


def load_data(path: str) -> np.ndarray:
    """
    Load 4D GW data from npz file.
    
    Expected keys: mass_1_source, mass_2_source, lambda_1, lambda_2
    
    Returns:
        np.ndarray: Shape (4, n_samples) with [m1, m2, lambda1, lambda2]
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")
    
    data = np.load(path)
    
    required_keys = ["mass_1_source", "mass_2_source", "lambda_1", "lambda_2"]
    for key in required_keys:
        if key not in data.keys():
            raise KeyError(f"Required key '{key}' not found in {path}. Available keys: {list(data.keys())}")
    
    m1 = data["mass_1_source"].flatten()
    m2 = data["mass_2_source"].flatten()
    lambda1 = data["lambda_1"].flatten()
    lambda2 = data["lambda_2"].flatten()
    
    # Check all arrays have same length
    lengths = [len(arr) for arr in [m1, m2, lambda1, lambda2]]
    if not all(l == lengths[0] for l in lengths):
        raise ValueError(f"All data arrays must have same length. Got: {lengths}")
    
    print(f"Loaded {len(m1)} samples from {path}")
    return np.array([m1, m2, lambda1, lambda2])


def make_cornerplot(data_samples: np.ndarray, 
                   nf_samples: np.ndarray,
                   output_path: str):
    """
    Create corner plot comparing data and NF samples.
    
    Args:
        data_samples: Shape (4, n_samples) - original data
        nf_samples: Shape (n_samples, 4) - NF samples
        output_path: Where to save the plot
    """
    # Get plotting range from data
    plot_range = []
    for i in range(4):
        data_min, data_max = np.min(data_samples[i]), np.max(data_samples[i])
        padding = 0.1 * (data_max - data_min)
        plot_range.append([data_min - padding, data_max + padding])
    
    labels = [r"$m_1$ [M$_{\odot}$]", r"$m_2$ [M$_{\odot}$]", r"$\Lambda_1$", r"$\Lambda_2$"]
    
    # Plot training data
    corner_kwargs = copy.deepcopy(default_corner_kwargs)
    corner_kwargs["color"] = "blue"
    corner_kwargs["hist_kwargs"] = {"density": True, "color": "blue"}
    
    fig = corner.corner(data_samples.T, range=plot_range, labels=labels, **corner_kwargs)
    
    # Plot NF samples
    corner_kwargs["color"] = "red"
    corner_kwargs["hist_kwargs"] = {"density": True, "color": "red"}
    corner.corner(nf_samples, range=plot_range, fig=fig, **corner_kwargs)
    
    # Add legend
    plt.figtext(0.75, 0.75, "Training data", fontsize=20, color="blue")
    plt.figtext(0.75, 0.70, "Normalizing flow", fontsize=20, color="red")
    
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close()


def train_flow(data_path: str, 
               output_dir: str = "./NFs/",
               num_epochs: int = 2000,
               learning_rate: float = 5e-4,
               batch_size: int = 1024,
               max_patience: int = 200,
               nn_depth: int = 5,
               nn_block_dim: int = 8,
               n_plot_samples: int = 5_000,
               seed: int = 0):
    """
    Train normalizing flow on 4D GW data.
    
    Args:
        data_path: Path to npz file with GW data
        output_dir: Directory to save outputs
        num_epochs: Number of training epochs
        learning_rate: Learning rate for training
        batch_size: Batch size for training
        max_patience: Early stopping patience
        nn_depth: Number of neural network layers
        nn_block_dim: Neural network block dimension
        n_plot_samples: Number of samples for plotting
        seed: Random seed for reproducibility
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print(f"Loading data from {data_path}...")
    data = load_data(data_path)
    n_dim, n_samples = data.shape
    print(f"Data shape: {n_dim} dimensions, {n_samples} samples")
    
    # Prepare data for training (samples x features)
    x = data.T
    
    # Initialize flow
    flow_key, train_key, sample_key = jax.random.split(jax.random.key(seed), 3)
    
    flow = block_neural_autoregressive_flow(
        key=flow_key,
        base_dist=Normal(jnp.zeros(n_dim)),
        nn_depth=nn_depth,
        nn_block_dim=nn_block_dim,
    )
    
    print(f"Training flow for {num_epochs} epochs...")
    print(f"Network architecture: {nn_depth} layers, {nn_block_dim} block dimension")
    print(f"Learning rate: {learning_rate}, Batch size: {batch_size}, Patience: {max_patience}")
    
    flow, losses = fit_to_data(
        key=train_key,
        dist=flow,
        x=x,
        learning_rate=learning_rate,
        max_epochs=num_epochs,
        max_patience=max_patience,
        batch_size=batch_size,
    )
    
    # Plot training losses
    plt.figure(figsize=(8, 6))
    plt.plot(losses["train"], label="Train", color="red")
    plt.plot(losses["val"], label="Validation", color="blue")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    loss_path = os.path.join(output_dir, "training_losses.png")
    plt.savefig(loss_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Training losses saved to {loss_path}")
    
    # Sample from trained flow
    nf_samples = flow.sample(sample_key, (n_plot_samples,))
    nf_samples_np = np.array(nf_samples)
    
    # Create corner plot
    corner_path = os.path.join(output_dir, "corner_plot.png")
    make_cornerplot(data, nf_samples_np, corner_path)
    print(f"Corner plot saved to {corner_path}")
    
    # Save model
    model_path = os.path.join(output_dir, "model.eqx")
    eqx.tree_serialise_leaves(model_path, flow)
    print(f"Model saved to {model_path}")
    
    # Test loading model
    print("Testing model loading...")
    loaded_flow = eqx.tree_deserialise_leaves(model_path, like=flow)
    test_samples = loaded_flow.sample(sample_key, (1000,))
    print(f"Successfully loaded model and generated {len(test_samples)} test samples")
    
    print("Training complete!")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Train a normalizing flow on 4D gravitational wave data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        "data_path", 
        type=str,
        help="Path to npz file with keys: mass_1_source, mass_2_source, lambda_1, lambda_2"
    )
    
    # Optional arguments
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="./NFs/",
        help="Directory to save outputs"
    )
    
    # Training hyperparameters
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=3000,
        help="Number of training epochs"
    )
    
    parser.add_argument(
        "--learning-rate", 
        type=float, 
        default=5e-4,
        help="Learning rate for training"
    )
    
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=1024,
        help="Batch size for training"
    )
    
    parser.add_argument(
        "--patience", 
        type=int, 
        default=200,
        help="Early stopping patience (epochs)"
    )
    
    # Network architecture
    parser.add_argument(
        "--nn-depth", 
        type=int, 
        default=5,
        help="Number of neural network layers"
    )
    
    parser.add_argument(
        "--nn-block-dim", 
        type=int, 
        default=8,
        help="Neural network block dimension"
    )
    
    # Plotting
    parser.add_argument(
        "--n-plot-samples", 
        type=int, 
        default=10000,
        help="Number of samples to generate for corner plot"
    )
    
    # Random seed
    parser.add_argument(
        "--seed", 
        type=int, 
        default=0,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    # Print configuration
    print("=" * 60)
    print("NORMALIZING FLOW TRAINING CONFIGURATION")
    print("=" * 60)
    print(f"Data path:          {args.data_path}")
    print(f"Output directory:   {args.output_dir}")
    print(f"Random seed:        {args.seed}")
    print()
    print("Training parameters:")
    print(f"  Epochs:           {args.epochs}")
    print(f"  Learning rate:    {args.learning_rate}")
    print(f"  Batch size:       {args.batch_size}")
    print(f"  Patience:         {args.patience}")
    print()
    print("Network architecture:")
    print(f"  NN depth:         {args.nn_depth}")
    print(f"  NN block dim:     {args.nn_block_dim}")
    print()
    print(f"Plot samples:       {args.n_plot_samples}")
    print("=" * 60)
    print()
    
    train_flow(
        data_path=args.data_path,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        max_patience=args.patience,
        nn_depth=args.nn_depth,
        nn_block_dim=args.nn_block_dim,
        n_plot_samples=args.n_plot_samples,
        seed=args.seed
    )


if __name__ == "__main__":
    main()