#!/usr/bin/env python3
"""
Load and display the architecture of normalizing flow models.

This script can load both FlowJAX (.eqx) and GlasFlow (.pt) models
and display their architecture in a human-readable format.

Usage:
    python check_architecture.py <model_directory>

Example:
    python check_architecture.py ./models/uniform/bns/radio_flowjax/
"""

import os
import sys
import json
import argparse
from pathlib import Path


def print_section(title, char="=", width=80):
    """Print a section header"""
    print(f"\n{char * width}")
    print(f"{title}")
    print(f"{char * width}")


def print_subsection(title, char="-", width=60):
    """Print a subsection header"""
    print(f"\n{char * width}")
    print(f"{title}")
    print(f"{char * width}")


def load_flowjax_model(model_dir):
    """Load and display FlowJAX model architecture"""
    import jax
    import jax.numpy as jnp
    import jax.random as jr
    from flowjax.flows import (
        block_neural_autoregressive_flow,
        coupling_flow,
        masked_autoregressive_flow
    )
    from flowjax.distributions import Normal, Transformed
    from flowjax import bijections as bij
    import equinox as eqx

    print_section("FLOWJAX MODEL ARCHITECTURE")

    # Load model kwargs
    kwargs_path = os.path.join(model_dir, "model_kwargs.json")
    with open(kwargs_path, "r") as f:
        kwargs = json.load(f)

    # Display training configuration
    print_subsection("Training Configuration")
    print(f"  Model Type: {kwargs.get('model_type', 'Unknown')}")
    print(f"  Source Type: {kwargs.get('source_type', 'Unknown')}")
    print(f"  Number of Inputs: {kwargs['n_inputs']}")
    print(f"  Parameter Names: {', '.join(kwargs['names'])}")
    print(f"  Training Samples: {kwargs.get('N_samples_training', 'Unknown'):,}")
    print(f"  Epochs: {kwargs.get('num_epochs', 'Unknown')}")
    print(f"  Learning Rate: {kwargs.get('learning_rate', 'Unknown')}")
    print(f"  Batch Size: {kwargs.get('batch_size', 'Unknown')}")
    print(f"  Max Patience: {kwargs.get('max_patience', 'Unknown')}")
    print(f"  Constrained Distribution: {kwargs.get('constrain_flowjax_dist', 'Unknown')}")

    # Display flow architecture details
    print_subsection("Flow Architecture")
    model_type = kwargs.get('model_type', '')

    if 'block_neural_autoregressive' in model_type:
        print(f"  Flow Type: Block Neural Autoregressive Flow (BNAF)")
        print(f"  Flow Layers: {kwargs.get('flow_layers', 'N/A')}")
        print(f"  Neural Network Depth: {kwargs.get('nn_depth', 'N/A')}")
        print(f"  Neural Network Block Dimension: {kwargs.get('nn_block_dim', 'N/A')}")
        hidden_dim = kwargs['n_inputs'] * kwargs.get('nn_block_dim', 8)
        print(f"  Hidden Layer Width: {hidden_dim} neurons")

    elif 'coupling' in model_type:
        print(f"  Flow Type: Coupling Flow (NSF)")
        print(f"  Number of Transforms: {kwargs.get('n_transforms', 'N/A')}")
        print(f"  Number of Neurons: {kwargs.get('n_neurons', 'N/A')}")
        print(f"  Number of Bins: {kwargs.get('num_bins', 'N/A')}")

    elif 'masked_autoregressive' in model_type:
        print(f"  Flow Type: Masked Autoregressive Flow (MAF)")
        print(f"  Flow Layers: {kwargs.get('flow_layers', 'N/A')}")
        print(f"  Neural Network Depth: {kwargs.get('nn_depth', 'N/A')}")

    # Load the actual model
    print_subsection("Loading Model")
    model_path = os.path.join(model_dir, "model.eqx")

    try:
        # Create base distribution
        n_inputs = kwargs['n_inputs']
        base_dist = Normal(jnp.zeros(n_inputs))
        key = jr.key(42)

        # Create flow based on model type
        if 'block_neural_autoregressive' in model_type:
            flow = block_neural_autoregressive_flow(
                key=key,
                base_dist=base_dist,
                nn_depth=kwargs.get('nn_depth', 1),
                nn_block_dim=kwargs.get('nn_block_dim', 8),
                flow_layers=kwargs.get('flow_layers', 1),
            )
        elif 'coupling' in model_type:
            flow = coupling_flow(
                key=key,
                base_dist=base_dist,
                transformer=kwargs.get('transformer', 'rq_coupling'),
                nn_width=kwargs.get('n_neurons', 64),
                num_layers=kwargs.get('n_transforms', 4),
                num_bins=kwargs.get('num_bins', 10),
            )
        elif 'masked_autoregressive' in model_type:
            flow = masked_autoregressive_flow(
                key=key,
                base_dist=base_dist,
                nn_depth=kwargs.get('nn_depth', 1),
                flow_layers=kwargs.get('flow_layers', 1),
            )
        else:
            print(f"  Unknown model type: {model_type}")
            return

        # Load saved parameters
        loaded_flow = eqx.tree_deserialise_leaves(model_path, flow)
        print(f"  ✓ Model loaded successfully")
        print(f"  Model Type: {type(loaded_flow).__name__}")

        # Check for parameter constraints
        print_subsection("Parameter Constraints")

        if kwargs.get('constrain_flowjax_dist', 'False') == 'True':
            print("  ✓ Model uses constrained distributions")
            print("  Parameter-specific constraints:")

            for i, name in enumerate(kwargs['names']):
                if 'mass_ratio' in name or 'q' == name:
                    print(f"    - {name}: Bounded to [0.1, 1.0] via Sigmoid + Affine")
                elif 'lambda' in name and 'delta_lambda_tilde' not in name:
                    print(f"    - {name}: Positive constraint via Softplus")
                elif 'luminosity_distance' in name or 'dL' in name:
                    print(f"    - {name}: Positive constraint via Softplus")
                else:
                    print(f"    - {name}: Unbounded (Identity transformation)")
        else:
            print("  ✗ Model uses unbounded distributions")
            print("  Warning: This may assign probability mass outside physical bounds")

        # Display model statistics
        print_subsection("Model Statistics")

        # Count parameters
        flat_params, _ = jax.tree_util.tree_flatten(loaded_flow)
        total_params = sum(p.size for p in flat_params if hasattr(p, 'size'))
        print(f"  Total Parameters: {total_params:,}")

        # Try to sample from the model
        try:
            print("\n  Testing sampling capability...")
            key = jr.key(123)
            test_samples = loaded_flow.sample(key, (5,))
            print(f"  ✓ Successfully generated {test_samples.shape[0]} test samples")
            print(f"  Sample shape: {test_samples.shape}")
            print(f"  Sample preview (first 2):")
            for i in range(min(2, test_samples.shape[0])):
                sample_dict = {name: test_samples[i, j].item()
                              for j, name in enumerate(kwargs['names'])}
                print(f"    Sample {i+1}:")
                for name, val in sample_dict.items():
                    print(f"      {name:20s}: {val:12.6f}")
        except Exception as e:
            print(f"  ✗ Sampling failed: {e}")

    except Exception as e:
        print(f"  ✗ Failed to load model: {e}")
        import traceback
        traceback.print_exc()


def load_glasflow_model(model_dir):
    """Load and display GlasFlow model architecture"""
    import torch
    import joblib
    from glasflow.flows.nsf import CouplingNSF
    from glasflow.flows.autoregressive import (
        MaskedPiecewiseRationalQuadraticAutoregressiveFlow,
        MaskedAffineAutoregressiveFlow
    )

    print_section("GLASFLOW MODEL ARCHITECTURE")

    # Load model kwargs
    kwargs_path = os.path.join(model_dir, "model_kwargs.json")
    with open(kwargs_path, "r") as f:
        kwargs = json.load(f)

    # Display training configuration
    print_subsection("Training Configuration")
    print(f"  Model Type: {kwargs.get('model_type', 'Unknown')}")
    print(f"  Source Type: {kwargs.get('source_type', 'Unknown')}")
    print(f"  Number of Inputs: {kwargs['n_inputs']}")
    print(f"  Parameter Names: {', '.join(kwargs['names'])}")
    print(f"  Training Samples: {kwargs.get('N_samples_training', 'Unknown'):,}")
    print(f"  Epochs: {kwargs.get('num_epochs', 'Unknown')}")
    print(f"  Learning Rate: {kwargs.get('learning_rate', 'Unknown')}")
    print(f"  Batch Size: {kwargs.get('batch_size', 'Unknown')}")
    print(f"  Max Patience: {kwargs.get('max_patience', 'Unknown')}")

    # Display flow architecture details
    print_subsection("Flow Architecture")
    model_type = kwargs.get('model_type', '')

    if 'nsf' in model_type.lower() or 'coupling' in model_type.lower():
        print(f"  Flow Type: Neural Spline Flow (Coupling NSF)")
        print(f"  Number of Transforms: {kwargs.get('n_transforms', 'N/A')}")
        print(f"  Number of Neurons: {kwargs.get('n_neurons', 'N/A')}")
        print(f"  Blocks per Transform: {kwargs.get('n_blocks_per_transform', 'N/A')}")
        print(f"  Number of Bins: {kwargs.get('num_bins', 'N/A')}")

    elif 'autoregressive' in model_type.lower():
        print(f"  Flow Type: Masked Autoregressive Flow")
        print(f"  Number of Transforms: {kwargs.get('n_transforms', 'N/A')}")
        print(f"  Number of Neurons: {kwargs.get('n_neurons', 'N/A')}")

    # Load the actual model
    print_subsection("Loading Model")
    model_path = os.path.join(model_dir, "model.pt")

    try:
        # Load model state dict
        state_dict = torch.load(model_path, map_location='cpu')
        print(f"  ✓ Model loaded successfully")
        print(f"  State dict keys: {len(state_dict)} parameter tensors")

        # Count total parameters
        total_params = sum(p.numel() for p in state_dict.values())
        print(f"  Total Parameters: {total_params:,}")

        # Display some layer information
        print(f"\n  Layer structure:")
        for key in list(state_dict.keys())[:10]:  # Show first 10 layers
            shape = state_dict[key].shape
            print(f"    {key:50s}: {str(shape):20s}")
        if len(state_dict) > 10:
            print(f"    ... and {len(state_dict) - 10} more layers")

    except Exception as e:
        print(f"  ✗ Failed to load model: {e}")
        import traceback
        traceback.print_exc()

    # Check for scaler
    print_subsection("Data Preprocessing")
    scaler_path = os.path.join(model_dir, "scaler.gz")
    if os.path.exists(scaler_path):
        try:
            scaler = joblib.load(scaler_path)
            print(f"  ✓ Scaler found: {type(scaler).__name__}")
            if hasattr(scaler, 'mean_'):
                print(f"  Data means: {scaler.mean_}")
                print(f"  Data scales: {scaler.scale_}")
        except:
            print(f"  ✗ Failed to load scaler")
    else:
        print(f"  No scaler found (raw data used)")


def main():
    # Default to gaussian population with radio EOS constraints (GlasFlow/PyTorch)
    default_model_dir = "./models/gaussian/bns/radio/"

    parser = argparse.ArgumentParser(
        description='Load and display normalizing flow model architecture',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python check_architecture.py
  python check_architecture.py ./models/gaussian/bns/radio/
  python check_architecture.py ./models/uniform/bns/radio/
        """
    )
    parser.add_argument(
        'model_dir',
        type=str,
        nargs='?',
        default=default_model_dir,
        help=f'Path to model directory containing model files and kwargs (default: {default_model_dir})'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output'
    )

    args = parser.parse_args()

    # Check if directory exists
    if not os.path.isdir(args.model_dir):
        print(f"Error: Directory not found: {args.model_dir}")
        sys.exit(1)

    # Check for kwargs file
    kwargs_path = os.path.join(args.model_dir, "model_kwargs.json")
    if not os.path.exists(kwargs_path):
        print(f"Error: model_kwargs.json not found in {args.model_dir}")
        sys.exit(1)

    # Determine model type
    model_eqx = os.path.join(args.model_dir, "model.eqx")
    model_pt = os.path.join(args.model_dir, "model.pt")

    print(f"\nModel Directory: {args.model_dir}")

    if os.path.exists(model_eqx):
        print("Model Backend: FlowJAX (.eqx)")
        try:
            load_flowjax_model(args.model_dir)
        except ImportError as e:
            print(f"\nError: FlowJAX not available: {e}")
            print("Install with: pip install flowjax equinox")
            sys.exit(1)

    elif os.path.exists(model_pt):
        print("Model Backend: GlasFlow (.pt)")
        try:
            load_glasflow_model(args.model_dir)
        except ImportError as e:
            print(f"\nError: GlasFlow/PyTorch not available: {e}")
            print("Install with: pip install torch glasflow")
            sys.exit(1)
    else:
        print(f"Error: No model file found (neither .eqx nor .pt)")
        sys.exit(1)

    print("\n" + "="*80)
    print("Architecture inspection complete!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
