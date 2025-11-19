# NFprior

This directory contains scripts for training and evaluating normalizing flow (NF) models that encode equation of state (EOS) constraints for neutron star parameter estimation.

*Note from human*: Please use the `glasflow` models, not the `flowjax` models, since the latter have NOT been tested in GW inference.

## Overview

Normalizing flows are trained on EOS-derived samples of neutron star masses and tidal deformabilities, then used as physics-informed priors in gravitational wave parameter estimation. The trained models enable Bayesian inference to incorporate nuclear physics constraints directly into the analysis.

## Key Scripts

### Training

**`train_NF_prior.py`** - Main script for training normalizing flow models
- Trains NF models on EOS samples (masses and tidal deformabilities)
- Supports both PyTorch (glasflow) and JAX (flowjax) backends
- Implements multiple flow architectures:
  - Coupling Neural Spline Flow (CouplingNSF) - recommended
  - Block Neural Autoregressive Flow (BNAF)
  - Masked Autoregressive Flow (MAF)
- Handles different neutron star population models (uniform, Gaussian, double Gaussian)
- Implements parameter constraints (mass ratio bounds, positive lambdas)
- Saves trained models to `models/` subdirectories

**Usage:**
```bash
# Train flowjax CouplingNSF model with uniform mass distribution
python train_NF_prior.py --use-flowjax --population-type uniform --eos-samples-name radio

# Train glasflow model with Gaussian mass distribution
python train_NF_prior.py --population-type gaussian --eos-samples-name radio_chiEFT

# Customize hyperparameters
python train_NF_prior.py --use-flowjax --flow-layers 4 --nn-depth 4 --learning-rate 1e-3
```

### Evaluation

**`evaluate_flows.py`** - Comprehensive evaluation of trained NF models
- Generates samples from trained models
- Creates corner plots comparing NF samples to training data
- Computes validation metrics (log-likelihood, sample quality)
- Produces diagnostic visualizations
- Supports both conditional and unconditional flows

**Usage:**
```bash
# Evaluate flowjax model
python evaluate_flows.py --use-flowjax --population-type uniform --eos-samples-name radio

# Evaluate with custom sample size
python evaluate_flows.py --use-flowjax --nb-samples 50000 --population-type gaussian
```

### Diagnostic Scripts

**`check_architecture.py`** - Inspect NF model architecture and parameters
- Displays flow architecture details
- Shows parameter counts and layer configurations
- Useful for debugging and understanding model complexity

**`assess_JSD.py`** - Calculate Jensen-Shannon Divergence between distributions
- Quantifies similarity between NF samples and training data
- Provides numerical assessment of model quality

**`inspect_saved_model.py`** - Load and inspect saved model files
- Examines model file structure and metadata
- Verifies model compatibility and format

**`test_loading_unconditional.py`** - Test unconditional NF model loading
- Validates model loading and sampling functionality
- Used for debugging model serialization issues

**`check_ln_prob.py`** - Verify log-probability calculations
- Tests probability density evaluations
- Checks numerical stability of likelihood computations

**`debug_boundary.py`** - Debug parameter space boundary handling
- Investigates boundary effects in NF training
- Tests constraint enforcement at parameter limits

## Directory Structure

```
NFprior/
├── models/                    # Trained NF models organized by configuration
│   ├── uniform/              # Models trained on uniform mass population
│   ├── gaussian/             # Models trained on Gaussian mass population
│   ├── double_gaussian/      # Models trained on double Gaussian population
│   ├── GW170817/            # Event-specific conditional models
│   ├── GW190425/            # Event-specific conditional models
│   └── GW230529/            # Event-specific conditional models
├── train_NF_prior.py        # Main training script
├── evaluate_flows.py        # Model evaluation and visualization
└── [diagnostic scripts]     # Various debugging and testing utilities
```

## Model Storage

Trained models are saved in subdirectories of `models/` with the naming convention:
- PyTorch (glasflow): `{population_type}/{eos_samples_name}/model.pth`
- JAX (flowjax): `{population_type}/{eos_samples_name}/flowjax_model.eqx`

Model metadata (hyperparameters, training history) is stored alongside model weights.

## Flow Architectures

### Coupling Neural Spline Flow (CouplingNSF)
- **Recommended** for EOS applications
- Analytically invertible (no numerical inversion failures)
- Smooth transformations ideal for mass-lambda correlations
- Fast sampling and stable training

### Block Neural Autoregressive Flow (BNAF)
- Maximum expressivity but requires numerical inversion
- May encounter convergence issues with large-capacity models
- Requires careful tuning of `maxiter` parameter

### Masked Autoregressive Flow (MAF)
- Autoregressive architecture with flexible transformations
- Slower sampling due to sequential operations
- Good for complex distributions but computational overhead

## Parameter Constraints

FlowJAX models support built-in parameter constraints:
- **Mass ratio (q)**: Constrained to [0.1, 1.0] using Sigmoid + ScalarAffine
- **Tidal deformabilities (λ₁, λ₂)**: Constrained to positive values using Softplus
- **Luminosity distance**: Positive values enforced for GW event models

Enable with `--constrain-flowjax-dist` (default: True)

## Hyperparameter Tuning

Key hyperparameters for flowjax models:
- `--flow-layers`: Number of flow transformations (default: 4, recommended: 3-5)
- `--nn-depth`: Neural network depth per layer (default: 4)
- `--nn-block-dim`: Width multiplier for hidden layers (default: 24)
- `--learning-rate`: Optimizer learning rate (default: 1e-3)
- `--max-patience`: Early stopping patience in epochs (default: 250)
- `--num-epochs`: Maximum training epochs (default: 2000)

Larger values increase model capacity but require longer training and more data.

## Data Sources

Training data is sourced from `../data/eos/` containing EOS posterior samples:
- `radio/`: Pulsar radio observations
- `radio_chiEFT/`: Radio + chiral effective field theory
- `radio_NICER/`: Radio + NICER X-ray observations
- `radio_GW170817/`: Radio + GW170817 constraints
- `radio_chiEFT_NICER/`: Combined radio, χEFT, and NICER

Each dataset contains samples of neutron star masses (m₁, m₂) and tidal deformabilities (λ₁, λ₂).

## Integration with Parameter Estimation

Trained NF models are loaded by `../GW_runs/pe.py` as custom priors in bilby parameter estimation. The models replace standard uniform or informative priors, incorporating EOS physics directly into gravitational wave inference.

## Common Workflows

1. **Train new model**: `python train_NF_prior.py --use-flowjax --population-type uniform --eos-samples-name radio`
2. **Evaluate model**: `python evaluate_flows.py --use-flowjax --population-type uniform --eos-samples-name radio`
3. **Inspect architecture**: `python check_architecture.py --use-flowjax --population-type uniform`
4. **Use in PE**: Model automatically loaded by `pe.py` when matching population and EOS name specified
