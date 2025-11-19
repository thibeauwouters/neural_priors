# Jester EOS Inference Scripts

This directory stores the scripts used to perform the EOS inference with [jester](https://github.com/nuclear-multimessenger-astronomy/jester). The results of the inference are too heavy to store on Github, but can be obtained by contacting me (or raise a Git issue).

For more information about the jester framework and the underlying methodology, please visit: **https://github.com/nuclear-multimessenger-astronomy/jester**

## File Descriptions

### `train_NF.py`
Trains normalizing flows on 4D gravitational wave marginal posteriors containing `mass_1_source`, `mass_2_source`, `lambda_1`, and `lambda_2`. The script:
- Uses FlowJAX's block neural autoregressive flow architecture
- Loads posterior samples from NPZ files
- Trains the flow model using JAX
- Generates corner plots comparing training data to flow samples
- Saves trained models in Equinox format (`.eqx` files)

**Key features:**
- GPU acceleration support via JAX
- Configurable flow hyperparameters
- Validation through corner plot visualization

### `inference.py`
Full-scale EOS inference script using `jim` (JAX Interface for Markov chain Monte Carlo) as a flowMC wrapper. This is the main driver for performing parameter estimation on gravitational wave events and pulsar observations.

**Key features:**
- Extensive command-line interface for configuring inference runs
- Support for multiple GW events (GW170817, GW190425, GW231109)
- Multiple prior choices: default, small NEP, different nbreak distributions
- Integration with trained normalizing flows from GW posteriors
- Support for pulsar mass measurements (J0030, J0740, radio timing pulsars)
- NICER mass measurement constraints
- Binary Love relations option
- Injection recovery capabilities

**Main use cases:**
- Joint EOS inference from GW events and pulsar observations
- Testing different nuclear physics priors
- Recovery studies with injections
- Comparison of different crust models

### `postprocessing.py`
Modular postprocessing script for analyzing EOS inference results and generating publication-quality plots.

**Capabilities:**
- Load EOS samples from inference outputs
- Calculate credible intervals using highest density intervals (HDI)
- Generate mass-radius diagrams with posterior probability coloring
- Compare posteriors against priors
- Unit conversions for nuclear physics quantities (densities, pressures, energies)
- Integration with ArviZ for statistical analysis

**Output plots:**
- Mass-radius relationships for neutron stars
- EOS curves (pressure vs density, energy density vs density)
- Sound speed profiles

### `utils.py`
Core utilities for EOS inference including likelihood calculations, prior definitions, and data handling.

**Main components:**
- `MetaModel_EOS_model` and `MetaModel_with_CSE_EOS_model` integration from jesterTOV
- NEP (Nuclear Empirical Parameters) constant definitions
- Support for pulsar mass-radius measurements from NICER observations
- Custom likelihood classes for jim/flowMC integration
- Prior transformations and coordinate conversions
- Data loading for pulsar observations (J0030, J0740)
- KDE handling for observational constraints

**Key constants:**
- Nuclear saturation parameters (E_sym, L_sym, K_sym, etc.)
- CSE (Constant Sound speed Extension) parameters
- Pulsar data file paths

### `utils_plotting.py`
Plotting utilities for visualizing EOS inference results and posterior distributions.

**Features:**
- Corner plot generation with customizable styling
- EOS curve plotting (microphysical properties)
- Multiple color schemes for different analyses (Amsterdam/Maryland, CREX/PREX)
- Default matplotlib styling for consistent publication-quality figures
- Integration with jesterTOV utilities for unit conversions

**Plot types:**
- Corner plots for posterior parameter distributions
- Pressure-density (p-n) curves
- Energy density-density (e-n) curves
- Sound speed-density (cs²-n) curves
- Energy-pressure (e-p) diagrams

## Workflow

The typical workflow for using these scripts is:

1. **Train NF on GW posterior** (optional): Use `train_NF.py` to create a normalizing flow representation of a GW event's posterior
2. **Configure and run inference**: Use `inference.py` with appropriate command-line flags to perform EOS inference
3. **Analyze results**: Use `postprocessing.py` to generate plots and calculate credible intervals
4. **Visualize**: Use utilities from `utils_plotting.py` for custom visualizations

## Dependencies

These scripts require:
- JAX and jax-numpy (GPU support recommended)
- jesterTOV (the core EOS modeling framework)
- jim (flowMC wrapper for sampling)
- FlowJAX (normalizing flow library)
- Standard scientific Python stack (numpy, matplotlib, scipy)
- ArviZ (for statistical analysis)
- Corner (for corner plots)