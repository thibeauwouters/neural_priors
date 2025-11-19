# GW_runs

This directory contains scripts and configuration files for running gravitational wave parameter estimation (PE) with normalizing flow priors.

*Note from human*: These might be outdated, as we switched to using `bilby_pipe` to perform the runs. Please reach out if you need the exact same config files used for this work, but have a look at `GW_runs_jarvis`, which is used more recently.

## Overview

The parameter estimation pipeline analyzes gravitational wave events using bilby with custom normalizing flow priors that encode equation of state constraints. Runs can be configured for different source types (BNS vs NSBH), population models, and EOS datasets.

## Key Scripts

### Parameter Estimation

**`pe.py`** - Main parameter estimation script
- Runs Bayesian inference on gravitational wave strain data
- Integrates NF priors from `../NFprior/models/`
- Supports multiple source hypotheses (BNS, NSBH)
- Uses relative binning for computational efficiency
- Handles both default (agnostic) and NF-informed priors

**Key Arguments:**
- `--GW-event`: Event name (GW170817, GW190425, GW230529)
- `--prior-name`: Source type (bns, nsbh, default)
- `--eos-samples-name`: EOS dataset for NF prior (radio, radio_chiEFT, etc.)
- `--population-type`: Mass population model (uniform, gaussian, double_gaussian)
- `--use-flowjax`: Use JAX-based flows instead of PyTorch
- `--waveform-model`: Waveform approximant (default: IMRPhenomXP_NRTidalv3)
- `--relative-binning-delta`: Likelihood approximation error tolerance (default: 1e-2)

**Usage:**
```bash
# Run BNS analysis with uniform population and radio EOS prior
python pe.py --GW-event GW170817 --prior-name bns --population-type uniform --eos-samples-name radio --use-flowjax

# Run NSBH analysis with default agnostic prior
python pe.py --GW-event GW170817 --prior-name nsbh --population-type default

# Run with custom waveform and sampling settings
python pe.py --GW-event GW190425 --prior-name bns --waveform-model IMRPhenomD_NRTidalv2 --nlive 2000
```

**Important Notes:**
- Requires modified bilby branch with NF support (see main README.md)
- Sets MPI-compatible environment variables for NF prior stability
- Uses relative binning for fast likelihood evaluation
- Results saved to `{output_dir}/{event}/{source}/{population}/{eos}/`

### Utilities

**`extract_posterior_samples.py`** - Extract samples from PE results
- Converts result files to standardized NPZ format
- Extracts posterior samples for downstream analysis
- Handles both HDF5 and legacy formats

**Usage:**
```bash
# Extract samples from specific run
python extract_posterior_samples.py --event GW170817 --source bns --population uniform --eos radio

# Extract from multiple runs
python extract_posterior_samples.py --event GW170817 --source bns --all-populations
```

**`generate_dag.py`** - Generate HTCondor DAG files for batch processing
- Creates directed acyclic graphs for parallel PE runs
- Automates submission of multiple configurations
- Organizes dependencies between analysis stages

**Usage:**
```bash
# Generate DAG for all GW170817 analyses
python generate_dag.py --event GW170817

# Generate DAG for specific configurations
python generate_dag.py --event GW230529 --sources bns nsbh --populations uniform gaussian
```

## Directory Structure

```
GW_runs/
├── pe.py                          # Main PE script
├── extract_posterior_samples.py   # Sample extraction utility
├── generate_dag.py                # HTCondor DAG generation
├── dag_files/                     # HTCondor submission files
├── final_results/                 # Production PE results (NPZ format)
│   ├── GW170817/
│   │   ├── bns/
│   │   │   ├── uniform/
│   │   │   │   └── radio/samples.npz
│   │   │   └── default/samples.npz
│   │   └── nsbh/
│   │       └── [similar structure]
│   ├── GW190425/
│   └── GW230529/
└── [event-specific config directories]
    ├── GW170817/
    ├── GW190425/
    └── GW230529/
```

## Result File Format

### Standard Runs
Path: `final_results/{event}/{source}/{population}/{eos}/samples.npz`

Example: `final_results/GW170817/bns/uniform/radio/samples.npz`

### Default Runs
Path: `final_results/{event}/{source}/default/samples.npz`

Example: `final_results/GW170817/bns/default/samples.npz`

All results use NPZ format containing posterior samples as numpy arrays accessible via:
```python
data = np.load('samples.npz')
samples = data['posterior']  # Or specific parameter keys
```

## Source Types

### BNS (Binary Neutron Star)
- Both objects are neutron stars
- Both λ₁ and λ₂ > 0
- Constraint: λ₂ ≥ λ₁ (enforced via mass ordering m₁ ≥ m₂)

### NSBH (Neutron Star - Black Hole)
- Primary is black hole (λ₁ = 0)
- Secondary is neutron star (λ₂ > 0)
- No tidal deformability constraint on primary

### Default
- Agnostic uniform priors without EOS constraints
- Used as baseline for Bayes factor calculations
- No NF prior applied

## Population Types

- **uniform**: Uniform distribution over neutron star masses
- **gaussian**: Single Gaussian mass distribution (μ=1.33, σ=0.09)
- **double_gaussian**: Bimodal Gaussian mass distribution
- **{event_name}**: Event-specific conditional models (advanced usage)

## EOS Datasets

Available EOS posterior samples (matched to NF model names):
- `radio`: Pulsar radio observations only
- `radio_chiEFT`: Radio + chiral effective field theory
- `radio_NICER`: Radio + NICER constraints
- `radio_GW170817`: Radio + GW170817 multi-messenger
- `radio_chiEFT_NICER`: Combined radio, χEFT, and NICER

## Waveform Models

Supported waveform approximants:
- `IMRPhenomXP_NRTidalv3` (default): Precessing waveform with NRTidal calibration
- `IMRPhenomD_NRTidalv2`: Non-precessing alternative
- Other IMRPhenom variants supported by bilby

## Computational Considerations

### Relative Binning
- Speeds up likelihood evaluation by ~100x
- Controlled by `--relative-binning-delta` (error tolerance)
- Lower delta = more bins = slower but more accurate
- Default 1e-2 provides good speed/accuracy tradeoff

### Sampling
- Default nested sampling with dynesty
- Typical `nlive=2000` for production runs
- Can use parallel tempering or other samplers via bilby options

### MPI Compatibility
- Script sets environment variables for thread safety with NF priors
- Compatible with HTCondor and SLURM job schedulers
- Use `--outdir` to specify cluster-specific output paths

## Common Workflows

### Single Event Analysis
```bash
# Run BNS and NSBH for comparison
python pe.py --GW-event GW170817 --prior-name bns --population-type uniform --eos-samples-name radio --use-flowjax
python pe.py --GW-event GW170817 --prior-name nsbh --population-type uniform --eos-samples-name radio --use-flowjax
```

### Batch Processing
```bash
# Generate DAG for all configurations
python generate_dag.py --event GW170817

# Submit to HTCondor
condor_submit_dag dag_files/GW170817_analysis.dag
```

### Result Extraction
```bash
# Extract samples after PE completes
python extract_posterior_samples.py --event GW170817 --source bns --population uniform --eos radio
```

## Integration with Other Components

- **Input Data**: Strain data and PSDs from `../data/{event}/`
- **NF Models**: Loaded from `../NFprior/models/{population}/{eos}/`
- **Output**: Saved to `final_results/` for use by `../plots/` and `../bayes_factors/`

## Troubleshooting

**NF Prior Loading Issues**
- Verify NF model exists at expected path
- Check population and EOS names match exactly
- Ensure flowjax/glasflow backend matches training

**Relative Binning Failures**
- Increase `--relative-binning-delta` if too few bins
- Check `--minimum-bin-threshold` (default 1000)
- Verify reference parameters are reasonable

**Memory Issues**
- Reduce `--nlive` for lower memory usage
- Use fewer parallel workers in MPI runs
- Decrease `--relative-binning-delta` to reduce bin count
