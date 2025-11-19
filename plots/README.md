# plots

This directory contains scripts for visualizing parameter estimation results, creating publication-quality figures, and analyzing gravitational wave posteriors.

## Overview

Visualization scripts generate corner plots, prior comparisons, log-likelihood distributions, and other diagnostic plots for gravitational wave parameter estimation results. Scripts range from quick diagnostic plots to publication-ready figures with careful styling.

## Key Scripts

### Publication Figures

**`money_plots.py`** - Publication-ready corner plots for paper figures
- Creates carefully configured corner plots for specific events and parameters
- Hardcoded settings for publication quality (fonts, colors, ranges)
- Compares BNS vs NSBH posteriors
- Supports event-specific customizations

**Usage:**
```bash
# Generate main corner plots for all events
python money_plots.py

# Create specific event plot
python money_plots.py --event GW170817 --parameters chirp_mass lambda_tilde
```

**`cornerplots.py`** - Batch corner plot generation
- Automated corner plot creation for multiple configurations
- Loops over events, sources, populations, and EOS datasets
- Produces diagnostic corner plots for all parameter combinations
- Less polished than money_plots but comprehensive coverage

**Usage:**
```bash
# Generate corner plots for all runs
python cornerplots.py

# Generate for specific event
python cornerplots.py --event GW170817
```

### Prior Analysis

**`plot_priors.py`** - Visualize normalizing flow priors
- Plots NF prior distributions for masses and tidal parameters
- Compares different EOS datasets and population models
- Shows 1D and 2D marginal distributions
- Validates NF training quality

**Usage:**
```bash
# Plot priors for uniform population
python plot_priors.py --population-type uniform

# Compare multiple EOS datasets
python plot_priors.py --population-type uniform --eos-names radio radio_chiEFT radio_NICER
```

**`plot_priors_bns_zoom.py`** - Zoomed prior plots for BNS region
- Focuses on BNS-relevant parameter space
- Higher resolution in key regions
- Useful for understanding prior behavior near constraints

**`prior_vs_posterior.py`** - Compare priors and posteriors
- Overlays prior and posterior distributions
- Quantifies prior informativeness
- Identifies parameters constrained by data vs prior

**Usage:**
```bash
# Compare prior/posterior for specific run
python prior_vs_posterior.py --event GW170817 --source bns --population uniform --eos radio
```

### Diagnostic Plots

**`logL.py`** - Log-likelihood distribution analysis
- Plots log-likelihood values across posterior samples
- Identifies prior-dominated vs likelihood-dominated samples
- Diagnoses convergence and sampling issues
- Compares likelihood distributions across runs

**Usage:**
```bash
# Plot log-likelihood for GW170817
python logL.py --event GW170817

# Compare BNS and NSBH log-likelihoods
python logL.py --event GW170817 --compare-sources
```

**`m1m2.py`** - Component mass analysis
- Plots component masses (m₁, m₂) in 2D
- Shows mass constraints from different priors
- Compares BNS and NSBH mass posteriors

**`spin_plots.py`** - Spin parameter analysis
- Analyzes spin magnitude and tilt angle posteriors
- Compares spin distributions across sources
- Identifies spin constraints from waveform data

**`gw230529_spins.py`** - GW230529-specific spin analysis
- Detailed spin investigation for ambiguous source classification
- Explores precession effects
- Correlates spins with tidal parameters

### Utilities

**`utils.py`** - Shared plotting utilities and helper functions
- Path construction for result files
- Data loading and preprocessing
- Parameter transformations (component masses ↔ chirp mass + q)
- Tidal parameter conversions (λ₁, λ₂ ↔ λ̃, δλ̃)
- Color schemes and style configurations
- Common plotting functions

**Key Functions:**
- `construct_result_path()`: Build paths to result files
- `load_posterior_data()`: Load samples from NPZ/HDF5 files
- `setup_matplotlib_style()`: Apply publication formatting
- `convert_lambdas_with_verbose()`: Transform tidal parameters
- `calculate_corner_plot_ranges()`: Determine plot bounds

**`tell_me_Mc_source_and_z.py`** - Quick parameter summary
- Prints chirp mass, redshift, and source-frame mass for event
- Useful for quick checks and understanding posteriors

**Usage:**
```bash
# Get parameters for specific run
python tell_me_Mc_source_and_z.py --event GW170817 --source bns --population uniform --eos radio
```

**`make_cosmology_interpolator.py`** - Create cosmology lookup tables
- Builds interpolators for luminosity distance ↔ redshift
- Used by other scripts for efficient cosmological conversions
- Caches results for repeated use

### Event-Specific Scripts

**`GW170817.py`** - GW170817-specific analyses
- Custom plots and diagnostics for the first multi-messenger event
- Historical reference for GW170817 analysis workflow

**`old_plot_priors.py`** - Legacy prior plotting script
- Older version of prior visualization
- Kept for compatibility with older analysis notebooks

## Directory Structure

```
plots/
├── money_plots.py              # Publication corner plots
├── cornerplots.py              # Batch corner plot generation
├── plot_priors.py              # NF prior visualization
├── prior_vs_posterior.py       # Prior/posterior comparison
├── logL.py                     # Log-likelihood analysis
├── utils.py                    # Shared utilities
├── [other analysis scripts]
├── figures/                    # Generated plot outputs
│   ├── money_plots/           # Publication-ready figures
│   ├── priors/                # Prior distribution plots
│   ├── prior_vs_posterior/    # Comparison plots
│   ├── logL/                  # Log-likelihood distributions
│   ├── logPosterior/          # Log-posterior distributions
│   ├── logPrior/              # Log-prior distributions
│   ├── mass_comparison/       # Component mass plots
│   ├── spin_analysis/         # Spin parameter plots
│   ├── spin_plots/            # Detailed spin figures
│   └── networkSNR/            # Signal-to-noise ratio plots
└── data/                       # Cached data for plotting
```

## Common Workflows

### Generate Publication Figures
```bash
# Create all publication corner plots
python money_plots.py

# Generate prior comparison plots
python plot_priors.py --population-type uniform
```

### Diagnostic Analysis
```bash
# Check log-likelihood distributions
python logL.py --event GW170817

# Compare prior and posterior
python prior_vs_posterior.py --event GW170817 --source bns --population uniform --eos radio

# Quick parameter summary
python tell_me_Mc_source_and_z.py --event GW170817 --source bns --population uniform --eos radio
```

### Batch Visualization
```bash
# Generate corner plots for all runs
python cornerplots.py

# Create spin analysis for all events
python spin_plots.py
```

## Plotting Conventions

### Colors

**EOS Datasets** (defined in `utils.py`):
- Radio: Specific color scheme
- Radio + χEFT: Alternative color
- Radio + NICER: Third color
- Etc. (see `EOS_COLORS` in utils.py)

**Source Types**:
- BNS: Typically blue/teal
- NSBH: Typically orange/red
- Default: Gray

### Parameter Labels

LaTeX-formatted labels defined in `PARAMETER_LATEX_LABELS` (utils.py):
- `chirp_mass`: $\mathcal{M}_c$ [$M_{\odot}$]
- `mass_ratio`: $q$
- `lambda_tilde`: $\tilde{\Lambda}$
- `lambda_1`, `lambda_2`: $\Lambda_1$, $\Lambda_2$
- Etc.

### Style Parameters

Publication plots use:
- Font: Computer Modern Serif (LaTeX-like)
- Label fontsize: 28
- Tick fontsize: 24
- No grid (clean appearance)
- usetex: True (for LaTeX rendering)

## Output Formats

Figures saved as:
- PDF: Vector graphics for publications
- PNG: Raster graphics for presentations/web

Typical DPI: 300 for high-resolution outputs

## Integration with Analysis Pipeline

**Input Data**: Loads posterior samples from `../final_results/` or `../GW_runs/final_results/`

**NF Prior Data**: Can load NF samples from `../NFprior/models/` for prior visualization

**Output**: Saves figures to `figures/` subdirectories organized by plot type

## Parameter Transformations

Scripts automatically handle conversions between parameterizations:
- Component masses (m₁, m₂) ↔ Chirp mass ($\mathcal{M}_c$) + mass ratio (q)
- Detector-frame masses ↔ Source-frame masses (via redshift)
- Individual lambdas (λ₁, λ₂) ↔ Tilde parameters (λ̃, δλ̃)
- Luminosity distance ↔ Redshift (via cosmology)

Transformations implemented in `utils.py` with proper Jacobian handling where needed.

## Tips for Custom Plots

1. **Start with cornerplots.py** for quick diagnostics
2. **Use money_plots.py** as template for publication figures
3. **Import from utils.py** for consistent styling and data loading
4. **Check parameter names** with `tell_me_Mc_source_and_z.py` before plotting
5. **Use PARAMETER_LATEX_LABELS** for consistent parameter naming across plots
