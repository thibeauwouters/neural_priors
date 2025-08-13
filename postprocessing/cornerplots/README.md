# Corner Plot Generation for GW Parameter Estimation

This directory contains tools for generating corner plots from gravitational wave parameter estimation results.

## Files

**`cornerplots.py`** - Main plotting script with command-line interface
- Creates corner plots comparing different priors/sources/populations
- Supports external data integration (Hauke, Adrian)
- Lambda parameter conversion (λ₁,λ₂ → λ̃,δλ̃)

**`utils.py`** - Utility functions and configuration
- Data loading functions (`load_posterior_data`, `load_hauke_data`, `load_adrian_data`)
- Path construction and directory scanning
- Matplotlib styling and plot constants
- Fast cosmology interpolation
- Parameter conversion utilities

## Example Usage

```bash
# Basic corner plot for GW170817 comparing BNS vs NSBH
python cornerplots.py --gw-event GW170817 --comparison-mode source

# Include external data comparisons
python cornerplots.py --gw-event GW170817 --plot-hauke --plot-adrian

# Compare across different population priors
python cornerplots.py --gw-event GW170817 --comparison-mode population --source-type bns

# Generate all corner plots for an event
python cornerplots.py --gw-event GW170817 --run-all
```