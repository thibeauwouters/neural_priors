# debug

This directory contains diagnostic scripts for troubleshooting analysis issues, investigating anomalies, and validating computational methods.

## Overview

Debug scripts are used to investigate specific problems encountered during analysis, test computational components in isolation, and create diagnostic visualizations. These are typically ad-hoc scripts developed during active debugging sessions.

## Key Scripts

### `taper.py` - Tapering window investigation

Investigates frequency-domain tapering and windowing effects on gravitational wave data analysis.

**Purpose:**
- Analyze impact of data conditioning on likelihood
- Test different tapering schemes
- Diagnose edge effects and artifacts
- Validate frequency-domain transformations

**Usage:**
```bash
# Test tapering on GW170817 data
python taper.py --event GW170817

# Compare tapering methods
python taper.py --methods tukey hann --plot-comparison

# Investigate specific frequency range
python taper.py --fmin 20 --fmax 2048
```

**Output:**
- Frequency-domain plots showing taper effects
- Time-domain comparisons
- Likelihood sensitivity analysis
- Diagnostic figures in `figures/` subdirectory

**When to Use:**
- Unexpected frequency-domain artifacts
- Likelihood evaluation issues
- Suspected edge effects in data
- Validating data preprocessing pipeline

### `check_q_comparison.py` - Mass ratio comparison

Compares mass ratio (q) definitions and conventions across different analyses.

**Purpose:**
- Verify q = m2/m1 vs q = m1/m2 conventions
- Compare mass ratio posteriors
- Diagnose mass parameter inconsistencies
- Validate parameter transformations

**Usage:**
```bash
# Compare mass ratio conventions
python check_q_comparison.py --event GW170817 --source bns

# Check against reference data
python check_q_comparison.py --compare-hauke --compare-adrian
```

**Output:**
- Mass ratio distributions
- Component mass correlations
- Convention comparison plots

### `make_cornerplot.py` - Quick diagnostic corner plots

Generates simple corner plots for rapid inspection of posteriors without full plotting pipeline.

**Purpose:**
- Quick visual check of posterior samples
- Minimal dependencies (faster than full plotting scripts)
- Ad-hoc parameter combinations
- Initial data quality assessment

**Usage:**
```bash
# Quick corner plot for specific run
python make_cornerplot.py --samples ../final_results/GW170817/bns/uniform/radio/samples.npz

# Select specific parameters
python make_cornerplot.py --samples results.npz --params chirp_mass mass_ratio lambda_tilde

# Save figure
python make_cornerplot.py --samples results.npz --output debug_corner.pdf
```

**Output:**
- Simple corner plot (no fancy styling)
- Basic parameter statistics printed to stdout

**When to Use:**
- Quick sanity checks during PE runs
- Investigating suspicious posteriors
- Before running full plotting pipeline

## Directory Structure

```
debug/
├── taper.py                    # Tapering investigation
├── check_q_comparison.py       # Mass ratio validation
├── make_cornerplot.py          # Quick corner plots
├── figures/                    # Debug plots
│   ├── taper_effects.pdf
│   ├── q_comparison.pdf
│   └── [diagnostic plots]
└── README.md
```

## Common Debug Workflows

### Investigate Likelihood Issues

**Symptom:** Unexpected log-likelihood values or distributions

**Steps:**
1. Check data loading: verify strain and PSD files
2. Test tapering: `python taper.py --event GW170817`
3. Validate frequency range: check fmin, fmax settings
4. Inspect relative binning: verify bin construction

### Diagnose Parameter Inconsistencies

**Symptom:** Posteriors don't match expected values or literature

**Steps:**
1. Check mass convention: `python check_q_comparison.py`
2. Verify parameter transformations: component masses ↔ chirp mass + q
3. Validate cosmology: luminosity distance ↔ redshift
4. Quick visual check: `python make_cornerplot.py --samples results.npz`

### Debug NF Prior Issues

**Symptom:** NF prior behaves unexpectedly

**Steps:**
1. Check model loading: verify file paths
2. Test sampling: generate samples from NF alone
3. Validate constraints: ensure bounds respected
4. Compare to training data: corner plot NF samples vs training set

### Validate Bayes Factors

**Symptom:** Unexpected Bayes factor values

**Steps:**
1. Check log evidence extraction: verify HDF5 file format
2. Compare runs: ensure consistent settings across BNS/NSBH
3. Validate prior volumes: check prior ranges match
4. Inspect log-likelihood distributions: look for prior domination

## Debug Best Practices

### Quick Iteration

**Use minimal samples for testing:**
```bash
# Fast test run with low nlive
python pe.py --nlive 500 --max-samples 5000
```

**Cache intermediate results:**
- Save data products (strain, PSDs)
- Reuse relative binning bins
- Store NF samples for reuse

### Isolate Problems

**Test components independently:**
1. Data loading (strain, PSD) separately from PE
2. NF prior sampling isolated from likelihood
3. Parameter transformations with known inputs
4. Likelihood evaluation with fixed parameters

### Document Findings

**Record debug session results:**
- What was the problem?
- What diagnostic revealed it?
- What was the solution?
- Add comments to code explaining non-obvious fixes

### Clean Up

**After debugging:**
- Archive useful debug scripts
- Delete obsolete/one-off scripts
- Document any permanent changes
- Update main code if debug revealed bugs

## Integration with Main Analysis

Debug scripts typically:
1. **Import from main codebase:** Use same data loading, transformations
2. **Operate on real data:** Use actual PE results, not synthetic
3. **Create diagnostic plots:** Save to `debug/figures/` for review
4. **Inform code fixes:** Findings lead to improvements in main pipeline

## Troubleshooting Debug Scripts

### Import Errors
- Add parent directory to Python path
- Install missing diagnostic dependencies
- Check for circular imports

### File Not Found
- Use absolute paths for robustness
- Verify relative path assumptions
- Check current working directory

### Unexpected Results
- Validate input data first
- Test with known-good examples
- Compare to reference implementations

## Tips for Effective Debugging

1. **Start simple:** Minimal test case first
2. **Isolate variables:** Change one thing at a time
3. **Use prints liberally:** Debug output is free
4. **Plot everything:** Visual inspection reveals issues
5. **Compare to working code:** Reference implementations
6. **Ask for help:** Discuss with collaborators
7. **Document solutions:** Future you will thank you

## Related Directories

- `../normalization/`: Validation of NF normalization
- `../plots/`: Full production plotting scripts
- `../GW_runs/`: Main PE pipeline being debugged
- `../NFprior/`: NF training debugging utilities
