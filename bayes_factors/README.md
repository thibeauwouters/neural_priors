# bayes_factors

This directory contains scripts for calculating and visualizing Bayes factors from gravitational wave parameter estimation runs to assess source classification (BNS vs NSBH).

## Overview

Bayes factors quantify the relative evidence for different source hypotheses (BNS vs NSBH) given the gravitational wave data. Scripts here collect log evidence values from parameter estimation results, compute Bayes factors, and generate publication-ready tables.

## Key Scripts

### `collect_all_bayes_factors.py` - Main Bayes factor calculation script

Systematically collects log evidence values from all PE runs and computes Bayes factors for source classification.

**Functionality:**
- Scans directory structure: `{event}/{source}/{population}/{eos}/`
- Extracts log evidence from HDF5 result files
- Computes Bayes factors: log(BF) = log(Z_BNS) - log(Z_NSBH)
- Organizes results in source-first structure for table generation
- Handles both NF-informed and default (agnostic) priors
- Generates LaTeX tables for publication

**Output Structure:**
```python
{
    "bns": {
        "uniform": {
            "radio": {
                "GW170817": bayes_factor_value,
                "GW190425": bayes_factor_value,
                ...
            }
        }
    },
    "nsbh": {...},
    "defaults": {
        "bns": {"GW170817": bf_value, ...},
        "nsbh": {...}
    }
}
```

**Usage:**
```bash
# Collect Bayes factors from default location
python collect_all_bayes_factors.py

# Specify custom base directory
python collect_all_bayes_factors.py --base-dir /path/to/results

# Ignore GW170817-constrained EOS datasets
python collect_all_bayes_factors.py --ignore-gw170817-eos
```

**Generated Outputs:**
- `bayes_factors.json`: Complete Bayes factor data
- LaTeX tables printed to stdout
- Organized by source type and population

### `old_collect_all_bayes_factors.py` - Legacy version

Previous implementation of Bayes factor collection with different organization scheme. Kept for reference and backward compatibility with older analyses.

### `jeffreys_colorbar.py` - Jeffreys scale visualization

Creates colorbar showing Jeffreys scale interpretation for Bayes factors.

**Jeffreys Scale:**
- |log BF| < 1: Inconclusive
- 1 ≤ |log BF| < 2.5: Substantial evidence
- 2.5 ≤ |log BF| < 5: Strong evidence
- |log BF| ≥ 5: Decisive evidence

**Usage:**
```bash
# Generate Jeffreys scale colorbar
python jeffreys_colorbar.py
```

## Bayes Factor Interpretation

### Definition

Bayes factor for BNS vs NSBH:
```
BF_{BNS/NSBH} = P(data | BNS) / P(data | NSBH)
log(BF) = log(Z_BNS) - log(Z_NSBH)
```

Where Z is the Bayesian evidence (marginal likelihood).

### Interpretation

- **log(BF) > 0**: Favors BNS hypothesis
- **log(BF) < 0**: Favors NSBH hypothesis
- **log(BF) ≈ 0**: Data cannot distinguish between hypotheses

### Significance Levels

Following Jeffreys (1961):
- **Decisive** (|log BF| > 5): Very strong evidence for one hypothesis
- **Strong** (2.5 < |log BF| ≤ 5): Strong evidence
- **Substantial** (1 < |log BF| ≤ 2.5): Moderate evidence
- **Inconclusive** (|log BF| ≤ 1): Insufficient evidence to decide

## Directory Structure

```
bayes_factors/
├── collect_all_bayes_factors.py       # Main collection script
├── old_collect_all_bayes_factors.py   # Legacy version
├── jeffreys_colorbar.py               # Scale visualization
└── [output files]
    ├── bayes_factors.json             # Complete BF data
    └── [LaTeX table files]            # Publication tables
```

## Data Sources

### Input Files

Reads log evidence from PE results stored in HDF5 format:
```
{base_dir}/{event}/{source}/{population}/{eos}/result/result.hdf5
```

**Key HDF5 Fields:**
- `log_evidence`: Bayesian evidence value
- `log_evidence_err`: Uncertainty on evidence (if available)

### Configuration

**Events:** GW170817, GW190425, GW230529

**Source Types:** bns, nsbh

**Population Models:** uniform, gaussian, double_gaussian, {event_name}

**EOS Datasets:** radio, radio_chiEFT, radio_NICER, radio_GW170817, radio_chiEFT_NICER

### Default Runs

Special case: `{event}/{source}/default/`
- Uses agnostic uniform priors
- No NF-informed EOS constraints
- Serves as baseline comparison

## Output Format

### JSON Structure

Complete Bayes factor data saved to `bayes_factors.json`:
```json
{
  "bns": {
    "uniform": {
      "radio": {
        "GW170817": 15.3,
        "GW190425": -2.1,
        "GW230529": 0.4
      }
    }
  },
  "defaults": {
    "bns": {
      "GW170817": 12.7
    }
  },
  "log_evidence_errors": [
    "Missing error for GW170817/bns/uniform/radio"
  ]
}
```

### LaTeX Tables

Generated tables follow publication format:
- Rows: EOS datasets
- Columns: GW events
- Cells: Bayes factor values with precision
- Color-coded by Jeffreys scale (if using colorbar)

**Display Name Mappings:**

Populations:
- uniform → "Uniform"
- gaussian → "Gaussian"
- double_gaussian → "Double Gaussian"

EOS datasets:
- radio → "PSRs"
- radio_chiEFT → "PSRs+χEFT"
- radio_NICER → "PSRs+NICER"
- radio_GW170817 → "+GW170817"
- radio_chiEFT_NICER → "PSRs+χEFT+NICER"

## Common Workflows

### Calculate All Bayes Factors
```bash
# Collect from default results directory
cd bayes_factors/
python collect_all_bayes_factors.py

# Save output
python collect_all_bayes_factors.py > bayes_factor_tables.tex
```

### Generate Publication Tables
```bash
# Create tables for paper
python collect_all_bayes_factors.py --ignore-gw170817-eos > bf_table.tex

# Include Jeffreys scale colorbar
python jeffreys_colorbar.py
```

### Compare Priors
```bash
# Compare NF prior vs default prior results
python collect_all_bayes_factors.py | grep -A 20 "Default Prior"
```

## Integration with Analysis Pipeline

**Input:** Log evidence values from `../GW_runs/final_results/`

**Output:** Bayes factor tables for publication in `../money_table/`

**Visualization:** Jeffreys scale used in publication figures

## Troubleshooting

### Missing Evidence Values
- Check that PE runs completed successfully
- Verify HDF5 files contain `log_evidence` field
- Check file paths match expected structure

### Inconsistent Results
- Ensure same sampler settings across runs
- Verify convergence of nested sampling
- Check for numerical precision issues in evidence calculation

### LaTeX Table Formatting
- Adjust precision in `collect_all_bayes_factors.py`
- Modify display name mappings for custom datasets
- Update column/row ordering as needed

## Mathematical Background

### Evidence Calculation

Bayesian evidence (marginal likelihood):
```
Z = ∫ P(data | θ) P(θ) dθ
```

Where:
- P(data | θ): Likelihood
- P(θ): Prior
- θ: Model parameters

Computed by nested sampling (dynesty) during parameter estimation.

### Prior Dependence

Bayes factors depend on prior choice:
- Broad priors → lower evidence
- Informative priors (NF) → higher evidence if data consistent
- Prior volume effect important for interpretation

### Comparison Across Priors

When comparing NF-informed vs default priors:
```
ΔBF = BF_NF - BF_default
```

Positive ΔBF indicates NF prior increases evidence for that source hypothesis.