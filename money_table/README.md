# money_table

This directory contains scripts for generating publication-ready summary tables of Bayes factors and parameter estimation results.

## Overview

The "money table" consolidates key results from all analyses into comprehensive LaTeX tables for publication. Tables organize Bayes factors, source classifications, and parameter constraints across events, priors, and EOS datasets.

## Key Script

### `get_money_table.py` - Generate comprehensive results tables

Main script for creating publication-quality LaTeX tables summarizing all analysis results.

**Functionality:**
- Collects Bayes factors from `../bayes_factors/`
- Organizes results by event, source type, and prior
- Formats tables with proper LaTeX syntax
- Applies color coding based on evidence strength
- Generates multiple table variants (main, supplementary, etc.)

**Table Types:**

**1. Bayes Factor Summary Table**
- Rows: EOS datasets (priors)
- Columns: GW events
- Cells: log(BF) values for BNS vs NSBH
- Color coding by Jeffreys scale

**2. Source Classification Table**
- Categorical classifications (BNS favored, NSBH favored, Inconclusive)
- Evidence strength (Decisive, Strong, Substantial)
- Combined across different priors

**3. Parameter Comparison Table**
- Chirp mass, mass ratio, tidal parameters
- Median values with credible intervals
- Comparison across priors and source hypotheses

**4. Default vs NF Prior Comparison**
- Bayes factor differences: ΔBF = BF_NF - BF_default
- Shows impact of physics-informed priors
- Highlights cases where priors affect classification

**Usage:**
```bash
# Generate all tables
python get_money_table.py

# Generate specific table type
python get_money_table.py --table-type bayes_factors

# Output to file
python get_money_table.py > results_table.tex

# Include only specific events
python get_money_table.py --events GW170817 GW190425

# Customize formatting
python get_money_table.py --precision 2 --color-scale jeffreys
```

**Output Format:**

Example LaTeX table:
```latex
\begin{table}
\centering
\begin{tabular}{lccc}
\hline
Prior & GW170817 & GW190425 & GW230529 \\
\hline
PSRs & 15.3 & -2.1 & 0.4 \\
PSRs+$\chi_{\rm{EFT}}$ & 16.1 & -1.8 & 0.6 \\
PSRs+NICER & 17.2 & -2.3 & 0.2 \\
\hline
\end{tabular}
\caption{Bayes factors (log BF) for BNS vs NSBH classification.}
\end{table}
```

## Table Components

### Headers

**Event Names:**
- GW170817: First multi-messenger BNS
- GW190425: High-mass candidate
- GW230529: Ambiguous classification

**Prior Names:**
- PSRs: Radio pulsar observations only
- PSRs+χEFT: Radio + chiral effective field theory
- PSRs+NICER: Radio + NICER constraints
- +GW170817: Including GW170817 constraints
- PSRs+χEFT+NICER: Combined constraints
- Default: Agnostic uniform priors

### Color Coding

Based on Jeffreys scale for evidence interpretation:

**Decisive (|log BF| > 5):**
- Dark green (BNS favored) or dark red (NSBH favored)
- Very strong evidence for classification

**Strong (2.5 < |log BF| ≤ 5):**
- Green (BNS) or red (NSBH)
- Strong evidence

**Substantial (1 < |log BF| ≤ 2.5):**
- Light green (BNS) or light red (NSBH)
- Moderate evidence

**Inconclusive (|log BF| ≤ 1):**
- White or light gray
- Data insufficient for classification

### Precision and Formatting

**Bayes Factors:**
- 1 decimal place typical (e.g., 15.3)
- Scientific notation for very large values

**Parameters:**
- Median with 90% credible interval
- Format: $1.186^{+0.004}_{-0.003}$ M☉

**Percentages:**
- Classification confidence
- Format: 95% (for probability source is BNS)

## Data Sources

**Input:**
- Bayes factors from `../bayes_factors/bayes_factors.json`
- Posterior samples from `../final_results/`
- Log evidence values from PE result files

**Configuration:**
- Event list, prior list, EOS dataset names
- Table formatting parameters
- Color scheme definitions

## Directory Structure

```
money_table/
├── get_money_table.py          # Main table generation script
├── tables/                     # Generated LaTeX tables
│   ├── main_results.tex        # Primary results table
│   ├── supplementary.tex       # Extended results
│   ├── bayes_factors.tex       # BF-only table
│   └── parameters.tex          # Parameter summary
└── figures/                    # Supporting visualizations
    ├── bf_heatmap.pdf          # Bayes factor heatmap
    └── classification_pie.pdf  # Source classification breakdown
```

## Table Variants

### Main Paper Table

**Content:**
- Core results for all three events
- Primary EOS datasets only
- Bayes factors with color coding
- Compact format for space constraints

### Supplementary Material

**Content:**
- Extended results with all EOS datasets
- Additional parameter estimates
- Detailed credible intervals
- Comparison with literature values

### Comparison Tables

**Default vs NF Prior:**
- Side-by-side comparison
- ΔBF highlighting prior impact
- Cases where priors change classification

**Population Model Comparison:**
- Uniform vs Gaussian vs Double Gaussian
- Shows sensitivity to mass distribution assumptions

## Customization

### Modify EOS Display Names

Edit display name mappings in `get_money_table.py`:
```python
EOS_DISPLAY = {
    "radio": "PSRs",
    "radio_chiEFT": "PSRs+$\\chi_{\\rm{EFT}}$",
    # Add custom names here
}
```

### Adjust Color Thresholds

Modify Jeffreys scale boundaries:
```python
COLOR_THRESHOLDS = {
    'decisive': 5.0,
    'strong': 2.5,
    'substantial': 1.0
}
```

### Change Table Format

Switch between formats:
- Standard LaTeX `tabular`
- `booktabs` (professional publication style)
- `longtable` (multi-page tables)
- CSV (for external tools)

## Integration with Analysis Pipeline

**Workflow:**
1. Run all PE analyses (`../GW_runs/pe.py`)
2. Collect Bayes factors (`../bayes_factors/collect_all_bayes_factors.py`)
3. Generate money table (`get_money_table.py`)
4. Copy LaTeX tables to manuscript

**Automation:**
```bash
# Complete pipeline
cd bayes_factors/
python collect_all_bayes_factors.py

cd ../money_table/
python get_money_table.py > main_results.tex

# Include in LaTeX document
# \input{main_results.tex}
```

## Common Workflows

### Generate Main Results Table
```bash
python get_money_table.py --table-type main > main_results.tex
```

### Create Supplementary Tables
```bash
python get_money_table.py --table-type supplementary > supplementary_results.tex
```

### Export to CSV for Analysis
```bash
python get_money_table.py --format csv > results.csv
```

### Generate All Table Variants
```bash
python get_money_table.py --all-variants --output-dir tables/
```

## Validation Checks

**Before Publication:**
1. Verify Bayes factors match `bayes_factors.json`
2. Check all events have entries
3. Confirm color coding correct
4. Validate LaTeX syntax compiles
5. Cross-check with posterior plots
6. Ensure units and notation consistent

**Common Issues:**
- Missing results: Check PE runs completed
- Wrong sign BF: Verify BNS/NSBH ordering
- LaTeX errors: Escape special characters
- Inconsistent precision: Apply rounding uniformly

## Tips for Publication

1. **Simplify for main text:** Include only key results
2. **Use supplementary for details:** Full tables with all datasets
3. **Color judiciously:** Too much color distracts
4. **Include caption context:** Define log BF, explain Jeffreys scale
5. **Reference data sources:** Cite EOS constraints used
6. **Show uncertainties:** Include log evidence errors when available

## References

Tables formatted following:
- AAS Journals LaTeX guidelines
- Physical Review publication standards
- Astrophysical Journal style requirements
