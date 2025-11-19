# Figure1

This directory contains scripts for generating Figure 1 of the publication, which provides a visual overview of the analysis methodology and workflow.

## Overview

Figure 1 is a schematic diagram illustrating the complete analysis pipeline from EOS constraints to gravitational wave parameter estimation. It combines multiple auxiliary plots assembled in Inkscape to create a comprehensive visual summary.

## Key Script

### `make_figure1.py` - Generate Figure 1 components

Creates individual plot elements that are assembled into the final figure using Inkscape or similar vector graphics software.

**Generated Components:**

**1. EOS Constraint Panel**
- Visualizes neutron star mass-radius relation from different observations
- Shows posterior distributions from PSRs, NICER, χEFT, GW170817
- Illustrates how constraints combine to restrict EOS parameter space

**2. Normalizing Flow Architecture**
- Schematic of flow transformation
- Training data (EOS samples) → Flow model → Prior distribution
- Highlights learned mass-lambda correlations

**3. Parameter Space Transformation**
- Unit cube → Gaussian base → Physical parameters
- Shows bijective mapping with constraints
- Illustrates q ∈ [0.1, 1.0], λ > 0 enforcement

**4. GW Strain Data**
- Time-domain strain for representative event (e.g., GW170817)
- Shows H, L, V detector data
- Highlights signal region

**5. Posterior Samples**
- Corner plot excerpt showing key parameters
- Comparison of default vs NF-informed priors
- Demonstrates prior impact on inference

**6. Bayes Factor Visualization**
- Schematic showing BNS vs NSBH evidence
- Jeffreys scale interpretation
- Color-coded classification results

**Usage:**
```bash
# Generate all Figure 1 components
python make_figure1.py

# Generate specific panel
python make_figure1.py --panel eos_constraints

# Customize output format
python make_figure1.py --format pdf --dpi 600

# Save to specific directory
python make_figure1.py --output-dir figures/
```

**Output:**
Individual PDF/PNG files saved to `figures/` subdirectory, ready for assembly in Inkscape.

## Figure Assembly Workflow

### 1. Generate Components
```bash
cd Figure1/
python make_figure1.py
```

### 2. Open in Inkscape
```bash
inkscape figures/figure1_template.svg
```

### 3. Import Components
- Import each PDF panel into Inkscape
- Position and align elements
- Add annotations, arrows, labels
- Adjust colors for consistency

### 4. Export Final Figure
- Export as high-resolution PDF
- Verify text is not converted to paths (editable)
- Check dimensions match journal requirements

### 5. Add to Manuscript
```latex
\begin{figure*}
\centering
\includegraphics[width=\textwidth]{figure1.pdf}
\caption{Analysis pipeline schematic...}
\label{fig:pipeline}
\end{figure*}
```

## Panel Descriptions

### EOS Constraints Panel

**Purpose:** Show how nuclear physics observations constrain neutron star properties

**Elements:**
- Mass-radius (M-R) plane with posterior distributions
- Different colored regions for PSRs, NICER, χEFT, GW170817
- Causality limit (c_s = c boundary)
- Representative EOS curves

**Key Message:** Multiple complementary observations combine to tightly constrain EOS

### NF Training Panel

**Purpose:** Illustrate how NF learns EOS-informed prior

**Elements:**
- Training data scatter (m₁, m₂, λ₁, λ₂)
- Flow architecture schematic
- Learned prior samples overlaid on training data
- Loss curve showing training convergence

**Key Message:** NF successfully captures complex correlations from EOS physics

### GW Data Panel

**Purpose:** Show real gravitational wave observation

**Elements:**
- Whitened strain timeseries for H1, L1, V1
- Highlighted merger signal
- Time-frequency representation (optional)
- GPS time and event name

**Key Message:** Clear signal detected in multiple detectors

### Posterior Comparison Panel

**Purpose:** Demonstrate impact of NF prior on inference

**Elements:**
- 2D posterior contours (e.g., chirp mass vs λ̃)
- Default prior (gray) vs NF prior (colored)
- Injected/true values (for validation plots)
- Credible intervals

**Key Message:** Physics-informed priors improve parameter constraints

### Classification Panel

**Purpose:** Visualize source classification via Bayes factors

**Elements:**
- Bayes factor values for each event
- Jeffreys scale with color coding
- BNS vs NSBH model schematics
- Evidence ratio visualization

**Key Message:** Method successfully classifies sources with quantified confidence

## Style Guidelines

### Color Scheme

**EOS Datasets:**
- Radio: Blue
- NICER: Orange
- χEFT: Green
- GW170817: Red

**Source Types:**
- BNS: Teal/cyan
- NSBH: Coral/red-orange

**Evidence Strength:**
- Decisive: Dark shades
- Strong: Medium shades
- Substantial: Light shades
- Inconclusive: Gray

### Fonts

**Main Text:**
- Font: Helvetica or Arial (sans-serif)
- Size: 10-12 pt for labels
- Size: 8-10 pt for axis labels

**Math:**
- Use LaTeX rendering for equations
- Consistent notation with manuscript

### Layout

**Aspect Ratio:**
- Two-column figure: 7 inches width
- Single-column: 3.5 inches width
- Height: Maintain readability

**Spacing:**
- Adequate whitespace between panels
- Clear visual hierarchy
- Aligned elements

## Directory Structure

```
Figure1/
├── make_figure1.py          # Main script
├── figures/                 # Generated components
│   ├── panel_a_eos.pdf
│   ├── panel_b_nf.pdf
│   ├── panel_c_gw.pdf
│   ├── panel_d_posterior.pdf
│   ├── panel_e_bayes.pdf
│   └── figure1_final.pdf    # Assembled figure
└── README.md
```

## Customization

### Modify Panel Content

Edit `make_figure1.py` to change:
- Which events to show
- Parameter combinations for posteriors
- EOS datasets to include
- Color schemes and styling

### Adjust Figure Dimensions

For different journal requirements:
```python
# In make_figure1.py
FIG_WIDTH = 7.0  # inches (two-column)
FIG_HEIGHT = 9.0  # inches
DPI = 600  # high resolution for publication
```

### Change Plot Style

Apply consistent matplotlib style:
```python
import matplotlib.pyplot as plt

# Publication style
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'sans-serif',
    'axes.linewidth': 1.0,
    'grid.linewidth': 0.5,
    'lines.linewidth': 1.5
})
```

## Integration with Manuscript

### Caption Template

```latex
\caption{
    \textbf{Analysis pipeline for EOS-informed GW source classification.}
    \textbf{(a)} Nuclear physics constraints on neutron star equation of state
    from pulsar observations (blue), NICER X-ray measurements (orange),
    chiral effective field theory (green), and GW170817 (red).
    \textbf{(b)} Normalizing flow training on EOS posterior samples to learn
    mass-lambda correlations.
    \textbf{(c)} Gravitational wave strain data from LIGO Hanford (H1),
    Livingston (L1), and Virgo (V1) detectors for GW170817.
    \textbf{(d)} Posterior distributions comparing default agnostic prior (gray)
    and NF-informed prior (colored), showing improved constraints.
    \textbf{(e)} Bayes factors for BNS vs NSBH classification across events,
    color-coded by evidence strength (Jeffreys scale).
}
\label{fig:pipeline}
```

## Common Workflows

### Generate Fresh Figure Components
```bash
python make_figure1.py --regenerate-all
```

### Update Specific Panel
```bash
python make_figure1.py --panel posterior_comparison --event GW170817
```

### Export High-Resolution Final Figure
```bash
# After assembly in Inkscape
inkscape --export-pdf=figure1_final.pdf --export-dpi=600 figure1.svg
```

## Troubleshooting

### Text Overlaps or Too Small
- Increase font sizes in script
- Adjust panel dimensions
- Reduce number of tick labels

### Colors Not Matching Manuscript
- Define color palette centrally
- Use consistent RGB/hex codes
- Test print vs screen appearance

### PDF Import Issues in Inkscape
- Ensure fonts embedded in PDF
- Convert text to paths if needed (last resort)
- Check PDF version compatibility

### Figure Too Large for Journal
- Reduce DPI for submission (300 sufficient)
- Compress with `gs` or similar tools
- Simplify complex elements

## Tips for Publication-Quality Figures

1. **Vector graphics preferred:** Use PDF format, not PNG
2. **High DPI for rasters:** 600 DPI for any embedded images
3. **Editable text:** Don't convert fonts to paths until final
4. **Consistent notation:** Match manuscript symbols exactly
5. **Colorblind-friendly:** Test with ColorBrewer or similar tools
6. **Black/white legible:** Ensure clarity without color
7. **Size appropriately:** Match journal column widths
8. **Label clearly:** All axes, panels, units specified
