# data

This directory contains gravitational wave strain data, power spectral densities (PSDs), EOS posterior samples, and reference datasets for parameter estimation analyses.

*Note from human:* Some files mentioned below might not exist in the Github repo if they are too large -- please reach out if you'd like to use them.

## Overview

The data directory organizes all input data required for gravitational wave parameter estimation, including:
- GW strain data (frame files) for real events
- Detector PSDs for likelihood calculation
- EOS posterior samples for normalizing flow training
- Reference posteriors from external analyses (Hauke, Adrian)
- Calibration uncertainty files

## Directory Structure

```
data/
├── GW170817/              # GW170817 event data
├── GW190425/              # GW190425 event data
├── GW230529/              # GW230529 event data
├── eos/                   # EOS posterior samples
├── adrian/                # Adrian's reference analyses
├── hauke/                 # Hauke's reference analyses
└── psds/                  # Design sensitivity PSDs
```

## GW Event Data

### GW170817/ - First multi-messenger BNS event

**Strain Data (Frame Files):**
- `H-H1_LOSC_CLN_16_V1-1187007040-2048.gwf` - LIGO Hanford data
- `L-L1_LOSC_CLN_16_V1-1187007040-2048.gwf` - LIGO Livingston data
- `V-V1_LOSC_CLN_16_V1-1187007040-2048.gwf` - Virgo data

**PSDs:**
- `h1_psd.txt` - Hanford power spectral density
- `l1_psd.txt` - Livingston power spectral density
- `v1_psd.txt` - Virgo power spectral density

**Calibration Uncertainties:**
- `Feb-20-2018_O2_LHO_*_RelativeResponseUncertainty_FinalResults.txt` - LIGO Hanford calibration
- `Feb-20-2018_O2_LLO_*_RelativeResponseUncertainty_FinalResults.txt` - LIGO Livingston calibration
- `V_calibrationUncertaintyEnvelope_*.txt` - Virgo calibration

**Configuration:**
- `GW170817.prior` - Prior file for parameter estimation
- `config.ini` - Analysis configuration

**Utilities:**
- `compare_datasets.py` - Compare different data versions
- `psd_comparison.png`, `strain_comparison.png` - Diagnostic plots

### GW190425/ - High-mass BNS or NSBH candidate

**Strain Data:**
- `L-L1_GWOSC_16KHZ_R1-1240213455-4096.gwf` - LIGO Livingston data
- `V-V1_GWOSC_16KHZ_R1-1240213455-4096.gwf` - Virgo data

Note: Only L1 and V1 available (H1 not observing)

**PSDs:**
- `glitch_median_PSD_forLI_L1_srate8192.txt` - L1 PSD (glitch-resistant median)
- `glitch_median_PSD_forLI_V1_srate8192.txt` - V1 PSD

**Reference Data:**
- `posterior_samples.h5` - Published posterior samples

### GW230529/ - Ambiguous source classification

**Strain Data:**
- `L-L1_GWOSC_16KHZ_R1-1369417271-4096.gwf` - LIGO Livingston data

Note: Single detector observation (high-mass event during O4)

**PSDs:**
- `psd_4096.dat` - Livingston PSD estimate

**Configuration:**
- `ini.ini` - Analysis configuration

**Reference Data:**
- `posterior_samples.h5` - Published posterior samples

**Utilities:**
- `extract_params.py` - Extract parameters from reference posteriors

## EOS Posterior Samples

### eos/ - Neutron star equation of state constraints

Contains posterior samples from various nuclear physics observations used to train normalizing flows.

**Datasets:**

**`radio/`** - Pulsar radio observations
- Mass and radius measurements from binary pulsars
- Provides mass-lambda correlations from timing observations

**`radio_chiEFT/`** - Radio + chiral effective field theory
- Combines pulsar observations with low-density χEFT constraints
- Tighter constraints on EOS at lower densities

**`radio_NICER/`** - Radio + NICER X-ray observations
- Includes radius measurements from NICER mission
- Improved constraints on NS radius

**`radio_GW170817/`** - Radio + GW170817 multi-messenger
- Incorporates GW170817 tidal deformability constraints
- Most restrictive dataset for BNS physics

**`radio_chiEFT_NICER/`** - Combined radio, χEFT, and NICER
- Full combination of complementary observations
- Strongest constraints on nuclear EOS

**`hauke/`** - Hauke's EOS dataset
- Alternative EOS posteriors for comparison

**`all/`** - Combined EOS samples

**File Format:**
Each EOS directory contains NPZ files with neutron star parameters:
- `m1`, `m2`: Component masses [M☉]
- `lambda_1`, `lambda_2`: Tidal deformabilities
- Derived parameters: chirp mass, mass ratio, tilde parameters

## Reference Analyses

### adrian/ - Adrian's parameter estimation results

Contains reference PE results from Adrian's analyses for comparison.

**Structure:**
```
adrian/
├── GW170817/
│   ├── [posterior samples]
│   └── [configuration files]
├── GW190425/
└── [utility scripts]
    ├── check_default_cosmology.py
    ├── marginalize_adrian_data.py
    └── report_priors.py
```

**Purpose:**
- Validate analysis setup and configuration
- Compare posteriors from different analysis pipelines
- Check cosmology and prior consistency

### hauke/ - Hauke's parameter estimation results

Contains reference PE results from Hauke's analyses.

**Structure:**
```
hauke/
├── GW170817/
├── GW190425/
└── [utility scripts]
    ├── check_bilby_default.py
    ├── marginalize_hauke_data.py
    └── report_priors.py
```

**Purpose:**
- Cross-validate analysis methods
- Compare different prior choices
- Benchmark computational approaches

## Design Sensitivity PSDs

### psds/ - Future detector sensitivities

**`AplusDesign_PSD.txt`** - Advanced LIGO Plus design sensitivity
- Target sensitivity for A+ upgrade
- Used for injection studies

**`avirgo_O5high_NEW_PSD.txt`** - Advanced Virgo O5 high sensitivity
- Expected Virgo sensitivity for O5 observing run
- Used for future event simulations

## File Formats

### Frame Files (.gwf)
Gravitational Wave Frame format containing strain time series.
- Standard format for LIGO/Virgo data
- Read with `gwpy` or `pycbc` libraries
- Contains metadata: GPS time, sample rate, calibration

### PSD Files (.txt, .dat)
Text files with two columns:
- Column 1: Frequency [Hz]
- Column 2: Power spectral density [Hz^-1]

### HDF5 Files (.h5)
Hierarchical data format containing:
- Posterior samples (structured arrays)
- Metadata (configuration, priors, etc.)
- Log evidence values

### NPZ Files (.npz)
NumPy compressed archives containing:
- Parameter samples as named arrays
- Accessed via `np.load('file.npz')['parameter_name']`

## Data Access

### LIGO Open Science Center (LOSC/GWOSC)

Strain data and PSDs downloaded from:
https://www.gw-openscience.org/

**Events:**
- GW170817: https://www.gw-openscience.org/events/GW170817/
- GW190425: https://www.gw-openscience.org/events/GW190425/
- GW230529: https://www.gw-openscience.org/events/GW230529/

### EOS Data Sources

Nuclear physics constraints from:
- **Pulsars**: Antoniadis et al., Fonseca et al., Miller et al.
- **NICER**: Riley et al. (2019, 2021)
- **χEFT**: Hebeler et al., Drischler et al.
- **GW170817**: Abbott et al. (2017, 2018)

## Usage in Analysis Pipeline

### Parameter Estimation (`../GW_runs/pe.py`)
```python
# Load strain data
data_files = {
    'H1': 'data/GW170817/H-H1_LOSC_CLN_16_V1-1187007040-2048.gwf',
    'L1': 'data/GW170817/L-L1_LOSC_CLN_16_V1-1187007040-2048.gwf',
    'V1': 'data/GW170817/V-V1_LOSC_CLN_16_V1-1187007040-2048.gwf'
}

# Load PSDs
psd_files = {
    'H1': 'data/GW170817/h1_psd.txt',
    'L1': 'data/GW170817/l1_psd.txt',
    'V1': 'data/GW170817/v1_psd.txt'
}
```

### NF Training (`../NFprior/train_NF_prior.py`)
```python
# Load EOS samples
eos_data = np.load('data/eos/radio/samples.npz')
m1 = eos_data['m1']
m2 = eos_data['m2']
lambda_1 = eos_data['lambda_1']
lambda_2 = eos_data['lambda_2']
```

## Data Validation

### Strain Data Checks
- GPS times match published values
- Sample rates appropriate (16384 Hz typical)
- Data duration sufficient (typically 4096 s around event)

### PSD Verification
- Frequency range covers analysis band
- No unphysical features (negative values, discontinuities)
- Consistent with detector noise characteristics

### EOS Sample Quality
- Physical mass ranges (0.1-3 M☉ typical)
- Positive tidal deformabilities
- Causality constraints satisfied (c_s < c)

## Troubleshooting

### Missing Data Files
- Check GWOSC for latest data releases
- Verify file paths in analysis scripts
- Ensure sufficient disk space for downloads

### Frame File Reading Issues
- Install required libraries: `gwpy`, `pycbc`, or `lalframe`
- Check frame file integrity with `FrCheck` utility
- Verify channel names match event

### EOS Data Format
- Ensure consistent parameter naming across datasets
- Check for NaN or infinite values
- Validate mass ordering (m1 ≥ m2 convention)
