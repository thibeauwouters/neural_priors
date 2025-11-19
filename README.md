# Incorporating Neutron Star Physics into Gravitational Wave Inference with Neural Priors

Code and data for the paper *Incorporating neutron star physics into gravitational wave inference with neural priors*.

## Overview

This repository implements a novel approach to gravitational wave (GW) source classification using physics-informed priors based on neutron star equations of state (EOS). We train normalizing flows on EOS constraints from nuclear physics observations, then use these learned priors in Bayesian parameter estimation to improve binary neutron star (BNS) vs neutron star-black hole (NSBH) classification.

### Key Innovation

Traditional GW analysis uses agnostic uniform priors that ignore known astrophysical constraints. This work incorporates **nuclear physics knowledge** directly into GW inference through normalizing flow priors, enabling:

- Improved parameter constraints via physics-informed priors
- Robust source classification (BNS vs NSBH) with quantified confidence
- Efficient integration of multi-messenger observations
- Scalable framework for future GW events

### Scientific Context

Neutron star observations from pulsars, NICER X-ray measurements, chiral effective field theory, and multi-messenger GW events provide complementary constraints on the nuclear equation of state. Normalizing flows learn the complex correlations between neutron star masses and tidal deformabilities from these observations, encoding them as flexible priors for GW parameter estimation.

## Repository Structure

```
eos_source_classification/
├── NFprior/              # Normalizing flow training and evaluation
├── GW_runs/              # Parameter estimation scripts and results
├── data/                 # GW strain data and EOS samples
├── plots/                # Visualization and publication figures
├── bayes_factors/        # Bayes factor calculation
├── money_table/          # Publication-ready results tables
├── final_results/        # Consolidated PE results
├── Figure1/              # Main paper figure generation
├── injections/           # Injection studies
├── normalization/        # Prior normalization validation
├── debug/                # Diagnostic scripts
└── backup/               # Archived results
```

### Core Directories

**[NFprior/](NFprior/)** - Normalizing flow model training
- Train flows on EOS posterior samples
- Evaluate model quality and sample generation
- Supports flowjax (JAX) and glasflow (PyTorch)
- Implements parameter constraints (q ∈ [0.1, 1.0], λ > 0)

**[GW_runs/](GW_runs/)** - Gravitational wave parameter estimation
- Run bilby PE with NF or default priors
- Relative binning for computational efficiency
- BNS and NSBH source hypotheses
- HTCondor DAG generation for batch processing

**[data/](data/)** - Input data
- GW strain and PSDs for GW170817, GW190425, GW230529
- EOS posterior samples from various observations
- Reference analyses from collaborators

**[plots/](plots/)** - Visualization and analysis
- Publication-quality corner plots
- Prior vs posterior comparisons
- Log-likelihood distributions
- Parameter space visualizations

**[bayes_factors/](bayes_factors/)** - Model comparison
- Calculate Bayes factors for source classification
- Generate LaTeX tables with Jeffreys scale interpretation
- Compare NF-informed vs agnostic priors

**[final_results/](final_results/)** - Consolidated results
- PE posteriors in standardized NPZ format
- Organized by event, source, population, and EOS dataset

## Quick Start

### Prerequisites

**Modified bilby with NF support:**
```bash
git clone https://github.com/ThibeauWouters/bilby.git
cd bilby
git checkout eos_source_classification
pip install -e .
```

**Python dependencies:**
```bash
pip install jax jaxlib flowjax equinox
pip install torch glasflow
pip install numpy scipy matplotlib corner
pip install gwpy pycbc lalsuite
```

### Basic Workflow

**1. Train normalizing flow on EOS data:**
```bash
cd NFprior/
python train_NF_prior.py --use-flowjax --population-type uniform --eos-samples-name radio
```

**2. Run parameter estimation:**
```bash
cd ../GW_runs/
python pe.py --GW-event GW170817 --prior-name bns --population-type uniform --eos-samples-name radio --use-flowjax
```

**3. Generate corner plots:**
```bash
cd ../plots/
python money_plots.py
```

**4. Calculate Bayes factors:**
```bash
cd ../bayes_factors/
python collect_all_bayes_factors.py
```

## Analysis Pipeline

### Complete Analysis Workflow

```
EOS Observations → NF Training → GW Parameter Estimation → Visualization → Results Tables
     (data/)        (NFprior/)         (GW_runs/)          (plots/)     (money_table/)
```

**Step-by-step:**

1. **Prepare EOS data** (`data/eos/`)
   - Collect posterior samples from nuclear physics observations
   - Organize by constraint type (radio, NICER, χEFT, etc.)

2. **Train normalizing flows** (`NFprior/`)
   - Learn mass-lambda correlations from EOS samples
   - Validate model quality via corner plots and metrics
   - Save trained models for PE

3. **Run parameter estimation** (`GW_runs/`)
   - Analyze GW events with BNS and NSBH hypotheses
   - Use NF-informed and default priors for comparison
   - Generate posterior samples and log evidence values

4. **Visualize results** (`plots/`)
   - Create corner plots showing posteriors
   - Compare priors and posteriors
   - Analyze log-likelihood distributions

5. **Compute Bayes factors** (`bayes_factors/`)
   - Calculate evidence ratios for model selection
   - Classify sources based on Jeffreys scale
   - Generate publication tables

6. **Create paper figures** (`Figure1/`, `plots/`, `money_table/`)
   - Assemble publication-ready figures
   - Generate results tables with LaTeX formatting
   - Export high-resolution graphics

## Gravitational Wave Events

### GW170817 - First Multi-Messenger BNS

- **Date:** August 17, 2017
- **Classification:** Confident BNS (electromagnetic counterpart)
- **Significance:** First GW+EM observation, tight λ̃ constraints
- **Detectors:** LIGO Hanford, Livingston, Virgo
- **Usage:** Validation case, EOS constraint source

### GW190425 - High-Mass Candidate

- **Date:** April 25, 2019
- **Classification:** Likely BNS, unusually high mass
- **Significance:** Challenges mass distribution assumptions
- **Detectors:** LIGO Livingston, Virgo (H1 offline)
- **Usage:** Tests prior sensitivity, ambiguous classification

### GW230529 - Ambiguous Source

- **Date:** May 29, 2023
- **Classification:** Uncertain (BNS or NSBH?)
- **Significance:** Primary science target for method demonstration
- **Detectors:** LIGO Livingston only
- **Usage:** Source classification showcase

## Normalizing Flow Priors

### Population Models

**Uniform** - Flat mass distribution
- Broadest coverage
- Minimal astrophysical assumptions
- Baseline for comparisons

**Gaussian** - Single Gaussian NS mass distribution
- μ = 1.33 M☉, σ = 0.09 M☉
- Motivated by observed pulsar masses
- Intermediate informativeness

**Double Gaussian** - Bimodal mass distribution
- Captures potential NS subpopulations
- More flexible than single Gaussian
- Matches some population synthesis models

### EOS Datasets

**Radio** (PSRs) - Pulsar observations only
- Mass measurements from binary pulsar timing
- Weakest constraints on tidal deformability
- Baseline nuclear physics prior

**Radio + χEFT** - Including chiral effective field theory
- Low-density EOS constraints from χEFT
- Tighter constraints than radio alone
- Theoretically motivated

**Radio + NICER** - Including X-ray observations
- Radius measurements from pulse profile modeling
- Strong mass-radius correlations
- Observationally driven

**Radio + GW170817** - Including multi-messenger event
- Tidal deformability from GW170817
- Tightest constraints for BNS analysis
- Potential circularity for GW170817 re-analysis

**Radio + χEFT + NICER** - Combined constraints
- Strongest overall constraints
- Multiple complementary observations
- Most informative prior

## Implementation Details

### Normalizing Flow Architecture

**Coupling Neural Spline Flow (CouplingNSF)** - Recommended
- Analytically invertible transformations
- Rational quadratic spline bijections
- Fast sampling, stable training
- Well-suited for smooth mass-lambda correlations

**Block Neural Autoregressive Flow (BNAF)** - Alternative
- Maximum expressivity
- Requires numerical inversion (bisection search)
- Potential convergence issues with large models
- Use with caution, increase `maxiter` if needed

### Parameter Constraints

**Mass ratio:** q ∈ [0.1, 1.0]
- Enforced via Sigmoid + ScalarAffine bijection
- Ensures m₂ ≤ m₁ convention

**Tidal deformabilities:** λ₁, λ₂ > 0
- Enforced via Softplus bijection
- Smooth, differentiable constraint

**Chirp mass:** Unbounded
- No constraint applied
- Derived from component masses

### Computational Efficiency

**Relative Binning:**
- ~100x speedup for likelihood evaluation
- Controlled error via `--relative-binning-delta`
- Typical delta = 1e-2 for production runs

**MPI Parallelization:**
- Compatible with HTCondor and SLURM
- Thread-safe environment variables set automatically
- Scales to thousands of cores

## Source Classification Methodology

### Bayes Factor Calculation

For BNS vs NSBH classification:

```
BF = P(data | BNS) / P(data | NSBH)
log(BF) = log(Z_BNS) - log(Z_NSBH)
```

Where Z is Bayesian evidence from nested sampling.

### Interpretation (Jeffreys Scale)

- **|log BF| < 1:** Inconclusive
- **1 ≤ |log BF| < 2.5:** Substantial evidence
- **2.5 ≤ |log BF| < 5:** Strong evidence
- **|log BF| ≥ 5:** Decisive evidence

**Positive log BF:** Favors BNS
**Negative log BF:** Favors NSBH

### Prior Dependence

Bayes factors depend on prior choice:
- **NF priors:** Incorporate EOS physics, higher evidence if data consistent
- **Default priors:** Agnostic, broader coverage, lower evidence
- **Comparison:** ΔBF = BF_NF - BF_default shows prior impact

## File Formats

**NPZ (NumPy Archives)** - Posterior samples
```python
data = np.load('samples.npz')
chirp_mass = data['chirp_mass']
lambda_tilde = data['lambda_tilde']
```

**HDF5** - Bilby result files
```python
import h5py
with h5py.File('result.hdf5', 'r') as f:
    log_evidence = f['log_evidence'][()]
```

**GWF (Gravitational Wave Frames)** - Strain data
```python
from gwpy.timeseries import TimeSeries
strain = TimeSeries.read('H-H1_GWOSC-1187008882-4096.gwf')
```

**EQX (Equinox)** - FlowJAX models
```python
import equinox as eqx
flow = eqx.tree_deserialise_leaves('flowjax_model.eqx', flow)
```

**PTH (PyTorch)** - GlasFlow models
```python
import torch
model = torch.load('model.pth')
```

## Results Organization

### Directory Naming Convention

**PE Results:** `{event}/{source}/{population}/{eos}/samples.npz`

Examples:
- `final_results/GW170817/bns/uniform/radio/samples.npz`
- `final_results/GW190425/nsbh/gaussian/radio_chiEFT/samples.npz`
- `final_results/GW230529/bns/default/samples.npz` (agnostic prior)

### Special Cases

**Default runs:** `{event}/{source}/default/`
- Uses agnostic uniform priors
- No EOS constraints
- Baseline for comparison

**Conditional models:** `{event}/` directory in `NFprior/models/`
- Event-specific NF models (advanced)
- Currently unconditional models preferred

## Validation and Testing

### Injection Studies (`injections/`)

Validate analysis pipeline with simulated signals:
- Inject known BNS/NSBH signals
- Recover with different priors
- Assess bias and coverage
- Quantify classification performance

### Normalization Validation (`normalization/`)

Ensure NF priors properly normalized:
- Monte Carlo integration ≈ 1
- No probability leakage
- Jacobian corrections correct
- bilby compatibility verified

### Debug Diagnostics (`debug/`)

Troubleshoot analysis issues:
- Inspect likelihood calculations
- Verify parameter transformations
- Quick diagnostic plots
- Data conditioning tests

## Known Issues and Solutions

### FlowJAX Numerical Inversion

**Issue:** Large-capacity BNAF models fail with `Maximum steps reached` during sampling

**Solution:** Increase `maxiter` in `NumericalInverse`:
```python
from flowjax.bijections import NumericalInverse
inverter = NumericalInverse(maxiter=200)
```

Or use CouplingNSF instead (analytically invertible).

### Bilby Cosmology

**Issue:** Default cosmology may differ from reference analyses

**Solution:** Explicitly set cosmology in PE:
```python
from astropy.cosmology import Planck15
bilby.gw.cosmology.set_cosmology(Planck15)
```

### GW230529 Frequency Range

**Issue:** Single-detector, high-mass event requires careful frequency setup

**Solution:** Check `fmin`, `fmax` settings for L1 sensitivity

## Contributing

### Adding New EOS Dataset

1. Place samples in `data/eos/new_dataset/`
2. Train NF: `python NFprior/train_NF_prior.py --eos-samples-name new_dataset`
3. Run PE: `python GW_runs/pe.py --eos-samples-name new_dataset`
4. Update display names in `bayes_factors/collect_all_bayes_factors.py`

### Adding New Event

1. Download strain/PSD to `data/new_event/`
2. Create prior file (if needed)
3. Run PE for BNS and NSBH hypotheses
4. Update event lists in plotting/analysis scripts
5. Regenerate Bayes factor tables

## Citation

If you use this code or methodology, please cite:

```bibtex
@article{wouters2024neural,
  title={Incorporating neutron star physics into gravitational wave inference with neural priors},
  author={Wouters, Thibeau and ...},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2024}
}
```

## Related Software

**bilby:** Bayesian inference for gravitational waves
- Repository: https://git.ligo.org/lscsoft/bilby
- Modified branch: https://github.com/ThibeauWouters/bilby/tree/eos_source_classification
- Documentation: https://lscsoft.docs.ligo.org/bilby/

**flowjax:** JAX-based normalizing flows
- Repository: https://github.com/danielward27/flowjax
- Documentation: https://flowjax.readthedocs.io/

**glasflow:** PyTorch normalizing flows (nflows wrapper)
- Repository: https://github.com/uofgravity/glasflow
- Paper: https://arxiv.org/abs/2011.12320

## License

[Specify license here]

## Contact

Thibeau Wouters - [email/website]

For questions about the code or methodology, please open an issue on GitHub or contact the authors directly.

## Acknowledgments

This research uses:
- Gravitational wave data from LIGO/Virgo/KAGRA observatories
- EOS constraints from pulsar observations, NICER, and nuclear theory
- Open-source software: bilby, flowjax, glasflow, numpy, scipy, matplotlib

Computational resources provided by [institution/cluster].
