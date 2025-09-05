# bilby_pipe Configuration Generator for EOS Source Classification

## Overview

This directory contains an automated system for generating bilby_pipe configurations for gravitational wave parameter estimation with normalizing flow (NF) priors. The goal is to systematically analyze gravitational wave events to classify the source type (binary neutron star vs neutron star-black hole) using equation of state constraints.

## Architecture

### Core Components

1. **`generate_bilby_pipe_configs.py`** - Main script that generates all configurations
2. **`template_config.ini`** - Template for bilby_pipe configuration files
3. **`template_prior.prior`** - Template for normalizing flow prior files  
4. **`template_default_prior.prior`** - Template for standard bilby prior files

### How It Works

The generator script automatically creates comprehensive parameter estimation configurations by:

1. **Reading event data** from hardcoded configurations (trigger times, detector networks, PSDs, etc.)
2. **Copying templates** to target directories for each run configuration
3. **Performing substitutions** to customize templates with event-specific parameters
4. **Organizing outputs** into a hierarchical directory structure

## Configuration Matrix

The script generates configurations for all combinations of:

- **Events**: GW170817, GW190425, GW230529
- **Population types**: uniform, gaussian, double_gaussian, default
- **Prior scenarios**: bns (binary neutron star), nsbh (neutron star-black hole)  
- **EOS constraints**: radio, radio_chiEFT, radio_NICER, radio_chiEFT_NICER

This creates up to **96 total configurations**:
- 6 default runs (3 events × 2 scenarios) using standard bilby priors
- Up to 90 NF runs (3 events × 3 populations × 2 scenarios × 5 EOS types)

## Directory Structure

Generated configurations follow the pattern:
```
runs/
├── {event}/
│   ├── {population}/
│   │   ├── {prior_scenario}/
│   │   │   ├── {eos_constraint}/
│   │   │   │   ├── config.ini
│   │   │   │   └── prior.prior
```

For example:
```
runs/GW170817/double_gaussian/bns/radio/config.ini
runs/GW170817/default/nsbh/config.ini
```

## Key Features

### Event-Specific Configuration
- **Trigger times**: From reference parameter files
- **Detector networks**: H1+L1+V1 for GW170817, L1+V1 for GW190425, L1 only for GW230529
- **Frequency ranges**: 23.1-2048 Hz for GW170817, 20-2048 Hz for GW190425, 20-1792 Hz for GW230529
- **Data paths**: Automatically configured frame files and PSDs

### Astrophysically Motivated Priors
- **Spin constraints**: 
  - BNS default: a₁,a₂ ≤ 0.05 (conservative neutron star spins)
  - BNS NF runs: a₁,a₂ ≤ 0.05 (matching observations)
  - NSBH: a₁ ≤ 0.5, a₂ ≤ 0.05 (black hole can spin faster)
- **Mass priors**: Use bilby.gw.prior.UniformInComponentsChirpMass/UniformInComponentsMassRatio for default runs
- **Tidal deformabilities**: λ₁ = 0 for NSBH (black holes have no tides), λ₁,λ₂ from NF/uniform for BNS

### Normalizing Flow Integration
- **Model loading**: Automatically finds NF models at `../NFprior/models/{population}/{scenario}/{eos}/model.pt`
- **Parameter names**: Extracted from `*_kwargs.json` files alongside models
- **NF configuration**: Uses NFDist with `use_tilde=False`, `use_component_masses=False` for simplicity
- **Validation**: Skips configurations where NF models or kwargs files don't exist

### bilby_pipe Configuration
- **Likelihood**: MBGravitationalWaveTransient (multibanding for efficiency)
- **Sampler**: dynesty with 4096 live points, 96 CPU cores
- **Distance marginalization**: Enabled for computational efficiency
- **Output**: HDF5 format with 20,000 final samples

## Usage

Simply run the generator script:
```bash
cd GW_runs_jarvis/
python generate_bilby_pipe_configs.py
```

No command-line arguments needed - the script generates all possible configurations automatically.

The script will:
1. Create the `runs/` directory structure
2. Generate config.ini and prior.prior files for each valid combination
3. Skip combinations where NF models don't exist (with warnings)
4. Print progress as it creates each configuration

## Key Design Decisions

### Template-Based Approach
Rather than hardcoding configuration content, the script uses templates with placeholders like `{{{TRIGGER_TIME}}}`. This makes it easy to modify configurations without touching the Python code.

### Hierarchical Organization  
The directory structure mirrors the conceptual hierarchy: event → population → source type → EOS constraint. This makes it intuitive to navigate and run specific subsets of analyses.

### Robust Validation
The script checks for both `.pt` model files and `_kwargs.json` metadata files before generating NF configurations, preventing incomplete setups.

### Resource Alignment
CPU requests match the number of sampling cores (both 96) to avoid resource waste on compute clusters.

## Relationship to Original pe.py

This system replaces the manual configuration process in `../GW_runs/pe.py` with an automated, template-driven approach. Key advantages:

- **Systematic coverage**: No missed parameter combinations
- **Consistency**: All runs use identical settings except for intended variations
- **Maintainability**: Templates are easier to modify than embedded strings
- **bilby_pipe integration**: Leverages bilby_pipe's job management instead of custom MPI handling

The generated configurations should produce equivalent scientific results to manually configured `pe.py` runs while being more systematic and less error-prone.

## Future Extensions

The template system makes it straightforward to:
- Add new events by extending `EVENT_CONFIGS`
- Include additional EOS constraints in `eos_samples`  
- Modify sampling parameters (live points, CPU cores) in one location
- Add new population types as NF models become available

This framework provides a scalable foundation for comprehensive gravitational wave source classification studies using equation of state constraints.