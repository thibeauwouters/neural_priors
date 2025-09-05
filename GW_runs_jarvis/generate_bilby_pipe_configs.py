#!/usr/bin/env python3
"""
Generate bilby_pipe configuration files for EOS source classification runs.

This script creates config.ini and prior.prior files for all combinations of:
- GW events (GW170817, GW190425, GW230529) 
- Population types (uniform, gaussian, double_gaussian)
- Prior types (bns, nsbh)
- EOS constraints (radio, radio_chiEFT, radio_NICER, etc.)

Uses multibanding (MBGravitationalWaveTransient) instead of relative binning.
"""

import os
import itertools
from pathlib import Path
import shutil
import json


# Absolute base directory is one up
BASE_DIR = ".."

# Event configurations extracted from pe.py and reference_parameters.json
EVENT_CONFIGS = {
    'GW170817': {
        'trigger_time': 1187008882.43,
        'detectors': ['H1', 'L1', 'V1'],
        'duration': 128.0,
        'minimum_frequency': 23.1,
        'maximum_frequency': 2048.0,
        'sampling_frequency': 4096.0,
        'frame_files': {
            'H1': f'{BASE_DIR}/data/GW170817/H-H1_LOSC_CLN_16_V1-1187007040-2048.gwf',
            'L1': f'{BASE_DIR}/data/GW170817/L-L1_LOSC_CLN_16_V1-1187007040-2048.gwf',
            'V1': f'{BASE_DIR}/data/GW170817/V-V1_LOSC_CLN_16_V1-1187007040-2048.gwf'
        },
        'channels': {
            'H1': 'H1:LOSC-STRAIN',
            'L1': 'L1:LOSC-STRAIN',
            'V1': 'V1:LOSC-STRAIN'
        },
        'psd_files': {
            'H1': f'{BASE_DIR}/data/GW170817/h1_psd.txt',
            'L1': f'{BASE_DIR}/data/GW170817/l1_psd.txt',
            'V1': f'{BASE_DIR}/data/GW170817/v1_psd.txt'
        },
        'default_priors': {
            'chirp_mass_min': 1.18,
            'chirp_mass_max': 1.21,
            'distance_min': 1.0,
            'distance_max': 75.0
        }
    },
    'GW190425': {
        'trigger_time': 1240215503.017147,
        'detectors': ['L1', 'V1'],
        'duration': 128.0,
        'minimum_frequency': 20.0,
        'maximum_frequency': 2048.0,
        'sampling_frequency': 4096.0,
        'frame_files': {
            'L1': f'{BASE_DIR}/data/GW190425/L-L1_GWOSC_16KHZ_R1-1240213455-4096.gwf',
            'V1': f'{BASE_DIR}/data/GW190425/V-V1_GWOSC_16KHZ_R1-1240213455-4096.gwf'
        },
        'channels': {
            'L1': 'L1:GWOSC-16KHZ_R1_STRAIN',
            'V1': 'V1:GWOSC-16KHZ_R1_STRAIN'
        },
        'psd_files': {
            'L1': f'{BASE_DIR}/data/GW190425/glitch_median_PSD_forLI_L1_srate8192.txt',
            'V1': f'{BASE_DIR}/data/GW190425/glitch_median_PSD_forLI_V1_srate8192.txt'
        },
        'default_priors': {
            'chirp_mass_min': 1.485,
            'chirp_mass_max': 1.490,
            'distance_min': 10.0,
            'distance_max': 300.0
        }
    },
    'GW230529': {
        'trigger_time': 1369419318.7460938,
        'detectors': ['L1'],
        'duration': 128.0,
        'minimum_frequency': 20.0,
        'maximum_frequency': 1792.0,
        'sampling_frequency': 4096.0,
        'frame_files': {
            'L1': f'{BASE_DIR}/data/GW230529/L-L1_GWOSC_16KHZ_R1-1369417271-4096.gwf'
        },
        'channels': {
            'L1': 'L1:GWOSC-16KHZ_R1_STRAIN'
        },
        'psd_files': {
            'L1': f'{BASE_DIR}/data/GW230529/psd_4096.dat'
        },
        'default_priors': {
            'chirp_mass_min': 2.02,
            'chirp_mass_max': 2.04,
            'distance_min': 1.0,
            'distance_max': 500.0
        }
    }
}

# NF model configurations
NF_CONFIGS = {
    'population_types': ['uniform', 'gaussian', 'double_gaussian'],
    'prior_names': ['bns', 'nsbh'],
    'eos_samples': ['radio']  # Start with radio, can expand later
}

def get_nf_model_path(population_type, prior_name, eos_samples_name):
    """Get the path to the NF model file."""
    return f"{BASE_DIR}/NFprior/models/{population_type}/{prior_name}/{eos_samples_name}/model.pt"

def create_config_ini(event_name, population_type, prior_name, eos_samples_name, config_dir, npool=192):
    """Copy template and modify config.ini in the target directory."""
    config = EVENT_CONFIGS[event_name]
    
    # Copy template to target directory
    template_path = Path(__file__).parent / "template_config.ini"
    config_file_path = config_dir / "config.ini"
    shutil.copy2(template_path, config_file_path)
    
    # Read the copied file
    config_content = config_file_path.read_text()
    
    # Create label
    if population_type == 'default':
        label = f"{event_name}_default_{prior_name}"
    else:
        label = f"{event_name}_{population_type}_{prior_name}_{eos_samples_name}"
    
    # Build data-dict and channel-dict strings
    data_dict_parts = []
    channel_dict_parts = []
    psd_dict_parts = []
    
    for detector in config['detectors']:
        data_dict_parts.append(f"'{detector}': '{config['frame_files'][detector]}'")
        channel_dict_parts.append(f"'{detector}': '{config['channels'][detector]}'")
        psd_dict_parts.append(f"'{detector}': '{config['psd_files'][detector]}'")
    
    data_dict = "{" + ", ".join(data_dict_parts) + "}"
    channel_dict = "{" + ", ".join(channel_dict_parts) + "}"
    psd_dict = "{" + ", ".join(psd_dict_parts) + "}"
    detectors_list = str(config['detectors'])
    
    # Template substitutions
    substitutions = {
        'TRIGGER_TIME': str(config['trigger_time']),
        'DATA_DICT': data_dict,
        'CHANNEL_DICT': channel_dict,
        'DETECTORS': detectors_list,
        'PSD_DICT': psd_dict,
        'SAMPLING_FREQUENCY': str(config['sampling_frequency']),
        'MAXIMUM_FREQUENCY': str(config['maximum_frequency']),
        'MINIMUM_FREQUENCY': str(config['minimum_frequency']),
        'LABEL': label,
        'TIME_REFERENCE': config['detectors'][0],
        'REFERENCE_FREQUENCY': str(config['minimum_frequency']),
        'NPOOL': str(npool)
    }
    
    # Perform substitutions
    for key, value in substitutions.items():
        config_content = config_content.replace(f"{{{{{{{key}}}}}}}", value)
    
    # Write modified content back
    config_file_path.write_text(config_content)

def create_prior_file(event_name, population_type, prior_name, eos_samples_name, config_dir):
    """Copy template and modify prior.prior in the target directory."""
    config = EVENT_CONFIGS[event_name]
    
    # Check if this is a default run (no NF model)
    is_default_run = population_type == 'default'
    
    if is_default_run:
        # Copy default template to target directory
        template_path = Path(__file__).parent / "template_default_prior.prior"
        prior_file_path = config_dir / "prior.prior"
        shutil.copy2(template_path, prior_file_path)
        
        # Read the copied file
        prior_content = prior_file_path.read_text()
        
        # For NSBH, override lambda_1 to be DeltaFunction(0)
        nsbh_override = ""
        if prior_name == 'nsbh':
            nsbh_override = "lambda_1 = DeltaFunction(0.0, name='lambda_1', latex_label='$\\Lambda_1$')"
        
        # Set spin maxima based on prior type 
        if prior_name == 'nsbh':
            a1_max = "0.5"  # Primary (BH) can have higher spin
            a2_max = "0.05"  # Secondary (NS) lower spin
        else:  # bns
            a1_max = "0.05"  # Both NS, conservative spin
            a2_max = "0.05"
        
        # Template substitutions for default run
        substitutions = {
            'CHIRP_MASS_MIN': str(config['default_priors']['chirp_mass_min']),
            'CHIRP_MASS_MAX': str(config['default_priors']['chirp_mass_max']),
            'DISTANCE_MIN': str(config['default_priors']['distance_min']),
            'DISTANCE_MAX': str(config['default_priors']['distance_max']),
            'GEOCENT_TIME_MIN': str(config['trigger_time'] - 0.1),
            'GEOCENT_TIME_MAX': str(config['trigger_time'] + 0.1),
            'A1_MAX': a1_max,
            'A2_MAX': a2_max,
            'NSBH_LAMBDA1_OVERRIDE': nsbh_override
        }
    else:
        # Copy NF template to target directory
        template_path = Path(__file__).parent / "template_prior.prior"
        prior_file_path = config_dir / "prior.prior"
        shutil.copy2(template_path, prior_file_path)
        
        # Read the copied file
        prior_content = prior_file_path.read_text()
        
        model_path = get_nf_model_path(population_type, prior_name, eos_samples_name)
        
        # Load NF kwargs  
        kwargs_path = model_path.replace('.pt', '_kwargs.json')
        if os.path.exists(kwargs_path):
            with open(kwargs_path, 'r') as f:
                nf_kwargs = json.load(f)
            nf_params = str(nf_kwargs["names"])
        else:
            # Fallback to defaults if kwargs file not found
            nf_params = '["chirp_mass_source", "mass_ratio", "lambda_1", "lambda_2"]'
        
        # Hardcode for simplicity
        use_tilde = "false"
        use_component_masses = "false"
        
        # For NSBH, override lambda_1 to be DeltaFunction(0)
        nsbh_override = ""
        if prior_name == 'nsbh':
            nsbh_override = "lambda_1 = DeltaFunction(0.0, name='lambda_1', latex_label='$\\Lambda_1$')"
        
        # Set spin maxima based on prior type for NF runs
        if prior_name == 'nsbh':
            a1_max = "0.5"  # Primary (BH) can have higher spin  
            a2_max = "0.05"  # Secondary (NS) lower spin
        else:  # bns
            a1_max = "0.05"  # Both NS, NF runs use higher values
            a2_max = "0.05"
        
        # Template substitutions for NF run
        substitutions = {
            'GEOCENT_TIME_MIN': str(config['trigger_time'] - 0.1),
            'GEOCENT_TIME_MAX': str(config['trigger_time'] + 0.1),
            'NF_NAMES': nf_params,
            'MODEL_PATH': model_path,
            'USE_TILDE': use_tilde,
            'USE_COMPONENT_MASSES': use_component_masses,
            'A1_MAX': a1_max,
            'A2_MAX': a2_max,
            'NSBH_LAMBDA1_OVERRIDE': nsbh_override
        }
    
    # Perform substitutions
    for key, value in substitutions.items():
        prior_content = prior_content.replace(f"{{{{{{{key}}}}}}}", value)
    
    # Write modified content back
    prior_file_path.write_text(prior_content)

def main():
    # Configuration parameters
    events = ['GW170817', 'GW190425', 'GW230529']
    population_types = ['uniform', 'gaussian', 'double_gaussian', 'default']
    prior_names = ['bns', 'nsbh']
    eos_samples = ['radio', 'radio_chiEFT', 'radio_NICER', 'radio_chiEFT_NICER']
    output_base = Path('configs')
    npool = 96  # Number of cores for sampling
    
    # Create base output directory
    output_base.mkdir(exist_ok=True)
    
    print(f"Generating bilby_pipe configurations...")
    print(f"Events: {events}")
    print(f"Population types: {population_types}")
    print(f"Prior names: {prior_names}")
    print(f"EOS samples: {eos_samples}")
    print()
    
    total_configs = 0
    
    for event, pop_type, prior_name, eos_sample in itertools.product(
        events, population_types, prior_names, eos_samples
    ):
        # For default runs, skip NF model checking and use simplified naming
        if pop_type == 'default':
            config_name = f"{event}_default_{prior_name}"
        else:
            # Check if NF model and kwargs exist for non-default runs
            model_path = get_nf_model_path(pop_type, prior_name, eos_sample)
            kwargs_path = model_path.replace('.pt', '_kwargs.json')
            if not os.path.exists(model_path):
                print(f"Warning: Skipping {event}_{pop_type}_{prior_name}_{eos_sample} - model not found: {model_path}")
                continue
            if not os.path.exists(kwargs_path):
                print(f"Warning: Skipping {event}_{pop_type}_{prior_name}_{eos_sample} - kwargs not found: {kwargs_path}")
                continue
            config_name = f"{event}_{pop_type}_{prior_name}_{eos_sample}"
            
        # Create output directory
        config_dir = output_base / config_name
        config_dir.mkdir(exist_ok=True)
        
        # Generate files by copying and modifying templates
        create_config_ini(event, pop_type, prior_name, eos_sample, config_dir, npool)
        create_prior_file(event, pop_type, prior_name, eos_sample, config_dir)
        
        print(f"Generated: {config_name}")
        total_configs += 1
    
    print(f"\nGenerated {total_configs} configurations in {output_base}/")

if __name__ == "__main__":
    main()