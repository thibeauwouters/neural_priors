'''
Script to create gravitational wave injections and perform parameter estimation
The goal is to create injections in realistic data streams, where we take the PSD of some events and inject the signal
into the data, and then perform parameter estimation on these injections, to better understand how the recovery made with NF priors work.
'''

# Put on some flags since MPI seems to mess up with the NF priors
import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["MKL_DYNAMIC"] = "FALSE"
os.environ["OMP_NUM_THREADS"] = "1"

import sys
import bilby 
import numpy as np
from bilby.core.prior.analytical import Uniform, Sine, Cosine, DeltaFunction
from bilby.gw.prior import UniformComovingVolume, AlignedSpin
from bilby.core.prior.joint import NFDist, NFPrior
import argparse
import json
from bilby.core.utils import logger
import warnings
import signal
from contextlib import contextmanager
import lalsimulation as lalsim
warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")

# Custom filter is to suppress specific warnings from the bilby library that keep on appearing again and again and bloat the output files
import logging
class MaskingFilter(logging.Filter):
    def filter(self, record):
        # Return False if message matches the undesired warnings
        msg = record.getMessage()
        if "Masking >3 elements" in msg or "masked frequencies" in msg:
            return False
        return True
logger.addFilter(MaskingFilter())

# Timeout context manager for testing
@contextmanager
def timeout(duration):
    """Context manager that raises TimeoutError if operation takes longer than duration seconds"""
    def timeout_handler(_signum, _frame):
        raise TimeoutError(f"Operation timed out after {duration} seconds")
    
    # Set the signal handler and a timeout alarm
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(duration)
    
    try:
        yield
    finally:
        # Disable the alarm
        signal.alarm(0)

################
### ARGPARSE ###
################

parser = argparse.ArgumentParser()
parser.add_argument('--run-dir',
                    type = str,
                    required = True,
                    help = "Directory containing injection_parameters.json and prior.prior files")
parser.add_argument('--prior-name',
                    type = str,
                    default = 'bns',
                    help = "Prior to use for parameter estimation")
# FIXME: need to consider toggling to design PSD for the event
parser.add_argument('--eos-samples-name',
                    type = str,
                    default = "radio",
                    help = "Name of the EOS posterior dataset, must match the NF name")
parser.add_argument('--label',
                    type = str,
                    default = 'injection',
                    help = "Label to give to the injection")
parser.add_argument('--seed', 
                    type = int,
                    default = 2024,
                    help = "Seed for the random number generator")
parser.add_argument('--event-name',
                    type = str,
                    required = True,
                    help = "Name of the event to use for injection configuration (e.g., GW170817)")
parser.add_argument('--relative-binning-delta', 
                    type = float,
                    default = 1e-2,
                    help = "The total error on the relative binning likelihood at the reference value")
parser.add_argument('--minimum-bin-threshold', 
                    type = int,
                    default = 1000,
                    help = "The minimum number of bins allowed to have")
parser.add_argument('--dry-run',
                    action = 'store_true',
                    help = "Run through the script without sampling to validate setup")
parser.add_argument('--test-sampling',
                    action = 'store_true',
                    help = "Test sampling with 30 second timeout for debugging")
parser.add_argument('--n-pool',
                    type = int,
                    default = 64,
                    help = "How many cores to use for the sampling")
parser.add_argument('--psd-type',
                    type = str,
                    default = 'event',
                    choices = ['design', 'event'],
                    help = "Type of PSD to use: design sensitivity or event-specific PSDs")

args = parser.parse_args()

# Event configuration mapping
# TODO: check if duration must be changed
EVENT_CONFIG = {
    'GW170817': {
        'ifo_list': ['H1', 'L1', 'V1'],
        'duration': 128.0,
        'minimum_frequency': 23.1,
        'sampling_frequency': 2*2048.0
    },
    'GW190425': {
        'ifo_list': ['L1', 'V1'],
        'duration': 128.0,
        'minimum_frequency': 20.0,
        'sampling_frequency': 2*2048.0
    },
    'GW230529': {
        'ifo_list': ['L1'],
        'duration': 128.0,
        'minimum_frequency': 20.0,
        'sampling_frequency': 2*2048.0
    },
    # # FIXME: finish support for an arbitrary event
    # 'design': {
    #     'ifo_list': ['H1', 'L1', 'V1'],
    #     'duration': 128.0,
    #     'minimum_frequency': 20.0,
    #     'sampling_frequency': 2*2048.0
    # }
}

################
### PREAMBLE ###
################

# Check if some of the given arguments are valid
SUPPORTED_PRIORS = ['default', 'bns', 'nsbh']

if args.prior_name not in SUPPORTED_PRIORS:
    raise ValueError(f"Invalid prior name provided. Please provide one of {SUPPORTED_PRIORS}")

# Auto-detect environment and set paths
def detect_environment() -> str:
    """Auto-detect whether running locally or on cluster"""
    if os.path.exists("/data/gravwav/twouters/projects/eos_source_classification"):
        logger.info("You are running on the Nikhef cluster, setting paths accordingly.")
        sys.path.append('/data/gravwav/twouters/uu_relative_binning/uu_relative_binning') # on the cluster
        base_dir = "/data/gravwav/twouters/projects/eos_source_classification/eos_source_classification/"
        return base_dir
    else:
        logger.info("You are testing locally, setting paths accordingly.")
        sys.path.append("/Users/Woute029/Documents/Code/projects/eos_source_classification/uu_relative_binning/uu_relative_binning") # locally
        base_dir = "/Users/Woute029/Documents/Code/projects/eos_source_classification/eos_source_classification/"
        return base_dir

base_dir = detect_environment()
cwd = os.getcwd()
from LikelihoodRB import RelBinning
    
# Validate event name and get configuration
if args.event_name not in EVENT_CONFIG:
    raise ValueError(f"Invalid event name provided. Please provide one of {list(EVENT_CONFIG.keys())}")

event_config = EVENT_CONFIG[args.event_name]

# Setup file paths
run_dir = os.path.abspath(args.run_dir)
injection_parameters_file = os.path.join(run_dir, "injection_parameters.json")
prior_filename = os.path.join(run_dir, "prior.prior")

# Setup output directory
full_outdir = os.path.join(cwd, args.run_dir, args.label)
os.makedirs(full_outdir, exist_ok=True)

bilby.core.utils.setup_logger(outdir=full_outdir, label=args.label)
logger.info(f"Output directory set to {full_outdir}")

# Load injection parameters from JSON file
logger.info(f"Loading injection parameters from {injection_parameters_file}")
with open(injection_parameters_file, 'r') as f:
    injection_parameters = json.load(f)

logger.info(f"Injection parameters:")
for key, value in injection_parameters.items():
    logger.info(f"  {key}: {value}")

# Calculate merger frequency using LALSimulation
# Convert parameters for LALSim function
mass_1_msun = injection_parameters['mass_1']
mass_2_msun = injection_parameters['mass_2'] 
mtot_msun = mass_1_msun + mass_2_msun
q = max(mass_1_msun, mass_2_msun) / min(mass_1_msun, mass_2_msun)  # q >= 1
lambda_1 = injection_parameters.get('lambda_1', 0.0)
lambda_2 = injection_parameters.get('lambda_2', 0.0)
chi1_as = injection_parameters.get('chi_1', 0.0)  # aligned spin component
chi2_as = injection_parameters.get('chi_2', 0.0)  # aligned spin component

# Call LALSim merger frequency function
merger_freq = lalsim.SimNRTunedTidesMergerFrequency_v3(
    mtot_msun, lambda_1, lambda_2, q, chi1_as, chi2_as
)

logger.info(f"Merger frequency: {merger_freq:.2f} Hz")
logger.info(f"1.2 × merger frequency: {1.2 * merger_freq:.2f} Hz")

# Set random seed
np.random.seed(args.seed)
bilby.core.utils.random.seed(args.seed)

# Event parameters from configuration
duration = event_config['duration']
sampling_frequency = event_config['sampling_frequency']
minimum_frequency = event_config['minimum_frequency']
reference_frequency = minimum_frequency
approximant = 'IMRPhenomXP_NRTidalv3'

# Use geocent_time from injection parameters
geocent_time = injection_parameters['geocent_time']
start_time = geocent_time - duration + 2.0

logger.info(f"Using waveform: {approximant}")
logger.info(f"Duration: {duration}s")
logger.info(f"Sampling frequency: {sampling_frequency} Hz")
logger.info(f"Minimum frequency: {minimum_frequency} Hz")

# Set up waveform generation
waveform_arguments = dict(
    waveform_approximant=approximant, 
    reference_frequency=reference_frequency,
    minimum_frequency=minimum_frequency,
    maximum_frequency=0.5*sampling_frequency)

waveform_generator = bilby.gw.WaveformGenerator(
    duration=duration,
    sampling_frequency=sampling_frequency,
    start_time=start_time,
    frequency_domain_source_model=bilby.gw.source.lal_binary_neutron_star,                    
    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_neutron_star_parameters,
    waveform_arguments=waveform_arguments)

# If a dry run, then make sure we always get passed the relative binning
if args.dry_run:
    dry_run_bin_threshold = 10 # just a low value
    logger.info(f"Running in dry run mode, setting minimum_bin_threshold to {dry_run_bin_threshold}")
    args.minimum_bin_threshold = dry_run_bin_threshold

# Reference parameters are the injection parameters
reference_parameters = injection_parameters.copy()

# FIXME: make this a separate function, to avoid duplicating code
# Compute merger frequency for reference parameters
mtot_msun = reference_parameters['mass_1'] + reference_parameters['mass_2']
q = max(reference_parameters['mass_1'], reference_parameters['mass_2']) / min(reference_parameters['mass_1'], reference_parameters['mass_2'])  # q >= 1
lambda_1 = reference_parameters.get('lambda_1', 0.0)
lambda_2 = reference_parameters.get('lambda_2', 0.0)
chi1_as = reference_parameters.get('chi_1', 0.0)  # aligned spin component
chi2_as = reference_parameters.get('chi_2', 0.0)  # aligned spin component
# Call LALSim merger frequency function
merger_freq_ref = lalsim.SimNRTunedTidesMergerFrequency_v3(
    mtot_msun, lambda_1, lambda_2, q, chi1_as, chi2_as
)
logger.info(f"Reference merger frequency: {merger_freq_ref:.2f} Hz")

logger.info(f"Reference parameters for PE:")
for key, value in reference_parameters.items():
    logger.info(f"{key}: {value}")
    
##############
### PRIORS ###
##############

# This is to easily and flexibly import the necessary classes below, also for NFs
safe_globals = {
    '__builtins__': {
        'abs': abs, 'min': min, 'max': max, 'round': round,
        'int': int, 'float': float, 'str': str, 'bool': bool,
    },
    'np': np,
    'bilby': bilby,
    'Uniform': Uniform,
    'Sine': Sine, 
    'Cosine': Cosine,
    'UniformComovingVolume': UniformComovingVolume,
    'AlignedSpin': AlignedSpin
}

# Prior file is automatically set from run directory
logger.info(f"Loading priors from {prior_filename}")

# Read the prior file as text (same as pe.py)
if os.path.exists(prior_filename):
    with open(prior_filename, "r") as f:
        prior_lines = f.readlines()

    # Replace 'UniformComovingVolume' with full path
    modified_lines = [
        line.replace(
            "UniformComovingVolume", "bilby.gw.prior.UniformComovingVolume"
        ) for line in prior_lines
    ]

    # Evaluate the prior lines in a safe namespace
    prior_dict = {}
    exec("".join(modified_lines), safe_globals, prior_dict)
else:
    raise FileNotFoundError(f"Prior file not found: {prior_filename}")

if args.prior_name == "default":
    logger.info("Using the default priors")
    
    # Build PriorDict straight from all given priors
    priors = bilby.core.prior.PriorDict(prior_dict)
    
    # Lambdas are not in the prior files, so we add them manually here
    priors["lambda_1"] = bilby.core.prior.Uniform(minimum=0.0, maximum=5000.0, name='lambda_1', latex_label='$\\Lambda_1$')
    priors["lambda_2"] = bilby.core.prior.Uniform(minimum=0.0, maximum=5000.0, name='lambda_2', latex_label='$\\Lambda_2$')
    
else:
    logger.info(f"Sampling with an NF prior, with name {args.prior_name}")
    
    # First, drop chirp_mass, mass_ratio and luminosity_distance from the prior_dict, as these are modelled by the NF
    prior_dict.pop('chirp_mass', None)
    prior_dict.pop('mass_ratio', None)
    prior_dict.pop('luminosity_distance', None)
    
    # Path to NF model - use the same event as injection for consistency
    nf_model_path = os.path.join(base_dir, f"NFprior/models/{args.event_name}/{args.eos_samples_name}_{args.prior_name}/model.pt")
    nf_model_path = os.path.abspath(nf_model_path)
    logger.info(f"Using NF model path: {nf_model_path}")
    
    nf_kwargs_filename = nf_model_path.replace('.pt', '_kwargs.json')
    if os.path.exists(nf_kwargs_filename):
        with open(nf_kwargs_filename, 'r') as f:
            nf_kwargs = json.load(f)
        logger.info(f"Loaded NF kwargs from {nf_kwargs_filename}")
    else:
        raise FileNotFoundError(f"NF kwargs file not found: {nf_kwargs_filename}")
    
    # TODO: this is hard-coded for now, but might be more flexible in the future
    nf_dist = NFDist(
        names=nf_kwargs["names"],
        flow_filename=nf_model_path,
        include_dL=True,
        use_tilde=False,
        use_component_masses=False
        )
    
    # Add NFPrior for each name that is in the dist
    for name in nf_dist.names:
        prior_dict[name] = NFPrior(dist=nf_dist, name=name)
        
    if args.prior_name == "nsbh":
        # Put lambda_1 = 0.0
        prior_dict["lambda_1"] = DeltaFunction(0.0, name='lambda_1', latex_label='$\\Lambda_1$')
        
    priors = bilby.core.prior.PriorDict(prior_dict)
        
logger.info("Going to show priors:")
for key, value in priors.items():
    logger.info(f"      {key}: {value}")

##########################
### IFOS and injection ###
##########################

# Set up interferometers from event configuration
ifo_list = event_config['ifo_list']
logger.info(f"Setting up interferometers: {ifo_list}")

ifos = bilby.gw.detector.InterferometerList(ifo_list)
# Make sure the minimum/maximum frequency is set correctly
for ifo in ifos:
    ifo.minimum_frequency = minimum_frequency
    ifo.maximum_frequency = 0.5*sampling_frequency

# Set up PSDs
logger.info("Setting up PSDs...")
if args.psd_type == 'design':
    # Use design sensitivity curves - the interferometers already have default PSDs set
    # No additional setup needed for design sensitivity
    raise NotImplementedError("Design sensitivity PSDs are not implemented yet. Please use event-specific PSDs for now.")
    # logger.info("Using design sensitivity PSDs (default for interferometers)")
    
else:
    # Use event-specific PSDs - fetch from data directory like pe.py
    logger.info("Loading event-specific PSDs...")
    data_path = os.path.join(base_dir, 'data', args.event_name)
    
    if args.event_name == 'GW190425':
        psd_files = {
            "L1": os.path.join(data_path, "glitch_median_PSD_forLI_L1_srate8192.txt"),
            "V1": os.path.join(data_path, "glitch_median_PSD_forLI_V1_srate8192.txt")
        }
    elif args.event_name == 'GW170817':
        psd_files = {
            "H1": os.path.join(data_path, "h1_psd.txt"),
            "L1": os.path.join(data_path, "l1_psd.txt"),
            "V1": os.path.join(data_path, "v1_psd.txt")
        }
    elif args.event_name == 'GW230529':
        psd_files = {
            "L1": os.path.join(data_path, "L1_psd.dat"),
        }
    else:
        raise ValueError(f"Event-specific PSDs not available for {args.event_name}")
    
    # Load PSDs for each interferometer
    for ifo in ifos:
        if ifo.name in psd_files:
            logger.info(f"Loading PSD for {ifo.name} from {psd_files[ifo.name]}")
            ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(psd_file=psd_files[ifo.name])
        else:
            raise ValueError(f"No PSD file found for interferometer {ifo.name} in {args.event_name}")

# Inject the signal
logger.info("Generating injection...")
for ifo in ifos:
    ifo.set_strain_data_from_power_spectral_density(
        sampling_frequency=sampling_frequency,
        duration=duration,
        start_time=start_time)

ifos.inject_signal(waveform_generator=waveform_generator, 
                   parameters=injection_parameters)
logger.info("Injection generated successfully!")

# Compute the network SNR:
snr_list = []
for ifo in ifos:
    snr = ifo.meta_data['optimal_SNR']
    snr_list.append(snr)
network_snr = np.sqrt(sum(s**2 for s in snr_list))
logger.info(f"Network SNR: {network_snr:.2f}")

##################
### LIKELIHOOD ###
##################

# Setup likelihood
sampling_waveform_generator = bilby.gw.WaveformGenerator(
    duration=duration,
    sampling_frequency=sampling_frequency,
    start_time=start_time,
    frequency_domain_source_model=bilby.gw.source.binary_neutron_star_frequency_sequence,
    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_neutron_star_parameters,
    waveform_arguments=waveform_arguments)

logger.info(f"Going to construct bins with:")
logger.info(f"  - delta: {args.relative_binning_delta}")
logger.info(f"  - minimum_bin_threshold: {args.minimum_bin_threshold}")
likelihood = RelBinning(
    interferometers=ifos,
    waveform_generator=sampling_waveform_generator, 
    ref_injection=reference_parameters,
    priors=priors,
    delta=args.relative_binning_delta,
    minimum_bin_threshold=args.minimum_bin_threshold)

if args.dry_run:
    logger.info("DRY RUN: Setup validation complete. Skipping sampling.")
    logger.info(f"Would run sampling with:")
    logger.info(f"  - Label: {args.label}")
    logger.info(f"  - Prior: {args.prior_name}")
    logger.info(f"  - Output directory: {full_outdir}")
    logger.info(f"  - Interferometers: {[ifo.name for ifo in ifos]}")
    logger.info(f"  - Duration: {duration}s")
    logger.info(f"  - Minimum frequency: {minimum_frequency}Hz")
    logger.info(f"  - Approximant: {approximant}")
    logger.info("DRY RUN: All checks passed. Ready for actual sampling.")
else:
    logger.info("Starting up the sampler now . . .")
    
    if args.test_sampling:
        logger.info("Test sampling mode enabled - using 30 second timeout")
        try:
            with timeout(10):
                result = bilby.run_sampler(likelihood=likelihood,
                                           priors=priors,
                                           npool=args.n_pool,
                                           verbose=True, 
                                           sampler='dynesty',
                                           nlive=1024,
                                           outdir=full_outdir,
                                           label=args.prior_name,
                                           naccept=60,
                                           check_point_plot=True,
                                           check_point_delta_t=3600,
                                           dlogz=0.1,
                                           print_method='interval-60',
                                           sample = 'acceptance-walk',
                                           )
                result.plot_corner(priors = True)
                logger.info("✓ Test sampling completed successfully within timeout!")
        except TimeoutError as e:
            logger.error(f"✗ {e}")
            logger.error("Sampling appears to be stuck or very slow. Check conditional NF implementation.")
            sys.exit(1)
    else:
        result = bilby.run_sampler(likelihood=likelihood,
                                   priors=priors,
                                   npool=args.n_pool,
                                   verbose=True, 
                                   sampler='dynesty',
                                   nlive=1024,
                                   outdir=full_outdir,
                                   label=args.prior_name,
                                   naccept=60,
                                   check_point_plot=True,
                                   check_point_delta_t=3600,
                                   dlogz=0.1,
                                   print_method='interval-60',
                                   sample = 'acceptance-walk',
                                   )
        result.plot_corner(priors = True)
        logger.info(f"Parameter estimation complete! Results saved to {full_outdir}")