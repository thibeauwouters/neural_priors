'''
Script to do parameter estimation using relative binning code
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
from bilby.core.prior.dict import NFConditionalPrior
from bilby.core.prior.joint import NFDist, NFPrior
import argparse
import json
import matplotlib.pyplot as plt
from bilby.core.utils import logger
import warnings
import signal
from contextlib import contextmanager
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
parser.add_argument('--GW-event',
                    type = str,
                    default = 'GW170817',
                    help = "GW-event to analyze, which is also used as label.")
parser.add_argument('--prior-name',
                    type = str,
                    default = 'bns',
                    help = "Output directory for the run")
parser.add_argument('--eos-samples-name',
                    type = str,
                    default = "radio",
                    help = "Name of the EOS posterior dataset, must match the NF name. Used to construct the path to the NF model data.") # TODO: does it have to be relative?
parser.add_argument('--population-type',
                    type = str,
                    default = "uniform",
                    help = "Population type for NF model path construction (e.g., uniform)")
parser.add_argument('--use-flowjax',
                    action = 'store_true',
                    help = "Use FlowJAX for the NF model. If not set, will use PyTorch.")
parser.add_argument('--waveform-model',
                    type = str,
                    default = 'IMRPhenomXP_NRTidalv3',
                    help = "Waveform model to be used for the run")
parser.add_argument('--relative-binning-delta', 
                    type = float,
                    default = 1e-2,
                    help = "The total error on the relative binning likelihood at the reference value.")
parser.add_argument('--minimum-bin-threshold', 
                    type = int,
                    default = 1000,
                    help = "The minimum number of bins allowed to have, otherwise rerun the bin construction method in uu_relative_binning.")
parser.add_argument('--GW170817-HV',
                    action = 'store_true',
                    help = "Run GW170817 only with HV, not HLV")
parser.add_argument('--output-dir',
                    type = str,
                    default = '',
                    help = "Base path, with subdirs being the priors, to which we store the results. If empty, will use the current working directory.")
parser.add_argument('--seed', 
                    type = int,
                    default = 2024,
                    help = "Seed for the random number generator")
parser.add_argument('--dry-run',
                    action = 'store_true',
                    help = "Run through the script without sampling to validate setup")
parser.add_argument('--test-sampling',
                    action = 'store_true',
                    help = "Test sampling with 30 second timeout for debugging")
parser.add_argument('--n-pool',
                    type = int,
                    default = 64,
                    help = "How many cores to use for the sampling.")
group = parser.add_mutually_exclusive_group()
group.add_argument('--use-relative-binning',
                   dest='use_relative_binning',
                   action='store_true',
                   help='Use relative binning likelihood (default).')
group.add_argument('--no-use-relative-binning',
                   dest='use_relative_binning',
                   action='store_false',
                   help='Do not use relative binning (use regular GW likelihood).')
parser.set_defaults(use_relative_binning=True)
group.add_argument('--use-analytic-binning-scheme',
                   dest='use_analytic_binning_scheme',
                   action='store_true',
                   help='Use the analytical formula for the relative binning bin construction methods')
group.add_argument('--no-use-analytic-binning-scheme',
                   dest='use_analytic_binning_scheme',
                   action='store_false',
                   help='Use UU relative binning code for the construction of the bins for relative binning.')
parser.set_defaults(use_analytic_binning_scheme=True)

args = parser.parse_args()

# Event configuration mapping
# TODO: check if duration must be changed
EVENT_CONFIG = {
    'GW170817': {
        'ifo_list': ['H1', 'L1', 'V1'],
        'duration': 128.0,
        'minimum_frequency': 23.1,
        'maximum_frequency': 2048.0,
        'sampling_frequency': 2*2048.0
    },
    'GW190425': {
        'ifo_list': ['L1', 'V1'],
        'duration': 128.0,
        'minimum_frequency': 20.0,
        'maximum_frequency': 2048.0,
        'sampling_frequency': 2*2048.0
    },
    'GW230529': {
        'ifo_list': ['L1'],
        'duration': 128.0,
        'minimum_frequency': 20.0,
        'maximum_frequency': 1792.0,
        'sampling_frequency': 2*2048.0
    }
}

################
### PREAMBLE ###
################

# Check if some of the given arguments are valid
SUPPORTED_PRIORS = ['default', 'default_nsbh', 'default_nsbh_primary', 'bns', 'nsbh', 'bbh']
SUPPORTED_POPULATIONS = ['uniform', 'gaussian', 'double_gaussian', 'GW170817', 'GW190425', 'GW230529']
SUPPORTED_EVENTS = list(EVENT_CONFIG.keys())

if args.prior_name not in SUPPORTED_PRIORS:
    raise ValueError(f"Invalid prior name provided. Please provide one of {SUPPORTED_PRIORS}")

if args.population_type not in SUPPORTED_POPULATIONS:
    raise ValueError(f"Invalid population type provided. Please provide one of {SUPPORTED_POPULATIONS}")

if "GW" in args.population_type and args.population_type != args.GW_event:
    raise ValueError(f"GW-event population type {args.population_type} is different from the GW-event {args.GW_event}. Please ensure they match.")

if args.GW_event not in SUPPORTED_EVENTS:
    raise ValueError(f"Invalid event provided. Please provide one of {SUPPORTED_EVENTS}")


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
from LikelihoodRB import RelBinning

output_dir = args.output_dir
if len(output_dir) == 0:
    # If no output directory is given, use the script directory as base
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = script_dir
    logger.info("No output directory provided, using script directory as base.")
    
logger.info(f"Output directory is set to: {output_dir}. Make sure it exists!")

full_outdir = os.path.join(output_dir, args.GW_event, args.population_type, args.prior_name, args.eos_samples_name)
reference_parameters_filename = os.path.join(output_dir, args.GW_event, "reference_parameters.json")
prior_filename = os.path.join(output_dir, args.GW_event, "prior.prior")
    
bilby.core.utils.setup_logger(outdir=full_outdir, label=args.GW_event)
logger.info(f"We set full_outdir to {full_outdir}")

# If a dry run, then make sure we always get pased the relative binning
if args.dry_run:
    dry_run_bin_threshold = 10 # just a low value
    logger.info(f"Running in dry run mode, setting minimum_bin_threshold to {dry_run_bin_threshold}")
    args.minimum_bin_threshold = dry_run_bin_threshold

# Load reference parameters and auto-configure event settings
with open(reference_parameters_filename, 'r') as f:
    reference_parameters = json.load(f)

# Get event configuration
event_config = EVENT_CONFIG[args.GW_event]
if args.GW_event == 'GW170817' and args.GW170817_HV:
    # If we only want to use HV, then we change the ifo_list -- this is mainly to understand what is going on here
    logger.info("Running GW170817 with only H1 and V1 interferometers")
    event_config['ifo_list'] = ['H1', 'V1']
ifo_list = event_config['ifo_list']
duration = event_config['duration']

# Use geocent_time from reference parameters if available, otherwise from config
if 'geocent_time' in reference_parameters:
    geocent_time = reference_parameters['geocent_time']
    del reference_parameters['geocent_time']  # Remove to avoid confusion later
    logger.info(f"Using geocent_time from reference parameters: {geocent_time}")
else:
    raise ValueError(f"geocent_time not found in reference parameters for {args.GW_event} -- PE cannot be run without this information.")
    
# Need to do a few conversions on these parameters 
chirp_mass = reference_parameters["chirp_mass"]
symmetric_mass_ratio = reference_parameters["symmetric_mass_ratio"]
mass_ratio = bilby.gw.conversion.symmetric_mass_ratio_to_mass_ratio(symmetric_mass_ratio)
mass_1, mass_2 = bilby.gw.conversion.chirp_mass_and_mass_ratio_to_component_masses(chirp_mass, mass_ratio)
    
# Now save the conversions
reference_parameters["mass_ratio"] = mass_ratio
reference_parameters["mass_1"] = mass_1 # TODO: source frame or detector frame? To check!
reference_parameters["mass_2"] = mass_2

logger.info(f"Showing the reference parameters for {args.GW_event}")
for key, value in reference_parameters.items():
    logger.info(f"{key}: {value}")

# Event parameters are now auto-configured above
start_time = geocent_time - duration + 2
logger.info(f"Geocent time is {geocent_time} and duration is {duration}")

reference_parameters['geocent_time'] = geocent_time
reference_parameters['duration'] = duration

# Other fixed kwargs
# TODO: do we want fmin and fmax to be set here from the per-event config?
approximant = args.waveform_model

logger.info(f"Using waveform: {approximant}")
logger.info(f"Relative binning delta {args.relative_binning_delta}")

waveform_arguments = dict(
    waveform_approximant=approximant, 
    reference_frequency=event_config["minimum_frequency"],
    minimum_frequency=event_config["minimum_frequency"],
    maximum_frequency=event_config["maximum_frequency"])

waveform_generator = bilby.gw.WaveformGenerator(
    duration=duration,
    sampling_frequency=event_config["sampling_frequency"],
    start_time=start_time,
    frequency_domain_source_model=bilby.gw.source.lal_binary_neutron_star,                    
    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_neutron_star_parameters,
    waveform_arguments=waveform_arguments)

logger.info(f"Running with label {args.GW_event} and ifo_list {ifo_list}")
np.random.seed(args.seed)
bilby.core.utils.random.seed(args.seed)

ifos = bilby.gw.detector.InterferometerList(ifo_list)
# Make sure the minimum/maximum frequency is set correctly
for ifo in ifos:
    ifo.minimum_frequency=event_config["minimum_frequency"]
    ifo.maximum_frequency=event_config["maximum_frequency"]

# TODO: this needs to be moved to a common utils file, so that we can use it in the NF prior training script as well
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

logger.info(f"Loading priors from {prior_filename}")

### PRIORS
# Read the prior file as text
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

if args.prior_name == "default":
    logger.info("Using the default priors")
    # Build PriorDict straight from all given priors
    priors = bilby.core.prior.PriorDict(prior_dict)
    
    # Lambdas are not in the prior files, so we add them manually here
    priors["lambda_1"] = bilby.core.prior.Uniform(minimum=0.0, maximum=5000.0, name='lambda_1', latex_label='$\Lambda_1$')
    priors["lambda_2"] = bilby.core.prior.Uniform(minimum=0.0, maximum=5000.0, name='lambda_2', latex_label='$\Lambda_2$')
    
elif args.prior_name == "default_nsbh":
    logger.info("Using the default priors but with the NSBH assumption (i.e., lambda_1 = 0.0)")
    
    # Build PriorDict straight from all given priors
    priors = bilby.core.prior.PriorDict(prior_dict)
    
    # Lambdas are not in the prior files, so we add them manually here
    priors["lambda_1"] = DeltaFunction(0.0, name='lambda_1', latex_label='$\Lambda_1$')
    priors["lambda_2"] = bilby.core.prior.Uniform(minimum=0.0, maximum=5000.0, name='lambda_2', latex_label='$\Lambda_2$')
    
elif args.prior_name == "default_nsbh_primary":
    logger.info("Using the default priors but with the NSBH assumption, where the NS is the primary (i.e., lambda_2 = 0.0)")
    
    # Build PriorDict straight from all given priors
    priors = bilby.core.prior.PriorDict(prior_dict)
    
    # Lambdas are not in the prior files, so we add them manually here
    priors["lambda_1"] = bilby.core.prior.Uniform(minimum=0.0, maximum=5000.0, name='lambda_1', latex_label='$\Lambda_1$')
    priors["lambda_2"] = DeltaFunction(0.0, name='lambda_2', latex_label='$\Lambda_2$')
    
else:
    logger.info(f"Sampling with an NF prior, with name {args.prior_name}")
    
    # First, drop chirp_mass, mass_ratio and luminosity_distance from the prior_dict, as these are modelled by the NF
    prior_dict.pop('chirp_mass', None)
    prior_dict.pop('mass_ratio', None)
    if "GW" in args.population_type:
        logger.info(f"This is a GW population type run, so popping the luminosity_distance from the prior_dict")
        prior_dict.pop('luminosity_distance', None)
        
    # Path to NF model - updated to match new folder structure
    if args.use_flowjax:
        nf_model_path = os.path.join(base_dir, f"NFprior/models/{args.population_type}/{args.prior_name}/{args.eos_samples_name}_flowjax/model.eqx")
    else:
        nf_model_path = os.path.join(base_dir, f"NFprior/models/{args.population_type}/{args.prior_name}/{args.eos_samples_name}/model.pt")
    nf_model_path = os.path.abspath(nf_model_path)
    logger.info(f"Using NF model path: {nf_model_path}")
    
    nf_kwargs_filename = nf_model_path.replace('.pt', '_kwargs.json').replace('.eqx', '_kwargs.json')
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
        use_tilde=nf_kwargs["use_tilde"],
        use_component_masses=nf_kwargs["use_component_masses"]
        )
    
    # Add NFPrior for each name that is in the dist
    for name in nf_dist.names:
        prior_dict[name] = NFPrior(dist=nf_dist, name=name)
        
    if args.prior_name == "nsbh":
        # Put lambda_1 = 0.0
        prior_dict["lambda_1"] = DeltaFunction(0.0, name='lambda_1', latex_label='$\Lambda_1$')
        
    priors = bilby.core.prior.PriorDict(prior_dict)
        
logger.info("Going to show priors in priors:")
for key, value in priors.items():
    logger.info(f"      {key}: {value}")
    
### GW DATA GENERATION
logger.info(f"Running through data generation steps now")
data_path = os.path.join(base_dir, 'data', args.GW_event)
if args.GW_event == 'GW190425':
    frame_files = {
        "L1": os.path.join(data_path, "L-L1_GWOSC_16KHZ_R1-1240213455-4096.gwf"),
        "V1": os.path.join(data_path, "V-V1_GWOSC_16KHZ_R1-1240213455-4096.gwf")
    }
    channels_dict = {
        "L1": "L1:GWOSC-16KHZ_R1_STRAIN", 
        "V1": "V1:GWOSC-16KHZ_R1_STRAIN"
    }
    psd_files = {
        "L1": os.path.join(data_path, "glitch_median_PSD_forLI_L1_srate8192.txt"),
        "V1": os.path.join(data_path, "glitch_median_PSD_forLI_V1_srate8192.txt")
    }
elif args.GW_event == 'GW170817':
    frame_files = {
        "H1": os.path.join(data_path, "H-H1_LOSC_CLN_16_V1-1187007040-2048.gwf"),
        "L1": os.path.join(data_path, "L-L1_LOSC_CLN_16_V1-1187007040-2048.gwf"),
        "V1": os.path.join(data_path, "V-V1_LOSC_CLN_16_V1-1187007040-2048.gwf")
    }
    channels_dict = {
        "H1": "H1:LOSC-STRAIN",
        "L1": "L1:LOSC-STRAIN", 
        "V1": "V1:LOSC-STRAIN"
    }
    psd_files = {
        "H1": os.path.join(data_path, "h1_psd.txt"),
        "L1": os.path.join(data_path, "l1_psd.txt"),
        "V1": os.path.join(data_path, "v1_psd.txt")
    }
elif args.GW_event == 'GW230529':
    channels_dict = {
        "L1": "L1:GWOSC-16KHZ_R1_STRAIN",
    }
    frame_files = {
        "L1": os.path.join(data_path, "L-L1_GWOSC_16KHZ_R1-1369417271-4096.gwf"),
    }
    psd_files = {
        "L1": os.path.join(data_path, "psd_4096.dat"),
    }
else:
    raise ValueError("GW event not recognized. Please provide one of 'GW170817', 'GW190425', or 'GW230529'.")

# Load the data into the ifos:
for ifo in ifos:
    ifo.set_strain_data_from_frame_file(
        frame_files[ifo.name],
        event_config["sampling_frequency"],
        duration,
        start_time=start_time,
        channel=channels_dict[ifo.name])
    
    # TODO: Ensure PSD, not ASD, for different events
    ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(psd_file=psd_files[ifo.name])

# Log frequency arrays for each IFO
for ifo in ifos:
    logger.info(f"Frequency array for {ifo.name}: {ifo.frequency_array}")
    
logger.info("Priors loaded:")

### LIKELIHOOD
sampling_waveform_generator = bilby.gw.WaveformGenerator(
    duration=duration,
    sampling_frequency=event_config["sampling_frequency"],
    start_time=start_time,
    frequency_domain_source_model=bilby.gw.source.binary_neutron_star_frequency_sequence,
    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_neutron_star_parameters,
    waveform_arguments=waveform_arguments)

if args.use_relative_binning:
    logger.info(f"Using relative binning likelihood")
    logger.info(f"Going to construct bins with:")
    logger.info(f"  - delta: {args.relative_binning_delta}")
    logger.info(f"  - minimum_bin_threshold: {args.minimum_bin_threshold}")
    likelihood = RelBinning(
        interferometers=ifos,
        waveform_generator=sampling_waveform_generator, 
        ref_injection=reference_parameters,
        priors=priors,
        delta=args.relative_binning_delta,
        minimum_bin_threshold=args.minimum_bin_threshold,
        use_analytic_method=args.use_analytic_binning_scheme)
else:
    logger.info(f"Using regular GW likelihood")
    likelihood = bilby.gw.GravitationalWaveTransient(
        interferometers=ifos,
        waveform_generator=waveform_generator)


if args.dry_run:
    logger.info("DRY RUN: Setup validation complete. Skipping sampling.")
    logger.info(f"Would run sampling with:")
    logger.info(f"  - Event: {args.GW_event}")
    logger.info(f"  - Prior: {args.prior_name}")
    logger.info(f"  - Output directory: {full_outdir}")
    logger.info(f"  - Interferometers: {[ifo.name for ifo in ifos]}")
    logger.info(f"  - Duration: {duration}s")
    logger.info(f"  - Minimum frequency: {event_config['minimum_frequency']}Hz")
    logger.info(f"  - Approximant: {approximant}")
    logger.info(f"  - npool: {args.npool}")
    logger.info("DRY RUN: All checks passed. Ready for actual sampling.")
else:
    logger.info("Starting up the sampler now . . .")
    
    logger.info(f"npool is set to: {args.n_pool}")
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
                                           sample='acceptance-walk',
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

############ The following block is to validate the PE results #####################


# N = 1000
# posterior = result.posterior
# calculate_efficiency = True
# if calculate_efficiency:
#     choice_idx = np.random.choice(np.arange(len(posterior)), N, replace = False)
#     collect_relative_binning_likelihood = []
#     collect_exact_likelihood = []

#     for ii in choice_idx:
#         parameters = dict(posterior.iloc[ii])
#         parameters.pop('log_prior')
#         rb_logl = parameters.pop('log_likelihood')

#         collect_relative_binning_likelihood.append(rb_logl)

#         exact_likelihood.parameters = parameters
#         collect_exact_likelihood.append(exact_likelihood.log_likelihood_ratio())

    
#     collect_exact_likelihood = np.array(collect_exact_likelihood)
#     collect_relative_binning_likelihood = np.array(collect_relative_binning_likelihood)
#     np.savetxt(f'{args.outdir}/{args.GW_event}_comparison_likelihoods.dat', np.vstack([choice_idx, collect_relative_binning_likelihood, collect_exact_likelihood]).T)



#     weights = np.exp(collect_exact_likelihood - collect_relative_binning_likelihood)
#     efficiency = np.mean(weights)**2 / np.mean(weights**2)
#     logger.info(f'Efficiency of the run: {efficiency}')

#     likelihood_differences = collect_relative_binning_likelihood - collect_exact_likelihood
#     fig, ax = plt.subplots(1, 1)
#     ax.hist(likelihood_differences, bins = 20)
#     ax.set_xlabel('LRelBin - LExact')
#     ax.grid()
#     fig.savefig(f"{args.outdir}/{args.GW_event}_likelihood_errors.png")