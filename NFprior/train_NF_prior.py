"""
Train a normalizing flow to approximate a distribution on masses and Lambdas to replicate an EOS dataset and be used in inference.
This might be joint prior or conditional prior, we will have to check which works best later on. 
"""

# FIXME: global NF (i.e., not per event) seems broken

import os
import argparse
import numpy as np
import json

### bilby imports
import bilby 
from bilby.core.prior.analytical import Uniform, Sine, Cosine
from bilby.gw.prior import UniformComovingVolume, AlignedSpin

from bilby.gw.conversion import (
    luminosity_distance_to_redshift,
    chirp_mass_and_mass_ratio_to_component_masses,
    lambda_1_lambda_2_to_lambda_tilde,
    lambda_1_lambda_2_to_delta_lambda_tilde
)

### glasflow imports
from glasflow.flows import RealNVP
from glasflow.flows.autoregressive import MaskedAffineAutoregressiveFlow
from glasflow.flows.nsf import CouplingNSF
import torch
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

### flowjax imports
import jax
import jax.numpy as jnp
import jax.random as jr
from flowjax.flows import masked_autoregressive_flow, coupling_flow
from flowjax.flows import block_neural_autoregressive_flow
from flowjax.train import fit_to_data
from flowjax.distributions import Normal
import equinox as eqx

import tqdm
import time
import joblib

import matplotlib.pyplot as plt
params = {"axes.grid": True,
          "text.usetex" : False}
plt.rcParams.update(params)

from sklearn.preprocessing import MinMaxScaler

if torch.cuda.is_available():
    print(f"torch: CUDA is available. Number of devices: {torch.cuda.device_count()}")
    print(f"torch: Current CUDA device: {torch.cuda.current_device()}")
else:
    print("torch: CUDA is not available.")

# Also using JAX:
jax_devices = jax.devices()
print(f"JAX: devices available: {jax_devices}")

# Get the device as well:
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description="Train a normalizing flow prior on EOS samples.")
parser.add_argument("--eos-samples-name", 
                    type=str, 
                    default="radio", 
                    help="EOS samples name (default: radio)")
parser.add_argument("--source-type", 
                    type=str, 
                    default="bns", 
                    choices=["bns", "nsbh"], 
                    help="Type of source to model")
parser.add_argument("--event-name", 
                    type=str, 
                    default="", 
                    help="The name of the GW event for which to train the NF prior. This will load in the priors and use them for creating specific training data.")
parser.add_argument("--conditional", 
                    action="store_true", 
                    help="Use conditional NF training")
parser.add_argument("--no-conditional", 
                    dest="conditional", 
                    action="store_false", 
                    help="Disable conditional NF training")
parser.set_defaults(conditional=False)
parser.add_argument("--use-tilde", 
                    action="store_true", 
                    help="Use tilde parameterization for lambdas (lambda_tilde, delta_lambda_tilde) instead of (lambda_1, lambda_2)")
parser.add_argument("--no-use-tilde", 
                    dest="use_tilde", 
                    action="store_false")
parser.set_defaults(use_tilde=False)
parser.add_argument("--use-component-masses", 
                    action="store_true", 
                    help="Use component masses (m1, m2) instead of (Mc, q)")
parser.add_argument("--no-use-component-masses", 
                    dest="use_component_masses", 
                    action="store_false")
parser.set_defaults(use_component_masses=False)
parser.add_argument("--include-dL", 
                    action="store_true", 
                    help="Include luminosity distance in unconditional tilde models for 5D training")
parser.add_argument("--no-include-dL", 
                    dest="include_dL", 
                    action="store_false")
parser.set_defaults(include_dL=True)
parser.add_argument("--N-samples-training", 
                    type=int, 
                    default=50_000, 
                    help="Number of training samples")
parser.add_argument("--N-samples-plot", 
                    type=int, 
                    default=10_000, 
                    help="Number of samples for plotting")
parser.add_argument("--m-max-BH", 
                    type=float, 
                    default=5.0, 
                    help="Maximum BH mass for NSBH sources")
parser.add_argument("--take-log-lambda", 
                    action="store_true", 
                    help="Take log of Lambda before training")
parser.add_argument("--no-take-log-lambda", 
                    dest="take_log_lambda", 
                    action="store_false")
parser.set_defaults(take_log_lambda=False)
parser.add_argument("--use-flowjax", 
                    action="store_true", 
                    help="Use flowJAX instead of glasflow")
parser.add_argument("--no-use-flowjax", 
                    dest="use_flowjax", 
                    action="store_false")
parser.set_defaults(use_flowjax=False)
parser.add_argument("--scale-input", 
                    action="store_true", 
                    help="Scale input before NF training")
parser.add_argument("--no-scale-input", 
                    dest="scale_input", 
                    action="store_false")
parser.set_defaults(scale_input=True)
parser.add_argument("--num-epochs", 
                    type=int, 
                    default=500,
                    help="Number of training epochs")
parser.add_argument("--learning-rate", 
                    type=float, 
                    default=1e-3, 
                    help="Learning rate")
parser.add_argument("--batch-size", 
                    type=int, 
                    default=1024, 
                    help="Batch size")
parser.add_argument("--max-patience", 
                    type=int, 
                    default=50, 
                    help="Max patience for early stopping")
parser.add_argument("--n-transforms", 
                    type=int, 
                    default=4, 
                    help="Number of NF transforms")
parser.add_argument("--n-neurons", 
                    type=int, 
                    default=64, 
                    help="Number of neurons per layer")
parser.add_argument("--n-blocks-per-transform", 
                    type=int, 
                    default=1, 
                    help="Number of blocks per transform")
parser.add_argument("--num-bins", 
                    type=int, 
                    default=10, 
                    help="Number of bins for spline flows")
parser.add_argument("--nn-depth", 
                    type=int, 
                    default=5, 
                    help="Depth of flowJAX network")
parser.add_argument("--nn-block-dim", 
                    type=int, 
                    default=8, 
                    help="Block dimension of flowJAX network")

    
class NFPriorCreator:
    """
    Class to construct the NF prior and train it.
    """
    
    def __init__(self,
                 eos_samples_name: str = "radio",
                 source_type: str = "bns",
                 event_name: str = "",
                 use_tilde: bool = False,
                 use_component_masses: bool = True,
                 conditional: bool = True,
                 include_dL: bool = False,
                 N_samples_training: int = 100_000,
                 N_samples_plot: int = 10_000,
                 m_max_BH: float = 5.0,
                 take_log_lambda: bool = False,
                 use_flowjax: bool = False,
                 num_epochs: int = 500,
                 learning_rate: float = 1e-3,
                 max_patience: int = 50,
                 batch_size: int = 1024,
                 scale_input: bool = True,
                 # glasflow-specific training arguments:
                 n_transforms: int = 4,
                 n_neurons: int = 128,
                 n_blocks_per_transform: int = 4,
                 num_bins: int = 10,
                 # flowJAX-specific training arguments:
                 nn_depth: int = 5,
                 nn_block_dim: int = 8
                 ):
        """
        Initialize the NFPriorCreator class with the necessary parameters.

        Args:
            eos_samples_name (str, optional): Name of the run from which we load the EOS samples from, which will be converted into the training data for the NF for binary systems. Defaults to `radio`, which only uses the radio timing constraints on MTOV.
            source_type (str, optional): Which kind of source to model: `bns` or `nsbh`. Defaults to "bns".
            event_name (str, optional): The name of the GW event for which to train the NF prior. This will load in the priors and use them for creating specific training data. Defaults to "".
            N_samples_training (int, optional): Number of training samples to create.. Defaults to 100_000.
            N_samples_plot (int, optional): Number of samples to create the plots. Defaults to 10_000.
            m_max_BH (float, optional): If generating NSBH training data with an NF that is not conditioned on the masses, this is up to which the masses are taken. Defaults to 5.0.
            save_name (str, optional): Where to save the models etc to. Defaults to "".
            conditional (bool, optional): Whether to train the NF in a conditional manner, i.e., Lambdas as function of masses. Defaults to True as this is recommended for our bilby setup.
            take_log_lambda (bool, optional): Whether to take the log of the Lambdas before training to deal with their massive scaling, to improve training the NF. Defaults to True.
            use_flowjax (bool, optional): Whether to use flowJAX instead of glasflow for training. Defaults to False.
            use_tilde (bool, optional): Whether to use tilde parameterization for lambdas (lambda_tilde, delta_lambda_tilde) instead of (lambda_1, lambda_2). Defaults to False.
            use_component_masses (bool, optional): Whether to use component masses (m1, m2) instead of (Mc, q). Defaults to True.
            num_epochs (int, optional): Number of training epochs. Defaults to 100.
            learning_rate (float, optional): Learning rate for training. Defaults to 1e-3.
            max_patience (int, optional): Max stops to wait before employing early stopping. Defaults to 50.
            n_transforms (int, optional): Number of transforms in NF. Defaults to 2.
            n_neurons (int, optional): Number of neurons in NF. Defaults to 64.
            batch_size (int, optional): Batch size for NF training. Defaults to 256.
            n_blocks_per_transform (int, optional): Number of blocks per transform for NF. Defaults to 2.
            scale_input (bool, optional): Whether to scale the input for the NF before training. Defaults to False.

        Raises:
            ValueError: If source type is not one of the supported types, i.e., "bns" or "nsbh".
        """
        
        self.eos_samples_name = eos_samples_name
        self.eos_samples_filename = f"../data/eos/{self.eos_samples_name}/eos_samples.npz"
        
        if not os.path.exists(self.eos_samples_filename):
            # Show all subdirs in 
            available_subdirs = os.listdir("../data/eos/")
            raise ValueError(f"File {self.eos_samples_filename} does not exist. Please check the path or the `eos_samples_name` argument. Available subdirs: {available_subdirs}.")
        print(f"Training data will be loaded from: {self.eos_samples_filename}")
        
        SUPPORTED_SOURCE_TYPES = ["bns", "nsbh"]
        if source_type not in SUPPORTED_SOURCE_TYPES:
            raise ValueError(f"source_type must be one of {SUPPORTED_SOURCE_TYPES}, got {source_type} instead.")
        
        self.source_type = source_type
        
        if len(event_name) > 0:
            # An event name was given, check if we have the PE ready for that event
            base_GW_events_dir = "../GW_runs/"
            all_GW_events = os.listdir(base_GW_events_dir)
            all_GW_events = [ev for ev in all_GW_events if ev.startswith("GW")]
            
            if not event_name in all_GW_events:
                raise ValueError(f"Event {event_name} not found in {base_GW_events_dir}. Available events: {all_GW_events}.")
            
        self.event_name = event_name
        print(f"Training a normalizing flow for {self.source_type} sources with event name set to {self.event_name}")
        self.N_samples_training = N_samples_training
        self.N_samples_plot = N_samples_plot
        self.m_max_BH = m_max_BH
        self.conditional = conditional
        self.take_log_lambda = take_log_lambda
        self.use_flowjax = use_flowjax
        self.use_tilde = use_tilde
        self.use_component_masses = use_component_masses
        self.include_dL = include_dL

        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.max_patience = max_patience
        self.n_transforms = n_transforms
        self.n_neurons = n_neurons
        self.nn_depth = nn_depth
        self.nn_block_dim = nn_block_dim
        self.batch_size = batch_size
        self.n_blocks_per_transform = n_blocks_per_transform
        self.scale_input = scale_input
        self.num_bins = num_bins
        
        if self.scale_input and self.take_log_lambda:
            raise ValueError("Cannot scale input and take log of Lambdas at the same time. \
                Please set either scale_input=False or take_log_lambda=False.\
                Recommended to use scaling and not log transformation for training (empirically gave best results).")
        
        if not self.use_component_masses and self.source_type == "nsbh" and self.conditional:
            raise ValueError("Combination of use_component_masses=False, source_type='nsbh', and conditional=True is not supported. \
                Please use component masses for NSBH conditional training or switch to unconditional training.")
        
        if self.include_dL and self.conditional:
            raise ValueError("include_dL=True is only supported for unconditional models. Set conditional=False to use dL.")
        
        # Check flowJAX availability if requested
        if self.use_flowjax and not globals().get('jax'):
            raise ImportError("flowJAX requested but JAX/flowJAX not available. Install flowJAX or set use_flowjax=False")
        
        # Make an outdir based on the given name etc, so that everything is stored in one directory for later on
        backend_suffix = "_flowjax" if self.use_flowjax else ""
        
        if self.event_name:
            # If an event name was given, we save the model in a specific directory for that event
            base_dir = f"./models/{self.event_name}/"
        else:
            # If no event name was given, we save the model in a general directory for the source type
            base_dir = f"./models/{self.source_type}/"
            
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
            print(f"Created base directory {base_dir}")
        
        if self.conditional:
            self.outdir = os.path.join(f"./{base_dir}{self.eos_samples_name}_conditional_{self.source_type}{backend_suffix}")
        else:
            self.outdir = os.path.join(f"./{base_dir}{self.eos_samples_name}_{self.source_type}{backend_suffix}")
            
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)
            print(f"Created output directory {self.outdir}")
        print(f"Everything for this model will be saved to {self.outdir}")
    
    def load_eos_samples_from_file(self) -> tuple[np.array, np.array, np.array]:
        """
        Load in the EOS samples from the file and clean them. This returns the masses, radii and Lambdas of the EOS samples.
        """
        if not os.path.exists(self.eos_samples_filename):
            raise ValueError(f"File {self.eos_samples_filename} does not exist.")
        
        print(f"Reading the EOS data from {self.eos_samples_filename}")
        eos_samples = np.load(self.eos_samples_filename)

        # Get the data
        masses_EOS, radii_EOS, Lambdas_EOS = eos_samples["masses_EOS"], eos_samples["radii_EOS"], eos_samples["Lambdas_EOS"]
        
        # Iterate over EOS and keep those that are fine
        nb_samples = len(masses_EOS)
        good_idx = np.ones(nb_samples, dtype=bool)

        # There are sometimes a few (not many) bad EOSs, so get rid of them first
        for i in range(nb_samples):
            # First, sometimes the radius can be very large for low mass stars, which is unphysical
            bad_radii = (masses_EOS[i] > 1.0) * (radii_EOS[i] > 20.0)
            if any(bad_radii):
                good_idx[i] = False
                continue
            # Second, sometimes a negative Lambda was computed, remove that
            bad_Lambdas = (Lambdas_EOS[i] < 0.0)
            if any(bad_Lambdas):
                good_idx[i] = False
                continue
            # Finally, we want the TOV mass to be above 2.0 M_odot
            bad_MTOV = np.max(masses_EOS) < 2.0
            if bad_MTOV:
                good_idx[i] = False
                continue

        print("Number of good samples: ", np.sum(good_idx) / nb_samples)

        masses_EOS = masses_EOS[good_idx]
        radii_EOS = radii_EOS[good_idx]
        Lambdas_EOS = Lambdas_EOS[good_idx]
        
        return masses_EOS, radii_EOS, Lambdas_EOS
    
    def create_data(self):
        """
        Switch to the correct data generation method.
        In case no event name was specified, this means we generate m1, m2 data from [1, MTOV] Msun from the EOS data.
        In case an event name was specified, we load in the priors, and generate m1, m2 (source-frame) training data from the priors.
        """
        
        if self.event_name:
            print(f"Creating training data for event {self.event_name} using the priors")
            self.create_data_from_priors()
        else:
            print("Creating training data from EOS samples")
            self.create_data_uniform()
            
    def create_data_from_priors(self):
        """
        Generate an NF prior that is trained specifically on masses generated from the priors of the event.
        """
        
        # Locate the prior file for the event
        prior_file = os.path.join("../GW_runs", self.event_name, "prior.prior")
        prior_file = os.path.abspath(prior_file)
        
        # The first three lines, by design, have the Mc, q and dL priors. Use them to instantiate Bilby prior
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
        
        print(f"Loading priors from bilby prior filename: {prior_file}")
        with open(prior_file, "r") as f:
            prior_lines = f.readlines()[:3]
            
        # Replace 'UniformComovingVolume' with full path
        modified_lines = [
            line.replace(
                "UniformComovingVolume", "bilby.gw.prior.UniformComovingVolume"
            ) for line in prior_lines
        ]

        # Evaluate the prior lines in a safe namespace
        prior_dict = {}
        exec("".join(modified_lines), safe_globals, prior_dict)

        # Now we can create the Bilby prior
        prior = bilby.core.prior.PriorDict(prior_dict)
        print(f"Loaded priors: {prior}")
        
        # Sample parameters:
        print(f"Sampling {self.N_samples_training} parameters from the prior and converting to source frame component masses")
        samples = prior.sample(self.N_samples_training)
        chirp_mass = samples["chirp_mass"]
        mass_ratio = samples["mass_ratio"]
        luminosity_distance = samples["luminosity_distance"]
        z = luminosity_distance_to_redshift(luminosity_distance)
        mc_source = chirp_mass / (1 + z)
        m1_source, m2_source = chirp_mass_and_mass_ratio_to_component_masses(mc_source, mass_ratio)
        print(f"Sampling {self.N_samples_training} parameters DONE")
        
        # Now get Lambdas from the EOS samples
        masses_EOS, _, Lambdas_EOS = self.load_eos_samples_from_file()
        
        Lambda1_list = np.zeros_like(m1_source)
        Lambda2_list = np.zeros_like(m2_source)
        for i, (m1, m2) in enumerate(zip(m1_source, m2_source)):
            # We might have to do a few tries to get a good sample, so we set a flag
            good_sample = False
            
            # Choose an EOS index
            while not good_sample:
                # Choose a posterior sample EOS MRL curve
                idx = np.random.randint(0, len(masses_EOS))
                m, l = masses_EOS[idx], Lambdas_EOS[idx]
                
                # Generate Lambdas:
                lambda_1 = np.interp(m1, m, l)
                lambda_2 = np.interp(m2, m, l)
                
                if lambda_1 < 0.0 or lambda_2 < 0.0:
                    # If we got a negative Lambda, try again
                    print(f"Negative lambda for m1={m1}, m2={m2}, lambda_1={lambda_1}, lambda_2={lambda_2}. Trying again.")
                    continue
                
                if lambda_1 > lambda_2:
                    # If we got unphysical lambda12 pair, try again
                    print(f"Wrong lambda order for m1={m1}, m2={m2}, lambda_1={lambda_1}, lambda_2={lambda_2}. Trying again.")
                    continue
                
                # If we got here, we have a good sample
                good_sample = True
            
            Lambda1_list[i] = lambda_1
            Lambda2_list[i] = lambda_2
        
        # For numerical stability, we turn zero into a very small number with np.clip:
        Lambda1_list = np.clip(Lambda1_list, a_min=1e-4, a_max=None)
        Lambda2_list = np.clip(Lambda2_list, a_min=1e-4, a_max=None)
        
        # Also get lambda_tilde, delta_lambda_tilde for the training data
        lambda_tilde = lambda_1_lambda_2_to_lambda_tilde(Lambda1_list, Lambda2_list, m1_source, m2_source)
        delta_lambda_tilde = lambda_1_lambda_2_to_delta_lambda_tilde(Lambda1_list, Lambda2_list, m1_source, m2_source)
                
        full_save_path = os.path.join(self.outdir, f"training_data.npz")
        self.training_filename = full_save_path
        print(f"Saving training data to {full_save_path}")
        np.savez(full_save_path, 
                 m1=np.array(m1_source),
                 m2=np.array(m2_source),
                 lambda_1=np.array(Lambda1_list),
                 lambda_2=np.array(Lambda2_list),
                 # Also add the bilby stuff here just to be complete
                 chirp_mass=np.array(chirp_mass),
                 mass_ratio=np.array(mass_ratio),
                 luminosity_distance=np.array(luminosity_distance),
                 redshift=np.array(z),
                 lambda_tilde=np.array(lambda_tilde),
                 delta_lambda_tilde=np.array(delta_lambda_tilde)
                 )
        
    def create_data_uniform(self):
        """
        Create the dataset from the EOS to neutron star systems: m1, m2, Lambda1, Lambda2.
        Masses are sampled uniformly between 1.0 and MTOV for each EOS sample, therefore, such a prior can be used for any GW event.
        Always generates NS-NS pairs - the BNS vs NSBH distinction is handled in load_training_data().
        """
        
        # Load the EOS samples
        masses_EOS, _, Lambdas_EOS = self.load_eos_samples_from_file()

        # Make everything ready for sampling
        m1_list = np.empty(self.N_samples_training)
        m2_list = np.empty(self.N_samples_training)
        Lambda1_list = np.empty(self.N_samples_training)
        Lambda2_list = np.empty(self.N_samples_training)

        # Construct the prior from sampling from the EOS set - always generate NS-NS pairs
        for i in range(self.N_samples_training):
            idx = np.random.randint(0, len(masses_EOS))
            m, l = masses_EOS[idx], Lambdas_EOS[idx]
            
            # Sample two masses between 1 and MTOV for this EOS
            mtov = np.max(m)
            mass_samples = np.random.uniform(1.0, mtov, 2)
            m1 = np.max(mass_samples)
            m2 = np.min(mass_samples)
            
            Lambda_1 = np.interp(m1, m, l)
            Lambda_2 = np.interp(m2, m, l)
            
            m1_list[i] = m1
            m2_list[i] = m2
            Lambda1_list[i] = Lambda_1
            Lambda2_list[i] = Lambda_2
               
        # For numerical stability, we turn zero into a very small number with np.clip:
        Lambda1_list = np.clip(Lambda1_list, a_min=1e-4, a_max=None)
        Lambda2_list = np.clip(Lambda2_list, a_min=1e-4, a_max=None)
        
        # Also get lambda_tilde, delta_lambda_tilde for the training data
        lambda_tilde = lambda_1_lambda_2_to_lambda_tilde(Lambda1_list, Lambda2_list, m1_list, m2_list)
        delta_lambda_tilde = lambda_1_lambda_2_to_delta_lambda_tilde(Lambda1_list, Lambda2_list, m1_list, m2_list)
                
        full_save_path = os.path.join(self.outdir, f"training_data.npz")
        self.training_filename = full_save_path
        print(f"Saving training data to {full_save_path}")
        np.savez(full_save_path, 
                 m1=np.array(m1_list),
                 m2=np.array(m2_list),
                 lambda_1=np.array(Lambda1_list),
                 lambda_2=np.array(Lambda2_list),
                 lambda_tilde=np.array(lambda_tilde),
                 delta_lambda_tilde=np.array(delta_lambda_tilde)
        )
        print(f"Saving training data DONE")
        
    def load_training_data(self, training_filename: str):
        """
        Loads in the preprocessed training data before feeding it into the NF for training.
        Creates parameterization-agnostic training arrays that the training methods can use.
        
        The training methods use generic arrays:
        - train_mass_1, train_mass_2: Can be (m1, m2) or (Mc, q) depending on use_component_masses
        - train_lambda_1, train_lambda_2: Can be (lambda_1, lambda_2) or (lambda_tilde, delta_lambda_tilde) depending on use_tilde
        
        For NSBH systems, concatenates all NS data into single arrays for p(Lambda|m) modeling.
        """
        data = np.load(training_filename)
        m1_raw = data["m1"]
        m2_raw = data["m2"]
        lambda_1_raw = data["lambda_1"]
        lambda_2_raw = data["lambda_2"]
        
        # Always store the original arrays first
        self.m1_raw = m1_raw
        self.m2_raw = m2_raw
        self.lambda_1_raw = lambda_1_raw
        self.lambda_2_raw = lambda_2_raw
        
        # Handle mass parameterization
        if self.use_component_masses:
            print("Using component masses (m1, m2) for training")
            train_mass_1 = m1_raw
            train_mass_2 = m2_raw
        else:
            print("Using chirp mass and mass ratio (Mc, q) for training")
            # Load or calculate chirp mass and mass ratio
            if "chirp_mass" in data and "mass_ratio" in data:
                train_mass_1 = data["chirp_mass"]
                train_mass_2 = data["mass_ratio"]
            else:
                from bilby.gw.conversion import component_masses_to_chirp_mass, component_masses_to_mass_ratio
                train_mass_1 = component_masses_to_chirp_mass(m1_raw, m2_raw)
                train_mass_2 = component_masses_to_mass_ratio(m1_raw, m2_raw)
        
        # Handle lambda parameterization
        if self.use_tilde:
            print("Using tilde parameterization for lambdas (lambda_tilde, delta_lambda_tilde)")
            # Load or calculate tilde parameters
            if "lambda_tilde" in data and "delta_lambda_tilde" in data:
                train_lambda_1 = data["lambda_tilde"]
                train_lambda_2 = data["delta_lambda_tilde"]
            else:
                train_lambda_1 = lambda_1_lambda_2_to_lambda_tilde(lambda_1_raw, lambda_2_raw, m1_raw, m2_raw)
                train_lambda_2 = lambda_1_lambda_2_to_delta_lambda_tilde(lambda_1_raw, lambda_2_raw, m1_raw, m2_raw)
        else:
            print("Using component parameterization for lambdas (lambda_1, lambda_2)")
            train_lambda_1 = lambda_1_raw
            train_lambda_2 = lambda_2_raw
        
        # Handle luminosity distance if requested
        if self.include_dL:
            if "luminosity_distance" in data:
                self.train_dL = data["luminosity_distance"]
            else:
                raise ValueError("include_dL=True but no luminosity_distance found in training data")
        
        if self.source_type == "nsbh":
            # For NSBH: create concatenated arrays for single NS modeling
            print("NSBH mode: concatenating all neutron star data for single NS modeling")
            # Store original arrays for unconditional training
            self.train_mass_1 = train_mass_1
            self.train_mass_2_original = train_mass_2
            self.train_lambda_1 = train_lambda_1
            self.train_lambda_2_original = train_lambda_2
            # Create concatenated arrays for conditional training
            self.train_mass_2 = np.concatenate([train_mass_1, train_mass_2])
            self.train_lambda_2 = np.concatenate([train_lambda_1, train_lambda_2])
        else:
            # For BNS: use arrays as-is
            self.train_mass_1 = train_mass_1
            self.train_mass_2 = train_mass_2
            self.train_lambda_1 = train_lambda_1
            self.train_lambda_2 = train_lambda_2
        
        if self.take_log_lambda:
            # Take the log of the Lambda parameters
            print("Taking the log of the Lambda parameters")
            
            if self.source_type == "nsbh":
                self.train_lambda_2 = np.log(np.abs(self.train_lambda_2) + 1e-6)
            else:
                self.train_lambda_1 = np.log(np.abs(self.train_lambda_1) + 1e-6)
                self.train_lambda_2 = np.log(np.abs(self.train_lambda_2) + 1e-6)
        
    def train(self):
        """
        Just branching off to the unconditional or conditional training function, depending on the type of training we want to do.
        """
        
        # Load the data, from which we infer what system we are training for
        training_filename = os.path.join(self.outdir, "training_data.npz")
        self.load_training_data(training_filename)
        
        print("Sanity checking the ranges of the training data")
        
        if self.source_type == "nsbh":
            print(f"train_mass_2 (concatenated) ranges from {np.min(self.train_mass_2)} to {np.max(self.train_mass_2)}")
            print(f"train_lambda_2 (concatenated) ranges from {np.min(self.train_lambda_2)} to {np.max(self.train_lambda_2)}")
        else:
            print(f"train_mass_1 ranges from {np.min(self.train_mass_1)} to {np.max(self.train_mass_1)}")
            print(f"train_mass_2 ranges from {np.min(self.train_mass_2)} to {np.max(self.train_mass_2)}")
            print(f"train_lambda_1 ranges from {np.min(self.train_lambda_1)} to {np.max(self.train_lambda_1)}")
            print(f"train_lambda_2 ranges from {np.min(self.train_lambda_2)} to {np.max(self.train_lambda_2)}")
        
        # Start storing kwargs here to dump later on
        self.nf_kwargs = {"n_transforms": self.n_transforms,
                          "n_neurons": self.n_neurons,
                          "n_blocks_per_transform": self.n_blocks_per_transform,
                          "nn_depth": self.nn_depth,
                          "nn_block_dim": self.nn_block_dim
                          }
        
        # Build depending on source and conditional using training arrays
        if self.source_type == "bns" and self.conditional:
            print(f"We are training BNS and conditional")
            self.x = np.array([self.train_lambda_1, self.train_lambda_2]).T
            self.u = np.array([self.train_mass_1, self.train_mass_2]).T
            
            self.n_inputs = 2
            self.n_conditional_inputs = 2
            
            # Set names based on parameterization
            if self.use_tilde:
                self.nf_kwargs["names"] = ["lambda_tilde", "delta_lambda_tilde"]
            else:
                self.nf_kwargs["names"] = ["lambda_1", "lambda_2"]
            
            if self.use_component_masses:
                self.nf_kwargs["names_conditional"] = ["m_1", "m_2"]
            else:
                self.nf_kwargs["names_conditional"] = ["chirp_mass", "mass_ratio"]
            
        elif self.source_type == "bns" and not self.conditional:
            print(f"We are training BNS and not conditional")
            
            if self.include_dL:
                # 5D model with luminosity distance
                print("Including luminosity distance in 5D model")
                self.x = np.array([self.train_mass_1, self.train_mass_2, self.train_dL, self.train_lambda_1, self.train_lambda_2]).T
                self.n_inputs = 5
                
                # Set names based on parameterization
                mass_names = ["m_1", "m_2"] if self.use_component_masses else ["chirp_mass", "mass_ratio"]
                lambda_names = ["lambda_tilde", "delta_lambda_tilde"] if self.use_tilde else ["lambda_1", "lambda_2"]
                self.nf_kwargs["names"] = mass_names + ["luminosity_distance"] + lambda_names
            else:
                # 4D model
                self.x = np.array([self.train_mass_1, self.train_mass_2, self.train_lambda_1, self.train_lambda_2]).T
                self.n_inputs = 4
                
                # Set names based on parameterization
                mass_names = ["m_1", "m_2"] if self.use_component_masses else ["chirp_mass", "mass_ratio"]
                lambda_names = ["lambda_tilde", "delta_lambda_tilde"] if self.use_tilde else ["lambda_1", "lambda_2"]
                self.nf_kwargs["names"] = mass_names + lambda_names
            
            self.u = None
            self.n_conditional_inputs = None
            self.nf_kwargs["names_conditional"] = []
            
        elif self.source_type == "nsbh" and self.conditional:
            # Note: use_tilde=True with NSBH conditional is not supported (validation check above)
            print(f"We are training NSBH and conditional - using concatenated NS data")
            self.x = self.train_lambda_2.reshape(-1, 1)  # Shape: (2*N_samples, 1)
            self.u = self.train_mass_2.reshape(-1, 1)    # Shape: (2*N_samples, 1)
            
            self.n_inputs = 1
            self.n_conditional_inputs = 1
            
            self.nf_kwargs["names"] = ["lambda_2"]
            self.nf_kwargs["names_conditional"] = ["m_2"]
            
        elif self.source_type == "nsbh" and not self.conditional:
            print(f"We are training NSBH and not conditional")
            
            if not self.use_component_masses:
                # For chirp mass/mass ratio parameterization
                if self.include_dL:
                    # Use 4 parameters: Mc, q, dL, lambda_2 (only NS has lambda)
                    print("Using 4 parameters (Mc, q, dL, lambda_2) for NSBH unconditional training")
                    self.x = np.array([self.train_mass_1, self.train_mass_2_original, self.train_dL, self.train_lambda_2_original]).T
                    self.n_inputs = 4
                    
                    mass_names = ["chirp_mass", "mass_ratio"]
                    lambda_name = "lambda_tilde" if self.use_tilde else "lambda_2"
                    self.nf_kwargs["names"] = mass_names + ["luminosity_distance"] + [lambda_name]
                else:
                    # Use 3 parameters: Mc, q, lambda_2 (only NS has lambda)
                    print("Using 3 parameters (Mc, q, lambda_2) for NSBH unconditional training")
                    self.x = np.array([self.train_mass_1, self.train_mass_2_original, self.train_lambda_2_original]).T
                    self.n_inputs = 3
                    
                    mass_names = ["chirp_mass", "mass_ratio"]
                    lambda_name = "lambda_tilde" if self.use_tilde else "lambda_2"
                    self.nf_kwargs["names"] = mass_names + [lambda_name]
            else:
                # For component mass parameterization, use concatenated NS data
                print("Using concatenated NS data for NSBH unconditional training")
                self.x = np.array([self.train_mass_2, self.train_lambda_2]).T
                self.n_inputs = 2
                
                lambda_name = "lambda_tilde" if self.use_tilde else "lambda_2"
                self.nf_kwargs["names"] = ["m2", lambda_name]
            
            self.u = None
            self.n_conditional_inputs = None
            self.nf_kwargs["names_conditional"] = []
            
        else:
            raise ValueError(f"Something wrong. To check. Source type is {self.source_type} and conditional is {self.conditional}. ")
        
        # Save to kwargs for ease of use later on after training
        self.nf_kwargs["n_inputs"] = self.n_inputs
        self.nf_kwargs["n_conditional_inputs"] = self.n_conditional_inputs
        
        # Show some stuff
        print("np.shape(self.x)")
        print(np.shape(self.x))
        
        if self.u is not None:
            print("np.shape(self.u)")
            print(np.shape(self.u))
            
        if self.scale_input:
            print(f"Using MinMaxScaler to scale the input data x")
            scaler = MinMaxScaler()
            self.x = scaler.fit_transform(self.x)
            
            # Save the scaler
            scaler_savename = os.path.join(self.outdir, "scaler.gz")
            joblib.dump(scaler, scaler_savename)
            print(f"Saved sklearn scaler to {scaler_savename}")
        
        backend_name = "flowJAX" if self.use_flowjax else "glasflow"
        print(f"Going to start training for {backend_name} . . .")
        start_time = time.time()
        
        if self.use_flowjax:
            if self.conditional:
                flow = self._train_conditional_flowjax()
            else:
                flow = self._train_unconditional_flowjax()
        else:
            if self.conditional:
                flow = self._train_conditional_glasflow()
            else:
                flow = self._train_unconditional_glasflow()
                
        end_time = time.time()
        print(f"Training done. Took around {(end_time - start_time)/60:.2f} minutes")
        
        # Save the model
        if self.use_flowjax:
            save_path = os.path.join(self.outdir, "model.eqx")
            print(f"Saving the flowJAX model to {save_path}")
            eqx.tree_serialise_leaves(save_path, flow)
        else:
            save_path = os.path.join(self.outdir, "model.pt")
            print(f"Saving the model weights to {save_path}")
            torch.save(flow.state_dict(), save_path)
        
        # Save the model kwargs
        if self.use_flowjax:
            save_path = save_path.replace(".eqx", "_kwargs.json")
        else:
            save_path = save_path.replace(".pt", "_kwargs.json")
        
        # Just dump any last kwargs we want to save here before finally saving it
        self.nf_kwargs["source_type"] = self.source_type
        self.nf_kwargs["N_samples_training"] = self.N_samples_training
        self.nf_kwargs["m_max_BH"] = self.m_max_BH
        self.nf_kwargs["num_epochs"] = self.num_epochs
        self.nf_kwargs["learning_rate"] = self.learning_rate
        self.nf_kwargs["max_patience"] = self.max_patience
        self.nf_kwargs["batch_size"] = self.batch_size
        self.nf_kwargs["num_bins"] = self.num_bins
        self.nf_kwargs["eos_samples_filename"] = self.eos_samples_filename
        self.nf_kwargs["training_filename"] = self.training_filename
        self.nf_kwargs["take_log_lambda"] = str(self.take_log_lambda)
        self.nf_kwargs["use_flowjax"] = str(self.use_flowjax)
        self.nf_kwargs["use_tilde"] = str(self.use_tilde)
        self.nf_kwargs["use_component_masses"] = str(self.use_component_masses)
        self.nf_kwargs["include_dL"] = str(self.include_dL)
        
        print(f"Saving the model kwargs to {save_path}")
        with open(save_path, "w") as f:
            json.dump(self.nf_kwargs, f, indent=4)
            
        print("DONE training!")
        
    def _train_unconditional_glasflow(self) -> CouplingNSF:
        """
        Simple wrapper around a glasflow model to train an unconditional normalizing flow on the data.
        """
        
        # Initialize the flow model
        self.nf_kwargs["model_type"] = "CouplingNSF"
        
        # FIXME: num bins not passed
        flow = CouplingNSF(n_inputs=self.n_inputs,
                           n_transforms=self.n_transforms,
                           n_neurons=self.n_neurons,
                           n_blocks_per_transform=self.n_blocks_per_transform
        )
        flow.to(DEVICE)
        
        # Initialize early stopping parameters
        best_loss = np.inf
        early_stop_counter = 0
        
        # DataLoader for batching the data
        x_tensor = torch.tensor(self.x, dtype=torch.float32)
        dataset = TensorDataset(x_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Training loop with tqdm
        optimizer = optim.Adam(flow.parameters(), lr=self.learning_rate)
        train_loss = []
        
        for epoch in tqdm.tqdm(range(self.num_epochs), desc="Training", unit="epoch",):
            epoch_loss = 0.0
            flow.train()

            for (batch,) in dataloader:
                batch = batch.to(DEVICE)
                
                optimizer.zero_grad()
                loss = -flow.log_prob(inputs=batch).mean()
                loss.backward()
                optimizer.step()

                batch_loss = loss.item()
                epoch_loss += batch_loss

            epoch_loss /= len(dataloader)
            train_loss.append(epoch_loss)

            # Early stopping check
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                if early_stop_counter >= self.max_patience:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break

        self.plot_loss(np.array(train_loss))
        return flow
    
    def _train_conditional_glasflow(self):
        """
        Simple wrapper around a glasflow model to train a conditional normalizing flow on the data.
        Uses appropriate flow type based on dimensionality.
        """
        
        print(f"Initializing a glasflow model for conditional training with kwargs:")
        print(f"    n_inputs: {self.n_inputs}")
        print(f"    n_conditional_inputs: {self.n_conditional_inputs}")
        print(f"    n_transforms: {self.n_transforms}")
        print(f"    n_neurons: {self.n_neurons}")
        
        # Choose appropriate flow based on dimensionality
        if self.n_inputs == 1:
            # For 1D distributions (e.g., NSBH case), use autoregressive flow
            print("Using MaskedAffineAutoregressiveFlow for 1D conditional distribution")
            self.nf_kwargs["model_type"] = "MaskedAffineAutoregressiveFlow"
            flow = MaskedAffineAutoregressiveFlow(
                n_inputs=self.n_inputs,
                n_conditional_inputs=self.n_conditional_inputs,
                n_transforms=self.n_transforms,
                n_neurons=self.n_neurons,
                # n_blocks_per_transform=self.n_blocks_per_transform # FIXME: ignore it for now, to be consistent with BNS
            )
            
        else:
            # For >1D distributions (e.g., BNS case), use coupling flow
            print("Using RealNVP for >1D conditional distribution")
            self.nf_kwargs["model_type"] = "RealNVP"
            flow = RealNVP(
                n_inputs=self.n_inputs,
                n_transforms=self.n_transforms,
                n_conditional_inputs=self.n_conditional_inputs,
                n_neurons=self.n_neurons,
                batch_norm_between_transforms=True,
            )
        
        flow.to(DEVICE)
        print(f"Initialized flow: {type(flow).__name__}")
        
        # Initialize early stopping parameters
        best_loss = np.inf
        early_stop_counter = 0
        
        # DataLoader for batching the data
        x_tensor = torch.tensor(self.x, dtype=torch.float32)
        u_tensor = torch.tensor(self.u, dtype=torch.float32)
        dataset = TensorDataset(x_tensor, u_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Training loop with tqdm
        optimizer = optim.Adam(flow.parameters(), lr=self.learning_rate)
        train_loss = []
        
        # TODO: have to use validation data and early stopping with that!
        for epoch in tqdm.tqdm(range(self.num_epochs), desc="Training", unit="epoch"):
            flow.train()
            epoch_loss = 0.0

            for batch in dataloader:
                x, u = batch
                x = x.to(DEVICE)
                u = u.to(DEVICE)
        
                optimizer.zero_grad()
                loss = -flow.log_prob(x, conditional=u).mean()
                
                loss.backward()
                optimizer.step()

                batch_loss = loss.item()
                epoch_loss += batch_loss

            epoch_loss /= len(dataloader)
            train_loss.append(epoch_loss)

            # Early stopping check
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                if early_stop_counter >= self.max_patience:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break

        self.plot_loss(np.array(train_loss))
        return flow
    
    def _train_unconditional_flowjax(self):
        """
        Train an unconditional normalizing flow using flowJAX.
        """
        
        raise NotImplementedError("Unconditional flowJAX training not implemented yet.")
    
    # TODO: clean up this code and train it
    #     # Create base distribution
    #     base_dist = Normal(jnp.zeros(self.n_inputs))
        
    #     # Initialize flow
    #     key = jr.key(42)
    #     if self.n_inputs == 1:
    #         # For 1D, use masked autoregressive flow
    #         print("Using masked_autoregressive_flow for unconditional 1D distribution")
    #         self.nf_kwargs["model_type"] = "masked_autoregressive_flow"
    #         flow = masked_autoregressive_flow(
    #             key=key,
    #             base_dist=base_dist,
    #             flow_layers=self.n_transforms,
    #             nn_width=self.n_neurons,
    #             nn_depth=self.n_blocks_per_transform
    #         )
    #     else:
    #         # For >1D, use coupling flow
    #         print("Using coupling_flow for unconditional >1D distribution")
    #         self.nf_kwargs["model_type"] = "coupling_flow"
    #         flow = coupling_flow(
    #             key=key,
    #             base_dist=base_dist,
    #             flow_layers=self.n_transforms,
    #             nn_width=self.n_neurons,
    #             nn_depth=self.n_blocks_per_transform
    #         )
        
    #     # Train the flow
    #     train_key = jr.key(123)
    #     x_data = jnp.array(self.x, dtype=jnp.float32)
        
    #     print(f"Training flowJAX unconditional model on data shape: {x_data.shape}")
    #     flow, losses = fit_to_data(
    #         key=train_key,
    #         dist=flow,
    #         data=x_data,
    #         learning_rate=self.learning_rate,
    #         max_epochs=self.num_epochs,
    #         batch_size=self.batch_size,
    #         patience=self.max_patience
    #     )
        
    #     self.plot_loss(np.array(losses))
    #     return flow
    
    def _train_conditional_flowjax(self):
        """
        Train a conditional normalizing flow using flowJAX.
        """
        # Create base distribution
        base_dist = Normal(jnp.zeros(self.n_inputs))
        
        # Initialize flow
        key = jr.key(42)
        if self.n_inputs == 1:
            raise NotImplementedError("Conditional 1D flowJAX training not implemented yet.")
            
            # TODO: clean up
            # # For 1D conditional, use masked autoregressive flow
            # print("Using masked_autoregressive_flow for conditional 1D distribution")
            # self.nf_kwargs["model_type"] = "masked_autoregressive_flow"
            # flow = masked_autoregressive_flow(
            #     key=key,
            #     base_dist=base_dist,
            #     cond_dim=self.n_conditional_inputs,
            #     flow_layers=self.n_transforms,
            #     nn_width=self.n_neurons,
            #     nn_depth=self.n_blocks_per_transform
            # )
        else:
            # For >1D conditional, use coupling flow
            print("Using block_neural_autoregressive_flow for conditional >1D distribution")
            self.nf_kwargs["model_type"] = "block_neural_autoregressive_flow"
            flow = block_neural_autoregressive_flow(
                key=key,
                base_dist=base_dist,
                cond_dim=self.n_conditional_inputs,
                # flow_layers=self.n_transforms, # TODO: perhaps tune this, for now, use default value
                nn_depth=self.nn_depth,
                nn_block_dim=self.nn_block_dim
            )
        
        # Train the flow with conditional data
        train_key = jr.key(123)
        x_data = jnp.array(self.x, dtype=jnp.float32)
        u_data = jnp.array(self.u, dtype=jnp.float32)
        
        print(f"Training flowJAX conditional model:")
        print(f"  x_data shape: {x_data.shape}")
        print(f"  u_data shape: {u_data.shape}")
        
        # Create combined dataset
        combined_data = (x_data, u_data)

        flow, losses = fit_to_data(
            train_key,
            flow,
            data=combined_data,
            learning_rate=self.learning_rate,
            max_epochs=self.num_epochs,
            max_patience=self.max_patience,
            batch_size=self.batch_size,
        )
        
        # TODO: use the losses for diagnosis later on!
        
        return flow
    
    def plot_loss(self, loss: np.array) -> None:
        
        # Make a plot of the loss trajectory
        plt.figure(figsize=(12, 6))
        plt.plot(loss, label="Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss Trajectory")
        save_path = os.path.join(self.outdir, "training_loss.pdf")
        print(f"Saving the training loss plot to {save_path}")
        if all(loss > 0.0):
            plt.yscale("log")
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
            
def main():
    args = parser.parse_args()
    
    print(f"Starting training with the following parameters:")
    for key, value in vars(args).items():
        print(f"    - {key}: {value}")
    trainer = NFPriorCreator(**vars(args))
    trainer.create_data()
    trainer.train()
    
if __name__ == "__main__":
    main()