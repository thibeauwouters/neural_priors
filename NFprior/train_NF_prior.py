"""
Train a normalizing flow to approximate a distribution on masses and Lambdas to replicate an EOS dataset and be used in inference.
This might be joint prior or conditional prior, we will have to check which works best later on. 

TODO:
- support for different neutron star populations -- later on
"""

import os
import argparse
import numpy as np
import json

### bilby imports
from bilby.core.prior.analytical import Uniform
from bilby.gw.prior import UniformComovingVolume
from bilby.core.prior import PriorDict
from bilby.gw.conversion import (
    luminosity_distance_to_redshift,
    chirp_mass_and_mass_ratio_to_component_masses,
    component_masses_to_chirp_mass,
    component_masses_to_mass_ratio,
    lambda_1_lambda_2_to_lambda_tilde,
    lambda_1_lambda_2_to_delta_lambda_tilde
)

### glasflow imports
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
jax_devices = jax.devices()
print(f"JAX: devices available: {jax_devices}")
import equinox as eqx

import tqdm
import time
import joblib
import subprocess

import matplotlib.pyplot as plt
params = {"axes.grid": True,
          "text.usetex" : False}
plt.rcParams.update(params)

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

if torch.cuda.is_available():
    print(f"torch: CUDA is available. Number of devices: {torch.cuda.device_count()}")
    print(f"torch: Current CUDA device: {torch.cuda.current_device()}")
else:
    print("torch: CUDA is not available.")

# Get the device as well:
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

script_dir = os.path.dirname(os.path.abspath(__file__))
EOS_DIR = os.path.join(script_dir, "..", "data", "eos")


def sample_ns_mass_gaussian(nb_mass_samples: int):
    """
    Sample from single Gaussian distribution, found the hyperparams in https://arxiv.org/pdf/2407.16669
    """
    mu = 1.33
    sigma = 0.09
    mass_samples = np.random.normal(mu, sigma, nb_mass_samples)
    
    if len(mass_samples) == 1:
        return mass_samples[0]
    else:
        return mass_samples
    
def sample_ns_mass_double_gaussian(nb_mass_samples: int):
    """
    Sample from double Gaussian, found the hyperparams in https://arxiv.org/pdf/2407.16669
    """
    mu_1 = 1.34
    sigma_1 = 0.07
    
    mu_2 = 1.80
    sigma_2 = 0.21
    w = 0.65
    
    # Sample from mixture of gaussians
    u = np.random.rand(nb_mass_samples) # uniform [0,1], to determine the mode
    mass_samples = np.where(
        u < w,
        np.random.normal(mu_1, sigma_1, size=nb_mass_samples),
        np.random.normal(mu_2, sigma_2, size=nb_mass_samples)
    )
    
    if len(mass_samples) == 1:
        return mass_samples[0]
    else:
        return mass_samples

parser = argparse.ArgumentParser(description="Train a normalizing flow prior on EOS samples.")
parser.add_argument("--population-type", 
                    type=str, 
                    default="uniform", 
                    choices=["uniform", "gaussian", "double_gaussian", "GW170817", "GW190425", "GW230529"], 
                    help="Type of source to model")
parser.add_argument("--source-type", 
                    type=str, 
                    default="bns", 
                    choices=["bns", "nsbh"],
                    help="Type of source to model")
parser.add_argument("--eos-samples-name", 
                    type=str, 
                    default="radio", 
                    choices=["radio", "radio_chiEFT", "radio_chiEFT_NICER", ],
                    help="EOS samples name (default: radio)")
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
parser.add_argument("--N-samples-training", 
                    type=int, 
                    default=200_000, 
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
                    default=1_000,
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
                    default=100, 
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
                    default=1, 
                    help="Depth of flowJAX network")
parser.add_argument("--nn-block-dim", 
                    type=int, 
                    default=8, 
                    help="Block dimension of flowJAX network")
parser.add_argument("--flow-layers", 
                    type=int, 
                    default=1, 
                    help="Number of flow layers (for flowJAX)")
parser.add_argument("--validation-split-fraction", 
                    type=float, 
                    default=0.2, 
                    help="Fraction of data to use for validation (default: 0.2)")
parser.add_argument("--setup-submission", 
                    action="store_true", 
                    help="Setup .sub file for cluster submission instead of training")
parser.add_argument("--submit", 
                    action="store_true", 
                    help="Setup .sub file and submit the job to cluster")

    
class NFPriorCreator:
    """
    Class to construct the NF prior and train it.
    """
    
    def __init__(self,
                 population_type: str = "uniform",
                 eos_samples_name: str = "radio",
                 source_type: str = "bns",
                 use_tilde: bool = False,
                 use_component_masses: bool = True,
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
                 nn_depth: int = 1,
                 nn_block_dim: int = 8,
                 flow_layers: int = 1,
                 validation_split_fraction: float = 0.2
                 ):
        """
        Initialize the NFPriorCreator class with the necessary parameters.

        Args:
            eos_samples_name (str, optional): Name of the run from which we load the EOS samples from, which will be converted into the training data for the NF for binary systems. Defaults to `radio`, which only uses the radio timing constraints on MTOV.
            source_type (str, optional): Which kind of source to model: `bns` or `nsbh`. Defaults to "bns".
            N_samples_training (int, optional): Number of training samples to create.. Defaults to 100_000.
            N_samples_plot (int, optional): Number of samples to create the plots. Defaults to 10_000.
            m_max_BH (float, optional): If generating NSBH training data with an NF that is not conditioned on the masses, this is up to which the masses are taken. Defaults to 5.0.
            save_name (str, optional): Where to save the models etc to. Defaults to "".
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
        self.eos_samples_filename = os.path.abspath(os.path.join(EOS_DIR, f"{self.eos_samples_name}/eos_samples.npz"))
        
        if not os.path.exists(self.eos_samples_filename):
            # Show all subdirs in 
            available_subdirs = os.listdir(EOS_DIR)
            raise ValueError(f"File {self.eos_samples_filename} does not exist. Please check the path or the `eos_samples_name` argument. Available subdirs: {available_subdirs}.")
        print(f"Training data will be loaded from: {self.eos_samples_filename}")
        
        SUPPORTED_SOURCE_TYPES = ["bns", "nsbh"]
        if source_type not in SUPPORTED_SOURCE_TYPES:
            raise ValueError(f"source_type must be one of {SUPPORTED_SOURCE_TYPES}, got {source_type} instead.")
        
        SUPPORTED_POPULATION_TYPES = ["uniform", "gaussian", "double_gaussian", "GW170817", "GW190425", "GW230529"]
        if population_type not in SUPPORTED_POPULATION_TYPES:
            raise ValueError(f"population_type must be one of {SUPPORTED_POPULATION_TYPES}, got {population_type} instead.")
        
        self.source_type = source_type
        self.population_type = population_type
        
        print(f"Training a normalizing flow for population =  {self.population_type} and {self.source_type} sources")
        self.N_samples_training = N_samples_training
        self.N_samples_plot = N_samples_plot
        self.m_max_BH = m_max_BH
        self.take_log_lambda = take_log_lambda
        self.use_flowjax = use_flowjax
        self.use_tilde = use_tilde
        self.use_component_masses = use_component_masses

        self.num_epochs = num_epochs
        if learning_rate >= 1e-3 and self.use_flowjax:
            learning_rate = 5e-4
            print(f"Using a smaller learning rate {learning_rate} for flowJAX training, as it is more stable with this value.")
        self.learning_rate = learning_rate
        self.max_patience = max_patience
        self.n_transforms = n_transforms
        self.n_neurons = n_neurons
        self.nn_depth = nn_depth
        self.flow_layers = flow_layers
        self.nn_block_dim = nn_block_dim
        self.batch_size = batch_size
        self.n_blocks_per_transform = n_blocks_per_transform
        self.scale_input = scale_input
        self.num_bins = num_bins
        self.validation_split_fraction = validation_split_fraction
        
        # Store the NF kwargs here to dump later on
        self.nf_kwargs = {"n_transforms": self.n_transforms,
                          "n_neurons": self.n_neurons,
                          "n_blocks_per_transform": self.n_blocks_per_transform,
                          "nn_depth": self.nn_depth,
                          "flow_layers": self.flow_layers,
                          "nn_block_dim": self.nn_block_dim
                          }
        
        # Set whether this is specific for a GW event
        if "GW" in self.population_type:
            self.is_gw_event = True
        else:
            self.is_gw_event = False
        
        # Set names based on parameterization
        if self.use_component_masses:
            mass_names = ["m_1", "m_2"]
        else:
            if self.is_gw_event:
                mass_names = ["chirp_mass", "mass_ratio"]
            else:
                mass_names = ["chirp_mass_source", "mass_ratio"]
        
        if self.source_type == "nsbh" and not self.use_tilde:
            lambda_names = ["lambda_2"]
        else:
            lambda_names = ["lambda_tilde", "delta_lambda_tilde"] if self.use_tilde else ["lambda_1", "lambda_2"]
            
        all_names = mass_names + lambda_names
        if self.is_gw_event:
            all_names += ["luminosity_distance"]
        self.nf_kwargs["names"] = all_names
        self.nf_kwargs["n_inputs"] = len(self.nf_kwargs["names"])
        
        print(f"Before training, we built the following NF kwargs: {self.nf_kwargs}")
        
        if self.scale_input and self.take_log_lambda:
            raise ValueError("Cannot scale input and take log of Lambdas at the same time. \
                Please set either scale_input=False or take_log_lambda=False.\
                Recommended to use scaling and not log transformation for training (empirically gave best results).")
        
        # Make an outdir based on the given name etc, so that everything is stored in one directory for later on
        backend_suffix = "_flowjax" if self.use_flowjax else ""
        self.outdir = os.path.join("./models/", self.population_type, self.source_type, f"{self.eos_samples_name}{backend_suffix}")
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)
            print(f"Created output directory {self.outdir}")
        print(f"Everything for this model will be saved to {self.outdir}")
        
        # FIXME: this is suboptimal, but turns out flowJAX has its own support for train-validation split! Use the default there
        # NOTE: I still have to figure out a nice modular way to do logic branching for the train-validation split step if flowJAX is/is not used, but for now, a hacky way is to throw away part of the data to avoid errors and use the "train data" as all data in case flowJAX is used. Need to update this later on.
        
        if self.use_flowjax:
            self.validation_split_fraction = 0.01
            print(f"Using flowJAX, so setting validation_split_fraction to {self.validation_split_fraction} (no validation split)")
    
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
            bad_MTOV = np.max(masses_EOS[i]) < 2.0
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
        Create the dataset from the EOS to neutron star systems: m1, m2, Lambda1, Lambda2.
        Masses are sampled uniformly between 1.0 and MTOV for each EOS sample, therefore, such a prior can be used for any GW event.
        Always generates NS-NS pairs - the BNS vs NSBH distinction is handled in load_training_data().
        """
        
        # TODO: at some point, use self.population_type to change the way we generate the training data
        
        # Load the EOS samples
        masses_EOS, _, Lambdas_EOS = self.load_eos_samples_from_file()
        
        if self.is_gw_event:
            print(f"create_data is following GW_event source type: {self.population_type}")
            
            # Read the prior file and process the first three lines for the masses
            prior_filename = f"../GW_runs/{self.population_type}/prior.prior"
            Mc_det_list = np.empty(self.N_samples_training)
            dL_list = np.empty(self.N_samples_training)
                
            # Read the first three lines of the prior file to get the priors
            with open(prior_filename, "r") as f:
                lines = f.readlines()
                Mc_prior = lines[0].strip()
                q_prior = lines[1].strip()
                dL_prior = lines[2].strip()
                
            # For each line, extract the minimum and maximum values
            def parse_prior_line(line):
                """Extract minimum and maximum values from a prior definition line."""
                import re
                # Pattern to match minimum=value and maximum=value
                min_match = re.search(r'minimum=([^,\)]+)', line)
                max_match = re.search(r'maximum=([^,\)]+)', line)
                
                min_val = None
                max_val = None
                
                if min_match:
                    min_str = min_match.group(1).strip()
                    try:
                        min_val = float(min_str)
                    except ValueError:
                        # Handle expressions like "1369419318.7460938-0.1"
                        try:
                            min_val = eval(min_str)
                        except:
                            min_val = None
                
                if max_match:
                    max_str = max_match.group(1).strip()
                    try:
                        max_val = float(max_str)
                    except ValueError:
                        # Handle expressions like "1369419318.7460938+0.1"
                        try:
                            max_val = eval(max_str)
                        except:
                            max_val = None
                
                return min_val, max_val
            
            # Parse the prior bounds
            Mc_min, Mc_max = parse_prior_line(Mc_prior)
            q_min, q_max = parse_prior_line(q_prior)
            dL_min, dL_max = parse_prior_line(dL_prior)
            
            print(f"Parsed prior bounds:")
            print(f"  Chirp mass: {Mc_min} - {Mc_max}")
            print(f"  Mass ratio: {q_min} - {q_max}")
            print(f"  Luminosity distance: {dL_min} - {dL_max}")
            
            # Create the bilby priors
            priors_dict = {}
            priors_dict["chirp_mass"] = Uniform(minimum=Mc_min, maximum=Mc_max, name="chirp_mass")
            priors_dict["mass_ratio"] = Uniform(minimum=q_min, maximum=q_max, name="mass_ratio")
            priors_dict["luminosity_distance"] = UniformComovingVolume(minimum=dL_min, maximum=dL_max, name='luminosity_distance', latex_label='$D_L$')
            
            gw_priors = PriorDict(priors_dict)
            print(f"Loaded priors: {gw_priors}")

        # Make everything ready for sampling
        m1_list = np.empty(self.N_samples_training)
        m2_list = np.empty(self.N_samples_training)
        Lambda1_list = np.empty(self.N_samples_training)
        Lambda2_list = np.empty(self.N_samples_training)

        # Construct the prior from sampling from the EOS set - always generate NS-NS pairs
        for i in range(self.N_samples_training):
            if i % (self.N_samples_training // 10) == 0:
                print(f"{i}/{self.N_samples_training}")
            idx = np.random.randint(0, len(masses_EOS))
            m, l = masses_EOS[idx], Lambdas_EOS[idx]
            mtov = np.max(m)
            
            if not self.is_gw_event:
                if self.source_type == "bns":
                    if self.population_type == "uniform":
                        # Sample two masses uniformly between 1.0 and MTOV, and ensure m1 >= m2
                        mass_samples = np.random.uniform(1.0, mtov, 2)
                    elif self.population_type == "gaussian":
                        mass_samples = sample_ns_mass_gaussian(2)
                    elif self.population_type == "double_gaussian":
                        mass_samples = sample_ns_mass_double_gaussian(2)
                    else:
                        raise ValueError(f"Unsupported population type: {self.population_type}")
                        
                    # Ensure that m1 >= m2
                    m1 = np.max(mass_samples)
                    m2 = np.min(mass_samples)
                    
                    Lambda_1 = np.interp(m1, m, l)
                    Lambda_2 = np.interp(m2, m, l)
                
                elif self.population_type == "nsbh":
                    # This automatically satisfies m1 >= m2
                    m1 = np.random.uniform(mtov, self.m_max_BH, 1)[0]
                    
                    if self.population_type == "uniform":
                        # Sample two masses uniformly between 1.0 and MTOV, and ensure m1 >= m2
                        m2 = np.random.uniform(1.0, mtov, 1)
                    elif self.population_type == "gaussian":
                        m2 = sample_ns_mass_gaussian(1)
                    elif self.population_type == "double_gaussian":
                        m2 = sample_ns_mass_double_gaussian(1)
                    else:
                        raise ValueError(f"Unsupported population type: {self.population_type}")
                    
                    Lambda_1 = 0.0
                    Lambda_2 = np.interp(m2, m, l)
                
            else:
                # For GW events, we sample from the given bilby priors
                samples = gw_priors.sample(size = 1)
                
                Mc, q, dL = samples["chirp_mass"][0], samples["mass_ratio"][0], samples["luminosity_distance"][0]
                Mc_det_list[i] = Mc
                dL_list[i] = dL
                
                # Convert to source frame component masses
                m1_det, m2_det = chirp_mass_and_mass_ratio_to_component_masses(Mc, q)
                z = luminosity_distance_to_redshift(dL)
                m1 = m1_det / (1 + z)
                m2 = m2_det / (1 + z)
                
                # Get Lambda values by interpolation
                Lambda_1 = np.interp(m1, m, l)
                Lambda_2 = np.interp(m2, m, l)
                    
            # Save the sampled values
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
        
        # Also get chirp mass and mass ratio for the training data
        chirp_mass_source = component_masses_to_chirp_mass(m1_list, m2_list)
        # mass_ratio = component_masses_to_mass_ratio(m1_list, m2_list) # TODO: remove for now?
        
        save_dict = {
            "m1": np.array(m1_list),
            "m2": np.array(m2_list),
            "mass_ratio": np.array(m2_list)/np.array(m1_list),
            "lambda_1": np.array(Lambda1_list),
            "lambda_2": np.array(Lambda2_list),
            "lambda_tilde": np.array(lambda_tilde),
            "delta_lambda_tilde": np.array(delta_lambda_tilde),
            "chirp_mass_source": np.array(chirp_mass_source),
        }
        
        if self.is_gw_event:
            save_dict["chirp_mass"] = Mc_det_list
            save_dict["luminosity_distance"] = dL_list
            
        print(f"Create data will save the following data:")
        for key, value in save_dict.items():
            print(f"  {key}: range = [{np.min(value)}, {np.max(value)}]")
        
        full_save_path = os.path.join(self.outdir, f"training_data.npz")
        self.training_filename = full_save_path
        print(f"Saving to {full_save_path}:")
        np.savez(full_save_path, **save_dict)
        print(f"Saving to {full_save_path} DONE")
        
    def load_training_data(self, training_filename: str):
        """
        Loads in the preprocessed training data before feeding it into the NF for training.
        Creates parameterization-agnostic training arrays that the training methods can use.
        
        The training methods use generic arrays:
        - train_mass_1, train_mass_2: Can be (m1, m2) or (Mc, q) depending on use_component_masses
        - train_lambda_1, train_lambda_2: Can be (lambda_1, lambda_2) or (lambda_tilde, delta_lambda_tilde) depending on use_tilde
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
            if self.is_gw_event:
                print("Using detector-frame chirp mass from the GW event and mass ratio (Mc_det, q) for training")
                train_mass_1 = data["chirp_mass"]
                train_mass_2 = data["mass_ratio"]
            else:
                print("Using source-frame chirp mass and mass ratio (Mc, q) for training")
                train_mass_1 = data["chirp_mass_source"]
                train_mass_2 = data["mass_ratio"]
        
        # Handle lambda parameterization
        if self.use_tilde:
            print("Using tilde parameterization for lambdas (lambda_tilde, delta_lambda_tilde)")
            # Load or calculate tilde parameters
            train_lambda_1 = data["lambda_tilde"]
            train_lambda_2 = data["delta_lambda_tilde"]
        else:
            print("Using component parameterization for lambdas (lambda_1, lambda_2)")
            train_lambda_1 = lambda_1_raw
            train_lambda_2 = lambda_2_raw
        
        # Create train-validation split using sklearn
        arrays_to_split = [train_mass_1, train_mass_2, train_lambda_1, train_lambda_2]
        if self.is_gw_event:
            # If we have GW event data, also include the luminosity distance
            arrays_to_split.append(data["luminosity_distance"])
        split_result = train_test_split(*arrays_to_split,
                                        test_size=self.validation_split_fraction,
                                        train_size=1.0-self.validation_split_fraction,
                                        random_state=42)
        
        # Unpack results
        self.train_mass_1, self.val_mass_1 = split_result[0], split_result[1]
        self.train_mass_2, self.val_mass_2 = split_result[2], split_result[3]
        self.train_lambda_1, self.val_lambda_1 = split_result[4], split_result[5]
        self.train_lambda_2, self.val_lambda_2 = split_result[6], split_result[7]
        if self.is_gw_event:
            self.train_dL, self.val_dL = split_result[8], split_result[9]
        
        print(f"Training samples: {len(self.train_mass_1)}, Validation samples: {len(self.val_mass_1)}")
        
        if self.take_log_lambda:
            # Take the log of the Lambda parameters
            print("Taking the log of the Lambda parameters")
            
            if self.source_type == "nsbh":
                self.train_lambda_2 = np.log(np.abs(self.train_lambda_2) + 1e-6)
                self.val_lambda_2 = np.log(np.abs(self.val_lambda_2) + 1e-6)
            else:
                self.train_lambda_1 = np.log(np.abs(self.train_lambda_1) + 1e-6)
                self.train_lambda_2 = np.log(np.abs(self.train_lambda_2) + 1e-6)
                self.val_lambda_1 = np.log(np.abs(self.val_lambda_1) + 1e-6)
                self.val_lambda_2 = np.log(np.abs(self.val_lambda_2) + 1e-6)
        
    def train(self):
        """
        Function to build the training data based on some specifications, and then branch off to the desired subfunction for the actual training.
        """
        # Load the data, from which we infer what system we are training for
        training_filename = os.path.join(self.outdir, "training_data.npz")
        self.load_training_data(training_filename)
        
        print("Creating the training data arrays")
        if self.source_type == "nsbh" and not self.use_tilde:
            print("NSBH and training on lambda_2 (3D)")
            x = [self.train_mass_1, self.train_mass_2, self.train_lambda_2]
            x_val = [self.val_mass_1, self.val_mass_2, self.val_lambda_2]
        else:
            # This is always a 4D model
            print(f"4D training data model for {self.source_type}")
            x = [self.train_mass_1, self.train_mass_2, self.train_lambda_1, self.train_lambda_2]
            x_val = [self.val_mass_1, self.val_mass_2, self.val_lambda_1, self.val_lambda_2]
                
        if self.is_gw_event:
            # If we have GW event data, also include the detector-frame chirp mass and luminosity distance
            x.append(self.train_dL)
            x_val.append(self.val_dL)
            print("Including luminosity distance in training data")
                
        self.x = np.array(x).T
        self.x_val = np.array(x_val).T
                
        # Show some stuff
        print("np.shape(self.x)")
        print(np.shape(self.x))
            
        if self.scale_input:
            print(f"Using MinMaxScaler to scale the input data x")
            scaler = MinMaxScaler()
            self.x = scaler.fit_transform(self.x)
            self.x_val = scaler.transform(self.x_val)  # Apply same scaling to validation
            
            # Save the scaler to a file we can unpickle later on
            scaler_savename = os.path.join(self.outdir, "scaler.gz")
            joblib.dump(scaler, scaler_savename)
            print(f"Saved sklearn scaler to {scaler_savename}")
        
        backend_name = "flowJAX" if self.use_flowjax else "glasflow"
        print(f"Going to start training for {backend_name} . . .")
        start_time = time.time()
        
        if self.use_flowjax:
            flow = self._train_flowjax()
        else:
            flow = self._train_glasflow()
                
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
        
        print(f"Saving the model kwargs to {save_path}")
        with open(save_path, "w") as f:
            json.dump(self.nf_kwargs, f, indent=4)
        print("DONE training!")
        
    def _train_glasflow(self) -> CouplingNSF:
        """
        Simple wrapper around a glasflow model to train an unconditional normalizing flow on the data.
        """
        
        # Initialize the flow model
        self.nf_kwargs["model_type"] = "CouplingNSF"
        
        flow = CouplingNSF(n_inputs=self.nf_kwargs["n_inputs"],
                           n_transforms=self.n_transforms,
                           n_neurons=self.n_neurons,
                           n_blocks_per_transform=self.n_blocks_per_transform,
                           num_bins=self.num_bins
        )
        flow.to(DEVICE)
        
        # Initialize early stopping parameters
        best_val_loss = np.inf
        early_stop_counter = 0
        
        # DataLoader for batching the data
        x_tensor = torch.tensor(self.x, dtype=torch.float32)
        x_val_tensor = torch.tensor(self.x_val, dtype=torch.float32)
        dataset = TensorDataset(x_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Training loop with tqdm
        optimizer = optim.Adam(flow.parameters(), lr=self.learning_rate)
        train_losses = []
        val_losses = []
        
        for epoch in tqdm.tqdm(range(self.num_epochs), desc="Training", unit="epoch"):
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
            train_losses.append(epoch_loss)
            
            # Validation loss calculation
            flow.eval()
            with torch.no_grad():
                x_val_batch = x_val_tensor.to(DEVICE)
                val_loss = -flow.log_prob(inputs=x_val_batch).mean().item()
                val_losses.append(val_loss)

            # Early stopping check using validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stop_counter = 0
                # Save best model
                best_model_state = flow.state_dict().copy()
            else:
                early_stop_counter += 1
                if early_stop_counter >= self.max_patience:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break

        # Load best model
        flow.load_state_dict(best_model_state)
        self.plot_loss(np.array(train_losses), np.array(val_losses))
        return flow
    
    def _train_flowjax(self):
        """
        Train an unconditional normalizing flow using flowJAX.
        """
        # Create base distribution
        base_dist = Normal(jnp.zeros(self.nf_kwargs["n_inputs"]))
        
        # Initialize flow
        key = jr.key(42)
        # TODO: decide whether we want to be flexible here and restore this piece of code, or rather deprecate coupling and instead hard-code BNAF?
        # if self.nf_kwargs["n_inputs"] == 1:
        #     # For 1D, use masked autoregressive flow
        #     print("Using masked_autoregressive_flow for unconditional 1D distribution")
        #     self.nf_kwargs["model_type"] = "masked_autoregressive_flow"
        #     flow = masked_autoregressive_flow(
        #         key=key,
        #         base_dist=base_dist,
        #         flow_layers=self.n_transforms,
        #         nn_width=self.n_neurons,
        #         nn_depth=self.nn_depth
        #     )
        print("Using block_neural_autoregressive_flow for flowJAX unconditional distribution")
        self.nf_kwargs["model_type"] = "block_neural_autoregressive_flow"
        flow = block_neural_autoregressive_flow(
            key=key,
            base_dist=base_dist,
            nn_depth=self.nn_depth,
            nn_block_dim=self.nn_block_dim,
            flow_layers=self.flow_layers,
        )
        
        # Train the flow with validation data
        train_key = jr.key(123)
        x_train_data = jnp.array(self.x, dtype=jnp.float32)
        x_val_data = jnp.array(self.x_val, dtype=jnp.float32)
        
        print(f"Training flowJAX unconditional model on data shape: {x_train_data.shape}")
        print(f"Validation data shape: {x_val_data.shape}")
        
        flow, losses = fit_to_data(
            key=train_key,
            dist=flow,
            data=x_train_data,
            # x_val=x_val_data, # TODO: figure out
            learning_rate=self.learning_rate,
            max_epochs=self.num_epochs,
            batch_size=self.batch_size,
            max_patience=self.max_patience,
            val_prop=self.validation_split_fraction
        )
        
        self.plot_loss(np.array(losses["train"]), np.array(losses["val"]))
        return flow
    
    def plot_loss(self, train_loss: np.array, val_loss: np.array = None) -> None:
        
        # Make a plot of the loss trajectory
        plt.figure(figsize=(12, 6))
        plt.plot(train_loss, label="Training Loss")
        if val_loss is not None:
            plt.plot(val_loss, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss Trajectory")
        plt.legend()
        save_path = os.path.join(self.outdir, "training_loss.pdf")
        print(f"Saving the training loss plot to {save_path}")
        if all(train_loss > 0.0) and (val_loss is None or all(val_loss > 0.0)):
            plt.yscale("log")
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()


    def setup_submission(self, args, submit=False):
        """
        Setup .sub file for cluster submission based on current arguments.
        
        Args:
            args: Parsed command line arguments
            submit: Whether to submit the job after creating .sub file
        """
        
        # Read the template file
        template_path = "train.sub"
        if not os.path.exists(template_path):
            raise FileNotFoundError(f"Template file {template_path} not found")
        
        with open(template_path, 'r') as f:
            template_content = f.read()
        
        # Get the absolute path to this training script
        train_script_path = os.path.abspath(__file__)
        
        # FIXME: is this still necessary given main()?
        # Build the training arguments string from current args, excluding submission-specific ones
        training_args = []
        exclude_args = {'setup_submission', 'template_file', 'submit'}
        
        for key, value in args.items():
            if key in exclude_args:
                continue
                
            arg_name = f"--{key.replace('_', '-')}"
            
            # Handle boolean flags
            if isinstance(value, bool):
                if value:
                    training_args.append(arg_name)
                else:
                    # For boolean args with "no-" variants, add the negative form if False
                    if key == 'use_tilde' and not value:
                        training_args.append("--no-use-tilde")
                    elif key == 'use_component_masses' and not value:
                        training_args.append("--no-use-component-masses")
                    elif key == 'take_log_lambda' and not value:
                        training_args.append("--no-take-log-lambda")
                    elif key == 'use_flowjax' and not value:
                        training_args.append("--no-use-flowjax")
                    elif key == 'scale_input' and not value:
                        training_args.append("--no-scale-input")
            else:
                # Handle non-boolean arguments
                training_args.extend([arg_name, str(value)])
        
        training_args_str = " ".join(training_args)
        
        # Modify the template content
        lines = template_content.strip().split('\n')
        modified_lines = []
        
        for line in lines:
            if line.startswith('arguments ='):
                # Replace the arguments line with our specific arguments
                modified_lines.append(f'arguments = "{train_script_path} {training_args_str}"')
            elif line.startswith('Log ='):
                # Update log file path
                log_path = os.path.join(self.outdir, "log.log")
                modified_lines.append(f'Log = {log_path}')
            elif line.startswith('Error ='):
                # Update error file path
                err_path = os.path.join(self.outdir, "err.err")
                modified_lines.append(f'Error = {err_path}')
            elif line.startswith('Output ='):
                # Update output file path
                out_path = os.path.join(self.outdir, "out.out")
                modified_lines.append(f'Output = {out_path}')
            else:
                # Keep the line as is
                modified_lines.append(line)
        
        # Write the modified .sub file
        sub_file_path = os.path.join(self.outdir, "train.sub")
        with open(sub_file_path, 'w') as f:
            f.write('\n'.join(modified_lines) + '\n')
        
        print(f"Created .sub file: {sub_file_path}")
        
        # Submit the job if requested
        if submit:
            try:
                subprocess.run(
                    ["condor_submit", sub_file_path], 
                    capture_output=True, 
                    text=True, 
                    check=True
                )
                print(f"Job submitted successfully!\n\n\n")
            except Exception as e:
                print(f"Error submitting job: {e}")
                return 1
        else:
            print(f"To submit the job, run: condor_submit {sub_file_path}")
        
        return 0

            
def main():
    args = parser.parse_args()
    
    # Filter out # TODO: this can likely be improved in the future
    trainer_args = {}
    submit_args = {}
    submit_args_keys = ['setup_submission', 'template_file', 'submit']
    
    for key, value in vars(args).items():
        if key in submit_args_keys:
            submit_args[key] = value
        else:
            trainer_args[key] = value
    trainer = NFPriorCreator(**trainer_args)
    
    # Handle submission modes
    if args.submit:
        print(f"Setting up submission and submitting job with the following parameters:")
        return trainer.setup_submission(vars(args), submit=True)
    elif args.setup_submission:
        print(f"Setting up submission (without submitting) with the following parameters:")
        return trainer.setup_submission(vars(args), submit=False)
    
    print(f"Starting training with the following parameters:")
    trainer.create_data()
    trainer.train()
    
if __name__ == "__main__":
    exit(main() or 0)