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

# ### flowjax imports
# import jax
# import jax.numpy as jnp
# import jax.random as jr
# from flowjax.flows import masked_autoregressive_flow, coupling_flow
# from flowjax.flows import block_neural_autoregressive_flow
# from flowjax.train import fit_to_data
# from flowjax.distributions import Normal
# jax_devices = jax.devices()
# print(f"JAX: devices available: {jax_devices}")
import equinox as eqx

import tqdm
import time
import joblib

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
parser.add_argument("--use-tilde", 
                    action="store_true", 
                    help="Use tilde parameterization for lambdas (lambda_tilde, delta_lambda_tilde) instead of (lambda_1, lambda_2)")
parser.add_argument("--no-use-tilde", 
                    dest="use_tilde", 
                    action="store_false")
parser.set_defaults(use_tilde=True)
parser.add_argument("--use-component-masses", 
                    action="store_true", 
                    help="Use component masses (m1, m2) instead of (Mc, q)")
parser.add_argument("--no-use-component-masses", 
                    dest="use_component_masses", 
                    action="store_false")
parser.set_defaults(use_component_masses=False)
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
parser.add_argument("--validation-split-fraction", 
                    type=float, 
                    default=0.2, 
                    help="Fraction of data to use for validation (default: 0.2)")

    
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
                 nn_depth: int = 5,
                 nn_block_dim: int = 8,
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
        
        print(f"Training a normalizing flow for {self.source_type} sources")
        self.N_samples_training = N_samples_training
        self.N_samples_plot = N_samples_plot
        self.m_max_BH = m_max_BH
        self.take_log_lambda = take_log_lambda
        self.use_flowjax = use_flowjax
        self.population_type = population_type
        self.use_tilde = use_tilde
        self.use_component_masses = use_component_masses

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
        self.validation_split_fraction = validation_split_fraction
        
        # Store the NF kwargs here to dump later on
        self.nf_kwargs = {"n_transforms": self.n_transforms,
                          "n_neurons": self.n_neurons,
                          "n_blocks_per_transform": self.n_blocks_per_transform,
                          "nn_depth": self.nn_depth,
                          "nn_block_dim": self.nn_block_dim
                          }
        
        # Set names based on parameterization
        mass_names = ["m_1", "m_2"] if self.use_component_masses else ["chirp_mass", "mass_ratio"]
        if self.source_type == "nsbh" and not self.use_tilde:
            lambda_names = ["lambda_2"]
        else:
            lambda_names = ["lambda_tilde", "delta_lambda_tilde"] if self.use_tilde else ["lambda_1", "lambda_2"]
        self.nf_kwargs["names"] = mass_names + lambda_names
        self.nf_kwargs["n_inputs"] = len(self.nf_kwargs["names"])
        
        print(f"Before training, we built the following NF kwargs: {self.nf_kwargs}")
        
        if self.scale_input and self.take_log_lambda:
            raise ValueError("Cannot scale input and take log of Lambdas at the same time. \
                Please set either scale_input=False or take_log_lambda=False.\
                Recommended to use scaling and not log transformation for training (empirically gave best results).")
        
        # Check flowJAX availability if requested
        if self.use_flowjax and not globals().get('jax'):
            raise ImportError("flowJAX requested but JAX/flowJAX not available. Install flowJAX or set use_flowjax=False")
        
        # Make an outdir based on the given name etc, so that everything is stored in one directory for later on
        backend_suffix = "_flowjax" if self.use_flowjax else ""
        self.outdir = os.path.join("./models/", self.population_type, self.source_type, f"{self.eos_samples_name}{backend_suffix}")
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
            if self.source_type == "bns":
                # Sample two masses uniformly between 1.0 and MTOV, and ensure m1 >= m2
                mass_samples = np.random.uniform(1.0, mtov, 2)
                m1 = np.max(mass_samples)
                m2 = np.min(mass_samples)
                
                Lambda_1 = np.interp(m1, m, l)
                Lambda_2 = np.interp(m2, m, l)
            else:
                # This automatically satisfies m1 >= m2
                m1 = np.random.uniform(mtov, self.m_max_BH, 1)[0]
                m2 = np.random.uniform(1.0, mtov, 1)[0]
                
                Lambda_1 = 0.0
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
        
        # Also get chirp mass and mass ratio for the training data
        chirp_mass = component_masses_to_chirp_mass(m1_list, m2_list)
        mass_ratio = component_masses_to_mass_ratio(m1_list, m2_list)
        
        full_save_path = os.path.join(self.outdir, f"training_data.npz")
        self.training_filename = full_save_path
        print(f"Saving training data to {full_save_path}")
        np.savez(full_save_path, 
                 m1=np.array(m1_list),
                 m2=np.array(m2_list),
                 lambda_1=np.array(Lambda1_list),
                 lambda_2=np.array(Lambda2_list),
                 lambda_tilde=np.array(lambda_tilde),
                 delta_lambda_tilde=np.array(delta_lambda_tilde),
                 chirp_mass=np.array(chirp_mass),
                 mass_ratio=np.array(mass_ratio)
        )
        print(f"Saving training data DONE")
        
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
            print("Using chirp mass and mass ratio (Mc, q) for training")
            # Load or calculate chirp mass and mass ratio
            train_mass_1 = component_masses_to_chirp_mass(m1_raw, m2_raw)
            train_mass_2 = component_masses_to_mass_ratio(m1_raw, m2_raw)
        
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
        split_result = train_test_split(*arrays_to_split, test_size=self.validation_split_fraction, random_state=42)
        
        # Unpack results
        self.train_mass_1, self.val_mass_1 = split_result[0], split_result[1]
        self.train_mass_2, self.val_mass_2 = split_result[2], split_result[3]
        self.train_lambda_1, self.val_lambda_1 = split_result[4], split_result[5]
        self.train_lambda_2, self.val_lambda_2 = split_result[6], split_result[7]
        
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
        
        if self.source_type == "bns":
            # This is always a 4D model
            print(f"We are training for BNS")
            self.x = np.array([self.train_mass_1, self.train_mass_2, self.train_lambda_1, self.train_lambda_2]).T
            self.x_val = np.array([self.val_mass_1, self.val_mass_2, self.val_lambda_1, self.val_lambda_2]).T
            
        elif self.source_type == "nsbh":
            print(f"We are training NSBH")
            if self.use_tilde:
                # This is a 4D model
                print("NSBH: training lambda tildes, so 4D model")
                self.x = np.array([self.train_mass_1, self.train_mass_2, self.train_lambda_2]).T
                self.x_val = np.array([self.val_mass_1, self.val_mass_2, self.val_lambda_2]).T
            else:
                # This is a 3D model, since Lambda_1 is always 0 for NSBH
                print("NSBH: training lambda_2")
                self.x = np.array([self.train_mass_1, self.train_mass_2, self.train_lambda_2]).T
                self.x_val = np.array([self.val_mass_1, self.val_mass_2, self.val_lambda_2]).T
        else:
            raise ValueError(f"Something wrong. To check. Source type is {self.source_type}")
        
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
        raise NotImplementedError("Unconditional flowJAX training not trusted enough yet, and we will stick to the .")
    
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