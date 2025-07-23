"""
Train a normalizing flow to approximate a distribution on masses and Lambdas to replicate an EOS dataset and be used in inference.
This might be joint prior or conditional prior, we will have to check which works best later on. 
"""

import os
import sys
import numpy as np
import json

### glasflow imports
from glasflow.flows import RealNVP
from glasflow.flows.autoregressive import MaskedPiecewiseRationalQuadraticAutoregressiveFlow, MaskedAffineAutoregressiveFlow
from glasflow.flows.nsf import CouplingNSF # from a few testing experiments, this is the best trade-off between working code and performance
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
    
class NFPriorCreator:
    """
    Class to construct the NF prior and train it.
    """
    
    def __init__(self,
                 eos_samples_filename: str = None,
                 source_type: str = "bns",
                 N_samples_training: int = 100_000,
                 N_samples_plot: int = 10_000,
                 m_max_BH: float = 5.0,
                 save_name: str = "",
                 conditional: bool = True,
                 take_log_lambda: bool = False,
                 use_flowjax: bool = False,  # Toggle between glasflow and flowJAX
                 num_epochs: int = 250,
                 learning_rate: float = 1e-3,
                 batch_size: int = 256,
                 scale_input: bool = True,
                 # glasflow-specific training arguments:
                 max_patience: int = 50,
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
            eos_samples_filename (str, optional): Filename where we load the EOS samples from, which will be converted into the training data for the NF for binary systems. Defaults to None.
            source_type (str, optional): Which kind of source to model: `bns` or `nsbh`. Defaults to "bns".
            N_samples_training (int, optional): Number of training samples to create.. Defaults to 100_000.
            N_samples_plot (int, optional): Number of samples to create the plots. Defaults to 10_000.
            m_max_BH (float, optional): If generating NSBH training data with an NF that is not conditioned on the masses, this is up to which the masses are taken. Defaults to 5.0.
            save_name (str, optional): Where to save the models etc to. Defaults to "".
            conditional (bool, optional): Whether to train the NF in a conditional manner, i.e., Lambdas as function of masses. Defaults to True as this is recommended for our bilby setup.
            take_log_lambda (bool, optional): Whether to take the log of the Lambdas before training to deal with their massive scaling, to improve training the NF. Defaults to True.
            use_flowjax (bool, optional): Whether to use flowJAX instead of glasflow for training. Defaults to False.
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
        self.eos_samples_filename = eos_samples_filename
        SUPPORTED_SOURCE_TYPES = ["bns", "nsbh"]
        if source_type not in SUPPORTED_SOURCE_TYPES:
            raise ValueError(f"source_type must be one of {SUPPORTED_SOURCE_TYPES}, got {source_type} instead. Defaulting to `bns`")
        self.source_type = source_type
        self.N_samples_training = N_samples_training
        self.N_samples_plot = N_samples_plot
        self.m_max_BH = m_max_BH
        self.conditional = conditional
        self.take_log_lambda = take_log_lambda
        self.use_flowjax = use_flowjax

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
        
        # Check flowJAX availability if requested
        if self.use_flowjax and not globals().get('jax'):
            raise ImportError("flowJAX requested but JAX/flowJAX not available. Install flowJAX or set use_flowjax=False")
        
        # Make an outdir based on the given name etc, so that everything is stored in one directory for later on
        backend_suffix = "_flowjax" if self.use_flowjax else ""
        if len(save_name) > 0:
            if self.conditional:
                self.outdir = os.path.join(f"./models/{save_name}_conditional_{self.source_type}{backend_suffix}")
            else:
                self.outdir = os.path.join(f"./models/{save_name}_{self.source_type}{backend_suffix}")
        else:
            if self.conditional:
                self.outdir = os.path.join(f"./models/conditional_{self.source_type}{backend_suffix}")
            else:
                self.outdir = os.path.join(f"./models/{self.source_type}{backend_suffix}")
            
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)
            print(f"Created output directory {self.outdir}")
        print(f"Everything for this model will be saved to {self.outdir}")
    
    
    def load_eos_samples_from_file(self) -> tuple[np.array, np.array, np.array]:
        """
        Load in the EOS samples from the file and clean them. This returns the masses, radii and Lambdas of the EOS samples.
        """
        if self.eos_samples_filename is None:
            # By default, we use the one from an EOS inference on only the radio timing (which ensures MTOV > 2)
            self.eos_samples_filename = "../data/eos/eos_samples.npz"
            print(f"No EOS samples filename was provided, so defaulted to: {self.eos_samples_filename}")
            
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
        Create the dataset from the EOS to neutron star systems: m1, m2, Lambda1, Lambda2.
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
                
        full_save_path = os.path.join(self.outdir, f"training_data.npz")
        self.training_filename = full_save_path
        print(f"Saving training data to {full_save_path}")
        np.savez(full_save_path, 
                 m1=np.array(m1_list),
                 m2=np.array(m2_list),
                 lambda_1=np.array(Lambda1_list),
                 lambda_2=np.array(Lambda2_list))
        print(f"Saving training data DONE")
        
    def load_training_data(self, training_filename: str):
        """
        Loads in the preprocessed training data before feeding it into the NF for training.
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
        
        if self.source_type == "nsbh":
            # For NSBH: create concatenated arrays for single NS modeling
            print("NSBH mode: concatenating all neutron star data for single NS modeling")
            self.m2 = np.concatenate([m1_raw, m2_raw])
            self.lambda_2 = np.concatenate([lambda_1_raw, lambda_2_raw])
        else:
            # For BNS: use original arrays
            self.m1 = m1_raw
            self.m2 = m2_raw
            self.lambda_1 = lambda_1_raw
            self.lambda_2 = lambda_2_raw
        
        if self.take_log_lambda:
            # Take the log of the Lambdas
            print("Taking the log of the Lambdas")
            
            if self.source_type == "nsbh":
                self.lambda_2 = np.log(self.lambda_2)
            else:
                self.lambda_1 = np.log(self.lambda_1)
                self.lambda_2 = np.log(self.lambda_2)
        
    def train(self):
        """
        Just branching off to the unconditional or conditional training function, depending on the type of training we want to do.
        """
        
        # Load the data, from which we infer what system we are training for
        training_filename = os.path.join(self.outdir, "training_data.npz")
        self.load_training_data(training_filename)
        
        print("Sanity checking the ranges of the training data")
        
        if self.source_type == "nsbh":
            print(f"m2 (concatenated) ranges from {np.min(self.m2)} to {np.max(self.m2)}")
            print(f"lambda_2 (concatenated) ranges from {np.min(self.lambda_2)} to {np.max(self.lambda_2)}")
        else:
            print(f"m1 ranges from {np.min(self.m1)} to {np.max(self.m1)}")
            print(f"m2 ranges from {np.min(self.m2)} to {np.max(self.m2)}")
            print(f"lambda_1 ranges from {np.min(self.lambda_1)} to {np.max(self.lambda_1)}")
            print(f"lambda_2 ranges from {np.min(self.lambda_2)} to {np.max(self.lambda_2)}")
        
        # Start storing kwargs here to dump later on
        self.nf_kwargs = {"n_transforms": self.n_transforms,
                          "n_neurons": self.n_neurons,
                          "n_blocks_per_transform": self.n_blocks_per_transform,
                          "nn_depth": self.nn_depth,
                          "nn_block_dim": self.nn_block_dim
                          }
        
        # Build depending on source and conditional
        if self.source_type == "bns" and self.conditional:
            print(f"We are training BNS and conditional")
            self.x = np.array([self.lambda_1, self.lambda_2]).T
            self.u = np.array([self.m1, self.m2]).T
            
            self.n_inputs = 2
            self.n_conditional_inputs = 2
            
            self.nf_kwargs["names"] = ["lambda_1", "lambda_2"]
            self.nf_kwargs["names_conditional"] = ["m_1", "m_2"]
            
        elif self.source_type == "bns" and not self.conditional:
            print(f"We are training BNS and not conditional")
            self.x = np.array([self.m1, self.m2, self.lambda_1, self.lambda_2]).T
            self.u = None
            
            self.n_inputs = 4
            self.n_conditional_inputs = None
            
            self.nf_kwargs["names"] = ["m_1", "m_2", "lambda_1", "lambda_2"]
            self.nf_kwargs["names_conditional"] = []
            
        elif self.source_type == "nsbh" and self.conditional:
            print(f"We are training NSBH and conditional - using concatenated NS data")
            self.x = self.lambda_2.reshape(-1, 1)  # Shape: (2*N_samples, 1)
            self.u = self.m2.reshape(-1, 1)       # Shape: (2*N_samples, 1)
            
            self.n_inputs = 1
            self.n_conditional_inputs = 1
            
            self.nf_kwargs["names"] = ["lambda_2"]
            self.nf_kwargs["names_conditional"] = ["m_2"]
            
        elif self.source_type == "nsbh" and not self.conditional:
            print(f"We are training NSBH and not conditional - using concatenated NS data")
            
            self.x = np.array([self.m2, self.lambda_2]).T
            self.u = None
            
            self.n_inputs = 2
            self.n_conditional_inputs = None
            
            self.nf_kwargs["names"] = ["m2", "lambda_2"]
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
            
        # Print something about ranges:
        if self.source_type == "nsbh":
            print(f"m2 (concatenated) ranges from {np.min(self.m2)} to {np.max(self.m2)}")
            print(f"lambda_2 (concatenated) ranges from {np.min(self.lambda_2)} to {np.max(self.lambda_2)}")
        else:
            print(f"m1 ranges from {np.min(self.m1)} to {np.max(self.m1)}")
            print(f"m2 ranges from {np.min(self.m2)} to {np.max(self.m2)}")
            print(f"lambda_1 ranges from {np.min(self.lambda_1)} to {np.max(self.lambda_1)}")
            print(f"lambda_2 ranges from {np.min(self.lambda_2)} to {np.max(self.lambda_2)}")
           
        if self.scale_input:
            print(f"Using MinMaxScaler to scale the input data x")
            scaler = MinMaxScaler()
            self.x = scaler.fit_transform(self.x)
            
            # Save the scaler
            scaler_savename = os.path.join(self.outdir, "scaler.gz")
            print(f"The scaler will be saved with joblib to {scaler_savename}")
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
        
        print(f"Saving the model kwargs to {save_path}")
        with open(save_path, "w") as f:
            json.dump(self.nf_kwargs, f, indent=4)
            
        # FIXME: deprecate this?
        # # Eval the flow and generate samples for visualizations
        # flow.eval()
        # with torch.no_grad():
        #     nf_samples_np = flow.sample(self.N_samples_plot)
        # nf_samples_np = nf_samples_np.cpu().numpy()
        
        # # Dump the NF samples -- this is mainly for ease of plotting stuff later on
        # nf_samples_save_path = os.path.join(self.outdir, "nf_samples.npz")
        # np.savez(nf_samples_save_path,
        #          nf_samples = nf_samples_np)
        
        print("DONE training!")
        
    def _train_unconditional_glasflow(self) -> CouplingNSF:
        """
        Simple wrapper around a glasflow model to train an unconditional normalizing flow on the data.
        """
        
        # Initialize the flow model
        self.nf_kwargs["model_type"] = "CouplingNSF"
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

        self.plot_loss(train_loss)
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
                
                # TODO: is this fixed now?
                # print("Debug flow loss")
                # print(loss)
                # print(loss.requires_grad)  # Should be True

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

        self.plot_loss(train_loss)
        return flow
    
    def _train_unconditional_flowjax(self):
        """
        Train an unconditional normalizing flow using flowJAX.
        """
        # Create base distribution
        base_dist = Normal(jnp.zeros(self.n_inputs))
        
        # Initialize flow
        key = jr.key(42)
        if self.n_inputs == 1:
            # For 1D, use masked autoregressive flow
            print("Using masked_autoregressive_flow for unconditional 1D distribution")
            self.nf_kwargs["model_type"] = "masked_autoregressive_flow"
            flow = masked_autoregressive_flow(
                key=key,
                base_dist=base_dist,
                flow_layers=self.n_transforms,
                nn_width=self.n_neurons,
                nn_depth=self.n_blocks_per_transform
            )
        else:
            # For >1D, use coupling flow
            print("Using coupling_flow for unconditional >1D distribution")
            self.nf_kwargs["model_type"] = "coupling_flow"
            flow = coupling_flow(
                key=key,
                base_dist=base_dist,
                flow_layers=self.n_transforms,
                nn_width=self.n_neurons,
                nn_depth=self.n_blocks_per_transform
            )
        
        # Train the flow
        train_key = jr.key(123)
        x_data = jnp.array(self.x, dtype=jnp.float32)
        
        print(f"Training flowJAX unconditional model on data shape: {x_data.shape}")
        flow, losses = fit_to_data(
            key=train_key,
            dist=flow,
            data=x_data,
            learning_rate=self.learning_rate,
            max_epochs=self.num_epochs,
            batch_size=self.batch_size,
            patience=self.max_patience
        )
        
        self.plot_loss(np.array(losses))
        return flow
    
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
            
            # FIXME: do this
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
        
        # For conditional training, we need to create a custom loss function
        def conditional_nll_loss(params, data_batch):
            """Negative log-likelihood loss for conditional flow."""
            x_batch, u_batch = data_batch
            return -flow.log_prob(x_batch, context=u_batch).mean()
        
        # Create combined dataset
        combined_data = (x_data, u_data)

        subkey = jr.key(1234)        
        flow, losses = fit_to_data(
            train_key,
            flow,
            data=combined_data,
            learning_rate=self.learning_rate,
            max_epochs=self.num_epochs,
            max_patience=self.max_patience,
            batch_size=self.batch_size,
        )
        
        # TODO: use the losses
        
        return flow
    
    # def _train_conditional_flowjax_custom(self, flow, x_data, u_data, key):
    #     """
    #     Custom training loop for conditional flowJAX models.
    #     """
    #     import optax
        
    #     # Initialize optimizer
    #     optimizer = optax.adam(self.learning_rate)
    #     opt_state = optimizer.init(eqx.filter(flow, eqx.is_inexact_array))
        
    #     losses = []
    #     best_loss = float('inf')
    #     patience_counter = 0
        
    #     # Training loop
    #     n_batches = len(x_data) // self.batch_size
        
    #     for epoch in tqdm.tqdm(range(self.num_epochs), desc="Training flowJAX"):
    #         epoch_loss = 0.0
    #         key, epoch_key = jr.split(key)
            
    #         # Shuffle data
    #         perm_key, epoch_key = jr.split(epoch_key)
    #         perm = jr.permutation(perm_key, len(x_data))
    #         x_shuffled = x_data[perm]
    #         u_shuffled = u_data[perm]
            
    #         for batch_idx in range(n_batches):
    #             start_idx = batch_idx * self.batch_size
    #             end_idx = min((batch_idx + 1) * self.batch_size, len(x_data))
                
    #             x_batch = x_shuffled[start_idx:end_idx]
    #             u_batch = u_shuffled[start_idx:end_idx]
                
    #             # Compute loss and gradients
    #             def loss_fn(model):
    #                 try:
    #                     log_probs = model.log_prob(x_batch, context=u_batch)
    #                     return -log_probs.mean()
    #                 except:
    #                     # Fallback if context parameter doesn't work
    #                     return -model.log_prob(x_batch).mean()
                
    #             loss, grads = eqx.filter_value_and_grad(loss_fn)(flow)
                
    #             # Update parameters
    #             updates, opt_state = optimizer.update(grads, opt_state, flow)
    #             flow = eqx.apply_updates(flow, updates)
                
    #             epoch_loss += loss
            
    #         epoch_loss /= n_batches
    #         losses.append(float(epoch_loss))
            
    #         # Early stopping
    #         if epoch_loss < best_loss:
    #             best_loss = epoch_loss
    #             patience_counter = 0
    #         else:
    #             patience_counter += 1
    #             if patience_counter >= self.max_patience:
    #                 print(f"Early stopping triggered at epoch {epoch+1}")
    #                 break
        
    #     self.plot_loss(np.array(losses))
    #     return flow
        
    def plot_loss(self, loss: np.array) -> None:
        
        # Make a plot of the loss trajectory
        plt.figure(figsize=(12, 6))
        plt.plot(loss, label="Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss Trajectory")
        plt.yscale("log")
        save_path = os.path.join(self.outdir, "training_loss.pdf")
        print(f"Saving the training loss plot to {save_path}")
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
            
def main():
    if len(sys.argv) > 1:
        source_type = sys.argv[1]
        print(f"source_type is set by user to {source_type}")
    else:
        source_type = "bns" # Default value
        print(f"source_type is set to the default value of {source_type}")
    
    trainer = NFPriorCreator(source_type=source_type)
    trainer.create_data()
    trainer.train()
    
if __name__ == "__main__":
    main()