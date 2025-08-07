"""
Evaluate the performance of the flows, by plotting some comparison cornerplots and also calculating eg KL divergences.
"""

import os
import sys
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import corner
import json
import copy
import torch
import joblib
from scipy.special import kl_div
from scipy.stats import gaussian_kde, chisquare

from glasflow.flows.nsf import CouplingNSF
from bilby.gw.conversion import component_masses_to_chirp_mass

### flowjax imports
import jax
import jax.numpy as jnp
import jax.random as jr
from flowjax.flows import coupling_flow, masked_autoregressive_flow, block_neural_autoregressive_flow
from flowjax.distributions import Normal
import equinox as eqx

params = {"axes.grid": True,
        "text.usetex" : False,
        "font.family" : "serif",
        "ytick.color" : "black",
        "xtick.color" : "black",
        "axes.labelcolor" : "black",
        "axes.edgecolor" : "black",
        # "font.serif" : ["Computer Modern Serif"],
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "axes.labelsize": 16,
        "legend.fontsize": 16,
        "legend.title_fontsize": 16,
        "figure.titlesize": 16}
plt.rcParams.update(params)

# Improved corner kwargs
default_corner_kwargs = dict(bins=40, 
                        smooth=1., 
                        show_titles=False,
                        label_kwargs=dict(fontsize=16),
                        title_kwargs=dict(fontsize=16), 
                        color="blue",
                        # quantiles=[],
                        # levels=[0.9],
                        plot_density=True, 
                        plot_datapoints=False, 
                        fill_contours=True,
                        max_n_ticks=4, 
                        min_n_ticks=3,
                        truth_color = "red",
                        save=False)


def make_cornerplot(chains_1: np.array, 
                    chains_2: np.array,
                    name: str,
                    my_range: list[float] = None,
                    truths: list[float] = None):
    """
    Plot a cornerplot of the true data samples and the NF samples
    Note: the shape use is a bit inconsistent below, watch out.
    """

    # The training data:
    corner_kwargs = copy.deepcopy(default_corner_kwargs)
    hist_1d_kwargs = {"density": True, "color": "blue"}
    corner_kwargs["color"] = "blue"
    corner_kwargs["hist_kwargs"] = hist_1d_kwargs
    fig = corner.corner(chains_1, range=my_range, truths=truths, labels=None, **corner_kwargs)

    # The data from the normalizing flow
    corner_kwargs["color"] = "red"
    hist_1d_kwargs = {"density": True, "color": "red"}
    corner_kwargs["hist_kwargs"] = hist_1d_kwargs
    corner.corner(chains_2,  truths=truths, range=my_range, fig=fig, **corner_kwargs)

    # Make a textbox just because that makes the plot cooler
    if len(my_range) <= 2:
        fs = 14
    else:
        fs = 24
    plt.text(0.75, 0.75, "Training data", fontsize = fs, color = "blue", transform = plt.gcf().transFigure)
    plt.text(0.75, 0.65, "Normalizing flow", fontsize = fs, color = "red", transform = plt.gcf().transFigure)

    plt.savefig(name, bbox_inches = "tight")
    plt.close()
    
class Checker:
    """
    Base class for both BNS and NSBH specific checkers.
    Just to unify data loading and initialization.
    """
    
    def __init__(self,
                 path: str,
                 N_samples: int = 10_000,
                 N_masses: int = 5):
        """
        Initialize the checker for the BNS conditional model.
        
        Args:
            path (str): Path to the directory where the model is stored.
            N_samples (int): Number of samples to generate for comparison for a single mass pair.
            N_masses (int): Number of mass pairs to sample from the training data, for the conditioning part of conditional flows.
        """
        self.path = path
        self.figures_outdir = os.path.join(self.path, "figures")
        os.makedirs(self.figures_outdir, exist_ok=True)
        
        self.N_samples = N_samples
        self.N_masses = N_masses
        
    def load_training_data(self):
        """
        Load training data from the model directory.
        """
        training_data_path = os.path.join(self.path, "training_data.npz")
        if os.path.exists(training_data_path):
            data = np.load(training_data_path)
            return data
        else:
            raise FileNotFoundError(f"Training data not found at {training_data_path}. Please ensure the model was trained with training data saved.")
    
# class CheckerBNS(Checker):
    
#     def __init__(self, path: str, N_samples: int = 10_000, N_masses: int = 5):
#         """
#         Initialize the checker for the BNS conditional model.
        
#         Args:
#             path (str): Path to the directory where the model is stored.
#             N_samples (int): Number of samples to generate for comparison for a single mass pair.
#             N_masses (int): Number of mass pairs to sample from the training data, for the conditioning part of conditional flows.
#         """
#         super().__init__(path, N_samples, N_masses)
#         self.flow, self.nf_kwargs, self.masses_EOS, self.Lambdas_EOS, self.scaler = self.load_model_and_data()
    
#     def load_model_and_data(self):
#         """
#         Load the NF model and EOS data for the BNS case.
        
#         Returns:
#             flow: The loaded normalizing flow model (glasflow or flowJAX).
#             nf_kwargs (dict): The configuration parameters for the NF.
#             masses_EOS (np.ndarray): Masses from the EOS samples.
#             Lambdas_EOS (np.ndarray): Corresponding Lambdas from the EOS samples.
#             scaler: The MinMaxScaler used during training (None if not used).
#         """
#         nf_kwargs_path = os.path.join(self.path, "model_kwargs.json")
        
#         with open(nf_kwargs_path, "r") as f:
#             nf_kwargs = json.load(f)
#         self.nf_kwargs = nf_kwargs
        
#         # Check if this is a flowJAX model - all booleans are stored as strings
#         use_flowjax = nf_kwargs.get("use_flowjax", "False") == "True"
        
#         if use_flowjax:
#             nf_path = os.path.join(self.path, "model.eqx")
            
#             # Create base distribution and flow structure
#             base_dist = Normal(jnp.zeros(2))
#             key = jr.key(42)
            
#             # Recreate the flow architecture to match training code
#             flow = block_neural_autoregressive_flow(
#                 key=key,
#                 base_dist=base_dist,
#                 nn_depth=self.nf_kwargs["nn_depth"],
#                 nn_block_dim=self.nf_kwargs["nn_block_dim"],
#                 flow_layers=self.nf_kwargs["flow_layers"],
#             )
            
#             print(f"Loading flowJAX model from {nf_path}")
#             flow = eqx.tree_deserialise_leaves(nf_path, flow)
            
#         else:
#             print("Loading glasflow BNS model")
#             nf_path = os.path.join(self.path, "model.pt")
            
#             flow = CouplingNSF(
#                 n_inputs=nf_kwargs["n_inputs"],
#                 n_transforms=nf_kwargs["n_transforms"],
#                 n_neurons=nf_kwargs["n_neurons"],
#                 n_blocks_per_transform=nf_kwargs["n_blocks_per_transform"],
#                 num_bins=nf_kwargs["num_bins"]
#             )
            
#             print(f"Loading glasflow model from {nf_path}")
#             flow.load_state_dict(torch.load(nf_path, map_location=torch.device('cpu')))
#             flow.eval()
#             flow.compile()
        
#         # Load the EOS posterior samples from jester from which we have created the training data
#         eos_samples_filename = nf_kwargs["eos_samples_filename"]
#         print(f"Comparing against the EOS samples taken from {eos_samples_filename}")
#         data = np.load(eos_samples_filename)
#         masses_EOS, Lambdas_EOS = data["masses_EOS"], data["Lambdas_EOS"]
        
#         # Load the scaler if it exists AND if scale_input was True during training
#         scaler_path = os.path.join(self.path, "scaler.gz")
#         scaler = None
#         scale_input = nf_kwargs.get("scale_input", "False") # FIXME: putting False for  # Default to True for backwards compatibility
        
#         print(f"DEBUG: scale_input = {scale_input}")
        
#         if os.path.exists(scaler_path) and scale_input:
#             print(f"Loading scaler from {scaler_path}")
#             scaler = joblib.load(scaler_path)
#         else:
#             if os.path.exists(scaler_path) and not scale_input:
#                 print("Scaler file exists but scale_input=False - not using scaler")
#             else:
#                 print("No scaler found - assuming input was not scaled during training")
        
#         return flow, nf_kwargs, masses_EOS, Lambdas_EOS, scaler

#     def check_conditional_bns_model(self):
#         """
#         Load the NF model and check the 2D Lambdas distribution conditioned on the given masses.

#         Args:
#             path (str): Path pointing to the directory where the model is stored.
#             m1_value (float): Mass 1 value to condition on.
#             m2_value (float): Mass 2 value to condition on.
#         """
        
#         mass_1 = np.linspace(1.25, 2.0, self.N_masses)
#         mass_2 = np.linspace(1.0, 2.0, self.N_masses)

#         m1_grid, m2_grid = np.meshgrid(mass_1, mass_2)

#         # Ensure m1 >= m2
#         mask = m2_grid <= m1_grid
#         mass_grid = np.column_stack((
#             m1_grid[mask].ravel(),
#             m2_grid[mask].ravel()
#         ))

#         print(f"mass_grid: len = {len(mass_grid)}")
#         print(mass_grid)
        
#         # Do the check for each pair:
#         for m1_value, m2_value in tqdm.tqdm(mass_grid, desc="Checking BNS conditional model"):
#             self.check_single_bns(m1_value, m2_value)
    
#     def check_single_bns(self, m1_value: float, m2_value: float):
#         """
#         Check the conditional BNS model by sampling from the training data and the NF.
#         Args:
#             m1_value (float): Mass 1 value to condition on.
#             m2_value (float): Mass 2 value to condition on.
#         """
#         # Check parameterization flags
#         use_tilde = self.nf_kwargs.get("use_tilde", "False") == "True"
#         use_component_masses = self.nf_kwargs.get("use_component_masses", "True") == "True"
        
#         # Load the training samples based on parameterization
#         if use_tilde:
#             # For tilde parameterization, we need to create lambda_tilde and delta_lambda_tilde
#             lambda_tilde_train = np.zeros(self.N_samples)
#             delta_lambda_tilde_train = np.zeros(self.N_samples)
#         else:
#             # For component parameterization, use lambda_1 and lambda_2
#             lambda_1_train = np.zeros(self.N_samples)
#             lambda_2_train = np.zeros(self.N_samples)
        
#         counter = 0
        
#         # Create the validation set from the EOS samples
#         while counter < self.N_samples:
#             idx = np.random.choice(len(self.masses_EOS), 1)
#             m, l = self.masses_EOS[idx][0], self.Lambdas_EOS[idx][0]
            
#             # Interpolate on the given masses to get the corresponding Lambdas
#             lambda_1_value = np.interp(m1_value, m, l)
#             lambda_2_value = np.interp(m2_value, m, l)
            
#             # Sanity check it:
#             if lambda_1_value < 0.0 or lambda_2_value < 0.0:
#                 continue
#             else:
#                 if use_tilde:
#                     # Convert to tilde parameters
#                     from bilby.gw.conversion import lambda_1_lambda_2_to_lambda_tilde, lambda_1_lambda_2_to_delta_lambda_tilde
#                     lambda_tilde_value = lambda_1_lambda_2_to_lambda_tilde(lambda_1_value, lambda_2_value, m1_value, m2_value)
#                     delta_lambda_tilde_value = lambda_1_lambda_2_to_delta_lambda_tilde(lambda_1_value, lambda_2_value, m1_value, m2_value)
                    
#                     lambda_tilde_train[counter] = lambda_tilde_value
#                     delta_lambda_tilde_train[counter] = delta_lambda_tilde_value
#                 else:
#                     lambda_1_train[counter] = lambda_1_value
#                     lambda_2_train[counter] = lambda_2_value
#                 counter += 1
        
#         # Create training samples array based on parameterization        
#         if use_tilde:
#             training_samples = np.array([lambda_tilde_train, delta_lambda_tilde_train]).T
#         else:
#             training_samples = np.array([lambda_1_train, lambda_2_train]).T
            
#         # Condition on those masses to sample from the conditional NF
#         nf_samples = []
#         use_flowjax = self.nf_kwargs.get("use_flowjax", "False") == "True"
        
#         # Prepare conditioning variables based on mass parameterization
#         if use_component_masses:
#             # For component mass parameterization, condition on m1 and m2
#             conditioning_vars = [m1_value, m2_value]
#         else:
#             # For chirp mass/mass ratio parameterization, condition on Mc and q
#             from bilby.gw.conversion import component_masses_to_chirp_mass, component_masses_to_mass_ratio
#             mc_cond = component_masses_to_chirp_mass(m1_value, m2_value)
#             q_cond = component_masses_to_mass_ratio(m1_value, m2_value)
#             conditioning_vars = [mc_cond, q_cond]
        
#         if use_flowjax:
#             # flowJAX sampling
#             key = jr.key(123)

#             u_jax = jnp.array([conditioning_vars], dtype=jnp.float32)

#             # Generate N_samples random keys
#             keys = jr.split(key, self.N_samples)

#             # Define a single sample function
#             @jax.jit
#             def sample_fn(sample_key):
#                 val = self.flow.sample(sample_key, (1,), condition=u_jax).flatten()
#                 if self.nf_kwargs["take_log_lambda"] == "True":
#                     val = np.exp(val)  # Scale back to original Lambda_2
#                 return val

#             # Vectorize over keys
#             nf_samples_jax = jax.vmap(sample_fn)(keys)

#             # Convert to numpy
#             nf_samples = np.array(nf_samples_jax)
        
#         else:
#             # glasflow sampling
#             u = torch.tensor([conditioning_vars], dtype=torch.float32)
#             with torch.no_grad():
#                 for _ in range(self.N_samples):
#                     value = self.flow.sample(1, conditional=u).cpu().numpy().flatten()
#                     if self.nf_kwargs.get("take_log_lambda", "False") == "True":
#                         value = np.exp(value)  # Scale back to original Lambda_2
#                     nf_samples.append(value)
                    
#         nf_samples = np.array(nf_samples)
        
#         # Apply inverse scaling if scaler was used during training
#         if self.scaler is not None:
#             nf_samples = self.scaler.inverse_transform(nf_samples)
        
#         # Determine range from data, use quantiles
#         param_1_nf, param_2_nf = nf_samples[:, 0], nf_samples[:, 1]
        
#         param_1_lower, param_1_upper = np.quantile(param_1_nf, [0.01, 0.99])
#         param_2_lower, param_2_upper = np.quantile(param_2_nf, [0.01, 0.99])
        
#         my_range = [[param_1_lower, param_1_upper], [param_2_lower, param_2_upper]]
        
#         # Create parameter labels and filename based on parameterization
#         if use_tilde:
#             labels = [r"$\tilde{\Lambda}$", r"$\delta\tilde{\Lambda}$"]
#         else:
#             labels = [r"$\Lambda_1$", r"$\Lambda_2$"]
        
#         if use_component_masses:
#             name = os.path.join(self.figures_outdir, f"m1_{m1_value:.2f}_m2_{m2_value:.2f}.pdf")
#         else:
#             name = os.path.join(self.figures_outdir, f"Mc_{conditioning_vars[0]:.2f}_q_{conditioning_vars[1]:.2f}.pdf")
        
#         print(f"Saving cornerplot to {name}")
#         make_cornerplot(training_samples, nf_samples, name, my_range=my_range, labels=labels)
        
#         # # Compute the KL divergence
#         # all_samples = np.concatenate([training_samples.flatten(), nf_samples.flatten()])
#         # min_value, max_value = np.min(all_samples), np.max(all_samples)
        
#         # p_kde = gaussian_kde(training_samples.flatten())
#         # q_kde = gaussian_kde(nf_samples.flatten())

#         # x_grid = np.linspace(min_value, max_value, 1000)

#         # p_vals = p_kde(x_grid)
#         # q_vals = q_kde(x_grid)

#         # kl = np.sum(p_vals * np.log(p_vals / q_vals)) * (x_grid[1] - x_grid[0])

#         # print("kl")        
#         # print(kl)        

# class CheckerNSBH(Checker):
#     """
#     Checker for the NSBH conditional model.
#     """
    
#     def __init__(self, path: str, N_samples: int = 10_000, N_masses: int = 5):
#         """
#         Initialize the checker for the NSBH conditional model.
        
#         Args:
#             path (str): Path to the directory where the model is stored.
#             N_samples (int): Number of samples to generate for comparison for a single mass pair.
#             N_masses (int): Number of mass pairs to sample from the training data.
#         """
#         super().__init__(path, N_samples, N_masses)
        
#         # Load the model and data
#         self.flow, self.nf_kwargs, self.masses_EOS, self.Lambdas_EOS, self.scaler = self.load_model_and_data()
        
#     def load_model_and_data(self):
#         """
#         Load the NF model and EOS data for the NSBH case.
        
#         Returns:
#             flow: The loaded normalizing flow model (glasflow or flowJAX).
#             nf_kwargs (dict): The configuration parameters for the NF.
#             masses_EOS (np.ndarray): Masses from the EOS samples.
#             Lambdas_EOS (np.ndarray): Corresponding Lambdas from the EOS samples.
#             scaler: The MinMaxScaler used during training (None if not used).
#         """
#         nf_kwargs_path = os.path.join(self.path, "model_kwargs.json")
        
#         with open(nf_kwargs_path, "r") as f:
#             nf_kwargs = json.load(f)
        
#         # Check if this is a flowJAX model
#         use_flowjax = nf_kwargs.get("use_flowjax", "False") == "True"
        
#         if use_flowjax:
#             print("Loading flowJAX NSBH model")
#             nf_path = os.path.join(self.path, "model.eqx")
            
#             # Create base distribution and flow structure for 1D
#             base_dist = Normal(jnp.zeros(1))
#             key = jr.key(42)
            
#             # Recreate the flow architecture
#             flow = masked_autoregressive_flow(
#                 key=key,
#                 base_dist=base_dist,
#                 cond_dim=1,
#                 flow_layers=nf_kwargs["n_transforms"],
#                 nn_width=nf_kwargs["n_neurons"],
#                 nn_depth=nf_kwargs["n_blocks_per_transform"]
#             )
            
#             print(f"Loading flowJAX NSBH model from {nf_path}")
#             flow = eqx.tree_deserialise_leaves(nf_path, flow)
            
#         else:
#             print("Loading glasflow NSBH model")
#             nf_path = os.path.join(self.path, "model.pt")
            
#             flow = CouplingNSF(
#                 n_inputs=nf_kwargs["n_inputs"],
#                 n_transforms=nf_kwargs["n_transforms"],
#                 n_neurons=nf_kwargs["n_neurons"],
#                 n_blocks_per_transform=nf_kwargs["n_blocks_per_transform"],
#                 num_bins=nf_kwargs["num_bins"]
#             )
#             print(f"Loading glasflow NSBH model from {nf_path}")
#             flow.load_state_dict(torch.load(nf_path, map_location=torch.device('cpu')))
#             flow.eval()
#             flow.compile()
        
#         print(f"NSBH model loaded successfully")
        
#         # Load the EOS posterior samples from jester from which we have created the training data
#         eos_samples_filename = nf_kwargs["eos_samples_filename"]
#         print(f"Comparing against the EOS samples taken from {eos_samples_filename}")
#         data = np.load(eos_samples_filename)
#         masses_EOS, Lambdas_EOS = data["masses_EOS"], data["Lambdas_EOS"]
        
#         # Load the scaler if it exists AND if scale_input was True during training
#         scaler_path = os.path.join(self.path, "scaler.gz")
#         scaler = None
#         scale_input = nf_kwargs.get("scale_input", "True") == "True"  # Default to True for backwards compatibility
        
#         if os.path.exists(scaler_path) and scale_input:
#             print(f"Loading scaler from {scaler_path}")
#             scaler = joblib.load(scaler_path)
#         else:
#             if os.path.exists(scaler_path) and not scale_input:
#                 print("Scaler file exists but scale_input=False - not using scaler")
#             else:
#                 print("No scaler found - assuming input was not scaled during training")
        
#         return flow, nf_kwargs, masses_EOS, Lambdas_EOS, scaler
    
#     def check_conditional_nsbh_model(self):
#         """
#         Iterate over an array of m2 values and check the conditional NSBH model.
#         """
#         # Generate a grid of m2 values
#         m2_values = np.linspace(1.0, 2.0, self.N_masses)

#         for m2_value in tqdm.tqdm(m2_values, desc="Checking NSBH conditional model"):
#             self.check_single_nsbh(m2_value)
    
#     def check_single_nsbh(self, m2_value: float = 1.4):
#         """
#         Load the NSBH conditional NF model and compare Lambda_2 marginal distributions.
        
#         Args:
#             m2_value (float): Mass 2 value (NS mass) to condition on.
#         """
#         # Generate training data Lambda_2 values for the given m2_value
#         lambda_2_train = np.zeros(self.N_samples)
        
#         counter = 0
        
#         while counter < self.N_samples:
#             idx = np.random.choice(len(self.masses_EOS), 1)
#             m, l = self.masses_EOS[idx][0], self.Lambdas_EOS[idx][0]
#             lambda_2_value = np.interp(m2_value, m, l)
            
#             # Check if not negative:
#             if lambda_2_value < 0:
#                 continue
            
#             lambda_2_train[counter] = lambda_2_value
#             counter += 1
#         # Generate samples from the conditional NF
#         nf_samples = []
#         use_flowjax = self.nf_kwargs.get("use_flowjax", "False") == "True"
        
#         if use_flowjax:
#             # flowJAX sampling
#             key = jr.key(123)
#             u_jax = jnp.array([[m2_value]], dtype=jnp.float32)
            
#             for _ in range(self.N_samples):
#                 key, sample_key = jr.split(key)
#                 try:
#                     # Try different context parameter names
#                     value = self.flow.sample(sample_key, (1,), context=u_jax).flatten()[0]
#                 except:
#                     try:
#                         value = self.flow.sample(sample_key, (1,), conditional=u_jax).flatten()[0]
#                     except:
#                         # Fallback - sample without conditioning
#                         value = self.flow.sample(sample_key, (1,)).flatten()[0]
                        
#                 value = float(value)
#                 if self.nf_kwargs["take_log_lambda"] == "True":
#                     value = np.exp(value)
#                 nf_samples.append(value)
#         else:
#             # glasflow sampling  
#             u = torch.tensor([[m2_value]], dtype=torch.float32)  # Conditional input
#             with torch.no_grad():
#                 for _ in range(self.N_samples):
#                     value = self.flow.sample(1, conditional=u).cpu().numpy().flatten()[0]
#                     if self.nf_kwargs.get("take_log_lambda", "False") == "True":
#                         value = np.exp(value)  # Scale back to original Lambda_2
#                     nf_samples.append(value)
                    
#         nf_samples = np.array(nf_samples)
        
#         # Apply inverse scaling if scaler was used during training
#         if self.scaler is not None:
#             nf_samples = self.scaler.inverse_transform(nf_samples.reshape(-1, 1)).flatten()
        
#         # Compare the marginal distributions
#         plt.figure(figsize=(10, 6))
#         # Plot histograms
#         lambda_2_lower, lambda_2_upper = np.quantile(lambda_2_train, [0.01, 0.99])
#         bins = np.linspace(lambda_2_lower, lambda_2_upper, 50)
#         plt.hist(lambda_2_train, bins=bins, alpha=0.7, density=True,
#                  label='Training Data', color='blue', edgecolor='black')
#         plt.hist(nf_samples, bins=bins, alpha=0.7, density=True,
#                  label='NF Samples', color='red', edgecolor='black')
        
#         plt.xlabel(r'$\Lambda_2$')
#         plt.ylabel('Density')
        
#         plt.title(f'NSBH Conditional Prior: $p(\Lambda_2 | m_2 = {m2_value:.1f} M_\odot)$')
#         plt.legend()
#         plt.grid(True, alpha=0.3)
        
#         # Save the plot
#         name = os.path.join(self.figures_outdir, f"m2_{m2_value:.2f}.pdf")
#         plt.savefig(name, bbox_inches="tight")
#         plt.close()
        
#         # TODO: KL divergence calculation

class CheckerUnconditional(Checker):
    """
    Checker for unconditional models (both BNS and NSBH).
    """
    
    def __init__(self, path: str, N_samples: int = 10_000):
        """
        Initialize the checker for unconditional models.
        
        Args:
            path (str): Path to the directory where the model is stored.
            N_samples (int): Number of samples to generate for comparison.
        """
        super().__init__(path, N_samples, N_masses=0)  # N_masses not used for unconditional
        self.flow, self.nf_kwargs, self.scaler = self.load_model_and_data()
        print(f"Loading training data")
        self.training_data = self.load_training_data()
        print(f"Loading training data DONE")
        
    def load_model_and_data(self):
        """
        Load the unconditional NF model.
        
        Returns:
            flow: The loaded normalizing flow model.
            nf_kwargs (dict): The configuration parameters for the NF.
            scaler: The MinMaxScaler used during training (None if not used).
        """
        nf_kwargs_path = os.path.join(self.path, "model_kwargs.json")
        
        with open(nf_kwargs_path, "r") as f:
            nf_kwargs = json.load(f)
        
        # Check if this is a flowJAX model
        use_flowjax = nf_kwargs.get("use_flowjax", "False") == "True"
        n_inputs = nf_kwargs["n_inputs"]
        
        if use_flowjax:
            print(f"Loading flowJAX unconditional model with {n_inputs} inputs")
            nf_path = os.path.join(self.path, "model.eqx")
            
            # Create base distribution
            base_dist = Normal(jnp.zeros(n_inputs))
            key = jr.key(42)
            
            # Choose flow type based on dimensionality and model type
            model_type = nf_kwargs.get("model_type", "coupling_flow")
            if model_type == "coupling_flow":
                flow = coupling_flow(
                    key=key,
                    base_dist=base_dist,
                    flow_layers=nf_kwargs["n_transforms"],
                    nn_width=nf_kwargs["n_neurons"],
                    nn_depth=nf_kwargs["nn_depth"]
                )
            elif model_type == "masked_autoregressive_flow":
                flow = masked_autoregressive_flow(
                    key=key,
                    base_dist=base_dist,
                    flow_layers=nf_kwargs["n_transforms"],
                    nn_width=nf_kwargs["n_neurons"],
                    nn_depth=nf_kwargs["nn_depth"]
                )
            elif model_type == "block_neural_autoregressive_flow":
                flow = block_neural_autoregressive_flow(
                    key=key,
                    base_dist=base_dist,
                    flow_layers=nf_kwargs["flow_layers"],
                    nn_depth=nf_kwargs["nn_depth"],
                    nn_block_dim=nf_kwargs["nn_block_dim"]
                )
            else:
                raise ValueError(f"Unsupported flowJAX model type: {model_type}")
            
            print(f"Loading flowJAX model from {nf_path}")
            flow = eqx.tree_deserialise_leaves(nf_path, flow)
            
        else:
            print(f"Loading glasflow unconditional model with {n_inputs} inputs")
            nf_path = os.path.join(self.path, "model.pt")
            
            flow = CouplingNSF(
                n_inputs=n_inputs,
                n_transforms=nf_kwargs["n_transforms"],
                n_neurons=nf_kwargs["n_neurons"],
                n_blocks_per_transform=nf_kwargs["n_blocks_per_transform"],
                num_bins=nf_kwargs["num_bins"]
            )
            
            print(f"Loading glasflow model from {nf_path}")
            flow.load_state_dict(torch.load(nf_path, map_location=torch.device('cpu')))
            flow.eval()
            flow.compile()
        
        # Load the scaler if it exists AND if scale_input was True during training
        scaler_path = os.path.join(self.path, "scaler.gz")
        scaler = None
        scale_input = nf_kwargs.get("scale_input", "True") == "True"
        
        if os.path.exists(scaler_path) and scale_input:
            print(f"Loading scaler from {scaler_path}")
            scaler = joblib.load(scaler_path)
        else:
            if os.path.exists(scaler_path) and not scale_input:
                print("Scaler file exists but scale_input=False - not using scaler")
            else:
                print("No scaler found - assuming input was not scaled during training")
        
        return flow, nf_kwargs, scaler
    
    def check_unconditional_model(self):
        """
        Check the unconditional model by comparing training data with NF samples.
        """
        if self.training_data is None:
            print("No training data available for comparison")
            return
        
        # Generate samples from the NF
        nf_samples = self.generate_nf_samples()
        
        # Get training samples in the same format
        training_samples = self.get_training_samples()
        
        # Create parameter labels based on use_tilde setting
        use_tilde = self.nf_kwargs.get("use_tilde", "False") == "True"
        parameter_names = self.nf_kwargs.get("names", [])
        
        labels = self.get_parameter_labels(parameter_names)
        
        # Determine range from combined data
        all_data = np.concatenate([training_samples, nf_samples], axis=0)
        ranges = []
        for i in range(all_data.shape[1]):
            lower, upper = np.quantile(all_data[:, i], [0.01, 0.99])
            ranges.append([lower, upper])
        
        # Create corner plot
        name = os.path.join(self.figures_outdir, "unconditional_comparison.pdf")
        print(f"Saving unconditional model comparison to {name}")
        
        self.make_cornerplot_with_labels(training_samples, nf_samples, name, 
                                       my_range=ranges, labels=labels)
        
        # For non-tilde lambda models, check if lambda_1 < lambda_2 for all generated samples
        if self.nf_kwargs.get("use_tilde", "False") == "False" and self.nf_kwargs.get("source_type", "bns") == "bns":
            print("\nChecking if lambda_1 < lambda_2 for all generated samples...")
            
            # Find indices of lambda_1 and lambda_2 in the parameter names
            parameter_names = self.nf_kwargs.get("names", [])
            if "lambda_1" in parameter_names and "lambda_2" in parameter_names:
                lambda_1_idx = parameter_names.index("lambda_1")
                lambda_2_idx = parameter_names.index("lambda_2")
                
                print(f"lambda_1 range: [{np.min(nf_samples[:, lambda_1_idx]):.3f}, {np.max(nf_samples[:, lambda_1_idx]):.3f}]")
                print(f"lambda_2 range: [{np.min(nf_samples[:, lambda_2_idx]):.3f}, {np.max(nf_samples[:, lambda_2_idx]):.3f}]")
                
                ok_samples = nf_samples[:, lambda_1_idx] < nf_samples[:, lambda_2_idx]
                percentage_ok = np.sum(ok_samples) / len(ok_samples) * 100
                print(f"Percentage of samples with lambda_1 < lambda_2: {percentage_ok:.2f}%")
            else:
                print("Could not find lambda_1 and lambda_2 in parameter names - skipping constraint check")
    
    def get_parameter_labels(self, parameter_names):
        """Get parameter labels based on parameterization flags and parameter names."""
        translation_dict = {"chirp_mass": r"$M_c$ [M$_\odot$]",
                            "chirp_mass_source": r"$M_c^{\rm{source}}$ [M$_\odot$]",
                            "q": r"$q$",
                            "mass_ratio": r"$q$",
                            "lambda_1": r"$\Lambda_1$",
                            "lambda_2": r"$\Lambda_2$",
                            "lambda_tilde": r"$\tilde{\Lambda}$",
                            "delta_lambda_tilde": r"$\delta\tilde{\Lambda}$",
                            "luminosity_distance": r"$d_L$ [Mpc]",
                            "redshift": r"$z$",
                            "m_1": r"$m_1$ [M$_\odot$]",
                            "m_2": r"$m_2$ [M$_\odot$]"
                            }
        
        return [translation_dict[name] for name in parameter_names]
        
    
    def make_cornerplot_with_labels(self, chains_1: np.array, chains_2: np.array,
                                  name: str, my_range: list = None, 
                                  truths: list = None, labels: list = None):
        """
        Create corner plot with parameter labels.
        """
        # The training data:
        corner_kwargs = copy.deepcopy(default_corner_kwargs)
        hist_1d_kwargs = {"density": True, "color": "blue"}
        corner_kwargs["color"] = "blue"
        corner_kwargs["hist_kwargs"] = hist_1d_kwargs
        fig = corner.corner(chains_1, range=my_range, truths=truths, 
                          labels=labels, **corner_kwargs)

        # The data from the normalizing flow
        corner_kwargs["color"] = "red"
        hist_1d_kwargs = {"density": True, "color": "red"}
        corner_kwargs["hist_kwargs"] = hist_1d_kwargs
        corner.corner(chains_2, truths=truths, range=my_range, 
                     fig=fig, labels=labels, **corner_kwargs)

        # Make a textbox
        if my_range is None or len(my_range) <= 2:
            fs = 14
        else:
            fs = 24
        plt.text(0.75, 0.75, "Training data", fontsize=fs, color="blue", 
                transform=plt.gcf().transFigure)
        plt.text(0.75, 0.65, "Normalizing flow", fontsize=fs, color="red", 
                transform=plt.gcf().transFigure)

        plt.savefig(name, bbox_inches="tight")
        plt.close()
    
    def generate_nf_samples(self):
        """
        Generate samples from the NF model.
        """
        nf_samples = []
        use_flowjax = self.nf_kwargs.get("use_flowjax", "False") == "True"
        
        if use_flowjax:
            # flowJAX sampling
            key = jr.key(123)
            keys = jr.split(key, self.N_samples)
            
            @jax.jit
            def sample_fn(sample_key):
                return self.flow.sample(sample_key, (1,)).flatten()
            
            # Vectorize over keys
            nf_samples_jax = jax.vmap(sample_fn)(keys)
            nf_samples = np.array(nf_samples_jax)
        else:
            # glasflow sampling
            with torch.no_grad():
                for _ in range(self.N_samples):
                    value = self.flow.sample(1).cpu().numpy().flatten()
                    nf_samples.append(value)
            nf_samples = np.array(nf_samples)
        
        # Apply inverse scaling if scaler was used during training
        if self.scaler is not None:
            nf_samples = self.scaler.inverse_transform(nf_samples)
        
        # Apply inverse log transform if log was taken during training
        if self.nf_kwargs.get("take_log_lambda", "False") == "True":
            # Only apply to lambda parameters (last 1 or 2 columns)
            use_tilde = self.nf_kwargs.get("use_tilde", "False") == "True"
            source_type = self.nf_kwargs.get("source_type", "bns")
            
            if source_type == "bns":
                if use_tilde:
                    # lambda_tilde, delta_lambda_tilde are last 2 columns
                    nf_samples[:, -2:] = np.exp(nf_samples[:, -2:])
                else:
                    # lambda_1, lambda_2 are last 2 columns
                    nf_samples[:, -2:] = np.exp(nf_samples[:, -2:])
            elif source_type == "nsbh":
                # lambda_2 is last column
                nf_samples[:, -1] = np.exp(nf_samples[:, -1])
        
        return nf_samples
    
    def generate_nf_samples_for_test(self, N_test_samples: int):
        """
        Generate samples from the NF model for testing purposes.
        """
        print(f"Generating {N_test_samples} samples from the NF model for testing")
        nf_samples = []
        use_flowjax = self.nf_kwargs.get("use_flowjax", "False") == "True"
        
        if use_flowjax:
            # flowJAX sampling
            key = jr.key(456)  # Different seed for test
            keys = jr.split(key, N_test_samples)
            
            @jax.jit
            def sample_fn(sample_key):
                return self.flow.sample(sample_key, (1,)).flatten()
            
            # Vectorize over keys
            nf_samples_jax = jax.vmap(sample_fn)(keys)
            nf_samples = np.array(nf_samples_jax)
        else:
            # glasflow sampling
            with torch.no_grad():
                for _ in range(N_test_samples):
                    value = self.flow.sample(1).cpu().numpy().flatten()
                    nf_samples.append(value)
            nf_samples = np.array(nf_samples)
        
        # Apply inverse scaling if scaler was used during training
        if self.scaler is not None:
            nf_samples = self.scaler.inverse_transform(nf_samples)
        
        # Apply inverse log transform if log was taken during training
        if self.nf_kwargs.get("take_log_lambda", "False") == "True":
            # Only apply to lambda parameters (last 1 or 2 columns)
            use_tilde = self.nf_kwargs.get("use_tilde", "False") == "True"
            source_type = self.nf_kwargs.get("source_type", "bns")
            
            if source_type == "bns":
                if use_tilde:
                    # lambda_tilde, delta_lambda_tilde are last 2 columns
                    nf_samples[:, -2:] = np.exp(nf_samples[:, -2:])
                else:
                    # lambda_1, lambda_2 are last 2 columns
                    nf_samples[:, -2:] = np.exp(nf_samples[:, -2:])
            elif source_type == "nsbh":
                # lambda_2 is last column
                nf_samples[:, -1] = np.exp(nf_samples[:, -1])
        
        # Print the ranges to the screen:
        for i in range(nf_samples.shape[1]):
            lower, upper = np.min(nf_samples[:, i]), np.max(nf_samples[:, i])
            print(f"Parameter {i} range: [{lower:.3f}, {upper:.3f}]")
        
        print(f"Generating {N_test_samples} samples from the NF model for testing DONE")
        return nf_samples
    
    def get_training_samples(self):
        """
        Get training samples in the same format as NF samples by using nf_kwargs["names"].
        """
        parameter_names = self.nf_kwargs.get("names", [])
        print(f"Constructing training samples for parameters: {parameter_names}")
        
        # FIXME: this is a small typo/bug and is now fixed, but old NFs might not have this yet
        # Handle backwards compatibility between 'q' and 'mass_ratio'
        parameter_names = parameter_names.copy()  # Don't modify original list
        for i, param_name in enumerate(parameter_names):
            if param_name == "mass_ratio" and "mass_ratio" not in self.training_data:
                if "q" in self.training_data:
                    parameter_names[i] = "q"
                elif "m1" in self.training_data and "m2" in self.training_data:
                    # We'll handle this case below
                    pass
                else:
                    raise KeyError(f"Cannot find mass ratio parameter. Expected 'mass_ratio' but found neither 'q' nor 'm1'/'m2' in training data keys: {list(self.training_data.keys())}")
            elif param_name == "q" and "q" not in self.training_data:
                if "mass_ratio" in self.training_data:
                    parameter_names[i] = "mass_ratio"
                elif "m1" in self.training_data and "m2" in self.training_data:
                    # We'll handle this case below
                    pass
                else:
                    raise KeyError(f"Cannot find mass ratio parameter. Expected 'q' but found neither 'mass_ratio' nor 'm1'/'m2' in training data keys: {list(self.training_data.keys())}")
        
        # Build training samples by iterating over the parameter names in order
        training_columns = []
        for param_name in parameter_names:
            if param_name in self.training_data:
                training_columns.append(self.training_data[param_name])
            elif param_name == "mass_ratio" and "m1" in self.training_data and "m2" in self.training_data:
                # Calculate mass_ratio = m2/m1 on the fly
                mass_ratio = self.training_data["m2"] / self.training_data["m1"]
                training_columns.append(mass_ratio)
            elif param_name == "q" and "m1" in self.training_data and "m2" in self.training_data:
                # Calculate q = m2/m1 on the fly
                q = self.training_data["m2"] / self.training_data["m1"]
                training_columns.append(q)
            elif param_name == "chirp_mass_source" and "m1" in self.training_data and "m2" in self.training_data:
                # Calculate chirp_mass_source from component masses using bilby
                chirp_mass_source = component_masses_to_chirp_mass(self.training_data["m1"], self.training_data["m2"])
                training_columns.append(chirp_mass_source)
            else:
                raise KeyError(f"Parameter '{param_name}' not found in training data and cannot be computed from available data. Available keys: {list(self.training_data.keys())}")
        
        result = np.column_stack(training_columns)
        print(f"Training samples shape: {result.shape}")
        return result

def main():
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="Evaluate a normalizing flow model")
    parser.add_argument("model_path", help="Path to the model directory")
    parser.add_argument("--test-only", action="store_true", 
                       help="Only test if model can generate samples (quick test)")
    parser.add_argument("--n-samples", type=int, default=10_000,
                       help="Number of samples to generate for evaluation")
    parser.add_argument("--n-test-samples", type=int, default=10_000,
                       help="Number of samples for quick test")
    
    args = parser.parse_args()
    
    try:
        if args.test_only:
            # Quick test - just see if model can generate samples
            print(f"Quick test for model: {args.model_path}")
            checker = CheckerUnconditional(args.model_path, N_samples=args.n_test_samples)
            
            # Try to generate samples
            samples = checker.generate_nf_samples_for_test(args.n_test_samples)
            
            print(f"✓ Success - Shape: {samples.shape}")
            print(f"  Ranges: {[f'[{samples[:, i].min():.3f}, {samples[:, i].max():.3f}]' for i in range(samples.shape[1])]}")
            
            # Check for issues
            if np.any(np.isnan(samples)) or np.any(np.isinf(samples)):
                print("✗ Warning: Found NaN or infinite values")
                return False
                
            return True
        else:
            # Full evaluation
            print(f"Full evaluation for model: {args.model_path}")
            checker = CheckerUnconditional(args.model_path, N_samples=args.n_samples)
            checker.check_unconditional_model()
            print("✓ Full evaluation completed successfully")
            return True
            
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)