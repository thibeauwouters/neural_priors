"""
Evaluate the performance of the flows, by plotting some comparison cornerplots and also calculating eg KL divergences.
"""

import os
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import corner
import json
import copy
import torch
from scipy.special import kl_div
from scipy.stats import gaussian_kde

from glasflow.flows import RealNVP

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
    
class CheckerBNS:
    
    def __init__(self,
                 path: str,
                 N_samples: int = 10_000,
                 N_masses: int = 5):
        """
        Initialize the checker for the BNS conditional model.
        
        Args:
            path (str): Path to the directory where the model is stored.
            N_samples (int): Number of samples to generate for comparison for a single mass pair.
            N_masses (int): Number of mass pairs to sample from the training data.
        """
        self.path = path
        self.N_samples = N_samples
        self.N_masses = N_masses
        
        # Load the model and data
        self.flow, self.nf_kwargs, self.masses_EOS, self.Lambdas_EOS = self.load_model_and_data()
    
    def load_model_and_data(self):
        """
        Load the NF model and EOS data for the BNS case.
        
        Returns:
            flow (RealNVP): The loaded normalizing flow model.
            nf_kwargs (dict): The configuration parameters for the NF.
            masses_EOS (np.ndarray): Masses from the EOS samples.
            Lambdas_EOS (np.ndarray): Corresponding Lambdas from the EOS samples.
        """
        nf_path = os.path.join(self.path, "model.pt")
        nf_kwargs_path = os.path.join(self.path, "model_kwargs.json")
        
        with open(nf_kwargs_path, "r") as f:
            nf_kwargs = json.load(f)
        self.nf_kwargs = nf_kwargs
        
        flow = RealNVP(
                n_inputs=2,
                n_conditional_inputs=2,
                n_transforms=self.nf_kwargs["n_transforms"],
                n_neurons=self.nf_kwargs["n_neurons"],
                batch_norm_between_transforms=True)
        
        print(f"Loading in the NF from {nf_path}")
        flow.load_state_dict(torch.load(nf_path, map_location=torch.device('cpu')))
        flow.eval()
        flow.compile()
        
        # Load the EOS posterior samples from jester from which we have created the training data
        eos_samples_filename = nf_kwargs["eos_samples_filename"]
        print(f"Comparing against the EOS samples taken from {eos_samples_filename}")
        data = np.load(eos_samples_filename)
        masses_EOS, Lambdas_EOS = data["masses_EOS"], data["Lambdas_EOS"]
        
        return flow, nf_kwargs, masses_EOS, Lambdas_EOS

    def check_conditional_bns_model(self):
        """
        Load the NF model and check the 2D Lambdas distribution conditioned on the given masses.

        Args:
            path (str): Path pointing to the directory where the model is stored.
            m1_value (float): Mass 1 value to condition on.
            m2_value (float): Mass 2 value to condition on.
        """
        
        mass_1 = np.linspace(1.25, 2.0, self.N_masses)
        mass_2 = np.linspace(1.0, 2.0, self.N_masses)

        m1_grid, m2_grid = np.meshgrid(mass_1, mass_2)

        # Ensure m1 >= m2
        mask = m2_grid <= m1_grid
        mass_grid = np.column_stack((
            m1_grid[mask].ravel(),
            m2_grid[mask].ravel()
        ))

        print(f"mass_grid: len = {len(mass_grid)}")
        print(mass_grid)
        
        # Do the check for each pair:
        for m1_value, m2_value in tqdm.tqdm(mass_grid, desc="Checking BNS conditional model"):
            self.check_single_bns(m1_value, m2_value)
    
    def check_single_bns(self, m1_value: float, m2_value: float):
        """
        Check the conditional BNS model by sampling from the training data and the NF.
        Args:
            m1_value (float): Mass 1 value to condition on.
            m2_value (float): Mass 2 value to condition on.
        """
        # Load the Lambdas from the training data by sampling
        lambda_1_train = np.zeros(self.N_samples)
        lambda_2_train = np.zeros(self.N_samples)
        
        counter = 0
        
        # Create the validation set from the EOS samples
        while counter < self.N_samples:
            idx = np.random.choice(len(self.masses_EOS), 1)
            m, l = self.masses_EOS[idx][0], self.Lambdas_EOS[idx][0]
            
            # Interpolate on the given masses to get the corresponding Lambdas
            lambda_1_value = np.interp(m1_value, m, l)
            lambda_2_value = np.interp(m2_value, m, l)
            
            # Sanity check it:
            if lambda_1_value < 0.0 or lambda_2_value < 0.0:
                continue
            else:
                lambda_1_train[counter] = lambda_1_value
                lambda_2_train[counter] = lambda_2_value
                counter += 1
                
        training_samples = np.array([lambda_1_train, lambda_2_train]).T
            
        # Condition on those masses to sample from the conditional NF
        u = torch.tensor([[m1_value, m2_value]], dtype=torch.float32)
        nf_samples = []
        
        with torch.no_grad():
            for _ in range(self.N_samples):
                value = self.flow.sample(1, conditional=u).cpu().numpy().flatten()
                if self.nf_kwargs["take_log_lambda"] == "True":
                    value = np.exp(value)
                nf_samples.append(value)
        nf_samples = np.array(nf_samples)
        
        print("np.shape(training_samples)")
        print(np.shape(training_samples))
        
        print("np.shape(nf_samples)")
        print(np.shape(nf_samples))
        
        # Print the ranges: 
        lambda_1_nf, lambda_2_nf = nf_samples[:, 0], nf_samples[:, 1]
        
        # Determine range from data but ensure positive, use quantiles
        lambda_1_lower, lambda_1_upper = np.quantile(lambda_1_nf, [0.01, 0.99])
        lambda_2_lower, lambda_2_upper = np.quantile(lambda_2_nf, [0.01, 0.99])
        
        my_range = [[lambda_1_lower, lambda_1_upper], [lambda_2_lower, lambda_2_upper]]
        
        name = f"./figures/check_bns/m1_{m1_value:.2f}_m2_{m2_value:.2f}.pdf"
        print(f"Saving cornerplot to {name}")
        make_cornerplot(training_samples, nf_samples, name, my_range=my_range)
        
        # # Compute the KL divergence
        # all_samples = np.concatenate([training_samples.flatten(), nf_samples.flatten()])
        # min_value, max_value = np.min(all_samples), np.max(all_samples)
        
        # p_kde = gaussian_kde(training_samples.flatten())
        # q_kde = gaussian_kde(nf_samples.flatten())

        # x_grid = np.linspace(min_value, max_value, 1000)

        # p_vals = p_kde(x_grid)
        # q_vals = q_kde(x_grid)

        # kl = np.sum(p_vals * np.log(p_vals / q_vals)) * (x_grid[1] - x_grid[0])

        # print("kl")        
        # print(kl)        

class CheckerNSBH:
    """
    Checker for the NSBH conditional model.
    """
    
    def __init__(self, path: str, N_samples: int = 10_000, N_masses: int = 5):
        """
        Initialize the checker for the NSBH conditional model.
        
        Args:
            path (str): Path to the directory where the model is stored.
            N_samples (int): Number of samples to generate for comparison for a single mass pair.
            N_masses (int): Number of mass pairs to sample from the training data.
        """
        self.path = path
        self.N_samples = N_samples
        self.N_masses = N_masses
        
        # Load the model and data
        self.flow, self.nf_kwargs, self.masses_EOS, self.Lambdas_EOS = self.load_model_and_data()
        
    def load_model_and_data(self):
        """
        Load the NF model and EOS data for the NSBH case.
        
        Returns:
            flow (MaskedAffineAutoregressiveFlow): The loaded normalizing flow model.
            nf_kwargs (dict): The configuration parameters for the NF.
            masses_EOS (np.ndarray): Masses from the EOS samples.
            Lambdas_EOS (np.ndarray): Corresponding Lambdas from the EOS samples.
        """
        from glasflow.flows.autoregressive import MaskedAffineAutoregressiveFlow
        
        nf_path = os.path.join(self.path, "model.pt")
        nf_kwargs_path = os.path.join(self.path, "model_kwargs.json")
        
        with open(nf_kwargs_path, "r") as f:
            nf_kwargs = json.load(f)
        
        flow = MaskedAffineAutoregressiveFlow(
            n_inputs=1,  # fixed for NSBH case
            n_conditional_inputs=1,  # fixed for NSBH case
            n_transforms=nf_kwargs["n_transforms"],
            n_neurons=nf_kwargs["n_neurons"],
            n_blocks_per_transform=nf_kwargs["n_blocks_per_transform"]
        )
        print(f"Loading in the NF from {nf_path}")
        flow.load_state_dict(torch.load(nf_path, map_location=torch.device('cpu')))
        flow.eval()
        flow.compile()
        print(f"NSBH model loaded successfully")
        
        # Load the EOS posterior samples from jester from which we have created the training data
        eos_samples_filename = nf_kwargs["eos_samples_filename"]
        print(f"Comparing against the EOS samples taken from {eos_samples_filename}")
        data = np.load(eos_samples_filename)
        masses_EOS, Lambdas_EOS = data["masses_EOS"], data["Lambdas_EOS"]
        
        return flow, nf_kwargs, masses_EOS, Lambdas_EOS
    
    def check_conditional_nsbh_model(self):
        """
        Iterate over an array of m2 values and check the conditional NSBH model.
        """
        # Generate a grid of m2 values
        m2_values = np.linspace(1.0, 2.0, self.N_masses)

        for m2_value in tqdm.tqdm(m2_values, desc="Checking NSBH conditional model"):
            self.check_single_nsbh(m2_value)
    
    def check_single_nsbh(self, m2_value: float = 1.4):
        """
        Load the NSBH conditional NF model and compare Lambda_2 marginal distributions.
        
        Args:
            m2_value (float): Mass 2 value (NS mass) to condition on.
        """
        # Generate training data Lambda_2 values for the given m2_value
        lambda_2_train = np.zeros(self.N_samples)
        
        counter = 0
        
        while counter < self.N_samples:
            idx = np.random.choice(len(self.masses_EOS), 1)
            m, l = self.masses_EOS[idx][0], self.Lambdas_EOS[idx][0]
            lambda_2_value = np.interp(m2_value, m, l)
            
            # Check if not negative:
            if lambda_2_value < 0:
                continue
            
            lambda_2_train[counter] = lambda_2_value
            counter += 1
        # Generate samples from the conditional NF
        u = torch.tensor([[m2_value]], dtype=torch.float32)  # Conditional input
        nf_samples = []
        with torch.no_grad():
            for _ in range(self.N_samples):
                value = self.flow.sample(1, conditional=u).cpu().numpy().flatten()[0]
                if self.nf_kwargs["take_log_lambda"] == "True":
                    value = np.exp(value)
                nf_samples.append(value)
        nf_samples = np.array(nf_samples)
        
        # Compare the marginal distributions
        plt.figure(figsize=(10, 6))
        # Plot histograms
        lambda_2_lower, lambda_2_upper = np.quantile(lambda_2_train, [0.01, 0.99])
        bins = np.linspace(lambda_2_lower, lambda_2_upper, 50)
        plt.hist(lambda_2_train, bins=bins, alpha=0.7, density=True,
                 label='Training Data', color='blue', edgecolor='black')
        plt.hist(nf_samples, bins=bins, alpha=0.7, density=True,
                 label='NF Samples', color='red', edgecolor='black')
        
        plt.xlabel(r'$\Lambda_2$')
        plt.ylabel('Density')
        
        plt.title(f'NSBH Conditional Prior: $p(\Lambda_2 | m_2 = {m2_value:.1f} M_\odot)$')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save the plot
        name = f"./figures/check_nsbh/m2_{m2_value:.2f}.pdf"
        os.makedirs("./figures/check_nsbh", exist_ok=True)
        plt.savefig(name, bbox_inches="tight")
        plt.close()
        
        # TODO: KL divergence calculation


    
def main():
    bns_checker = CheckerBNS("./models/conditional_bns/", N_samples=10_000, N_masses=5)
    print("\n" + "="*50)
    print("Testing BNS conditional model")
    print("="*50)
    bns_checker.check_conditional_bns_model()
    
    # nsbh_checker = CheckerNSBH("./models/conditional_nsbh/", N_samples=10_000, N_masses=5)
    # print("\n" + "="*50)
    # print("Testing NSBH conditional model")
    # print("="*50)
    # nsbh_checker.check_conditional_nsbh_model()
    
if __name__ == "__main__":
    main()