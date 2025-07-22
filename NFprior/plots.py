# TODO: will deprecate this

# """
# Make some diagnosis plots for the NF flow priors. This is isolated from the training script just to keep things cleaner.
# """

# import os
# import numpy as np
# import matplotlib.pyplot as plt
# import corner
# import json
# import copy
# import torch

# params = {"axes.grid": True,
#         "text.usetex" : False,
#         "font.family" : "serif",
#         "ytick.color" : "black",
#         "xtick.color" : "black",
#         "axes.labelcolor" : "black",
#         "axes.edgecolor" : "black",
#         # "font.serif" : ["Computer Modern Serif"],
#         "xtick.labelsize": 16,
#         "ytick.labelsize": 16,
#         "axes.labelsize": 16,
#         "legend.fontsize": 16,
#         "legend.title_fontsize": 16,
#         "figure.titlesize": 16}
# plt.rcParams.update(params)

# # Improved corner kwargs
# default_corner_kwargs = dict(bins=40, 
#                         smooth=1., 
#                         show_titles=False,
#                         label_kwargs=dict(fontsize=16),
#                         title_kwargs=dict(fontsize=16), 
#                         color="blue",
#                         # quantiles=[],
#                         # levels=[0.9],
#                         plot_density=True, 
#                         plot_datapoints=False, 
#                         fill_contours=True,
#                         max_n_ticks=4, 
#                         min_n_ticks=3,
#                         truth_color = "red",
#                         save=False)


# def convert_tilde(params: dict):
#     lambda_1, lambda_2 = params['lambda_1'], params['lambda_2']
#     q = params['q']
#     eta = q / (1 + q)**2
#     lambda_plus = lambda_1 + lambda_2
#     lambda_minus = lambda_1 - lambda_2
#     lambda_tilde = 8 / 13 * (
#         (1 + 7 * eta - 31 * eta**2) * lambda_plus +
#         (1 - 4 * eta)**0.5 * (1 + 9 * eta - 11 * eta**2) * lambda_minus)

#     delta_lambda_tilde = 1 / 2 * (
#         (1 - 4 * eta) ** 0.5 * (1 - 13272 / 1319 * eta + 8944 / 1319 * eta**2) *
#         lambda_plus + (1 - 15910 / 1319 * eta + 32850 / 1319 * eta ** 2 +
#                        3380 / 1319 * eta ** 3) * lambda_minus)

#     print(f"Lambda tilde ranges from {np.min(lambda_tilde):.2f} to {np.max(lambda_tilde):.2f}")
#     print(f"Delta lambda tilde ranges from {np.min(delta_lambda_tilde):.2f} to {np.max(delta_lambda_tilde):.2f}")

#     params['lambda_tilde'] = lambda_tilde
#     params['delta_lambda_tilde'] = delta_lambda_tilde

#     return params

# def make_cornerplot(chains_1: np.array, 
#                     chains_2: np.array,
#                     name: str,
#                     my_range: list[float] = None,
#                     truths: list[float] = None):
#     """
#     Plot a cornerplot of the true data samples and the NF samples
#     Note: the shape use is a bit inconsistent below, watch out.
#     """

#     # The training data:
#     corner_kwargs = copy.deepcopy(default_corner_kwargs)
#     hist_1d_kwargs = {"density": True, "color": "blue"}
#     corner_kwargs["color"] = "blue"
#     corner_kwargs["hist_kwargs"] = hist_1d_kwargs
#     fig = corner.corner(chains_1, range=my_range, truths=truths, labels=None, **corner_kwargs)

#     # The data from the normalizing flow
#     corner_kwargs["color"] = "red"
#     hist_1d_kwargs = {"density": True, "color": "red"}
#     corner_kwargs["hist_kwargs"] = hist_1d_kwargs
#     corner.corner(chains_2,  truths=truths, range=my_range, fig=fig, **corner_kwargs)

#     # Make a textbox just because that makes the plot cooler
#     fs = 32
#     plt.text(0.75, 0.75, "Training data", fontsize = fs, color = "blue", transform = plt.gcf().transFigure)
#     plt.text(0.75, 0.65, "Normalizing flow", fontsize = fs, color = "red", transform = plt.gcf().transFigure)

#     plt.savefig(name, bbox_inches = "tight")
#     plt.close()

# def check_conditional_bns_model(path: str,
#                             m1_value: float = 1.6,
#                             m2_value: float = 1.4,
#                             N_samples: int = 1_000):
#     """
#     Load the NF model and check the 2D Lambdas distribution conditioned on the given masses.

#     Args:
#         path (str): Path pointing to the directory where the model is stored.
#         m1_value (float): Mass 1 value to condition on.
#         m2_value (float): Mass 2 value to condition on.
#     """
    
#     from glasflow.flows import RealNVP
    
#     # Ensure that masses are in the right order
#     m1_value_ = max(m1_value, m2_value)
#     m2_value_ = min(m1_value, m2_value)
    
#     m1_value = float(m1_value_)
#     m2_value = float(m2_value_)
    
#     # Load the NF weights
#     nf_path = os.path.join(path, "model.pt")
#     nf_kwargs_path = os.path.join(path, "model_kwargs.json")
    
#     with open(nf_kwargs_path, "r") as f:
#         nf_kwargs = json.load(f)
    
#     # TODO: fix this with kwargs if retrained! And make it more general
#     flow = RealNVP(
#             n_inputs=2, # nf_kwargs["n_inputs"]
#             n_transforms=nf_kwargs["n_transforms"],
#             n_conditional_inputs=2, # nf_kwargs["n_conditional_inputs"]
#             n_neurons=nf_kwargs["n_neurons"],
#             batch_norm_between_transforms=True,
#         )
    
#     print(f"Loading in the NF")
#     flow.load_state_dict(torch.load(nf_path, map_location=torch.device('cpu')))
#     flow.eval()
#     flow.compile()
#     print(f"Loading in the NF DONE")
    
#     # FIXME: this should be loaded from the NF kwargs
#     eos_samples_filename = "../data/eos/eos_samples.npz"
#     data = np.load(eos_samples_filename)
    
#     masses_EOS, Lambdas_EOS = data["masses_EOS"], data["Lambdas_EOS"]
    
#     # Load the Lambdas from the training data by sampling
#     lambda_1_train = np.zeros(N_samples)
#     lambda_2_train = np.zeros(N_samples)
    
#     random_idx = np.random.choice(len(masses_EOS), N_samples, replace=False)
    
#     for i in range(N_samples):
#         idx = random_idx[i]
#         m, l = masses_EOS[idx], Lambdas_EOS[idx]
        
#         # Interpolate on the given masses to get the corresponding Lambdas
#         lambda_1_value = np.interp(m1_value, m, l)
#         lambda_2_value = np.interp(m2_value, m, l)
        
#         lambda_1_train[i] = lambda_1_value
#         lambda_2_train[i] = lambda_2_value
        
#     training_samples = np.array([lambda_1_train, lambda_2_train]).T
        
#     # Get samples from the conditional NF
#     u = torch.tensor([[m1_value, m2_value]], dtype=torch.float32)  # Conditional input for the NF
#     nf_samples = []
    
#     with torch.no_grad():
#         for i in range(N_samples):
#             value = flow.sample(1, conditional=u).cpu().numpy().flatten()
#             if nf_kwargs["take_log_lambda"] == "True":
#                 # If the NF was trained on log Lambdas, we need to exponentiate them
#                 value = np.exp(value)
#             if i == 0:
#                 print("value")
#                 print(value)
#             nf_samples.append(value)
#     nf_samples = np.array(nf_samples)
    
#     print("np.shape(training_samples)")
#     print(np.shape(training_samples))
    
#     print("np.shape(nf_samples)")
#     print(np.shape(nf_samples))
    
#     # Print the ranges: 
#     lambda_1_nf, lambda_2_nf = nf_samples[:, 0], nf_samples[:, 1]
    
#     print(f"Lambda_1 training data range: [{np.min(lambda_1_train):.1f}, {np.max(lambda_1_train):.1f}]")
#     print(f"Lambda_1 NF samples range: [{np.min(lambda_1_nf):.1f}, {np.max(lambda_1_nf):.1f}]")
    
#     print(f"Lambda_2 training data range: [{np.min(lambda_2_train):.1f}, {np.max(lambda_2_train):.1f}]")
#     print(f"Lambda_2 NF samples range: [{np.min(lambda_2_nf):.1f}, {np.max(lambda_2_nf):.1f}]")
    
#     # Determine range from data but ensure positive
#     lambda_1_lower = min(np.min(lambda_1_train), np.min(lambda_1_nf))
#     lambda_1_lower = max(lambda_1_lower, 0.0)  # Ensure non-negative
#     lambda_1_upper = max(np.max(lambda_1_train), np.max(lambda_1_nf))
    
#     lambda_2_lower = min(np.min(lambda_2_train), np.min(lambda_2_nf))
#     lambda_2_lower = max(lambda_2_lower, 0.0)  # Ensure non-negative
#     lambda_2_upper = max(np.max(lambda_2_train), np.max(lambda_2_nf))
    
#     my_range = [[lambda_1_lower, lambda_1_upper], [lambda_2_lower, lambda_2_upper]]
    
#     name = f"./figures/check_bns/m1_{m1_value:.2f}_m2_{m2_value:.2f}.pdf"
#     print(f"Saving cornerplot to {name}")
#     make_cornerplot(training_samples, nf_samples, name, my_range=my_range)
    
#     print("DONE")
    

# def check_conditional_nsbh_model(path: str,
#                                  m2_value: float = 1.4,
#                                  N_samples: int = 10_000):
#     """
#     Load the NSBH conditional NF model and compare Lambda_2 marginal distributions.
    
#     Args:
#         path (str): Path pointing to the directory where the NSBH model is stored.
#         m2_value (float): Mass 2 value (NS mass) to condition on.
#         N_samples (int): Number of samples for comparison.
#     """
    
#     from glasflow.flows.autoregressive import MaskedAffineAutoregressiveFlow
    
#     # Load the NF weights and configuration
#     nf_path = os.path.join(path, "model.pt")
#     nf_kwargs_path = os.path.join(path, "model_kwargs.json")
    
#     with open(nf_kwargs_path, "r") as f:
#         nf_kwargs = json.load(f)
    
#     # Initialize the flow with correct architecture for NSBH
#     flow = MaskedAffineAutoregressiveFlow(
#         n_inputs=nf_kwargs["n_inputs"],  # Should be 1
#         n_conditional_inputs=nf_kwargs["n_conditional_inputs"],  # Should be 1  
#         n_transforms=nf_kwargs["n_transforms"],
#         n_neurons=nf_kwargs["n_neurons"],
#         n_blocks_per_transform=nf_kwargs["n_blocks_per_transform"]
#     )
    
#     print(f"Loading NSBH conditional NF model from {nf_path}")
#     flow.load_state_dict(torch.load(nf_path, map_location=torch.device('cpu')))
#     flow.eval()
#     print(f"NSBH model loaded successfully")
    
#     # Load EOS data for training comparison
#     eos_samples_filename = nf_kwargs["eos_samples_filename"]
#     if not os.path.exists(eos_samples_filename):
#         # Fallback to relative path
#         eos_samples_filename = "/Users/Woute029/Documents/Code/projects/eos_source_classification/eos_source_classification/data/eos/eos_samples.npz"
    
#     data = np.load(eos_samples_filename)
#     masses_EOS, Lambdas_EOS = data["masses_EOS"], data["Lambdas_EOS"]
    
#     # Generate training data Lambda_2 values for the given m2_value
#     lambda_2_train = np.zeros(N_samples)
#     random_idx = np.random.choice(len(masses_EOS), N_samples, replace=False)
    
#     for i in range(N_samples):
#         idx = random_idx[i]
#         m, l = masses_EOS[idx], Lambdas_EOS[idx]
#         lambda_2_value = np.interp(m2_value, m, l)
        
#         # TODO: perhaps create a plot of these to see the full curve, diagnose anything bad happening?
#         # Check if not negative:
#         if lambda_2_value < 0:
#             print(f"Warning: Negative Lambda_2 value {lambda_2_value:.2f} for m2 = {m2_value:.1f} M☉ at index {i}. Skipping this sample.")
#             continue
        
#         lambda_2_train[i] = lambda_2_value
    
#     # Generate samples from the conditional NF
#     u = torch.tensor([[m2_value]], dtype=torch.float32)  # Conditional input
#     nf_samples = []
    
#     with torch.no_grad():
#         for i in range(N_samples):
#             value = flow.sample(1, conditional=u).cpu().numpy().flatten()[0]
#             if nf_kwargs["take_log_lambda"] == "True":
#                 value = np.exp(value)  # Convert back from log space
#             nf_samples.append(value)
    
#     nf_samples = np.array(nf_samples)
    
#     # Create marginal comparison plot
#     plt.figure(figsize=(10, 6))
    
#     # Plot histograms
#     bins = np.linspace(min(np.min(lambda_2_train), np.min(nf_samples)), 
#                       max(np.max(lambda_2_train), np.max(nf_samples)), 50)
    
#     # Print the ranges: 
#     print(f"Lambda_2 training data range: [{np.min(lambda_2_train):.1f}, {np.max(lambda_2_train):.1f}]")
#     print(f"Lambda_2 NF samples range: [{np.min(nf_samples):.1f}, {np.max(nf_samples):.1f}]")
    
#     plt.hist(lambda_2_train, bins=bins, alpha=0.7, density=True, 
#              label='Training Data', color='blue', edgecolor='black')
#     plt.hist(nf_samples, bins=bins, alpha=0.7, density=True, 
#              label='NF Samples', color='red', edgecolor='black')
    
#     plt.xlabel(r'$\Lambda_2$')
#     plt.ylabel('Density')
#     plt.title(f'NSBH Conditional Prior: $p(\Lambda_2 | m_2 = {m2_value:.1f} M_\odot)$')
#     plt.legend()
#     plt.grid(True, alpha=0.3)
    
#     # Save the plot
#     os.makedirs("./figures", exist_ok=True)
#     save_name = f"./figures/check_nsbh/m2_{m2_value:.1f}.pdf"
#     plt.savefig(save_name, bbox_inches="tight")
#     print(f"Saved NSBH marginal comparison plot to {save_name}")
#     plt.close()
    
#     # Print statistics
#     print(f"\nStatistics for m_2 = {m2_value:.1f} M_☉:")
#     print(f"Training data Lambda_2: mean={np.mean(lambda_2_train):.1f}, std={np.std(lambda_2_train):.1f}")
#     print(f"NF samples Lambda_2: mean={np.mean(nf_samples):.1f}, std={np.std(nf_samples):.1f}")
#     print(f"Range - Training: [{np.min(lambda_2_train):.1f}, {np.max(lambda_2_train):.1f}]")
#     print(f"Range - NF: [{np.min(nf_samples):.1f}, {np.max(nf_samples):.1f}]")
    
# def main():
#     # Test BNS conditional model
#     path_bns = "./models/conditional_bns/"
#     check_conditional_bns_model(path_bns, m1_value=1.8, m2_value=1.2, N_samples=10_000)
    
#     # Test NSBH conditional model  
#     path_nsbh = "./models/conditional_nsbh/"
#     print("\n" + "="*50)
#     print("Testing NSBH conditional model")
#     print("="*50)
#     check_conditional_nsbh_model(path_nsbh, m2_value=1.2, N_samples=10_000)
    
# if __name__ == "__main__":
#     main()