import numpy as np
import matplotlib.pyplot as plt

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
    
mass_samples = sample_ns_mass_double_gaussian(10_000)

# Make a histogram
plt.hist(mass_samples, bins=100, density=True, alpha=0.5, label='Sampled Masses')
plt.savefig('./figures/double_gaussian_mass_samples.png')
plt.close()