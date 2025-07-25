import os
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import corner

# Import the joint prior implementation
from bilby.core.prior.joint import NFPrior, NFDist
from bilby.core.prior.analytical import Uniform
from bilby.core.prior.dict import PriorDict

from bilby.gw.conversion import lambda_1_lambda_2_to_lambda_tilde, lambda_1_lambda_2_to_delta_lambda_tilde
from bilby.gw.conversion import chirp_mass_and_mass_ratio_to_component_masses

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
                        # color="blue",
                        # quantiles=[],
                        # levels=[0.9],
                        plot_density=True, 
                        plot_datapoints=False, 
                        fill_contours=True,
                        max_n_ticks=4, 
                        min_n_ticks=3,
                        truth_color = "red",
                        save=False)

# Joint prior model files (GW170817 radio BNS model)
bns_model_dir = "./models/GW170817/radio_bns/"
bns_glasflow_filename = os.path.join(bns_model_dir, "model.pt")
bns_training_data_path = os.path.join(bns_model_dir, "training_data.npz")
bns_figures_dir = os.path.join(bns_model_dir, "figures/")

def test_nf_joint_prior():
    """Test the NFPrior joint prior implementation for BNS"""
    
    print("Testing BNS NFPrior joint prior implementation...")
    
    # Create NFDist with the GW170817 radio BNS model
    print("\n=== Creating NFDist ====")
    try:
        # Parameter names as defined in the model
        param_names = ["chirp_mass", "mass_ratio", "luminosity_distance", "lambda_1", "lambda_2"]
        
        # Create NFDist 
        nf_dist = NFDist(
            names=param_names,
            flow_filename=bns_glasflow_filename,
            include_dL=True,
            use_tilde=False,
            use_component_masses=False
        )
        
        # Test its own sampling:
        samples = nf_dist.sample(size=10_000)
        
        print("samples")
        print(samples)

        
        print(f"✓ NFDist created with {nf_dist.num_vars} parameters")
        print(f"  Parameter names: {nf_dist.names}")
        
        # Create NFPrior objects for each parameter
        priors = {}
        for name in param_names:
            priors[name] = NFPrior(dist=nf_dist, name=name)
        
        print(f"✓ NFPrior objects created for parameters: {list(priors.keys())}")
        
    except Exception as e:
        print(f"✗ NFDist/NFPrior creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test basic sampling
    print("\n=== Testing Sampling ====")
    try:

        # Transform into PriorDict
        priors = PriorDict(priors)

        samples = priors.sample(size=1_000)
        for key, value in samples.items():
            print(f"  {key}: {value[:5]}... (shape: {value.shape})")
            
    except Exception as e:
        print(f"✗ Sampling failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test rescaling from unit hypercube
    print("\n=== Testing Rescaling ====")
    try:
        # Generate random unit hypercube sample
        unit_cube_sample = np.random.uniform(0, 1, len(param_names))
        print(f"  Unit cube sample: {unit_cube_sample}")
        
        # Rescale using NFDist
        rescaled_sample = nf_dist.rescale(unit_cube_sample)
        print(f"  Rescaled sample: {rescaled_sample}")
        
        print("✓ Rescaling successful")
        
    except Exception as e:
        print(f"✗ Rescaling failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test log probability evaluation
    print("\n=== Testing Log Probability ====")
    try:
        # Test with realistic parameter values
        realistic_params = np.array([1.4, 0.8, 100.0, 100.0, 500.0])  # chirp_mass, mass_ratio, dL, lambda_1, lambda_2
        ln_prob_realistic = nf_dist.ln_prob(realistic_params)
        print(f"  Realistic params ln_prob: {ln_prob_realistic:.3f}")
        
        # Test with unrealistic parameter values
        unrealistic_params = np.array([1.4, 0.8, 100.0, 50000.0, 50000.0])  # very large lambdas
        ln_prob_unrealistic = nf_dist.ln_prob(unrealistic_params)
        print(f"  Unrealistic params ln_prob: {ln_prob_unrealistic:.3f}")
        
        if ln_prob_realistic > ln_prob_unrealistic:
            print("✓ Realistic parameters have higher probability than unrealistic ones!")
        else:
            print("✗ Probability ordering unexpected")
            
        # Test with unrealistic parameter values
        negative_lambdas = np.array([1.4, 0.8, 100.0, -10.0, -10.0])  # negative lambdas
        ln_prob_negative = nf_dist.ln_prob(negative_lambdas)
        print(f"  Negative lambdas ln_prob: {ln_prob_negative:.3f}")
            
        # Test with unrealistic parameter values
        negative_lambdas = np.array([1.4, 0.8, 100.0, -10.0, -10.0])  # negative lambdas
        ln_prob_negative = nf_dist.ln_prob(negative_lambdas)
        print(f"  Negative lambdas ln_prob: {ln_prob_negative:.3f}")
            
        if ln_prob_negative == -np.inf:
            print("✓ Negative lambdas have correct ln prob!")
        else:
            print("✗ Negative lambdas have wrong ln prob")
            
        # Test with wrong Lambdas order
        wrong_order_lambdas = np.array([1.4, 0.8, 100.0, 500.0, 400.0])  # wrong order in lambdas
        ln_prob_wrong_order = nf_dist.ln_prob(wrong_order_lambdas)
        print(f"  Wrong order lambdas ln_prob: {ln_prob_wrong_order:.3f}")
            
        if ln_prob_wrong_order == -np.inf:
            print("✓ Wrong order lambdas have correct ln prob!")
        else:
            print("✗ Wrong order lambdas have wrong ln prob")
            
    except Exception as e:
        print(f"✗ Log probability evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n✓ NFPrior joint prior test completed successfully!")
    return True

def generate_corner_plots():
    """Generate corner plots comparing NF samples with training data"""
    
    print("Generating corner plots for NF joint prior...")
    
    # Create NFDist 
    param_names = ["chirp_mass", "mass_ratio", "luminosity_distance", "lambda_1", "lambda_2"]
    nf_dist = NFDist(
        names=param_names,
        flow_filename=bns_glasflow_filename,
        include_dL=True,
        use_tilde=False,
        use_component_masses=False
    )
    
    # Add the NFPriors
    priors = {}
    for name in param_names:
        priors[name] = NFPrior(dist=nf_dist, name=name)
    priors = PriorDict(priors)
    
    # Generate NF samples
    N_samples = 10_000
    print(f"Generating {N_samples} NF samples...")
    nf_samples_dict = priors.sample(size=N_samples)
    nf_samples = np.array(list(nf_samples_dict.values())).T
    print(f"NF samples shape: {nf_samples.shape}")
    
    # Generate unit cube rescaled samples
    print(f"Generating {N_samples} unit cube rescaled samples...")
    unit_cube_samples = []
    for _ in tqdm.tqdm(range(N_samples), desc="Rescaling unit cube samples"):
        unit_cube = np.random.uniform(0, 1, len(param_names))
        rescaled = nf_dist.rescale(unit_cube)
        unit_cube_samples.append(rescaled)
    unit_cube_samples = np.array(unit_cube_samples)
    print(f"Unit cube samples shape: {unit_cube_samples.shape}")
    
    # Load training data
    print("Loading training data...")
    training_data = np.load(bns_training_data_path)
    print(f"Available keys in training data: {list(training_data.keys())}")
    
    # Extract training data arrays - check which keys are available
    if 'chirp_mass' in training_data:
        training_samples = np.column_stack([
            training_data['chirp_mass'],
            training_data['mass_ratio'], 
            training_data['luminosity_distance'],
            training_data['lambda_1'],
            training_data['lambda_2']
        ])
    else:
        # Alternative key names
        training_samples = np.column_stack([
            training_data['mc'],  # chirp mass
            training_data['q'],   # mass ratio
            training_data['dL'],  # luminosity distance  
            training_data['lambda_1'],
            training_data['lambda_2']
        ])
    
    print(f"Training samples shape: {training_samples.shape}")
    
    # Create parameter labels
    param_labels = [r"$M_c$ [M$_\odot$]", r"$q$", r"$d_L$ [Mpc]", r"$\Lambda_1$", r"$\Lambda_2$"]
    
    # Create ranges for corner plot
    all_samples = np.vstack([nf_samples, training_samples])
    ranges = [[np.percentile(col, 0.5), np.percentile(col, 99.5)] for col in all_samples.T]
    
    # Plot 1: NF samples vs training data
    print("Creating corner plot: NF samples vs training data...")
    fig1 = corner.corner(
        nf_samples,
        range=ranges,
        labels=param_labels,
        color="blue",
        **default_corner_kwargs
    )
    
    # Overlay training data
    corner_kwargs_training = default_corner_kwargs.copy()
    corner_kwargs_training["color"] = "red"
    corner_kwargs_training["hist_kwargs"] = {"color": "red", "density": True}
    
    corner.corner(
        training_samples,
        range=ranges,
        labels=param_labels,
        fig=fig1,
        **corner_kwargs_training
    )
    
    # Save plot 1
    output_path1 = os.path.join(bns_figures_dir, "nf_samples_vs_training_data.pdf")
    print(f"Saving corner plot to {output_path1}")
    plt.savefig(output_path1, bbox_inches="tight")
    plt.close()
    
    # Plot 2: Unit cube rescaled vs training data  
    print("Creating corner plot: Unit cube rescaled vs training data...")
    fig2 = corner.corner(
        unit_cube_samples,
        range=ranges,
        labels=param_labels,
        color="green",
        **default_corner_kwargs
    )
    
    # Overlay training data
    corner.corner(
        training_samples,
        range=ranges,
        labels=param_labels,
        fig=fig2,
        **corner_kwargs_training
    )
    
    # Save plot 2
    output_path2 = os.path.join(bns_figures_dir, "unit_cube_rescaled_vs_training_data.pdf")
    print(f"Saving corner plot to {output_path2}")
    plt.savefig(output_path2, bbox_inches="tight")
    plt.close()
    
    # Transform to tilde space and create comparison
    print("Creating tilde space comparison...")
    
    # Convert NF samples to tilde space
    chirp_mass_nf = nf_samples[:, 0]
    mass_ratio_nf = nf_samples[:, 1]
    m1_nf, m2_nf = chirp_mass_and_mass_ratio_to_component_masses(chirp_mass_nf, mass_ratio_nf)
    lambda_tilde_nf = lambda_1_lambda_2_to_lambda_tilde(nf_samples[:, 3], nf_samples[:, 4], m1_nf, m2_nf)
    delta_lambda_tilde_nf = lambda_1_lambda_2_to_delta_lambda_tilde(nf_samples[:, 3], nf_samples[:, 4], m1_nf, m2_nf)
    
    nf_tilde = np.column_stack([m1_nf, m2_nf, lambda_tilde_nf, delta_lambda_tilde_nf])
    
    # Convert training data to tilde space
    if 'chirp_mass' in training_data:
        chirp_mass_train = training_data['chirp_mass']
        mass_ratio_train = training_data['mass_ratio']
    else:
        chirp_mass_train = training_data['mc']
        mass_ratio_train = training_data['q']
        
    m1_train, m2_train = chirp_mass_and_mass_ratio_to_component_masses(chirp_mass_train, mass_ratio_train)
    lambda_tilde_train = lambda_1_lambda_2_to_lambda_tilde(training_data['lambda_1'], training_data['lambda_2'], m1_train, m2_train)
    delta_lambda_tilde_train = lambda_1_lambda_2_to_delta_lambda_tilde(training_data['lambda_1'], training_data['lambda_2'], m1_train, m2_train)
    
    training_tilde = np.column_stack([m1_train, m2_train, lambda_tilde_train, delta_lambda_tilde_train])
    
    # Create tilde space plot
    tilde_labels = [r"$m_1$ [M$_\odot$]", r"$m_2$ [M$_\odot$]", r"$\tilde{\Lambda}$", r"$\delta\tilde{\Lambda}$"]
    
    all_tilde = np.vstack([nf_tilde, training_tilde])
    tilde_ranges = [[np.percentile(col, 0.5), np.percentile(col, 99.5)] for col in all_tilde.T]
    
    fig3 = corner.corner(
        nf_tilde,
        range=tilde_ranges,
        labels=tilde_labels,
        color="blue",
        **default_corner_kwargs
    )
    
    corner.corner(
        training_tilde,
        range=tilde_ranges,
        labels=tilde_labels,
        fig=fig3,
        **corner_kwargs_training
    )
    
    # Save tilde space plot
    output_path3 = os.path.join(bns_figures_dir, "nf_samples_vs_training_tilde_space.pdf")
    print(f"Saving tilde space corner plot to {output_path3}")
    plt.savefig(output_path3, bbox_inches="tight")
    plt.close()
    
    print("✓ Corner plots generated successfully!")

def main():
    """Test the NFPrior joint prior implementation and generate plots"""
    print("="*60)
    print("TESTING NFPrior JOINT PRIOR IMPLEMENTATION")
    print("="*60)
    
    # Test basic functionality
    try:
        success = test_nf_joint_prior()
        if success:
            print("\n✓ Basic functionality test passed!")
        else:
            print("\n✗ Basic functionality test failed!")
            return
    except Exception as e:
        print(f"\n✗ Basic functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Generate corner plots
    try:
        generate_corner_plots()
        print("\n✓ Corner plot generation passed!")
    except Exception as e:
        print(f"\n✗ Corner plot generation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()