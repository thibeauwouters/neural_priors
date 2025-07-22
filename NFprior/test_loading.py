import os
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import corner

# Import only the new NFConditionalPrior implementation
from bilby.core.prior.dict import NFConditionalPrior
from bilby.core.prior import ConditionalPriorDict

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

# Conditional model files
bns_model_dir = "./models/conditional_bns/"
conditional_glasflow_filename = os.path.join(bns_model_dir, "model.pt")

bns_glasflow_filename = os.path.join(bns_model_dir, "model.pt")
bns_training_data_path = os.path.join(bns_model_dir, "training_data.npz")

# NSBH model files
nsbh_model_dir = "./models/conditional_nsbh/"
nsbh_glasflow_filename = os.path.join(nsbh_model_dir, "model.pt")
nsbh_training_data_path = os.path.join(nsbh_model_dir, "training_data.npz")

def test_bns_conditional_prior():
    """Test the new NFConditionalPrior implementation with ConditionalPriorDict"""
    
    print("Testing BNS implementation...")
    
    from bilby.core.prior.analytical import Uniform
    
    # Create ConditionalPriorDict
    priors = ConditionalPriorDict()
    
    # Add base priors
    priors['chirp_mass'] = Uniform(minimum=1.3, maximum=1.5, name='chirp_mass')
    priors['mass_ratio'] = Uniform(minimum=0.125, maximum=1.0, name='mass_ratio')
    priors['luminosity_distance'] = Uniform(minimum=20.0, maximum=400.0, name='luminosity_distance')
    
    # Add conditional NF priors
    priors['lambda_1'] = NFConditionalPrior(
        nf_model_path=conditional_glasflow_filename,
        target_param='lambda_1',
        minimum=0.0,
        maximum=10000.0
    )
    
    priors['lambda_2'] = NFConditionalPrior(
        nf_model_path=conditional_glasflow_filename,
        target_param='lambda_2',
        minimum=0.0,
        maximum=10000.0
    )
    
    print(f"✓ ConditionalPriorDict created with {len(priors)} parameters")
    print(f"  Parameter keys: {list(priors.keys())}")
    
    # Test sampling
    print("\n1. Testing sampling...")
    try:
        sample = priors.sample()
        print(f"  Sample keys: {list(sample.keys())}")
        print(f"  Sample values:\n   chirp_mass={sample['chirp_mass']:.3f},\n   mass_ratio={sample['mass_ratio']:.3f},\n   lambda_1={sample['lambda_1']:.1f},\n   lambda_2={sample['lambda_2']:.1f}")
    except Exception as e:
        print(f"  ✗ Sampling failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test rescaling
    print("\n2. Testing rescaling...")
    rescaling_failed = False
    try:
        unit_cube = np.random.uniform(0, 1, len(priors))
        keys = list(priors.keys())
        rescaled = priors.rescale(keys, unit_cube)
        rescaled = np.array(rescaled)  # Ensure it's an array
        print(f"  Unit cube shape: {unit_cube.shape}")
        print(f"  Rescaled shape: {rescaled.shape}")
        print(f"  Rescaled values (first 5): {rescaled[:5]}")
    except Exception as e:
        print(f"  ✗ Rescaling failed: {e}")
        import traceback
        traceback.print_exc()
        rescaling_failed = True
    
    # Test log probability
    print("\n3. Testing log probability...")
    ln_prob_failed = False
    try:
        if not rescaling_failed:
            # Construct an example array
            param_dict = {key: val for key, val in zip(keys, rescaled)}
            print(f"  Parameter dictionary: {param_dict}")
            print(f"  Parameter keys: {list(param_dict.keys())}")
            print(f"  Parameter values: {list(param_dict.values())}")
            
            # Compute ln prob
            ln_prob = priors.ln_prob(param_dict)
            print(f"  Log probability: {ln_prob}")
        else:
            print("  Skipped due to rescaling failure")
            ln_prob_failed = True
    except Exception as e:
        print(f"  ✗ Log probability failed: {e}")
        import traceback
        traceback.print_exc()
        ln_prob_failed = True
    
    if rescaling_failed or ln_prob_failed:
        print("\n✗ New NFConditionalPrior test had failures!")
        raise RuntimeError("BNS test had failures")
    else:
        print("\n✓ New NFConditionalPrior test completed successfully!")

def test_corner_plot_new_implementation():
    """Generate a corner plot using the new implementation"""
    print("\nGenerating corner plot with new implementation...")
    
    from bilby.core.prior.analytical import Uniform
    
    # Create ConditionalPriorDict
    priors = ConditionalPriorDict()
    
    # Add base priors
    priors['chirp_mass'] = Uniform(minimum=1.0, maximum=2.0, name='chirp_mass') 
    priors['mass_ratio'] = Uniform(minimum=0.5, maximum=1.0, name='mass_ratio')
    priors['luminosity_distance'] = Uniform(minimum=50.0, maximum=200.0, name='luminosity_distance')
    
    # Add conditional NF priors
    priors['lambda_1'] = NFConditionalPrior(
        nf_model_path=conditional_glasflow_filename,
        target_param='lambda_1',
        minimum=0.0,
        maximum=10_000.0
    )
    
    priors['lambda_2'] = NFConditionalPrior(
        nf_model_path=conditional_glasflow_filename,
        target_param='lambda_2',
        minimum=0.0,
        maximum=10_000.0
    )
    
    # Generate samples
    N_samples = 10_000
    
    print(f"Generating {N_samples} samples...")
    samples = []
    for i in tqdm.tqdm(range(N_samples), desc="Sampling . . ."):
        sample = priors.sample()
        samples.append([sample['chirp_mass'], sample['mass_ratio'], 
                       sample['luminosity_distance'], sample['lambda_1'], sample['lambda_2']])
    
    samples_og = np.array(samples)
    param_names = ['chirp_mass', 'mass_ratio', 'luminosity_distance', 'lambda_1', 'lambda_2']
    
    # Create ranges for corner plot
    ranges = [[np.percentile(col, 0.5), np.percentile(col, 99.5)] for col in samples_og.T]
    
    # Create corner plot
    print("Creating corner plot...")
    fig = corner.corner(
        samples_og, 
        range=ranges, 
        labels=param_names,
        **default_corner_kwargs
    )
    
    # Save the figure
    output_path = "./figures/bns_conditional_prior_cornerplot.pdf"
    print(f"Saving corner plot to {output_path}")
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    
    print("Also creating the cornerplot with the rescale method to check that")
    
    # Do the same using rescaling rather than sample to test that functionality
    samples = []
    keys = list(priors.keys())
    for i in tqdm.tqdm(range(N_samples), desc="Rescaling unit cube samples . . ."):
        unit_cube = np.random.uniform(0, 1, len(priors))
        rescaled = priors.rescale(keys, unit_cube)
        samples.append([rescaled[0], rescaled[1], rescaled[2], rescaled[3], rescaled[4]])
    samples = np.array(samples)
    
    print(np.shape(samples))
    
    # Create ranges for corner plot
    ranges = [[np.percentile(col, 0.5), np.percentile(col, 99.5)] for col in samples.T]
    
    # Make the corner plot with rescaled values
    print("Creating corner plot with rescaled values...")
    fig_rescaled = corner.corner(
        samples,
        range=ranges, 
        labels=param_names,
        **default_corner_kwargs
    )
    
    # Save the figure
    output_path = "./figures/bns_conditional_prior_cornerplot_rescaling.pdf"
    print(f"Saving corner plot to {output_path}")
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    
    # Now overlay the second cornerplot on the first, using a different color
    default_corner_kwargs["color"] = "red"
    hist_kwargs = {"color": "red",
                   "density": True}
    default_corner_kwargs["color"] = "red"
    default_corner_kwargs["hist_kwargs"] = hist_kwargs
    
    print("Creating corner plot to compare rescale with sample...")
    fig = corner.corner(
        samples_og, 
        range=ranges, 
        labels=param_names,
        **default_corner_kwargs
    )
    
    corner.corner(
        samples,
        range=ranges, 
        labels=param_names,
        fig=fig,
        **default_corner_kwargs)
    
    # Save the figure
    output_path = "./figures/bns_conditional_prior_cornerplot_comparison.pdf"
    print(f"Saving corner plot to {output_path}")
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    
    print("✓ Corner plot generated successfully!")

def test_nsbh_conditional_prior():
    """Test and plot the NSBH conditional prior against training data"""
    print("\n" + "="*60)
    print("TESTING NSBH CONDITIONAL PRIOR")
    print("="*60)
    
    # Load training data
    print("Loading NSBH training data...")
    try:
        training_data = np.load(nsbh_training_data_path)
        m2_training = training_data['m2']  # Concatenated NS masses
        lambda2_training = training_data['lambda_2']  # Corresponding lambdas (in log space)
        print(f"✓ Loaded {len(m2_training)} training samples")
        print(f"  Mass range: {m2_training.min():.3f} - {m2_training.max():.3f} M☉")
        print(f"  Log Lambda range: {lambda2_training.min():.3f} - {lambda2_training.max():.3f}")
    except Exception as e:
        print(f"✗ Failed to load training data: {e}")
        return
    
    # Create a proper prior setup for testing NSBH
    from bilby.core.prior.analytical import Uniform
    
    # For NSBH, we still need the base parameters that get converted to masses
    priors = ConditionalPriorDict()
    
    # Add base priors that will be used to derive the NS mass
    priors['chirp_mass'] = Uniform(minimum=1.0, maximum=3.0, name='chirp_mass')
    priors['mass_ratio'] = Uniform(minimum=0.1, maximum=1.0, name='mass_ratio')  # q = m2/m1, for NSBH m2 is NS
    priors['luminosity_distance'] = Uniform(minimum=1.0, maximum=500.0, name='luminosity_distance')
    
    # Add the conditional lambda prior for the NS
    priors['lambda_2'] = NFConditionalPrior(
        nf_model_path=nsbh_glasflow_filename,
        target_param='lambda_2',
        minimum=0.0,  # Will be converted from log space
        maximum=10000.0
    )
    
    print(f"✓ Created NSBH priors with parameters: {list(priors.keys())}")
    
    # Test basic functionality
    print("\n1. Testing NSBH sampling...")
    try:
        sample = priors.sample()
        # For NSBH, we sample chirp_mass, mass_ratio, dL, then derive m_2 and sample lambda_2
        print(f"  Sample keys: {list(sample.keys())}")
        print(f"  Sample: chirp_mass={sample['chirp_mass']:.3f}, mass_ratio={sample['mass_ratio']:.3f}, lambda_2={sample['lambda_2']:.1f}")
    except Exception as e:
        print(f"  ✗ Sampling failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Generate samples for comparison
    N_samples = 10_000
    print(f"\n2. Generating {N_samples} NSBH samples for comparison...")
    
    nf_lambdas = []
    
    for i in tqdm.tqdm(range(N_samples), desc="Sampling . . ."):
        
        try:
            sample = priors.sample()
            nf_lambdas.append(sample['lambda_2'])
        except Exception as e:
            print(f"  ✗ Sampling failed at iteration {i}: {e}")
            import traceback
            traceback.print_exc()
            break
    
    nf_lambdas = np.array(nf_lambdas)
    
    print(f"Range of sampled lambda_2: {nf_lambdas.min():.1f} - {nf_lambdas.max():.1f}")
    print(f"✓ Generated {len(nf_lambdas)} NF samples correctly")
    
def main():
    """Test the new NFConditionalPrior implementation"""
    print("="*60)
    print("TESTING NFConditionalPrior")
    print("="*60)
    
    # Test the new implementation
    try:
        test_bns_conditional_prior()
        print("\n✓ Basic functionality test passed!")
    except Exception as e:
        print(f"\n✗ Basic functionality test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Generate corner plot
    try:
        test_corner_plot_new_implementation()
        print("\n✓ Corner plot generation passed!")
    except Exception as e:
        print(f"\n✗ Corner plot generation failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test NSBH conditional prior
    try:
        test_nsbh_conditional_prior()
        print("\n✓ NSBH conditional prior test passed!")
    except Exception as e:
        print(f"\n✗ NSBH conditional prior test failed: {e}")
        import traceback
        traceback.print_exc()
    
if __name__ == "__main__":
    main()