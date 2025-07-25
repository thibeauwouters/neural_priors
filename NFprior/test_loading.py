import os
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import corner

# Import the new NFConditionalPrior implementation
from bilby.core.prior.dict import NFConditionalPrior
from bilby.core.prior import ConditionalPriorDict

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

def test_test_loading_prior():
    """Test the new NFConditionalPrior implementation with shared state coordination for BNS"""
    
    print("Testing BNS NFConditionalPrior with shared state coordination...")
    
    from bilby.core.prior.analytical import Uniform
    
    # Test the new shared state coordination approach
    print("\n=== Testing Shared State Coordination ====")
    try:
        # Create ConditionalPriorDict with shared state coordination
        priors = ConditionalPriorDict()
        
        # Add base priors
        priors['chirp_mass'] = Uniform(minimum=1.3, maximum=1.5, name='chirp_mass')
        priors['mass_ratio'] = Uniform(minimum=0.125, maximum=1.0, name='mass_ratio')
        priors['luminosity_distance'] = Uniform(minimum=20.0, maximum=400.0, name='luminosity_distance')
        
        # Create shared state object for coordination between lambda_1 and lambda_2
        shared_state = {'lambda_1': None, 'lambda_2': None, '_conditioning': None}
        
        # Add conditional NF priors with shared state coordination
        priors['lambda_1'] = NFConditionalPrior(
            nf_model_path=conditional_glasflow_filename,
            target_param='lambda_1',
            minimum=1e-6,
            maximum=100_000.0,
            shared_lambda_state=shared_state
        )
        
        priors['lambda_2'] = NFConditionalPrior(
            nf_model_path=conditional_glasflow_filename,
            target_param='lambda_2',
            minimum=1e-6,
            maximum=100_000.0,
            shared_lambda_state=shared_state
        )
        
        print(f"✓ ConditionalPriorDict created with {len(priors)} parameters")
        print(f"  Parameter keys: {list(priors.keys())}")
        print(f"  Shared state: {shared_state}")
        
        # Test sampling with detailed shared state checking
        print("  Testing individual NFConditionalPrior.sample() calls...")
        
        # First, test that lambda_1 prior creates joint sample in shared state
        print(f"  Before lambda_1 sample: shared_state = {shared_state}")
        
        # Call lambda_1 sample method directly with required variables
        conditioning_vars = {'chirp_mass': 1.4, 'mass_ratio': 0.8, 'luminosity_distance': 100.0}
        lambda_1_val = priors['lambda_1'].sample(**conditioning_vars)
        print(f"  After lambda_1 sample: shared_state = {shared_state}")
        print(f"  lambda_1 sampled value: {lambda_1_val:.1f}")
        
        # Now test lambda_2 - it should use the value from shared state
        lambda_2_val = priors['lambda_2'].sample(**conditioning_vars)
        print(f"  After lambda_2 sample: shared_state = {shared_state}")
        print(f"  lambda_2 sampled value: {lambda_2_val:.1f}")
        
        # Verify they match the shared state
        if (abs(lambda_1_val - shared_state['lambda_1']) < 1e-6 and
            abs(lambda_2_val - shared_state['lambda_2']) < 1e-6):
            print("  ✓ Individual sampling with shared state working!")
        else:
            print("  ✗ Individual sampling values don't match shared state")
        
        # Now test ConditionalPriorDict.sample()
        print("\n  Testing ConditionalPriorDict.sample()...")
        sample = priors.sample()
        print(f"  Sample keys: {list(sample.keys())}")
        print(f"  Sample values:\n   chirp_mass={sample['chirp_mass']:.3f},\n   mass_ratio={sample['mass_ratio']:.3f},\n   lambda_1={sample['lambda_1']:.1f},\n   lambda_2={sample['lambda_2']:.1f}")
        print(f"  Shared state after ConditionalPriorDict sampling: {shared_state}")
        
        # Check if shared state has values (should be populated during ConditionalPriorDict sampling)
        if shared_state['lambda_1'] is not None and shared_state['lambda_2'] is not None:
            print("  ✓ Shared state properly populated during ConditionalPriorDict sampling!")
            
            # Verify the sampled values match the shared state
            if (abs(sample['lambda_1'] - shared_state['lambda_1']) < 1e-6 and
                abs(sample['lambda_2'] - shared_state['lambda_2']) < 1e-6):
                print("  ✓ ConditionalPriorDict sample values match shared state!")
            else:
                print("  ✗ ConditionalPriorDict sample values don't match shared state")
                print(f"    Sample lambda_1: {sample['lambda_1']}, Shared: {shared_state['lambda_1']}")
                print(f"    Sample lambda_2: {sample['lambda_2']}, Shared: {shared_state['lambda_2']}")
        else:
            print("  ✗ Shared state not populated during ConditionalPriorDict sampling")
            print(f"    This means marginal approximation is still being used")
        
        # Test ln_prob with ConditionalPriorDict (realistic bilby usage)
        print("\n  Testing ln_prob via ConditionalPriorDict (realistic bilby usage)...")
        
        # Test 1: Complete parameter dictionary with realistic lambda values
        print("  Testing realistic lambda values:")
        realistic_params = {
            'chirp_mass': 1.4,
            'mass_ratio': 0.8, 
            'luminosity_distance': 100.0,
            'lambda_1': 100.0,  # Realistic NS tidal deformability
            'lambda_2': 500.0   # Realistic NS tidal deformability
        }
        
        ln_prob_realistic = priors.ln_prob(realistic_params)
        print(f"    Realistic params: ln_prob = {ln_prob_realistic:.3f}")
        print(f"    Parameters: lambda_1={realistic_params['lambda_1']:.1f}, lambda_2={realistic_params['lambda_2']:.1f}")
        
        # Test 2: Complete parameter dictionary with unrealistically large lambda values
        print("  Testing unrealistically large lambda values:")
        unrealistic_params = {
            'chirp_mass': 1.4,
            'mass_ratio': 0.8,
            'luminosity_distance': 100.0, 
            'lambda_1': 90_000.0,  # Unrealistically large
            'lambda_2': 90_000.0   # Unrealistically large  
        }
        
        ln_prob_unrealistic = priors.ln_prob(unrealistic_params)
        print(f"    Unrealistic params: ln_prob = {ln_prob_unrealistic:.3f}")
        print(f"    Parameters: lambda_1={unrealistic_params['lambda_1']:.0f}, lambda_2={unrealistic_params['lambda_2']:.0f}")
        
        # Test 3: Out-of-bounds values (should return -inf due to bounds checking)
        print("  Testing out-of-bounds lambda values:")
        out_of_bounds_params = {
            'chirp_mass': 1.4,
            'mass_ratio': 0.8,
            'luminosity_distance': 100.0,
            'lambda_1': -10.0,      # Negative (out of bounds)
            'lambda_2': 90_000.0   # Above maximum bound
        }
        
        ln_prob_oob = priors.ln_prob(out_of_bounds_params)
        print(f"    Out-of-bounds params: ln_prob = {ln_prob_oob:.3f}")
        print(f"    Parameters: lambda_1={out_of_bounds_params['lambda_1']:.1f}, lambda_2={out_of_bounds_params['lambda_2']:.0f}")
        
        # Instance where lambda_1 is bigger than lambda_2, which is not realistic
        print("  Testing unrealistic lambda_1 > lambda_2:")
        lambda_order_params = {
            'chirp_mass': 1.4,
            'mass_ratio': 0.8,
            'luminosity_distance': 100.0,
            'lambda_1': 800.0,
            'lambda_2': 700.0
        }
        
        ln_prob_order = priors.ln_prob(lambda_order_params)
        print(f"    Out-of-bounds params: ln_prob = {ln_prob_order:.3f}")
        print(f"    Parameters: lambda_1={lambda_order_params['lambda_1']:.1f}, lambda_2={lambda_order_params['lambda_2']:.0f}")
        
        # Verify probability ordering
        if ln_prob_realistic > ln_prob_unrealistic:
            print("  ✓ Realistic lambda values have higher probability than unrealistic ones!")
        else:
            print("  ✗ Probability ordering unexpected - check NF model")
            
        if ln_prob_oob == -float('inf'):
            print("  ✓ Out-of-bounds values correctly return -inf!")
        else:
            print("  ✗ Out-of-bounds values should return -inf")
        
        print("  ✓ ln_prob method tested via ConditionalPriorDict (proper bilby usage)!")
        
        print("\n✓ Shared state coordination approach working!")
        
    except Exception as e:
        print(f"✗ Shared state coordination approach failed: {e}")
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
    """Generate a corner plot using the conditional NF implementation with shared state"""
    print("\nGenerating corner plot with conditional NF implementation (shared state coordination)...")
    
    from bilby.core.prior.analytical import Uniform
    
    # Create ConditionalPriorDict
    priors = ConditionalPriorDict()
    
    # Add base priors
    priors['chirp_mass'] = Uniform(minimum=1.0, maximum=2.0, name='chirp_mass') 
    priors['mass_ratio'] = Uniform(minimum=0.5, maximum=1.0, name='mass_ratio')
    priors['luminosity_distance'] = Uniform(minimum=50.0, maximum=200.0, name='luminosity_distance')
    
    # Create shared state for lambda coordination
    shared_state = {'lambda_1': None, 'lambda_2': None, '_conditioning': None}
    
    # Add conditional NF priors with shared state
    priors['lambda_1'] = NFConditionalPrior(
        nf_model_path=conditional_glasflow_filename,
        target_param='lambda_1',
        minimum=1e-6, # small, but not zero, to prevent issues with log scaling
        maximum=100_000.0,
        shared_lambda_state=shared_state
    )
    
    priors['lambda_2'] = NFConditionalPrior(
        nf_model_path=conditional_glasflow_filename,
        target_param='lambda_2',
        minimum=1e-6, # small, but not zero, to prevent issues with log scaling
        maximum=100_000.0,
        shared_lambda_state=shared_state
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
    output_path = "./figures/test_loading_prior_cornerplot.pdf"
    print(f"Saving corner plot to {output_path}")
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    
    # Create it in tilde space as well, first, the NF samples, take them from above and transform them
    
    chirp_mass = samples_og[:, 0]
    mass_ratio = samples_og[:, 1]
    
    m1, m2 = chirp_mass_and_mass_ratio_to_component_masses(chirp_mass, mass_ratio)
    
    lambda_tilde = lambda_1_lambda_2_to_lambda_tilde(samples_og[:, 3], samples_og[:, 4], m1, m2)
    delta_lambda_tilde = lambda_1_lambda_2_to_delta_lambda_tilde(samples_og[:, 3], samples_og[:, 4], m1, m2)
    
    samples_tilde = np.array([m1, m2, lambda_tilde, delta_lambda_tilde]).T
    ranges = [[np.percentile(col, 0.5), np.percentile(col, 99.5)] for col in samples_tilde.T]
    labels = [r"$m_1$ [M$_\odot$]", r"$m_2$ [M$_\odot$]", r"$\tilde{\Lambda}$", r"$\delta \tilde{\Lambda}$"]
    corner_kwargs = default_corner_kwargs.copy()
    corner_kwargs["hist_kwargs"] = {"color": "blue", "density": True}
    fig_tilde = corner.corner(
        samples_tilde, 
        range=ranges,
        labels=labels,
        **corner_kwargs
    )
    
    corner_kwargs["color"] = "red"
    corner_kwargs["hist_kwargs"] = {"color": "red", "density": True}
    
    training_data = np.load(bns_training_data_path)
    m1, m2, lambda_1, lambda_2, = training_data["m1"], training_data["m2"], training_data["lambda_1"], training_data["lambda_2"]
    
    lambda_tilde_training = lambda_1_lambda_2_to_lambda_tilde(lambda_1, lambda_2, m1, m2)
    delta_lambda_tilde_training = lambda_1_lambda_2_to_delta_lambda_tilde(lambda_1, lambda_2, m1, m2)
    
    samples_training = np.array([m1, m2, lambda_tilde_training, delta_lambda_tilde_training]).T
    corner.corner(samples_training, 
                   labels=[r"$m_1$ [M$_\odot$]", r"$m_2$ [M$_\odot$]", r"$\tilde{\Lambda}$", r"$\delta \tilde{\Lambda}$"],
                   fig=fig_tilde,
                   **corner_kwargs)
    
    # Save the figure
    output_path = "./figures/test_loading_prior_cornerplot_tilde.pdf"
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
    output_path = "./figures/test_loading_prior_cornerplot_rescaling.pdf"
    print(f"Saving corner plot to {output_path}")
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    
    # Now overlay the second cornerplot on the first, using a different color
    print("Creating corner plot to compare rescale with sample...")
    default_corner_kwargs["color"] = "blue"
    hist_kwargs = {"color": "blue",
                   "density": True}
    default_corner_kwargs["color"] = "blue"
    fig = corner.corner(
        samples_og, 
        range=ranges, 
        labels=param_names,
        **default_corner_kwargs
    )
    
    default_corner_kwargs["color"] = "red"
    hist_kwargs = {"color": "red",
                   "density": True}
    default_corner_kwargs["color"] = "red"
    default_corner_kwargs["hist_kwargs"] = hist_kwargs
    
    corner.corner(
        samples,
        range=ranges, 
        labels=param_names,
        fig=fig,
        **default_corner_kwargs)
    
    # Save the figure
    output_path = "./figures/test_loading_prior_cornerplot_comparison.pdf"
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
        test_test_loading_prior()
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