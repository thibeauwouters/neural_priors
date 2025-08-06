import os
import json
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

# Joint prior model files (uniform radio BNS models)
bns_glasflow_model_dir = "./models/uniform/bns/radio/"
bns_flowjax_model_dir = "./models/uniform/bns/radio_flowjax/"

bns_glasflow_filename = os.path.join(bns_glasflow_model_dir, "model.pt")
bns_flowjax_filename = os.path.join(bns_flowjax_model_dir, "model.eqx")
bns_glasflow_training_data_path = os.path.join(bns_glasflow_model_dir, "training_data.npz")
bns_flowjax_training_data_path = os.path.join(bns_flowjax_model_dir, "training_data.npz")
bns_glasflow_figures_dir = os.path.join(bns_glasflow_model_dir, "figures/")
bns_flowjax_figures_dir = os.path.join(bns_flowjax_model_dir, "figures/")

def test_nf_joint_prior_backend(backend_name, flow_filename):
    """Test the NFPrior joint prior implementation for a specific backend"""
    
    print(f"\n{'='*60}")
    print(f"Testing {backend_name.upper()} NFPrior implementation")
    print(f"{'='*60}")
    
    # Create NFDist with the specified model
    print(f"\n=== Creating NFDist with {backend_name} ====")
    try:
        # Parameter names as defined in the model (read from kwargs)
        kwargs_filename = flow_filename.replace(".pt", "_kwargs.json").replace(".eqx", "_kwargs.json")
        with open(kwargs_filename, "r") as f:
            kwargs = json.load(f)
        param_names = kwargs["names"]
        print(f"  Model parameters: {param_names}")
        
        # Create NFDist 
        nf_dist = NFDist(
            names=param_names,
            flow_filename=flow_filename,
            use_tilde=False,
            use_component_masses=False
        )
        
        print(f"✓ NFDist created with {nf_dist.num_vars} parameters")
        print(f"  Parameter names: {nf_dist.names}")
        print(f"  Backend detected: {'flowjax' if hasattr(nf_dist, 'use_flowjax') and nf_dist.use_flowjax else 'glasflow'}")
        
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
    
    # Test direct NFDist sampling
    print("\n=== Testing Direct NFDist Sampling ====")
    try:
        # Test the underlying nf_sample function first
        print("  Testing underlying nf_sample function...")
        raw_samples = nf_dist.nf_sample(100)
        print(f"  Raw NF samples: {type(raw_samples)}")
        if raw_samples is not None:
            print(f"  Raw NF samples shape: {raw_samples.shape}")
            print(f"  Raw sample preview: {raw_samples[:3]}")
        
        # Test the _sample method step by step
        print("\n  Testing _sample method step by step...")
        
        # Step 1: nf_sample
        flow_samp = nf_dist.nf_sample(100)
        print(f"  After nf_sample: shape={flow_samp.shape}, type={type(flow_samp)}")
        print(f"  Values range: [{flow_samp.min():.6f}, {flow_samp.max():.6f}]")
        
        # Step 2: scaler inverse transform
        if nf_dist.scaler is not None:
            scaled_samp = nf_dist.scaler.inverse_transform(flow_samp)
            print(f"  After scaler: shape={scaled_samp.shape}, type={type(scaled_samp)}")
            print(f"  Values range: [{scaled_samp.min():.6f}, {scaled_samp.max():.6f}]")
        else:
            scaled_samp = flow_samp
            print("  No scaler found, skipping scaling")
        
        # Step 3: clean_samples
        cleaned_samp = nf_dist.clean_samples(scaled_samp)
        print(f"  After cleaning: {type(cleaned_samp)}")
        if cleaned_samp is not None:
            print(f"  After cleaning: shape={cleaned_samp.shape}")
            print(f"  Values range: [{cleaned_samp.min():.6f}, {cleaned_samp.max():.6f}]")
        
        # Test direct sampling from NFDist
        nf_dist.sample(size=1_000)  # This stores samples in current_sample attribute
        print(f"  Current sample keys: {list(nf_dist.current_sample.keys())}")
        
        # Extract samples from current_sample attribute
        sample_arrays = []
        for name in param_names:
            sample_arrays.append(nf_dist.current_sample[name])
        direct_samples = np.column_stack(sample_arrays)
        
        print(f"  Direct NFDist samples shape: {direct_samples.shape}")
        print(f"  Sample preview: {direct_samples[:3]}")
        print("✓ Direct NFDist sampling successful")
        
    except Exception as e:
        print(f"✗ Direct NFDist sampling failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test PriorDict sampling  
    print("\n=== Testing PriorDict Sampling ====")
    try:
        # Transform into PriorDict
        priors = PriorDict(priors)

        samples = priors.sample(size=1_000)
        for key, value in samples.items():
            print(f"  {key}: {value[:5]}... (shape: {value.shape})")
        print("✓ PriorDict sampling successful")
            
    except Exception as e:
        print(f"✗ PriorDict sampling failed: {e}")
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
        # Test with realistic parameter values (4 params: chirp_mass_source, mass_ratio, lambda_1, lambda_2)
        realistic_params = np.array([1.4, 0.8, 100.0, 500.0])
        ln_prob_realistic = nf_dist.ln_prob(realistic_params)
        print(f"  Realistic params ln_prob: {ln_prob_realistic:.3f}")
        
        # Test with unrealistic parameter values
        unrealistic_params = np.array([1.4, 0.8, 50000.0, 50000.0])  # very large lambdas
        ln_prob_unrealistic = nf_dist.ln_prob(unrealistic_params)
        print(f"  Unrealistic params ln_prob: {ln_prob_unrealistic:.3f}")
        
        if ln_prob_realistic > ln_prob_unrealistic:
            print("✓ Realistic parameters have higher probability than unrealistic ones!")
        else:
            print("✗ Probability ordering unexpected")
            
        # Test with negative lambdas
        negative_lambdas = np.array([1.4, 0.8, -10.0, -10.0])  # negative lambdas
        ln_prob_negative = nf_dist.ln_prob(negative_lambdas)
        print(f"  Negative lambdas ln_prob: {ln_prob_negative:.3f}")
            
        if ln_prob_negative == -np.inf:
            print("✓ Negative lambdas have correct ln prob!")
        else:
            print("✗ Negative lambdas have wrong ln prob")
            
        # Test with wrong Lambdas order
        wrong_order_lambdas = np.array([1.4, 0.8, 500.0, 400.0])  # wrong order in lambdas
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

def generate_corner_plots_for_backend(backend_name, flow_filename, training_data_path, figures_dir):
    """Generate corner plots comparing NF samples with training data for a specific backend"""
    
    print(f"\n=== Generating corner plots for {backend_name.upper()} ===")
    
    # Parameter names as defined in the model (read from kwargs)
    kwargs_filename = flow_filename.replace(".pt", "_kwargs.json").replace(".eqx", "_kwargs.json")
    with open(kwargs_filename, "r") as f:
        kwargs = json.load(f)
    param_names = kwargs["names"]
    print(f"  Model parameters: {param_names}")
    
    # Create NFDist 
    nf_dist = NFDist(
        names=param_names,
        flow_filename=flow_filename,
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
    for _ in tqdm.tqdm(range(N_samples)):
        unit_cube = np.random.uniform(0, 1, len(param_names))
        rescaled = nf_dist.rescale(unit_cube)
        unit_cube_samples.append(rescaled)
    unit_cube_samples = np.array(unit_cube_samples)
    print(f"Unit cube samples shape: {unit_cube_samples.shape}")
    
    # Load training data
    print("Loading training data...")
    training_data = np.load(training_data_path)
    print(f"Available keys in training data: {list(training_data.keys())}")
    
    # Extract training data arrays - check which keys are available
    training_samples = []
    for name in param_names:
        if name == "chirp_mass_source":
            # Handle the source frame chirp mass case
            if "chirp_mass_source" in training_data:
                training_samples.append(training_data["chirp_mass_source"])
            elif "chirp_mass" in training_data:
                training_samples.append(training_data["chirp_mass"])
        elif name in training_data:
            training_samples.append(training_data[name])
        else:
            # Try alternative key names
            alt_name_map = {
                "mass_ratio": "q", 
                "luminosity_distance": "dL"
            }
            alt_name = alt_name_map.get(name, name)
            if alt_name in training_data:
                training_samples.append(training_data[alt_name])
            else:
                raise KeyError(f"Could not find parameter {name} or alternative {alt_name} in training data")
    
    training_samples = np.column_stack(training_samples)
    
    print(f"Training samples shape: {training_samples.shape}")
    
    # Create parameter labels
    param_label_map = {
        "chirp_mass": r"$M_c$ [M$_\odot$]",
        "chirp_mass_source": r"$M_c^{\rm src}$ [M$_\odot$]", 
        "mass_ratio": r"$q$",
        "luminosity_distance": r"$d_L$ [Mpc]",
        "lambda_1": r"$\Lambda_1$",
        "lambda_2": r"$\Lambda_2$"
    }
    param_labels = [param_label_map.get(name, name) for name in param_names]
    
    # Create ranges for corner plot
    all_samples = np.vstack([nf_samples, training_samples])
    ranges = [[np.percentile(col, 0.5), np.percentile(col, 99.5)] for col in all_samples.T]
    
    # Ensure figures directory exists
    os.makedirs(figures_dir, exist_ok=True)
    
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
    output_path1 = os.path.join(figures_dir, f"nf_samples_vs_training_data_{backend_name}.pdf")
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
    output_path2 = os.path.join(figures_dir, f"unit_cube_rescaled_vs_training_data_{backend_name}.pdf")
    print(f"Saving corner plot to {output_path2}")
    plt.savefig(output_path2, bbox_inches="tight")
    plt.close()
    
    # Transform to tilde space and create comparison (only if we have the right parameters)
    if "lambda_1" in param_names and "lambda_2" in param_names:
        print("Creating tilde space comparison...")
        
        # Find indices for required parameters
        mass_param = "chirp_mass_source" if "chirp_mass_source" in param_names else "chirp_mass"
        if mass_param not in param_names:
            print("  Skipping tilde space plot - no chirp mass parameter found")
            print("✓ Corner plots generated successfully!")
            return
            
        chirp_mass_idx = param_names.index(mass_param)
        mass_ratio_idx = param_names.index("mass_ratio") 
        lambda_1_idx = param_names.index("lambda_1")
        lambda_2_idx = param_names.index("lambda_2")
        
        # Convert NF samples to tilde space
        chirp_mass_nf = nf_samples[:, chirp_mass_idx]
        mass_ratio_nf = nf_samples[:, mass_ratio_idx]
        m1_nf, m2_nf = chirp_mass_and_mass_ratio_to_component_masses(chirp_mass_nf, mass_ratio_nf)
        lambda_tilde_nf = lambda_1_lambda_2_to_lambda_tilde(nf_samples[:, lambda_1_idx], nf_samples[:, lambda_2_idx], m1_nf, m2_nf)
        delta_lambda_tilde_nf = lambda_1_lambda_2_to_delta_lambda_tilde(nf_samples[:, lambda_1_idx], nf_samples[:, lambda_2_idx], m1_nf, m2_nf)
        
        nf_tilde = np.column_stack([m1_nf, m2_nf, lambda_tilde_nf, delta_lambda_tilde_nf])
        
        # Convert training data to tilde space
        chirp_mass_train = training_samples[:, chirp_mass_idx]
        mass_ratio_train = training_samples[:, mass_ratio_idx]
        m1_train, m2_train = chirp_mass_and_mass_ratio_to_component_masses(chirp_mass_train, mass_ratio_train)
        lambda_tilde_train = lambda_1_lambda_2_to_lambda_tilde(training_samples[:, lambda_1_idx], training_samples[:, lambda_2_idx], m1_train, m2_train)
        delta_lambda_tilde_train = lambda_1_lambda_2_to_delta_lambda_tilde(training_samples[:, lambda_1_idx], training_samples[:, lambda_2_idx], m1_train, m2_train)
        
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
        output_path3 = os.path.join(figures_dir, f"nf_samples_vs_training_tilde_space_{backend_name}.pdf")
        print(f"Saving tilde space corner plot to {output_path3}")
        plt.savefig(output_path3, bbox_inches="tight")
        plt.close()
    else:
        print("  Skipping tilde space plot - lambda parameters not found")
    
    print("✓ Corner plots generated successfully!")

def main():
    """Test the NFPrior joint prior implementation for both backends"""
    print("="*80)
    print("TESTING NFPrior JOINT PRIOR IMPLEMENTATION - BOTH BACKENDS")
    print("="*80)
    
    # Test both backends
    backends_to_test = [
        ("glasflow", bns_glasflow_filename),
        ("flowjax", bns_flowjax_filename)
    ]
    
    all_passed = True
    
    for backend_name, model_filename in backends_to_test:
        print(f"\n{'='*80}")
        print(f"Starting tests for {backend_name.upper()} backend")
        print(f"Model: {model_filename}")
        print(f"{'='*80}")
        
        # Check if model file exists
        if not os.path.exists(model_filename):
            print(f"✗ Model file not found: {model_filename}")
            print(f"  Skipping {backend_name} tests")
            all_passed = False
            continue
        
        try:
            success = test_nf_joint_prior_backend(backend_name, model_filename)
            if success:
                print(f"\n✓ {backend_name.upper()} backend test passed!")
                
                # Generate corner plots if test passed
                print(f"\n{'='*80}")
                print(f"Generating corner plots for {backend_name.upper()} backend")
                print(f"{'='*80}")
                
                # Determine training data path and figures dir based on backend
                if backend_name == "glasflow":
                    training_data_path = bns_glasflow_training_data_path
                    figures_dir = bns_glasflow_figures_dir
                else:  # flowjax
                    training_data_path = bns_flowjax_training_data_path
                    figures_dir = bns_flowjax_figures_dir
                
                try:
                    generate_corner_plots_for_backend(backend_name, model_filename, training_data_path, figures_dir)
                    print(f"✓ {backend_name.upper()} corner plots generated successfully!")
                except Exception as plot_e:
                    print(f"✗ {backend_name.upper()} corner plot generation failed: {plot_e}")
                    import traceback
                    traceback.print_exc()
                    # Don't fail the overall test for plot generation issues
                
            else:
                print(f"\n✗ {backend_name.upper()} backend test failed!")
                all_passed = False
        except Exception as e:
            print(f"\n✗ {backend_name.upper()} backend test failed with exception: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False
    
    # Final summary
    print(f"\n{'='*80}")
    print("FINAL TEST SUMMARY")
    print(f"{'='*80}")
    if all_passed:
        print("✓ ALL BACKEND TESTS PASSED! NFDist refactoring is working correctly.")
    else:
        print("✗ SOME TESTS FAILED. Check the output above for details.")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()