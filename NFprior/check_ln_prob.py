#!/usr/bin/env python3
"""
Check the ln_prob calculation for NFDist class with Jacobian correction.
This script:
1. Loads a trained normalizing flow model
2. Generates samples from the flow
3. Computes ln_prob using the Bilby NFDist class
4. Creates scatter/corner plots colored by ln_prob values
5. Tests normalization using Monte Carlo integration

Example: python check_ln_prob.py models/uniform/bns/radio --n-samples 5000 --n-mc-samples 20000

NOTE: this was a quick sanity check coded up with Claude Code, not using it as official result or anything!
"""

import os
import sys
import corner
import numpy as np
import matplotlib.pyplot as plt
import json
import argparse
from pathlib import Path
from scipy.special import logsumexp

# Add bilby to path
sys.path.insert(0, "/Users/Woute029/Documents/Code/projects/eos_source_classification/bilby")

# Import bilby NFDist class
from bilby.core.prior.joint import NFDist, NFPrior
from bilby.core.prior import PriorDict

# Import flow evaluation tools
from evaluate_flows import CheckerUnconditional

def load_model_and_generate_samples(model_path: str, n_samples: int = 10000):
    """Load a flow model and generate samples similar to evaluate_flows.py"""
    print(f"Loading model from: {model_path}")
    
    # Use the existing checker to load the model
    checker = CheckerUnconditional(model_path, N_samples=n_samples)
    
    # Generate samples from the NF
    nf_samples = checker.generate_nf_samples()
    
    print(f"Generated {len(nf_samples)} samples with shape {nf_samples.shape}")
    return nf_samples

def create_bilby_nf_prior(model_path: str):
    """Create a bilby NFDist and NFPrior objects from the model path"""
    
    # Find the model file
    model_files = list(Path(model_path).glob("model.*"))
    if not model_files:
        raise FileNotFoundError(f"No model files found in {model_path}")
    
    model_file = str(model_files[0])  # Take the first model file found
    print(f"Using model file: {model_file}")
    
    # Load kwargs to get parameter names
    kwargs_file = model_file.replace(".pt", "_kwargs.json").replace(".eqx", "_kwargs.json")
    with open(kwargs_file, "r") as f:
        kwargs = json.load(f)
    
    parameter_names = kwargs["names"]
    use_tilde = kwargs.get("use_tilde", "False") == "True"
    use_component_masses = kwargs.get("use_component_masses", "False") == "True"
    
    print(f"Parameter names: {parameter_names}")
    print(f"Use tilde: {use_tilde}")
    print(f"Use component masses: {use_component_masses}")
    
    # Create NFDist object
    nf_dist = NFDist(
        names=parameter_names,
        flow_filename=model_file,
        use_tilde=use_tilde,
        use_component_masses=use_component_masses
    )
    
    # Create NFPrior objects for each parameter
    nf_priors = {}
    for param_name in parameter_names:
        nf_priors[param_name] = NFPrior(
            dist=nf_dist,
            name=param_name
        )
    
    # Create PriorDict
    prior_dict = PriorDict(nf_priors)
    
    return nf_dist, prior_dict

def compute_ln_prob_for_samples(nf_dist, samples):
    """Compute ln_prob for all samples using the NFDist"""
    print("Computing ln_prob for samples...")
    
    n_samples = len(samples)
    ln_probs = np.zeros(n_samples)
    
    # Compute ln_prob for each sample
    for i in range(n_samples):
        if i % 1000 == 0:
            print(f"  Processing sample {i}/{n_samples}")
        
        sample = samples[i]
        ln_probs[i] = nf_dist.ln_prob(sample)
    
    print(f"ln_prob range: [{np.min(ln_probs):.3f}, {np.max(ln_probs):.3f}]")
    print(f"Mean ln_prob: {np.mean(ln_probs):.3f}")
    print(f"Finite ln_probs: {np.sum(np.isfinite(ln_probs))}/{len(ln_probs)}")
    
    return ln_probs


def test_normalization_mc(nf_dist, 
                          parameter_names,
                          n_test_samples=100_000,
                          use_mc: bool = False,
                          use_uniform: bool = True):
    """Test if the probability density integrates to 1 using Monte Carlo integration"""
    print(f"\nTesting normalization using Monte Carlo with {n_test_samples} samples...")
    
    # Generate samples from the prior using the _sample method to get proper shape
    test_samples = nf_dist._sample(n_test_samples)
    
    print(f"Generated test samples shape: {test_samples.shape}")
    
    # Compute log probabilities
    ln_probs = np.array([nf_dist.ln_prob(sample) for sample in test_samples])
    
    # Filter finite log probabilities
    finite_mask = np.isfinite(ln_probs)
    finite_ln_probs = ln_probs[finite_mask]
    finite_samples = test_samples[finite_mask]
    
    print(f"Finite log probabilities: {len(finite_ln_probs)}/{len(ln_probs)}")
    
    if use_mc:
        
        if len(finite_ln_probs) == 0:
            print("❌ No finite log probabilities found!")
            return
        
        # Estimate the volume of parameter space using sample ranges
        sample_mins = np.percentile(finite_samples, 2.5, axis=0)  # 2.5th percentile
        sample_maxs = np.percentile(finite_samples, 97.5, axis=0)  # 97.5th percentile
        
        # Add some padding to ensure we cover the full range
        padding = (sample_maxs - sample_mins) * 0.1
        param_mins = sample_mins - padding
        param_maxs = sample_maxs + padding
        
        param_ranges = param_maxs - param_mins
        volume = float(np.prod(param_ranges))  # Convert to scalar
        
        print(f"Parameter ranges:")
        for i, param_name in enumerate(parameter_names):
            print(f"  {param_name}: [{param_mins[i]:.3f}, {param_maxs[i]:.3f}] -> range = {param_ranges[i]:.3f}")
        
        print(f"Estimated parameter space volume: {volume:.3e}")
        
        # Monte Carlo estimate using logsumexp for numerical stability
        # For importance sampling: ∫ p(x) dx ≈ (1/N) * Σ p(x_i) * volume_element
        # In log space: log(∫ p(x) dx) ≈ log(volume/N) + logsumexp(ln_probs)
        log_volume = np.log(volume)
        log_n_samples = np.log(len(finite_ln_probs))
        
        # Use logsumexp for numerical stability
        log_integral_estimate = log_volume - log_n_samples + logsumexp(finite_ln_probs)
        integral_estimate = np.exp(log_integral_estimate)
        
        print(f"Log integral estimate: {log_integral_estimate:.6f}")
        print(f"MC integral estimate: {integral_estimate:.6f}")
        
        if 0.5 <= integral_estimate <= 2.0:  # More lenient bounds for MC estimation
            print("✅ Normalization looks reasonable!")
        else:
            print("❌ Normalization may be incorrect!")
    
    if use_uniform:
        
        # Alternative test using uniform sampling in parameter space
        print("\nAlternative test using uniform sampling in parameter space...")
        
        n_uniform = n_test_samples
        
        # Generate uniform samples using the sample bounds
        bounds_low = np.percentile(finite_samples, 2.5, axis=0)  # 2.5th percentile
        bounds_high = np.percentile(finite_samples, 97.5, axis=0)  # 97.5th percentile

        uniform_samples = np.random.uniform(
            low=bounds_low,
            high=bounds_high,
            size=(n_uniform, len(parameter_names))
        )
        
        print(f"Generated {n_uniform} uniform samples")
        
        # Compute ln_prob for uniform samples
        uniform_ln_probs = []
        for i, sample in enumerate(uniform_samples):
            if i % (n_uniform // 10) == 0:
                print(f"  Processing uniform sample {i}/{n_uniform}")
            ln_prob = nf_dist.ln_prob(sample)
            uniform_ln_probs.append(ln_prob)
        
        uniform_ln_probs = np.array(uniform_ln_probs)
        
        # Filter finite values
        uniform_finite_mask = np.isfinite(uniform_ln_probs)
        uniform_finite_ln_probs = uniform_ln_probs[uniform_finite_mask]
        
        print(f"Finite uniform log probabilities: {len(uniform_finite_ln_probs)}/{len(uniform_ln_probs)}")
        
        if len(uniform_finite_ln_probs) > 0:
            # Compute volume of uniform sampling region
            uniform_volume = float(np.prod(bounds_high - bounds_low))
            
            # Use logsumexp for uniform integral estimate
            log_uniform_volume = np.log(uniform_volume)
            log_n_uniform_samples = np.log(len(uniform_finite_ln_probs))
            log_uniform_integral = log_uniform_volume - log_n_uniform_samples + logsumexp(uniform_finite_ln_probs)
            uniform_integral_estimate = np.exp(log_uniform_integral)
            
            print(f"Uniform sampling volume: {uniform_volume:.3e}")
            print(f"Uniform sampling integral estimate: {uniform_integral_estimate:.6f}")
            
            if 0.5 <= uniform_integral_estimate <= 2.0:
                print("✅ Uniform sampling normalization looks reasonable!")
            else:
                print("❌ Uniform sampling suggests normalization issues!")

def main():
    parser = argparse.ArgumentParser(description="Check ln_prob calculation for NFDist")
    parser.add_argument("model_path", help="Path to the trained model directory")
    parser.add_argument("--n-samples", type=int, default=10000,
                       help="Number of samples to generate for ln_prob testing")
    parser.add_argument("--n-mc-samples", type=int, default=50000,
                       help="Number of samples for Monte Carlo normalization test")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_path):
        print(f"❌ Model path does not exist: {args.model_path}")
        return 1
    
    print("=== Checking ln_prob calculation for NFDist ===\n")
    
    try:
        # Step 1: Load model and generate samples
        print("Step 1: Loading model and generating samples")
        nf_samples = load_model_and_generate_samples(args.model_path, args.n_samples)
        
        # Step 2: Create Bilby NFDist
        print("\nStep 2: Creating Bilby NFDist")
        nf_dist, _ = create_bilby_nf_prior(args.model_path)
        
        # Step 3: Compute ln_prob for samples
        print("\nStep 3: Computing ln_prob for samples")
        ln_probs = compute_ln_prob_for_samples(nf_dist, nf_samples)
        
        # Step 4: Create visualizations
        print("\nStep 4: Creating visualizations")
        output_dir = Path(args.model_path) / "ln_prob_check"
        output_dir.mkdir(exist_ok=True)
        
        # Step 5: Test normalization
        print("\nStep 5: Testing normalization")
        test_normalization_mc(nf_dist, nf_dist.names, args.n_mc_samples)
        
        return 0
        
    except Exception as e:
        print(f"❌ Error during ln_prob check: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())