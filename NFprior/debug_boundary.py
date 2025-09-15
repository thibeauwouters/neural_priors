#!/usr/bin/env python3
"""
Analyze normalizing flow prior behavior on high mass ratio training data.

This script investigates how the normalizing flow prior evaluates probability
for training samples with high mass ratios (q > threshold). It:

1. Loads a trained normalizing flow model
2. Loads training data and filters by mass ratio threshold
3. Evaluates ln_prob for filtered samples
4. Creates 2D scatter plots showing ln_prob vs parameter combinations
5. Analyzes the probability landscape for high mass ratio systems

Example: python debug_boundary.py models/uniform/bns/radio --q-threshold 0.95

NOTE: Diagnostic script to understand prior behavior on training data subsets.
"""

# COMMENTED OUT: Original boundary trajectory analysis code
# The original implementation generated trajectories approaching q=1 boundary
# and analyzed ln_prob behavior along these synthetic paths.
# This has been replaced with analysis of actual training data filtered by mass ratio.
"""
ORIGINAL IMPLEMENTATION (COMMENTED OUT):
- generate_physical_boundary_trajectory(): Created synthetic trajectories
- analyze_boundary_behavior(): Analyzed ln_prob along trajectories  
- create_boundary_plots(): Visualized trajectory behavior
These functions created artificial paths in parameter space rather than
analyzing the model's performance on real training data.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import json
import argparse
from pathlib import Path
import warnings
from scipy.optimize import curve_fit
from scipy.stats import binned_statistic

# Add bilby to path
sys.path.insert(0, "/Users/Woute029/Documents/Code/projects/eos_source_classification/bilby")

# Import bilby NFDist class
from bilby.core.prior.joint import NFDist, NFPrior
from bilby.core.prior import PriorDict

# Import flow evaluation tools
from evaluate_flows import CheckerUnconditional

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
    
    return nf_dist, parameter_names

def load_training_data(model_path):
    """Load training data from the model directory."""
    training_data_path = os.path.join(model_path, "training_data.npz")
    if os.path.exists(training_data_path):
        data = np.load(training_data_path)
        return data
    else:
        raise FileNotFoundError(f"Training data not found at {training_data_path}")

def lambda_1_lambda_2_to_delta_lambda_tilde(lambda_1, lambda_2, m1, m2):
    """Convert lambda_1, lambda_2 to delta_lambda_tilde using bilby conversion."""
    from bilby.gw.conversion import lambda_1_lambda_2_to_delta_lambda_tilde
    return lambda_1_lambda_2_to_delta_lambda_tilde(lambda_1, lambda_2, m1, m2)

def lambda_tilde_delta_lambda_tilde_to_lambda_1_lambda_2(lambda_tilde, delta_lambda_tilde, m1, m2):
    """Convert lambda_tilde, delta_lambda_tilde back to lambda_1, lambda_2."""
    from bilby.gw.conversion import lambda_tilde_delta_lambda_tilde_to_lambda_1_lambda_2
    return lambda_tilde_delta_lambda_tilde_to_lambda_1_lambda_2(lambda_tilde, delta_lambda_tilde, m1, m2)

def component_masses_to_chirp_mass(m1, m2):
    """Convert component masses to chirp mass."""
    from bilby.gw.conversion import component_masses_to_chirp_mass
    return component_masses_to_chirp_mass(m1, m2)

def chirp_mass_and_mass_ratio_to_component_masses(chirp_mass, q):
    """Convert chirp mass and mass ratio to component masses."""
    from bilby.gw.conversion import chirp_mass_and_mass_ratio_to_component_masses
    return chirp_mass_and_mass_ratio_to_component_masses(chirp_mass, q)

# COMMENTED OUT: Original trajectory generation function
# def generate_physical_boundary_trajectory(model_path, parameter_names, n_points=100):
#     """Original implementation that created synthetic trajectories toward q=1 boundary"""
#     # [Original implementation commented out - see git history if needed]
#     pass

def filter_training_data_by_mass_ratio(training_data, parameter_names, q_threshold=0.95):
    """
    Filter training data to keep only samples with mass ratio above threshold.
    
    Args:
        training_data: Loaded training data from npz file
        parameter_names: List of parameter names
        q_threshold: Minimum mass ratio to keep (default: 0.95)
    
    Returns:
        dict: Filtered data with high mass ratio samples
    """
    # Get mass ratio values
    if "mass_ratio" in training_data:
        q_values = training_data["mass_ratio"]
    elif "q" in training_data:
        q_values = training_data["q"]
    else:
        raise ValueError("No mass ratio parameter found in training data")
    
    # Create filter mask
    high_q_mask = q_values >= q_threshold
    n_total = len(q_values)
    n_filtered = np.sum(high_q_mask)
    
    print(f"Filtering training data by q >= {q_threshold:.3f}")
    print(f"Keeping {n_filtered}/{n_total} samples ({100*n_filtered/n_total:.1f}%)")
    
    # Filter all parameters
    filtered_data = {}
    for param_name in parameter_names:
        if param_name in training_data:
            filtered_data[param_name] = training_data[param_name][high_q_mask]
        else:
            print(f"Warning: Parameter {param_name} not found in training data")
    
    # Also store the original parameter names for reference
    for key in training_data.keys():
        if key not in filtered_data:
            filtered_data[key] = training_data[key][high_q_mask]
    
    # Store filter info
    filtered_data["filter_info"] = {
        "q_threshold": q_threshold,
        "n_total": n_total,
        "n_filtered": n_filtered,
        "q_min": float(np.min(q_values[high_q_mask])),
        "q_max": float(np.max(q_values[high_q_mask])),
        "q_mean": float(np.mean(q_values[high_q_mask]))
    }
    
    return filtered_data

def evaluate_ln_prob_on_filtered_data(nf_dist, filtered_data, parameter_names):
    """
    Evaluate ln_prob for all filtered training samples.
    
    Args:
        nf_dist: NFDist object
        filtered_data: Dictionary with filtered training data
        parameter_names: List of parameter names in correct order
    
    Returns:
        dict: Results including ln_prob values and sample info
    """
    n_samples = len(filtered_data[parameter_names[0]])
    ln_probs = np.zeros(n_samples)
    
    print(f"Evaluating ln_prob for {n_samples} high mass ratio samples...")
    
    # Create sample matrix in correct parameter order
    samples = np.zeros((n_samples, len(parameter_names)))
    for i, param_name in enumerate(parameter_names):
        samples[:, i] = filtered_data[param_name]
    
    # Evaluate ln_prob for all samples
    n_successful = 0
    n_failed = 0
    
    for i in range(n_samples):
        if i % max(1, n_samples // 20) == 0:  # Progress every 5%
            print(f"  Progress: {i}/{n_samples} ({100*i/n_samples:.1f}%)")
        
        try:
            ln_prob_val = nf_dist.ln_prob(samples[i])
            ln_probs[i] = ln_prob_val
            n_successful += 1
        except Exception as e:
            if n_failed < 5:  # Only show first few errors
                print(f"Warning: Error evaluating ln_prob for sample {i}: {e}")
            n_failed += 1
            ln_probs[i] = np.nan
    
    print(f"Evaluation complete: {n_successful} successful, {n_failed} failed")
    
    # Calculate statistics on valid ln_probs
    valid_mask = np.isfinite(ln_probs)
    valid_ln_probs = ln_probs[valid_mask]
    
    results = {
        "ln_probs": ln_probs,
        "samples": samples,
        "valid_mask": valid_mask,
        "n_successful": n_successful,
        "n_failed": n_failed,
        "ln_prob_stats": {
            "n_valid": len(valid_ln_probs),
            "min": float(np.min(valid_ln_probs)) if len(valid_ln_probs) > 0 else np.nan,
            "max": float(np.max(valid_ln_probs)) if len(valid_ln_probs) > 0 else np.nan,
            "mean": float(np.mean(valid_ln_probs)) if len(valid_ln_probs) > 0 else np.nan,
            "std": float(np.std(valid_ln_probs)) if len(valid_ln_probs) > 0 else np.nan,
            "median": float(np.median(valid_ln_probs)) if len(valid_ln_probs) > 0 else np.nan
        }
    }
    
    return results

# COMMENTED OUT: Original trajectory analysis function
# def analyze_boundary_behavior(trajectories, ln_probs_dict, parameter_names):
#     """Original implementation that analyzed synthetic trajectory behavior"""
#     # [Original implementation commented out - see git history if needed]
#     pass

def analyze_high_q_data_behavior(filtered_data, evaluation_results, parameter_names):
    """
    Analyze the ln_prob behavior for high mass ratio training data.
    
    Args:
        filtered_data: Dictionary with filtered training data
        evaluation_results: Results from evaluate_ln_prob_on_filtered_data
        parameter_names: List of parameter names
    
    Returns:
        dict: Analysis results for high mass ratio data
    """
    ln_probs = evaluation_results["ln_probs"]
    valid_mask = evaluation_results["valid_mask"]
    ln_prob_stats = evaluation_results["ln_prob_stats"]
    
    analysis = {
        "filter_info": filtered_data["filter_info"],
        "evaluation_stats": {
            "n_successful": evaluation_results["n_successful"],
            "n_failed": evaluation_results["n_failed"],
            "success_rate": evaluation_results["n_successful"] / len(ln_probs) if len(ln_probs) > 0 else 0
        },
        "ln_prob_distribution": ln_prob_stats
    }
    
    if ln_prob_stats["n_valid"] > 0:
        valid_ln_probs = ln_probs[valid_mask]
        
        # Analyze ln_prob distribution characteristics
        analysis["distribution_analysis"] = {
            "percentiles": {
                "5%": float(np.percentile(valid_ln_probs, 5)),
                "25%": float(np.percentile(valid_ln_probs, 25)),
                "50%": float(np.percentile(valid_ln_probs, 50)),
                "75%": float(np.percentile(valid_ln_probs, 75)),
                "95%": float(np.percentile(valid_ln_probs, 95))
            },
            "outliers": {
                "very_high": int(np.sum(valid_ln_probs > ln_prob_stats["mean"] + 3*ln_prob_stats["std"])),
                "very_low": int(np.sum(valid_ln_probs < ln_prob_stats["mean"] - 3*ln_prob_stats["std"]))
            }
        }
        
        # Parameter correlations with ln_prob
        param_correlations = {}
        for i, param_name in enumerate(parameter_names):
            if param_name in filtered_data:
                param_values = filtered_data[param_name][valid_mask]
                if len(param_values) > 1 and len(valid_ln_probs) > 1:
                    correlation = np.corrcoef(param_values, valid_ln_probs)[0, 1]
                    param_correlations[param_name] = float(correlation) if np.isfinite(correlation) else 0.0
        
        analysis["parameter_correlations"] = param_correlations
    
    return analysis

# COMMENTED OUT: Original boundary plotting functions
# def create_boundary_plots(trajectories, ln_probs_dict, output_dir):
#     """Original implementation that plotted synthetic trajectory behavior"""
#     # [Original implementation commented out - see git history if needed]
#     pass

def create_high_q_scatter_plots(filtered_data, evaluation_results, parameter_names, output_dir):
    """
    Create 2D scatter plots showing ln_prob vs parameter combinations for high mass ratio data.
    
    Args:
        filtered_data: Dictionary with filtered training data
        evaluation_results: Results from evaluate_ln_prob_on_filtered_data
        parameter_names: List of parameter names
        output_dir: Output directory for plots
    """
    ln_probs = evaluation_results["ln_probs"]
    valid_mask = evaluation_results["valid_mask"]
    
    if np.sum(valid_mask) == 0:
        print("No valid ln_prob evaluations - cannot create plots")
        return
    
    # Filter to valid samples only
    valid_ln_probs = ln_probs[valid_mask]
    
    # Set up plotting parameters
    plt.style.use('default')
    
    # Get key parameters for plotting
    plot_params = []
    plot_labels = []
    
    # Always include mass ratio if available
    if "mass_ratio" in filtered_data:
        plot_params.append(filtered_data["mass_ratio"][valid_mask])
        plot_labels.append("Mass Ratio (q)")
    elif "q" in filtered_data:
        plot_params.append(filtered_data["q"][valid_mask])
        plot_labels.append("Mass Ratio (q)")
    
    # Include chirp mass
    if "chirp_mass_source" in filtered_data:
        plot_params.append(filtered_data["chirp_mass_source"][valid_mask])
        plot_labels.append("Chirp Mass [M☉]")
    elif "chirp_mass" in filtered_data:
        plot_params.append(filtered_data["chirp_mass"][valid_mask])
        plot_labels.append("Chirp Mass [M☉]")
    
    # Include tidal deformabilities
    if "lambda_1" in filtered_data:
        plot_params.append(filtered_data["lambda_1"][valid_mask])
        plot_labels.append("λ₁")
    
    if "lambda_2" in filtered_data:
        plot_params.append(filtered_data["lambda_2"][valid_mask])
        plot_labels.append("λ₂")
    
    # Include distance if available
    if "luminosity_distance" in filtered_data:
        plot_params.append(filtered_data["luminosity_distance"][valid_mask])
        plot_labels.append("Luminosity Distance [Mpc]")
    
    # Compute and include delta_lambda_tilde if we have the required parameters
    if ("lambda_1" in filtered_data and "lambda_2" in filtered_data and 
        ("chirp_mass_source" in filtered_data or "chirp_mass" in filtered_data) and
        ("mass_ratio" in filtered_data or "q" in filtered_data)):
        
        # Get the required parameters
        lambda_1_vals = filtered_data["lambda_1"][valid_mask]
        lambda_2_vals = filtered_data["lambda_2"][valid_mask]
        
        if "chirp_mass_source" in filtered_data:
            mc_vals = filtered_data["chirp_mass_source"][valid_mask]
        else:
            mc_vals = filtered_data["chirp_mass"][valid_mask]
            
        if "mass_ratio" in filtered_data:
            q_vals = filtered_data["mass_ratio"][valid_mask]
        else:
            q_vals = filtered_data["q"][valid_mask]
        
        # Convert to component masses
        m1_vals = []
        m2_vals = []
        delta_lambda_tilde_vals = []
        
        for i in range(len(lambda_1_vals)):
            m1, m2 = chirp_mass_and_mass_ratio_to_component_masses(mc_vals[i], q_vals[i])
            m1_vals.append(m1)
            m2_vals.append(m2)
            
            # Compute delta_lambda_tilde
            delta_lambda_tilde = lambda_1_lambda_2_to_delta_lambda_tilde(
                lambda_1_vals[i], lambda_2_vals[i], m1, m2
            )
            delta_lambda_tilde_vals.append(delta_lambda_tilde)
        
        plot_params.append(np.array(delta_lambda_tilde_vals))
        plot_labels.append("δλ̃")
    
    n_params = len(plot_params)
    if n_params == 0:
        print("No recognized parameters found for plotting")
        return
    
    print(f"Creating scatter plots for {n_params} parameters")
    
    # Create comprehensive figure with ln_prob vs each parameter
    n_cols = min(3, n_params)
    n_rows = (n_params + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_params == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Create color map based on ln_prob values
    vmin, vmax = np.percentile(valid_ln_probs, [5, 95])  # Use 5th-95th percentiles to avoid outliers
    
    for i in range(n_params):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col] if n_rows > 1 else axes[col]
        
        # Create scatter plot colored by ln_prob
        scatter = ax.scatter(plot_params[i], valid_ln_probs, 
                           c=valid_ln_probs, cmap='viridis', 
                           alpha=0.6, s=20, vmin=vmin, vmax=vmax)
        
        ax.set_xlabel(plot_labels[i])
        ax.set_ylabel('ln(probability)')
        ax.set_title(f'NF Prior: ln(prob) vs {plot_labels[i]}')
        ax.grid(True, alpha=0.3)
        
        # Add colorbar for first plot only
        if i == 0:
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('ln(probability)')
    
    # Hide unused subplots
    for i in range(n_params, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        if n_rows > 1:
            axes[row, col].set_visible(False)
        elif n_cols > 1:
            axes[col].set_visible(False)
    
    # Add overall title with filter information
    filter_info = filtered_data["filter_info"]
    fig.suptitle(f'High Mass Ratio Analysis (q ≥ {filter_info["q_threshold"]:.3f})\n'
                f'{filter_info["n_filtered"]} samples, q ∈ [{filter_info["q_min"]:.3f}, {filter_info["q_max"]:.3f}]', 
                fontsize=14)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    
    # Save main scatter plot
    main_plot_path = os.path.join(output_dir, "high_q_scatter_plots.pdf")
    plt.savefig(main_plot_path, bbox_inches="tight", dpi=300)
    plt.close()
    
    # Create pairwise parameter scatter plots if we have multiple parameters
    if n_params >= 2:
        fig, axes = plt.subplots(n_params-1, n_params-1, figsize=(4*(n_params-1), 4*(n_params-1)))
        
        if n_params == 2:
            axes = np.array([[axes]])
        elif n_params == 3:
            axes = axes.reshape(2, 2)
        
        for i in range(n_params-1):
            for j in range(n_params-1):
                if j > i:
                    # Upper triangle: parameter vs parameter colored by ln_prob
                    scatter = axes[i, j].scatter(plot_params[j+1], plot_params[i], 
                                               c=valid_ln_probs, cmap='viridis',
                                               alpha=0.6, s=20, vmin=vmin, vmax=vmax)
                    axes[i, j].set_xlabel(plot_labels[j+1])
                    axes[i, j].set_ylabel(plot_labels[i])
                    axes[i, j].grid(True, alpha=0.3)
                    
                elif j == i:
                    # Diagonal: histogram of ln_prob
                    axes[i, j].hist(valid_ln_probs, bins=30, alpha=0.7, density=True)
                    axes[i, j].set_xlabel('ln(probability)')
                    axes[i, j].set_ylabel('Density')
                    axes[i, j].set_title('ln(prob) Distribution')
                    axes[i, j].grid(True, alpha=0.3)
                    
                else:
                    # Lower triangle: hide
                    axes[i, j].set_visible(False)
        
        plt.tight_layout()
        pairwise_plot_path = os.path.join(output_dir, "high_q_pairwise_plots.pdf")
        plt.savefig(pairwise_plot_path, bbox_inches="tight", dpi=300)
        plt.close()
    
    # Create ln_prob distribution plot
    plt.figure(figsize=(12, 8))
    
    # Main histogram
    plt.subplot(2, 2, 1)
    plt.hist(valid_ln_probs, bins=50, alpha=0.7, density=True, edgecolor='black')
    plt.xlabel('ln(probability)')
    plt.ylabel('Density')
    plt.title('Distribution of ln(probability) for High Mass Ratio Samples')
    plt.grid(True, alpha=0.3)
    
    # Add statistics
    stats = evaluation_results["ln_prob_stats"]
    stats_text = f"Mean: {stats['mean']:.2f}\nStd: {stats['std']:.2f}\nMedian: {stats['median']:.2f}\nMin: {stats['min']:.2f}\nMax: {stats['max']:.2f}"
    plt.text(0.7, 0.7, stats_text, transform=plt.gca().transAxes, 
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    # Q-Q plot (if we have enough samples)
    if len(valid_ln_probs) > 50:
        from scipy import stats as scipy_stats
        plt.subplot(2, 2, 2)
        scipy_stats.probplot(valid_ln_probs, dist="norm", plot=plt)
        plt.title('Q-Q Plot (Normal Distribution)')
        plt.grid(True, alpha=0.3)
    
    # Box plot
    plt.subplot(2, 2, 3)
    plt.boxplot(valid_ln_probs, vert=True)
    plt.ylabel('ln(probability)')
    plt.title('Box Plot of ln(probability)')
    plt.grid(True, alpha=0.3)
    
    # Cumulative distribution
    plt.subplot(2, 2, 4)
    sorted_ln_probs = np.sort(valid_ln_probs)
    cumulative = np.linspace(0, 1, len(sorted_ln_probs))
    plt.plot(sorted_ln_probs, cumulative, 'b-', linewidth=2)
    plt.xlabel('ln(probability)')
    plt.ylabel('Cumulative Probability')
    plt.title('Cumulative Distribution of ln(probability)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    dist_plot_path = os.path.join(output_dir, "high_q_ln_prob_distribution.pdf")
    plt.savefig(dist_plot_path, bbox_inches="tight", dpi=300)
    plt.close()
    
    print(f"Created plots:")
    print(f"  Main scatter plots: {main_plot_path}")
    if n_params >= 2:
        print(f"  Pairwise plots: {pairwise_plot_path}")
    print(f"  Distribution plots: {dist_plot_path}")

def main():
    parser = argparse.ArgumentParser(description="Analyze NF prior behavior on high mass ratio training data")
    parser.add_argument("model_path", nargs='?', default="models/uniform/bns/radio", 
                       help="Path to the trained model directory (default: models/uniform/bns/radio)")
    parser.add_argument("--q-threshold", type=float, default=0.95,
                       help="Mass ratio threshold for filtering data (default: 0.95)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_path):
        print(f"❌ Model path does not exist: {args.model_path}")
        return 1
    
    print("=== Analyzing NF Prior on High Mass Ratio Training Data ===\n")
    
    try:
        # Create output directory
        output_dir = Path(args.model_path) / "boundary_debug"
        output_dir.mkdir(exist_ok=True)
        
        # Step 1: Create Bilby NFDist
        print("Step 1: Creating Bilby NFDist")
        nf_dist, parameter_names = create_bilby_nf_prior(args.model_path)
        
        # Step 2: Load and filter training data
        print(f"\nStep 2: Loading training data and filtering by mass ratio >= {args.q_threshold:.3f}")
        training_data = load_training_data(args.model_path)
        filtered_data = filter_training_data_by_mass_ratio(training_data, parameter_names, args.q_threshold)
        
        # Step 3: Evaluate ln_prob on filtered data
        print(f"\nStep 3: Evaluating ln_prob for filtered samples")
        evaluation_results = evaluate_ln_prob_on_filtered_data(nf_dist, filtered_data, parameter_names)
        
        # Print quick summary
        stats = evaluation_results["ln_prob_stats"]
        print(f"\nQuick Summary:")
        print(f"  Successful evaluations: {evaluation_results['n_successful']}/{evaluation_results['n_successful'] + evaluation_results['n_failed']}")
        if stats["n_valid"] > 0:
            print(f"  ln_prob range: [{stats['min']:.3f}, {stats['max']:.3f}]")
            print(f"  ln_prob mean ± std: {stats['mean']:.3f} ± {stats['std']:.3f}")
            print(f"  ln_prob median: {stats['median']:.3f}")
        
        # Step 4: Analyze results
        print(f"\nStep 4: Analyzing high mass ratio data behavior")
        analysis = analyze_high_q_data_behavior(filtered_data, evaluation_results, parameter_names)
        
        # Print detailed analysis results
        print("\nDetailed Analysis Results:")
        print("=" * 60)
        
        filter_info = analysis["filter_info"]
        print(f"\nData Filtering:")
        print(f"  Threshold: q >= {filter_info['q_threshold']:.3f}")
        print(f"  Samples kept: {filter_info['n_filtered']}/{filter_info['n_total']} ({100*filter_info['n_filtered']/filter_info['n_total']:.1f}%)")
        print(f"  Mass ratio range: [{filter_info['q_min']:.4f}, {filter_info['q_max']:.4f}]")
        print(f"  Mean mass ratio: {filter_info['q_mean']:.4f}")
        
        eval_stats = analysis["evaluation_stats"]
        print(f"\nEvaluation Results:")
        print(f"  Success rate: {100*eval_stats['success_rate']:.1f}% ({eval_stats['n_successful']}/{eval_stats['n_successful'] + eval_stats['n_failed']})")
        
        ln_prob_dist = analysis["ln_prob_distribution"]
        if ln_prob_dist["n_valid"] > 0:
            print(f"\nln_prob Distribution:")
            print(f"  Valid samples: {ln_prob_dist['n_valid']}")
            print(f"  Range: [{ln_prob_dist['min']:.3f}, {ln_prob_dist['max']:.3f}]")
            print(f"  Mean ± Std: {ln_prob_dist['mean']:.3f} ± {ln_prob_dist['std']:.3f}")
            print(f"  Median: {ln_prob_dist['median']:.3f}")
        
        if "distribution_analysis" in analysis:
            dist_analysis = analysis["distribution_analysis"]
            print(f"\nDistribution Percentiles:")
            percentiles = dist_analysis["percentiles"]
            print(f"  5%: {percentiles['5%']:.3f}")
            print(f"  25%: {percentiles['25%']:.3f}")
            print(f"  50%: {percentiles['50%']:.3f}")
            print(f"  75%: {percentiles['75%']:.3f}")
            print(f"  95%: {percentiles['95%']:.3f}")
            
            outliers = dist_analysis["outliers"]
            print(f"\nOutliers (±3σ):")
            print(f"  Very high: {outliers['very_high']} samples")
            print(f"  Very low: {outliers['very_low']} samples")
        
        if "parameter_correlations" in analysis:
            correlations = analysis["parameter_correlations"]
            print(f"\nParameter Correlations with ln_prob:")
            for param, corr in correlations.items():
                print(f"  {param}: {corr:.3f}")
        
        # Step 5: Create visualization plots
        print(f"\nStep 5: Creating 2D scatter plots")
        create_high_q_scatter_plots(filtered_data, evaluation_results, parameter_names, str(output_dir))
        
        # Save analysis results to JSON
        analysis_json = {
            "model_path": args.model_path,
            "parameter_names": parameter_names,
            "q_threshold": args.q_threshold,
            "analysis": analysis
        }
        
        json_path = output_dir / "high_q_analysis.json"
        with open(json_path, "w") as f:
            json.dump(analysis_json, f, indent=2)
        
        print(f"\n✓ High mass ratio analysis completed!")
        print(f"  Results saved to: {output_dir}")
        print(f"  Analysis data: {json_path}")
        print(f"  Plots: high_q_scatter_plots.pdf + distribution plots")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error during high mass ratio analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())