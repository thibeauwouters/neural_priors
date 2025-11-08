"""
Calculate Jensen-Shannon Divergence (JSD) between training data and NF samples for all models.
Outputs a JSON file with JSD values (in bits) for each parameter of each model.
"""

import os
import sys
import json
import numpy as np
import warnings
from scipy.stats import entropy
from scipy.special import kl_div
import tqdm

# Import the CheckerUnconditional class from evaluate_flows
from evaluate_flows import CheckerUnconditional, VERBOSE

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


def calculate_jsd_bits(p_samples: np.ndarray, q_samples: np.ndarray, n_bins: int = 50) -> float:
    """
    Calculate Jensen-Shannon Divergence in bits between two sample distributions.

    Args:
        p_samples: Samples from distribution P
        q_samples: Samples from distribution Q
        n_bins: Number of bins for histogram estimation

    Returns:
        float: JSD in bits
    """
    # Determine common range for both distributions
    min_val = min(p_samples.min(), q_samples.min())
    max_val = max(p_samples.max(), q_samples.max())
    bins = np.linspace(min_val, max_val, n_bins + 1)

    # Create histograms
    p_hist, _ = np.histogram(p_samples, bins=bins, density=True)
    q_hist, _ = np.histogram(q_samples, bins=bins, density=True)

    # Normalize to get probability distributions
    p_prob = p_hist / p_hist.sum()
    q_prob = q_hist / q_hist.sum()

    # Add small epsilon to avoid log(0)
    eps = 1e-10
    p_prob = np.maximum(p_prob, eps)
    q_prob = np.maximum(q_prob, eps)

    # Calculate mixture distribution M = 0.5 * (P + Q)
    m_prob = 0.5 * (p_prob + q_prob)

    # Calculate JSD = 0.5 * KL(P||M) + 0.5 * KL(Q||M)
    kl_pm = np.sum(kl_div(p_prob, m_prob))
    kl_qm = np.sum(kl_div(q_prob, m_prob))
    jsd_nats = 0.5 * kl_pm + 0.5 * kl_qm

    # Convert from nats to bits
    jsd_bits = jsd_nats / np.log(2)

    return float(jsd_bits)


def assess_single_model(model_path: str, n_bins: int = 50, skip_flowjax: bool = False) -> dict:
    """
    Assess a single NF model by calculating JSD for each parameter.
    Uses the same number of samples as the training data for fair comparison.

    Args:
        model_path: Path to the model directory
        n_bins: Number of bins for histogram-based JSD calculation
        skip_flowjax: If True, skip flowjax models

    Returns:
        dict: Dictionary mapping parameter names to JSD values in bits, or None if skipped
    """
    try:
        # Check if this is a flowjax model and skip if requested
        if skip_flowjax:
            model_kwargs_path = os.path.join(model_path, "model_kwargs.json")
            if os.path.exists(model_kwargs_path):
                with open(model_kwargs_path, "r") as f:
                    model_kwargs = json.load(f)
                use_flowjax = model_kwargs.get("use_flowjax", "False") == "True"
                if use_flowjax:
                    if VERBOSE:
                        print(f"  Skipping flowjax model: {model_path}")
                    return None

        # First load with minimal samples to get training data size
        checker = CheckerUnconditional(model_path, N_samples=1)

        # Get training samples to determine actual training size
        training_samples = checker.get_training_samples()
        n_training_samples = len(training_samples)

        if VERBOSE:
            print(f"  Training data size: {n_training_samples} samples")

        # Now reload with matching sample size
        checker = CheckerUnconditional(model_path, N_samples=n_training_samples)

        # Generate NF samples (same size as training data)
        nf_samples = checker.generate_nf_samples()

        # Get parameter names
        parameter_names = checker.nf_kwargs.get("names", [f"param_{i}" for i in range(training_samples.shape[1])])

        # Calculate JSD for each parameter
        jsd_results = {}
        for i, param_name in enumerate(parameter_names):
            train_param = training_samples[:, i]
            nf_param = nf_samples[:, i]

            jsd_value = calculate_jsd_bits(train_param, nf_param, n_bins=n_bins)
            jsd_results[param_name] = jsd_value

            if VERBOSE:
                print(f"  {param_name}: JSD = {jsd_value:.6f} bits")

        # Calculate overall multivariate JSD (flatten all dimensions)
        train_flat = training_samples.flatten()
        nf_flat = nf_samples.flatten()
        overall_jsd = calculate_jsd_bits(train_flat, nf_flat, n_bins=n_bins)
        jsd_results["overall_flattened"] = overall_jsd

        return jsd_results

    except Exception as e:
        print(f"Error processing {model_path}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def find_all_models(base_path: str = "./models",
                    populations: list = None,
                    sources: list = None,
                    eos_constraints: list = None) -> list:
    """
    Find all model directories containing model_kwargs.json.

    Args:
        base_path: Base path to search for models
        populations: List of valid population names (e.g., ["uniform", "gaussian"])
        sources: List of valid source types (e.g., ["bns", "nsbh"])
        eos_constraints: List of valid EOS constraints (e.g., ["radio", "radio_chiEFT"])

    Returns:
        list: List of paths to model directories
    """
    model_paths = []

    for root, dirs, files in os.walk(base_path):
        if "model_kwargs.json" in files:
            # Check if model file exists (either .pt or .eqx)
            has_model = os.path.exists(os.path.join(root, "model.pt")) or \
                       os.path.exists(os.path.join(root, "model.eqx"))

            if has_model:
                # If filters are specified, check if model follows the structure
                if populations is not None or sources is not None or eos_constraints is not None:
                    # Get relative path from base_path
                    rel_path = os.path.relpath(root, base_path)
                    parts = rel_path.split(os.sep)

                    # Expected structure: population/source/eos
                    if len(parts) >= 3:
                        pop, source, eos = parts[0], parts[1], parts[2]

                        # Check each filter
                        if populations is not None and pop not in populations:
                            continue
                        if sources is not None and source not in sources:
                            continue
                        if eos_constraints is not None and eos not in eos_constraints:
                            continue

                        model_paths.append(root)
                    # If structure doesn't match, skip this model
                else:
                    # No filters, add all models
                    model_paths.append(root)

    return sorted(model_paths)


def assess_all_models(base_path: str = "./models",
                      output_file: str = "jsd_assessment.json",
                      n_bins: int = 50,
                      skip_flowjax: bool = False,
                      populations: list = None,
                      sources: list = None,
                      eos_constraints: list = None) -> dict:
    """
    Assess all models in the base path and save results to JSON.
    Uses the same number of samples as each model's training data for fair comparison.

    Args:
        base_path: Base path containing model directories
        output_file: Output JSON file path
        n_bins: Number of bins for histogram-based JSD calculation
        skip_flowjax: If True, skip flowjax models
        populations: List of valid population names to filter
        sources: List of valid source types to filter
        eos_constraints: List of valid EOS constraints to filter

    Returns:
        dict: Complete results dictionary
    """
    # Print filter information
    if populations or sources or eos_constraints:
        print("Filtering models with:")
        if populations:
            print(f"  Populations: {', '.join(populations)}")
        if sources:
            print(f"  Sources: {', '.join(sources)}")
        if eos_constraints:
            print(f"  EOS constraints: {', '.join(eos_constraints)}")
        print("=" * 80)

    # Find all models
    model_paths = find_all_models(base_path, populations=populations,
                                   sources=sources, eos_constraints=eos_constraints)

    if not model_paths:
        print(f"No models found in {base_path}")
        return {}

    print(f"Found {len(model_paths)} models to assess")
    print("=" * 80)

    # Assess each model
    all_results = {}

    for model_path in tqdm.tqdm(model_paths, desc="Assessing models"):
        # Use relative path as key
        relative_path = os.path.relpath(model_path, base_path)

        jsd_results = assess_single_model(model_path, n_bins=n_bins, skip_flowjax=skip_flowjax)

        if jsd_results is not None:
            all_results[relative_path] = jsd_results
        else:
            print(f"Failed: {relative_path}")

    # Save results to JSON
    output_path = os.path.join(base_path, output_file)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 80)
    print(f"Results saved to: {output_path}")
    print(f"Total models assessed: {len(all_results)}/{len(model_paths)}")

    # Print summary statistics
    print("\n" + "=" * 80)
    print("Summary Statistics:")
    print("-" * 80)

    # Collect all JSD values by parameter
    param_jsds = {}
    for model_path, results in all_results.items():
        for param_name, jsd_value in results.items():
            if param_name not in param_jsds:
                param_jsds[param_name] = []
            param_jsds[param_name].append(jsd_value)

    # Print statistics for each parameter
    for param_name, jsd_values in param_jsds.items():
        jsd_array = np.array(jsd_values)
        print(f"\n{param_name}:")
        print(f"  Mean JSD: {np.mean(jsd_array):.6f} bits")
        print(f"  Std JSD:  {np.std(jsd_array):.6f} bits")
        print(f"  Min JSD:  {np.min(jsd_array):.6f} bits")
        print(f"  Max JSD:  {np.max(jsd_array):.6f} bits")
        print(f"  Median:   {np.median(jsd_array):.6f} bits")

    # Find best and worst models based on max JSD across all parameters
    if all_results:
        max_jsds = {}
        for path, results in all_results.items():
            # Get max JSD across all parameters (excluding overall_flattened)
            param_values = [jsd for param, jsd in results.items()
                           if param != "overall_flattened"]
            if param_values:
                max_jsds[path] = max(param_values)
            else:
                max_jsds[path] = float('inf')

        if max_jsds:
            best_model = min(max_jsds, key=max_jsds.get)
            worst_model = max(max_jsds, key=max_jsds.get)

            print("\n" + "=" * 80)
            print("Best Model (lowest max JSD across parameters):")
            print(f"  Path: {best_model}")
            print(f"  Max JSD:  {max_jsds[best_model]:.6f} bits")

            print("\nWorst Model (highest max JSD across parameters):")
            print(f"  Path: {worst_model}")
            print(f"  Max JSD:  {max_jsds[worst_model]:.6f} bits")

    return all_results


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Calculate Jensen-Shannon Divergence for all normalizing flow models"
    )
    parser.add_argument(
        "--base-path",
        type=str,
        default="./models",
        help="Base path containing model directories"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="jsd_assessment.json",
        help="Output JSON file name (saved in base_path)"
    )
    parser.add_argument(
        "--n-bins",
        type=int,
        default=50,
        help="Number of bins for histogram-based JSD calculation"
    )
    parser.add_argument(
        "--single-model",
        type=str,
        default=None,
        help="Assess only a single model at this path"
    )
    parser.add_argument(
        "--include-flowjax",
        action="store_false",
        dest="skip_flowjax",
        help="Include flowjax models during assessment (default: skip them)"
    )
    parser.add_argument(
        "--populations",
        type=str,
        nargs="+",
        default=["uniform", "gaussian", "double_gaussian"],
        help="Filter by population names (default: uniform gaussian double_gaussian GW170817)"
    )
    parser.add_argument(
        "--sources",
        type=str,
        nargs="+",
        default=["bns", "nsbh"],
        help="Filter by source types (default: bns nsbh)"
    )
    parser.add_argument(
        "--eos-constraints",
        type=str,
        nargs="+",
        default=["radio", "radio_chiEFT", "radio_NICER"],
        help="Filter by EOS constraints (default: radio radio_chiEFT radio_NICER)"
    )

    args = parser.parse_args()

    try:
        if args.single_model:
            # Assess single model
            print(f"Assessing single model: {args.single_model}")
            jsd_results = assess_single_model(
                args.single_model,
                n_bins=args.n_bins,
                skip_flowjax=args.skip_flowjax
            )

            if jsd_results is not None:
                print("\nResults:")
                for param_name, jsd_value in jsd_results.items():
                    print(f"  {param_name}: {jsd_value:.6f} bits")

                # Save single model result
                output_path = os.path.join(args.single_model, "jsd_results.json")
                with open(output_path, "w") as f:
                    json.dump(jsd_results, f, indent=2)
                print(f"\nResults saved to: {output_path}")
                return True
            else:
                return False
        else:
            # Assess all models
            all_results = assess_all_models(
                base_path=args.base_path,
                output_file=args.output_file,
                n_bins=args.n_bins,
                skip_flowjax=args.skip_flowjax,
                populations=args.populations,
                sources=args.sources,
                eos_constraints=args.eos_constraints
            )
            return len(all_results) > 0

    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
