#!/usr/bin/env python
"""
Validate normalizing flow normalization using bilby evidence calculation.

This script treats the NF's ln_prob as a likelihood function, sets up uniform priors
based on the NF's parameter ranges, and uses bilby's nested sampling to compute
the evidence. For a properly normalized NF, Volume × Evidence should ≈ 1.

Usage:
    python validate_nf_normalization.py --model-path /path/to/nf/model/
"""

import os
import sys
import argparse
import json
import numpy as np
import bilby
from bilby.core.utils import logger

# Add the bilby directory to path to import NFDist
sys.path.append(os.path.join(os.path.dirname(__file__), '../../bilby'))
from bilby.core.prior.joint import NFDist


class NFNormalizationLikelihood(bilby.Likelihood):
    """
    Custom likelihood that uses the normalizing flow's ln_prob as the likelihood function.
    This allows us to use bilby's evidence calculation to validate NF normalization.
    """
    
    def __init__(self, nf_dist):
        """
        Initialize the likelihood using an NFDist instance.
        
        Parameters
        ----------
        nf_dist : NFDist
            Loaded NFDist with access to the flow and ln_prob method
        """
        self.nf_dist = nf_dist
        self.parameter_names = self.nf_dist.names
        
        # Initialize bilby likelihood with the parameter names
        parameters = {name: None for name in self.parameter_names}
        super().__init__(parameters=parameters)
        
        logger.info(f"Initialized NFNormalizationLikelihood with parameters: {self.parameter_names}")
        
    def log_likelihood(self):
        """
        Return the NF's log probability as the likelihood.
        
        Returns
        -------
        float
            Log probability from the normalizing flow
        """
        # Get parameter values in the correct order
        param_values = [self.parameters[name] for name in self.parameter_names]
        param_array = np.array(param_values)
        
        # Get NF log probability using the NFDist's ln_prob method
        log_prob = self.nf_dist.ln_prob(param_array)
            
        return float(log_prob)


def create_uniform_priors(model_path):
    """
    Create uniform priors based on the NF's parameter ranges.
    
    Parameters
    ----------
    model_path : str
        Path to the NF model directory
        
    Returns
    -------
    dict
        Dictionary of bilby uniform priors
    float
        Total volume of the prior space
    list
        Parameter names
    """
    # Load model kwargs to get parameter names
    kwargs_path = os.path.join(model_path, "model_kwargs.json")
    with open(kwargs_path, 'r') as f:
        model_kwargs = json.load(f)
    
    parameter_names = model_kwargs.get("names", [])
    priors = {}
    volume = 1.0
    
    logger.info("Creating uniform priors")
    
    # Load scaler if available
    scaler_path = os.path.join(model_path, "scaler.gz")
    scaler = None
    if os.path.exists(scaler_path):
        import joblib
        scaler = joblib.load(scaler_path)
        logger.info(f"Loaded scaler from {scaler_path}")
    
    for i, param_name in enumerate(parameter_names):
        if scaler is not None:
            # Use scaler bounds
            min_val = scaler.data_min_[i] 
            max_val = scaler.data_max_[i]
        else:
            raise ValueError("Scaler not found. Ensure model is properly trained with scaling.")
        
        prior_min = min_val
        prior_max = max_val
        
        # # Apply physical constraints
        # if param_name in ["lambda_1", "lambda_2", "lambda_tilde"]:
        #     # Tidal deformabilities must be positive
        #     prior_min = max(prior_min, 1e-3)
        # elif param_name in ["mass_ratio", "q"]:
        #     # Mass ratio must be in [0, 1]
        #     prior_min = max(prior_min, 0.01)  # Avoid exactly 0
        #     prior_max = min(prior_max, 1.0)
        # elif param_name in ["chirp_mass", "chirp_mass_source"]:
        #     # Chirp mass must be positive
        #     prior_min = max(prior_min, 0.5)
            
        # Create uniform prior
        priors[param_name] = bilby.core.prior.Uniform(
            minimum=prior_min, 
            maximum=prior_max, 
            name=param_name
        )
        
        # Update volume
        param_volume = prior_max - prior_min
        volume *= param_volume
        
        logger.info(f"  {param_name}: [{prior_min:.4f}, {prior_max:.4f}] (range: {param_volume:.4f})")
    
    logger.info(f"Total prior volume: {volume:.4e}")
    return priors, volume, parameter_names


def validate_normalization(model_path, nlive=1000, output_dir=None):
    """
    Validate NF normalization using bilby evidence calculation.
    
    Parameters
    ----------
    model_path : str
        Path to the NF model directory
    nlive : int
        Number of live points for nested sampling
    output_dir : str, optional
        Output directory for results
        
    Returns
    -------
    dict
        Results dictionary with evidence, volume, and normalization estimate
    """
    logger.info(f"Starting normalization validation for model: {model_path}")
    
    # Create uniform priors first to get parameter info
    logger.info("Setting up uniform priors...")
    priors, prior_volume, parameter_names = create_uniform_priors(model_path)
    
    # Determine model file path
    model_pt_path = os.path.join(model_path, "model.pt")
    model_eqx_path = os.path.join(model_path, "model.eqx")
    
    if os.path.exists(model_pt_path):
        model_file_path = model_pt_path
    elif os.path.exists(model_eqx_path):
        model_file_path = model_eqx_path
    else:
        raise FileNotFoundError("No model file found (model.pt or model.eqx)")
    
    # Load the NF model using NFDist
    logger.info(f"Loading normalizing flow model from {model_file_path}...")
    nf_dist = NFDist(
        names=parameter_names,
        flow_filename=model_file_path
    )
    
    # Create the likelihood
    logger.info("Creating NFNormalizationLikelihood...")
    likelihood = NFNormalizationLikelihood(nf_dist)
    
    # Set up output directory
    if output_dir is None:
        output_dir = os.path.join(model_path, "normalization_validation")
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up bilby run parameters
    label = "nf_normalization_validation"
    
    logger.info(f"Running bilby inference with {nlive} live points...")
    logger.info(f"Output directory: {output_dir}")
    
    # Check if the results_file exists and is valid
    results_file = os.path.join(output_dir, "normalization_results.json")
    bilby_result_file = os.path.join(output_dir, f"{label}_result.json")
    
    result = None
    if os.path.exists(bilby_result_file):
        logger.info(f"Loading existing bilby result from: {bilby_result_file}")
        try:
            result = bilby.core.result.Result.from_json(bilby_result_file)
            logger.info("Successfully loaded bilby result file")
        except Exception as e:
            logger.warning(f"Failed to load bilby result file: {e}")
            result = None
    
    if result is None:
        if os.path.exists(results_file):
            logger.info(f"Attempting to load normalization results from: {results_file}")
            try:
                with open(results_file, 'r') as f:
                    result_dict = json.load(f)
                result = bilby.core.result.Result.from_dict(result_dict)
                logger.info("Successfully loaded normalization results file")
            except Exception as e:
                logger.warning(f"Failed to load normalization results file: {e}")
                result = None
    
    if result is None:
        logger.info("No valid results found, running bilby sampler...")
        # Run the sampler
        result = bilby.run_sampler(
            likelihood=likelihood,
            priors=priors,
            sampler="dynesty",
            nlive=nlive,
            outdir=output_dir,
            label=label,
            clean=True,  # Clean up intermediate files
            verbose=True,  # Reduce output
        )
    
    # Extract results
    evidence = result.log_evidence
    evidence_error = result.log_evidence_err
    
    # Calculate normalization
    # For uniform priors: Evidence = (1/Volume) * ∫ NF_density dθ
    # If NF is normalized: ∫ NF_density dθ = 1
    # Therefore: Normalization = Volume * Evidence ≈ 1
    
    # Use the NFDist's internal volume (from log Jacobian correction) instead of padded volume
    # Note: log_jacobian_correction = log(1/Volume), so log(Volume) = -log_jacobian_correction
    if hasattr(nf_dist, 'log_jacobian_correction'):
        nf_log_volume = -nf_dist.log_jacobian_correction  # Flip sign to get log(Volume)
        nf_volume = np.exp(nf_log_volume)
        logger.info(f"Using NFDist internal volume: log_jacobian_correction={nf_dist.log_jacobian_correction:.6f}")
        logger.info(f"  -> log(Volume)={nf_log_volume:.6f}, Volume={nf_volume:.4e}")
        log_normalization = nf_log_volume + evidence
        volume_used = nf_volume
        volume_source = "NFDist_internal"
    else:
        logger.warning("NFDist log_jacobian_correction not available, using padded prior volume")
        log_normalization = np.log(prior_volume) + evidence  
        volume_used = prior_volume
        volume_source = "uniform_priors"
        
    normalization = np.exp(log_normalization)
    
    # Estimate error (rough approximation)
    normalization_error = normalization * evidence_error
    
    # Validation result
    tolerance = 0.2  # 20% tolerance
    is_normalized = abs(normalization - 1.0) < tolerance
    
    results = {
        "model_path": model_path,
        "evidence": evidence,
        "evidence_error": evidence_error,
        "prior_volume": prior_volume,
        "volume_used": volume_used,
        "volume_source": volume_source,
        "log_normalization": log_normalization,
        "normalization": normalization,
        "normalization_error": normalization_error,
        "tolerance": tolerance,
        "parameter_names": parameter_names,
        "n_parameters": len(parameter_names),
        "nlive": nlive
    }
    
    # Add is_normalized as a separate field for logic (not saved to JSON)
    results["is_normalized"] = is_normalized
    
    # Save results
    results_file = os.path.join(output_dir, "normalization_results.json")
    with open(results_file, 'w') as f:
        # Convert numpy types for JSON serialization, excluding is_normalized
        json_results = {}
        for key, value in results.items():
            if key == "is_normalized":
                continue  # Skip boolean field that's not JSON serializable
            elif isinstance(value, np.ndarray):
                json_results[key] = value.tolist()
            elif isinstance(value, (np.float64, np.float32)):
                json_results[key] = float(value)
            elif isinstance(value, (np.int64, np.int32)):
                json_results[key] = int(value)
            else:
                json_results[key] = value
        json.dump(json_results, f, indent=2)
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("NORMALIZATION VALIDATION RESULTS")
    logger.info("="*60)
    logger.info(f"Model: {os.path.basename(model_path)}")
    logger.info(f"Parameters: {', '.join(parameter_names)}")
    logger.info(f"Volume used ({volume_source}): {volume_used:.4e}")
    if volume_source == "NFDist_internal":
        logger.info(f"Prior volume: {prior_volume:.4e} (for comparison)")
    logger.info(f"Log evidence: {evidence:.4f} ± {evidence_error:.4f}")
    logger.info(f"Normalization: {normalization:.4f} ± {normalization_error:.4f}")
    logger.info(f"Status: {'✓ NORMALIZED' if is_normalized else '✗ NOT NORMALIZED'}")
    if not is_normalized:
        deviation = abs(normalization - 1.0)
        logger.info(f"Deviation from 1.0: {deviation:.4f} (>{tolerance:.2f} tolerance)")
    logger.info(f"Results saved to: {results_file}")
    logger.info("="*60)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Validate normalizing flow normalization using bilby evidence calculation"
    )
    parser.add_argument(
        "--model-path", 
        required=True,
        help="Path to the NF model directory"
    )
    parser.add_argument(
        "--nlive", 
        type=int, 
        default=1000,
        help="Number of live points for nested sampling (default: 1000)"
    )
    parser.add_argument(
        "--output-dir", 
        help="Output directory for results (default: model_path/normalization_validation)"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logger.setLevel("INFO")
    else:
        logger.setLevel("WARNING")
    
    # Validate arguments
    if not os.path.exists(args.model_path):
        logger.error(f"Model path does not exist: {args.model_path}")
        return 1
        
    # Check for required files
    model_files = ["model_kwargs.json"]
    has_glasflow = os.path.exists(os.path.join(args.model_path, "model.pt"))
    has_flowjax = os.path.exists(os.path.join(args.model_path, "model.eqx"))
    
    if not (has_glasflow or has_flowjax):
        logger.error("No model file found (model.pt or model.eqx)")
        return 1
        
    for required_file in model_files:
        file_path = os.path.join(args.model_path, required_file)
        if not os.path.exists(file_path):
            logger.error(f"Required file not found: {file_path}")
            return 1
    
    try:
        # Run validation
        results = validate_normalization(
            model_path=args.model_path,
            nlive=args.nlive,
            output_dir=args.output_dir
        )
        
        # Return appropriate exit code
        return 0 if results["is_normalized"] else 1
        
    except Exception as e:
        logger.error(f"Error during validation: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)