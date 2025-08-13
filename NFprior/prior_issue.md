# Normalizing Flow Prior Normalization Issues

## Problem Statement

The log prior values from NF-based distributions are sometimes above 0, indicating that the normalizing flow prior is unnormalized (i.e., the probability density function integrates to a value greater than 1 over the parameter space).

## Why This Is Critical

### Theoretical Violation
Normalizing flows should be normalized by construction:
- They use invertible transformations with proper Jacobian determinants
- They transform a normalized base distribution (e.g., standard Gaussian) to the target distribution
- The change of variables formula ensures normalization is preserved: if p_base integrates to 1, then p_target should also integrate to 1

### Practical Impact
1. **Bayes Factor Calculations**: The Bayes factor depends on the evidence, which includes the prior normalization. An unnormalized prior will give systematically incorrect Bayes factors, undermining the entire source classification analysis.

2. **Parameter Estimation Interpretation**: The posterior samples may not represent the true posterior distribution if the prior normalization is wrong, leading to biased parameter estimates.

3. **Model Comparison**: When comparing different priors (uniform vs. NF-based), they need to be properly normalized for fair comparison. Unnormalized priors invalidate these comparisons.

4. **Scientific Conclusions**: All downstream scientific interpretations (BNS vs NSBH classification, λ̃ bimodality analysis) become unreliable.

## Potential Root Causes

### 1. Coordinate Transformation Chain Issues
The full transformation chain is:
```
Unit cube → Gaussian → NF → Physical parameters
```

**Suspected Issues:**
- Missing Jacobian corrections in the unit cube ↔ Gaussian transformation
- Incorrect handling of the transformation from Gaussian base to physical parameter space
- Accumulated numerical errors across multiple transformations

### 2. FlowJAX Implementation Bugs

**Constrained Distribution Implementation:**
- Bug in the `Transformed(Normal, bijection)` implementation with `Stack` bijections
- Incorrect Jacobian calculation for composed transformations (Softplus, ScalarAffine, Sigmoid)
- Parameter-specific transformation errors (mass ratio bounds, lambda positivity constraints)

**Flow Architecture Issues:**
- Incorrect Jacobian computation in Block Neural Autoregressive Flow (BNAF)
- Numerical instability in the autoregressive transformations, especially with large-capacity models
- Incorrect handling of the numerical inversion process during sampling

### 3. Bilby Integration Problems

**NFPrior Class Issues:**
- Coordinate system mismatches between bilby's parameter space and NF parameter space
- Incorrect rescaling between bilby's bounded parameters and NF's unbounded space
- Missing or double-counted Jacobian corrections in the bilby-NF interface

**Parameter Bounds Handling:**
- Even with FlowJAX constraints, there might be edge cases where probability mass leaks outside physically valid bounds
- Inconsistency between bilby's parameter bounds and NF's learned support

### 4. Training Data Issues

**Normalization During Training:**
- Training data may not be properly normalized before fitting the NF
- Inconsistent parameter ranges between training data and evaluation data
- Data preprocessing (standardization, bounds clipping) affecting the learned distribution normalization

**Validation Issues:**
- Training loss plateauing doesn't guarantee proper normalization
- Lack of explicit normalization constraints during training
- Insufficient validation of learned distribution properties

## Diagnostic Strategy

### Phase 1: Isolate the Source
1. **Direct Flow Sampling Test:**
   ```python
   # Sample directly from trained NF (outside bilby)
   key = jax.random.PRNGKey(42)
   samples = flow.sample(key, (10000,))
   log_probs = flow.log_prob(samples)
   
   print(f"Max log prob: {log_probs.max()}")
   print(f"Mean log prob: {log_probs.mean()}")
   print(f"Samples with log_prob > 0: {(log_probs > 0).sum()}")
   ```

   **Interpretation:**
   - If max log prob > 0: Bug is in NF training/implementation
   - If max log prob ≤ 0: Bug is in bilby-NF interface

2. **Base Distribution Verification:**
   ```python
   # Check if base distribution is normalized
   base_samples = flow.base_distribution.sample(key, (10000,))
   base_log_probs = flow.base_distribution.log_prob(base_samples)
   
   print(f"Base max log prob: {base_log_probs.max()}")
   ```

3. **Jacobian Inspection:**
   ```python
   # Check Jacobian computation
   test_samples = flow.sample(key, (1000,))
   log_probs, log_det_jacobians = flow.log_prob_and_jacobian(test_samples)
   
   print(f"Log det Jacobian stats: mean={log_det_jacobians.mean()}, std={log_det_jacobians.std()}")
   print(f"Suspicious large Jacobians: {(log_det_jacobians > 10).sum()}")
   ```

### Phase 2: Parameter-Specific Analysis
1. **Check Individual Parameter Transformations:**
   ```python
   # Test each parameter transformation separately
   from flowjax.bijections import Softplus, ScalarAffine, Sigmoid, Chain
   
   # Mass ratio transformation: [0.1, 1.0]
   q_transform = Chain([Sigmoid(), ScalarAffine(scale=0.9, loc=0.1)])
   
   # Lambda transformations: > 0
   lambda_transform = Softplus()
   
   # Test normalization preservation
   for transform, name in [(q_transform, "mass_ratio"), (lambda_transform, "lambda")]:
       test_input = jax.random.normal(key, (10000,))
       transformed = transform.transform(test_input)
       log_det_jac = transform.log_det_jacobian(test_input)
       
       # Check if transformation preserves normalization
       input_log_prob = -0.5 * test_input**2 - 0.5 * np.log(2 * np.pi)  # Standard normal
       output_log_prob = input_log_prob - log_det_jac
       
       print(f"{name} - Max output log prob: {output_log_prob.max()}")
   ```

2. **Parameter Range Validation:**
   ```python
   # Check if all samples are within expected bounds
   samples = flow.sample(key, (10000,))
   
   # Assuming parameter order: [chirp_mass, mass_ratio, lambda_1, lambda_2]
   chirp_mass, q, lambda_1, lambda_2 = samples[:, 0], samples[:, 1], samples[:, 2], samples[:, 3]
   
   print(f"Mass ratio range: [{q.min():.3f}, {q.max():.3f}] (expected: [0.1, 1.0])")
   print(f"Lambda_1 range: [{lambda_1.min():.3f}, {lambda_1.max():.3f}] (expected: [0, ∞))")
   print(f"Lambda_2 range: [{lambda_2.min():.3f}, {lambda_2.max():.3f}] (expected: [0, ∞))")
   print(f"Lambda ordering violations: {(lambda_2 > lambda_1).sum()} / {len(samples)}")
   ```

### Phase 3: Bilby Interface Analysis
1. **NFPrior Evaluation Test:**
   ```python
   # Test bilby NFPrior directly
   from bilby_path import NFPrior  # Adjust import path
   
   # Load trained model and create prior
   nf_prior = NFPrior(model_path="path/to/model", parameter_names=["chirp_mass", "mass_ratio", "lambda_1", "lambda_2"])
   
   # Test evaluation at multiple points
   test_points = {
       "chirp_mass": np.array([1.2, 1.3, 1.4]),
       "mass_ratio": np.array([0.8, 0.9, 1.0]), 
       "lambda_1": np.array([400, 500, 600]),
       "lambda_2": np.array([200, 300, 400])
   }
   
   log_probs = []
   for i in range(len(test_points["chirp_mass"])):
       point = {key: val[i] for key, val in test_points.items()}
       log_prob = nf_prior.prob(point, return_log=True)
       log_probs.append(log_prob)
       print(f"Point {i}: {point} -> log_prob = {log_prob}")
   
   print(f"Max log prob from bilby: {max(log_probs)}")
   ```

2. **Parameter Rescaling Verification:**
   ```python
   # Check if rescaling introduces normalization errors
   # This depends on the specific implementation in bilby NFPrior
   ```

## Action Items

### Immediate (High Priority)
1. **Run Phase 1 diagnostics** to isolate whether the bug is in the NF implementation or bilby interface
2. **Check training logs** for any normalization-related warnings or errors
3. **Verify constrained FlowJAX implementation** by comparing constrained vs unconstrained model outputs

### Short Term
1. **Implement explicit normalization check** in the NF training pipeline
2. **Add normalization validation** to the model evaluation script (`evaluate_flows.py`)
3. **Create minimal reproduction script** to test NF normalization in isolation
4. **Review bilby NFPrior implementation** for coordinate transformation bugs

### Medium Term
1. **Implement post-hoc normalization correction** if needed:
   ```python
   # Monte Carlo estimation of normalization constant
   def estimate_normalization_constant(flow, n_samples=100000):
       key = jax.random.PRNGKey(42)
       # Sample from a uniform distribution over parameter bounds
       uniform_samples = sample_uniformly_over_bounds(key, n_samples)
       log_probs = flow.log_prob(uniform_samples)
       # Estimate integral using Monte Carlo
       return log_probs.mean() + np.log(volume_of_bounds)
   ```

2. **Benchmark against analytical solutions** where possible (e.g., simple test distributions)
3. **Implement comprehensive unit tests** for all transformation components

### Long Term
1. **Consider alternative NF architectures** (Coupling flows, Neural Spline Flows) if BNAF proves problematic
2. **Implement numerical stability improvements** for the transformation chain
3. **Develop automated validation pipeline** for all trained models

## References and Context

- **FlowJAX Documentation**: Check for known issues with `Transformed` distributions and constrained bijections
- **BNAF Paper**: Review numerical stability considerations and normalization guarantees
- **Bilby Prior Implementation**: Understand how bilby handles coordinate transformations and Jacobians
- **GW Parameter Estimation Best Practices**: Ensure compatibility with standard bilby workflows

## Notes

- This issue is blocking reliable Bayes factor calculations for the entire GW source classification project
- The unnormalized priors may explain unexpected results in the GW170817 NSBH vs BNS analysis
- Resolution of this issue is critical before proceeding with any scientific conclusions or paper writing
- Consider consulting with FlowJAX developers if the issue persists in the library implementation