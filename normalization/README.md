# normalization

This directory contains scripts for validating the normalization of normalizing flow priors in the bilby parameter estimation framework.

## Overview

Proper normalization of custom priors is critical for Bayesian inference. These scripts verify that normalizing flow priors integrate to unity, handle parameter transformations correctly, and produce valid probability densities compatible with bilby's sampling framework.

## Key Scripts

### `validate_nf_normalization.py` - Comprehensive normalization validation

Main validation script that tests NF prior normalization through multiple methods.

**Validation Tests:**

**1. Monte Carlo Integration**
- Samples from prior extensively
- Numerically integrates probability density
- Checks if integral ≈ 1 within tolerance

**2. Log Probability Consistency**
- Evaluates log_prob at sampled points
- Verifies probability densities are positive
- Checks for numerical stability issues

**3. Parameter Transform Jacobians**
- Tests bijective transformations (unit cube ↔ physical parameters)
- Validates Jacobian corrections applied correctly
- Ensures volume elements preserved

**4. Boundary Behavior**
- Tests probability at parameter bounds
- Verifies constraints enforced (q ∈ [0.1, 1], λ > 0)
- Checks for probability leakage outside valid region

**5. Comparison with bilby Built-ins**
- Samples from NF prior via bilby
- Compares to direct NF sampling
- Validates bilby integration correctness

**Usage:**
```bash
# Validate specific NF model
python validate_nf_normalization.py --use-flowjax --population-type uniform --eos-samples-name radio

# Run all validation tests
python validate_nf_normalization.py --all-tests --verbose

# Test with high sample count for precision
python validate_nf_normalization.py --n-samples 1000000

# Validate specific parameter bounds
python validate_nf_normalization.py --check-bounds --plot-boundary
```

**Output:**
- Normalization integral value and uncertainty
- Pass/fail status for each test
- Diagnostic plots showing probability densities
- Warnings for any normalization issues

**Expected Results:**
- Integral within [0.99, 1.01] (1% tolerance)
- No negative probabilities
- Smooth behavior at boundaries
- Consistent bilby/direct sampling

### `bilby_tutorial.py` - bilby custom prior tutorial

Educational script demonstrating how to implement custom priors in bilby.

**Covers:**
- Creating custom prior classes
- Implementing `prob()` and `ln_prob()` methods
- Handling parameter transformations
- Rescaling between parameter spaces
- Integration with bilby samplers

**Usage:**
```bash
# Run tutorial
python bilby_tutorial.py
```

**Use Cases:**
- Learning bilby prior interface
- Testing custom prior implementations
- Debugging prior integration issues
- Template for new prior types

### `plot_bilby_corner.py` - Plot bilby prior samples

Generates corner plots from bilby prior samples to visually validate prior behavior.

**Purpose:**
- Visual check of prior distributions
- Compare bilby sampling to expected distribution
- Identify sampling artifacts or biases
- Quick diagnostic for prior issues

**Usage:**
```bash
# Sample and plot bilby prior
python plot_bilby_corner.py --prior-file ../GW_runs/GW170817/bns_prior.json

# Use NF prior
python plot_bilby_corner.py --use-nf --population-type uniform --eos-samples-name radio

# High sample count for smooth distributions
python plot_bilby_corner.py --n-samples 100000 --output prior_samples.pdf
```

**Output:**
- Corner plot of prior samples
- Marginal distributions
- Correlation structure

## Normalization Issues and Solutions

### Common Normalization Problems

**Problem 1: Integral ≠ 1**

**Symptoms:**
- Monte Carlo integral significantly different from 1
- Bayesian evidence values unreliable
- Bayes factors incorrect

**Causes:**
- Missing Jacobian corrections in transforms
- Incorrect probability density calculation
- Parameter bounds not properly enforced

**Solutions:**
- Review transform implementation
- Add missing Jacobian factors
- Validate against analytic examples

**Problem 2: Negative Probabilities**

**Symptoms:**
- `prob()` returns negative values
- Runtime errors in sampler
- NaN in log probabilities

**Causes:**
- Numerical instability in transformations
- Incorrect handling of log-space operations
- Boundary effects near parameter limits

**Solutions:**
- Use `log_prob()` instead of `prob()` where possible
- Add numerical safeguards (clip, abs)
- Validate boundary conditions

**Problem 3: bilby Sampling Issues**

**Symptoms:**
- Sampler fails to initialize
- Very low acceptance rates
- Posteriors don't match expected behavior

**Causes:**
- Prior rescaling incorrect
- Unit cube mapping wrong
- Incompatible with bilby's assumptions

**Solutions:**
- Test rescale methods independently
- Verify unit cube ↔ parameter mapping
- Check bilby prior interface implementation

### Validation Workflow

**Step 1: Direct NF Validation**
```bash
# Test NF model in isolation
cd ../NFprior/
python evaluate_flows.py --use-flowjax --population-type uniform
```

**Step 2: bilby Integration Test**
```bash
cd ../normalization/
python validate_nf_normalization.py --use-flowjax --population-type uniform
```

**Step 3: Visual Inspection**
```bash
python plot_bilby_corner.py --use-nf --population-type uniform --n-samples 50000
```

**Step 4: PE Test Run**
```bash
cd ../GW_runs/
python pe.py --GW-event GW170817 --prior-name bns --population-type uniform --nlive 500
```

**Step 5: Validate Evidence**
```bash
# Check log evidence is reasonable
# Compare BNS vs NSBH evidence values
```

## Directory Structure

```
normalization/
├── validate_nf_normalization.py    # Main validation script
├── bilby_tutorial.py               # Custom prior tutorial
├── plot_bilby_corner.py            # Prior sampling plots
├── figures/                        # Diagnostic plots
│   ├── normalization_test.pdf
│   ├── prior_samples.pdf
│   └── boundary_behavior.pdf
└── README.md
```

## Mathematical Background

### Normalization Condition

For valid probability density p(θ):
```
∫ p(θ) dθ = 1
```

Where integration is over full parameter space.

### Change of Variables

When transforming parameters θ → φ:
```
p_φ(φ) = p_θ(θ(φ)) |det J|
```

Where J is Jacobian matrix of transformation.

**Critical:** bilby rescaling handles unit cube [0,1] ↔ parameter space mapping. NF must properly account for this.

### Log Probability

Numerical stability requires working in log-space:
```
ln p(θ) = ln p_base(z) - ln |det J|
```

Where:
- p_base(z): Base distribution (e.g., Gaussian)
- z: Latent variable
- J: Jacobian of NF transformation

### Constrained Parameters

For constrained parameters (e.g., λ > 0):
```
p_constrained(λ) = p_unconstrained(f^{-1}(λ)) |df^{-1}/dλ|
```

Where f: R → R+ is bijection (e.g., softplus).

## Integration with bilby

### Custom Prior Class

NF priors inherit from `bilby.core.prior.Prior`:

**Required Methods:**
- `prob(val)`: Return probability density
- `ln_prob(val)`: Return log probability
- `sample(N)`: Generate N samples
- `rescale(val)`: Map unit cube [0,1] → parameter value

**Optional Methods:**
- `cdf(val)`: Cumulative distribution
- `is_in_prior_range(val)`: Check if value valid

### bilby Sampler Integration

**Unit Cube Sampling:**
1. Sampler proposes points in [0,1]^D
2. `rescale()` maps to parameter space
3. `ln_prob()` evaluated at rescaled point
4. Jacobian corrections applied automatically (if implemented correctly)

**Critical:** Ensure `rescale()` and `ln_prob()` are consistent and properly normalized.

## Common Workflows

### Validate New NF Model
```bash
# After training new NF
cd normalization/
python validate_nf_normalization.py --use-flowjax --population-type uniform --eos-samples-name new_eos --all-tests
```

### Debug Normalization Issue
```bash
# Detailed diagnostic
python validate_nf_normalization.py --verbose --plot-diagnostics --n-samples 1000000
```

### Quick Visual Check
```bash
# Generate prior samples
python plot_bilby_corner.py --use-nf --n-samples 50000
# Inspect for artifacts, biases, correct ranges
```

## Troubleshooting

### Normalization Off by Constant Factor

**Issue:** Integral consistently 0.9 or 1.1

**Fix:** Check for missing/extra Jacobian factors in transformations

### Normalization Test Passes but PE Fails

**Issue:** Validation succeeds but sampler crashes

**Fix:** Test with actual bilby sampler, not just Monte Carlo integration

### Prior Samples Outside Bounds

**Issue:** Corner plot shows samples violating constraints

**Fix:** Verify constraint enforcement in NF bijections and rescale method

## Best Practices

1. **Validate before using:** Always test normalization before production runs
2. **Use high sample counts:** N ≥ 100,000 for reliable integration
3. **Test edge cases:** Boundary behavior, extreme parameters
4. **Compare implementations:** Cross-check flowjax vs glasflow if available
5. **Document assumptions:** Note any approximations or numerical tricks
6. **Monitor evidence:** Track log evidence across runs for consistency