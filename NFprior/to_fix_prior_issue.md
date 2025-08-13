# Critical Normalization Issue in NFDist Implementation

## Problem Summary

The `NFDist` class in `bilby/bilby/core/prior/joint.py` (to find it, use grep!!!) has a critical normalization issue that breaks proper probability density calculation for nested sampling. The implementation is missing the Jacobian correction for the MinMaxScaler transformation, leading to unnormalized prior distributions.

## Mathematical Details

### The Issue

In the `_ln_prob` method (lines 1252-1257), the code performs the following steps:

1. Transform samples to scaled space: `samp_scaled = scaler.transform(samp)`
2. Evaluate normalizing flow log probability: `log_prob = nf.log_prob(samp_scaled)`
3. Return `log_prob` directly **without Jacobian correction**

### The Mathematics

For a probability density transformation from space X to space Y via function f: X → Y, the correct probability density is:

```
p_X(x) = p_Y(f(x)) * |det(J_f(x))|
```

In log space:
```
log p_X(x) = log p_Y(f(x)) + log |det(J_f(x))|
```

### MinMaxScaler Transformation

The MinMaxScaler applies element-wise transformation:
```
x_scaled[i] = (x[i] - min[i]) / (max[i] - min[i])
```

The Jacobian matrix is diagonal:
```
J[i,i] = 1 / (max[i] - min[i])
J[i,j] = 0 for i ≠ j
```

The log determinant is:
```
log |det(J)| = sum_i log(1 / (max[i] - min[i])) = -sum_i log(max[i] - min[i])
```

### Current Implementation Error

**Current code:**
```python
if self.scaler is not None:
    samp = self.scaler.transform(samp)
log_probs = self.nf_ln_prob(samp)
# Missing: log_probs += log_det_jacobian
```

**Correct implementation:**
```python
if self.scaler is not None:
    samp = self.scaler.transform(samp)
    log_det_jacobian = -np.sum(np.log(self.scaler.data_max_ - self.scaler.data_min_))
else:
    log_det_jacobian = 0.0

log_probs = self.nf_ln_prob(samp) + log_det_jacobian
```

## Impact on Results

### Single Model Analysis
- Prior probabilities are systematically offset by constant `log_det_jacobian`
- Affects evidence calculation: `Z_corrected = Z_raw * exp(log_det_jacobian)`
- Parameter posteriors maintain correct shape but wrong normalization

### Model Comparison (Critical Issue)
When comparing flows trained on different datasets (e.g., BNS vs NSBH):

- Flow 1 (BNS): `correction_1 = -sum(log(max1[i] - min1[i]))`
- Flow 2 (NSBH): `correction_2 = -sum(log(max2[i] - min2[i]))`

**Bayes Factor Error:**
```
log(BF_raw) = log(Z_BNS_raw) - log(Z_NSBH_raw)
log(BF_correct) = log(BF_raw) + (correction_1 - correction_2)
```

The correction terms do NOT cancel because different datasets have different parameter ranges, leading to **systematically biased source classification**.

## Files Affected

1. **Primary:** `bilby/bilby/core/prior/joint.py` - Lines 1252-1257 in `NFDist._ln_prob`
2. **Analysis:** All gravitational wave parameter estimation results using NFDist
3. **Scientific Impact:** BNS vs NSBH classification conclusions

## Proposed Fix

### Code Change Required
```python
# In NFDist._ln_prob method around line 1254:
if self.scaler is not None:
    samp = self.scaler.transform(samp)
    
    # Add Jacobian correction for MinMaxScaler transformation
    log_det_jacobian = -np.sum(np.log(self.scaler.data_max_ - self.scaler.data_min_))
else:
    log_det_jacobian = 0.0
    
log_probs = self.nf_ln_prob(samp)

# Apply Jacobian correction to ensure proper normalization
log_probs = log_probs + log_det_jacobian
```

### Verification Steps
1. Confirm scalers exist for all trained flows
2. Calculate correction constants for each flow model
3. Verify that corrected posteriors integrate to 1 (within numerical precision)
4. Recompute Bayes factors with corrected evidence values

## Urgency

**HIGH PRIORITY** - This issue affects the fundamental scientific conclusions about neutron star source classification. All existing results using NFDist priors should be considered potentially unreliable for model comparison purposes until this correction is applied.

## Alternative: Postprocessing Correction

If reruns are computationally prohibitive, the correction can be applied in postprocessing since the Jacobian term is constant for each flow:

```python
# For each flow model:
jacobian_correction = -np.sum(np.log(scaler.data_max_ - scaler.data_min_))
evidence_corrected = evidence_raw * np.exp(jacobian_correction)
```

However, direct code fix is strongly preferred for future reliability.