# BoTorch VI-HGP to hetGPy MLE: Conversion Guide

This document explains how the original BoTorch variational inference heteroscedastic GP implementation has been adapted to use hetGPy with MLE.

## Original Architecture (BoTorch VI-HGP)

### Model Structure
```
Two Latent Processes:
├── f(x): Mean function
│   └── Uses MaternKernel(nu=2.5) with inducing points
└── g(x): Log-noise process
    └── Uses MaternKernel(nu=2.5) with inducing points

Likelihood:
├── Observational model: y_i ~ N(f(x_i), exp(g(x_i)))
└── Expected log-probability computed via:
    - Moment-based ELBO (not sampling-based)
    - KL divergence for variational distribution

Optimization:
└── L-BFGS-B on ELBO (includes KL term + expected log-likelihood)
```

### Key Components (from BoTorch)

1. **HeteroscedasticLatentGP**: Two-latent process VI model
   - Variational distribution per latent process
   - Independent multitask variational strategy
   - Learnable inducing locations

2. **HeteroscedasticGaussianLikelihood**: Custom likelihood
   - Computes expected log-probability under both latents
   - Handles noise transformation: σ²(x) = exp(g(x))

3. **HeteroscedasticELBO**: Custom variational ELBO
   - Sums expected log-likelihood and negative KL divergence

4. **Output Handler**: Standardization
   - Transforms data to N(0,1)
   - Inverse transform for predictions

## Adapted Architecture (hetGPy MLE)

### Model Structure
```
Two Processes (not explicitly latent):
├── Mean process: GP with covariance kernel
│   └── Lengthscale: theta
│   └── Nugget: Lambda/mult (derived from noise process)
└── Noise process: GP for log-noise (or noise directly)
    └── Lengthscale: theta_g
    └── Nugget: g
    └── Smoothed predictions: Lambda

Inference Model:
├── Joint MLE of all hyperparameters
├── Noise variance modeled as: Λ = smooth_gp(Delta, X0)
│   where Delta are point-wise noise estimates
└── Covariance: K = C + diag(Lambda/mult)

Optimization:
└── L-BFGS-B directly on marginal likelihood (with optional penalty)
```

### Key Components (from hetGPy)

1. **hetGP class**: Main model object
   - `mleHetGP()`: Fits heteroscedastic GP via MLE
   - `predict()`: Returns mean, epistemic uncertainty, aleatoric uncertainty

2. **Covariance Functions**: Multiple kernel options
   - Gaussian (RBF)
   - Matern5_2 (ν=2.5)
   - Matern3_2 (ν=3/2)

3. **Hyperparameter Linking**:
   - Joint: theta_g = k_theta_g * theta
   - Constrained: lower bound from homoskedastic fit
   - Independent: separate optimization

4. **Output Handler**: Same standardization logic
   - Transforms data to N(0,1)
   - Inverse transform for predictions

## Mapping of Components

### Data Handling

**BoTorch VI-HGP:**
```python
# Input: PyTorch tensors with GPU support
train_x: Tensor (k*n, d)
train_y: Tensor (k*n,)
train_n: Tensor (k,) # Replicates at each point
sigma2_hat: Tensor (k,) # Empirical variance estimates
```

**hetGPy MLE:**
```python
# Input: NumPy arrays
X: ndarray (n, d)  # All observations (with replicates)
Z: ndarray (n,)    # All observations
# OR
X0: ndarray (k, d)   # Unique locations
Z0: ndarray (k,)     # Mean at unique locations
mult: ndarray (k,)   # Replicates at each location
```

**Mapping:**
- hetGPy automatically detects replicates via `find_reps()`
- No need to manually compute `mult` or `Z0`
- Simpler input interface

### Model Fitting

**BoTorch VI-HGP:**
```python
vhgp = VI_HGP(n_u=int(n_u), iters=800, standardise=True)
model, out_transform, hyperparameters = vhgp.get_VI_HGP_model(
    train_x=train_x,
    train_n=train_n,
    train_y=train_y,
    sigma2_hat=sigma2_hat
)

# Returns:
# - model: GPyTorch model with variational distribution
# - out_transform: StandardScaler object
# - hyperparameters: dict with theta, k_theta_g, etc.
```

**hetGPy MLE:**
```python
model = hetGP()
model.mleHetGP(
    X=train_x,
    Z=train_y,
    covtype="Matern5_2",
    maxit=100
)

# Model stored as dictionary-like object with:
# - theta, theta_g, g, Delta, Lambda
# - nu_hat, beta0, nmean
# - Ki, Kgi (inverse covariance matrices)
```

**Key Differences:**
- No inducing points needed (uses full data)
- No variational distribution (direct MLE)
- Simpler hyperparameter set
- Returns covariance matrix inverses for efficient prediction

### Predictions

**BoTorch VI-HGP:**
```python
preds = _predict_vihgp(
    grid_x=test_x,
    model=model,
    outcome_transform=out_transform
)
# Returns tuple: (mean, sd2, nugs)
# Need to unstandardize manually

# With uncertainty:
preds['upper'] = norm.ppf(0.95, loc=mean, scale=sqrt(sd2 + nugs))
preds['lower'] = norm.ppf(0.05, loc=mean, scale=sqrt(sd2 + nugs))
```

**hetGPy MLE:**
```python
preds = model.predict(test_x)
# Returns dict with keys:
#   - mean: posterior mean
#   - sd2: epistemic variance (model uncertainty)
#   - nugs: aleatoric variance (noise prediction)
#   - sd2_var: variance of noise process (optional)
# Automatically handles standardization with model object
```

**Key Differences:**
- Dictionary-based output (clearer semantics)
- Built-in standardization handling
- Optional noise process variance
- Simpler API overall

### Hyperparameter Structure

**BoTorch VI-HGP hyperparameters:**
```python
{
    'theta_f': [lengthscale(s) for mean process],
    'theta_g': [lengthscale(s) for noise process],
    'outputscale_f': output scale for mean process,
    'outputscale_g': output scale for noise process,
    'k_theta_g': ratio linking theta_f and theta_g,
    'means': [mean constants for each latent],
    'nugs': [nugget estimates],
}
```

**hetGPy MLE hyperparameters:**
```python
{
    'theta': lengthscale(s) for mean process,
    'theta_g': lengthscale(s) for noise process,
    'g': nugget for noise process,
    'nu_hat': variance estimate (output scale),
    'beta0': mean constant,
    'k_theta_g': ratio if linkThetas='joint',
    'Delta': point-wise noise estimates,
    'Lambda': smoothed noise predictions,
    'nmean': mean of noise process,
    'll': log-likelihood value,
}
```

**Key Differences:**
- hetGPy is more explicit about noise model components
- All hyperparameters at training data locations available
- Log-likelihood value directly accessible
- No separate output scales (implicit in nu_hat)

## Algorithm Comparison

### BoTorch VI-HGP (Variational Inference)

```
1. Initialize inducing points (subset of data)
2. Initialize variational distribution q(u) where u is inducing variables
3. Repeat:
   a. Sample from q(u) for f and g
   b. Compute expected log p(y|f,g) - KL[q(u)||p(u)]
   c. Update variational parameters via gradient descent
   d. Update hyperparameters via L-BFGS-B on ELBO
   e. Check convergence
4. Return model with learned variational distribution
```

**Complexity:** O(m³) where m = num inducing points
**Iterations:** 800 (user-specified)
**Time:** Minutes to hours (depending on data size and GPU)

### hetGPy MLE (Direct Maximum Likelihood)

```
1. Initialize hyperparameters (via homoskedastic GP or user-specified)
2. Identify replicates and compute Delta (point-wise noise estimates)
3. Repeat:
   a. Compute Lambda (smoothed noise via noise GP)
   b. Compute K = C + diag(Lambda/mult)
   c. Compute marginal likelihood (with optional penalty)
   d. Update hyperparameters via L-BFGS-B
   e. Check convergence
4. Return model with learned hyperparameters
```

**Complexity:** O(n³) for Cholesky decomposition
**Iterations:** 100 (typical, much fewer needed)
**Time:** Seconds to minutes (CPU-based, usually faster for moderate n < 5000)

## Uncertainty Decomposition

### How hetGPy Returns Uncertainties

The `predict()` method returns two components:

1. **Epistemic Uncertainty (sd2):**
   - Model uncertainty: σ²_f(x) = cov_f(x,x) - cov_f(x,X) K⁻¹ cov_f(X,x)
   - Measures uncertainty about the function value
   - Decreases with more data or longer lengthscale

2. **Aleatoric Uncertainty (nugs):**
   - Noise prediction: σ²_ε(x) = exp(g_mean(x)) or smooth_gp(Delta)(x)
   - Predicts the noise level at location x
   - Independent of training data directly

**Total Predictive Uncertainty:**
```
σ²_total(x) = sd2(x) + nugs(x)
```

This matches the BoTorch formulation where:
```
σ²_total(x) = E[σ²_f(x|u)] + E[exp(g(x))]
```

## Practical Conversion Example

### Original BoTorch Code:
```python
import torch
from exp_utils import VI_HGP

TKWARGS = {"dtype": torch.double, "device": torch.device("cpu")}

# Load data
train_x = torch.load("data/train_x.pt").to(**TKWARGS)
train_y = torch.load("data/train_y.pt").to(**TKWARGS)

# Fit model
vhgp = VI_HGP(n_u=10, iters=800, standardise=True)
model, transform, hypers = vhgp.get_VI_HGP_model(
    train_x=train_x,
    train_n=train_n,
    train_y=train_y,
    sigma2_hat=sigma2_hat
)

# Predict
preds = predict_vihgp(grid_x, model, transform)
mean, sd2, nugs = preds

# Export
torch.save(mean, "results/pred_f.pt")
```

### Equivalent hetGPy Code:
```python
import numpy as np
from hetgp_exp_utils import HetGP_Model

# Load data
train_x = np.load("data/train_x.npy")
train_y = np.load("data/train_y.npy")

# Fit model
model = HetGP_Model(covtype="Matern5_2", maxit=100)
hypers = model.fit(train_x, train_y, standardise=True)

# Predict
preds = model.predict(grid_x)
mean = preds['mean']
sd2 = preds['sd2']
nugs = preds['nugs']

# Export
np.save("results/pred_f.npy", mean)
np.save("results/pred_sigma2_f.npy", sd2)
np.save("results/pred_sigma2_eps.npy", nugs)
```

## Summary of Trade-offs

| Aspect | BoTorch VI-HGP | hetGPy MLE |
|--------|---|---|
| **Scalability** | Better (inducing points) | Limited (full data) |
| **GPU Support** | Yes | No |
| **Customization** | High | Low |
| **Simplicity** | Complex | Simple |
| **Training Time** | Longer | Shorter |
| **Memory Usage** | Depends on m (inducing) | O(n²) for covariance |
| **Reproducibility** | PyTorch version dependent | Highly reproducible |
| **R Compatibility** | No | Yes (matches R hetGP) |
| **Hyperparameter Interpretation** | Indirect (learned) | Direct MLE estimates |

## When to Use Each

**Use BoTorch VI-HGP when:**
- Large-scale problems (n > 10,000)
- GPU acceleration needed
- Complex acquisition functions for BO
- Integrating with BoTorch ecosystem

**Use hetGPy when:**
- Moderate-sized problems (100 < n < 5000)
- Reproducibility with R hetGP important
- Simple, interpretable MLE estimates preferred
- Don't need GPU or inducing point approximation
- Fast training more important than extreme scalability
