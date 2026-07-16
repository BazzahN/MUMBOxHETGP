# hetGPy Experiments: Adaptation from BoTorch VI-HGP

This directory contains an adaptation of the heteroscedastic GP experiments from the `Investigating Inference` folder, converted from the BoTorch variational inference approach to use the hetGPy package with NumPy arrays.

## Overview

The original implementation in `/home/newtonh3/Investigating Inference` used:
- **BoTorch** for Bayesian optimization and GP modeling
- **PyTorch** tensors for all computations
- **Variational Inference** with inducing points (SVGP)
- GPU-accelerated computations

This adapted version uses:
- **hetGPy** (Python implementation of R's hetGP package)
- **NumPy** arrays for all computations
- **Maximum Likelihood Estimation** (MLE) with L-BFGS-B optimizer
- CPU-based, but faster for moderate-sized datasets

## Key Files

### Main Scripts

1. **`hetgp_exp_utils.py`** - Utilities for hetGPy experiments
   - `output_handler`: Handles standardization and transformations
   - `HetGP_Model`: Wrapper class for hetGPy with simplified interface
   - Methods: `fit()`, `predict()`, `get_model_summary()`

2. **`hetgp_exp_script.py`** - Main experiment execution script
   - Loads data from Input directory
   - Fits heteroscedastic GP model
   - Generates predictions on test grid
   - Exports hyperparameters and predictions
   - Usage: `python hetgp_exp_script.py --config <config.yml> --macro <macro_index>`

3. **`hetgp_input_generation.py`** - Data generation script
   - Creates synthetic experimental data
   - Supports multiple test functions (Sine, Bump, Cosine)
   - Supports multiple noise models (Linear, Sine, Peak)
   - Generates numpy files for training/test data
   - Usage: `python hetgp_input_generation.py --config <config.yml> --n_macros <num>`

### Notebooks

1. **`data_fit_hetgpy.ipynb`** - Comprehensive example notebook
   - Demonstrates loading and fitting a hetGP model
   - Creates visualization (sausage plot) with uncertainty bands
   - Shows comparison to original BoTorch VI-HGP approach
   - Quantitative assessment of predictions

## Configuration Files

Example config files (should be placed in `configs/` subdirectory):

```yaml
problem:
  k: 10                      # Number of unique design points
  n: 1                       # Replicates per point
  x_min: 0.0
  x_max: 1.0
  test_function_index: 1     # 0/1: Sine, 2: Bump, 3: Cosine
  noise_function_index: 0    # 0: Linear, 1: Sine, 2: Peak
  phi: 0.6                   # Noise parameter
  tau: 0.7                   # Noise parameter
  seed: 12345

misc:
  n_grid: 200                # Test grid size

hetgp:
  covtype: "Matern5_2"       # Gaussian, Matern5_2, Matern3_2
  maxit: 100                 # Max optimization iterations
  standardise: true          # Standardize outputs before fitting
```

## Workflow

### 1. Generate Data

```bash
# Create config file in configs/exp_k10.yml
python hetgp_input_generation.py --config configs/exp_k10.yml --n_macros 10
```

Output structure:
```
exp_k10/
├── Input/
│   ├── train_x_m0.npy, train_y_m0.npy
│   ├── train_x_m1.npy, train_y_m1.npy
│   ├── ... (more macros)
│   ├── test_x.npy
│   ├── test_y.npy
│   └── test_sigma2.npy
```

### 2. Fit Models

```bash
# Fit single macro replicate
python hetgp_exp_script.py --config configs/exp_k10.yml --macro 0

# Or run multiple in a loop
for i in {0..9}; do
  python hetgp_exp_script.py --config configs/exp_k10.yml --macro $i
done
```

Output structure:
```
exp_k10/Data/
├── pred_f_m0.npy, pred_sigma2_f_m0.npy, pred_sigma2_eps_m0.npy  # standardized
├── pred_f_unstd_m0.npy, pred_sigma2_f_unstd_m0.npy, pred_sigma2_eps_unstd_m0.npy  # original scale
├── hyperparameters_m0.json
├── ... (more macros)
```

### 3. Analyze Results

Use the Jupyter notebook to load and visualize results:
```bash
jupyter notebook data_fit_hetgpy.ipynb
```

## Data Format

### Input Data

Training data should contain:
- **X**: Input locations (n × d array)
- **Z**: Observations (1D array of length n)

With replicates:
- Multiple observations at same location are handled automatically
- hetGPy internally finds replicates using `find_reps()`

### Output Predictions

From `model.predict(x)`:
- **mean**: Posterior mean prediction
- **sd2**: Epistemic uncertainty (model uncertainty)
- **nugs**: Aleatoric uncertainty (noise variance prediction)
- **sd2_var**: Variance of the noise prediction (optional)

## Hyperparameter Extraction

The fitted model contains:
- **theta**: Lengthscale(s) of mean process
- **theta_g**: Lengthscale(s) of noise process
- **g**: Nugget parameter for noise process
- **nu_hat**: Variance estimate of mean process
- **Delta**: Raw nugget estimates at training locations
- **Lambda**: Smoothed noise predictions
- **beta0**: Mean constant
- **k_theta_g**: Lengthscale ratio (if linkThetas='joint')
- **ll**: Log-likelihood value

## Key Differences from Original BoTorch Implementation

| Aspect | BoTorch VI-HGP | hetGPy MLE |
|--------|---|---|
| **Framework** | PyTorch, BoTorch | NumPy, hetGPy |
| **Inference** | Variational (SVGP) | MLE |
| **Optimization** | Adam, multi-GPU | L-BFGS-B |
| **Inducing Points** | Required for scaling | Not used |
| **GPU Support** | Yes | No |
| **Data Size** | Better for large (n > 5000) | Better for small-medium (n < 5000) |
| **Hyperparameters** | Variational + MLE | MLE only |
| **Standardization** | Optional | Optional |
| **Output** | Predicted latent processes | Mean + noise prediction |

## When to Use hetGPy

✅ **Use hetGPy when:**
- Working with small to medium-sized datasets (n < 5000)
- Need reproducibility with R hetGP package
- Prefer simplicity and straightforward MLE
- Working with 1D or low-dimensional inputs
- Don't need GPU acceleration
- Want faster training for moderate-sized problems

❌ **Use BoTorch VI-HGP when:**
- Working with large datasets (n > 10000)
- Need GPU acceleration for training
- Require extensive customization of kernels/likelihoods
- Building Bayesian optimization loops
- Want to leverage inducing point approximations

## API Examples

### Basic Usage

```python
from hetgp_exp_utils import HetGP_Model
import numpy as np

# Create and fit model
model = HetGP_Model(covtype="Matern5_2", maxit=100)
hyperparams = model.fit(train_x, train_y, standardise=True)

# Make predictions
preds = model.predict(test_x)
print(f"Mean predictions: {preds['mean']}")
print(f"Epistemic uncertainty: {preds['sd2']}")
print(f"Aleatoric uncertainty: {preds['nugs']}")

# Get model summary
summary = model.get_model_summary()
print(summary)
```

### Direct hetGPy Usage

```python
from hetgpy import hetGP
import numpy as np

# Create and fit model directly
model = hetGP()
model.mleHetGP(
    X=train_x,
    Z=train_y,
    covtype="Matern5_2",
    maxit=100
)

# Get predictions
preds = model.predict(test_x)
print(preds.keys())  # mean, sd2, nugs, sd2_var
```

## Troubleshooting

### Issue: "Module hetgpy not found"
**Solution:** Install hetGPy in your environment
```bash
pip install hetgpy
# or from source:
git clone --recurse-submodules https://github.com/davidogara/hetGPy.git
cd hetGPy && pip install -e .
```

### Issue: "Negative predictive variances"
**Solution:** This can happen due to numerical errors. hetGPy automatically thresholds these to zero.
To avoid: use higher precision or pass `eps` parameter to `mleHetGP()`.

### Issue: Model doesn't converge
**Solution:** Try increasing `maxit` or providing initial hyperparameter values in `init` dict.

## References

- **hetGPy**: https://hetgpy.readthedocs.io/
- **Original R hetGP**: https://cran.r-project.org/web/packages/hetGP/
- **JOSS Paper**: Binois & Gramacy (2021) - hetGP: Heteroskedastic Gaussian Process Modeling

## Notes

- This adaptation focuses on the core heteroscedastic GP modeling
- The experiment structure (configs, data generation, results export) mirrors the original
- All computations use NumPy, making it simpler to understand and debug
- Standardization is applied by default (can be disabled in config)
- Results are exported in both standardized and original scales
