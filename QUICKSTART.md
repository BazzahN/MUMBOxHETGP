# Quick Start Guide: hetGPy Heteroscedastic GP Experiments

This guide helps you get started with the hetGPy-based heteroscedastic GP experiments in 5 minutes.

## Installation (if needed)

```bash
# Install hetGPy
pip install hetgpy

# Or from source (recommended)
git clone --recurse-submodules https://github.com/davidogara/hetGPy.git
cd hetGPy && pip install -e .
```

## File Structure

```
hetGPy/examples/
├── data_fit_hetgpy.ipynb              # Main notebook with examples
├── hetgp_exp_utils.py                 # Model wrapper class
├── hetgp_exp_script.py                # Fit models and save results
├── hetgp_input_generation.py          # Generate synthetic data
├── run_example.sh                     # Automated workflow
├── configs/
│   └── exp_k25_example.yml            # Example configuration
├── HETGP_ADAPTATION_README.md         # Full documentation
├── CONVERSION_GUIDE.md                # BoTorch to hetGPy mapping
└── QUICKSTART.md                      # This file
```

## Option 1: Interactive Notebook (Easiest)

### Step 1: Open the Jupyter notebook
```bash
cd ~/hetGPy/examples
jupyter notebook data_fit_hetgpy.ipynb
```

### Step 2: Run through the cells
The notebook demonstrates:
1. Loading example data
2. Fitting a heteroscedastic GP model
3. Making predictions
4. Creating a visualization (sausage plot)
5. Comparing to the original BoTorch approach

**Note:** You need some data first. See "Option 2" below to generate data.

## Option 2: Automated Workflow (Fastest)

### Step 1: Run the example script
```bash
cd ~/hetGPy/examples
bash run_example.sh
```

This will:
1. Generate 5 synthetic datasets
2. Fit a model to each
3. Save predictions and hyperparameters
4. Create `exp_k25_example/` directory with all results

### Step 2: View results
```bash
ls exp_k25_example/
# Output:
# Input/    - generated training/test data
# Data/     - predictions and hyperparameters
```

### Step 3: Visualize (optional)
```bash
jupyter notebook data_fit_hetgpy.ipynb
# Then modify paths to point to exp_k25_example/
```

## Option 3: Manual Step-by-Step

### Step 1: Generate data
```bash
cd ~/hetGPy/examples

# Create your config first (see configs/exp_k25_example.yml)
# Then generate data:
python hetgp_input_generation.py --config configs/exp_k25_example.yml --n_macros 10
```

**Output:** Creates `exp_k25_example/Input/` with numpy files
- `train_x_m*.npy`, `train_y_m*.npy` (for each macro 0-9)
- `test_x.npy`, `test_y.npy`, `test_sigma2.npy` (shared)

### Step 2: Fit model(s)
```bash
# Fit a single macro replicate
python hetgp_exp_script.py --config configs/exp_k25_example.yml --macro 0

# Or fit all in a loop
for i in {0..9}; do
    python hetgp_exp_script.py --config configs/exp_k25_example.yml --macro $i
done
```

**Output:** Creates `exp_k25_example/Data/` with:
- Predictions: `pred_f_m*.npy`, `pred_sigma2_f_m*.npy`, `pred_sigma2_eps_m*.npy`
- Hyperparameters: `hyperparameters_m*.json`

### Step 3: Analyze results
```python
import json
import numpy as np

# Load predictions
pred_mean = np.load("exp_k25_example/Data/pred_f_m0.npy")
pred_epi = np.load("exp_k25_example/Data/pred_sigma2_f_m0.npy")  # epistemic
pred_ale = np.load("exp_k25_example/Data/pred_sigma2_eps_m0.npy")  # aleatoric

# Load hyperparameters
with open("exp_k25_example/Data/hyperparameters_m0.json") as f:
    hypers = json.load(f)
    print(f"Lengthscale: {hypers['theta']}")
    print(f"Nugget: {hypers['g']}")
    print(f"Log-likelihood: {hypers['llhood']}")
```

## Key Concepts

### Model Output

The hetGPy model returns three uncertainty components:

```
y(x) ~ N(mean(x), sd2(x) + nugs(x))
       └─────────┬──────────┘  └──┬──┘  └──┬──┘
         posterior mean   epistemic   aleatoric
                        (what the GP     (what the
                         doesn't know)  data/process tells us)
```

### Configuration File

A minimal config (see `configs/exp_k25_example.yml`):

```yaml
problem:
  k: 25                    # Design points
  n: 1                     # Replicates per point
  x_min: 0.0
  x_max: 1.0
  test_function_index: 1   # 0/1: Sine, 2: Bump, 3: Cosine
  noise_function_index: 0  # 0: Linear, 1: Sine, 2: Peak
  phi: 0.6
  tau: 0.7
  seed: 12345
misc:
  n_grid: 200
hetgp:
  covtype: "Matern5_2"
  maxit: 100
  standardise: true
```

## API Cheat Sheet

### Fit a model
```python
from hetgp_exp_utils import HetGP_Model

model = HetGP_Model(covtype="Matern5_2", maxit=100)
hyperparams = model.fit(train_x, train_y, standardise=True)
```

### Make predictions
```python
preds = model.predict(test_x)
# Keys: mean, sd2, nugs, mean_unstd, sd2_unstd, nugs_unstd
print(preds['mean'])              # Posterior mean
print(preds['sd2'])               # Epistemic uncertainty
print(preds['nugs'])              # Aleatoric uncertainty
print(preds['sd2'] + preds['nugs'])  # Total uncertainty
```

### Extract hyperparameters
```python
hypers = model._extract_hyperparameters()
print(f"Lengthscale: {hypers['theta']}")
print(f"Noise lengtscale: {hypers['theta_g']}")
print(f"Nugget: {hypers['g']}")
print(f"Variance: {hypers['nu_hat']}")
print(f"Log-likelihood: {hypers['ll']}")
```

## Data Formats

### Input files (generated by `hetgp_input_generation.py`)
- **train_x_mX.npy**: Training locations (n, 1) for macro X
- **train_y_mX.npy**: Training observations (n,) for macro X
- **test_x.npy**: Test grid (m, 1) - shared
- **test_y.npy**: True function values (m,) - shared
- **test_sigma2.npy**: True noise variance (m,) - shared

### Output files (created by `hetgp_exp_script.py`)
- **pred_f_mX.npy**: Predicted mean (m,)
- **pred_sigma2_f_mX.npy**: Predicted epistemic uncertainty (m,)
- **pred_sigma2_eps_mX.npy**: Predicted aleatoric uncertainty (m,)
- **pred_*_unstd_mX.npy**: Same, but in original (unstandardized) scale
- **hyperparameters_mX.json**: Hyperparameters as JSON

## Troubleshooting

### Problem: "hetgpy module not found"
```bash
pip install hetgpy
```

### Problem: Data files not found
- Check paths are correct (relative to current directory)
- Run `python hetgp_input_generation.py` first to generate data

### Problem: Model doesn't converge
- Increase `maxit` in config (e.g., 200 or 500)
- Try different `covtype` (e.g., "Gaussian" or "Matern3_2")
- Check data quality (remove outliers, normalize)

### Problem: Negative variances in output
- This shouldn't happen (hetGPy thresholds to zero automatically)
- If it does, consider increasing precision by passing `eps=1e-8` to `mleHetGP()`

## Next Steps

1. **Run the notebook:** `jupyter notebook data_fit_hetgpy.ipynb`
2. **Try different configs:** Create a new `configs/my_experiment.yml`
3. **Scale up:** Increase `k` (design points) or `n` (replicates)
4. **Compare:** See `CONVERSION_GUIDE.md` for BoTorch comparison

## Comparison: BoTorch vs hetGPy

| Feature | BoTorch VI-HGP | hetGPy MLE |
|---------|---|---|
| Data size | Large (n > 10k) | Medium (n < 5k) ✓ |
| Speed | Slow | Fast ✓ |
| GPU | Yes | No |
| Simplicity | Complex | Simple ✓ |
| R Compatible | No | Yes ✓ |

## Additional Resources

- **Documentation:** [hetGPy Docs](https://hetgpy.readthedocs.io/)
- **GitHub:** [hetGPy Repo](https://github.com/davidogara/hetGPy)
- **R Package:** [hetGP on CRAN](https://cran.r-project.org/web/packages/hetGP/)
- **Paper:** [Binois & Gramacy (2021)](https://www.jstatsoft.org/article/view/v098i13)

## File Descriptions

### Main Files

**hetgp_exp_utils.py** (150 lines)
- `output_handler`: Standardization/destandardization
- `HetGP_Model`: High-level model wrapper
  - `fit()`: Fit to training data
  - `predict()`: Make predictions
  - `_extract_hyperparameters()`: Get MLE estimates

**hetgp_exp_script.py** (150 lines)
- Main entry point for fitting experiments
- Loads data from numpy files
- Fits model and saves results as numpy + JSON
- Usage: `python hetgp_exp_script.py --config file.yml --macro 0`

**hetgp_input_generation.py** (200 lines)
- Generate synthetic experimental data
- Multiple test functions (Sine, Bump, Cosine)
- Multiple noise models (Linear, Sine, Peak)
- Creates numpy files in structured format
- Usage: `python hetgp_input_generation.py --config file.yml --n_macros 10`

### Notebooks

**data_fit_hetgpy.ipynb** (7 sections)
- Import libraries
- Load data
- Fit model
- Make predictions
- Visualize with sausage plot
- Quantitative assessment
- Comparison to BoTorch

### Documentation

**HETGP_ADAPTATION_README.md**
- Full documentation of the adaptation
- Workflow examples
- API documentation
- When to use hetGPy vs BoTorch

**CONVERSION_GUIDE.md**
- Detailed mapping of BoTorch to hetGPy components
- Algorithm comparison
- Hyperparameter structure differences
- Practical conversion examples

**QUICKSTART.md** (this file)
- Quick reference for getting started
- Common workflows
- API cheat sheet
- Troubleshooting
