"""
Experiment script for hetGPy heteroscedastic GP modeling.
Adapted from BoTorch VI-HGP implementation to use hetGPy with numpy arrays.

Algo takes the input data with specified macro prefix and fits the model.
It then calculates predictions over a shared test grid for:
- the latent function (mean)
- the models epistemic uncertainty (sd2)
- the systems aleatoric uncertainty (nugs)

Hyperparameters are also exported as JSON.
"""

import argparse
import yaml
import json
import numpy as np
from pathlib import Path
from hetgp_exp_utils import HetGP_Model,calc_log_likelihood

def export_json(exp_name, dict_data, macro):
    """
    Export dictionary as JSON file.
    
    Parameters
    ----------
    exp_name : str
        Experiment name
    dict_data : dict
        Data to export
    macro : int
        Macro/replicate index
    """
    outdir = Path(exp_name) / "Data"
    outdir.mkdir(parents=True, exist_ok=True)

    # Convert numpy arrays to lists for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.generic):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        return obj

    data_serializable = convert_to_serializable(dict_data)

    with open(outdir / f"hyperparameters_m{macro}.json", "w") as f:
        json.dump(data_serializable, f, indent=2)


def main():
    """Main experiment execution function."""
    
    # Import arguments from command line
    parser = argparse.ArgumentParser(
        description="Fit heteroscedastic GP model using hetGPy"
    )
    parser.add_argument("--config", required=True, help="Path to config YAML file")
    parser.add_argument(
        "--macro", type=int, required=False, default=0,
        help="Macro/replicate index"
    )
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Extract experiment name from config path
    exp_name = Path(args.config).stem
    macro = args.macro

    # Decant relevant variables from config
    exp_args = config["problem"]
    misc_args = config["misc"]
    gp_args = config.get("hetgp", {})

    # Problem constants
    k = exp_args["k"]  # Number of points
    n_grid = misc_args["n_grid"]

    # GP Arguments
    covtype = gp_args.get("covtype", "Matern5_2")
    maxit = gp_args.get("maxit", 100)
    standardise = gp_args.get("standardise", True)

    print(f"=== Experiment: {exp_name} ===")
    print(f"Macro: {macro}")
    print(f"Number of training points: {k}")
    print(f"Covariance type: {covtype}")
    
    # Step 1: Import Data
    indir = Path(exp_name) / "Input"
    
    # Load training data
    train_x = np.load(indir / f"train_x_m{macro}.npy")
    train_y = np.load(indir / f"train_y_m{macro}.npy")
    
    # Load test data
    test_x = np.load(indir / "test_x.npy")
    test_y = np.load(indir / "test_y.npy")
    test_sigma2 = np.load(indir / "test_sigma2.npy")

    print(f"Train data shape: X={train_x.shape}, Y={train_y.shape}")
    print(f"Test data shape: X={test_x.shape}, Y={test_y.shape}")

    outdir = Path(exp_name) / "Data"
    outdir.mkdir(parents=True, exist_ok=True)

    # Step 2: Fit hetGP Model
    print("\nFitting hetGP model...")
    model_wrapper = HetGP_Model(
        covtype=covtype, maxit=maxit, verbose=True
    )
    
    hyperparameters = model_wrapper.fit(
        train_x=train_x, 
        train_y=train_y, 
        standardise=standardise
    )

    print("Model fitted successfully!")
    print("\nModel summary:")
    summary = model_wrapper.get_model_summary()
    for key, value in summary.items():
        if isinstance(value, np.ndarray):
            print(f"  {key}: {value.tolist()}")
        else:
            print(f"  {key}: {value}")

    # Step 3: Obtain predictions over test grid
    print(f"\nMaking predictions on {len(test_x)} test points...")
    preds = model_wrapper.predict(test_x)
    ## Calculate log-likelihood of test data under the model
    preds_train = model_wrapper.predict(train_x)
    llhood = calc_log_likelihood(
        targets=train_y.flatten(),
        f_preds=preds_train['mean_unstd'],
        sigma2_preds=preds_train['nugs_unstd']
    )
    # Step 4: Save results
    
    # Save predictions (in standardised space)
    np.save(outdir / f"pred_f_m{macro}.npy", preds["mean"])
    np.save(outdir / f"pred_sigma2_f_m{macro}.npy", preds["sd2"])
    np.save(outdir / f"pred_sigma2_eps_m{macro}.npy", preds["nugs"])

    # Save predictions (in original scale)
    np.save(outdir / f"pred_f_unstd_m{macro}.npy", preds["mean_unstd"])
    np.save(outdir / f"pred_sigma2_f_unstd_m{macro}.npy", preds["sd2_unstd"])
    np.save(outdir / f"pred_sigma2_eps_unstd_m{macro}.npy", preds["nugs_unstd"])

    # Export Hyperparameters and likelihood as JSON
    hyperparameters["ll"] = float(llhood) if not np.isnan(llhood) else None
    # hyperparameters["standardised"] = standardise
    hyperparameters["standardisation_mean"] = float(model_wrapper.output_transform.mu_std)
    hyperparameters["standardisation_std"] = float(model_wrapper.output_transform.sig_std)
    
    export_json(exp_name, hyperparameters, macro)

    print(f"\nResults saved to {outdir}/")
    print(f"  - Predictions: pred_*_m{macro}.npy (standardised)")
    print(f"  - Predictions: pred_*_unstd_m{macro}.npy (original scale)")
    print(f"  - Hyperparameters: hyperparameters_m{macro}.json")
    print("\nExperiment complete!")


if __name__ == "__main__":
    main()
