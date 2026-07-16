"""
Utilities for hetGPy-based experiments.
Adapted from BoTorch VI-HGP implementation to use hetGPy with numpy arrays.
"""
import numpy as np
from hetgpy import hetGP
from scipy.stats import norm

def calc_log_likelihood(targets,f_preds,sigma2_preds):
    '''
    Calculates the loglikelihood for a gaussian 
    '''

    sqr_term = (targets - f_preds) ** 2
    print(f"sqr_term: {targets.shape}")  # Debug print to check values
    print(f"f_preds: {f_preds.shape}")  # Debug print to check values
    print(f"sigma2_preds: {sigma2_preds.shape}")  # Debug print to check values
    log_likelihood = -0.5 * (np.log(2.0 * np.pi) + np.log(sigma2_preds) + sqr_term/sigma2_preds)

    print(f"log_likelihood: {log_likelihood}")  # Debug print to check values
    ll_sum = log_likelihood.sum()

    return ll_sum
    



class output_handler:
    """Handles standardization and transformations of output data."""

    def __init__(self):
        """Sets parameters so that standardisation does nothing initially."""
        self.sig_std = 1.0
        self.mu_std = 0.0

    def standardise_and_update(self, y):
        """
        Standardises the input data to N(0,1) distribution and updates transformation parameters.
        
        Parameters
        ----------
        y : ndarray
            The training data to be standardised (shape: (n,) or (n, 1))
        
        Returns
        -------
        ndarray
            Standardized data
        """
        y = np.asarray(y).flatten()
        self.mu_std = y.mean()
        self.sig_std = y.std()
        if self.sig_std == 0:
            self.sig_std = 1.0  # Avoid division by zero
        return (y - self.mu_std) / self.sig_std

    def standardise(self, y):
        """
        Standardises the input data to N(0,1) distribution.
        
        Parameters
        ----------
        y : ndarray
            The training data to be standardised (shape: (n,) or (n, 1))
        
        Returns
        -------
        ndarray
            Standardized data
        """
        y = np.asarray(y).flatten()
        return (y - self.mu_std) / self.sig_std

    def unstandardise(self, y_std):
        """
        Reverts standardised input back to its original scale.
        
        Parameters
        ----------
        y_std : ndarray
            Standardized data
        
        Returns
        -------
        ndarray
            Data in original scale
        """
        y_std = np.asarray(y_std).flatten()
        return y_std * self.sig_std + self.mu_std


class HetGP_Model:
    """
    Wrapper for hetGPy heteroscedastic GP model.
    Provides a simplified interface for fitting and prediction.
    """

    def __init__(self, covtype="Matern5_2", maxit=100, verbose=False):
        """
        Initialize HetGP model wrapper.
        
        Parameters
        ----------
        covtype : str, optional
            Covariance kernel type: 'Gaussian', 'Matern5_2', or 'Matern3_2'
        maxit : int, optional
            Maximum iterations for optimization
        verbose : bool, optional
            Print optimization information
        """
        self.covtype = covtype
        self.maxit = maxit
        self.verbose = verbose
        self.model = None
        self.output_transform = None

    def fit(self, train_x, train_y, standardise=False):
        """
        Fit the heteroscedastic GP model.
        
        Parameters
        ----------
        train_x : ndarray
            Training input locations (shape: (n, d))
        train_y : ndarray
            Training outputs (shape: (n,))
        standardise : bool, optional
            Whether to standardise outputs before fitting
        
        Returns
        -------
        dict
            Dictionary with model hyperparameters
        """
        train_x = np.asarray(train_x)
        train_y = np.asarray(train_y).flatten()

        # Standardise outputs if requested
        self.output_transform = output_handler()
        if standardise:
            train_y = self.output_transform.standardise_and_update(train_y)
        else:
            self.output_transform.standardise_and_update(train_y)  # Initialize transform

        # Fit hetGP model
        trace = 1 if self.verbose else -1
        self.model = hetGP()
        self.model.mleHetGP(
            X=train_x,
            Z=train_y,
            covtype=self.covtype,
            maxit=self.maxit,
            settings={"trace": trace,"checkHom": False},
        )

        # Extract hyperparameters
        hyperparams = self._extract_hyperparameters()
        return hyperparams

    def predict(self, test_x):
        """
        Make predictions at test locations.
        
        Parameters
        ----------
        test_x : ndarray
            Test input locations (shape: (m, d))
        
        Returns
        -------
        dict
            Dictionary with keys:
            - 'mean': predicted means (shape: (m,))
            - 'sd2': epistemic uncertainty (shape: (m,))
            - 'nugs': aleatoric uncertainty (shape: (m,))
            - 'mean_unstd': predictions in original scale (if standardised)
        """
        test_x = np.asarray(test_x)
        if len(test_x.shape) == 1:
            test_x = test_x.reshape(-1, 1)

        if self.model is None:
            raise ValueError("Model must be fitted before making predictions")

        # Get predictions from hetGP
        preds = self.model.predict(test_x)

        # Unstandardise predictions if output was standardised
        if self.output_transform is not None:
            mean_unstd = self.output_transform.unstandardise(preds["mean"])
            # Scale uncertainties by the standardization factor
            sd2_unstd = preds["sd2"] * (self.output_transform.sig_std)
            nugs_unstd = preds["nugs"] * (self.output_transform.sig_std)
        else:
            mean_unstd = preds["mean"]
            sd2_unstd = preds["sd2"]
            nugs_unstd = preds["nugs"]

        return {
            "mean": preds["mean"],
            "sd2": preds["sd2"],
            "nugs": preds["nugs"],
            "mean_unstd": mean_unstd,
            "sd2_unstd": sd2_unstd,
            "nugs_unstd": nugs_unstd,
        }

    def _extract_hyperparameters(self):
        """
        Extract hyperparameters from the fitted model.
        
        Returns
        -------
        dict
            Dictionary with model hyperparameters
        """
        if self.model is None:
            return {}

        model = self.model
        print(f"Test: {model['theta'].item()} is {type(model['theta'].item())}")  # Debug print to check values
        hyperparams = {
            "l_f": model["theta"][0],  # Lengthscale for mean process
            "tau": model["nu_hat"],  # Variance estimate
            "l_g": model["theta_g"].item(),  # Lengthscale for noise process
            "g": model["g"],  # Nugget of noise process
            #"Delta": model["Delta"],  # Raw nugget estimates at training locations
            #"Lambda": model["Lambda"],  # Smoothed noise predictions
            "mu": model["beta0"],  # Mean constant
            ##"ll": model.get("ll"),  # Log-likelihood
            #"k_theta_g": model.get("k_theta_g"),  # Lengthscale ratio if present
        }
        return hyperparams

    def get_model_summary(self):
        """
        Get a summary of the fitted model.
        
        Returns
        -------
        dict
            Summary statistics of the model
        """
        if self.model is None:
            return {"status": "Model not fitted"}

        model = self.model
        return {
            "status": "fitted",
            "num_unique_locations": len(model["X0"]),
            "input_dim": model["X0"].shape[1],
            "l_f": model["theta"],
            "l_g": model["theta_g"],
            "tau": model["nu_hat"],
            "g": model["g"],
            "log_likelihood": model.get("ll"),
            "covtype": self.covtype,
        }
