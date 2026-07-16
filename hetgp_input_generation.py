"""
Input generation script for hetGPy experiments.
Generates synthetic experimental data using test functions.

Adapted from BoTorch version to use numpy arrays instead of torch tensors.
"""

import numpy as np
import argparse
import yaml
from pathlib import Path


class TestFunction:
    """Base class for test functions."""

    def __init__(self, noise_function=None, phi=0.6, tau=0.7, rng_state=None):
        """
        Initialize test function.
        
        Parameters
        ----------
        noise_function : callable
            Function to generate noise
        phi : float
            Noise function parameter
        tau : float
            Noise function parameter
        rng_state : np.random.RandomState, optional
            Random state for reproducibility
        """
        self.noise_function = noise_function
        self.phi = phi
        self.tau = tau
        self.rng = rng_state if rng_state is not None else np.random.RandomState()

    def __call__(self, x):
        """Evaluate function at location x."""
        raise NotImplementedError

    def get_noise(self, x):
        """Get noise level at location x."""
        if self.noise_function is None:
            return 0
        return self.noise_function(x, phi=self.phi, tau=self.tau)


class Sine1D(TestFunction):
    """1D sine test function."""

    def __call__(self, x):
        """Evaluate sine function."""
        x = np.asarray(x)
        return np.sin(6 * np.pi * x)


class Bump1D(TestFunction):
    """1D bump test function."""

    def __call__(self, x):
        """Evaluate bump function."""
        x = np.asarray(x)
        return np.exp(-20 * ((x - 0.5) ** 2))


class Cosine1D(TestFunction):
    """1D cosine test function."""

    def __call__(self, x):
        """Evaluate cosine function."""
        x = np.asarray(x)
        return np.cos(6 * np.pi * x)


def noise_linear(x, phi=0.6, tau=0.7):
    """Linear noise model."""
    x = np.asarray(x)
    return phi + tau * x


def noise_sine(x, phi=0.6, tau=0.7):
    """Sine-modulated noise model."""
    x = np.asarray(x)
    return phi + tau * np.sin(4 * np.pi * x)


def noise_peak(x, phi=0.6, tau=0.7):
    """Peak-modulated noise model."""
    x = np.asarray(x)
    return phi + tau * np.exp(-20 * ((x - 0.5) ** 2))


# Test function and noise function registries
TEST_FUNCTIONS = {
    0: Sine1D,
    1: Sine1D,  # Default
    2: Bump1D,
    3: Cosine1D,
}

NOISE_FUNCTIONS = {
    0: noise_linear,
    1: noise_sine,
    2: noise_peak,
}


def get_nxk_initial_evals(k, n, test_func, x_min, x_max, rng):
    """
    Generate n*k initial evaluations at k unique locations.
    
    Parameters
    ----------
    k : int
        Number of unique design locations
    n : int
        Number of replicates at each location
    test_func : TestFunction
        Test function to evaluate
    x_min : float
        Lower bound of domain
    x_max : float
        Upper bound of domain
    rng : np.random.RandomState
        Random state for reproducibility
    
    Returns
    -------
    tuple
        (train_x, train_y, test_x, test_y, test_sigma2)
    """
    # Generate k unique locations
    x_unique = rng.uniform(x_min, x_max, size=k)
    x_unique = np.sort(x_unique)

    # Generate n replicates at each location
    train_x = np.repeat(x_unique, n)

    # Evaluate function and add noise
    train_y = np.zeros_like(train_x)
    for i, x in enumerate(train_x):
        f_val = test_func(x)
        noise_std = np.sqrt(np.maximum(test_func.get_noise(x), 1e-8))
        noise = rng.normal(0, noise_std)
        train_y[i] = f_val + noise

    # Create test grid
    test_x = np.linspace(x_min, x_max, 200)
    test_y = test_func(test_x)
    
    # Get true noise variance at test points
    test_sigma2 = np.array([test_func.get_noise(x) for x in test_x])

    return train_x.reshape(-1, 1), train_y, test_x.reshape(-1, 1), test_y, test_sigma2


def spawn_generators(master_seed, M):
    """
    Spawn M independent random generators from a master seed.
    
    Parameters
    ----------
    master_seed : int
        Master random seed
    M : int
        Number of generators to spawn
    
    Returns
    -------
    list
        List of M np.random.RandomState objects
    """
    master_rng = np.random.RandomState(master_seed)
    seeds = master_rng.randint(0, 2**31 - 1, M)
    return [np.random.RandomState(int(s)) for s in seeds]


def main():
    """Main data generation function."""
    
    # Import arguments from command line
    parser = argparse.ArgumentParser(
        description="Generate synthetic experimental data for hetGPy modeling"
    )
    parser.add_argument("--config", required=True, help="Path to config YAML file")
    parser.add_argument(
        "--n_macros", type=int, required=True,
        help="Number of macro replicates to generate"
    )
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Extract experiment name from config path
    exp_name = Path(args.config).stem

    # Decant relevant variables from config
    exp_args = config["problem"]
    misc_args = config["misc"]

    M = args.n_macros  # Number of data replicates
    k = exp_args["k"]  # Number of unique locations
    n = exp_args["n"]  # Replicates at each location
    x_min = exp_args["x_min"]
    x_max = exp_args["x_max"]

    test_function_index = exp_args["test_function_index"]
    noise_function_index = exp_args["noise_function_index"]
    phi = exp_args["phi"]
    tau = exp_args["tau"]
    seed = exp_args["seed"]
    n_grid = misc_args["n_grid"]

    print(f"=== Data Generation: {exp_name} ===")
    print(f"Number of macro replicates: {M}")
    print(f"Number of unique locations: {k}")
    print(f"Replicates per location: {n}")
    print(f"Domain: [{x_min}, {x_max}]")
    print(f"Test function: {test_function_index}")
    print(f"Noise function: {noise_function_index}")
    print()

    # Create output directory
    outdir = Path(exp_name) / "Input"
    outdir.mkdir(parents=True, exist_ok=True)

    # Spawn random generators
    generators = spawn_generators(seed, M)

    # Get test function and noise function
    test_func_class = TEST_FUNCTIONS.get(test_function_index, Sine1D)
    noise_func = NOISE_FUNCTIONS.get(noise_function_index, noise_linear)

    # Generate data for each macro replicate
    train_x_all = []
    train_y_all = []
    
    print(f"Generating {M} datasets...")
    for macro_idx, rng in enumerate(generators):
        test_func = test_func_class(
            noise_function=noise_func,
            phi=phi,
            tau=tau,
            rng_state=rng
        )

        train_x, train_y, test_x, test_y, test_sigma2 = get_nxk_initial_evals(
            k, n, test_func, x_min, x_max, rng
        )

        # Save training data for this macro
        np.save(outdir / f"train_x_m{macro_idx}.npy", train_x)
        np.save(outdir / f"train_y_m{macro_idx}.npy", train_y)

        train_x_all.append(train_x)
        train_y_all.append(train_y)

        if macro_idx == 0:
            # Save test data only once (same for all macros)
            np.save(outdir / "test_x.npy", test_x)
            np.save(outdir / "test_y.npy", test_y)
            np.save(outdir / "test_sigma2.npy", test_sigma2)

        print(f"  Macro {macro_idx}: train_x shape {train_x.shape}, train_y shape {train_y.shape}")

    print(f"\nData generation complete!")
    print(f"Output directory: {outdir}/")
    print(f"Files generated:")
    print(f"  - train_x_m*.npy, train_y_m*.npy (for each macro {M} files)")
    print(f"  - test_x.npy, test_y.npy, test_sigma2.npy (shared across macros)")


if __name__ == "__main__":
    main()
