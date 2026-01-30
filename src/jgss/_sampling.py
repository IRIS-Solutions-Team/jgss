"""Latin Hypercube Sampling for basin-of-attraction search.

This module implements the global search phase of JGSS. It uses Latin Hypercube
Sampling (LHS) in the reduced subspace to efficiently explore the parameter
space and find a good starting point for local convergence.
"""

from typing import Callable, Optional, Tuple

import numpy as np
import scipy.stats.qmc as qmc
from numpy.typing import NDArray
from scipy.stats import norm


def find_basin(
    residual_fn: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    x0: NDArray[np.float64],
    W: NDArray[np.float64],
    n_samples: int,
    bounds: Optional[Tuple[NDArray[np.float64], NDArray[np.float64]]] = None,
    seed: Optional[int] = None,
    verbose: int = 0,
    callback: Optional[Callable[[NDArray[np.float64], NDArray[np.float64]], None]] = None,
) -> Tuple[NDArray[np.float64], float, int]:
    """Find the basin of attraction using Latin Hypercube Sampling in subspace.

    Performs a global search by sampling points in the reduced subspace
    defined by the projection matrix W. Each sample is projected back to
    the full space, evaluated, and the best point (lowest SSR) is returned.

    Args:
        residual_fn: Function f(x) -> residuals of shape (m,).
        x0: Initial guess / center point. Shape (n,).
        W: Projection matrix from extract_active_subspace. Shape (n, k).
        n_samples: Number of LHS samples to evaluate.
        bounds: Optional (lower, upper) bounds for clipping. Each shape (n,).
        seed: Random seed for reproducibility.
        verbose: Verbosity level. 0=silent, >=2 prints sampling progress.
        callback: Optional function called with (x, f) for each evaluation.

    Returns:
        Tuple of:
            - best_x: Point with lowest SSR found. Shape (n,).
            - best_ssr: Sum of squared residuals at best_x.
            - nfev: Number of function evaluations performed.
    """
    x0 = np.asarray(x0, dtype=np.float64).copy()
    n = len(x0)
    k = W.shape[1]  # Subspace dimension

    # Validate bounds
    if bounds is not None:
        lower, upper = bounds
        lower = np.asarray(lower, dtype=np.float64)
        upper = np.asarray(upper, dtype=np.float64)
    else:
        lower = None
        upper = None

    # Initialize tracking
    best_ssr = float("inf")
    best_x = x0.copy()
    nfev = 0

    # Evaluate initial guess first
    try:
        f0 = residual_fn(x0)
        nfev += 1
        ssr0 = float(np.sum(f0**2))
        if np.isfinite(ssr0) and ssr0 < best_ssr:
            best_ssr = ssr0
            best_x = x0.copy()
        if callback is not None:
            callback(x0, f0)
    except Exception:
        pass  # Initial guess failed, continue with sampling

    if verbose >= 2:
        print(f"    [Basin Search] Starting with {n_samples} LHS samples in {k}D subspace")
        if best_ssr < float("inf"):
            print(f"    [Basin Search] Initial SSR: {best_ssr:.4e}")

    # Latin Hypercube Sampling for optimal space coverage
    sampler = qmc.LatinHypercube(d=k, seed=seed)
    sample_points_01 = sampler.random(n=n_samples)

    # Map [0,1] to Gaussian-distributed coefficients via Inverse CDF (Probit)
    z_samples = norm.ppf(sample_points_01)

    # Log-uniform scaling for perturbation magnitude
    # Vary the "kick" size from 10^-3 to 10^0.5 (~3.16)
    rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()
    magnitudes = 10 ** rng.uniform(-3, 0.5, n_samples)

    # Subspace search loop
    improved_count = 0
    for i in range(n_samples):
        z_vec = z_samples[i]
        scale = magnitudes[i]

        # Project reduced vector z back to full space
        # x_trial = x0 + W @ (z * scale)
        perturbation = W @ (z_vec * scale)
        x_trial = x0 + perturbation

        # Apply bounds clipping if provided
        if lower is not None and upper is not None:
            x_trial = np.clip(x_trial, lower, upper)

        try:
            res = residual_fn(x_trial)
            nfev += 1
            ssr = float(np.sum(res**2))

            if np.isfinite(ssr) and ssr < best_ssr:
                best_ssr = ssr
                best_x = x_trial.copy()
                improved_count += 1

            if callback is not None:
                callback(x_trial, res)

        except Exception:
            # Skip failed evaluations
            nfev += 1
            continue

        # Progress reporting for verbose >= 2
        if verbose >= 2 and (i + 1) % 100 == 0:
            print(f"    [Basin Search] {i + 1}/{n_samples} samples, best SSR: {best_ssr:.4e}")

    if verbose >= 2:
        print(f"    [Basin Search] Complete. Found {improved_count} improvements.")
        print(f"    [Basin Search] Final best SSR: {best_ssr:.4e}")

    return best_x, best_ssr, nfev
