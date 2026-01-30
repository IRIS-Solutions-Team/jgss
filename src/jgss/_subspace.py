"""Jacobian computation and active subspace extraction utilities.

This module provides functions for computing Jacobians (analytically or via
finite differences) and extracting the active subspace using SVD. The active
subspace captures the directions of highest sensitivity in the residual function.
"""

from typing import Callable, Optional, Tuple

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray


def compute_jacobian(
    residual_fn: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    x: NDArray[np.float64],
    jacobian_fn: Optional[Callable[[NDArray[np.float64]], NDArray[np.float64]]] = None,
    epsilon: float = 1e-8,
) -> Tuple[NDArray[np.float64], int, int]:
    """Compute the Jacobian matrix of a residual function.

    If jacobian_fn is provided, uses the analytical Jacobian. Otherwise,
    computes a finite-difference approximation column by column.

    Args:
        residual_fn: Function f(x) -> residuals of shape (m,).
        x: Point at which to evaluate the Jacobian. Shape (n,).
        jacobian_fn: Optional analytical Jacobian function J(x) -> (m, n) matrix.
        epsilon: Step size for finite-difference approximation.

    Returns:
        Tuple of:
            - J: Jacobian matrix of shape (m, n)
            - nfev: Number of function evaluations used
            - njev: Number of Jacobian evaluations used (1 if analytical, 0 if FD)
    """
    x = np.asarray(x, dtype=np.float64)

    if jacobian_fn is not None:
        # Use analytical Jacobian
        J = jacobian_fn(x)
        # Handle sparse matrices (convert to dense)
        if hasattr(J, "toarray"):
            J = J.toarray()
        return np.asarray(J, dtype=np.float64), 0, 1

    # Finite-difference approximation
    f0 = residual_fn(x)
    m = len(f0)
    n = len(x)
    J = np.zeros((m, n), dtype=np.float64)

    nfev = 1  # Already called residual_fn once for f0

    for j in range(n):
        x_plus = x.copy()
        x_plus[j] += epsilon
        f_plus = residual_fn(x_plus)
        J[:, j] = (f_plus - f0) / epsilon
        nfev += 1

    return J, nfev, 0


def extract_active_subspace(
    J: NDArray[np.float64],
    k: int,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Extract the active subspace from a Jacobian matrix using SVD.

    The active subspace captures the k directions of highest sensitivity
    (largest singular values) in the Jacobian. This enables dimensionality
    reduction for high-dimensional optimization problems.

    Args:
        J: Jacobian matrix of shape (m, n).
        k: Number of singular vectors to retain (subspace dimension).

    Returns:
        Tuple of:
            - W: Projection matrix of shape (n, k). Maps k-dim subspace to
                 full n-dim space via x_full = x0 + W @ z_subspace.
            - singular_values: All singular values from SVD, for diagnostics.

    Note:
        The projection matrix W = Vt[:k, :].T contains the right singular
        vectors corresponding to the k largest singular values. These
        represent the directions of greatest sensitivity in the input space.
    """
    J = np.asarray(J, dtype=np.float64)

    # Handle sparse matrices
    if hasattr(J, "toarray"):
        J = J.toarray()

    # SVD: J = U @ diag(s) @ Vt
    # full_matrices=False is critical for performance on wide/tall matrices
    U, s, Vt = la.svd(J, full_matrices=False)

    # Clamp k to valid range
    n = J.shape[1]
    k = min(k, n, len(s))

    # Extract top-k rows of Vt (corresponding to largest singular values)
    # Transpose to get W: (n, k) mapping from subspace to full space
    W = Vt[:k, :].T

    return W, s
