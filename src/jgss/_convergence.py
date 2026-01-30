"""Levenberg-Marquardt convergence wrapper using scipy.optimize.least_squares.

This module wraps scipy's least_squares optimizer to provide the local
convergence phase of the JGSS algorithm. After the global basin search
finds a good starting point, LM converges to machine precision.
"""

from typing import Callable, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares


def levenberg_marquardt(
    residual_fn: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    x0: NDArray[np.float64],
    jacobian_fn: Optional[Callable[[NDArray[np.float64]], NDArray[np.float64]]] = None,
    tol: float = 1e-8,
    maxiter: int = 500,
    bounds: Optional[Tuple[NDArray[np.float64], NDArray[np.float64]]] = None,
    verbose: int = 0,
    callback: Optional[Callable[[NDArray[np.float64], NDArray[np.float64]], None]] = None,
) -> Tuple[NDArray[np.float64], bool, str, int, int, NDArray[np.float64]]:
    """Run Levenberg-Marquardt optimization for local convergence.

    Wraps scipy.optimize.least_squares to solve nonlinear least squares
    problems. Uses 'lm' method for unbounded problems or 'trf' (Trust Region
    Reflective) when bounds are provided.

    Args:
        residual_fn: Function f(x) -> residuals of shape (m,).
        x0: Initial guess. Shape (n,).
        jacobian_fn: Optional analytical Jacobian function.
        tol: Convergence tolerance for both ftol and xtol.
        maxiter: Maximum number of function evaluations = maxiter * n.
        bounds: Optional (lower, upper) bounds tuple.
        verbose: Verbosity level. >=1 prints iteration progress.
        callback: Optional function called with (x, f) after each iteration.

    Returns:
        Tuple of:
            - x_final: Solution vector. Shape (n,).
            - success: True if converged.
            - message: Status message from optimizer.
            - nfev: Number of function evaluations.
            - njev: Number of Jacobian evaluations.
            - fun: Residual vector at solution. Shape (m,).
    """
    x0 = np.asarray(x0, dtype=np.float64).copy()
    n = len(x0)

    # Configure Jacobian
    if jacobian_fn is not None:
        # Wrap to handle sparse matrices
        def jac_wrapper(x: NDArray[np.float64]) -> NDArray[np.float64]:
            J = jacobian_fn(x)
            if hasattr(J, "toarray"):
                J = J.toarray()
            return np.asarray(J, dtype=np.float64)

        jac: Union[str, Callable[[NDArray[np.float64]], NDArray[np.float64]]] = jac_wrapper
    else:
        jac = "2-point"  # Finite difference approximation

    # Track iterations for verbose output and callback
    iteration_count = [0]

    # Wrap residual function to support callback and verbose
    def wrapped_residual(x: NDArray[np.float64]) -> NDArray[np.float64]:
        res = residual_fn(x)
        iteration_count[0] += 1

        if callback is not None:
            callback(x, res)

        if verbose >= 1 and iteration_count[0] % 10 == 0:
            ssr = float(np.sum(res**2))
            residual_norm = float(np.linalg.norm(res))
            print(
                f"    [LM] Iteration {iteration_count[0]:4d}: "
                f"SSR={ssr:.4e}, |f|={residual_norm:.4e}"
            )

        return res

    # Determine method based on bounds
    if bounds is not None:
        lower, upper = bounds
        lower = np.asarray(lower, dtype=np.float64)
        upper = np.asarray(upper, dtype=np.float64)
        method = "trf"  # Trust Region Reflective supports bounds
        bounds_arg: Union[Tuple[NDArray, NDArray], str] = (lower, upper)
    else:
        method = "lm"  # Levenberg-Marquardt for unbounded
        bounds_arg = (-np.inf, np.inf)

    # Note: method='lm' doesn't support bounds, so we switch to 'trf' when bounded
    # max_nfev is the total function evaluations allowed
    max_nfev = maxiter * n

    if verbose >= 1:
        print(f"    [LM] Starting Levenberg-Marquardt (method={method}, tol={tol:.1e})")

    try:
        result = least_squares(
            wrapped_residual,
            x0,
            jac=jac,
            method=method,
            ftol=tol,
            xtol=tol,
            gtol=tol,
            max_nfev=max_nfev,
            bounds=bounds_arg,
            verbose=0,  # We handle verbosity ourselves
        )

        x_final = result.x
        success = result.success
        message = result.message
        nfev = result.nfev
        njev = result.njev if hasattr(result, "njev") and result.njev is not None else 0
        fun = result.fun

    except Exception as e:
        # Handle optimization failures gracefully
        x_final = x0.copy()
        success = False
        message = f"Optimization failed: {str(e)}"
        nfev = iteration_count[0]
        njev = 0
        fun = residual_fn(x0)

    if verbose >= 1:
        ssr = float(np.sum(fun**2))
        status = "CONVERGED" if success else "NOT CONVERGED"
        print(f"    [LM] {status}: SSR={ssr:.4e}, nfev={nfev}, njev={njev}")
        print(f"    [LM] {message}")

    return x_final, success, message, nfev, njev, fun
