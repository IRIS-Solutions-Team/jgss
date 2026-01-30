"""Core solve() function implementing the JGSS algorithm.

This module orchestrates the three phases of Jacobian-Guided Subspace Search:
1. Jacobian Geometry Analysis - Extract active subspace via SVD
2. Basin Finding - Latin Hypercube Sampling in the subspace
3. Local Convergence - Levenberg-Marquardt to machine precision

The solve() function is the main public API of the jgss package.
"""

from typing import Callable, List, Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

from jgss._convergence import levenberg_marquardt
from jgss._sampling import find_basin
from jgss._subspace import compute_jacobian, extract_active_subspace
from jgss._types import SolverResult


def solve(
    residual_fn: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    x0: ArrayLike,
    *,
    jacobian_fn: Optional[Callable[[NDArray[np.float64]], NDArray[np.float64]]] = None,
    tol: float = 1e-8,
    maxiter: int = 500,
    k_subspace: int = 30,
    n_samples: int = 600,
    bounds: Optional[Tuple[ArrayLike, ArrayLike]] = None,
    seed: Optional[int] = None,
    verbose: int = 0,
    callback: Optional[Callable[[NDArray[np.float64], NDArray[np.float64]], None]] = None,
    history: bool = False,
    return_jacobian: bool = False,
) -> SolverResult:
    """Solve a nonlinear system using Jacobian-Guided Subspace Search (JGSS).

    JGSS is a hybrid global-local optimization algorithm designed for
    high-dimensional nonlinear systems. It combines dimensionality reduction
    (via Jacobian SVD), global search (Latin Hypercube Sampling), and local
    convergence (Levenberg-Marquardt).

    Args:
        residual_fn: Function f(x) -> residuals. Must return array of shape (m,).
        x0: Initial guess. Array-like of shape (n,).
        jacobian_fn: Optional analytical Jacobian function J(x) -> (m, n) matrix.
            If None, uses finite-difference approximation.
        tol: Convergence tolerance for residual norm. Default 1e-8.
        maxiter: Maximum iterations for LM phase. Default 500.
        k_subspace: Dimension of active subspace for global search. Default 30.
        n_samples: Number of Latin Hypercube samples. Default 600.
        bounds: Optional (lower, upper) bounds tuple for constrained optimization.
        seed: Random seed for reproducibility. Default None.
        verbose: Verbosity level:
            -1: Silent (no output)
             0: Final summary only (default)
             1: Per-iteration output during LM
             2: Full debug (subspace info, basin search, LM)
        callback: Optional function called with (x, f) at each evaluation.
        history: If True, include list of solution iterates in result. Default False.
        return_jacobian: If True, include final Jacobian in result. Default False.

    Returns:
        SolverResult with fields:
            x: Solution vector
            success: True if converged (residual_norm < tol)
            message: Status message
            nfev: Total function evaluations
            njev: Total Jacobian evaluations
            fun: Residual vector at solution
            residual_norm: L2 norm of residual
            optimality: First-order optimality measure
            history: List of iterates (if requested)
            jac: Jacobian at solution (if requested)

    Raises:
        ValueError: If x0 is empty or bounds have wrong shape.

    Example:
        >>> import numpy as np
        >>> from jgss import solve
        >>> def system(x):
        ...     return x ** 2 - np.array([1.0, 4.0])
        >>> result = solve(system, np.array([0.5, 1.5]))
        >>> print(result.x)  # [1.0, 2.0]
        >>> print(result.success)  # True
    """
    # =========================================================================
    # Input Validation
    # =========================================================================
    x0 = np.asarray(x0, dtype=np.float64).copy()

    if x0.size == 0:
        raise ValueError("x0 must be non-empty")

    n = len(x0)

    # Validate and convert bounds
    bounds_arrays: Optional[Tuple[NDArray[np.float64], NDArray[np.float64]]] = None
    if bounds is not None:
        lower, upper = bounds
        lower = np.asarray(lower, dtype=np.float64)
        upper = np.asarray(upper, dtype=np.float64)

        if lower.shape != (n,) or upper.shape != (n,):
            raise ValueError(
                f"bounds must have shape ({n},) to match x0, "
                f"got lower={lower.shape}, upper={upper.shape}"
            )
        bounds_arrays = (lower, upper)

    # Initialize tracking
    total_nfev = 0
    total_njev = 0
    history_list: List[NDArray[np.float64]] = []

    # Wrap callback to track history if requested
    if history:
        original_callback = callback

        def tracking_callback(x: NDArray[np.float64], f: NDArray[np.float64]) -> None:
            history_list.append(x.copy())
            if original_callback is not None:
                original_callback(x, f)

        callback = tracking_callback

    # =========================================================================
    # Phase 1: Jacobian Geometry Analysis
    # =========================================================================
    if verbose >= 2:
        print(f"[JGSS] Phase 1: Computing Jacobian at initial guess (n={n})")

    J, nfev_jac, njev_jac = compute_jacobian(residual_fn, x0, jacobian_fn)
    total_nfev += nfev_jac
    total_njev += njev_jac

    # Determine effective subspace dimension
    k_effective = min(k_subspace, n, J.shape[0])

    W, singular_values = extract_active_subspace(J, k_effective)

    if verbose >= 2:
        condition_num = (
            singular_values[0] / singular_values[-1] if len(singular_values) > 0 else float("inf")
        )
        print(f"    Active subspace dimension: {k_effective}")
        print(f"    Condition number: {condition_num:.2e}")
        print(f"    Singular values: [{singular_values[0]:.2e}, ..., {singular_values[-1]:.2e}]")

    # =========================================================================
    # Phase 2: Basin Finding (LHS in Subspace)
    # =========================================================================
    if verbose >= 1:
        print(f"[JGSS] Phase 2: Basin search with {n_samples} LHS samples")

    best_x, best_ssr, nfev_basin = find_basin(
        residual_fn,
        x0,
        W,
        n_samples,
        bounds=bounds_arrays,
        seed=seed,
        verbose=verbose,
        callback=callback,
    )
    total_nfev += nfev_basin

    if verbose >= 1:
        print(f"    Basin found: SSR={best_ssr:.4e}")

    # =========================================================================
    # Phase 3: Levenberg-Marquardt Convergence
    # =========================================================================
    if verbose >= 1:
        print(f"[JGSS] Phase 3: Levenberg-Marquardt convergence (maxiter={maxiter})")

    x_final, lm_success, lm_message, nfev_lm, njev_lm, final_fun = levenberg_marquardt(
        residual_fn,
        best_x,
        jacobian_fn=jacobian_fn,
        tol=tol,
        maxiter=maxiter,
        bounds=bounds_arrays,
        verbose=verbose,
        callback=callback,
    )
    total_nfev += nfev_lm
    total_njev += njev_lm

    # =========================================================================
    # Build Result
    # =========================================================================
    residual_norm = float(np.linalg.norm(final_fun))
    success = residual_norm < tol

    # Determine status message
    if success:
        message = f"Converged: residual norm {residual_norm:.2e} < tol {tol:.1e}"
    else:
        message = f"Not converged: {lm_message}"

    # Compute optimality (gradient norm) if we have the Jacobian
    optimality = 0.0
    final_jac: Optional[NDArray[np.float64]] = None

    if return_jacobian or success:
        # Compute final Jacobian for optimality measure
        final_jac_computed, nfev_opt, njev_opt = compute_jacobian(residual_fn, x_final, jacobian_fn)
        total_nfev += nfev_opt
        total_njev += njev_opt

        # Optimality = ||J^T @ f|| (gradient of 0.5 * ||f||^2)
        grad = final_jac_computed.T @ final_fun
        optimality = float(np.linalg.norm(grad))

        if return_jacobian:
            final_jac = final_jac_computed

    # Final summary output
    if verbose >= 0:
        status = "CONVERGED" if success else "NOT CONVERGED"
        print(f"[JGSS] {status}")
        print(f"    Residual norm: {residual_norm:.4e}")
        print(f"    Evaluations: {total_nfev} function, {total_njev} Jacobian")

    return SolverResult(
        x=x_final,
        success=success,
        message=message,
        nfev=total_nfev,
        njev=total_njev,
        fun=final_fun,
        residual_norm=residual_norm,
        optimality=optimality,
        history=history_list if history else None,
        jac=final_jac,
    )
