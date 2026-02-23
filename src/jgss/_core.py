"""Core solve() function implementing the JGSS algorithm.

This module orchestrates the three phases of Jacobian-Guided Subspace Search:
1. Jacobian Geometry Analysis - Extract active subspace via SVD
2. Basin Finding - Latin Hypercube Sampling in the subspace
3. Local Convergence - Levenberg-Marquardt to machine precision

With v1.1.0, solve() additionally supports:
- Multi-restart (n_restarts, default 5) for stochastic robustness
- Soft variable clamping (clip_bounds) for domain safety
- Scaled LM polish as automatic fallback when restarts don't converge

The solve() function is the main public API of the jgss package.
"""

from typing import Callable, List, Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

from jgss._convergence import levenberg_marquardt, scaled_lm_polish
from jgss._sampling import find_basin
from jgss._subspace import compute_jacobian, extract_active_subspace
from jgss._types import SolverResult


def _single_pass(
    residual_fn: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    x0: NDArray[np.float64],
    jacobian_fn: Optional[Callable[[NDArray[np.float64]], NDArray[np.float64]]],
    tol: float,
    maxiter: int,
    k_subspace: int,
    n_samples: int,
    bounds_arrays: Optional[Tuple[NDArray[np.float64], NDArray[np.float64]]],
    seed: Optional[int],
    verbose: int,
    callback: Optional[Callable[[NDArray[np.float64], NDArray[np.float64]], None]],
) -> Tuple[NDArray[np.float64], bool, str, int, int, NDArray[np.float64]]:
    """Execute one complete JGSS pass (Phase 1 + Phase 2 + Phase 3).

    Returns:
        Tuple of (x_final, converged, message, nfev, njev, fun).
    """
    total_nfev = 0
    total_njev = 0

    # Phase 1: Jacobian Geometry Analysis
    if verbose >= 2:
        n = len(x0)
        print(f"[JGSS] Phase 1: Computing Jacobian at initial guess (n={n})")

    J, nfev_jac, njev_jac = compute_jacobian(residual_fn, x0, jacobian_fn)
    total_nfev += nfev_jac
    total_njev += njev_jac

    n = len(x0)
    k_effective = min(k_subspace, n, J.shape[0])
    W, singular_values = extract_active_subspace(J, k_effective)

    if verbose >= 2:
        condition_num = (
            singular_values[0] / singular_values[-1] if len(singular_values) > 0 else float("inf")
        )
        print(f"    Active subspace dimension: {k_effective}")
        print(f"    Condition number: {condition_num:.2e}")
        print(f"    Singular values: [{singular_values[0]:.2e}, ..., {singular_values[-1]:.2e}]")

    # Phase 2: Basin Finding (LHS in Subspace)
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

    # Phase 3: Levenberg-Marquardt Convergence
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

    residual_norm = float(np.linalg.norm(final_fun))
    converged = residual_norm < tol

    return x_final, converged, lm_message, total_nfev, total_njev, final_fun


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
    n_restarts: int = 5,
    clip_bounds: Optional[Tuple[ArrayLike, ArrayLike]] = None,
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

    By default, solve() runs multiple independent restarts (n_restarts=5)
    with different random seeds and keeps the best result. If no restart
    converges, a scaled LM polish is applied as a final fallback.

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
        n_restarts: Number of independent restart attempts. Default 5.
            Each restart uses a different random seed for the LHS basin search.
            Set to 1 to match v1.0.0 behavior (single pass).
        clip_bounds: Optional (lower, upper) for soft variable clamping. Default None.
            Unlike bounds (which constrain the LM optimizer), clip_bounds applies
            np.clip to x0 and all intermediate results. Useful for keeping
            variables in economically meaningful ranges.
        seed: Random seed for reproducibility. Default None.
            With n_restarts>1, restart i uses seed=(seed+i) if seed is not None,
            or seed=i if seed is None.
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
            nfev: Total function evaluations across all restarts
            njev: Total Jacobian evaluations across all restarts
            fun: Residual vector at solution
            residual_norm: L2 norm of residual
            optimality: First-order optimality measure
            history: List of iterates (if requested)
            jac: Jacobian at solution (if requested)

    Raises:
        ValueError: If x0 is empty, bounds have wrong shape, or n_restarts < 1.

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

    if n_restarts < 1:
        raise ValueError(f"n_restarts must be >= 1, got {n_restarts}")

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

    # Validate and convert clip_bounds
    clip_lower: Optional[NDArray[np.float64]] = None
    clip_upper: Optional[NDArray[np.float64]] = None
    if clip_bounds is not None:
        cl, cu = clip_bounds
        clip_lower = np.asarray(cl, dtype=np.float64)
        clip_upper = np.asarray(cu, dtype=np.float64)

        if clip_lower.shape != (n,) and clip_lower.shape != ():
            raise ValueError(
                f"clip_bounds lower must be scalar or shape ({n},), got {clip_lower.shape}"
            )
        if clip_upper.shape != (n,) and clip_upper.shape != ():
            raise ValueError(
                f"clip_bounds upper must be scalar or shape ({n},), got {clip_upper.shape}"
            )

    # Apply clip_bounds to x0
    if clip_lower is not None and clip_upper is not None:
        x0 = np.clip(x0, clip_lower, clip_upper)

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
    # Multi-Restart Loop
    # =========================================================================
    best_x: Optional[NDArray[np.float64]] = None
    best_fun: Optional[NDArray[np.float64]] = None
    best_residual_norm = float("inf")
    best_message = ""
    converged = False

    for restart_idx in range(n_restarts):
        if seed is not None:
            restart_seed: Optional[int] = seed + restart_idx
        else:
            restart_seed = restart_idx

        if verbose >= 1 and n_restarts > 1:
            print(f"[JGSS] Restart {restart_idx + 1}/{n_restarts} (seed={restart_seed})")

        x_pass, pass_converged, pass_message, nfev_pass, njev_pass, fun_pass = _single_pass(
            residual_fn,
            x0,
            jacobian_fn,
            tol,
            maxiter,
            k_subspace,
            n_samples,
            bounds_arrays,
            restart_seed,
            verbose,
            callback,
        )
        total_nfev += nfev_pass
        total_njev += njev_pass

        # Apply clip_bounds to result
        if clip_lower is not None and clip_upper is not None:
            x_pass = np.clip(x_pass, clip_lower, clip_upper)
            fun_pass = residual_fn(x_pass)
            total_nfev += 1

        pass_residual_norm = float(np.linalg.norm(fun_pass))

        if pass_residual_norm < best_residual_norm:
            best_x = x_pass.copy()
            best_fun = fun_pass.copy()
            best_residual_norm = pass_residual_norm
            best_message = pass_message

        if pass_converged:
            converged = True
            break

    # =========================================================================
    # Scaled LM Polish (fallback when restarts don't converge)
    # Skip when bounds are active -- polish runs unbounded LM which would
    # violate the user's constraints.
    # =========================================================================
    if not converged and best_x is not None and bounds_arrays is None:
        if verbose >= 1:
            print("[JGSS] Scaled LM polish (fallback)")

        x_polish, polish_success, polish_msg, nfev_p, njev_p, fun_polish = scaled_lm_polish(
            residual_fn,
            best_x,
            jacobian_fn=jacobian_fn,
            tol=tol,
            maxiter=maxiter,
        )
        total_nfev += nfev_p
        total_njev += njev_p

        # Apply clip_bounds to polish result
        if clip_lower is not None and clip_upper is not None:
            x_polish = np.clip(x_polish, clip_lower, clip_upper)
            fun_polish = residual_fn(x_polish)
            total_nfev += 1

        polish_residual_norm = float(np.linalg.norm(fun_polish))

        if polish_residual_norm < best_residual_norm:
            best_x = x_polish
            best_fun = fun_polish
            best_residual_norm = polish_residual_norm
            best_message = polish_msg

        converged = best_residual_norm < tol

    # =========================================================================
    # Build Result
    # =========================================================================
    assert best_x is not None
    assert best_fun is not None

    x_final = best_x
    final_fun = best_fun
    residual_norm = best_residual_norm
    success = converged

    # Determine status message
    if success:
        message = f"Converged: residual norm {residual_norm:.2e} < tol {tol:.1e}"
    else:
        message = f"Not converged: {best_message}"

    # Compute optimality (gradient norm) if we have the Jacobian
    optimality = 0.0
    final_jac: Optional[NDArray[np.float64]] = None

    if return_jacobian or success:
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
        if n_restarts > 1:
            print(f"    Restarts used: {min(restart_idx + 1, n_restarts)}/{n_restarts}")

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
