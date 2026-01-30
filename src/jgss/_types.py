"""Public type definitions for the JGSS nonlinear solver package.

This module defines the core data structures used by jgss.solve():
- SolverConfig: Documents available hyperparameters and their defaults
- SolverResult: Immutable result container returned by solve()

These types form the public API contract for the JGSS solver.
"""

from dataclasses import dataclass, field, fields
from typing import Iterator, List, Optional

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class SolverConfig:
    """Configuration parameters for JGSS solver (for reference/introspection).

    Note: solve() takes these as keyword arguments, not a config object.
    This class documents the available parameters and their defaults.

    Attributes:
        tol: Convergence tolerance for residual norm. Default 1e-8.
        maxiter: Maximum number of solver iterations. Default 500.
        k_subspace: Dimension of the Jacobian-guided subspace. Default 30.
        n_samples: Number of Latin Hypercube samples for global search. Default 600.
        seed: Random seed for reproducibility. Default None (non-deterministic).
        verbose: Verbosity level (0=silent, 1=summary, 2=iterations). Default 0.
    """

    tol: float = 1e-8
    maxiter: int = 500
    k_subspace: int = 30
    n_samples: int = 600
    seed: Optional[int] = None
    verbose: int = 0


@dataclass(frozen=True)
class SolverResult:
    """Result of jgss.solve() - immutable container with dict-like access.

    This dataclass provides both attribute access (result.x) and dict-style
    access (result['x']) for compatibility with different usage patterns.

    Attributes:
        x: Solution vector. Shape (n,) where n is the number of variables.
        success: True if solver converged within tolerance.
        message: Human-readable status message describing the outcome.
        nfev: Number of function evaluations performed.
        njev: Number of Jacobian evaluations performed.
        fun: Residual vector at the solution. Shape (m,) where m is number of equations.
        residual_norm: L2 norm of the residual at the solution.
        optimality: Measure of first-order optimality (gradient norm).
        history: Optional list of solution iterates for debugging/visualization.
        jac: Optional Jacobian matrix at the solution. Shape (m, n).
    """

    x: NDArray[np.float64]
    success: bool
    message: str
    nfev: int
    njev: int
    fun: NDArray[np.float64]
    residual_norm: float = field(default=0.0)
    optimality: float = field(default=0.0)
    history: Optional[List[NDArray[np.float64]]] = None
    jac: Optional[NDArray[np.float64]] = None

    def __getitem__(self, key: str) -> object:
        """Enable dict-style access: result['x']."""
        return getattr(self, key)

    def keys(self) -> List[str]:
        """Return list of field names for dict-like iteration."""
        return [f.name for f in fields(self)]

    def __iter__(self) -> Iterator[str]:
        """Iterate over field names."""
        return iter(self.keys())

    def __contains__(self, key: str) -> bool:
        """Check if key is a valid field name."""
        return key in self.keys()
