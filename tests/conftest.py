"""Pytest fixtures and benchmark functions for JGSS testing.

This module provides:
- Benchmark residual functions for testing solver convergence
- Expected solutions and starting points for each benchmark
- All functions return residual arrays suitable for jgss.solve()

The solver minimizes sum of squared residuals, so each benchmark is expressed
in residual form where ||residual(x*)||^2 = 0 at the known minimum.
"""

import numpy as np
from numpy.typing import NDArray


# =============================================================================
# Pytest Configuration
# =============================================================================


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "robustness: mark test as robustness/edge case test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow (run on limited CI matrix)"
    )

# =============================================================================
# Benchmark Residual Functions
# =============================================================================


def rosenbrock_residual(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """2D Rosenbrock in residual form.

    The classic Rosenbrock function f(x,y) = (1-x)^2 + 100(y-x^2)^2
    expressed as residuals [10*(y - x^2), 1 - x] so that
    ||residual||^2 = 100(y-x^2)^2 + (1-x)^2 = f(x,y)

    Minimum at (1, 1) where residual = [0, 0].
    """
    return np.array([10.0 * (x[1] - x[0] ** 2), 1.0 - x[0]])


def powell_residual(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Powell's function in residual form (4D).

    The Powell function has four terms that when squared and summed give:
    (x0 + 10*x1)^2 + 5*(x2 - x3)^2 + (x1 - 2*x2)^4 + 10*(x0 - x3)^4

    Expressed as residuals where ||residual||^2 matches the function.
    Note: The quartic terms require sqrt when expressing as residuals.

    Minimum at (0, 0, 0, 0).
    """
    return np.array(
        [
            x[0] + 10.0 * x[1],
            np.sqrt(5.0) * (x[2] - x[3]),
            (x[1] - 2.0 * x[2]) ** 2,
            np.sqrt(10.0) * (x[0] - x[3]) ** 2,
        ]
    )


def beale_residual(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Beale's function in residual form (2D).

    f(x,y) = (1.5 - x + xy)^2 + (2.25 - x + xy^2)^2 + (2.625 - x + xy^3)^2

    Expressed as 3 residuals for the 3 terms.
    Minimum at (3, 0.5) where residual = [0, 0, 0].
    """
    return np.array(
        [
            1.5 - x[0] + x[0] * x[1],
            2.25 - x[0] + x[0] * x[1] ** 2,
            2.625 - x[0] + x[0] * x[1] ** 3,
        ]
    )


def himmelblau_residual(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Himmelblau's function in residual form (2D).

    f(x,y) = (x^2 + y - 11)^2 + (x + y^2 - 7)^2

    Expressed as 2 residuals for the 2 terms.
    Has four identical local minima at:
    - (3.0, 2.0)
    - (-2.805118, 3.131312)
    - (-3.779310, -3.283186)
    - (3.584428, -1.848126)

    We test convergence to (3, 2) from origin.
    """
    return np.array([x[0] ** 2 + x[1] - 11.0, x[0] + x[1] ** 2 - 7.0])


def booth_residual(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Booth's function in residual form (2D).

    f(x,y) = (x + 2y - 7)^2 + (2x + y - 5)^2

    Expressed as 2 residuals for the 2 terms.
    Minimum at (1, 3) where residual = [0, 0].
    """
    return np.array([x[0] + 2.0 * x[1] - 7.0, 2.0 * x[0] + x[1] - 5.0])


# =============================================================================
# Expected Solutions
# =============================================================================

ROSENBROCK_SOLUTION = np.array([1.0, 1.0])
POWELL_SOLUTION = np.array([0.0, 0.0, 0.0, 0.0])
BEALE_SOLUTION = np.array([3.0, 0.5])
HIMMELBLAU_SOLUTION = np.array([3.0, 2.0])
BOOTH_SOLUTION = np.array([1.0, 3.0])


# =============================================================================
# Starting Points
# =============================================================================

ROSENBROCK_X0 = np.array([0.0, 0.0])
POWELL_X0 = np.array([3.0, -1.0, 0.0, 1.0])
BEALE_X0 = np.array([0.0, 0.0])
HIMMELBLAU_X0 = np.array([2.0, 1.5])  # Start closer to (3, 2) minimum
BOOTH_X0 = np.array([0.0, 0.0])
