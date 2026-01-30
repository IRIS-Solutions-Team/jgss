"""Convergence tests for JGSS solver on benchmark optimization functions.

This module tests that jgss.solve() converges to known minima for classic
optimization test functions. Tests use parametrization to run the same
convergence logic across multiple benchmarks.

Tests:
- 2D benchmarks: Rosenbrock, Powell (4D), Beale, Himmelblau, Booth
- High-dimensional: 10D and 100D Rosenbrock
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose

from jgss import solve
from tests.conftest import (
    rosenbrock_residual,
    powell_residual,
    beale_residual,
    himmelblau_residual,
    booth_residual,
    ROSENBROCK_X0,
    POWELL_X0,
    BEALE_X0,
    HIMMELBLAU_X0,
    BOOTH_X0,
    ROSENBROCK_SOLUTION,
    POWELL_SOLUTION,
    BEALE_SOLUTION,
    HIMMELBLAU_SOLUTION,
    BOOTH_SOLUTION,
)


# =============================================================================
# Benchmark Test Data
# =============================================================================

BENCHMARKS = [
    ("rosenbrock", rosenbrock_residual, ROSENBROCK_X0, ROSENBROCK_SOLUTION),
    ("powell", powell_residual, POWELL_X0, POWELL_SOLUTION),
    ("beale", beale_residual, BEALE_X0, BEALE_SOLUTION),
    ("himmelblau", himmelblau_residual, HIMMELBLAU_X0, HIMMELBLAU_SOLUTION),
    ("booth", booth_residual, BOOTH_X0, BOOTH_SOLUTION),
]


# =============================================================================
# Parametrized Benchmark Tests
# =============================================================================


@pytest.mark.parametrize(
    "name,func,x0,expected",
    BENCHMARKS,
    ids=[b[0] for b in BENCHMARKS],
)
def test_benchmark_convergence(name, func, x0, expected):
    """Test solve() converges to known minimum for benchmark functions."""
    result = solve(func, x0=x0, verbose=-1)
    assert result.success, f"{name} failed: {result.message}"
    assert_allclose(result.x, expected, rtol=0, atol=1e-8)


# =============================================================================
# High-Dimensional Rosenbrock Tests
# =============================================================================


def _rosenbrock_nd_residual(x):
    """N-dimensional Rosenbrock in residual form.

    The n-dimensional Rosenbrock is sum of terms:
    f(x) = sum_{i=0}^{n-2} [100*(x[i+1] - x[i]^2)^2 + (1 - x[i])^2]

    Expressed as residuals: for each i, we have two residuals:
    - 10*(x[i+1] - x[i]^2)  (when squared gives 100*(...)^2)
    - 1 - x[i]              (when squared gives (1 - x[i])^2)

    Minimum at x = [1, 1, ..., 1] where all residuals are zero.
    """
    residuals = []
    for i in range(len(x) - 1):
        residuals.append(10.0 * (x[i + 1] - x[i] ** 2))
        residuals.append(1.0 - x[i])
    return np.array(residuals)


@pytest.mark.parametrize(
    "n_dims",
    [10, pytest.param(100, marks=pytest.mark.slow)],
)
def test_rosenbrock_high_dimensional(n_dims):
    """Test convergence on n-dimensional Rosenbrock."""
    x0 = np.zeros(n_dims)
    result = solve(_rosenbrock_nd_residual, x0=x0, verbose=-1)
    expected = np.ones(n_dims)
    assert result.success, f"{n_dims}D Rosenbrock failed: {result.message}"
    assert_allclose(result.x, expected, rtol=0, atol=1e-8)
