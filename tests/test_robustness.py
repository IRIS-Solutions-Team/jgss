"""Robustness tests for edge cases and ill-conditioned problems."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from jgss import solve


@pytest.mark.robustness
def test_near_singular_jacobian():
    """Test behavior with near-singular Jacobian at starting point.

    This function has a near-singular Jacobian near origin due to cubic x term
    and near-zero coefficient on y. The solver handles this gracefully.
    """

    def ill_conditioned(x):
        # Function with near-singular Jacobian near origin
        return np.array([x[0] ** 3, 1e-10 * x[1]])

    x0 = np.array([0.1, 0.1])
    result = solve(ill_conditioned, x0=x0, verbose=-1)
    # Solver should converge to near origin
    assert result is not None
    assert result.success
    assert_allclose(result.x, [0.0, 0.0], atol=1e-3)


@pytest.mark.robustness
def test_already_at_solution():
    """Test behavior when starting at the solution."""

    def simple_residual(x):
        return x - np.array([1.0, 2.0])

    x0 = np.array([1.0, 2.0])  # Already at solution
    result = solve(simple_residual, x0=x0, verbose=-1)
    assert result.success
    assert_allclose(result.x, [1.0, 2.0], atol=1e-10)
    # Residual should be essentially zero
    assert result.residual_norm < 1e-10


@pytest.mark.robustness
def test_zero_residual():
    """Test function that is zero everywhere."""

    def zero_func(x):
        return np.zeros_like(x)

    x0 = np.array([5.0, 5.0])
    result = solve(zero_func, x0=x0, verbose=-1)
    # Should succeed immediately (any point is a solution)
    assert result.success


@pytest.mark.robustness
@pytest.mark.slow
def test_large_system():
    """Test moderately large system (500 variables)."""
    n = 500

    def large_residual(x):
        # Simple diagonal system: x - target = 0
        target = np.arange(n, dtype=float) / n
        return x - target

    x0 = np.zeros(n)
    result = solve(large_residual, x0=x0, verbose=-1)
    assert result.success
    expected = np.arange(n, dtype=float) / n
    assert_allclose(result.x, expected, atol=1e-8)


@pytest.mark.robustness
def test_bounded_optimization():
    """Test solve with bounds constraint."""

    def simple_residual(x):
        return x - np.array([5.0, 5.0])  # Solution at (5, 5)

    x0 = np.array([0.0, 0.0])
    bounds = ([0.0, 0.0], [2.0, 2.0])  # Constrain to [0,2]
    result = solve(simple_residual, x0=x0, bounds=bounds, verbose=-1)
    # Solution should be at boundary (2, 2) since true solution is outside
    # Note: gtol termination at boundary is expected behavior - solver found
    # the best feasible point even though residual isn't zero
    assert_allclose(result.x, [2.0, 2.0], atol=1e-6)
    # Verify we're at the boundary (best feasible solution)
    assert np.all(result.x <= 2.0 + 1e-10)


@pytest.mark.robustness
def test_callback_called():
    """Test that callback is called during optimization."""
    callback_count = [0]  # Use list for mutability in closure

    def counting_callback(x, f):
        callback_count[0] += 1

    def simple_residual(x):
        return x - np.array([1.0, 1.0])

    x0 = np.array([0.0, 0.0])
    result = solve(simple_residual, x0=x0, callback=counting_callback, verbose=-1)
    assert result.success
    assert callback_count[0] > 0, "Callback was never called"
