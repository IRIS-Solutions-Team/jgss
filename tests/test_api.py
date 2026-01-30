"""API contract tests for JGSS solver.

This module validates the public API of the jgss package:
- SolverResult field presence and types
- SolverResult dict-style access
- SolverConfig default values
- Verbose output behavior at all levels
"""

import pytest
import numpy as np

from jgss import solve, SolverConfig, SolverResult
from tests.conftest import rosenbrock_residual, ROSENBROCK_X0


# =============================================================================
# SolverResult Tests
# =============================================================================


def test_solver_result_fields():
    """Verify SolverResult has all required fields."""
    result = solve(rosenbrock_residual, x0=ROSENBROCK_X0, verbose=-1)

    # Required fields exist
    assert hasattr(result, "x")
    assert hasattr(result, "success")
    assert hasattr(result, "message")
    assert hasattr(result, "nfev")
    assert hasattr(result, "njev")
    assert hasattr(result, "fun")

    # Types are correct
    assert isinstance(result.x, np.ndarray)
    assert isinstance(result.success, bool)
    assert isinstance(result.message, str)
    assert isinstance(result.nfev, int)
    assert isinstance(result.njev, int)

    # nfev > 0 (must have called function)
    assert result.nfev > 0


def test_solver_result_dict_access():
    """Verify SolverResult supports dict-style access."""
    result = solve(rosenbrock_residual, x0=ROSENBROCK_X0, verbose=-1)
    assert np.array_equal(result["x"], result.x)
    assert result["success"] == result.success
    assert result["message"] == result.message
    assert result["nfev"] == result.nfev


def test_solver_result_is_immutable():
    """Verify SolverResult is frozen (immutable)."""
    result = solve(rosenbrock_residual, x0=ROSENBROCK_X0, verbose=-1)
    with pytest.raises(AttributeError):
        result.success = False  # type: ignore[misc]


def test_solver_result_contains():
    """Verify SolverResult supports 'in' operator."""
    result = solve(rosenbrock_residual, x0=ROSENBROCK_X0, verbose=-1)
    assert "x" in result
    assert "success" in result
    assert "nonexistent" not in result


def test_solver_result_keys():
    """Verify SolverResult.keys() returns field names."""
    result = solve(rosenbrock_residual, x0=ROSENBROCK_X0, verbose=-1)
    keys = result.keys()
    assert "x" in keys
    assert "success" in keys
    assert "message" in keys
    assert "nfev" in keys
    assert "njev" in keys
    assert "fun" in keys


# =============================================================================
# SolverConfig Tests
# =============================================================================


def test_solver_config_defaults():
    """Verify SolverConfig has correct defaults."""
    config = SolverConfig()
    assert config.k_subspace == 30
    assert config.n_samples == 600
    assert config.tol == 1e-8
    assert config.maxiter == 500
    assert config.seed is None
    assert config.verbose == 0


def test_solver_config_is_immutable():
    """Verify SolverConfig is frozen (immutable)."""
    config = SolverConfig()
    with pytest.raises(AttributeError):
        config.tol = 1e-6  # type: ignore[misc]


# =============================================================================
# Verbose Output Tests
# =============================================================================


def test_verbose_silent(capfd):
    """verbose=-1 produces no output."""
    solve(rosenbrock_residual, x0=ROSENBROCK_X0, verbose=-1)
    captured = capfd.readouterr()
    assert captured.out == ""


def test_verbose_summary(capfd):
    """verbose=0 produces summary output."""
    solve(rosenbrock_residual, x0=ROSENBROCK_X0, verbose=0)
    captured = capfd.readouterr()
    # Should have some output (summary)
    assert len(captured.out) > 0
    # Should contain CONVERGED or NOT CONVERGED
    assert "CONVERGED" in captured.out


def test_verbose_iteration(capfd):
    """verbose=1 produces per-iteration output."""
    solve(rosenbrock_residual, x0=ROSENBROCK_X0, verbose=1)
    captured = capfd.readouterr()
    # Should have multiple lines
    assert captured.out.count("\n") > 1
    # Should contain phase information
    assert "Phase" in captured.out


def test_verbose_debug(capfd):
    """verbose=2 produces debug output (most verbose)."""
    solve(rosenbrock_residual, x0=ROSENBROCK_X0, verbose=2)
    captured = capfd.readouterr()
    # Debug should be longest
    assert len(captured.out) > 100
    # Should contain subspace information
    assert "subspace" in captured.out.lower() or "Subspace" in captured.out


def test_verbose_increasing_output(capfd):
    """Higher verbose levels produce more output."""
    outputs = []
    for level in [-1, 0, 1, 2]:
        solve(rosenbrock_residual, x0=ROSENBROCK_X0, verbose=level)
        captured = capfd.readouterr()
        outputs.append(len(captured.out))

    # Each level should produce at least as much output as the previous
    # (verbose=-1 should be 0, others should be increasing)
    assert outputs[0] == 0  # verbose=-1 is silent
    assert outputs[1] > 0  # verbose=0 has summary
    assert outputs[2] > outputs[1]  # verbose=1 has more
    assert outputs[3] >= outputs[2]  # verbose=2 has most
