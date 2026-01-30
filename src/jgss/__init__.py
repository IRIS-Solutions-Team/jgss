"""JGSS: Jacobian-Guided Subspace Search solver for nonlinear systems."""

try:
    from jgss._version import __version__
except ImportError:
    __version__ = "0.1.0"

from jgss._core import solve
from jgss._types import SolverConfig, SolverResult

__all__ = ["__version__", "solve", "SolverConfig", "SolverResult"]
