# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.0] - 2026-02-03

### Added

- Initial release of JGSS nonlinear solver
- `solve()` function for high-dimensional nonlinear systems
- `SolverConfig` dataclass documenting hyperparameters and defaults
- `SolverResult` dataclass with solution, diagnostics, and optional history
- Jacobian Guided Subspace Search algorithm using SVD
- Latin Hypercube Sampling for basin finding
- Levenberg-Marquardt for local convergence
- Support for analytical Jacobians via `jacobian_fn` parameter
- Optional bounds constraints (box constraints)
- Verbose output levels (-1=silent to 2=debug)
- Callback support for progress monitoring
- History tracking for solution iterates
- Optional Jacobian return for post-analysis
- Python 3.10, 3.11, 3.12 support
- NumPy 1.x and 2.x compatibility
