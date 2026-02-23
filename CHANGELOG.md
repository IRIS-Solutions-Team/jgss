# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.1.0] - 2026-02-23

### Added

- `n_restarts` parameter (default 5) for multi-restart robustness
  - Each restart uses a different LHS seed for basin search diversity
  - Best-of-N selection with early exit on convergence
  - Seed semantics: `seed=42, n_restarts=5` → seeds [42, 43, 44, 45, 46]
- `clip_bounds` parameter for soft variable clamping
  - Separate from `bounds` (which constrains the LM optimizer)
  - Applies `np.clip` to x0 and all intermediate/final results
  - Useful for keeping DSGE variables in economically meaningful ranges
- Scaled LM polish as automatic fallback when restarts don't converge
  - `DiagonalScaler` normalizes variables by `max(|x|, 1.0)` before LM
  - Dramatically improves conditioning for multi-scale DSGE systems
  - Chain-rule Jacobian: `J_scaled = J_original * scales[np.newaxis, :]`
- `_scaling.py` module with `DiagonalScaler` frozen dataclass

### Changed

- Default `n_restarts` is 5 (was implicitly 1 in v1.0.0)
- `SolverResult.nfev`/`njev` now accumulate across all restarts
- `SolverConfig` documents new `n_restarts` and `clip_bounds` fields

### Performance

- 72-80% success rate on real DSGE benchmark (466x443, 25 trials)
  vs ~56% with single-restart v1.0.0

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
