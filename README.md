# Jacobian-Guided Subspace Search (JGSS)

**Jacobian Guided Subspace Search** — Nonlinear solver for macroeconomic and DSGE models.

[![CI](https://img.shields.io/github/actions/workflow/status/IRIS-Solutions-Team/jgss/ci.yml?branch=main&label=CI)](https://github.com/IRIS-Solutions-Team/jgss/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/jgss)](https://pypi.org/project/jgss/)
[![Python versions](https://img.shields.io/pypi/pyversions/jgss)](https://pypi.org/project/jgss/)
[![License](https://img.shields.io/pypi/l/jgss)](https://github.com/IRIS-Solutions-Team/jgss/blob/main/LICENSE)

## Installation

```bash
pip install jgss
```

## Quick Start

```python
import numpy as np
from jgss import solve

# Define a simple system: x^2 = target
def residual(x):
    target = np.array([1.0, 4.0, 9.0])
    return x**2 - target

# Solve from initial guess
result = solve(residual, x0=np.array([0.5, 1.5, 2.5]))

print(result.x)        # [1.0, 2.0, 3.0]
print(result.success)  # True
```

## Usage Examples

### Basic Usage

```python
import numpy as np
from jgss import solve

def system(x):
    """System of nonlinear equations."""
    return np.array([
        x[0]**2 + x[1]**2 - 4,    # Circle: x^2 + y^2 = 4
        x[0] - x[1]               # Line: x = y
    ])

result = solve(system, x0=np.array([1.0, 1.0]))

if result.success:
    print(f"Solution: {result.x}")
    print(f"Residual norm: {result.residual_norm:.2e}")
```

### Configuration Options

```python
import numpy as np
from jgss import solve

def large_system(x):
    """High-dimensional system."""
    return x**2 - np.arange(1, len(x) + 1, dtype=float)

result = solve(
    large_system,
    x0=np.ones(100),
    tol=1e-10,           # Tighter tolerance
    k_subspace=50,       # Larger subspace for high-dim problems
    verbose=1,           # Show iteration progress
    seed=42,             # Reproducible results
)

print(f"Converged: {result.success}")
print(f"Function evaluations: {result.nfev}")
```

### Advanced: Analytical Jacobian

```python
import numpy as np
from jgss import solve

def residual(x):
    return np.array([
        x[0]**2 - x[1] - 1,
        x[0] - x[1]**2 + 1,
    ])

def jacobian(x):
    """Analytical Jacobian matrix."""
    return np.array([
        [2*x[0], -1],
        [1, -2*x[1]],
    ])

result = solve(
    residual,
    x0=np.array([1.0, 1.0]),
    jacobian_fn=jacobian,  # Faster than finite differences
)

print(f"Solution: {result.x}")
print(f"Jacobian evaluations: {result.njev}")
```

### Multi-Restart for DSGE Models

```python
import numpy as np
from jgss import solve

def dsge_residual(x):
    """DSGE-like system with multiple scales."""
    return np.array([
        0.01 * x[0] - 0.03,      # Interest rate (~0.03)
        1000 * x[1] - 5000,       # GDP (~5.0)
        x[2]**2 - x[0] * x[1],   # Nonlinear coupling
    ])

result = solve(
    dsge_residual,
    x0=np.array([0.01, 4.0, 0.1]),
    n_restarts=5,                    # 5 independent LHS draws
    clip_bounds=(-10.0, 30.0),       # Keep variables in range
    seed=42,                         # Reproducible
    verbose=0,
)

print(f"Converged: {result.success}")
print(f"Solution: {result.x}")
print(f"Restarts used: see nfev={result.nfev}")
```

### Advanced: Bounds and Callbacks

```python
import numpy as np
from jgss import solve

def system(x):
    return x**3 - np.array([8.0, 27.0])

def monitor(x, f):
    """Callback for progress monitoring."""
    print(f"  x = {x}, ||f|| = {np.linalg.norm(f):.4e}")

result = solve(
    system,
    x0=np.array([1.0, 2.0]),
    bounds=(np.array([0.0, 0.0]), np.array([10.0, 10.0])),  # Constrain to [0, 10]
    callback=monitor,
    history=True,         # Track solution iterates
    return_jacobian=True, # Include final Jacobian
    verbose=-1,           # Suppress default output
)

print(f"Solution: {result.x}")
print(f"Iterations tracked: {len(result.history)}")
print(f"Final Jacobian shape: {result.jac.shape}")
```

## API Reference

### `solve()`

Main solver function for nonlinear systems.

```python
from jgss import solve

result = solve(residual_fn, x0, **options)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `residual_fn` | `Callable` | required | Function `f(x) -> residuals` returning array of shape `(m,)` |
| `x0` | `ArrayLike` | required | Initial guess, array-like of shape `(n,)` |
| `jacobian_fn` | `Callable` | `None` | Analytical Jacobian `J(x) -> (m, n)` matrix. If `None`, uses finite differences |
| `tol` | `float` | `1e-8` | Convergence tolerance for residual norm |
| `maxiter` | `int` | `500` | Maximum iterations for local convergence phase |
| `k_subspace` | `int` | `30` | Dimension of active subspace for global search |
| `n_samples` | `int` | `600` | Number of Latin Hypercube samples for basin finding |
| `bounds` | `tuple` | `None` | Optional `(lower, upper)` bounds, each of shape `(n,)` |
| `n_restarts` | `int` | `5` | Number of independent restart attempts with different LHS seeds |
| `clip_bounds` | `tuple` | `None` | Optional `(lower, upper)` for soft variable clamping (applied via `np.clip`) |
| `seed` | `int` | `None` | Random seed for reproducibility. Restart *i* uses `seed+i` |
| `verbose` | `int` | `0` | Verbosity: `-1`=silent, `0`=summary, `1`=iterations, `2`=debug |
| `callback` | `Callable` | `None` | Function `callback(x, f)` called at each evaluation |
| `history` | `bool` | `False` | If `True`, include solution iterates in result |
| `return_jacobian` | `bool` | `False` | If `True`, include final Jacobian in result |

**Returns:** `SolverResult`

### `SolverConfig`

Documents available hyperparameters (for reference only - `solve()` takes keyword arguments directly).

```python
from jgss import SolverConfig

# View default configuration
config = SolverConfig()
print(config.tol)        # 1e-8
print(config.k_subspace) # 30
```

**Fields:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `tol` | `float` | `1e-8` | Convergence tolerance for residual norm |
| `maxiter` | `int` | `500` | Maximum solver iterations |
| `k_subspace` | `int` | `30` | Dimension of Jacobian Guided subspace |
| `n_samples` | `int` | `600` | Number of Latin Hypercube samples |
| `n_restarts` | `int` | `5` | Number of independent restart attempts |
| `clip_bounds` | `tuple` | `None` | Soft variable clamping bounds |
| `seed` | `int` | `None` | Random seed for reproducibility |
| `verbose` | `int` | `0` | Verbosity level |

### `SolverResult`

Immutable result container with both attribute and dict-style access.

```python
# Attribute access
print(result.x)
print(result.success)

# Dict-style access
print(result['x'])
print('success' in result)  # True
```

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `x` | `NDArray[float64]` | Solution vector, shape `(n,)` |
| `success` | `bool` | `True` if converged within tolerance |
| `message` | `str` | Human-readable status message |
| `nfev` | `int` | Number of function evaluations |
| `njev` | `int` | Number of Jacobian evaluations |
| `fun` | `NDArray[float64]` | Residual vector at solution, shape `(m,)` |
| `residual_norm` | `float` | L2 norm of residual at solution |
| `optimality` | `float` | First-order optimality (gradient norm) |
| `history` | `list` | Solution iterates (if `history=True`) |
| `jac` | `NDArray[float64]` | Jacobian at solution (if `return_jacobian=True`) |

## How It Works

JGSS (Jacobian Guided Subspace Search) is a hybrid global-local optimization algorithm designed for high-dimensional nonlinear systems common in macroeconomic modeling.

The algorithm operates in three phases, repeated across multiple restarts:

1. **Jacobian Analysis** - Computes the Jacobian at the initial guess and uses SVD to identify an active subspace where the system is most sensitive to changes
2. **Basin Finding** - Uses Latin Hypercube Sampling within the active subspace to find a starting point in a promising convergence basin
3. **Local Convergence** - Applies Levenberg-Marquardt to refine the solution to machine precision

By default, JGSS runs 5 independent restarts with different random seeds for the LHS basin search. Each restart explores a different region of the subspace, and the best result (lowest residual norm) is kept. If no restart converges, a **scaled LM polish** normalizes variables by their characteristic magnitudes and runs a final LM pass in scaled space — this dramatically improves conditioning for multi-scale DSGE systems.

This approach is particularly effective for DSGE and macroeconomic models where the solution space is high-dimensional but the active dynamics lie in a lower-dimensional manifold.

## License

MIT
