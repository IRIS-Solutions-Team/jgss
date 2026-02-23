"""Diagonal variable scaling for well-conditioned LM operation.

DSGE variables span O(10^-2) to O(10^4). Without scaling, the LM Jacobian
is ill-conditioned. This module provides a minimal DiagonalScaler that
normalizes variables to O(1).

Chain rule: J_scaled = J_original * scales[np.newaxis, :]
"""

from __future__ import annotations

import dataclasses

import numpy as np
from numpy.typing import NDArray


@dataclasses.dataclass(frozen=True)
class DiagonalScaler:
    """Component-wise scaling by characteristic magnitudes.

    scale(x) = x / scales
    unscale(x_s) = x_s * scales
    """

    scales: NDArray[np.float64]

    @classmethod
    def from_x0(cls, x0: NDArray[np.float64]) -> DiagonalScaler:
        """Auto-detect scales from initial point magnitudes.

        Uses max(|x0_i|, 1.0) as the scale for each component,
        providing a floor of 1.0 for near-zero components.
        """
        scales = np.maximum(np.abs(x0), 1.0)
        return cls(scales=scales)

    def scale(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return x / self.scales

    def unscale(self, x_scaled: NDArray[np.float64]) -> NDArray[np.float64]:
        return x_scaled * self.scales
