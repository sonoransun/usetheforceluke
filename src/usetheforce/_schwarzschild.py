"""Internal helpers shared by Schwarzschild-form ``ForceField`` implementations.

The two blackhole models (``SchwarzschildGravity``, ``BlackHoleCounterDrive``)
both compute a probe-to-centre displacement, a Schwarzschild radius
``r_s = 2 G M / c²``, and an optional general-relativistic hover correction::

    a_hover(R) = (G M / R²) / sqrt(1 - r_s/R)   (proper acceleration to hover)

This factor diverges as ``R → r_s`` — the central physics finding of the
"blackhole explorer" mode. Centralising these formulas mirrors how
``_yukawa.py`` serves the two graviton models.

References
----------
- Schwarzschild, K. (1916). "Über das Gravitationsfeld eines Massenpunktes
  nach der Einsteinschen Theorie".
- Misner, Thorne, Wheeler, *Gravitation* (1973), §31 (Schwarzschild geometry).
"""

from __future__ import annotations

import math

import numpy as np
import scipy.constants as sc

# Smallest |R| we treat as physically meaningful before raising. The 1/R²
# term overflows below this; above, the float arithmetic is fine.
R_FLOOR_M: float = 1e-30

# Convenience re-exports for callers that want the constants without importing
# scipy.constants directly.
G_NEWTON: float = sc.G  # m³ kg⁻¹ s⁻²
C_LIGHT: float = sc.c  # m / s


def schwarzschild_radius(mass_kg: float) -> float:
    """Schwarzschild radius ``r_s = 2 G M / c²`` (m). Requires ``M > 0``."""
    if mass_kg <= 0:
        raise ValueError("mass_kg must be positive")
    return 2.0 * G_NEWTON * mass_kg / (C_LIGHT * C_LIGHT)


def displacement(r: np.ndarray, center: np.ndarray) -> tuple[np.ndarray, float]:
    """Return ``(d = r − r_c, R = |d|)``; raise if ``R`` falls below ``R_FLOOR_M``."""
    d = np.asarray(r, dtype=float) - center
    R = float(np.sqrt(np.dot(d, d)))
    if R < R_FLOOR_M:
        raise ValueError(
            f"probe within {R_FLOOR_M} m of centre (R={R}); Schwarzschild field is singular"
        )
    return d, R


def gr_hover_factor(R: float, r_s: float) -> float:
    """Proper-acceleration correction ``1 / sqrt(1 - r_s/R)`` for a hovering observer.

    Raises ``ValueError`` if ``R ≤ r_s`` — inside or on the horizon there is no
    stationary observer, and the formula has no real value.
    """
    if r_s >= R:
        raise ValueError(
            f"R={R} m is inside or on the event horizon r_s={r_s} m; no stationary frame"
        )
    return 1.0 / math.sqrt(1.0 - r_s / R)
