"""Shared pint UnitRegistry and SI-stripping helper.

All public APIs accept ``pint.Quantity`` and call ``to_si`` once at the boundary
to obtain bare floats/ndarrays in SI for numerical kernels — pint overhead in
inner loops (e.g. ODE RHS) is unacceptable for trajectory work.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pint

ureg: pint.UnitRegistry = pint.UnitRegistry(system="SI")
Q_ = ureg.Quantity


def to_si(quantity: Any, expected_units: str) -> float | np.ndarray:
    """Convert ``quantity`` to SI ``expected_units`` and return a bare float/ndarray.

    Raises ``pint.DimensionalityError`` when dimensions disagree. Plain
    floats/ndarrays pass through (treated as already-SI), so kernels can be
    called with either Quantities or raw numbers.
    """
    if isinstance(quantity, pint.Quantity):
        converted = quantity.to(expected_units)
        magnitude = converted.magnitude
        return np.asarray(magnitude) if isinstance(magnitude, list | tuple) else magnitude
    return quantity
