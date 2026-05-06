"""Boundary-unit handling: round-trip via ``to_si`` and dimension enforcement."""

from __future__ import annotations

import numpy as np
import pint
import pytest

from usetheforce.units import Q_, to_si, ureg


def test_to_si_round_trip_scalar() -> None:
    assert to_si(Q_(2.5, "km"), "m") == pytest.approx(2500.0)


def test_to_si_passes_through_floats() -> None:
    assert to_si(3.0, "m") == 3.0


def test_to_si_passes_through_arrays() -> None:
    arr = np.array([1.0, 2.0, 3.0])
    out = to_si(arr, "m")
    assert isinstance(out, np.ndarray)
    np.testing.assert_array_equal(out, arr)


def test_to_si_rejects_dimension_mismatch() -> None:
    with pytest.raises(pint.DimensionalityError):
        to_si(Q_(2.5, "kg"), "m")


def test_shared_registry_identity() -> None:
    """All quantities must come from the package-shared registry."""
    q = Q_(1.0, "m")
    assert q._REGISTRY is ureg
