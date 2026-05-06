"""Grid + Laplacian + sample_force_field smoke tests."""

from __future__ import annotations

import numpy as np
import pytest

from usetheforce.fields import RegularGrid3D, laplacian_3d, sample_force_field
from usetheforce.qfield import ShapedFieldAnsatz


def test_grid_axes_and_points() -> None:
    g = RegularGrid3D(origin=(0.0, 0.0, 0.0), spacing=(0.5, 0.5, 0.5), shape=(3, 3, 3))
    x, y, z = g.axes()
    assert x.shape == (3,) and y.shape == (3,) and z.shape == (3,)
    assert g.points().shape == (27, 3)


def test_grid_validates_input() -> None:
    with pytest.raises(ValueError):
        RegularGrid3D(origin=(0.0, 0.0, 0.0), spacing=(0.0, 0.5, 0.5), shape=(3, 3, 3))
    with pytest.raises(ValueError):
        RegularGrid3D(origin=(0.0, 0.0, 0.0), spacing=(0.5, 0.5, 0.5), shape=(1, 3, 3))


def test_laplacian_of_quadratic() -> None:
    """∇² of f(x,y,z) = x² + y² + z² is 6 everywhere on the interior."""
    n = 9
    g = RegularGrid3D(origin=(0.0, 0.0, 0.0), spacing=(0.1, 0.1, 0.1), shape=(n, n, n))
    X, Y, Z = g.meshgrid()
    field = X**2 + Y**2 + Z**2
    lap = laplacian_3d(field, g)
    np.testing.assert_allclose(lap[1:-1, 1:-1, 1:-1], 6.0, atol=1e-10)


def test_sample_force_field_shape() -> None:
    ff = ShapedFieldAnsatz(amplitude=1.0, sigma=1.0)
    g = RegularGrid3D(origin=(-1.0, -1.0, -1.0), spacing=(0.5, 0.5, 0.5), shape=(5, 5, 5))
    field = sample_force_field(ff, g, t=0.0)
    assert field.shape == (3, 5, 5, 5)
    assert np.all(np.isfinite(field))
