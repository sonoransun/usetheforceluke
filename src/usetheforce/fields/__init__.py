"""Generic field-simulation kernels: regular grids, finite-difference operators, sampling."""

from usetheforce.fields.grid import RegularGrid3D
from usetheforce.fields.laplacian import laplacian_3d
from usetheforce.fields.sample import sample_force_field

__all__ = ["RegularGrid3D", "laplacian_3d", "sample_force_field"]
