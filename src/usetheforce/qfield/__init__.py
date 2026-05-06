"""Exotic quantum-field shaping propulsion avenue. SPECULATIVE.

Three submodels:
- ``ShapedFieldAnsatz`` — Gaussian-well parametric ansatz.
- ``HeavyElementLattice`` — softened multi-site dipole array.
- ``StimulatedEmissionArray`` — coherent phased-array radiation field.
"""

from usetheforce.qfield.heavy_elements import HeavyElementLattice
from usetheforce.qfield.models import ShapedFieldAnsatz
from usetheforce.qfield.stimulated_emission import StimulatedEmissionArray

__all__ = ["HeavyElementLattice", "ShapedFieldAnsatz", "StimulatedEmissionArray"]
