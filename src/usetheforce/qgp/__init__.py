"""Quark–gluon plasma propulsion avenue.

Anchored: Stefan–Boltzmann energy density with lattice-QCD-flavoured T-dependent
effective degrees of freedom (Bjorken 1983; lattice QCD g_eff(T) crossover).

Speculative: the coupling that turns QGP energy throughput into a graviton
emission rate. Marked ``speculative=True`` on the resulting ``ForceField``.
"""

from usetheforce.qgp.graviton import QGPGravitonField
from usetheforce.qgp.source import QuarkGluonPlasmaSource, g_effective

__all__ = ["QGPGravitonField", "QuarkGluonPlasmaSource", "g_effective"]
