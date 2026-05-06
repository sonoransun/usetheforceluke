"""Antimatter-conversion propulsion avenue. SPECULATIVE.

Two submodels:
- ``AntimatterCounterGravity`` — local cancellation of a supplied background g(r).
- ``AntimatterGravitonField`` — Yukawa graviton field sourced by an annihilation hotspot.
"""

from usetheforce.antimatter.conversion import AntimatterCounterGravity
from usetheforce.antimatter.graviton import AntimatterGravitonField

__all__ = ["AntimatterCounterGravity", "AntimatterGravitonField"]
