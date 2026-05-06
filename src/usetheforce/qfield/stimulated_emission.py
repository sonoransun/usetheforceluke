"""Phased-array of coherent point emitters. SPECULATIVE.

A phased array of ``N`` coherent point emitters produces a complex amplitude
field at probe position ``r``::

    z(r) = Σₙ Aₙ / Rₙ · exp(i (k Rₙ + φₙ)),    Rₙ = |r − rₙ|

The time-averaged intensity is ``I(r) = |z(r)|²``. The propulsive (radiation)
potential is ``U(r) = -α I(r)``, so the force is::

    F(r) = -∇U(r) = α ∇I(r) = 2α Re[(∇z)(r) · z*(r)]

with::

    ∇z(r) = Σₙ Aₙ · (-1/Rₙ² + i k / Rₙ) · exp(i (k Rₙ + φₙ)) · d̂ₙ,
    d̂ₙ = (r − rₙ) / Rₙ.

Vectorized over emitters; emitter positions must not coincide with the probe
location (no softening — choose probe points away from emitters).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


class StimulatedEmissionArray:
    """Coherent phased-array emitter field exerting a radiation force on a probe."""

    metadata: dict[str, Any]

    def __init__(
        self,
        positions: np.ndarray | Sequence[Sequence[float]],
        amplitudes: np.ndarray | Sequence[float],
        phases: np.ndarray | Sequence[float],
        wavenumber: float,
        coupling: float = 1.0,
        min_distance_m: float = 0.0,
    ) -> None:
        self._pos = np.asarray(positions, dtype=float)
        if self._pos.ndim != 2 or self._pos.shape[1] != 3:
            raise ValueError(f"positions must have shape (N, 3), got {self._pos.shape}")
        n = self._pos.shape[0]
        self._amp = np.asarray(amplitudes, dtype=float)
        self._phi = np.asarray(phases, dtype=float)
        if self._amp.shape != (n,) or self._phi.shape != (n,):
            raise ValueError(
                f"amplitudes/phases shape must be ({n},), "
                f"got {self._amp.shape} and {self._phi.shape}"
            )
        self._k = float(wavenumber)
        self._alpha = float(coupling)
        self._min_R = float(min_distance_m)
        if self._k <= 0:
            raise ValueError("wavenumber must be positive")
        if self._min_R < 0:
            raise ValueError("min_distance_m must be non-negative")
        self.metadata = {
            "avenue": "qfield",
            "model": f"stimulated-emission phased array (N={n})",
            "speculative": True,
            "speculative_components": ["amplitudes", "phases", "coupling"],
            "citation": "coherent phased-array radiation-pressure ansatz; not derived from QED",
        }

    def _displacements(self, r: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        d = r[None, :] - self._pos  # (N, 3)
        R = np.sqrt(np.einsum("ij,ij->i", d, d))  # (N,)
        if np.any(self._min_R >= R):
            raise ValueError(
                f"probe within min_distance_m={self._min_R} of an emitter "
                f"(closest R={float(R.min())})"
            )
        d_hat = d / R[:, None]
        return d, R, d_hat

    def _z_and_e(self, R: np.ndarray) -> tuple[complex, np.ndarray]:
        e = np.exp(1j * (self._k * R + self._phi))  # (N,)
        z = complex(np.sum(self._amp * e / R))
        return z, e

    def potential(self, r: np.ndarray) -> float:
        _, R, _ = self._displacements(np.asarray(r, dtype=float))
        z, _ = self._z_and_e(R)
        return float(-self._alpha * (z * np.conj(z)).real)

    def force(self, t: float, r: np.ndarray) -> np.ndarray:  # noqa: ARG002
        _, R, d_hat = self._displacements(np.asarray(r, dtype=float))
        z, e = self._z_and_e(R)
        factor = self._amp * (-1.0 / R**2 + 1j * self._k / R) * e  # (N,)
        grad_z = np.einsum("i,ij->j", factor, d_hat)  # (3,) complex
        grad_I = 2.0 * np.real(grad_z * np.conj(z))
        return self._alpha * grad_I
