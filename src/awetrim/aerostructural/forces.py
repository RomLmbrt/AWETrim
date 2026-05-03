"""Force assembly helpers for aerostructural coupling."""

from __future__ import annotations

import numpy as np


def distribute_total_force_by_particle_mass(total_force, m_arr) -> np.ndarray:
    """Distribute a total 3D force over nodes proportional to particle masses."""
    total_force = np.asarray(total_force, dtype=float).reshape(3)
    masses = np.asarray(m_arr, dtype=float).reshape(-1)
    mass_sum = float(np.sum(masses))
    if mass_sum <= 1e-12:
        raise ValueError("Total particle mass must be positive to distribute force.")

    mass_fraction = masses / mass_sum
    return mass_fraction[:, None] * total_force[None, :]


__all__ = ["distribute_total_force_by_particle_mass"]
