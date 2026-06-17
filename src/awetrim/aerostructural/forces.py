# Copyright (c) 2023-2026 Oriol Cayon, Delft University of Technology
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

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
