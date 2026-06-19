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

"""Tidy aerodynamic-identification dataset: schema, IO and phi_a reconstruction.

The identification pipeline (AS solver and EKF flight data) produces a tidy
table with one row per aerodynamic sample.  Both sources share this schema so a
single :mod:`awetrim.identification.aero_polynomial` selection runs identically
on each, enabling a like-for-like comparison.

Schema (one row per alpha sample):

    anchor_id : str    identifier of the originating anchor / flight segment
    source    : str    "AS" or "EKF"
    v_a       : float  apparent wind speed [m/s]
    u_s       : float  steering input [-]
    u_p       : float  absolute power-tape length [m]
    alpha     : float  angle of attack [rad]
    cl        : float  wing lift coefficient [-]
    cd        : float  wing drag coefficient [-]
    phi_a     : float  aerodynamic roll [rad]
    converged : bool   whether the underlying solve/segment is trusted

``u_p`` is the absolute power-tape length, not a normalised depower input.
``v_a``, ``u_s`` and ``u_p`` are constant within an anchor's alpha sweep;
``alpha``, ``cl``, ``cd`` and ``phi_a`` vary along it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .aero_polynomial import DEFAULT_REGRESSORS

TIDY_COLUMNS: tuple[str, ...] = (
    "anchor_id",
    "source",
    "v_a",
    "u_s",
    "u_p",
    "alpha",
    "cl",
    "cd",
    "phi_a",
    "converged",
)


def aerodynamic_roll(
    total_aero_force: np.ndarray,
    va: np.ndarray,
    radial_axis: np.ndarray,
) -> float:
    """Aerodynamic roll angle phi_a [rad] of the aero force about the apparent wind.

    Reproduces the trim-solver definition
    (``awetrim.aerodynamics.vsm_quasi_steady``): the lift direction is the part
    of the radial axis perpendicular to the apparent wind, the side direction
    completes the right-handed frame, and phi_a is the tilt of the total aero
    force from the lift direction toward the side direction.

    Args:
        total_aero_force: total aerodynamic force vector (3,), any consistent frame.
        va: apparent wind vector (3,) in the same frame.
        radial_axis: radial (tether/zenith) unit vector (3,) in the same frame.

    Returns:
        phi_a in radians; ``nan`` if the apparent wind is degenerate.
    """
    f = np.asarray(total_aero_force, dtype=float).reshape(3)
    va = np.asarray(va, dtype=float).reshape(3)
    radial = np.asarray(radial_axis, dtype=float).reshape(3)
    va_norm = np.linalg.norm(va)
    if va_norm < 1e-9:
        return float("nan")
    va_unit = va / va_norm
    lift_dir = radial - np.dot(radial, va_unit) * va_unit
    lift_norm = np.linalg.norm(lift_dir)
    if lift_norm < 1e-9:
        return float("nan")
    lift_dir = lift_dir / lift_norm
    side_dir = np.cross(lift_dir, va_unit)
    return float(np.arctan2(np.dot(f, side_dir), np.dot(f, lift_dir)))


def regressor_arrays(
    df,
    regressors: Sequence[str] = DEFAULT_REGRESSORS,
) -> dict[str, np.ndarray]:
    """Extract a ``{regressor: array}`` mapping from a tidy DataFrame.

    Maps the canonical regressor names used by the polynomial model to the
    dataset columns (``v_a`` is stored as ``v_a``; ``alpha`` etc. match
    directly).
    """
    column_for = {
        "alpha": "alpha",
        "u_s": "u_s",
        "u_p": "u_p",
        "v_a": "v_a",
    }
    return {
        name: np.asarray(df[column_for.get(name, name)], dtype=float)
        for name in regressors
    }


def clean_dataset(df, *, require_converged: bool = True, targets=("cl", "cd", "phi_a")):
    """Drop non-finite/failed rows so the fit sees only trusted samples."""
    import pandas as pd  # local import keeps module import light

    out = df.copy()
    if require_converged and "converged" in out.columns:
        out = out[out["converged"].astype(bool)]
    finite_cols = [c for c in (*targets, *DEFAULT_REGRESSORS) if c in out.columns]
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=finite_cols)
    return out.reset_index(drop=True)


def save_dataset(df, path: str | Path) -> Path:
    """Write the tidy dataset to CSV (creating parent dirs)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def load_dataset(path: str | Path):
    """Load a tidy dataset CSV."""
    import pandas as pd

    return pd.read_csv(path)
