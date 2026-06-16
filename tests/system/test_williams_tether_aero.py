"""Per-segment aerodynamic verification for ``WilliamsTether``.

The tether resolves the apparent wind on each segment into a drag (along the
apparent wind) and a lift (perpendicular to it) using the cross-flow +
skin-friction decomposition:

    cd_t = C_N * sin^3(theta) + pi * cf * cos^3(theta)
    cl_t = C_N * sin^2(theta) * cos(theta) - pi * cf * sin(theta) * cos^2(theta)

where ``theta`` is the angle between the apparent wind and the segment axis.
For UNIT drag/lift directions, ``drag_n + lift_n`` must reproduce the physical
force exactly: a normal (cross-flow) component of magnitude ``q * C_N * sin^2``
and a tangential (skin-friction) component of magnitude ``q * pi * cf * cos^2``,
with ``q = 0.5 * rho * V^2 * L * d``. These tests pin that invariant down and
guard against the lift direction being left unnormalised.
"""

from __future__ import annotations

from dataclasses import dataclass

import casadi as ca
import numpy as np
import pytest

from awetrim.system.williams_tether import WilliamsTether


@dataclass
class _UniformWind:
    """Constant-speed wind exposing the interface the tether reads."""

    speed: float
    z0: float = 0.01

    def speed_wind_at_height(self, z):  # noqa: D401 - simple stub
        return self.speed


@dataclass
class _Env:
    rho: float
    g: float
    wind: object


def _evaluate_shape(tether, env, *, omega):
    """Evaluate the symbolic tether shape at a fully numeric configuration.

    All decision symbols are overridden with concrete values so the resulting
    CasADi graph is constant and can be evaluated with ``ca.evalf``.
    """
    length = 228.0
    shape = tether.tether_shape_symbolic(
        env=env,
        r_kite=ca.DM([100.0, 0.0, 200.0]),
        tension_kite=5000.0,
        omega=ca.DM(omega),
        tether_length=length,
        elevation_last=float(np.arctan2(200.0, 100.0)),
        azimuth_last=0.0,
    )
    keys = (
        "drag_per_node",
        "lift_per_node",
        "velocities_apparent_wind",
        "angle_va_tether",
        "tensions",
    )
    out = {k: np.asarray(ca.evalf(shape[k])) for k in keys}
    out["length"] = length
    return out


def test_node_aero_matches_crossflow_plus_friction():
    """drag_n + lift_n reproduces the closed-form cross-flow + friction force.

    The magnitude is independent of direction, so this directly catches an
    unnormalised lift direction (which would scale the lift by sin(theta)).
    """
    cf = 0.02
    n = 10
    diameter = 0.01
    tether = WilliamsTether(
        diameter=diameter, density=950.0, n_elements=n, elastic=False, cf=cf
    )
    env = _Env(rho=1.225, g=9.81, wind=_UniformWind(speed=12.0))

    res = _evaluate_shape(tether, env, omega=[0.0, 0.1, 0.0])

    c_n = tether.drag_coefficient_tether
    l_unstrained = res["length"] / n

    # Interior nodes 1..n-1 carry the segment aero (rows 0 and N are zero).
    for k in range(1, n):
        speed = float(res["velocities_apparent_wind"][k, 0])
        theta = float(res["angle_va_tether"][k, 0])
        force = res["drag_per_node"][k] + res["lift_per_node"][k]

        q = 0.5 * env.rho * speed**2 * l_unstrained * diameter
        expected = q * np.hypot(
            c_n * np.sin(theta) ** 2, np.pi * cf * np.cos(theta) ** 2
        )
        assert float(np.linalg.norm(force)) == pytest.approx(
            expected, rel=1e-6, abs=1e-9
        )


def test_node_aero_is_normal_to_segment_without_friction():
    """With cf = 0 the force is pure cross-flow drag: perpendicular to the
    segment axis. An unnormalised lift direction would leave a spurious
    tangential component even here."""
    n = 12
    tether = WilliamsTether(
        diameter=0.01, density=950.0, n_elements=n, elastic=False, cf=0.0
    )
    env = _Env(rho=1.225, g=9.81, wind=_UniformWind(speed=12.0))

    res = _evaluate_shape(tether, env, omega=[0.0, 0.1, 0.0])
    tensions = res["tensions"]  # row k is the tension in segment k (axis dir)

    for k in range(1, n):
        ej = tensions[k] / (np.linalg.norm(tensions[k]) + 1e-12)
        force = res["drag_per_node"][k] + res["lift_per_node"][k]
        magnitude = np.linalg.norm(force)
        assert abs(float(np.dot(force, ej))) <= 1e-6 * (magnitude + 1e-9)
