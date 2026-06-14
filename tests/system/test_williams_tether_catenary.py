"""Analytical catenary verification for ``WilliamsTether``.

When the tether hangs under its own weight alone — no wind, no aerodynamic
drag, no kite rotation — its shape is an inextensible catenary. This test
pins down a catenary by choosing the parameter ``a = H/w`` (horizontal
tension over weight-per-length) and the horizontal span ``x_k``, derives the
matching kite position, total length, kite-end tension, and tangent angle
analytically, hands those into ``WilliamsTether``, and checks that the
solver recovers the same shape.

The configuration places the catenary low point at the ground anchor
(``x_b = 0``), giving the closed form::

    z(x)        = a * (cosh(x/a) - 1)
    s(x)        = a * sinh(x/a)               (arc length from ground)
    T(x)        = w * a * cosh(x/a)            (tension magnitude)
    H           = w * a                        (horizontal tension, constant)
    elevation_k = atan(sinh(x_k/a))            (tangent angle at the kite)
"""

from __future__ import annotations

from dataclasses import dataclass

import casadi as ca
import numpy as np
import pytest
from scipy.optimize import least_squares

from awetrim.system.williams_tether import WilliamsTether


@dataclass
class _StandaloneEnv:
    """Minimal env object the standalone tether API consumes.

    The tether reads ``.wind`` (for the apparent-wind term), ``.rho`` and
    ``.g``. For pure-gravity catenary tests ``wind`` is ``None`` and
    ``rho = 0`` disables the aero term entirely.
    """

    rho: float
    g: float
    wind: object = None


# ---------------------------------------------------------------------------
# Analytical catenary fixture
# ---------------------------------------------------------------------------


def test_williams_elastic_defines_axial_stiffness():
    """WilliamsTether(elastic=True) must define EA = E·area and build the
    elastic shape without error (regression for the undefined self.EA)."""
    tether = WilliamsTether(
        E=100e9, diameter=0.01, density=950.0, n_elements=5, elastic=True
    )
    assert tether.EA == pytest.approx(tether.E * tether.area_tether)

    env = _StandaloneEnv(rho=0.0, g=9.81, wind=None)
    shape = tether.tether_shape_symbolic(
        env=env,
        r_kite=np.array([10.0, 0.0, 100.0]),
        force_kite_resultant=np.array([0.0, 0.0, 1000.0]),
    )
    # Elastic stretch path executed: stretched length is a real expression.
    assert shape["tether_length_stretched"] is not None


@pytest.fixture(scope="module")
def catenary():
    """Closed-form hanging-tether catenary with low point at the ground."""
    a = 80.0
    x_k = 100.0
    diameter = 0.01
    density = 970.0
    g = 9.81

    w = np.pi * diameter**2 / 4.0 * density * g
    z_k = a * (np.cosh(x_k / a) - 1.0)
    length = a * np.sinh(x_k / a)
    tension_kite = w * a * np.cosh(x_k / a)
    horizontal_tension = w * a
    elevation_kite = float(np.arctan(np.sinh(x_k / a)))

    return {
        "a": a,
        "x_k": x_k,
        "z_k": z_k,
        "length": length,
        "tension_kite": tension_kite,
        "horizontal_tension": horizontal_tension,
        "elevation_kite": elevation_kite,
        "diameter": diameter,
        "density": density,
        "g": g,
        "w": w,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _hanging_env(g: float) -> _StandaloneEnv:
    """Env disabling aero (rho=0) and wind (wind=None) for a pure-gravity
    catenary. Rotation is passed as ``omega=0`` at call sites."""
    return _StandaloneEnv(rho=0.0, g=g, wind=None)


def _pack_params(param_names, r_kite, force_kite):
    table = {
        "r_kite": np.asarray(r_kite, dtype=float).reshape(-1),
        "force_kite_resultant": np.asarray(force_kite, dtype=float).reshape(-1),
    }
    return np.concatenate([table[n] for n in param_names])


def _solve(tether: WilliamsTether, env: _StandaloneEnv, r_kite, force_kite):
    omega = ca.DM.zeros(3)
    residual_fun, names = tether.residual_function(env=env, omega=omega)
    jac_fun, _ = tether.residual_jacobian_function(env=env, omega=omega)
    p = _pack_params(names, r_kite, force_kite)

    # Initial guess: straight line from origin to kite, slightly stretched so
    # the solver has room to grow the length toward the curved catenary.
    distance = float(np.linalg.norm(r_kite))
    elev_guess = float(np.arctan2(r_kite[2], np.hypot(r_kite[0], r_kite[1])))
    az_guess = float(np.arctan2(r_kite[1], r_kite[0]))
    x0 = np.array([elev_guess, az_guess, distance * 1.05])

    sol = least_squares(
        lambda x: np.asarray(residual_fun(x=x, p=p)["residual"]).reshape(-1),
        x0,
        jac=lambda x: np.asarray(jac_fun(x=x, p=p)["jac"]),
        method="lm",
        xtol=1e-12,
        ftol=1e-12,
    )
    return sol, p


def _analytical_node_positions(catenary, n_elements):
    """Catenary node positions sampled at the same arc lengths Williams uses."""
    a = catenary["a"]
    length = catenary["length"]
    arc = np.linspace(0.0, length, n_elements + 1)
    x = a * np.arcsinh(arc / a)
    z = a * (np.cosh(x / a) - 1.0)
    return np.column_stack([x, np.zeros_like(x), z])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_recovers_catenary_length_and_kite_angle(catenary):
    """Solver recovers analytic length, tangent angle, and azimuth."""
    tether = WilliamsTether(
        diameter=catenary["diameter"],
        density=catenary["density"],
        n_elements=200,
        elastic=False,
    )
    env = _hanging_env(catenary["g"])

    r_kite = np.array([catenary["x_k"], 0.0, catenary["z_k"]])
    # Only the magnitude of force_kite_resultant is consumed by Williams; its
    # direction is left to the solver via (elevation_last, azimuth_last).
    force_kite = np.array([0.0, 0.0, catenary["tension_kite"]])

    sol, _ = _solve(tether, env, r_kite, force_kite)
    assert sol.success, sol.message
    elevation, azimuth, length = sol.x.tolist()

    assert np.max(np.abs(sol.fun)) < 1e-6  # ground residual closed
    assert length == pytest.approx(catenary["length"], rel=5e-3)
    assert elevation == pytest.approx(catenary["elevation_kite"], abs=5e-3)
    assert azimuth == pytest.approx(0.0, abs=1e-8)


def test_shape_matches_analytical_catenary(catenary):
    """Pointwise positions of the discrete tether match the closed-form curve."""
    n_elements = 200
    tether = WilliamsTether(
        diameter=catenary["diameter"],
        density=catenary["density"],
        n_elements=n_elements,
        elastic=False,
    )
    env = _hanging_env(catenary["g"])

    r_kite = np.array([catenary["x_k"], 0.0, catenary["z_k"]])
    force_kite = np.array([0.0, 0.0, catenary["tension_kite"]])

    sol, p = _solve(tether, env, r_kite, force_kite)
    shape_fun, _ = tether.shape_function(env=env, omega=ca.DM.zeros(3))
    positions = np.asarray(shape_fun(x=sol.x, p=p)["positions"])

    expected = _analytical_node_positions(catenary, n_elements)
    max_err = float(np.linalg.norm(positions - expected, axis=1).max())

    # 1% of the tether length is comfortably above the O(1/N) discretization
    # error from the lumped-mass model.
    assert max_err < 0.01 * catenary["length"], (
        f"max position error {max_err:.4f} m exceeds 1% of length "
        f"{catenary['length']:.4f} m"
    )


def test_horizontal_tension_is_constant_along_cable(catenary):
    """For a purely hanging cable, the horizontal tension is invariant."""
    tether = WilliamsTether(
        diameter=catenary["diameter"],
        density=catenary["density"],
        n_elements=200,
        elastic=False,
    )
    env = _hanging_env(catenary["g"])

    r_kite = np.array([catenary["x_k"], 0.0, catenary["z_k"]])
    force_kite = np.array([0.0, 0.0, catenary["tension_kite"]])

    sol, p = _solve(tether, env, r_kite, force_kite)
    shape_fun, _ = tether.shape_function(env=env, omega=ca.DM.zeros(3))
    tensions = np.asarray(shape_fun(x=sol.x, p=p)["tensions"])

    horizontal_norm = np.linalg.norm(tensions[:, :2], axis=1)
    # H is exactly constant in the discrete model too (only gravity acts on
    # nodes, which is purely vertical) — so this is a tight check.
    assert np.allclose(
        horizontal_norm, horizontal_norm[0], rtol=1e-8, atol=1e-8
    )
    # And the constant value matches the analytical H = w * a within the
    # discretization error.
    assert horizontal_norm[0] == pytest.approx(
        catenary["horizontal_tension"], rel=1e-2
    )


def test_converges_toward_catenary_as_n_increases(catenary):
    """Refining the discretization brings the discrete shape closer to the
    closed-form catenary monotonically."""
    errors = []
    for n in (20, 80, 320):
        tether = WilliamsTether(
            diameter=catenary["diameter"],
            density=catenary["density"],
            n_elements=n,
            elastic=False,
        )
        env = _hanging_env(catenary["g"])

        r_kite = np.array([catenary["x_k"], 0.0, catenary["z_k"]])
        force_kite = np.array([0.0, 0.0, catenary["tension_kite"]])

        sol, p = _solve(tether, env, r_kite, force_kite)
        shape_fun, _ = tether.shape_function(env=env, omega=ca.DM.zeros(3))
        positions = np.asarray(shape_fun(x=sol.x, p=p)["positions"])
        expected = _analytical_node_positions(catenary, n)
        errors.append(float(np.linalg.norm(positions - expected, axis=1).max()))

    assert errors[1] < errors[0]
    assert errors[2] < errors[1]
