"""Tests for awetrim.kinematics.Kinematics.ParametrizedKinematics.

Consolidated from the three former files:
  test_parametrized_kinematics_symbolic.py   — vtau relation
  test_parametrized_kinematics_more.py       — dot_vtau and chi expressions
  test_parametrized_kinematics_all_eqs.py    — full kinematic equation suite
"""

import casadi as ca

from awetrim.kinematics.Kinematics import ParametrizedKinematics


# ---------------------------------------------------------------------------
# Shared test fixtures
# ---------------------------------------------------------------------------


class DummyPattern:
    def elevation(self, r, s):
        return 0.2 + 0.001 * r + 0.01 * s

    def azimuth(self, r, s):
        return 0.1 + 0.002 * r + 0.02 * s


class DummyKiteModel:
    def __init__(self, distance_radial, speed_radial):
        self.distance_radial = distance_radial
        self.speed_radial = speed_radial


class DummyPhase:
    def __init__(self, s, kite_model, s_dot, s_ddot):
        self.s = s
        self.kite_model = kite_model
        self.s_dot = s_dot
        self.s_ddot = s_ddot


def _make_pk():
    """Return (pk, symbols) for a ParametrizedKinematics with symbolic inputs."""
    s = ca.MX.sym("s")
    r = ca.MX.sym("r")
    vr = ca.MX.sym("vr")
    s_dot = ca.MX.sym("s_dot")
    s_ddot = ca.MX.sym("s_ddot")

    pattern = DummyPattern()
    kite_model = DummyKiteModel(r, vr)
    phase = DummyPhase(s, kite_model, s_dot, s_ddot)
    pk = ParametrizedKinematics(pattern, phase)

    return pk, (s, r, vr, s_dot, s_ddot)


_NUMERIC_VALS = (0.5, 80.0, 0.8, 0.25, 0.01)


def _assert_close(val1, val2, tol=1e-8):
    assert abs(float(val1) - float(val2)) < tol


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_vtau_relation():
    """vk² = vr² + vtau² must hold for all valid inputs."""
    pk, syms = _make_pk()
    s, r, vr, s_dot, s_ddot = syms

    expr = pk.vk**2 - pk.vr**2 - pk.vtau**2
    f = ca.Function("check", [s, r, vr, s_dot, s_ddot], [expr])

    val = float(f(0.5, 50.0, 1.0, 0.2, 0.0)[0])
    assert abs(val) < 1e-6


def test_dot_vtau_and_chi_expressions():
    """dot_vtau and chi match their manually derived equivalents."""
    pk, syms = _make_pk()
    s, r, vr, s_dot, s_ddot = syms
    pattern = DummyPattern()

    expr_dot_vtau_manual = pk.sqrt_A * (
        pk.s_dot**2 * pk.dr_ds + pk.s_ddot * pk.r
    ) + pk.s_dot * pk.r * pk.dot_A / (2 * pk.sqrt_A)

    phi_expr = pattern.azimuth(r, s)
    beta_expr = pattern.elevation(r, s)
    dphi_manual = ca.gradient(phi_expr, s) + ca.gradient(phi_expr, r) * vr / s_dot
    dbeta_manual = ca.gradient(beta_expr, s) + ca.gradient(beta_expr, r) * vr / s_dot
    expr_chi_manual = ca.atan2(dphi_manual * ca.cos(beta_expr), dbeta_manual)

    f_dot = ca.Function("f_dot", list(syms), [pk.dot_vtau - expr_dot_vtau_manual])
    f_chi = ca.Function("f_chi", list(syms), [pk.chi - expr_chi_manual])

    _assert_close(f_dot(*_NUMERIC_VALS)[0], 0.0)
    _assert_close(f_chi(*_NUMERIC_VALS)[0], 0.0)


def test_all_kinematics_equations():
    """Every ParametrizedKinematics output matches its manual derivation."""
    pk, syms = _make_pk()
    s, r, vr, s_dot, s_ddot = syms
    pattern = DummyPattern()

    phi_expr = pattern.azimuth(r, s)
    beta_expr = pattern.elevation(r, s)

    dphi_ds = ca.gradient(phi_expr, s) + ca.gradient(phi_expr, r) * vr / s_dot
    dbeta_ds = ca.gradient(beta_expr, s) + ca.gradient(beta_expr, r) * vr / s_dot
    dr_ds = vr / s_dot

    dR_ds_manual = ca.vertcat(
        r * dphi_ds * ca.cos(beta_expr), r * dbeta_ds, dr_ds
    )
    vk_manual = ca.norm_2(dR_ds_manual) * s_dot
    vtau_manual = ca.sqrt(vk_manual**2 - vr**2)

    dr_ds2_manual = ca.gradient(dr_ds, s)
    dbeta_ds2_manual = (
        ca.gradient(dbeta_ds, s) + ca.gradient(dbeta_ds, r) * vr / s_dot
    )
    dphi_ds2_manual = (
        ca.gradient(dphi_ds, s) + ca.gradient(dphi_ds, r) * vr / s_dot
    )

    sqrt_A_manual = vtau_manual / (s_dot * r)
    dot_A_manual = (
        2
        * s_dot
        * (
            dbeta_ds * dbeta_ds2_manual
            + dphi_ds * dphi_ds2_manual * ca.cos(beta_expr) ** 2
            - dphi_ds**2 * dbeta_ds * ca.sin(beta_expr) * ca.cos(beta_expr)
        )
    )
    dot_vr_manual = dr_ds2_manual * s_dot**2 + s_ddot * dr_ds
    dot_vtau_manual = sqrt_A_manual * (
        s_dot**2 * dr_ds + s_ddot * r
    ) + s_dot * r * dot_A_manual / (2 * sqrt_A_manual)
    chi_manual = ca.atan2(dphi_ds * ca.cos(beta_expr), dbeta_ds)
    dot_chi_manual = (
        ca.gradient(chi_manual, s) * s_dot + ca.gradient(chi_manual, r) * vr
    )

    inputs = list(syms)
    checks = {
        "dR_ds":     (pk.dR_ds,     dR_ds_manual),
        "vk":        (pk.vk,        vk_manual),
        "vtau":      (pk.vtau,      vtau_manual),
        "dr_ds2":    (pk.dr_ds2,    dr_ds2_manual),
        "dbeta_ds2": (pk.dbeta_ds2, dbeta_ds2_manual),
        "sqrt_A":    (pk.sqrt_A,    sqrt_A_manual),
        "dot_A":     (pk.dot_A,     dot_A_manual),
        "dot_vr":    (pk.dot_vr,    dot_vr_manual),
        "dot_vtau":  (pk.dot_vtau,  dot_vtau_manual),
        "chi":       (pk.chi,       chi_manual),
        "dot_chi":   (pk.dot_chi,   dot_chi_manual),
    }

    for name, (impl, manual) in checks.items():
        f = ca.Function(f"f_{name}", inputs, [impl - manual])
        _assert_close(f(*_NUMERIC_VALS)[0], 0.0)

    # extract_function must return a callable
    fk = pk.extract_function("vk")
    assert callable(fk)
