import casadi as ca

from awetrim.kinematics.Kinematics import ParametrizedKinematics


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


def test_dot_vtau_and_chi_expressions():
    s = ca.MX.sym("s")
    r = ca.MX.sym("r")
    vr = ca.MX.sym("vr")
    s_dot = ca.MX.sym("s_dot")
    s_ddot = ca.MX.sym("s_ddot")

    pattern = DummyPattern()
    kite_model = DummyKiteModel(r, vr)
    phase = DummyPhase(s, kite_model, s_dot, s_ddot)

    pk = ParametrizedKinematics(pattern, phase)

    # Build the two equivalent forms for dot_vtau
    expr_code = pk.dot_vtau
    expr_manual = pk.sqrt_A * (
        pk.s_dot**2 * pk.dr_ds + pk.s_ddot * pk.r
    ) + pk.s_dot * pk.r * pk.dot_A / (2 * pk.sqrt_A)

    f_dot = ca.Function("f_dot", [s, r, vr, s_dot, s_ddot], [expr_code - expr_manual])

    # Test chi expression (construct dphi/dbeta manually from pattern)
    phi_expr = pattern.azimuth(r, s)
    beta_expr = pattern.elevation(r, s)
    dphi_manual = ca.gradient(phi_expr, s) + ca.gradient(phi_expr, r) * vr / s_dot
    dbeta_manual = ca.gradient(beta_expr, s) + ca.gradient(beta_expr, r) * vr / s_dot
    expr_chi_code = pk.chi
    expr_chi_manual = ca.atan2(dphi_manual * ca.cos(beta_expr), dbeta_manual)
    f_chi = ca.Function(
        "f_chi", [s, r, vr, s_dot, s_ddot], [expr_chi_code - expr_chi_manual]
    )

    # Numeric evaluation
    vals = (0.5, 80.0, 0.8, 0.25, 0.01)
    diff_dot = float(f_dot(*vals)[0])
    diff_chi = float(f_chi(*vals)[0])

    assert abs(diff_dot) < 1e-8
    assert abs(diff_chi) < 1e-8
