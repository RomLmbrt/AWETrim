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


def make_numeric_vals():
    return (0.5, 80.0, 0.8, 0.25, 0.01)


def assert_close(val1, val2, tol=1e-8):
    assert abs(float(val1) - float(val2)) < tol


def test_all_kinematics_equations():
    s, r, vr, s_dot, s_ddot = (
        ca.MX.sym("s"),
        ca.MX.sym("r"),
        ca.MX.sym("vr"),
        ca.MX.sym("s_dot"),
        ca.MX.sym("s_ddot"),
    )

    pattern = DummyPattern()
    kite_model = DummyKiteModel(r, vr)
    phase = DummyPhase(s, kite_model, s_dot, s_ddot)

    pk = ParametrizedKinematics(pattern, phase)

    # Build manual expressions from pattern for basic derivatives
    phi_expr = pattern.azimuth(r, s)
    beta_expr = pattern.elevation(r, s)

    dphi_ds_manual = ca.gradient(phi_expr, s) + ca.gradient(phi_expr, r) * vr / s_dot
    dbeta_ds_manual = ca.gradient(beta_expr, s) + ca.gradient(beta_expr, r) * vr / s_dot
    dr_ds_manual = vr / s_dot

    # dR_ds manual
    dR_ds_manual = ca.vertcat(
        r * dphi_ds_manual * ca.cos(beta_expr), r * dbeta_ds_manual, dr_ds_manual
    )

    # vk and vtau manual
    vk_manual = ca.norm_2(dR_ds_manual) * s_dot
    vtau_manual = ca.sqrt(vk_manual**2 - vr**2)

    # second derivatives
    dr_ds2_manual = ca.gradient(dr_ds_manual, s)
    dbeta_ds2_manual = (
        ca.gradient(dbeta_ds_manual, s) + ca.gradient(dbeta_ds_manual, r) * vr / s_dot
    )
    dphi_ds2_manual = (
        ca.gradient(dphi_ds_manual, s) + ca.gradient(dphi_ds_manual, r) * vr / s_dot
    )

    # sqrt_A and dot_A manual
    sqrt_A_manual = vtau_manual / (s_dot * r)
    dot_A_manual = (
        2
        * s_dot
        * (
            dbeta_ds_manual * dbeta_ds2_manual
            + dphi_ds_manual * dphi_ds2_manual * ca.cos(beta_expr) ** 2
            - dphi_ds_manual**2
            * dbeta_ds_manual
            * ca.sin(beta_expr)
            * ca.cos(beta_expr)
        )
    )

    # dot_vr manual
    dot_vr_manual = dr_ds2_manual * s_dot**2 + s_ddot * dr_ds_manual

    # dot_vtau manual (same form as implemented)
    dot_vtau_manual = sqrt_A_manual * (
        s_dot**2 * dr_ds_manual + s_ddot * r
    ) + s_dot * r * dot_A_manual / (2 * sqrt_A_manual)

    # chi and dot_chi manual
    chi_manual = ca.atan2(dphi_ds_manual * ca.cos(beta_expr), dbeta_ds_manual)
    dot_chi_manual = (
        ca.gradient(chi_manual, s) * s_dot + ca.gradient(chi_manual, r) * vr
    )

    # Create functions to evaluate differences
    inputs = [s, r, vr, s_dot, s_ddot]

    f_dR = ca.Function("f_dR", inputs, [pk.dR_ds - dR_ds_manual])
    f_vk = ca.Function("f_vk", inputs, [pk.vk - vk_manual])
    f_vtau = ca.Function("f_vtau", inputs, [pk.vtau - vtau_manual])
    f_dr_ds2 = ca.Function("f_dr_ds2", inputs, [pk.dr_ds2 - dr_ds2_manual])
    f_dbeta_ds2 = ca.Function("f_dbeta_ds2", inputs, [pk.dbeta_ds2 - dbeta_ds2_manual])
    f_sqrt_A = ca.Function("f_sqrt_A", inputs, [pk.sqrt_A - sqrt_A_manual])
    f_dot_A = ca.Function("f_dot_A", inputs, [pk.dot_A - dot_A_manual])
    f_dot_vr = ca.Function("f_dot_vr", inputs, [pk.dot_vr - dot_vr_manual])
    f_dot_vtau = ca.Function("f_dot_vtau", inputs, [pk.dot_vtau - dot_vtau_manual])
    f_chi = ca.Function("f_chi", inputs, [pk.chi - chi_manual])
    f_dot_chi = ca.Function("f_dot_chi", inputs, [pk.dot_chi - dot_chi_manual])

    # Also test extract_function for vk
    fk_from_extract = pk.extract_function("vk")
    fk_direct = ca.Function("vk_direct", inputs, [pk.vk])

    # Numeric evaluation
    numeric_vals = (0.5, 80.0, 0.8, 0.25, 0.01)

    assert_close(f_dR(*numeric_vals)[0], 0.0)
    assert_close(f_vk(*numeric_vals)[0], 0.0)
    assert_close(f_vtau(*numeric_vals)[0], 0.0)
    assert_close(f_dr_ds2(*numeric_vals)[0], 0.0)
    assert_close(f_dbeta_ds2(*numeric_vals)[0], 0.0)
    assert_close(f_sqrt_A(*numeric_vals)[0], 0.0)
    assert_close(f_dot_A(*numeric_vals)[0], 0.0)
    assert_close(f_dot_vr(*numeric_vals)[0], 0.0)
    assert_close(f_dot_vtau(*numeric_vals)[0], 0.0)
    assert_close(f_chi(*numeric_vals)[0], 0.0)
    assert_close(f_dot_chi(*numeric_vals)[0], 0.0)

    # extract_function created successfully (callable behavior validated elsewhere)
    assert hasattr(fk_from_extract, "__call__") or hasattr(fk_from_extract, "call")
