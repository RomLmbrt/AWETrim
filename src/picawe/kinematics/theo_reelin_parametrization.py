# -*- coding: utf-8 -*-
import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
from picawe.kinematics.Kinematics import ParametrizedKinematics
from picawe.system.system_model import SystemModel
from picawe.kinematics.ReelInBspline_parametrized import ReelInBspline


# =========================================================
# Base class: angles-only pattern (radians) + numeric eval
# =========================================================
class ParametrizedPatternsAngles:
    def azimuth(self, s):  # radians, CasADi MX/SX
        raise NotImplementedError

    def elevation(self, s):  # radians, CasADi MX/SX
        raise NotImplementedError

    def eval_angles(self, u_vec):
        u_vec = np.asarray(u_vec).reshape(-1)
        N = u_vec.size
        s = ca.MX.sym("s")
        phi_s, beta_s = self.azimuth(s), self.elevation(s)
        f_ab = ca.Function("f_ab", [s], [phi_s, beta_s]).map(N)
        phi_row, beta_row = f_ab(ca.DM(u_vec).T)
        return np.array(phi_row).ravel(), np.array(beta_row).ravel()

    def eval_xyz(self, u_vec, r_vec):
        u_vec = np.asarray(u_vec).reshape(-1)
        r_vec = np.asarray(r_vec).reshape(-1)
        assert u_vec.shape == r_vec.shape, "u_vec and r_vec must have same length"
        phi, beta = self.eval_angles(u_vec)
        x = r_vec * np.cos(beta) * np.cos(phi)
        y = r_vec * np.cos(beta) * np.sin(phi)
        z = r_vec * np.sin(beta)
        return x, y, z


# =========================================================
# Demo plotting
# =========================================================
def azel_to_vec(az_deg, el_deg):
    az = np.deg2rad(az_deg)
    el = np.deg2rad(el_deg)
    return np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])


if __name__ == "__main__":

    from picawe.kinematics.ReelInBspline_fitting import ReelInBspline_fitting as ribfit

    fitted = ribfit(
    file_path_full = "/home/theophile/src/Simulation_Results/trial_Uri_valid/ProtoLogger_csv/2025-09-10_11-31-10_ProtoLogger.csv",
    file_path_cycle = "/home/theophile/src/Simulation_Results/trial_Uri_valid/cycles/cycle_data_sheet_lines.csv",
    cyc_idx=0,
    p=3,
    n_ctrl=8,
    c_penalty=1.0,
    v_penalty=0.0,
    eps_knot=1e-3
    )

    # u-grid
    N = 100
    u = np.linspace(0.0, 1.0, N)

    # initial radii
    r_start0, r_end0 = 300.0, 150.0

    # r(u) profiles
    def r_profiles(r0, r1):
        r_lin = r0 + (r1 - r0) * u
        r_quad = r0 + (r1 - r0) * (u**2)
        r_half = np.where(u <= 0.5, r0 + (r1 - r0) * (u / 0.5), r1)
        return r_lin, r_quad, r_half

    # Create pattern instance (update parameters directly here)

    # # Placeholder for C_interior, U_interior and u_vals
    # C_interior = None
    # U_interior = None
    # u_vals = None

    mode = "spherical"  # "spherical" or "cartesian"

    pat = ReelInBspline(
        p=fitted.p,
        n_ctrl=fitted.n_ctrl,
        r0=300.0,
        rf=150.0,
        crs0=fitted.ri_crs0,
        crsf=fitted.ri_crsf,
        phi0=fitted.ri_p0_sph[0],
        phif=fitted.ri_pf_sph[0],
        beta0=fitted.ri_p0_sph[1],
        betaf=fitted.ri_pf_sph[1],
        C_interior=fitted.C_sph[1:-1] if mode=="spherical" else fitted.C_cart[1:-1],
        u_vals=u,
        U_interior=fitted.U_sph[fitted.p+1:-(fitted.p+1)] if mode=="spherical" else fitted.U_cart
    )

    # Create kinematics object
    class Phase:
        def __init__(self):
            self.s = ca.MX.sym("s")
            self.s_dot = ca.MX.sym("s_dot")
            self.s_ddot = ca.MX.sym("s_ddot")
            self.vr = ca.MX.sym("vr")
            self.t = ca.MX.sym("t")
            self.kite_model = SystemModel()

    phase = Phase()
    kinematics = ParametrizedKinematics(pat, phase)

    # -------------------------------------------------
    # Plotting
    # -------------------------------------------------
    fig = plt.figure(figsize=(12, 7))
    ax3d = fig.add_subplot(121, projection="3d")
    axr = fig.add_subplot(222)
    axt = fig.add_subplot(224)
    plt.subplots_adjust(bottom=0.15, wspace=0.25, hspace=0.35)

    # r(u) profiles
    r_lin, r_quad, r_half = r_profiles(r_start0, r_end0)

    # Evaluate 3D
    x1, y1, z1 = pat.eval_xyz(u, r_lin)
    x2, y2, z2 = pat.eval_xyz(u, r_quad)
    x3, y3, z3 = pat.eval_xyz(u, r_half)

    # Draw 3D
    ax3d.plot(x1, y1, z1, lw=2, label="r linear")
    ax3d.plot(x2, y2, z2, lw=2, ls="--", label="r quadratic")
    ax3d.plot(x3, y3, z3, lw=2, ls="-.", label="r half→const")
    ax3d.set_xlabel("X")
    ax3d.set_ylabel("Y")
    ax3d.set_zlabel("Z")
    ax3d.set_title("φ(u), β(u) with r(u) profiles")
    ax3d.legend()
    all_pts = np.vstack([np.c_[x1, y1, z1], np.c_[x2, y2, z2], np.c_[x3, y3, z3]])
    ax3d.set_xlim(all_pts[:, 0].min(), all_pts[:, 0].max())
    ax3d.set_ylim(all_pts[:, 1].min(), all_pts[:, 1].max())
    ax3d.set_zlim(all_pts[:, 2].min(), all_pts[:, 2].max())

    # r(u) subplot
    axr.plot(u, r_lin, lw=2, label="linear")
    axr.plot(u, r_quad, lw=2, ls="--", label="quadratic")
    axr.plot(u, r_half, lw=2, ls="-.", label="half→const")
    axr.set_xlabel("u")
    axr.set_ylabel("r (m)")
    axr.set_title("r(u) profiles")
    axr.set_xlim(0, 1)
    axr.set_ylim(
        min(r_lin.min(), r_quad.min(), r_half.min()),
        max(r_lin.max(), r_quad.max(), r_half.max()),
    )
    axr.legend()

    # Angles subplot
    phi_u, beta_u = pat.eval_angles(u)
    axt.plot(u, np.rad2deg(phi_u), lw=2, label="φ(u)")
    axt.plot(u, np.rad2deg(beta_u), lw=2, label="β(u)")
    axt.set_xlabel("u")
    axt.set_title("Angles (deg)")
    lo = min(np.rad2deg(phi_u).min(), np.rad2deg(beta_u).min())
    hi = max(np.rad2deg(phi_u).max(), np.rad2deg(beta_u).max())
    pad = 0.05 * (hi - lo if hi > lo else 1.0)
    axt.set_ylim(lo - pad, hi + pad)
    axt.legend()

    plt.show()
