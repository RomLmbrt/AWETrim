import numpy as np
import matplotlib.pyplot as plt
from picawe.kinematics.ReelInBspline_fitting import ReelInBspline_fitting as ribfit
from picawe.kinematics.ReelInBspline_build import ReelInBspline_build as ribbuild
import casadi as ca

# -------------------------------
# Plotting
# -------------------------------
class ReelInBspline_plotting(ribfit, ribbuild):

    # something0 or somethingf means the start or end 0 for start and f for final
    # p - point eg. p0 start point
    # v - velocity
    # crs - course
    # idx - index
    # cyc - cycle
    # ri - reel-in
    # ro - reel-out
    # sph - spherical
    # cart - cartesian

    """
    Plotting class for B-spline fits using CasADi-based evaluation.
    Inherits from the fitting class and the spline builder.
    """

    def __init__(self, file_path_full, file_path_cycle, cyc_idx, p, n_ctrl, c_penalty=1, v_penalty=0, eps_knot=0.001):
        super().__init__(file_path_full, file_path_cycle, cyc_idx, p, n_ctrl, c_penalty, v_penalty, eps_knot)

    def plot_spline_fit_cart(self):
        if self.C_cart is None or self.U_cart is None:
            raise ValueError("Run fit_spline(mode='cartesian') before plotting.")

        builder = ribbuild()

        # Build symbolic spline + derivative
        C_sym = ca.MX.sym("C", self.C_cart.shape[0], self.C_cart.shape[1])
        u_sym = ca.MX.sym("u")
        S_sym, dS_sym, _ = builder.build_bspline(C_sym, self.p, self.U_cart, u_sym, return_derivative=True)

        # Function returns BOTH spline and derivative
        spline_func = ca.Function("spline_func_cart", [C_sym, u_sym], [S_sym, dS_sym], ["C", "u"], ["S", "dS"])

        # Evaluate spline (ignore derivative for plotting)
        S_fit_cart, _ = builder.eval_spline(spline_func, self.C_cart, self.u_vals)

        # 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(self.x_cyc, self.y_cyc, self.z_cyc, label="Trajectory", alpha=0.6)
        ax.plot(S_fit_cart[:,0], S_fit_cart[:,1], S_fit_cart[:,2], "r--", label="B-spline fit")

        # Control points
        ax.scatter(self.C_cart[:,0], self.C_cart[:,1], self.C_cart[:,2], color="black", s=30, label="Control points")
        # ri start/end
        ax.scatter(*self.ri_p0_cart, color="green", s=30, label="ri start")
        ax.scatter(*self.ri_pf_cart, color="red", s=30, label="ri end")

        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        ax.legend(); ax.set_box_aspect([1,1,1])
        plt.show()

    def plot_spline_fit_sph(self):
        if self.C_sph is None or self.U_sph is None:
            raise ValueError("Run fit_spline(mode='spherical') before plotting.")
        if self.az_ri is None or self.el_ri is None:
            raise ValueError("Run get_ri_ro_boundaries() before plotting.")

        builder = ribbuild()

        # Build symbolic spline + derivative
        C_sym = ca.MX.sym("C", self.C_sph.shape[0], self.C_sph.shape[1])
        u_sym = ca.MX.sym("u")
        S_sym, dS_sym, _ = builder.build_bspline(C_sym, self.p, self.U_sph, u_sym, return_derivative=True)

        # Function returns BOTH spline and derivative
        spline_func = ca.Function("spline_func_sph", [C_sym, u_sym], [S_sym, dS_sym], ["C", "u"], ["S", "dS"])

        # Evaluate spline (ignore derivative for plotting)
        S_fit_sph, _ = builder.eval_spline(spline_func, self.C_sph, self.u_vals)

        # Plot azimuth and elevation
        fig, (ax_az, ax_el) = plt.subplots(1, 2, figsize=(10,4), sharex=True)
        ax_az.plot(self.u_vals, self.az_ri, label="Azimuth (data)", color="C0")
        ax_az.plot(self.u_vals, S_fit_sph[:,0], "--", label="Azimuth (spline)", color="C1")
        ax_el.plot(self.u_vals, self.el_ri, label="Elevation (data)", color="C0")
        ax_el.plot(self.u_vals, S_fit_sph[:,1], "--", label="Elevation (spline)", color="C1")

        # Control points
        u_cp = np.linspace(self.u_vals[0], self.u_vals[-1], len(self.C_sph))
        ax_az.scatter(u_cp, self.C_sph[:,0], color="black", s=30, label="Control points")
        ax_el.scatter(u_cp, self.C_sph[:,1], color="black", s=30, label="Control points")

        # ri start/end
        ax_az.scatter(self.u_vals[0], self.ri_p0_sph[0], color="green", s=30, label="ri start")
        ax_az.scatter(self.u_vals[-1], self.ri_pf_sph[0], color="red", s=30, label="ri end")
        ax_el.scatter(self.u_vals[0], self.ri_p0_sph[1], color="green", s=30, label="ri start")
        ax_el.scatter(self.u_vals[-1], self.ri_pf_sph[1], color="red", s=30, label="ri end")

        ax_az.set_xlabel("u"); ax_az.set_ylabel("Azimuth [rad]"); ax_az.grid(True, alpha=0.3); ax_az.legend()
        ax_el.set_xlabel("u"); ax_el.set_ylabel("Elevation [rad]"); ax_el.grid(True, alpha=0.3); ax_el.legend()
        fig.tight_layout()
        plt.show()
