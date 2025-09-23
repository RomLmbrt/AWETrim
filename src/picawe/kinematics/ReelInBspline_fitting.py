import numpy as np
from scipy.optimize import least_squares
from picawe.kinematics.ReelInBspline_data_processing import ReelInBspline_data_processing as ribdata
from picawe.kinematics.ReelInBspline_build import ReelInBspline_build as ribbuild
import casadi as ca

class ReelInBspline_fitting(ribdata, ribbuild):

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
    B-spline fitting class for Reel-In trajectories using CasADi.
    Inherits all instance variables from ReelInBspline_data_processing.
    """

    def __init__(self, file_path_full, file_path_cycle, cyc_idx, p, n_ctrl,
                 c_penalty=1.0, v_penalty=0.0, eps_knot=1e-3):
        super().__init__(file_path_full, file_path_cycle, cyc_idx)

        self.p = p
        self.n_ctrl = n_ctrl
        self.c_penalty = c_penalty
        self.v_penalty = v_penalty
        self.eps_knot = eps_knot
        self.u_vals()
        self.fit_spline(mode="cartesian")
        self.fit_spline(mode="spherical")

    def u_vals(self):
        self.S_sph = np.vstack([self.az_ri, self.el_ri]).T
        self.S_cart = np.vstack([self.x_ri, self.y_ri, self.z_ri]).T

        dist = np.cumsum(np.linalg.norm(np.diff(self.S_cart, axis=0), axis=1))
        dist = np.insert(dist, 0, 0.0)
        self.u_vals = dist / dist[-1]

    # -------------------------------
    # Compute averages using CasADi derivatives
    # -------------------------------
    def compute_course_average_sph(self, dS_sph, S_sph, start=True):
        k = 10
        if start:
            course = -np.arctan2(dS_sph[:k, 0] * np.cos(S_sph[:k, 1]), dS_sph[:k, 1]) + 2*np.pi
        else:
            course = -np.arctan2(dS_sph[-k:, 0] * np.cos(S_sph[-k:, 1]), dS_sph[-k:, 1]) + 2*np.pi
        return np.mean(np.mod(course, 2*np.pi))

    def compute_velocity_average_cart(self, dS_cart, start=True):
        k = 10
        if start:
            velocity = dS_cart[:k, :]
        else:
            velocity = dS_cart[-k:, :]
        return np.mean(velocity, axis=0)

    # -------------------------------
    # Fit spline
    # -------------------------------
    def fit_spline(self, mode="spherical"):
        if not hasattr(self, 'u_vals'):
            self.u_vals()

        dim = 2 if mode=="spherical" else 3

        # Initial knot vector
        number_of_knots = self.n_ctrl + self.p + 1
        n_interior_knots = number_of_knots - 2*(self.p+1)
        if n_interior_knots <= 0:
            raise ValueError("Too few control points for spline order")

        U_interior_0 = np.linspace(0.15, 0.85, n_interior_knots+2)[1:-1]
        U0 = np.concatenate(([0]*(self.p+1), U_interior_0, [1]*(self.p+1)))

        # Select data and boundary points
        if mode=="spherical":
            S_data = self.S_sph
            ri_p0 = self.ri_p0_sph
            ri_pf = self.ri_pf_sph
        else:
            S_data = self.S_cart
            ri_p0 = self.ri_p0_cart
            ri_pf = self.ri_pf_cart

        # Initial control points
        C_inner_0 = np.zeros((self.n_ctrl-2, dim))
        print(f"Initial inner control points C_inner_0 has shape {C_inner_0.shape}")
        C0 = np.vstack([ri_p0, C_inner_0, ri_pf])
        print(f"Initial control points C0 has shape {C0.shape}")

        # Initial least-squares refinement (linear solve using basis functions)
        builder = ribbuild()
        # Pre-build symbolic spline function for initial knots
        C_sym = ca.MX.sym("C", self.n_ctrl, dim)
        u_sym = ca.MX.sym("u")
        S_sym, _, _ = builder.build_bspline(C_sym, self.p, U0, u_sym, return_derivative=True)
        spline_func = ca.Function("spline_func", [C_sym, u_sym], [S_sym], ["C", "u"], ["S"])

        # Build Nmat for initial linear solve
        Nmat = builder.build_Nmat(U0, self.p, self.u_vals)
        print(f"Nmat has shape {Nmat.shape}")
        rhs = S_data - (Nmat[:, [0, -1]] @ np.vstack([ri_p0, ri_pf]))
        C_inner_0, _, _, _ = np.linalg.lstsq(Nmat[:, 1:-1], rhs, rcond=None)
        print(f"Refined inner control points C_inner_0 has shape {C_inner_0.shape}")
        C0 = np.vstack([ri_p0, C_inner_0, ri_pf]).ravel()
        print(f"Initial control points C0 has shape {C0.shape}")

        # Knot increments
        du0 = np.ones(n_interior_knots) * 0.2
        print(f"Initial knot increments du0 has shape {du0.shape}")
        x0 = np.concatenate([C_inner_0.ravel(), du0])
        print(f"Initial guess x0 has shape {x0.shape}")

        # Bounds
        lb_C = np.full_like(C0[dim:-dim], -np.inf)
        ub_C = np.full_like(C0[dim:-dim], np.inf)
        lb_du = np.full_like(du0, self.eps_knot)
        ub_du = np.full_like(du0, 1-self.eps_knot)
        lb = np.concatenate([lb_C, lb_du])
        ub = np.concatenate([ub_C, ub_du])

        # -------------------------------
        # Residual function using CasADi evaluation
        # -------------------------------
        def residuals(params):
            C = params[:(self.n_ctrl-2)*dim].reshape(self.n_ctrl-2, dim)
            C = np.vstack([ri_p0, C, ri_pf])
            du = params[(self.n_ctrl-2)*dim:]
            U_interior = np.cumsum(du)
            U_interior /= (U_interior[-1] + 0.1)
            U = np.concatenate(([0]*(self.p+1), U_interior, [1]*(self.p+1)))

            # Build spline function for current knots
            S_sym, dS_sym, _ = builder.build_bspline(C_sym, self.p, U, u_sym, return_derivative=True)
            spline_func = ca.Function("spline_func", [C_sym, u_sym], [S_sym, dS_sym], ["C", "u"], ["S", "dS"])

            # Evaluate spline and derivative
            S_fit, dS_fit = builder.eval_spline(spline_func, C, self.u_vals)

            res_data = (S_fit - S_data).ravel()

            if mode=="spherical":
                avg_start = self.compute_course_average_sph(dS_fit, S_fit, start=True)
                avg_end = self.compute_course_average_sph(dS_fit, S_fit, start=False)
                res_extra = np.array([self.ri_crs0 - avg_start, self.ri_crsf - avg_end]) * self.c_penalty
            else:
                vel_start = self.compute_velocity_average_cart(dS_fit, start=True)
                vel_end = self.compute_velocity_average_cart(dS_fit, start=False)
                res_extra = np.ravel(np.array([vel_start - self.ri_v0, vel_end - self.ri_vf])) * self.v_penalty

            return np.concatenate([res_data, res_extra])

        # -------------------------------
        # Solve least squares
        # -------------------------------
        res = least_squares(residuals, x0, bounds=(lb, ub),
                            ftol=1e-8, xtol=1e-8, gtol=1e-8, verbose=2)

        # -------------------------------
        # Save optimized results
        # -------------------------------
        C_opt = res.x[:(self.n_ctrl-2)*dim].reshape(self.n_ctrl-2, dim)
        C_opt = np.vstack([ri_p0, C_opt, ri_pf])
        du_opt = res.x[(self.n_ctrl-2)*dim:]
        U_interior_opt = np.cumsum(du_opt)
        U_interior_opt /= (U_interior_opt[-1] + 0.1)
        U_opt = np.concatenate(([0]*(self.p+1), U_interior_opt, [1]*(self.p+1)))

        if mode=="spherical":
            self.C_sph, self.U_sph = C_opt, U_opt
            return self.C_sph, self.u_vals, self.U_sph
        else:
            self.C_cart, self.U_cart = C_opt, U_opt
            return self.C_cart, self.u_vals, self.U_cart

if __name__ == "__main__":
    # Test fitting class
    full_path = "/home/theophile/src/Simulation_Results/trial_Uri_valid/ProtoLogger_csv/2025-09-10_11-31-10_ProtoLogger.csv"
    cycle_path = "/home/theophile/src/Simulation_Results/trial_Uri_valid/cycles/cycle_data_sheet_lines.csv"
    cyc_idx = 0
    p = 3
    n_ctrl = 8
    c_penalty = 1
    v_penalty = 0
    eps_knot = 0.001

    fitter = ReelInBspline_fitting(full_path, cycle_path, cyc_idx,
                                   p, n_ctrl, c_penalty, v_penalty, eps_knot)