import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

# -------------------------------
# Helper Functions
# -------------------------------
def convert_time_to_seconds(time_array):
    """Convert HH:MM:SS.sss to seconds (float)."""
    seconds_array = []
    for time_str in time_array:
        h, m, s = map(float, str(time_str).split(":"))
        seconds_array.append(h * 3600 + m * 60 + s)
    return np.array(seconds_array)

def sph2cart(az, el, r):
    x = r * np.cos(el) * np.cos(az)
    y = r * np.cos(el) * np.sin(az)
    z = r * np.sin(el)
    return x, y, z

def cart2sph(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    az = np.arctan2(y, x)
    el = np.arcsin(z / r)
    return az, el, r

def evaluate_bspline(C, p, U, u, return_basis=False, return_derivative=False):
    """
    Evaluate B-spline at u (scalar or array).
    If return_basis=True, returns basis matrix Nmat.
    If return_derivative=True, also returns derivative basis matrix dNmat.
    """
    u = np.atleast_1d(u)
    n_ctrl = C.shape[0]
    Nmat = np.zeros((len(u), n_ctrl))
    dNmat = np.zeros((len(u), n_ctrl))

    def N(i, k, u_val):
        if k == 0:
            return 1.0 if (U[i] <= u_val <= U[i+1]) else 0.0
        left = 0.0
        right = 0.0
        if U[i+k] > U[i]:
            left = (u_val - U[i])/(U[i+k]-U[i]) * N(i, k-1, u_val)
        if U[i+k+1] > U[i+1]:
            right = (U[i+k+1]-u_val)/(U[i+k+1]-U[i+1]) * N(i+1, k-1, u_val)
        return left + right

    def dN(i, k, u_val):
        if k == 0:
            return 0.0
        left = 0.0
        right = 0.0
        if U[i+k] > U[i]:
            left = k/(U[i+k]-U[i]) * N(i, k-1, u_val)
        if U[i+k+1] > U[i+1]:
            right = k/(U[i+k+1]-U[i+1]) * N(i+1, k-1, u_val)
        return left - right

    for ui, u_val in enumerate(u):
        for i in range(n_ctrl):
            Nmat[ui, i] = N(i, p, u_val)
            if return_derivative:
                dNmat[ui, i] = dN(i, p, u_val)

    S = Nmat @ C
    if return_basis and return_derivative:
        return S, Nmat, dNmat
    elif return_basis:
        return S, Nmat
    elif return_derivative:
        return S, dNmat
    else:
        return S


# -------------------------------
# Cycle Class
# -------------------------------
class Cycle:
    def __init__(self, file_path_full, file_path_cycle, cycle_idx=0):
        """Initialize Cycle object from CSV files and cycle index."""
        # Load CSVs
        self.full_df = pd.read_csv(file_path_full)
        self.cycle_df = pd.read_csv(file_path_cycle)
        self.cycle_idx = cycle_idx

        # Preprocess times
        self.full_df['time_s'] = np.round(convert_time_to_seconds(self.full_df['time_of_day'].to_numpy()), 1)
        self.cycle_df['start_time_s'] = np.round(convert_time_to_seconds(self.cycle_df['start_time_cycle_LT'].to_numpy()), 1)

        # Extract variables
        self.time_full = self.full_df['time_s'].to_numpy()
        self.time_cycle = self.cycle_df['start_time_s'].to_numpy()
        self.az = self.full_df['kite_azimuth'].to_numpy()
        self.el = self.full_df['kite_elevation'].to_numpy()
        self.r = self.full_df['kite_distance'].to_numpy()
        self.phase = self.full_df['flight_phase'].to_numpy()

        # Compute cycle boundaries
        self._compute_cycle_indices()
        self._extract_cycle_data()

        # B-spline variables
        self.C_cart, self.C_sph, self.p, self.U = None, None, None, None
        self.u_vals = None

        # Reel-In points and velocities
        self.ri_start_point, self.ri_end_point = None, None
        self.ri_start_velocity, self.ri_end_velocity = None, None
        self.az_RI, self.el_RI, self.r_RI = None, None, None
        self.x_RI, self.y_RI, self.z_RI = None, None, None

    # -------------------------------
    # Internal methods
    # -------------------------------
    def _compute_cycle_indices(self):
        """Find start/end indices of the selected cycle."""
        self.start_indices = np.array([
            i for i, t in enumerate(self.time_full) 
            for tc in self.time_cycle if t == tc
        ])
        if self.cycle_idx >= len(self.start_indices)-1:
            raise IndexError("cycle_idx out of range")
        self.cycle_start_idx = self.start_indices[self.cycle_idx]
        self.cycle_end_idx   = self.start_indices[self.cycle_idx + 1] - 1 if self.cycle_idx + 1 < len(self.start_indices) else len(self.time_full)-1

    def _extract_cycle_data(self):
        """Extract only the selected cycle data (spherical and cartesian)."""
        self.az_cyc = self.az[self.cycle_start_idx:self.cycle_end_idx+1]
        self.el_cyc = self.el[self.cycle_start_idx:self.cycle_end_idx+1]
        self.r_cyc  = self.r[self.cycle_start_idx:self.cycle_end_idx+1]
        self.phase_cyc = self.phase[self.cycle_start_idx:self.cycle_end_idx+1]

        self.x_cyc, self.y_cyc, self.z_cyc = sph2cart(self.az_cyc, self.el_cyc, self.r_cyc)
        self.dx_cyc, self.dy_cyc, self.dz_cyc = np.gradient(self.x_cyc), np.gradient(self.y_cyc), np.gradient(self.z_cyc)
        self.num_points = len(self.x_cyc)

    # -------------------------------
    # Reel-In/Out boundaries
    # -------------------------------
    def get_RI_RO_boundaries(self):
        """Compute start/end points and velocities of Reel-In."""
        # Find start of Reel-In
        RI_start_idx = None
        for i, tag in enumerate(self.phase_cyc):
            if tag.lower() in ["pp-ri", "pp-rori", "pp-riro"]:
                RI_start_idx = i
                break
        if RI_start_idx is None:
            raise ValueError("Reel-In start not found in this cycle")

        RI_end_idx = len(self.phase_cyc) - 1  # Last point of cycle

        self.RI_start_idx, self.RI_end_idx = RI_start_idx, RI_end_idx

        # Store points and velocities
        self.ri_start_point = np.array([self.x_cyc[RI_start_idx], self.y_cyc[RI_start_idx], self.z_cyc[RI_start_idx]])
        self.ri_end_point   = np.array([self.x_cyc[RI_end_idx], self.y_cyc[RI_end_idx], self.z_cyc[RI_end_idx]])
        self.ri_start_velocity = np.array([self.dx_cyc[RI_start_idx], self.dy_cyc[RI_start_idx], self.dz_cyc[RI_start_idx]])
        self.ri_end_velocity   = np.array([self.dx_cyc[RI_end_idx], self.dy_cyc[RI_end_idx], self.dz_cyc[RI_end_idx]])

        # Spherical RI segment
        self.az_RI = self.az_cyc[RI_start_idx:RI_end_idx+1]
        self.el_RI = self.el_cyc[RI_start_idx:RI_end_idx+1]
        self.r_RI  = self.r_cyc[RI_start_idx:RI_end_idx+1]

        # Cartesian RI segment
        self.x_RI, self.y_RI, self.z_RI = self.x_cyc[RI_start_idx:RI_end_idx+1], self.y_cyc[RI_start_idx:RI_end_idx+1], self.z_cyc[RI_start_idx:RI_end_idx+1]

        return (self.ri_start_point, self.ri_end_point,
                self.ri_start_velocity, self.ri_end_velocity,
                self.az_RI, self.el_RI, self.r_RI,
                self.RI_start_idx, self.RI_end_idx)

    # -------------------------------
    # B-spline fitting (Cartesian)
    # -------------------------------

    def fit_cartesian_spline(self, p=3, n_ctrl=7, vel_penalty=15.0, eps_knot=1e-3):
        """
        Fit a B-spline to the Reel-In segment, optimizing control points
        and interior knots (via du increments), while keeping start/end
        points fixed and optionally penalizing start/end velocities.
        """
        # -------------------
        # Prepare data
        # -------------------
        self.S_cart = np.vstack([self.x_RI, self.y_RI, self.z_RI]).T
        dist = np.cumsum(np.linalg.norm(np.diff(self.S_cart, axis=0), axis=1))
        dist = np.insert(dist, 0, 0.0)
        self.u_vals = dist / dist[-1]

        # -------------------
        # Number of interior knots (unclamped knots)
        # -------------------
        number_of_knots = n_ctrl + p + 1
        n_interior_knots = (number_of_knots - 2)

        if n_interior_knots <= 0:
            raise ValueError("Too few control points for spline order")

        # -------------------
        # Initial guess
        # -------------------
        # LSQ ignoring velocities to get initial control points
        U0 = np.linspace(0, 1, number_of_knots)

        _, Nmat = evaluate_bspline(np.zeros((n_ctrl,3)), p, U0, self.u_vals, return_basis=True)
        C0, _, _, _ = np.linalg.lstsq(Nmat, self.S_cart, rcond=None)

        # Interior knots as evenly spaced increments
        du0 = np.ones(n_interior_knots) / (n_interior_knots + 1)

        # Flatten parameters: control points (interior only) + du increments
        # Fix start/end control points
        C_0 = C0.ravel()

        x0 = np.concatenate([C_0, du0])

        # -------------------
        # Bounds
        # -------------------
        lb_C = np.full_like(C_0, -1e4)
        ub_C = np.full_like(C_0, 1e4)

        lb_du = np.full_like(du0, eps_knot)
        ub_du = np.full_like(du0, 1.0 - eps_knot)
        lb = np.concatenate([lb_C, lb_du])
        ub = np.concatenate([ub_C, ub_du])

        # -------------------
        # Residual function
        # -------------------
        def residuals(params):
            times = 0
            # Unpack control points and du
            
            C = params[:C_0.size].reshape(n_ctrl,3)
            du = params[C_0.size:]
            
            # Reconstruct knots
            U_interior = np.cumsum(du)
            U_interior = U_interior / U_interior[-1]  # Normalize to [0,1]

            U = np.concatenate(([0], U_interior, [1]))
            # print("Current knot vector U:", U)

            # Evaluate spline and derivative matrices
            S_fit, Nmat = evaluate_bspline(C, p, U, self.u_vals, return_basis=True)
            _, _, dNmat0 = evaluate_bspline(C, p, U, np.array([0.0]), return_basis=True, return_derivative=True)
            _, _, dNmat1 = evaluate_bspline(C, p, U, np.array([1.0]), return_basis=True, return_derivative=True)

            # Data residual
            res_data = (S_fit - self.S_cart).ravel()

            # Velocity residual (start/end)
            S0_vel = dNmat0[0,:] @ C
            S1_vel = dNmat1[0,:] @ C
            res_vel = vel_penalty * np.concatenate([S0_vel - self.ri_start_velocity,
                                                S1_vel - self.ri_end_velocity])
            return np.concatenate([res_data, res_vel])

        # -------------------
        # Solve least squares
        # -------------------
        res = least_squares(residuals, x0, bounds=(lb, ub), ftol=1e-8, xtol=1e-8, gtol=1e-8, verbose=2)

        # -------------------
        # Extract optimized control points and knots
        # -------------------
        C_opt = res.x[:C_0.size].reshape(n_ctrl,3)
        du_opt = res.x[C_0.size:]
 
        U_interior_opt = np.cumsum(du_opt)
        U_interior_opt = U_interior_opt / U_interior_opt[-1]
        U_opt = np.concatenate(([0], U_interior_opt, [1]))

        # Save
        self.C_cart = C_opt
        self.U = U_opt
        self.p = p
        self.C_sph = np.array([cart2sph(*pt) for pt in C_opt])

        return self.C_cart, self.u_vals, self.U

    # -------------------------------
    # Spline evaluation
    # -------------------------------
    def eval_cartesian_spline(self, u):
        result = evaluate_bspline(self.C_cart, self.p, self.U, u)
        return result

    def eval_spherical_spline(self, u):
        xyz = self.eval_cartesian_spline(u)
        if xyz.ndim == 1:
            return cart2sph(*xyz)
        else:
            return np.array([cart2sph(*pt) for pt in xyz])

    # -------------------------------
    # Plotting
    # -------------------------------
    def plot_spline_fit(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(self.x_cyc, self.y_cyc, self.z_cyc, label="Trajectory", alpha=0.6)
        S_fit = np.vstack([self.eval_cartesian_spline(u) for u in self.u_vals])
        print(S_fit[0,:], S_fit[-1,:])
        ax.plot(S_fit[:,0], S_fit[:,1], S_fit[:,2], "r--", label="B-spline fit")
        if self.C_cart is not None:
            ax.scatter(self.C_cart[:,0], self.C_cart[:,1], self.C_cart[:,2],
                       color="black", s=30, label="Control points")
        # Plot RI start/end
        if self.ri_start_point is not None and self.ri_end_point is not None:
            ax.scatter(*self.ri_start_point, color="green", s=50, label="RI Start")
            ax.scatter(*self.ri_end_point, color="red", s=50, label="RI End")
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        ax.legend(); ax.set_box_aspect([1,1,1])
        plt.show()


if __name__ == "__main__":
# --- File paths ---
    full_df = "/home/theophile/src/Simulation_Results/trial_Uri_valid/ProtoLogger_csv/2025-09-10_11-31-10_ProtoLogger.csv"
    cycle_df = "/home/theophile/src/Simulation_Results/trial_Uri_valid/cycles/cycle_data_sheet_lines.csv"

    # Create a Cycle object for the first cycle (cycle_idx=0)
    cycle = Cycle(full_df, cycle_df, cycle_idx=1)

    # Compute Reel-In boundaries
    ri_start, ri_end, ri_v0, ri_vf, az_RI, el_RI, r_RI, ri_start_idx, ri_end_idx = cycle.get_RI_RO_boundaries()

    print("RI start point (x,y,z):", ri_start)
    print("RI end point (x,y,z):", ri_end)
    print("RI start velocity:", ri_v0)
    print("RI end velocity:", ri_vf)

    # Fit a B-spline to the full cycle (or later to RI only)
    C_cart, u_vals, U = cycle.fit_cartesian_spline()
    print("Control points (cartesian): \n", C_cart)
    print("Knot vector (U): \n", U)

    # Plot trajectory and spline
    cycle.plot_spline_fit()

