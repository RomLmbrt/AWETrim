import re
import numpy as np
import pandas as pd
import casadi as ca
import matplotlib.pyplot as plt

# Helpers

def convert_time_to_seconds(time_array):
    """Convert time in HH:MM:SS.sss format to total seconds."""
    seconds_array = []
    for time_str in time_array:
        parts = re.split(r"[:.]", str(time_str))
        h = int(parts[0])
        m = int(parts[1])
        s = float(parts[2]) + float("0." + parts[3]) if len(parts) > 3 else float(parts[2])
        seconds = h * 3600 + m * 60 + s
        seconds_array.append(seconds)
    return np.array(seconds_array)

# Numeric B-spline basis (numpy) using Cox-de Boor recursion
def bspline_basis_matrix(u_vals, p, U, n_ctrl):
    u_vals = np.asarray(u_vals)
    m = len(U) - 1
    N = np.zeros((len(u_vals), n_ctrl))
    for i in range(n_ctrl):
        left = U[i]
        right = U[i+1]
        if left <= right:
            N[:, i] = np.where((u_vals >= left) & (u_vals < right), 1.0, 0.0)
    N[np.isclose(u_vals, 1.0), -1] = 1.0
    for k in range(1, p+1):
        N_k = np.zeros_like(N)
        for i in range(n_ctrl):
            left_den = U[i+k] - U[i]
            right_den = U[i+k+1] - U[i+1]
            left = np.zeros(len(u_vals))
            right = np.zeros(len(u_vals))
            if left_den != 0:
                left = (u_vals - U[i]) / left_den * N[:, i]
            if right_den != 0:
                right = (U[i+k+1] - u_vals) / right_den * N[:, i+1] if i+1 < n_ctrl else 0
            N_k[:, i] = left + right
        N = N_k
    return N

p = 3
n_ctrl = 7
U = [0.0,0.0,0.0,0.0, 1.0/4.0, 2.0/4.0, 3.0/4.0, 1.0,1.0,1.0,1.0]

class Cycle:
    def __init__(self, full_df, cycle_df, cycle_idx=0):
        self.full_df = full_df.copy()
        self.cycle_df = cycle_df.copy()
        self.cycle_idx = cycle_idx
        self._prepare_time_and_columns()
        self._compute_useful_cycles_idx()
        self.x, self.y, self.z, self.time = self.get_cycle_cartesian(cycle_idx)
        self.dx, self.dy, self.dz = np.gradient(self.x), np.gradient(self.y), np.gradient(self.z)
        self.gradient = np.column_stack((self.dx, self.dy, self.dz))
        self.p0 = None
        self.v0 = None
        self.pf = None
        self.vf = None
        self.c2 = None
        self.c3 = None
        self.c4 = None
        self.C_mat = None
        self.S_fitted_sph = None
        self.S_fitted_cart = None

    @staticmethod
    def from_files(file_path_full, file_path_cycle, cycle_idx=0):
        full_data = pd.read_csv(file_path_full, header=0, sep=r"\s+")
        cycle_data = pd.read_csv(file_path_cycle, header=0)
        return Cycle(full_data, cycle_data, cycle_idx)

    def _prepare_time_and_columns(self):
        self.full_df['time_s'] = np.round(convert_time_to_seconds(self.full_df['time_of_day'].to_numpy()), 1)
        self.cycle_df['start_time_s'] = np.round(convert_time_to_seconds(self.cycle_df['start_time_cycle_LT'].to_numpy()), 1)
        self.az = self.full_df['kite_azimuth'].to_numpy()  # already in radians
        self.el = self.full_df['kite_elevation'].to_numpy()  # already in radians
        self.r = self.full_df['kite_distance'].to_numpy()
        self.time_full = self.full_df['time_s'].to_numpy()
        self.time_cycle = self.cycle_df['start_time_s'].to_numpy()

    def _compute_useful_cycles_idx(self):
        idx_start_cycle = np.array([i for i, t in enumerate(self.time_full) for j, tc in enumerate(self.time_cycle) if t == tc])
        if len(idx_start_cycle) < 3:
            raise ValueError('Not enough cycle indices found in full and cycle files')
        self.useful_cycles_idx = idx_start_cycle[1:-1]

    def sph2cart(self):
        """Convert stored spherical trajectory to Cartesian coordinates (radians assumed)."""
        x = self.r * np.cos(self.el) * np.cos(self.az)
        y = self.r * np.cos(self.el) * np.sin(self.az)
        z = self.r * np.sin(self.el)
        return x, y, z

    def sph2cart_cycle(self, az, el, r):
        x = r * np.cos(el) * np.cos(az)
        y = r * np.cos(el) * np.sin(az)
        z = r * np.sin(el)
        return x, y, z

    def get_cycle_cartesian(self, cycle_idx):
        start_idx = self.useful_cycles_idx[cycle_idx]
        end_idx = self.useful_cycles_idx[cycle_idx + 1]
        az_cyc = self.az[start_idx:end_idx]
        el_cyc = self.el[start_idx:end_idx]
        r_cyc = self.r[start_idx:end_idx]
        time_cyc = self.time_full[start_idx:end_idx] - self.time_full[start_idx]
        x, y, z = self.sph2cart_cycle(az_cyc, el_cyc, r_cyc)
        return np.array(x), np.array(y), np.array(z), time_cyc

    def find_true_RI_start(self, RI_start_est):
        distances = (self.x - RI_start_est[0])**2 + (self.y - RI_start_est[1])**2 + (self.z - RI_start_est[2])**2
        idx = np.argmin(distances)
        return idx, (self.x[idx], self.y[idx], self.z[idx])

    def find_true_RI_end(self):
        start_point = (self.x[0], self.y[0], self.z[0])
        end_point = (self.x[-1], self.y[-1], self.z[-1])
        RI_end_true = (np.mean([start_point[0], end_point[0]]),
                       np.mean([start_point[1], end_point[1]]),
                       np.mean([start_point[2], end_point[2]]))
        RI_end_gradient = (np.mean([self.dx[0], self.dx[-1]]),
                           np.mean([self.dy[0], self.dy[-1]]),
                           np.mean([self.dz[0], self.dz[-1]]))
        return RI_end_true, RI_end_gradient

    def fit_spline_least_squares(self, T, n_samples=100):
        s_vals = np.linspace(0, T, n_samples)
        u_vals = s_vals / T
        # Setup placeholder control points
        self.c2 = np.zeros(3)
        self.c3 = np.zeros(3)
        self.c4 = np.zeros(3)
        # Compute B-spline basis matrix (numeric)
        N = bspline_basis_matrix(u_vals, p, U, n_ctrl)
        # Solve least squares for each coordinate (x, y, z) in spherical coords
        S_sph = np.vstack([self.az[self.useful_cycles_idx[self.cycle_idx]:self.useful_cycles_idx[self.cycle_idx+1]],
                           self.el[self.useful_cycles_idx[self.cycle_idx]:self.useful_cycles_idx[self.cycle_idx+1]],
                           self.r[self.useful_cycles_idx[self.cycle_idx]:self.useful_cycles_idx[self.cycle_idx+1]]]).T
        # For simplicity, fit only the unknown control points c2,c3,c4, using N[:,2:5]
        C_unknown, residuals, rank, s = np.linalg.lstsq(N[:,2:5], S_sph, rcond=None)
        self.c2, self.c3, self.c4 = C_unknown.T
        # Store fitted spline in spherical and Cartesian coords
        self.S_fitted_sph = N @ np.column_stack((S_sph[0], S_sph[0], S_sph[0]))  # placeholder
        self.S_fitted_cart = np.array([self.sph2cart_cycle(*sph) for sph in self.S_fitted_sph])
        return self.S_fitted_sph, self.S_fitted_cart

    def plot_cycle(self, RI_start=None, RI_end=None):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(self.x, self.y, self.z, label=f'Cycle {self.cycle_idx+1} Trajectory')
        if RI_start is not None:
            ax.scatter(*RI_start, color='red', label='Reel-In Start Point', s=25)
        if RI_end is not None:
            ax.scatter(*RI_end, color='green', label='Reel-In End Point', s=25)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(f'Kite Trajectory for Cycle {self.cycle_idx+1}')
        ax.legend()
        ax.set_box_aspect([1,1,1])
        plt.show()

    def performance_metrics(self):
        pass



if __name__ == "__main__":
# --- File paths ---
    file_path_full = "/home/theophile/src/Simulation_Results/trial_Uri_valid/ProtoLogger_csv/2025-09-10_11-31-10_ProtoLogger.csv"
    file_path_cycle = "/home/theophile/src/Simulation_Results/trial_Uri_valid/cycles/cycle_data_sheet_lines.csv"


    # --- Create Cycle object ---
    cycle_idx = 0 # first useful cycle
    cycle = Cycle.from_files(file_path_full, file_path_cycle, cycle_idx)


    # --- Find Reel-In start and end ---
    RI_start_est = (260, -130, 140) # placeholder estimate
    idx_start, p0 = cycle.find_true_RI_start(RI_start_est)
    pf, vf = cycle.find_true_RI_end()


    # --- Plot the cycle ---
    cycle.plot_cycle(RI_start=p0, RI_end=pf)


    # --- Fit B-spline in spherical coordinates ---
    T = 10.0 # total normalized time for spline
    S_sph, S_cart = cycle.fit_spline_least_squares(T)


    # --- Display performance metrics ---
    metrics = cycle.performance_metrics()
    print("Cycle performance metrics:", metrics)