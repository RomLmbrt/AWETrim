import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import casadi as ca
import pickle
from scipy.optimize import least_squares
from picawe.kinematics.my_parametrized_patterns import CasadiSpline as build
from picawe.kinematics.my_RI_data_processing import RI_data_processing
from picawe.kinematics.my_RI_RO_data_processing import RI_RO_data_processing
from picawe.kinematics.my_RO_RI_data_processing import RO_RI_data_processing

class Fitting(build):
    def __init__(self, az_data, el_data, n_ctrl_pts=15):
        # Don't call super().__init__() here since we need custom initialization
        self.n_ctrl = n_ctrl_pts
        self.data_az = az_data
        self.data_el = el_data

        self.initial_params_az = np.linspace(np.min(az_data), np.max(az_data), num=n_ctrl_pts)[1:-1]
        self.initial_params_el = np.linspace(np.min(el_data), np.max(el_data), num=n_ctrl_pts)[1:-1]

        # Fix the shape - should be 1D arrays, not 2D
        self.init_params = np.concatenate([self.initial_params_az, self.initial_params_el])

        self.lower_bounds_az = np.full(n_ctrl_pts-2, -np.pi)
        self.upper_bounds_az = np.full(n_ctrl_pts-2, np.pi)
        self.lower_bounds_el = np.full(n_ctrl_pts-2, -np.pi)
        self.upper_bounds_el = np.full(n_ctrl_pts-2, np.pi)

        self.bounds = (
                np.concatenate([self.lower_bounds_az, self.lower_bounds_el]),
                np.concatenate([self.upper_bounds_az, self.upper_bounds_el])
            )

        self.fitted_params_az, self.fitted_params_el = self.Fit()

    def residuals(self, params):
        # Split the 1D params array back into az and el components
        params_az = params[:self.n_ctrl-2]
        params_el = params[self.n_ctrl-2:]

        obj = build(C_az=np.concatenate(([self.data_az[0]], params_az, [self.data_az[-1]])), C_el=np.concatenate(([self.data_el[0]], params_el, [self.data_el[-1]])))
        u_vals = np.linspace(0, 1, num=len(self.data_az))

        # Use dummy r=1.0 value since CasadiSpline doesn't really use it
        az = obj.azimuth(1.0, u_vals)
        el = obj.elevation(1.0, u_vals)

        res_az = az - self.data_az
        res_el = el - self.data_el

        return np.concatenate([res_az, res_el])
    
    def Fit(self):
        result = least_squares(
            self.residuals,
            self.init_params,
            bounds=self.bounds,
            verbose=0,
            xtol=1e-10,
            ftol=1e-10,
            gtol=1e-10,
        )
        
        # Split results back into az and el components
        fitted_params_az = result.x[:self.n_ctrl-2]
        fitted_params_el = result.x[self.n_ctrl-2:]
        
        print("Fitting completed.")
        return fitted_params_az, fitted_params_el

    def plot_fit(self, title_prefix="", ax=None, show_control_points=True):
        """
        Plot the original data vs fitted spline curves.
        
        Parameters:
        -----------
        title_prefix : str
            Prefix for plot titles (e.g., "RI", "RI_RO", "RO_RI")
        ax : matplotlib axes array, optional
            Array of 2 axes for azimuth and elevation plots. If None, creates new figure.
        show_control_points : bool
            Whether to show control points on the plot
            
        Returns:
        --------
        fig, axes : matplotlib figure and axes
        """
        if ax is None:
            fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        else:
            axes = ax
            fig = axes[0].get_figure()
        
        # Generate u values for plotting
        u_vals = np.linspace(0, 1, len(self.data_az))
        
        # Create fitted spline object
        C_az_full = np.concatenate(([self.data_az[0]], self.fitted_params_az, [self.data_az[-1]]))
        C_el_full = np.concatenate(([self.data_el[0]], self.fitted_params_el, [self.data_el[-1]]))
        fitted_spline = build(C_az=C_az_full, C_el=C_el_full)
        
        # Evaluate fitted spline
        fitted_az = fitted_spline.azimuth(1.0, u_vals)
        fitted_el = fitted_spline.elevation(1.0, u_vals)
        
        # Plot azimuth
        axes[0].plot(u_vals, self.data_az, 'b-', label='Data', linewidth=2)
        axes[0].plot(u_vals, fitted_az, 'r--', label='Fitted', linewidth=2)
        
        if show_control_points:
            u_ctrl_pts = np.linspace(0, 1, len(C_az_full))
            axes[0].scatter(u_ctrl_pts, C_az_full, c='red', s=50, marker='o', 
                          label='Control Points', zorder=5, edgecolors='black')
        
        axes[0].set_title(f'{title_prefix} Azimuth Fit')
        axes[0].set_xlabel('u parameter')
        axes[0].set_ylabel('Azimuth (rad)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot elevation
        axes[1].plot(u_vals, self.data_el, 'b-', label='Data', linewidth=2)
        axes[1].plot(u_vals, fitted_el, 'r--', label='Fitted', linewidth=2)
        
        if show_control_points:
            u_ctrl_pts = np.linspace(0, 1, len(C_el_full))
            axes[1].scatter(u_ctrl_pts, C_el_full, c='red', s=50, marker='o', 
                          label='Control Points', zorder=5, edgecolors='black')
        
        axes[1].set_title(f'{title_prefix} Elevation Fit')
        axes[1].set_xlabel('u parameter')
        axes[1].set_ylabel('Elevation (rad)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig, axes
    
    def save_data(self, segment_name=""):
        fitted_data = {
            'fitted_params_az': self.fitted_params_az,
            'fitted_params_el': self.fitted_params_el,
            'n_ctrl_pts': self.n_ctrl,
            'segment_name': segment_name,
            'original_data_az': self.data_az,
            'original_data_el': self.data_el,
            'full_control_points_az': np.concatenate(([self.data_az[0]], self.fitted_params_az, [self.data_az[-1]])),
            'full_control_points_el': np.concatenate(([self.data_el[0]], self.fitted_params_el, [self.data_el[-1]]))
        }

        # Create filename with segment name
        filename = f"fit_results_{segment_name}.pkl" if segment_name else "fit_results.pkl"
        
        # Save to disk
        with open(filename, "wb") as f:
            pickle.dump(fitted_data, f)
        
        print(f"Data saved to {filename}")

if __name__ == "__main__":
    waypoint_path = "/home/theophile/src/Simulation_Results/trial_Uri_valid_2/waypoints/2025-09-25_11-48-58_ProtoLogger_waypoints.csv"
    full_path = "/home/theophile/src/Simulation_Results/trial_Uri_valid_2/ProtoLogger_csv/2025-09-25_11-48-58_ProtoLogger.csv"
    cycle_path = "/home/theophile/src/Simulation_Results/trial_Uri_valid_2/cycles/cycle_data_sheet_lines.csv"

    # -------Fitting all three segments-------
    # -------Reel In-------
    RI = RI_data_processing(file_path_full=full_path, file_path_cycle=cycle_path, file_path_waypoints=waypoint_path, cyc_idx=0)
    data_RI_az = RI.RI_az
    data_RI_el = RI.RI_el

    print("Fitting RI data...")
    fitting_RI = Fitting(data_RI_az, data_RI_el)
    fitting_RI.save_data("RI")

    # -------Reel In to Reel Out-------
    RIRO = RI_RO_data_processing(file_path_full=full_path, file_path_cycle=cycle_path, file_path_waypoints=waypoint_path, cyc_idx=0)
    data_RIRO_az = RIRO.RI_RO_az
    data_RIRO_el = RIRO.RI_RO_el

    print("\nFitting RI_RO data...")
    fitting_RIRO = Fitting(data_RIRO_az, data_RIRO_el)
    fitting_RIRO.save_data("RI_RO")

    # -------Reel Out to Reel In-------
    RORI = RO_RI_data_processing(file_path_full=full_path, file_path_cycle=cycle_path, file_path_waypoints=waypoint_path, cyc_idx=0)
    data_RORI_az = RORI.RO_RI_az
    data_RORI_el = RORI.RO_RI_el

    print("\nFitting RO_RI data...")
    fitting_RORI = Fitting(data_RORI_az, data_RORI_el)
    fitting_RORI.save_data("RO_RI")

    # -------Plotting using the new method-------
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Plot each segment
    fitting_RI.plot_fit(title_prefix="RI", ax=axes[:, 0])
    fitting_RIRO.plot_fit(title_prefix="RI_RO", ax=axes[:, 1])
    fitting_RORI.plot_fit(title_prefix="RO_RI", ax=axes[:, 2])
    
    plt.tight_layout()
    plt.show()