from picawe.kinematics.parametrized_patterns import CST_Lissajous
from picawe.kinematics.ReelInBspline_data_processing import ReelInBspline_data_processing as ribdata
import numpy as np
import casadi as cs
import matplotlib.pyplot as plt

class ReelOutLoopDefinition(ribdata):
    def __init__(self, file_path_cycle=None, file_path_full=None, cyc_idx=0):
        super().__init__(file_path_cycle=file_path_cycle, file_path_full=file_path_full, cyc_idx=cyc_idx)

    def plot_reel_out_path(self):
        self.az_RO = self.az_cyc[:self.ri_idx0]
        self.el_RO = self.el_cyc[:self.ri_idx0]

        plt.figure()
        plt.plot(self.az_RO, self.el_RO)
        plt.xlabel('Azimuth (rad)')
        plt.ylabel('Elevation (rad)')
        plt.title('Kite Path During Reel-Out Phase')
        plt.show()

    def ID_lisajous_start_end(self):
        # Truncate the RO data to find the CST Lissajous pattern parameters
        pass

if __name__ == "__main__":
    full_path = "/home/theophile/src/Simulation_Results/trial_Uri_valid_2/ProtoLogger_csv/2025-09-25_11-48-58_ProtoLogger.csv"
    cycle_path = "/home/theophile/src/Simulation_Results/trial_Uri_valid_2/cycles/cycle_data_sheet_lines.csv"

    obj = ReelOutLoopDefinition(file_path_cycle=cycle_path, file_path_full=full_path, cyc_idx=0)
    obj.plot_reel_out_path()























class LSQ_reel_out(ReelOutLoopDefinition):
    def __init__(self):
        super().__init__()

