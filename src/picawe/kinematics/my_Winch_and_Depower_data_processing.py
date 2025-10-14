import numpy as np
from picawe.kinematics.my_data_processing import DataProcessing

class Winch_and_Depower_data_processing(DataProcessing):
    def __init__(self, file_path_full, file_path_cycle, file_path_waypoints, cyc_idx=0):
        super().__init__(file_path_full, file_path_cycle, file_path_waypoints, cyc_idx)

        # RIRO means reel in to reel out (Leftover before Lissajous)
        # RORI means reel out to reel in (Leftover after Lissajous)
        # RI means reel in (csv RI + RIRO + RORI)
        # RO means reel out (csv RO)

        # Start of RIRO:
        self.RIRO_idx0 = self.RI_RO_idx0

        # Start of reel out:
        self.RO_idx0 = self.Lissajous_idx0

        # Start of RORI:
        self.RORI_idx0 = self.RO_RI_idx0

        # Start of reel in:
        self.RI_idx0 = self.RI_idx0

        print(f"\nStarting indices: \n RIRO: {self.RIRO_idx0} \n RO: {self.RO_idx0} \n RORI: {self.RORI_idx0} \n RI: {self.RI_idx0} \n")
        print(f"Length of cycle: {len(self.az_cyc)} \n")

        # Waypoint data
        self._find_cycle_wp_limits()
        self._extrapolate_wp_names()
    
    # -------------------------
    # Waypoint inter/extrapolation
    # -------------------------

    def _find_cycle_wp_limits(self):

        self.wp0_idx = None
        self.wpf_idx = None
        self.wp0 = None
        self.wpf = None

        # print(self.time_waypoints, "\n")
        print(self.time_cyc[0], "\n", self.time_cyc[-1], "\n")

        for i, t in enumerate(self.time_waypoints):
            if t > self.time_cyc[0] and self.wp0 is None:
                print("Got the start!", t)
                self.wp0_idx = i-1
                self.wp0 = self.wp_names[self.wp0_idx]
            elif t >= self.time_cyc[-1] and self.wpf is None:
                print("got the end!", t)
                self.wpf_idx = i
                self.wpf = self.wp_names[self.wpf_idx]
                break

        print(f"Cycle starts at waypoint {self.wp0} (idx {self.wp0_idx}) and ends at waypoint {self.wpf} (idx {self.wpf_idx})")
        
    def _extrapolate_wp_names(self):

        self.cyc_switch_idx = []
        self.extrapolated_wp_names = []

        current_idx = self.wp0_idx

        for t in self.time_waypoints[self.wp0_idx+1:self.wpf_idx+1]:
            for i, t_cyc in enumerate(self.time_cyc):
                if t_cyc < t:
                    self.extrapolated_wp_names.append(self.wp_names[current_idx])
                else:
                    self.cyc_switch_idx.append(i)
                    current_idx += 1
                    break

        # print(self.extrapolated_wp_names)

        
        

if __name__ == "__main__":
    waypoint_path = "/home/theophile/src/Simulation_Results/trial_Uri_valid_2/waypoints/2025-09-25_11-48-58_ProtoLogger_waypoints.csv"
    full_path = "/home/theophile/src/Simulation_Results/trial_Uri_valid_2/ProtoLogger_csv/2025-09-25_11-48-58_ProtoLogger.csv"
    cycle_path = "/home/theophile/src/Simulation_Results/trial_Uri_valid_2/cycles/cycle_data_sheet_lines.csv"
    obj = Winch_and_Depower_data_processing(file_path_full=full_path, file_path_cycle=cycle_path, file_path_waypoints=waypoint_path, cyc_idx=0)