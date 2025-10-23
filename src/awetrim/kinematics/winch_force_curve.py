# This file is a copy of the winch force curve computation file from Kitepower's repository
# It has been added to the .gitignore to avoid issues with git tracking
# as it is not meant to be public, it is Kitepower property

import numpy as np
import matplotlib.pyplot as plt


class WinchControllerParameters:
    def __init__(
        self,
        v_cmd=-0.64,
        f_min=440.0,
        f_max=2550.0,
        p_gain_v=-1200.0,
        torque_min=-1700.0,
        torque_max=1700.0,
        p_gain_f=3.0,
        force_knee=2000.0,
        force_slope_factor=0.3,
        plot_comment="",
    ):
        self.v_cmd = v_cmd
        self.f_min = f_min
        self.f_max = f_max
        self.p_gain_v = p_gain_v
        self.torque_min = torque_min
        self.torque_max = torque_max
        self.p_gain_f = p_gain_f
        self.force_knee = force_knee
        self.force_slope_factor = force_slope_factor
        self.plot_comment = plot_comment


class WinchControllerCharacteristics:
    def __init__(self):
        self.v_max = 10.0
        self.v_m_list = np.linspace(-8.0, self.v_max, 1000)

        # self.torque_to_force = 9.99/(0.433*9.81) # GS2 from winch controller sheet
        self.torque_to_force = 13.1 / (0.433 * 9.81)  # GS3 from winch controller sheet
        # self.torque_to_force = 2.0  # empirical factor from dividing measured force with force setpoint.

        # Maximum safe specs limited by max force and motor power limit
        self.force_max = 2750.0
        self.reelspeed_max = 6.5

    @staticmethod
    def get_power_isolines(reelspeed, power_kW):
        return power_kW * 100 / reelspeed

    @staticmethod
    def cap_output(value, f_min, f_max):
        if value < f_min:
            return f_min
        elif value > f_max:
            return f_max
        else:
            return value

    @staticmethod
    def get_v_knee(v_cmd, f_knee, p_v_ctrl):
        # print(v_cmd, f_knee, p_v_ctrl)
        return v_cmd - f_knee / p_v_ctrl

    def get_force_controller_setpoint(
        self, v_m, v_cmd, p_gain_v, f_min, f_max, f_knee=None, force_slope_factor=None
    ):
        if not f_knee:
            f_knee = f_max
        if not force_slope_factor:
            force_slope_factor = 1.0

        v_knee = self.get_v_knee(v_cmd, f_knee, p_gain_v)
        # print(v_knee)
        error = v_cmd - v_m
        output = (
            p_gain_v * error
        )  # This term simulates the normal pid controller (i-gain=d-gain=0).

        if v_m > v_knee:
            output = (f_max - f_knee) * np.tanh(
                (v_m - v_knee) * force_slope_factor
            ) + f_knee

        return self.cap_output(output, f_min, f_max)

    def get_force_controller_setpoint_function(
        self, v_cmd, p_gain_v, f_min, f_max, p_gain_f, f_knee, force_slope_factor
    ):
        return [
            self.get_force_controller_setpoint(
                v_m=v_m,
                v_cmd=v_cmd,
                p_gain_v=p_gain_v,
                f_min=f_min,
                f_max=f_max,
                f_knee=f_knee,
                force_slope_factor=force_slope_factor,
            )
            for v_m in self.v_m_list
        ]

    def get_effective_controller_function(
        self, v_cmd, p_gain_v, f_min, f_max, p_gain_f, f_knee, force_slope_factor
    ):
        conversion_to_actual_force = 1.0 + 1.0 / (p_gain_f * self.torque_to_force)
        return [
            self.get_force_controller_setpoint(
                v_m=v_m,
                v_cmd=v_cmd,
                p_gain_v=p_gain_v,
                f_min=f_min,
                f_max=f_max,
                f_knee=f_knee,
                force_slope_factor=force_slope_factor,
            )
            / conversion_to_actual_force
            for v_m in self.v_m_list
        ]

    def get_controller_setpoint_linear(self, v_m, v_cmd, p_gain_v, f_min, f_max):
        output = p_gain_v * (v_cmd - v_m)
        return self.cap_output(output, f_min, f_max)

    def get_effective_controller_function_linear(
        self, v_cmd, p_gain_v, f_min, f_max, p_gain_f
    ):
        conversion_to_actual_force = 1.0 + 1.0 / (p_gain_f * self.torque_to_force)
        return [
            self.get_controller_setpoint_linear(
                v_m=v_m, v_cmd=v_cmd, p_gain_v=p_gain_v, f_min=f_min, f_max=f_max
            )
            / conversion_to_actual_force
            for v_m in self.v_m_list
        ]

    def plot_controller_characteristics(self, controller_parameters: list):
        # Plot the controller output trace
        power_values = [-25, 25, 50, 100, 150, 200, 250]

        plt.figure()
        for power in power_values:
            plt.plot(
                self.v_m_list,
                self.get_power_isolines(self.v_m_list, power),
                color="black",
                alpha=0.5,
                linewidth=0.5,
            )
            x_value_text = 0
            if power < 0:
                x_value_text = -self.v_max
            elif power >= 0:
                x_value_text = self.v_max
            plt.text(
                x_value_text,
                self.get_power_isolines(x_value_text, power),
                f"{power} kW",
                color="black",
                fontsize=10,
                ha="right",
            )

        for params in controller_parameters:
            controller_linear = self.get_effective_controller_function_linear(
                v_cmd=params.v_cmd,
                p_gain_v=params.p_gain_v,
                p_gain_f=params.p_gain_f,
                f_min=params.f_min,
                f_max=params.f_max,
            )
            plt.plot(self.v_m_list, controller_linear, alpha=0.3, color="grey")

        for params in controller_parameters:
            self.controller = self.get_effective_controller_function(
                v_cmd=params.v_cmd,
                p_gain_v=params.p_gain_v,
                p_gain_f=params.p_gain_f,
                f_min=params.f_min,
                f_max=params.f_max,
                f_knee=params.force_knee,
                force_slope_factor=params.force_slope_factor,
            )
            plt.plot(
                self.v_m_list,
                self.controller,
                label=f"{params.plot_comment}: f_knee={params.force_knee}, b={params.force_slope_factor}, "
                f"f_min={params.f_min}, f_max={params.f_max}, Kp_v={params.p_gain_v}, Kp_f={params.p_gain_f},"
                f" v_cmd={params.v_cmd}",
            )

        plt.xlabel("measured velocity [m/s]")
        plt.ylabel("Force [kgf]")
        plt.ylim(0.0, 3100.0)
        plt.title(f"Winch controller characteristics")
        plt.grid(True)
        plt.legend()

        # Add shading for areas that are too high in force or in reelspeed
        _, ymax = plt.gca().get_ylim()
        _, xmax = plt.gca().get_xlim()

        plt.axvspan(self.reelspeed_max, xmax, color="red", alpha=0.2)
        plt.axhspan(self.force_max, ymax, color="red", alpha=0.2)

    def plot_force_controller_setpoint(self, controller_parameters: list):
        plt.figure()

        for params in controller_parameters:
            controller = self.get_force_controller_setpoint_function(
                v_cmd=params.v_cmd,
                p_gain_v=params.p_gain_v,
                p_gain_f=params.p_gain_f,
                f_min=params.f_min,
                f_max=params.f_max,
                f_knee=params.force_knee,
                force_slope_factor=params.force_slope_factor,
            )
            plt.plot(
                self.v_m_list,
                controller,
                label=f"{params.plot_comment}: f_knee = {params.force_knee}, b={params.force_slope_factor}, "
                f"f_min={params.f_min}, f_max={params.f_max}, Kp_v={params.p_gain_v}, Kp_f={params.p_gain_f},"
                f" v_cmd={params.v_cmd}",
            )

        plt.xlabel("measured velocity [m/s]")
        plt.ylabel("Force controller setpoint")
        # plt.ylim(0.0, 3200.0)
        plt.title(f"Force controller setpoint")
        plt.grid(True)
        plt.legend()

        # Add shading for areas that are too high in force or in reelspeed
        _, ymax = plt.gca().get_ylim()
        _, xmax = plt.gca().get_xlim()

        plt.axvspan(self.reelspeed_max, xmax, color="red", alpha=0.2)
        plt.axhspan(self.force_max, ymax, color="red", alpha=0.2)


def main():
    parameters = [
        WinchControllerParameters(
            f_max=3500.0,
            f_min=440.0,
            force_knee=2500,
            force_slope_factor=0.1,
            p_gain_v=-1500.0,
            p_gain_f=4.0,
            v_cmd=-1.0,
        ),
    ]

    gsctrl = WinchControllerCharacteristics()
    gsctrl.plot_controller_characteristics(parameters)
    # gsctrl.plot_force_controller_setpoint(parameters)

    plt.show()


if __name__ == "__main__":
    main()
