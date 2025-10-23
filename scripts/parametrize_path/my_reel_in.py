import numpy as np
import json
import matplotlib.pyplot as plt
from awetrim import SystemModel, State
from awetrim.timeseries.phase_parametrized import PhaseParameterized
from awetrim.system.kite import Kite
from awetrim.system.tether import RigidLumpedTether
from awetrim.environment.Wind import Wind
import pickle
from awetrim.utils.color_palette import set_plot_style, get_color_list
from awetrim.utils.defaults import PLOT_LABELS


def run_sim(
    aero_input,
    pattern_config,
    label_prefix,
    mass_wing,
    area_wing,
    mass_kcu,
    tether_diameter,
    depower,
    start_state,
    wind,
    sim_type,
):

    if sim_type == "quasi_steady":
        quasi_steady = True
        inertia_free = False
    elif sim_type == "dynamic":
        quasi_steady = False
        inertia_free = False
    elif sim_type == "inertia_free":
        quasi_steady = True
        inertia_free = True
    elif sim_type == "no_mass":
        quasi_steady = True
        inertia_free = True

    print(f"Running simulation for {sim_type} with label: {label_prefix}")
    tether = RigidLumpedTether(
        diameter=tether_diameter,
    )
    kite = Kite(
        mass_wing=mass_wing,
        mass_kcu=mass_kcu,
        area_wing=area_wing,
        aero_input=aero_input,
        steering_control="asymmetric",
    )
    if inertia_free:
        kite.override_centripetal = True
        kite.override_coriolis = True

    model = SystemModel(
        dof=3, quasi_steady=quasi_steady, kite=kite, tether=tether, wind_model=wind
    )

    model.input_depower = depower
    if sim_type == "no_mass":
        model.mass_wing = 0
        start_state["input_steering"] = 0
    phase = PhaseParameterized(
        model, quasi_steady=quasi_steady, pattern_config=pattern_config
    )
    states = phase.run_simulation_phase(start_state=start_state, return_states=True)

    return phase, states


def main():

    # ---------- Config ----------
    mass_wing = 61
    mass_kcu = 30
    area_wing = 46.85
    tether_diameter = 0.01

    speed_wind_at_100 = (
        7.6374  # m/s (6 m/s at reference height of 6 m) got from KP software for LOG
    )
    wind = Wind(
        wind_model="logarithmic",
        z0=0.0002,
    )
    speed_friction = 0.41 * speed_wind_at_100 / np.log(100 / wind.z0)
    wind.speed_friction = speed_friction
    # wind.speed_wind_ref = speed_wind_at_100

    # color palette available via get_color_list() as needed

    with open("./data/LEI-V9-KITE/v9_aero_input.json", "r") as file:
        aero_input_v9 = json.load(file)

    # ---------- Load precomputed spline fit data ----------
    segment_name = "Single_Spline"  # input("Enter segment name (e.g., 'RI' or 'RI_RO' or 'RO_RI or 'Single_Spline'): ").strip()

    filename = f"fit_results_{segment_name}.pkl"
    with open(filename, "rb") as f:
        fit_data = pickle.load(f)

    r0 = fit_data["r0"]
    r1 = fit_data["r1"]
    C_az = fit_data["C_az"]
    C_el = fit_data["C_el"]
    s_norm_az = fit_data["s_norm_az"]
    s_norm_el = fit_data["s_norm_el"]

    # ---------Load winch and depower data ----------

    if segment_name == "Single_Spline":
        with open("fit_winch_results_Single_Spline_phase_settings.pkl", "rb") as f:
            winch_depower_data = pickle.load(f)

    init_condit_QS_dict = [
        {
            "t": 0,
            "s": 0,
            "s_dot": 0.05,
            "s_ddot": 0,
            "input_steering": 0,
            "tension_tether_ground": 1e8,
            "distance_radial": r0,
            "speed_radial": 0,
            "input_depower": 0,
        }
    ]

    init_condit_Dyn_dict = [
        {
            "t": 0,
            "s": 0,
            "s_dot": 0.05,
            "s_ddot": 0,
            "input_steering": 0,
            "tension_tether_ground": 1e8,
            "distance_radial": r0,
            "speed_radial": 0,
            "input_depower": 0,
        }
    ]

    quasi_steady_s_ends = []
    dynamic_s_ends = []
    globalstatesQS = []
    time = 0
    for phase_idx in range(len(winch_depower_data)):

        f_max = winch_depower_data[phase_idx]["max_tether_force"]
        f_min = winch_depower_data[phase_idx]["min_tether_force"]
        beta_plus = winch_depower_data[phase_idx]["softplus_beta"]
        beta_minus = winch_depower_data[phase_idx]["softminus_beta"]
        slope = winch_depower_data[phase_idx]["slope"]
        offset = winch_depower_data[phase_idx]["offset"]
        s_start = winch_depower_data[phase_idx]["s"]
        depower = winch_depower_data[phase_idx]["depower"]
        if phase_idx == len(winch_depower_data) - 1:
            s_end = 1
        else:
            s_end = winch_depower_data[phase_idx + 1]["s"]
        print(s_start)
        print(s_end)

        depower_norm = (
            (depower / 100) - 0.4
        ) / 0.28  # normalize depower between 0 and 1 for V9

        Realistic_RI_eg = {
            "reeling_strategy": "force",  # "force" or "constant"
            "force_model": "quadratic",  # "linear" or "quadratic"
            "reeling_speed": 0,  # m/s, only for constant reeling
            "max_tether_force": f_max,  # N, only for force reeling
            "min_tether_force": f_min,  # N, only for force reeling
            "softplus": True,
            "softplus_beta": beta_plus,
            "softminus": True,
            "softminus_beta": beta_minus,
            "slope": slope,  # N/(m/s)^2 for quadratic, N/(m/s) for linear
            "offset": offset,  # m/s
        }

        pattern_config = {
            "pattern_type": "spline",
            "path_parameters": {
                "r0": r0,
                "r1": r1,
                "C_az": C_az,
                "C_el": C_el,
                "s_norm_az": s_norm_az,
                "s_norm_el": s_norm_el,
            },
            "radial_parameters": Realistic_RI_eg,
            "start_angle": s_start,
            "end_angle": s_end,
            "n_points": 100,
            "optimization_parameters": [],
        }

        base_start_state_QS = State(**init_condit_QS_dict[0])
        base_start_state_Dyn = State(**init_condit_Dyn_dict[phase_idx])

        phaseQS, stateQS = run_sim(
            aero_input_v9,
            pattern_config,
            "V9",
            mass_wing,
            area_wing,
            mass_kcu,
            tether_diameter,
            depower_norm,
            base_start_state_QS,
            wind,
            "quasi_steady",
        )
        globalstatesQS.append(stateQS)
        base_start_state_Dyn = stateQS[0]

        phaseDyn, stateDyn = run_sim(
            aero_input_v9,
            pattern_config,
            "V9",
            mass_wing,
            area_wing,
            mass_kcu,
            tether_diameter,
            depower_norm,
            base_start_state_Dyn,
            wind,
            "dynamic",
        )

        quasi_steady_s_ends.append(stateQS[-1]["s"])
        dynamic_s_ends.append(stateDyn[-1]["s"])
        time += stateDyn[-1]["t"]
        # init_condit_QS_dict.append(
        #     {
        #         "t": stateQS[-1]["t"],
        #         "s": stateQS[-1]["s"],
        #         "s_dot": stateQS[-1]["s_dot"],
        #         "s_ddot": (
        #             stateQS[-1]["s_ddot"] if stateQS[-1]["s_ddot"] is not None else 0
        #         ),
        #         "input_steering": stateQS[-1]["input_steering"],
        #         "tension_tether_ground": stateQS[-1]["tension_tether_ground"],
        #         "input_depower": (
        #             stateQS[-1]["input_depower"]
        #             if stateQS[-1]["input_depower"] is not None
        #             else 0
        #         ),
        #         "speed_radial": stateQS[-1]["speed_radial"],
        #         "distance_radial": stateQS[-1]["distance_radial"],
        #     }
        # )

        init_condit_Dyn_dict.append(
            {
                "t": stateDyn[-1]["t"],
                "s": stateDyn[-1]["s"],
                "s_dot": stateDyn[-1]["s_dot"],
                "s_ddot": (
                    stateDyn[-1]["s_ddot"] if stateDyn[-1]["s_ddot"] is not None else 0
                ),
                "input_steering": stateDyn[-1]["input_steering"],
                "tension_tether_ground": stateDyn[-1]["tension_tether_ground"],
                "input_depower": (
                    stateDyn[-1]["input_depower"]
                    if stateDyn[-1]["input_depower"] is not None
                    else 0
                ),
                "speed_radial": stateDyn[-1]["speed_radial"],
                "distance_radial": stateDyn[-1]["distance_radial"],
            }
        )

        dynamic_phase = phaseDyn
        qs_phase = phaseQS

        # First series creates the overview figure
        fig, axes_map, scatter = dynamic_phase.plot_overview_3d(
            label="V9 Dynamic",
            color=get_color_list()[2],
            linestyle="-",
            variables=[
                "speed_tangential",
                "tension_tether_ground",
                "lift_coefficient",
                "speed_radial",
            ],
            x_param="t",
        )

        # Second series overlays on the same axes
        qs_phase.plot_overview_3d(
            label="V9 Quasi-Steady",
            color=get_color_list()[1],
            linestyle="--",
            variables=[
                "speed_tangential",
                "tension_tether_ground",
                "lift_coefficient",
                "speed_radial",
            ],
            x_param="t",
            axes=axes_map,
        )

        fig.legend(loc="upper center", bbox_to_anchor=(0.5, 0.95), ncol=2)
        set_plot_style()
        plt.tight_layout()
        # # Save the figure as pdf
        # plt.savefig("./results/figures/reelout_cst.pdf", bbox_inches="tight")
        plt.show()

        metrics = dynamic_phase.energy_metrics(qs_phase)
        print("\n--- V9 ---")
        print(
            f"Power QS: {metrics['avg_power_other']:.2f}, Power Dyn: {metrics['avg_power_self']:.2f}."
        )
        print(
            f"Mean power QS: {metrics['mean_power_other']:.2f}, Mean power Dyn: {metrics['mean_power_self']:.2f}"
        )
        print(f"Δ Power: {metrics['power_diff_percent']:.2f}%")
        print(f"Estimated time lag: {metrics['best_time_lag']:.3f} s")
        print(f"ΔF_t,mean: {metrics['delta_ft_mean_percent']:.2f}%")
        print(f"ΔF_t,max: {metrics['delta_ft_max_percent']:.2f}%")
        print(f"ΔF_t,min: {metrics['delta_ft_min_percent']:.2f}%")
        print(f"Δv_tau,max: {metrics['delta_vtau_max_percent']:.2f}%")
        print(f"Δv_tau,min: {metrics['delta_vtau_min_percent']:.2f}%")
        print(f"Δs_v_tau,max: {metrics['s_lag_vtau_max_deg']:.2f} deg")
        print(f"Δs_v_tau,min: {metrics['s_lag_vtau_min_deg']:.2f} deg")
        plt.show()
    print("Total time:", time)
    if globalstatesQS:
        set_plot_style()
        combined_time = []
        tension_series = []
        tangential_series = []
        radial_series = []
        time_offset = 0.0

        for phase_states in globalstatesQS:
            if not phase_states:
                continue
            phase_start = phase_states[0]["t"]
            phase_end = phase_states[-1]["t"]
            for state in phase_states:
                current_time = time_offset + (state["t"] - phase_start)
                combined_time.append(current_time)
                tension_series.append(state.get("tension_tether_ground"))
                tangential_series.append(state.get("speed_tangential"))
                radial_series.append(state.get("speed_radial"))
            time_offset += phase_end - phase_start

        fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 8))
        plot_config = [
            ("tension_tether_ground", tension_series, "Tether tension"),
            ("speed_tangential", tangential_series, "Tangential speed"),
            ("speed_radial", radial_series, "Radial speed"),
        ]

        for ax, (key, series, fallback_label) in zip(axes, plot_config):
            label = PLOT_LABELS.get(key, fallback_label)
            ax.plot(combined_time, series, label=label)
            ax.set_ylabel(label)
            ax.grid(True, linestyle="--", alpha=0.3)
            ax.legend(loc="best")

        axes[-1].set_xlabel("Time [s]")
        plt.tight_layout()
        plt.show()

    print("Quasi-steady end s values for each phase:", quasi_steady_s_ends)

    return (
        init_condit_QS_dict[-1],
        init_condit_Dyn_dict[-1],
    )  # Return the last state as initial condition for further use in the reelout_Lissajous segment


init_conditions_QS, init_conditions_Dyn = main()
