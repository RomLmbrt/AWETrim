import json
import pickle
import csv

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from awetrim import SystemModel, State
from awetrim.environment.Wind import Wind
from awetrim.kinematics.find_Lissajous_RO_start_end_angles import (
    find_Lissajous_RO_start_end_angles,
)
from awetrim.system.kite import Kite
from awetrim.system.tether import RigidLumpedTether
from awetrim.timeseries.phase_parametrized import PhaseParameterized
from awetrim.utils.color_palette import set_plot_style, get_color_list
from my_reel_in import init_conditions_QS as Single_Spline_final_state_QS
from my_reel_in import init_conditions_Dyn as Single_Spline_final_state_Dyn
from awetrim.utils.defaults import PLOT_LABELS


def define_system(
    tether_diameter,
    mass_wing,
    mass_kcu,
    area_wing,
    aero_input,
    wind_model,
):
    """Instantiate a SystemModel with the supplied components."""

    tether = RigidLumpedTether(diameter=tether_diameter)
    kite = Kite(
        mass_wing=mass_wing,
        mass_kcu=mass_kcu,
        area_wing=area_wing,
        aero_input=aero_input,
        steering_control="asymmetric",
    )

    model = SystemModel(
        dof=3,
        kite=kite,
        tether=tether,
        wind_model=wind_model,
    )
    return model


def run_sim(
    pattern_config,
    label_prefix,
    depower,
    start_state,
    model,
    quasi_steady,
):
    """Run a parametrized phase simulation and return the populated PhaseParameterized object."""

    sim_type = "quasi steady" if quasi_steady else "dynamic"
    print(f"Running simulation for {sim_type} with label: {label_prefix}")

    model.input_depower = depower

    phase = PhaseParameterized(
        model,
        quasi_steady=quasi_steady,
        pattern_config=pattern_config,
    )
    phase.run_simulation_phase(start_state=start_state, return_states=True)
    return phase


def main():
    plot_variables = [
        "speed_radial",
        "speed_tangential",
        "tension_tether_ground",
        "lift_coefficient",
        "drag_coefficient",
        "mechanical_power",
    ]
    variables_to_save = plot_variables + [
        "distance_radial",
        "angle_elevation",
        "angle_azimuth",
    ]
    derived_variables = ["x_position", "y_position", "z_position"]

    # ---------- Config ----------
    mass_wing = 61
    mass_kcu = 30
    area_wing = 46.85
    tether_diameter = 0.01

    speed_wind_at_100 = 7.6374  # m/s (6 m/s at reference height of 6 m)
    wind_model = Wind(
        wind_model="logarithmic",
        z0=0.0002,
    )
    speed_friction = 0.41 * speed_wind_at_100 / np.log(100 / wind_model.z0)
    wind_model.speed_friction = speed_friction

    with open("./data/LEI-V9-KITE/v9_aero_input.json", "r") as file:
        aero_input_v9 = json.load(file)

    # ---------- Load precomputed Lissajous fit data ----------
    segment_name = "LISSAJOUS"
    filename = f"fit_results_{segment_name}.pkl"
    with open(filename, "rb") as f:
        fit_data = pickle.load(f)

    r0 = fit_data["r0"]
    duration = fit_data["duration"]
    az_amp0 = fit_data["best_params"]["az_amp0"]
    beta_amp0 = fit_data["best_params"]["beta_amp0"]
    beta_coeffs = fit_data["best_params"]["beta_coeffs"]
    az_coeffs = fit_data["best_params"]["az_coeffs"]
    beta0 = fit_data["best_params"]["beta0"]

    pattern_type = "cst_lissajous"
    parameters = {
        "omega": 1.0,
        "r0": r0,
        "az_amp0": az_amp0,
        "beta_amp0": beta_amp0,
        "width_phi": 0.5,
        "width_beta": 0.5,
        "left_first": True,
        "normalize_bumps": False,
        "repeat_phi": True,
        "repeat_beta": True,
        "beta_coeffs": np.array(beta_coeffs),
        "az_coeffs": az_coeffs,
        "kbeta": 0,
        "beta0": beta0,
        "kappa": 0,
    }

    s_start_opt, range_opt, cycles = find_Lissajous_RO_start_end_angles(
        pattern_type, parameters
    )

    # --------- Load winch and depower data ----------
    with open("fit_winch_results_RO_phase_settings.pkl", "rb") as f:
        winch_depower_data = pickle.load(f)

    f_max = winch_depower_data[0]["max_tether_force"]
    f_min = winch_depower_data[0]["min_tether_force"]
    beta_plus = winch_depower_data[0]["softplus_beta"]
    beta_minus = winch_depower_data[0]["softminus_beta"]
    slope = winch_depower_data[0]["slope"]
    offset = winch_depower_data[0]["offset"]
    depower = winch_depower_data[0]["depower"]

    depower_norm = (
        (depower / 100) - 0.4
    ) / 0.28  # normalize depower between 0 and 1 for V9

    Realistic_RO_eg = {
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
        "pattern_type": pattern_type,
        "path_parameters": parameters,
        "radial_parameters": Realistic_RO_eg,
        "start_time": 0,
        "end_time": duration + 1,
        "start_angle": s_start_opt,
        "end_angle": s_start_opt + range_opt + cycles * (2 * np.pi),
        "n_points": 500,
        "optimization_parameters": [],
    }

    # ---------- Starting state ----------
    Single_Spline_final_state_QS["s"] = s_start_opt
    Single_Spline_final_state_QS["s_dot"] = 2
    Single_Spline_final_state_QS["tension_tether_ground"] = 1e8

    print("\nInitial quasi-steady state:")
    for key, value in Single_Spline_final_state_QS.items():
        print(f"{key}: {value}")
    print()

    base_start_state_QS = State(**Single_Spline_final_state_QS)
    base_start_state_Dyn = State(**Single_Spline_final_state_Dyn)

    system_model_qs = define_system(
        tether_diameter,
        mass_wing,
        mass_kcu,
        area_wing,
        aero_input_v9,
        wind_model,
    )
    phaseQS = run_sim(
        pattern_config,
        "V9",
        depower_norm,
        base_start_state_QS,
        system_model_qs,
        quasi_steady=True,
    )

    # Use the quasi-steady initial state as the baseline for the dynamic run
    if phaseQS.states:
        base_start_state_Dyn = phaseQS.states[0]

    system_model_dyn = define_system(
        tether_diameter,
        mass_wing,
        mass_kcu,
        area_wing,
        aero_input_v9,
        wind_model,
    )
    phaseDyn = run_sim(
        pattern_config,
        "V9",
        depower_norm,
        base_start_state_Dyn,
        system_model_dyn,
        quasi_steady=False,
    )

    qs_tension = [state["tension_tether_ground"] for state in phaseQS.states]
    dyn_tension = [state["tension_tether_ground"] for state in phaseDyn.states]

    # plt.figure()
    # plt.plot(qs_tension, label="Quasi-Steady")
    # plt.plot(dyn_tension, label="Dynamic")
    # plt.legend()
    # plt.show()

    dynamic_phase = phaseDyn
    qs_phase = phaseQS

    dynamic_phase = phaseDyn
    qs_phase = phaseQS
    qs_series = {"t": qs_phase.return_variable("t")}
    dyn_series = {"t": dynamic_phase.return_variable("t")}
    for var_name in variables_to_save:
        qs_series[var_name] = qs_phase.return_variable(var_name)
        dyn_series[var_name] = dynamic_phase.return_variable(var_name)
    for var_name in derived_variables:
        qs_series[var_name] = []
        dyn_series[var_name] = []

    # fig, axes_map, scatter = dynamic_phase.plot_overview_3d(
    #     label="V9 Dynamic",
    #     color=get_color_list()[2],
    #     linestyle="-",
    #     variables=[
    #         "speed_tangential",
    #         "tension_tether_ground",
    #         "input_steering",
    #         "speed_radial",
    #         "mechanical_power",
    #         "lift_coefficient",
    #         "drag_coefficient",
    #     ],
    #     x_param="t",
    # )

    # qs_phase.plot_overview_3d(
    #     label="V9 Quasi-Steady",
    #     color=get_color_list()[1],
    #     linestyle="--",
    #     variables=[
    #         "speed_tangential",
    #         "tension_tether_ground",
    #         "input_steering",
    #         "speed_radial",
    #         "mechanical_power",
    #         "lift_coefficient",
    #         "drag_coefficient",
    #     ],
    #     x_param="t",
    #     axes=axes_map,
    # )

    # fig.legend(loc="upper center", bbox_to_anchor=(0.5, 0.95), ncol=2)
    # set_plot_style()
    # plt.tight_layout()
    # # plt.savefig("./results/figures/reelout_cst.pdf", bbox_inches="tight")
    # plt.show()

    # metrics = dynamic_phase.energy_metrics(qs_phase)
    # print("\n--- V9 ---")
    # print(
    #     f"Power QS: {metrics['avg_power_other']:.2f}, Power Dyn: {metrics['avg_power_self']:.2f}."
    # )
    # print(
    #     f"Mean power QS: {metrics['mean_power_other']:.2f}, Mean power Dyn: {metrics['mean_power_self']:.2f}"
    # )
    # print(f"Delta Power: {metrics['power_diff_percent']:.2f}%")
    # print(f"Estimated time lag: {metrics['best_time_lag']:.3f} s")
    # print(f"Delta F_t,mean: {metrics['delta_ft_mean_percent']:.2f}%")
    # print(f"Delta F_t,max: {metrics['delta_ft_max_percent']:.2f}%")
    # print(f"Delta F_t,min: {metrics['delta_ft_min_percent']:.2f}%")
    # print(f"Delta v_tau,max: {metrics['delta_vtau_max_percent']:.2f}%")
    # print(f"Delta v_tau,min: {metrics['delta_vtau_min_percent']:.2f}%")
    # print(f"Delta s_v_tau,max: {metrics['s_lag_vtau_max_deg']:.2f} deg")
    # print(f"Delta s_v_tau,min: {metrics['s_lag_vtau_min_deg']:.2f} deg")
    # plt.show()

    
    aggregated_data = {
        "quasi_steady": qs_series,
        "dynamic": dyn_series
        }
    

    set_plot_style()
    fig, axes = plt.subplots(
        len(plot_variables),
        1,
        sharex=True,
        figsize=(10, 3 * len(plot_variables)),
    )
    axes = np.atleast_1d(axes)
    for idx, var_name in enumerate(plot_variables):
        ax = axes[idx]
        ylabel = PLOT_LABELS.get(var_name, var_name)
        for sim_key, sim_label in [
            ("quasi_steady", "Quasi-Steady"),
            ("dynamic", "Dynamic"),
        ]:
            times = aggregated_data[sim_key]["t"]
            values = aggregated_data[sim_key][var_name]
            ax.plot(times, values, label=sim_label)
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", alpha=0.3)
    axes[-1].set_xlabel("Time [s]")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(loc="best")
    plt.tight_layout()
    plt.show()

    fig3d = plt.figure(figsize=(8, 6))
    ax3d = fig3d.add_subplot(111, projection="3d")
    plotted_any = False
    for sim_key, sim_label in [
        ("quasi_steady", "Quasi-Steady"),
        ("dynamic", "Dynamic"),
    ]:
        times = aggregated_data[sim_key]["t"]

        r_vals = np.asarray(aggregated_data[sim_key].get("distance_radial", []), dtype=float)
        beta_vals = np.asarray(aggregated_data[sim_key].get("angle_elevation", []), dtype=float)
        phi_vals = np.asarray(aggregated_data[sim_key].get("angle_azimuth", []), dtype=float)
        if (
            r_vals.size == times.size
            and beta_vals.size == times.size
            and phi_vals.size == times.size
        ):
            x_vals = r_vals * np.cos(beta_vals) * np.cos(phi_vals)
            y_vals = r_vals * np.cos(beta_vals) * np.sin(phi_vals)
            z_vals = r_vals * np.sin(beta_vals)
        else:
            x_vals = np.full(times.shape, np.nan, dtype=float)
            y_vals = np.full(times.shape, np.nan, dtype=float)
            z_vals = np.full(times.shape, np.nan, dtype=float)

        aggregated_data[sim_key]["x_position"].extend(x_vals.tolist())
        aggregated_data[sim_key]["y_position"].extend(y_vals.tolist())
        aggregated_data[sim_key]["z_position"].extend(z_vals.tolist())

        x_vals = np.asarray(aggregated_data[sim_key]["x_position"], dtype=float)
        y_vals = np.asarray(aggregated_data[sim_key]["y_position"], dtype=float)
        z_vals = np.asarray(aggregated_data[sim_key]["z_position"], dtype=float)
        finite_mask = (
            np.isfinite(x_vals) & np.isfinite(y_vals) & np.isfinite(z_vals)
        )
        if finite_mask.any():
            ax3d.plot(
                x_vals[finite_mask],
                y_vals[finite_mask],
                z_vals[finite_mask],
                label=sim_label,
            )
            plotted_any = True
    if plotted_any:
        ax3d.set_xlabel(PLOT_LABELS.get("x", "x"))
        ax3d.set_ylabel(PLOT_LABELS.get("y", "y"))
        ax3d.set_zlabel(PLOT_LABELS.get("z", "z"))
        x_combined = []
        y_combined = []
        z_combined = []
        for sim_key in ["quasi_steady", "dynamic"]:
            x_arr = np.asarray(aggregated_data[sim_key]["x_position"], dtype=float)
            y_arr = np.asarray(aggregated_data[sim_key]["y_position"], dtype=float)
            z_arr = np.asarray(aggregated_data[sim_key]["z_position"], dtype=float)
            finite = np.isfinite(x_arr) & np.isfinite(y_arr) & np.isfinite(z_arr)
            if finite.any():
                x_combined.append(x_arr[finite])
                y_combined.append(y_arr[finite])
                z_combined.append(z_arr[finite])
        if x_combined:
            x_all = np.concatenate(x_combined)
            y_all = np.concatenate(y_combined)
            z_all = np.concatenate(z_combined)
            ranges = np.array([np.ptp(x_all), np.ptp(y_all), np.ptp(z_all)])
            overall = np.nanmax(ranges) if ranges.size else 0.0
            if overall > 0:
                mid_x = 0.5 * (np.nanmax(x_all) + np.nanmin(x_all))
                mid_y = 0.5 * (np.nanmax(y_all) + np.nanmin(y_all))
                mid_z = 0.5 * (np.nanmax(z_all) + np.nanmin(z_all))
                half = overall / 2.0
                ax3d.set_xlim(mid_x - half, mid_x + half)
                ax3d.set_ylim(mid_y - half, mid_y + half)
                ax3d.set_zlim(mid_z - half, mid_z + half)
                ax3d.set_box_aspect([1, 1, 1])
        ax3d.legend(loc="best")
        plt.tight_layout()
        plt.show()
    else:
        plt.close(fig3d)

    output_dir = Path("results/timeseries")
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "aggregated_timeseries.csv"
    header = ["simulation", "time"] + variables_to_save + derived_variables
    with csv_path.open("w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        for sim_key, sim_label in [
            ("quasi_steady", "quasi_steady"),
            ("dynamic", "dynamic"),
        ]:
            times = aggregated_data[sim_key]["t"]
            if times is None:
                continue
            for idx in range(len(times)):
                row = [sim_label, times[idx]]
                row.extend(
                    aggregated_data[sim_key][var][idx] for var in variables_to_save
                )
                row.extend(
                    aggregated_data[sim_key][var][idx] for var in derived_variables
                )
                writer.writerow(row)
    print(f"Saved aggregated timeseries to {csv_path}")

    total_qs_time = aggregated_data["quasi_steady"]["t"][-1] if len(aggregated_data["quasi_steady"]["t"]) > 0 else 0.0

    total_dyn_time = aggregated_data["dynamic"]["t"][-1] if len(aggregated_data["dynamic"]["t"]) > 0 else 0.0

    time = max(total_qs_time, total_dyn_time)
    if total_qs_time is not None:
        print(f"Total quasi-steady time: {total_qs_time:.3f} s")
    if total_dyn_time is not None:
        print(f"Total dynamic time: {total_dyn_time:.3f} s")
    print("Total time:", time)

    return phaseQS, phaseDyn, aggregated_data


phaseQS, phaseDyn, aggregated_data_RO = main()
