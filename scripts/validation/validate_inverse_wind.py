"""
Inverse solver: Given a prescribed tether force, solve for the wind speed.
Uses measured wind speed as initial condition.
"""

import h5py
import pandas as pd
import numpy as np
from awetrim import SystemModel
from awetrim.system.kite import Kite
from awetrim.system.tether import RigidLumpedTether
from awetrim.environment.Wind import Wind
import casadi as ca
import time
import json


def read_results(year, month, day, kite_model, addition="", path_to_main=""):
    path = "/flight_logs/"
    date = str(year) + "-" + str(month) + "-" + str(day)
    file_name = str(kite_model) + "_" + date
    hdf5_path = path_to_main + path + file_name + addition + ".h5"
    ekf_output_df, flight_data_df, config_data = read_results_from_hdf5(hdf5_path)
    return ekf_output_df, flight_data_df, config_data


def read_results_from_hdf5(hdf5_path):
    with h5py.File(hdf5_path, "r") as hf:
        ekf_group = hf["ekf_output"]
        ekf_output_df = pd.DataFrame(
            {
                col: (
                    ekf_group[col][:].astype(str)
                    if ekf_group[col].dtype.kind == "S"
                    else ekf_group[col][:]
                )
                for col in ekf_group.keys()
            }
        )

        flight_group = hf["flight_data"]
        flight_data_df = pd.DataFrame(
            {
                col: (
                    flight_group[col][:].astype(str)
                    if flight_group[col].dtype.kind == "S"
                    else flight_group[col][:]
                )
                for col in flight_group
                if isinstance(flight_group[col], h5py.Dataset)
            }
        )

        config_group = hf["config_data"]
        config_data = read_dict_from_group(config_group)

    return ekf_output_df, flight_data_df, config_data


def read_dict_from_group(group):
    config_dict = {}
    for key, value in group.attrs.items():
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        config_dict[key] = value

    for subgroup_name in group:
        subgroup = group[subgroup_name]
        config_dict[subgroup_name] = read_dict_from_group(subgroup)

    return config_dict


def setup_inverse_solver(kite_model):
    """
    Setup an inverse solver where we prescribe the tether force
    and solve for wind speed (speed_friction).
    """
    # Define the inverse problem:
    # Unknown variables: [tension_tether_ground, input_steering, speed_friction]
    unknown_vars = [
        "speed_tangential",
        "input_steering",
        "speed_friction",  # Changed from speed_tangential to speed_friction
    ]

    solver_options = {
        "ipopt": {
            "print_level": 0,
            "sb": "yes",
            "max_iter": 400,
        },
        "print_time": False,
    }

    kite_model.setup_qs_solver(unknown_vars, solver_options=solver_options)

    return unknown_vars, solver_options


def solve_inverse(
    kite_model, current_state, measured_vtau, measured_wind_speed, unknown_vars
):
    """
    Solve for wind speed given a prescribed tether force.

    Parameters
    ----------
    kite_model : SystemModel
        The system model
    current_state : dict
        Current state with known variables
    prescribed_force : float
        Prescribed tether force (in Newtons)
    measured_wind_speed : float
        Measured wind speed to use as initial condition (speed_friction)
    unknown_vars : list
        List of unknown variable names

    Returns
    -------
    dict
        Solution dictionary with solved variables
    """
    p = [current_state[name] for name in kite_model._qs_inputs]
    lbx, ubx, lbg, ubg = kite_model.get_boundaries(current_state, unknown_vars)

    # Prescribe the tension: set both lower and upper bounds to the prescribed value
    # tension_idx = unknown_vars.index("tension_tether_ground")
    # lbx[tension_idx] = prescribed_force
    # ubx[tension_idx] = prescribed_force

    # Initial guess: use measured wind speed for speed_friction
    speed_friction_idx = unknown_vars.index("speed_friction")
    qs_guess = np.array(
        [
            measured_vtau,  # speed_tangential
            0.0,  # input_steering
            0.5,  # speed_friction (initial guess from measurement)
        ]
    )

    sol = kite_model._qs_solver(x0=qs_guess, p=p, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)

    return sol


def run_inverse_validation(cycle_num=65):
    """
    Run inverse validation: given measured force, solve for wind speed.

    Parameters
    ----------
    cycle_num : int or list of int
        Cycle number(s) to process. Can be a single integer or a list of integers.
    """
    # Load data
    results, flight_data, config_data = read_results(
        "2019", "10", "08", "v3", addition="", path_to_main="./data/LEI-V3-KITE/"
    )
    print(f"Max cycle: {max(flight_data.cycle)}")

    # Filter to cycle(s)
    if isinstance(cycle_num, (list, tuple)):
        mask = flight_data.cycle.isin(cycle_num)
        print(f"Processing cycles: {cycle_num}")
    else:
        mask = flight_data.cycle == cycle_num
        print(f"Processing cycle: {cycle_num}")
    flight_data = flight_data[mask]
    results = results[mask]
    results = results.reset_index(drop=True)
    flight_data = flight_data.reset_index(drop=True)

    # Preprocess
    position = np.array(
        [results.kite_position_x, results.kite_position_y, results.kite_position_z]
    ).T
    velocity = np.array(
        [results.kite_velocity_x, results.kite_velocity_y, results.kite_velocity_z]
    ).T
    distance_radial = np.linalg.norm(position, axis=1)
    speed_tangential = np.linalg.norm(
        np.cross(position, velocity), axis=1
    ) / np.maximum(distance_radial, 1e-12)
    azimuth = np.arctan2(results.kite_position_y, results.kite_position_x)

    # Normalize depower
    flight_data["up"] = (flight_data["up"] - flight_data["up"].min()) / (
        flight_data["up"].max() - flight_data["up"].min()
    )

    # Setup system
    with open("./data/LEI-V3-KITE/v3_aero_input.json", "r") as file:
        aero_input = json.load(file)

    tether = RigidLumpedTether(diameter=0.01)
    wind_model = Wind(wind_model="logarithmic", z0=0.1)
    kite = Kite(
        mass_wing=14,
        area_wing=20,
        aero_input=aero_input,
        mass_kcu=16,
        steering_control="asymmetric",
    )
    kite_model = SystemModel(
        dof=3, quasi_steady=True, kite=kite, tether=tether, wind_model=wind_model
    )

    # Setup inverse solver
    unknown_vars, solver_options = setup_inverse_solver(kite_model)

    # Extract functions
    tension_func = kite_model.extract_function("tension_tether_ground")
    cl_func = kite_model.extract_function("lift_coefficient")
    cd_func = kite_model.extract_function("drag_coefficient")
    aoa_func = kite_model.extract_function("angle_of_attack")

    # Calculate reference wind speed
    uf = (
        results.wind_speed_horizontal
        * kite_model.wind.kappa
        / np.log(results.kite_position_z / kite_model.wind.z0)
    )

    # Run inverse solver
    solutions_inverse = []

    for i, row in flight_data.iterrows():
        # if i > 600:
        #     break
        print(f"\n{'='*60}")
        print(f"Processing row {i + 1}/{len(flight_data)}")
        print(f"{'='*60}")

        # Measured values (to be used as initial conditions or prescribed values)
        measured_wind_speed = uf[i]
        measured_force = float(flight_data.ground_tether_force[i])

        print(f"Measured wind speed (speed_friction): {measured_wind_speed:.2f} m/s")
        print(f"Measured tether force: {measured_force:.2f} N")

        # Define current state (all known quantities)
        current_state = {
            "distance_radial": distance_radial[i],
            "angle_course": row.kite_course,
            "speed_radial": row.tether_reelout_speed,
            "angle_azimuth": azimuth[i] - results.wind_direction[i],
            "angle_elevation": row.kite_elevation,
            "speed_friction": measured_wind_speed,  # Will be optimized in inverse
            "timeder_angle_course": 0.0,  # Approximate
            "input_depower": row.up,
            "tension_tether_ground": measured_force,  # Will be optimized in inverse
            "speed_tangential": speed_tangential[i],  # Will be optimized in inverse
        }

        # INVERSE SOLVE: Prescribe the force, solve for wind speed
        prescribed_force = measured_force
        sol = solve_inverse(
            kite_model,
            current_state,
            measured_vtau=speed_tangential[i],
            measured_wind_speed=measured_wind_speed,  # Initial guess
            unknown_vars=unknown_vars,
        )

        if np.linalg.norm(sol["g"]) < 1:
            # Extract solution
            solved_vtau = float(sol["x"][0])
            solved_steering = float(sol["x"][1])
            solved_wind_speed = float(sol["x"][2])

            print(f"\nInverse solve successful!")
            print(f"  Prescribed force: {prescribed_force:.2f} N")
            print(f"  Solved speed tangential: {solved_vtau:.2f} m/s")
            print(f"  Measured wind speed: {measured_wind_speed:.2f} m/s")
            print(f"  Solved wind speed: {solved_wind_speed:.2f} m/s")
            print(
                f"  Wind speed change: {(solved_wind_speed - measured_wind_speed):.2f} m/s"
            )
            print(f"  Constraint violation: {np.linalg.norm(sol['g']):.2e}")

            # Update current_state with solved values
            current_state["speed_tangential"] = solved_vtau
            current_state["input_steering"] = solved_steering
            current_state["speed_friction"] = solved_wind_speed

            # Compute derived quantities
            state_combined = dict(current_state)
            state_combined["speed_tangential"] = solved_vtau
            state_combined["input_steering"] = solved_steering
            state_combined["speed_friction"] = solved_wind_speed
            state_combined["speed_wind"] = (
                solved_wind_speed
                * np.log(row.kite_elevation * distance_radial[i] / kite_model.wind.z0)
                / kite_model.wind.kappa
            )
            state_combined["cl"] = float(
                cl_func(*[state_combined[name] for name in cl_func.name_in()])
            )
            state_combined["cd"] = float(
                cd_func(*[state_combined[name] for name in cd_func.name_in()])
            )
            state_combined["aoa"] = float(
                aoa_func(*[state_combined[name] for name in aoa_func.name_in()])
            )
            state_combined["time"] = row.time
            state_combined["index"] = i
            state_combined["measured_wind_speed"] = (
                measured_wind_speed
                * np.log(row.kite_elevation * distance_radial[i] / kite_model.wind.z0)
            ) / kite_model.wind.kappa
            state_combined["measured_force"] = measured_force

            solutions_inverse.append(state_combined)

        else:
            print(
                f"Inverse solve FAILED! Constraint violation: {np.linalg.norm(sol['g']):.2e}"
            )

    return solutions_inverse, flight_data


if __name__ == "__main__":
    print("Running inverse wind speed solver...")
    solutions_inverse, flight_data = run_inverse_validation(cycle_num=[62, 63, 64, 65])

    # Convert to DataFrame for easier analysis
    df_inverse = pd.DataFrame(solutions_inverse)

    # Convert wind speeds from reference height (100m) to kite height
    z0 = 0.1  # roughness length
    z_ref = 100.0
    z_kite = df_inverse["angle_elevation"].values * df_inverse["distance_radial"].values

    # Avoid division by zero in log
    z_kite = np.maximum(z_kite, 1.0)

    # Convert to wind at kite height
    df_inverse["measured_wind_at_kite"] = df_inverse["measured_wind_speed"] * np.log(
        z_kite / z0
    )
    df_inverse["speed_friction_at_kite"] = (
        df_inverse["speed_friction"] * np.log(z_kite / z0) / np.log(z_ref / z0)
    )

    print("\n" + "=" * 80)
    print("INVERSE SOLUTION SUMMARY")
    print("=" * 80)
    print(
        df_inverse[
            [
                "time",
                "measured_wind_speed",
                "speed_friction",
                "measured_force",
                "tension_tether_ground",
                "aoa",
            ]
        ]
    )

    # Plot measured vs solved wind speed
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    time = df_inverse["time"].values

    # Plot 1: Wind speed at kite height comparison
    axes[0].plot(
        time,
        df_inverse["measured_wind_speed"],
    )
    axes[0].plot(
        time,
        df_inverse["speed_wind"],
        label="Solved wind at kite height",
    )
    axes[0].set_ylabel("Wind Speed [m/s]", fontsize=12)
    axes[0].set_title(
        "Measured vs Solved Wind Speed at Kite Height", fontsize=13, fontweight="bold"
    )
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=11)

    # Plot 2: Wind speed difference at kite height
    wind_diff_kite = df_inverse["speed_wind"] - df_inverse["measured_wind_speed"]
    axes[1].plot(time, wind_diff_kite, "g-", linewidth=2, marker="^")
    axes[1].axhline(y=0, color="k", linestyle="--", alpha=0.5)
    axes[1].set_xlabel("Time [s]", fontsize=12)
    axes[1].set_ylabel("Wind Speed Difference [m/s]", fontsize=12)
    axes[1].set_title(
        "Solved - Measured Wind Speed at Kite Height", fontsize=13, fontweight="bold"
    )
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        "results/inverse_wind_speed_comparison.png", dpi=150, bbox_inches="tight"
    )
    print("\nPlot saved to results/inverse_wind_speed_comparison.png")
    plt.show()

    # Save results
    df_inverse.to_csv("results/inverse_wind_speed_validation.csv", index=False)
    print("\nResults saved to results/inverse_wind_speed_validation.csv")
