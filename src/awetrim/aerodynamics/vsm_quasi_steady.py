from __future__ import annotations

import copy
from time import perf_counter
from typing import Any, Callable, Mapping, Sequence

import casadi as ca
import numpy as np
from scipy.optimize import least_squares

from awetrim.aerodynamics.protocols import (
    AWETrimSystemModel,
    AxisDefinition,
    VsmBodyAerodynamics,
    VsmSolver,
)


DEFAULT_AXES = AxisDefinition(
    course=np.array([1.0, 0.0, 0.0], dtype=float),
    normal=np.array([0.0, 1.0, 0.0], dtype=float),
    radial=np.array([0.0, 0.0, 1.0], dtype=float),
)

DEFAULT_TRANSFORMATION_C_FROM_VSM = np.array(
    [
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=float,
)

# x = [speed_tangential, angle_roll_body_deg, angle_pitch_body_deg,
#      angle_yaw_body_deg, timeder_angle_course_body]
DEFAULT_BOUNDS_LOWER = np.array([-2.0, -15.0, -15.0, -15.0, -5.0], dtype=float)
DEFAULT_BOUNDS_UPPER = np.array([80.0, 15.0, 15.0, 15.0, 5.0], dtype=float)


def _default_vsm_solver(reference_point: np.ndarray) -> VsmSolver:
    try:
        from VSM.core.Solver import Solver
    except ImportError as exc:  # pragma: no cover - depends on optional package
        raise ImportError(
            "VSM is required when no solver is supplied. Install or expose "
            "`VSM.core.Solver.Solver`, or pass a solver implementing VsmSolver."
        ) from exc

    return Solver(reference_point=reference_point, gamma_initial_distribution_type="zero")


def _as_3vector(value: Any) -> np.ndarray:
    vector = np.asarray(value, dtype=float).reshape(-1)
    if vector.size != 3:
        raise ValueError(f"Expected a 3-vector, got shape {np.asarray(value).shape}")
    return vector


def _numeric_value_for_symbol(system_model: AWETrimSystemModel, name: str) -> Any:
    if name == "speed_wind_ref" and hasattr(system_model.wind, "speed_wind_ref_value"):
        value = system_model.wind.speed_wind_ref_value
        if value is not None:
            return value
    for owner in (
        system_model,
        getattr(system_model, "wind", None),
        getattr(system_model, "kite", None),
        getattr(system_model, "tether", None),
    ):
        if owner is not None and hasattr(owner, name):
            value = getattr(owner, name)
            if not isinstance(value, (ca.MX, ca.SX)):
                return value
    raise ValueError(f"No numeric value available for symbolic variable '{name}'.")


def _as_numeric_3vector(system_model: AWETrimSystemModel, value: Any) -> np.ndarray:
    try:
        return _as_3vector(value)
    except Exception as first_error:
        if not isinstance(value, (ca.MX, ca.SX, ca.DM)):
            raise first_error

    symbols = ca.symvar(value)
    if not symbols:
        return _as_3vector(ca.DM(value).full())
    inputs = [_numeric_value_for_symbol(system_model, symbol.name()) for symbol in symbols]
    func = ca.Function("awetrim_vsm_numeric_eval", symbols, [value])
    return _as_3vector(func(*inputs).full())


def _as_5vector(value: Any, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=float).reshape(-1)
    if vector.shape != (5,):
        raise ValueError(
            f"{name} must be shape (5,) for "
            "[speed_tangential, roll, pitch, yaw, timeder_angle_course_body]."
        )
    return vector


def _system_model_mass_wing(system_model: AWETrimSystemModel) -> float:
    if hasattr(system_model, "mass_wing"):
        return float(getattr(system_model, "mass_wing"))
    if hasattr(system_model, "kite") and hasattr(system_model.kite, "mass_wing"):
        return float(system_model.kite.mass_wing)
    raise AttributeError("system_model must expose mass_wing or kite.mass_wing.")


def _set_course_rate_body(
    system_model: AWETrimSystemModel, course_rate_body: float
) -> None:
    if hasattr(system_model, "timeder_angle_course_body"):
        system_model.timeder_angle_course_body = course_rate_body
    else:
        system_model.timeder_angle_course = course_rate_body


def _acceleration_course_body(system_model: AWETrimSystemModel) -> np.ndarray:
    if hasattr(system_model, "acceleration_course_body"):
        return _as_numeric_3vector(system_model, system_model.acceleration_course_body)
    return _as_numeric_3vector(system_model, system_model.acceleration)


def _force_gravity(system_model: AWETrimSystemModel) -> np.ndarray:
    if hasattr(system_model, "force_gravity"):
        return _as_numeric_3vector(system_model, system_model.force_gravity)
    if hasattr(system_model, "expression"):
        return _as_numeric_3vector(system_model, system_model.expression("force_gravity"))
    if hasattr(system_model, "kite"):
        return _as_numeric_3vector(
            system_model, system_model.kite.force_gravity_for(system_model)
        )
    raise AttributeError(
        "system_model must expose force_gravity, expression('force_gravity'), "
        "or kite.force_gravity_for(system_model)."
    )


def _rotation_matrix(axis: np.ndarray, angle_deg: float) -> np.ndarray:
    theta = np.deg2rad(angle_deg)
    axis_vec = _as_3vector(axis)
    axis_norm = np.linalg.norm(axis_vec)
    if axis_norm == 0.0:
        raise ValueError("Rotation axis must be non-zero.")
    axis_unit = axis_vec / axis_norm
    kx, ky, kz = axis_unit
    skew = np.array(
        [[0.0, -kz, ky], [kz, 0.0, -kx], [-ky, kx, 0.0]],
        dtype=float,
    )
    return np.eye(3) + np.sin(theta) * skew + (1.0 - np.cos(theta)) * (skew @ skew)


def _compose_attitude_rotation(
    *,
    roll_deg: float,
    pitch_deg: float,
    yaw_deg: float,
    axes: AxisDefinition,
) -> np.ndarray:
    roll_matrix = _rotation_matrix(axes.course, roll_deg)
    pitch_matrix = _rotation_matrix(axes.normal, pitch_deg)
    yaw_matrix = _rotation_matrix(axes.radial, yaw_deg)
    return yaw_matrix @ pitch_matrix @ roll_matrix


def _set_body_attitude_from_baseline(
    body: VsmBodyAerodynamics,
    *,
    baseline_sections: list[list[tuple[np.ndarray, np.ndarray]]],
    baseline_spanwise: list[np.ndarray],
    roll_deg: float,
    pitch_deg: float,
    yaw_deg: float,
    axes: AxisDefinition,
    reference_point: np.ndarray,
) -> None:
    combined_rotation = _compose_attitude_rotation(
        roll_deg=roll_deg,
        pitch_deg=pitch_deg,
        yaw_deg=yaw_deg,
        axes=axes,
    )
    origin = _as_3vector(reference_point)

    def rotate_point(point: np.ndarray) -> np.ndarray:
        return origin + combined_rotation @ (_as_3vector(point) - origin)

    for wing, wing_sections, spanwise_base in zip(
        body.wings, baseline_sections, baseline_spanwise
    ):
        for section, (le_base, te_base) in zip(wing.sections, wing_sections):
            section.LE_point = rotate_point(le_base)
            section.TE_point = rotate_point(te_base)

        rotated_span = combined_rotation @ spanwise_base
        span_norm = np.linalg.norm(rotated_span)
        if span_norm == 0.0:
            raise ValueError(
                "Combined attitude produced zero spanwise direction vector."
            )
        wing.spanwise_direction = rotated_span / span_norm

    body.geometry_rotation = combined_rotation
    body._build_panels()


def _baseline_geometry(
    body: VsmBodyAerodynamics,
) -> tuple[list[list[tuple[np.ndarray, np.ndarray]]], list[np.ndarray]]:
    baseline_sections: list[list[tuple[np.ndarray, np.ndarray]]] = []
    baseline_spanwise: list[np.ndarray] = []
    for wing in body.wings:
        baseline_sections.append(
            [
                (
                    np.asarray(section.LE_point, dtype=float).copy(),
                    np.asarray(section.TE_point, dtype=float).copy(),
                )
                for section in wing.sections
            ]
        )
        baseline_spanwise.append(
            np.asarray(wing.spanwise_direction, dtype=float).copy()
        )
    return baseline_sections, baseline_spanwise


def solve_vsm_quasi_steady_trim(
    body_aero: VsmBodyAerodynamics,
    center_of_gravity: np.ndarray,
    reference_point: np.ndarray,
    system_model: AWETrimSystemModel,
    x_guess: np.ndarray,
    *,
    solver: VsmSolver | None = None,
    bounds_lower: np.ndarray = DEFAULT_BOUNDS_LOWER,
    bounds_upper: np.ndarray = DEFAULT_BOUNDS_UPPER,
    transformation_c_from_vsm: np.ndarray = DEFAULT_TRANSFORMATION_C_FROM_VSM,
    include_gravity: bool = False,
    axes: AxisDefinition = DEFAULT_AXES,
    moment_tolerance: float = 1e-2,
    return_timing_breakdown: bool = False,
    max_nfev: int | None = None,
) -> tuple[dict[str, Any], VsmBodyAerodynamics]:
    """Solve one aerodynamic VSM quasi-steady trim state.

    The optimized state is ordered as
    `[speed_tangential, angle_roll_body_deg, angle_pitch_body_deg,
    angle_yaw_body_deg, timeder_angle_course_body]`.
    """

    bounds_lower = _as_5vector(bounds_lower, "bounds_lower")
    bounds_upper = _as_5vector(bounds_upper, "bounds_upper")
    x_guess = _as_5vector(x_guess, "x_guess")
    center_of_gravity = _as_3vector(center_of_gravity)
    reference_point = _as_3vector(reference_point)
    transformation_c_from_vsm = np.asarray(transformation_c_from_vsm, dtype=float)

    if transformation_c_from_vsm.shape != (3, 3):
        raise ValueError("transformation_c_from_vsm must be shape (3, 3).")
    if np.any(bounds_lower >= bounds_upper):
        raise ValueError("Each lower bound must be smaller than its upper bound.")

    if solver is None:
        solver = _default_vsm_solver(reference_point)

    def evaluate_kinematics(x: np.ndarray) -> dict[str, np.ndarray]:
        speed_tangential, _roll, _pitch, _yaw, course_rate_body = x
        _set_course_rate_body(system_model, course_rate_body)
        system_model.speed_tangential = speed_tangential

        inertial_force = -_system_model_mass_wing(system_model) * _as_3vector(
            transformation_c_from_vsm @ _acceleration_course_body(system_model)
        )
        gravity_force = _as_3vector(transformation_c_from_vsm @ _force_gravity(system_model))
        wind_velocity = _as_numeric_3vector(
            system_model,
            transformation_c_from_vsm @ system_model.wind.velocity_wind(system_model),
        )
        kite_velocity = _as_numeric_3vector(
            system_model, transformation_c_from_vsm @ system_model.velocity_kite
        )
        apparent_velocity = _as_numeric_3vector(
            system_model,
            transformation_c_from_vsm @ system_model.velocity_apparent_wind,
        )
        return {
            "va": apparent_velocity,
            "inertial_force": inertial_force,
            "gravity_force": gravity_force,
            "wind_velocity": wind_velocity,
            "kite_velocity": kite_velocity,
            "apparent_velocity": apparent_velocity,
        }

    timing_counters = {
        "residual_evaluations": 0,
        "residual_total_s": 0.0,
        "body_rotate_s": 0.0,
        "kinematics_s": 0.0,
        "solver_s": 0.0,
        "postprocess_s": 0.0,
    }
    cached_eval: dict[str, Any] = {"x": None, "payload": None}
    working_body = copy.deepcopy(body_aero)
    baseline_sections, baseline_spanwise = _baseline_geometry(working_body)

    def moment_residual(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        cached_x = cached_eval["x"]
        if cached_x is not None and np.array_equal(x, cached_x):
            return np.asarray(cached_eval["payload"]["residual"], dtype=float)

        eval_t0 = perf_counter()
        _speed_tangential, roll_deg, pitch_deg, yaw_deg, course_rate_body = x

        t0 = perf_counter()
        _set_body_attitude_from_baseline(
            working_body,
            baseline_sections=baseline_sections,
            baseline_spanwise=baseline_spanwise,
            roll_deg=roll_deg,
            pitch_deg=pitch_deg,
            yaw_deg=yaw_deg,
            axes=axes,
            reference_point=reference_point,
        )
        timing_counters["body_rotate_s"] += perf_counter() - t0

        t0 = perf_counter()
        kin = evaluate_kinematics(x)
        va = _as_3vector(kin["va"])
        inertial_force = _as_3vector(kin["inertial_force"])
        gravity_force = (
            _as_3vector(kin.get("gravity_force", np.zeros(3, dtype=float)))
            if include_gravity
            else np.zeros(3, dtype=float)
        )
        timing_counters["kinematics_s"] += perf_counter() - t0

        aoa_course_deg = np.rad2deg(np.arctan2(va[2], va[0]))
        beta_course_deg = np.rad2deg(np.arctan2(va[1], np.hypot(va[0], va[2])))
        umag = np.linalg.norm(va)
        if umag <= 0.0:
            raise ValueError("Apparent wind magnitude must be positive.")

        working_body.va_initialize(
            Umag=umag,
            angle_of_attack=aoa_course_deg,
            side_slip=beta_course_deg,
            body_rates=course_rate_body,
            body_axis=-axes.radial,
            reference_point=reference_point,
            rates_in_body_frame=False,
        )

        t0 = perf_counter()
        res = solver.solve(working_body)
        timing_counters["solver_s"] += perf_counter() - t0

        cmx = float(res.get("cmx", np.nan))
        cmy = float(res.get("cmy", np.nan))
        cmz = float(res.get("cmz", np.nan))
        total_aero_force = np.array(
            [
                float(res.get("Fx", np.nan)),
                float(res.get("Fy", np.nan)),
                float(res.get("Fz", np.nan)),
            ],
            dtype=float,
        )

        projected_area = float(working_body.wings[0].compute_projected_area())
        if projected_area <= 0.0:
            raise ValueError("VSM body projected area must be positive.")
        max_chord = max(float(panel.chord) for panel in working_body.panels)
        q_inf = 0.5 * float(solver.rho) * umag**2
        denom = q_inf * projected_area * max_chord if max_chord > 0.0 else 1.0

        moment_vec = np.cross(center_of_gravity - reference_point, inertial_force)
        if include_gravity:
            moment_vec += np.cross(center_of_gravity - reference_point, gravity_force)
        delta_cm = moment_vec / denom

        cmx += delta_cm[0]
        cmy += delta_cm[1]
        cmz += delta_cm[2]

        net_force = total_aero_force + inertial_force + gravity_force
        force_denom = q_inf * projected_area
        cfx = np.dot(net_force, axes.course) / force_denom
        cfy = np.dot(net_force, axes.normal) / force_denom

        t0 = perf_counter()
        residual = np.array([cmx, cmy, cmz, cfx, cfy], dtype=float)
        timing_counters["postprocess_s"] += perf_counter() - t0
        timing_counters["residual_evaluations"] += 1
        timing_counters["residual_total_s"] += perf_counter() - eval_t0
        cached_eval["x"] = x.copy()
        cached_eval["payload"] = {
            "residual": residual,
            "kin": kin,
            "va": va,
            "umag": umag,
            "res": res,
            "gravity_force": gravity_force,
            "inertial_force": inertial_force,
        }
        return residual

    opt = least_squares(
        lambda x: moment_residual(x),
        np.clip(x_guess, bounds_lower, bounds_upper),
        bounds=(bounds_lower, bounds_upper),
        max_nfev=max_nfev,
    )

    cm_best = moment_residual(opt.x)
    cmx, cmy, cmz, cfx, cfy = cm_best
    physical_success = bool(
        np.abs(cmx) < moment_tolerance
        and np.abs(cmy) < moment_tolerance
        and np.abs(cmz) < moment_tolerance
    )

    payload = cached_eval["payload"] if np.array_equal(opt.x, cached_eval["x"]) else None
    if payload is None:
        _ = moment_residual(opt.x)
        payload = cached_eval["payload"]

    kin = payload["kin"]
    va = _as_3vector(payload["va"])
    umag = float(payload["umag"])
    res = payload["res"]
    aoa_course_deg = float(np.rad2deg(np.arctan2(va[2], va[0])))
    beta_course_deg = float(np.rad2deg(np.arctan2(va[1], np.hypot(va[0], va[2]))))
    aoa_center_chord_deg = float(res.get("alpha_center_chord_deg", aoa_course_deg))
    beta_center_chord_deg = float(res.get("beta_center_chord_deg", beta_course_deg))

    total_aero_force = np.array(
        [
            float(res.get("Fx", np.nan)),
            float(res.get("Fy", np.nan)),
            float(res.get("Fz", np.nan)),
        ],
        dtype=float,
    )
    va_unit = va / np.linalg.norm(va)
    lift_dir = axes.radial - np.dot(axes.radial, va_unit) * va_unit
    side_dir = np.cross(lift_dir, va_unit)
    aero_roll_deg = float(
        np.rad2deg(
            np.arctan2(
                np.dot(total_aero_force, side_dir),
                np.dot(total_aero_force, lift_dir),
            )
        )
    )

    inertial_force = _as_3vector(payload["inertial_force"])
    gravity_force = _as_3vector(payload["gravity_force"])
    x_cp = res.get("center_of_pressure", np.nan)
    x_cp_arr = np.asarray(x_cp, dtype=float)
    x_cp_point = (
        x_cp_arr.reshape(3) if x_cp_arr.size == 3 else np.array([float(x_cp_arr), 0.0, 0.0])
    )
    tether_force = float(total_aero_force[2] + gravity_force[2] + inertial_force[2])

    result: dict[str, Any] = {
        "opt_x": np.asarray(opt.x, dtype=float),
        "cm": np.array([cmx, cmy, cmz], dtype=float),
        "cfx": float(cfx),
        "cfy": float(cfy),
        "side_slip_deg": beta_center_chord_deg,
        "side_slip_course_deg": beta_course_deg,
        "aero_roll_deg": aero_roll_deg,
        "aoa_deg": aoa_center_chord_deg,
        "aoa_course_deg": aoa_course_deg,
        "success": bool(opt.success),
        "success_physical": physical_success,
        "gravity_force": gravity_force,
        "inertial_force": inertial_force,
        "cl": res.get("cl"),
        "cd": res.get("cd"),
        "total_aero_force_vec": total_aero_force,
        "x_cp_point": x_cp_point,
        "wind_vel_world": _as_3vector(kin.get("wind_velocity", np.zeros(3))),
        "kite_vel_world": _as_3vector(kin.get("kite_velocity", np.zeros(3))),
        "va_vel_world": _as_3vector(kin.get("apparent_velocity", va)),
        "Umag": umag,
        "course_axis": axes.course,
        "radial_axis": axes.radial,
        "normal_axis": axes.normal,
        "F_distribution": res.get("F_distribution"),
        "panel_cp_locations": res.get("panel_cp_locations"),
        "alpha_at_ac": res.get("alpha_at_ac"),
        "gamma_distribution": res.get("gamma_distribution"),
        "tether_force": tether_force,
        "optimizer": opt,
    }

    if return_timing_breakdown:
        residual_total = float(timing_counters["residual_total_s"])
        if residual_total > 0.0:
            timing_counters["solver_share"] = timing_counters["solver_s"] / residual_total
            timing_counters["body_rotate_share"] = (
                timing_counters["body_rotate_s"] / residual_total
            )
            timing_counters["kinematics_share"] = (
                timing_counters["kinematics_s"] / residual_total
            )
            timing_counters["postprocess_share"] = (
                timing_counters["postprocess_s"] / residual_total
            )
        result["timing_breakdown"] = timing_counters

    return result, working_body


def compute_vsm_trim_stability_derivatives(
    body_aero: VsmBodyAerodynamics,
    center_of_gravity: np.ndarray,
    reference_point: np.ndarray,
    x_trim: np.ndarray,
    trim_result: Mapping[str, Any],
    *,
    solver: VsmSolver | None = None,
    axes: AxisDefinition = DEFAULT_AXES,
    mass: float = 15.0,
    inertia_xx: float = 100.0,
    inertia_yy: float = 19.43,
    inertia_zz: float = 100.0,
    distance_radial: float | None = None,
    eps_vel: float = 0.1,
    eps_angle_deg: float = 0.5,
    eps_rate: float = 0.01,
) -> dict[str, Any]:
    """Compute aerodynamic stability derivatives around a VSM trim state."""

    center_of_gravity = _as_3vector(center_of_gravity)
    reference_point = _as_3vector(reference_point)
    x_trim = _as_5vector(x_trim, "x_trim")
    if solver is None:
        solver = _default_vsm_solver(reference_point)

    speed_tangential, roll0, pitch0, yaw0, course_rate0 = x_trim
    va_trim = _as_3vector(trim_result["va_vel_world"])
    f_tether = np.array([0.0, 0.0, -float(trim_result["tether_force"])], dtype=float)
    r_arm = reference_point - center_of_gravity
    moment_tether_at_cg = np.cross(r_arm, f_tether)

    working_body = copy.deepcopy(body_aero)
    baseline_sections, baseline_spanwise = _baseline_geometry(working_body)
    projected_area = float(body_aero.wings[0].compute_projected_area())
    max_chord = max(float(panel.chord) for panel in body_aero.panels)

    def eval_force_moment(
        delta_v: np.ndarray,
        omega_perturb: np.ndarray,
        delta_roll_deg: float = 0.0,
        delta_pitch_deg: float = 0.0,
        delta_yaw_deg: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        _set_body_attitude_from_baseline(
            working_body,
            baseline_sections=baseline_sections,
            baseline_spanwise=baseline_spanwise,
            roll_deg=roll0 + delta_roll_deg,
            pitch_deg=pitch0 + delta_pitch_deg,
            yaw_deg=yaw0 + delta_yaw_deg,
            axes=axes,
            reference_point=reference_point,
        )
        va_pert = va_trim - delta_v
        umag = np.linalg.norm(va_pert)
        aoa_deg = np.rad2deg(np.arctan2(va_pert[2], va_pert[0]))
        beta_deg = np.rad2deg(np.arctan2(va_pert[1], np.hypot(va_pert[0], va_pert[2])))
        omega_total = -course_rate0 * axes.radial + omega_perturb
        omega_mag = np.linalg.norm(omega_total)
        omega_axis = omega_total / omega_mag if omega_mag > 1e-12 else axes.radial

        working_body.va_initialize(
            Umag=umag,
            angle_of_attack=aoa_deg,
            side_slip=beta_deg,
            body_rates=omega_mag,
            body_axis=omega_axis,
            reference_point=reference_point,
            rates_in_body_frame=False,
        )
        res = solver.solve(working_body)
        f_aero = np.array(
            [
                float(res.get("Fx", 0.0)),
                float(res.get("Fy", 0.0)),
                float(res.get("Fz", 0.0)),
            ],
            dtype=float,
        )
        q_inf = 0.5 * float(solver.rho) * umag**2
        denom = q_inf * projected_area * max_chord if projected_area > 0 else 1.0
        moment_aero_at_ref = (
            np.array(
                [
                    float(res.get("cmx", 0.0)),
                    float(res.get("cmy", 0.0)),
                    float(res.get("cmz", 0.0)),
                ],
                dtype=float,
            )
            * denom
        )

        speed_tangential_eff = float(speed_tangential) + float(np.dot(delta_v, axes.course))
        f_inertial = np.zeros(3, dtype=float)
        f_inertial[1] = mass * speed_tangential_eff * float(course_rate0)
        if distance_radial is not None and float(distance_radial) > 0.0:
            f_inertial[2] = mass * speed_tangential_eff**2 / float(distance_radial)

        moment_at_cg = moment_aero_at_ref + np.cross(r_arm, f_aero) + moment_tether_at_cg
        force_at_cg = f_aero + f_tether + f_inertial
        return force_at_cg, moment_at_cg

    zero3 = np.zeros(3, dtype=float)
    eps_angle_rad = np.deg2rad(eps_angle_deg)

    def central_diff_col(
        delta_v: np.ndarray,
        omega_perturb: np.ndarray,
        step: float,
        droll: float = 0.0,
        dpitch: float = 0.0,
        dyaw: float = 0.0,
    ) -> np.ndarray:
        force_plus, moment_plus = eval_force_moment(delta_v, omega_perturb, droll, dpitch, dyaw)
        force_minus, moment_minus = eval_force_moment(
            -delta_v, -omega_perturb, -droll, -dpitch, -dyaw
        )
        d_force = (force_plus - force_minus) / (2.0 * step)
        d_moment = (moment_plus - moment_minus) / (2.0 * step)
        return np.array(
            [d_force[0], d_force[1], d_force[2], d_moment[0], d_moment[1], d_moment[2]]
        )

    col_u = central_diff_col(+eps_vel * axes.course, zero3, eps_vel)
    col_theta = central_diff_col(zero3, zero3, eps_angle_rad, dpitch=eps_angle_deg)
    col_q = central_diff_col(zero3, eps_rate * axes.normal, eps_rate)
    j_long = np.array(
        [
            [col_u[0], col_theta[0], col_q[0]],
            [col_u[2], col_theta[2], col_q[2]],
            [col_u[4], col_theta[4], col_q[4]],
        ]
    )

    col_v = central_diff_col(+eps_vel * axes.normal, zero3, eps_vel)
    col_phi = central_diff_col(zero3, zero3, eps_angle_rad, droll=eps_angle_deg)
    col_psi = central_diff_col(zero3, zero3, eps_angle_rad, dyaw=eps_angle_deg)
    col_p = central_diff_col(zero3, eps_rate * axes.course, eps_rate)
    col_r = central_diff_col(zero3, eps_rate * axes.radial, eps_rate)
    j_lat = np.array(
        [
            [col_v[1], col_phi[1], col_psi[1], col_p[1], col_r[1]],
            [col_v[3], col_phi[3], col_psi[3], col_p[3], col_r[3]],
            [col_v[5], col_phi[5], col_psi[5], col_p[5], col_r[5]],
        ]
    )

    a_long = np.zeros((3, 3))
    a_long[0, :] = j_long[0, :] / mass
    a_long[1, :] = [0.0, 0.0, 1.0]
    a_long[2, :] = j_long[2, :] / inertia_yy

    a_lat = np.zeros((5, 5))
    a_lat[0, :] = j_lat[0, :] / mass
    a_lat[1, :] = [0.0, 0.0, 0.0, 1.0, 0.0]
    a_lat[2, :] = [0.0, 0.0, 0.0, 0.0, 1.0]
    a_lat[3, :] = j_lat[1, :] / inertia_xx
    a_lat[4, :] = j_lat[2, :] / inertia_zz

    eig_long, vec_long = np.linalg.eig(a_long)
    eig_lat, vec_lat = np.linalg.eig(a_lat)

    def timescales(eigvals: np.ndarray) -> np.ndarray:
        real_parts = np.real(eigvals)
        abs_re = np.abs(real_parts)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(abs_re > 1e-12, 1.0 / abs_re, np.inf)

    return {
        "J_long": j_long,
        "J_lat": j_lat,
        "A_long": a_long,
        "A_lat": a_lat,
        "eig_long": eig_long,
        "eig_lat": eig_lat,
        "vec_long": vec_long,
        "vec_lat": vec_lat,
        "Tfast_long": timescales(eig_long),
        "Tfast_lat": timescales(eig_lat),
        "stable_long": bool(np.all(np.real(eig_long) < 0.0)),
        "stable_lat": bool(np.all(np.real(eig_lat) < 0.0)),
        "F_tether": f_tether,
        "M_tether_at_CG": moment_tether_at_cg,
    }


def _as_sequence(value: Sequence[float] | float) -> list[float]:
    if isinstance(value, np.ndarray):
        return [float(v) for v in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [float(v) for v in value]
    return [float(value)]


def run_vsm_quasi_steady_sweep(
    *,
    build_body: Callable[[dict[str, float]], VsmBodyAerodynamics],
    system_model: AWETrimSystemModel,
    center_of_gravity: np.ndarray,
    reference_point: np.ndarray,
    x_guess: np.ndarray,
    principal_axis: str,
    secondary_axis: str,
    sweep_values: Mapping[str, Sequence[float] | float],
    update_system_model: Callable[[AWETrimSystemModel, dict[str, float]], None] | None = None,
    solver_factory: Callable[[np.ndarray], VsmSolver] | None = None,
    bounds_lower: np.ndarray = DEFAULT_BOUNDS_LOWER,
    bounds_upper: np.ndarray = DEFAULT_BOUNDS_UPPER,
    transformation_c_from_vsm: np.ndarray = DEFAULT_TRANSFORMATION_C_FROM_VSM,
    include_gravity: bool = False,
    axes: AxisDefinition = DEFAULT_AXES,
    moment_tolerance: float = 1e-4,
    return_timing_breakdown: bool = False,
    max_nfev: int | None = None,
) -> list[dict[str, Any]]:
    """Run a warm-started principal/secondary VSM aerodynamic trim sweep."""

    if principal_axis not in sweep_values:
        raise KeyError(f"principal_axis '{principal_axis}' missing from sweep_values")
    if secondary_axis not in sweep_values:
        raise KeyError(f"secondary_axis '{secondary_axis}' missing from sweep_values")

    principal_values = _as_sequence(sweep_values[principal_axis])
    secondary_values = _as_sequence(sweep_values[secondary_axis])
    if principal_axis == secondary_axis:
        secondary_values = [secondary_values[0]]

    base_values = {key: _as_sequence(value)[0] for key, value in sweep_values.items()}
    rows: list[dict[str, Any]] = []

    for secondary_value in secondary_values:
        current_guess = _as_5vector(x_guess, "x_guess").copy()
        for principal_value in principal_values:
            case_values = dict(base_values)
            case_values[principal_axis] = principal_value
            case_values[secondary_axis] = secondary_value
            if update_system_model is not None:
                update_system_model(system_model, case_values)

            solver = (
                solver_factory(reference_point)
                if solver_factory is not None
                else _default_vsm_solver(_as_3vector(reference_point))
            )
            result, body = solve_vsm_quasi_steady_trim(
                body_aero=build_body(case_values),
                center_of_gravity=center_of_gravity,
                reference_point=reference_point,
                system_model=system_model,
                x_guess=current_guess,
                solver=solver,
                bounds_lower=bounds_lower,
                bounds_upper=bounds_upper,
                transformation_c_from_vsm=transformation_c_from_vsm,
                include_gravity=include_gravity,
                axes=axes,
                moment_tolerance=moment_tolerance,
                return_timing_breakdown=return_timing_breakdown,
                max_nfev=max_nfev,
            )
            rows.append(
                {
                    "principal_axis": principal_axis,
                    "secondary_axis": secondary_axis,
                    "principal_value": principal_value,
                    "secondary_value": secondary_value,
                    "case_values": case_values,
                    "result": result,
                    "body": body,
                }
            )
            if result.get("success", False):
                current_guess = np.asarray(result["opt_x"], dtype=float)

    return rows


def vsm_quasi_steady_sweep_to_dataframe(sweep_rows: Sequence[Mapping[str, Any]]):
    """Convert VSM aerodynamic sweep rows into a flat pandas DataFrame."""
    import pandas as pd

    rows = []
    for row in sweep_rows:
        result = row["result"]
        opt_x = np.asarray(result["opt_x"], dtype=float)
        cmx, cmy, cmz = np.asarray(result["cm"], dtype=float)
        rows.append(
            {
                "principal_axis": row["principal_axis"],
                "secondary_axis": row["secondary_axis"],
                "principal_value": float(row["principal_value"]),
                "secondary_value": float(row["secondary_value"]),
                "speed_tangential": float(opt_x[0]),
                "angle_roll_body_deg": float(opt_x[1]),
                "angle_pitch_body_deg": float(opt_x[2]),
                "angle_yaw_body_deg": float(opt_x[3]),
                "timeder_angle_course_body": float(opt_x[4]),
                "aoa_center_deg": float(result["aoa_deg"]),
                "aoa_course_deg": float(result["aoa_course_deg"]),
                "beta_center_deg": float(result["side_slip_deg"]),
                "beta_course_deg": float(result["side_slip_course_deg"]),
                "aero_roll_deg": float(result["aero_roll_deg"]),
                "cl": float(result["cl"]),
                "cd": float(result["cd"]),
                "cmx": float(cmx),
                "cmy": float(cmy),
                "cmz": float(cmz),
                "norm_cm": float(np.linalg.norm([cmx, cmy, cmz])),
                "cfx": float(result["cfx"]),
                "cfy": float(result["cfy"]),
                "success": bool(result["success"]),
                "success_physical": bool(result["success_physical"]),
            }
        )
    return pd.DataFrame(rows)


def plot_vsm_quasi_steady_sweep(
    df: Any,
    principal_axis: str,
    secondary_axis: str,
    *,
    show: bool = True,
) -> tuple[Any, Any] | None:
    """Plot standard VSM aerodynamic quasi-steady sweep figures."""
    import matplotlib.pyplot as plt

    if df.empty:
        return None

    x_col = "principal_value"
    line_col = "secondary_value"
    fig1, ax1 = plt.subplots(3, 1, figsize=(7, 9), sharex=True)
    for sec_val in sorted(df[line_col].dropna().unique()):
        sub = df[df[line_col] == sec_val].sort_values(x_col)
        label = f"{secondary_axis}={sec_val:.3f}"
        ax1[0].plot(sub[x_col], sub["timeder_angle_course_body"], "o-", label=label)
        ax1[1].plot(sub[x_col], sub["beta_center_deg"], "o-", label=label)
        ax1[2].plot(sub[x_col], sub["aero_roll_deg"], "o-", label=label)

    ax1[0].axhline(0, color="k", linewidth=0.8)
    ax1[0].set_ylabel("course rate [rad/s]")
    ax1[0].legend()
    ax1[1].set_ylabel("Sideslip center [deg]")
    ax1[2].set_xlabel(principal_axis)
    ax1[2].set_ylabel("Aero roll angle [deg]")
    fig1.suptitle(
        f"VSM aerodynamic quasi-steady sweep (x={principal_axis}, series={secondary_axis})",
        y=0.995,
    )
    fig1.tight_layout()

    fig2, ax2 = plt.subplots(3, 1, figsize=(7, 9), sharex=True)
    for sec_val in sorted(df[line_col].dropna().unique()):
        sub = df[df[line_col] == sec_val].sort_values(x_col)
        label = f"{secondary_axis}={sec_val:.3f}"
        ax2[0].plot(sub[x_col], sub["aoa_center_deg"], "o-", label=label)
        ax2[1].plot(sub[x_col], sub["cl"], "o-", label=label)
        ax2[2].plot(sub[x_col], sub["cd"], "o-", label=label)

    ax2[0].set_ylabel("AoA center [deg]")
    ax2[0].legend()
    ax2[1].set_ylabel("Lift coeff")
    ax2[2].set_ylabel("Drag coeff")
    ax2[2].set_xlabel(principal_axis)
    fig2.tight_layout()
    if show:
        plt.show()
    return fig1, fig2


# Compatibility aliases for scripts migrated from Vortex-Step-Method.
solve_quasi_steady_state = solve_vsm_quasi_steady_trim
compute_stability_derivatives = compute_vsm_trim_stability_derivatives
run_quasi_steady_sweep = run_vsm_quasi_steady_sweep
quasi_steady_sweep_rows_to_dataframe = vsm_quasi_steady_sweep_to_dataframe
plot_quasi_steady_sweep_dataframe = plot_vsm_quasi_steady_sweep


__all__ = [
    "DEFAULT_AXES",
    "DEFAULT_BOUNDS_LOWER",
    "DEFAULT_BOUNDS_UPPER",
    "DEFAULT_TRANSFORMATION_C_FROM_VSM",
    "AxisDefinition",
    "compute_stability_derivatives",
    "compute_vsm_trim_stability_derivatives",
    "plot_quasi_steady_sweep_dataframe",
    "plot_vsm_quasi_steady_sweep",
    "quasi_steady_sweep_rows_to_dataframe",
    "run_quasi_steady_sweep",
    "run_vsm_quasi_steady_sweep",
    "solve_quasi_steady_state",
    "solve_vsm_quasi_steady_trim",
    "vsm_quasi_steady_sweep_to_dataframe",
]
