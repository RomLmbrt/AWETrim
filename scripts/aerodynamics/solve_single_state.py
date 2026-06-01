"""Solve one VSM aerodynamic trim state.

To trim on a deformed shape from an aerostructural run, pass
``--deformed-from <case_dir>`` (where ``case_dir`` contains the deformed
``aero_geometry.yaml`` and ``struc_geometry.yaml``). The deformed geometries
are used by VSM and for plotting, but mass, inertia tensor and centre of
gravity are still read from ``system.yaml`` in ``--config-folder`` -- this
script does NOT recompute inertia or CoG from the deformed shape.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from common import (
    add_common_arguments,
    build_body,
    build_system_model,
    output_dir,
    parsed_common,
    print_trim_summary,
    save_figure,
    write_json,
)

from awetrim.aerodynamics.vsm_quasi_steady import (
    DEFAULT_TRANSFORMATION_C_FROM_VSM,
    solve_vsm_qs_trim_with_williams_tether,
    solve_vsm_quasi_steady_trim,
)
from awetrim.system.williams_tether import WilliamsTether


def _euler_C_from_K(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Numpy version of reference_frames.transformation_C_from_K (Yaw·Pitch·Roll)."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    Roll = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Pitch = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Yaw = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Yaw @ Pitch @ Roll


def _T_Wind_from_C(system_model) -> np.ndarray:
    """Numerical wind<-course matrix evaluated from the current system_model angles."""
    import casadi as ca
    from awetrim.utils.reference_frames import transformation_C_from_W

    az = float(system_model.angle_azimuth)
    el = float(system_model.angle_elevation)
    chi = float(system_model.angle_course)
    dw = float(getattr(system_model.wind, "direction_wind", 0.0))
    T_C_from_W = np.array(
        ca.DM(transformation_C_from_W(az, el, chi)).full(), dtype=float
    )
    T_Wind_from_W = np.array(
        [
            [np.cos(-dw), -np.sin(-dw), 0.0],
            [np.sin(-dw), np.cos(-dw), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    return T_Wind_from_W @ T_C_from_W.T


def _load_struc_nodes_and_edges(struc_path: str | None):
    """Return (nodes_C, wing_edges, bridle_edges, wing_ids) from struc_geometry.yaml.

    Nodes are in the course frame with KCU at origin (node 0). Wing nodes come
    from ``wing_particles``; bridle nodes from ``bridle_particles``. Pulley
    rows (3 ids) are split into two segments. Returns ``None`` if unavailable.
    """
    if not struc_path:
        return None
    import yaml

    path = Path(struc_path)
    if not path.exists():
        return None
    data = yaml.safe_load(path.read_text(encoding="utf-8"))

    nodes: dict[int, np.ndarray] = {0: np.zeros(3, dtype=float)}
    wing_ids: list[int] = []
    for row in data.get("wing_particles", {}).get("data", []):
        pid = int(row[0])
        nodes[pid] = np.array([float(row[1]), float(row[2]), float(row[3])])
        wing_ids.append(pid)
    for row in data.get("bridle_particles", {}).get("data", []):
        pid = int(row[0])
        nodes[pid] = np.array([float(row[1]), float(row[2]), float(row[3])])

    def _edges(key: str) -> list[tuple[int, int]]:
        out: list[tuple[int, int]] = []
        for row in data.get(key, {}).get("data", []):
            ids = [int(v) for v in row[1:] if v is not None]
            if len(ids) == 2:
                out.append((ids[0], ids[1]))
            elif len(ids) == 3:
                out.append((ids[1], ids[0]))
                out.append((ids[0], ids[2]))
        return out

    return nodes, _edges("wing_connections"), _edges("bridle_connections"), wing_ids


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Solve one VSM aerodynamic trim state."
    )
    add_common_arguments(parser)
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()
    values = parsed_common(args)
    out_dir = output_dir(args, "single_state")

    body, body_props = build_body(args)
    system_model = build_system_model(args)

    use_williams = isinstance(getattr(system_model, "tether", None), WilliamsTether)

    if use_williams:
        print(
            "Tether model: WilliamsTether -> running joint trim+tether least_squares."
        )
        result, _ = solve_vsm_qs_trim_with_williams_tether(
            body_aero=body,
            center_of_gravity=values["center_of_gravity"],
            reference_point=values["reference_point"],
            system_model=system_model,
            x_guess=values["x_guess"],
            bounds_lower=values["bounds_lower"],
            bounds_upper=values["bounds_upper"],
            include_gravity=args.include_gravity,
            moment_tolerance=args.moment_tolerance,
            max_nfev=args.max_nfev,
        )
    else:
        result, _ = solve_vsm_quasi_steady_trim(
            body_aero=body,
            center_of_gravity=values["center_of_gravity"],
            reference_point=values["reference_point"],
            system_model=system_model,
            x_guess=values["x_guess"],
            bounds_lower=values["bounds_lower"],
            bounds_upper=values["bounds_upper"],
            include_gravity=args.include_gravity,
            moment_tolerance=args.moment_tolerance,
            return_timing_breakdown=True,
            max_nfev=args.max_nfev,
        )

    print_trim_summary(result)

    if use_williams:
        print("--- Williams tether ---")
        print(f"  elevation_last [deg] : {result['williams_elevation_last_deg']:.4f}")
        print(f"  azimuth_last   [deg] : {result['williams_azimuth_last_deg']:.4f}")
        print(f"  tether_length  [m]   : {result['williams_tether_length']:.4f}")
        print(f"  ground residual [m]  : {result['williams_ground_residual']}")
        print(
            f"  |F_kite_resultant|   : {np.linalg.norm(result['force_kite_resultant']):.3f} N"
        )

    json_path = (
        Path(args.output_json) if args.output_json else out_dir / "trim_result.json"
    )
    write_json(json_path, result)

    fig, ax = plt.subplots(figsize=(6, 4))
    labels = ["cmx", "cmy", "cmz", "cfx", "cfy"]
    values_plot = np.r_[
        np.asarray(result["cm"], dtype=float), result["cfx"], result["cfy"]
    ]
    ax.bar(
        labels,
        values_plot,
        color=["#4C78A8", "#4C78A8", "#4C78A8", "#F58518", "#F58518"],
    )
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_ylabel("Residual coefficient [-]")
    ax.set_title("VSM aerodynamic trim residuals")
    fig.tight_layout()
    save_figure(fig, out_dir / "trim_residuals.pdf")
    if args.no_show:
        plt.close(fig)

    if use_williams:
        positions = result["williams_positions"]
        rk = np.asarray(result["r_kite"], dtype=float)
        radius = float(result["williams_tether_length"])

        fig2 = plt.figure(figsize=(7, 6))
        ax2 = fig2.add_subplot(111, projection="3d")

        # Wind window: az in [-90, 90] deg, el in [0, 90] deg at radial = tether length.
        az_grid = np.deg2rad(np.linspace(-90.0, 90.0, 40))
        el_grid = np.deg2rad(np.linspace(0.0, 90.0, 20))
        az_mesh, el_mesh = np.meshgrid(az_grid, el_grid)
        ww_x = radius * np.cos(el_mesh) * np.cos(az_mesh)
        ww_y = radius * np.cos(el_mesh) * np.sin(az_mesh)
        ww_z = radius * np.sin(el_mesh)
        ax2.plot_surface(
            ww_x,
            ww_y,
            ww_z,
            color="#4C78A8",
            alpha=0.08,
            linewidth=0,
            shade=False,
        )

        # Color/linewidth convention:
        #   kite + tether: same color, inflatable tubes thickest, tether medium
        #   bridles: orange, thinnest
        kite_color = "black"
        lw_tube = 2.5
        lw_tether = 1.4
        lw_bridle = 0.7

        ax2.plot(
            positions[:, 0],
            positions[:, 1],
            positions[:, 2],
            color=kite_color,
            linewidth=lw_tether,
            marker="o",
            markersize=2,
            label="tether",
        )
        ax2.scatter([0.0], [0.0], [0.0], color="black", s=40, label="ground")

        # Kite + bridles, scaled up so they are visible against the tether length.
        # Convention: struc_geometry has the KCU at the origin and the wing
        # above it; we place the KCU at (0, 0, distance_radial) in the course
        # frame (= r_kite). The trim attitude (roll, pitch, yaw) is applied in
        # the VSM frame -- exactly like the solver does in
        # vsm_quasi_steady._compose_attitude_rotation (R = Yaw·Pitch·Roll
        # about VSM axes [1,0,0]/[0,1,0]/[0,0,1]) -- then mapped to course via
        # T_C_from_VSM and finally to the wind frame.
        kite_scale = 5.0
        opt_x = np.asarray(result["opt_x"], dtype=float)
        # _euler_C_from_K(roll, pitch, yaw) == solver's Yaw·Pitch·Roll. Treating
        # it as R_VSM (acts on VSM-frame points) reproduces the solver exactly.
        R_VSM = _euler_C_from_K(
            np.deg2rad(opt_x[1]), np.deg2rad(opt_x[2]), np.deg2rad(opt_x[3])
        )
        T_C_from_VSM = np.asarray(DEFAULT_TRANSFORMATION_C_FROM_VSM, dtype=float)
        T_Wind_from_C = _T_Wind_from_C(system_model)
        # Body-attitude rotation expressed in the course frame, for geometry
        # that is already in course coordinates (struc_geometry). Conjugation
        # through T_C_from_VSM = diag(-1,-1,1) flips the sign of roll & pitch.
        R_C = T_C_from_VSM @ R_VSM @ T_C_from_VSM

        def _to_wind_C(p_C_kcu: np.ndarray) -> np.ndarray:
            """Scale around KCU and place at rk in wind frame.

            Caller is responsible for having already applied the body attitude
            (in VSM or C frame, as appropriate for the source geometry).
            """
            p = (T_Wind_from_C @ p_C_kcu.T).T * kite_scale
            return p + rk

        struc_data = _load_struc_nodes_and_edges(body_props.get("struc_geometry_path"))
        wing_edges: list[tuple[int, int]] = []
        bridle_edges: list[tuple[int, int]] = []
        nodes_C: dict[int, np.ndarray] | None = None
        if struc_data is not None and struc_data[3]:
            nodes_C, wing_edges, bridle_edges, _wing_ids = struc_data

        def _draw_struc_edges(edges, *, color, linewidth, linestyle="-"):
            for ci, cj in edges:
                if ci not in nodes_C or cj not in nodes_C:
                    continue
                seg_C = (R_C @ np.vstack([nodes_C[ci], nodes_C[cj]]).T).T
                seg_W = _to_wind_C(seg_C)
                ax2.plot(
                    seg_W[:, 0],
                    seg_W[:, 1],
                    seg_W[:, 2],
                    color=color,
                    linewidth=linewidth,
                    linestyle=linestyle,
                )

        if nodes_C is not None and wing_edges:
            # Inflatable tubes (LE, struts, TE) from struc_geometry wing_connections.
            _draw_struc_edges(wing_edges, color=kite_color, linewidth=lw_tube)
            ax2.plot(
                [],
                [],
                color=kite_color,
                linewidth=lw_tube,
                label=f"kite tubes (x{kite_scale:g})",
            )
        else:
            # Fallback: VSM panel outline if no struc_geometry is available.
            panels = list(getattr(body, "panels", []) or [])
            for panel in panels:
                pts_VSM = np.asarray(panel.corner_points, dtype=float)
                pts_C = (T_C_from_VSM @ (R_VSM @ pts_VSM.T)).T
                loop_C = np.vstack([pts_C, pts_C[:1]])
                loop_W = _to_wind_C(loop_C)
                ax2.plot(
                    loop_W[:, 0],
                    loop_W[:, 1],
                    loop_W[:, 2],
                    color=kite_color,
                    linewidth=lw_tube,
                )
            if panels:
                ax2.plot(
                    [],
                    [],
                    color=kite_color,
                    linewidth=lw_tube,
                    label=f"kite panels (x{kite_scale:g})",
                )

        if nodes_C is not None and bridle_edges:
            _draw_struc_edges(
                bridle_edges, color="tab:grey", linewidth=lw_bridle, linestyle="--"
            )
            ax2.plot(
                [],
                [],
                color="tab:grey",
                linewidth=lw_bridle,
                linestyle="--",
                label="bridles",
            )

        ax2.scatter(
            rk[0], rk[1], rk[2], color=kite_color, marker="D", s=30, label="KCU"
        )

        tether = system_model.tether
        tether_info = type(tether).__name__
        if hasattr(tether, "n_elements"):
            tether_info += f", N={int(tether.n_elements)}"
        if hasattr(tether, "cf"):
            tether_info += f", cf={float(tether.cf):.3g}"

        ax2.set_xlabel("x [m]")
        ax2.set_ylabel("y [m]")
        ax2.set_zlabel("z [m]")
        ax2.set_title(f"Williams tether shape (kite -> ground)\n[{tether_info}]")
        ax2.legend()
        # Equal aspect so the quarter-sphere actually looks like a sphere.
        try:
            ax2.set_aspect("equal")
        except NotImplementedError:
            # Older matplotlib: fall back to box aspect with manual ranges.
            r = radius
            ax2.set_xlim(0.0, r)
            ax2.set_ylim(-r, r)
            ax2.set_zlim(0.0, r)
            ax2.set_box_aspect((1.0, 2.0, 1.0))
        fig2.tight_layout()
        save_figure(fig2, out_dir / "williams_tether_shape.pdf")
        if args.no_show:
            plt.close(fig2)

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
