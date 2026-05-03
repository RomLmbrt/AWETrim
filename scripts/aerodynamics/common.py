from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

DEFAULT_VSM_SRC = PROJECT_DIR.parent / "Vortex-Step-Method" / "src"
DEFAULT_POWERED_GEOMETRY_DIR = (
    PROJECT_DIR / "data" / "LEI-V3-KITE" / "kite_geometries" / "powered_geometry"
)
DEFAULT_GEOMETRY_YAML = DEFAULT_POWERED_GEOMETRY_DIR / "aero_geometry.yaml"
DEFAULT_BRIDLE_PATH = DEFAULT_POWERED_GEOMETRY_DIR / "struc_geometry.yaml"
DEFAULT_OUTPUT_ROOT = PROJECT_DIR / "results" / "aerodynamics"


def csv_vector(value: str, *, length: int, name: str) -> np.ndarray:
    parts = [float(part.strip()) for part in value.split(",") if part.strip()]
    if len(parts) != length:
        raise argparse.ArgumentTypeError(f"{name} must contain {length} values.")
    return np.asarray(parts, dtype=float)


def add_vsm_path(path: str | None) -> None:
    if path:
        vsm_src = Path(path).expanduser().resolve()
        if str(vsm_src) not in sys.path:
            sys.path.insert(0, str(vsm_src))


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--vsm-src",
        default=str(DEFAULT_VSM_SRC) if DEFAULT_VSM_SRC.exists() else None,
        help="Optional path to VSM/src.",
    )
    parser.add_argument(
        "--geometry-yaml",
        default=str(DEFAULT_GEOMETRY_YAML),
        help="VSM aero geometry YAML.",
    )
    parser.add_argument(
        "--bridle-path",
        default=str(DEFAULT_BRIDLE_PATH) if DEFAULT_BRIDLE_PATH.exists() else None,
        help="Optional VSM bridle YAML.",
    )
    parser.add_argument("--n-panels", type=int, default=18)
    parser.add_argument("--spanwise-panel-distribution", default="uniform")
    parser.add_argument("--reference-point", default="0,0,0")
    parser.add_argument("--center-of-gravity", default="0.5,0,5")
    parser.add_argument("--mass-wing", type=float, default=30.0)
    parser.add_argument("--tether-diameter", type=float, default=0.01)
    parser.add_argument("--wind-speed", type=float, default=6.0)
    parser.add_argument("--elevation-deg", type=float, default=0.0)
    parser.add_argument("--azimuth-deg", type=float, default=0.0)
    parser.add_argument("--course-deg", type=float, default=0.0)
    parser.add_argument("--radial-speed", type=float, default=0.0)
    parser.add_argument("--distance-radial", type=float, default=200.0)
    parser.add_argument("--x-guess", default="25,0,0,0,0")
    parser.add_argument("--bounds-lower", default="2,-15,-15,-15,-5")
    parser.add_argument("--bounds-upper", default="80,15,15,15,5")
    parser.add_argument("--moment-tolerance", type=float, default=1e-3)
    parser.add_argument("--include-gravity", action="store_true")
    parser.add_argument("--max-nfev", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory. Defaults to results/aerodynamics/<script>.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save plots without opening interactive windows.",
    )


def output_dir(args: argparse.Namespace, script_name: str) -> Path:
    path = (
        Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_ROOT / script_name
    )
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2), encoding="utf-8")
    print(f"Wrote {path}")


def save_figure(fig: Any, path: Path, *, dpi: int = 150) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"Wrote {path}")


def parsed_common(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "reference_point": csv_vector(
            args.reference_point, length=3, name="reference-point"
        ),
        "center_of_gravity": csv_vector(
            args.center_of_gravity, length=3, name="center-of-gravity"
        ),
        "x_guess": csv_vector(args.x_guess, length=5, name="x-guess"),
        "bounds_lower": csv_vector(args.bounds_lower, length=5, name="bounds-lower"),
        "bounds_upper": csv_vector(args.bounds_upper, length=5, name="bounds-upper"),
    }


def build_body(args: argparse.Namespace):
    add_vsm_path(args.vsm_src)
    from VSM.core.BodyAerodynamics import BodyAerodynamics

    return BodyAerodynamics.instantiate(
        n_panels=args.n_panels,
        file_path=Path(args.geometry_yaml),
        spanwise_panel_distribution=args.spanwise_panel_distribution,
        bridle_path=Path(args.bridle_path) if args.bridle_path else None,
    )


def build_system_model(args: argparse.Namespace):
    from awetrim.system.system_model import SystemModel
    from awetrim.system.tether import RigidLumpedTether

    system = SystemModel(tether=RigidLumpedTether(diameter=args.tether_diameter))
    system.mass_wing = args.mass_wing
    system.kite.mass_wing = args.mass_wing
    system.angle_elevation = np.deg2rad(args.elevation_deg)
    system.angle_azimuth = np.deg2rad(args.azimuth_deg)
    system.angle_course = np.deg2rad(args.course_deg)
    system.speed_radial = args.radial_speed
    system.distance_radial = args.distance_radial
    system.wind.speed_wind_ref = args.wind_speed
    system.wind.direction_wind = 0.0
    system.timeder_speed_tangential = 0.0
    system.timeder_speed_radial = 0.0
    system.timeder_angle_course = 0.0
    return system


def update_system_model(system: Any, case_values: dict[str, float]) -> None:
    if "wind_speed" in case_values:
        system.wind.speed_wind_ref = case_values["wind_speed"]
    if "elevation_deg" in case_values:
        system.angle_elevation = np.deg2rad(case_values["elevation_deg"])
    if "azimuth_deg" in case_values:
        system.angle_azimuth = np.deg2rad(case_values["azimuth_deg"])
    if "course_deg" in case_values:
        system.angle_course = np.deg2rad(case_values["course_deg"])
    if "radial_speed" in case_values:
        system.speed_radial = case_values["radial_speed"]
    if "distance_radial" in case_values:
        system.distance_radial = case_values["distance_radial"]


def to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if np.iscomplexobj(value):
            return to_jsonable(value.tolist())
        return value.tolist()
    if isinstance(value, np.generic):
        if np.iscomplexobj(value):
            return {"real": float(value.real), "imag": float(value.imag)}
        return value.item()
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, dict):
        return {
            key: to_jsonable(val) for key, val in value.items() if key != "optimizer"
        }
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    try:
        json.dumps(value)
    except TypeError:
        return str(value)
    return value


def print_trim_summary(result: dict[str, Any]) -> None:
    opt_x = np.asarray(result["opt_x"], dtype=float)
    cm = np.asarray(result["cm"], dtype=float)
    print("success_optimizer:", result["success"])
    print("success_physical:", result["success_physical"])
    print("speed_tangential [m/s]:", f"{opt_x[0]:.6g}")
    print("angle_roll_body_deg:", f"{opt_x[1]:.6g}")
    print("angle_pitch_body_deg:", f"{opt_x[2]:.6g}")
    print("angle_yaw_body_deg:", f"{opt_x[3]:.6g}")
    print("timeder_angle_course_body [rad/s]:", f"{opt_x[4]:.6g}")
    print("cm:", np.array2string(cm, precision=6))
    print("cfx/cfy:", f"{result['cfx']:.6g}", f"{result['cfy']:.6g}")
    print("aoa_center_deg:", f"{result['aoa_deg']:.6g}")
    print("side_slip_deg:", f"{result['side_slip_deg']:.6g}")
    print("cl/cd:", f"{float(result['cl']):.6g}", f"{float(result['cd']):.6g}")
