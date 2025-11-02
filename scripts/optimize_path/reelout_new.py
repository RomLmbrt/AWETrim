import json
from pathlib import Path

import numpy as np

from awetrim import SystemModel
from awetrim.environment.Wind import Wind
from awetrim.system.kite import Kite
from awetrim.system.tether import RigidLumpedTether
from awetrim.timeseries.reelout_phase import ReeloutSimple

# ---------------------------------------------------------------------------
# Configuration knobs – tweak these values to experiment with the setup.
# ---------------------------------------------------------------------------
PHYSICAL_CONFIG = {
    "mass_wing": 61,
    "mass_kcu": 30,
    "area_wing": 46.85,
    "tether_diameter": 0.014,
}

PATH_PARAMETERS = {
    "omega": 1.0,
    "r0": 230.0,
    "az_amp0": 0.4785941041623598,
    "beta_amp0": 0.08726648368043392,
    "width_phi": 0.5,
    "width_beta": 0.5,
    "left_first": True,
    "normalize_bumps": False,
    "repeat_phi": True,
    "repeat_beta": True,
    "beta_coeffs": np.array(
        [0.26689736, -0.99999995, 0.04902545, -0.84708337, 0.4426069]
    ),
    "az_coeffs": [0, 0, 0, 0, 0],
    "kbeta": 0,
    "beta0": 0.45030399611963495,
    "kappa": 0,
    "distance_radial_start": 220,
}

RADIAL_PARAMETERS = {
    "reeling_strategy": "force",  # "force" or "constant"
    "force_model": "quadratic",  # "linear" or "quadratic"
    "reeling_speed": 0.0,  # m/s, only for constant reeling
    "max_tether_force": 4000,  # N, only for force reeling
    "min_tether_force": 25000,  # N, only for force reeling
    "softplus": True,
    "softplus_beta": 5e-5,
    "softminus": True,
    "softminus_beta": 1e-3,
    "slope": 2700,  # N/(m/s)^2 for quadratic, N/(m/s) for linear
    "offset": 0,  # m/s
}

SIM_PARAMETERS = {
    "start_time": 0,
    "end_time": 35,
    "start_angle": np.pi / 2,
    "end_angle": 2 * np.pi + np.pi / 2,
    "n_points": 300,
}

REELOUT_CONFIG = {
    "pattern_type": "cst_lissajous",
    "path_parameters": PATH_PARAMETERS,
    "radial_parameters": RADIAL_PARAMETERS,
    "sim_parameters": SIM_PARAMETERS,
}

AERO_INPUT_FILE = Path("data/LEI-V9-KITE/v9_aero_input.json")


def load_aero_input(path: Path = AERO_INPUT_FILE):
    """Load aerodynamic input data from disk."""
    with path.open("r") as file:
        return json.load(file)


def build_wind_model(speed_wind_at_100=10, z0=0.0002, model_type="uniform"):
    """Create a wind model using the supplied parameters."""
    wind_model = Wind(
        wind_model=model_type,
        z0=z0,
    )
    speed_friction = 0.41 * speed_wind_at_100 / np.log(100 / wind_model.z0)
    wind_model.speed_friction = speed_friction
    wind_model.speed_wind_ref = speed_wind_at_100
    return wind_model


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


def create_system_model():
    """Assemble the system model using the configuration dictionaries above."""
    aero_input = load_aero_input()
    wind_model = build_wind_model()
    return define_system(
        tether_diameter=PHYSICAL_CONFIG["tether_diameter"],
        mass_wing=PHYSICAL_CONFIG["mass_wing"],
        mass_kcu=PHYSICAL_CONFIG["mass_kcu"],
        area_wing=PHYSICAL_CONFIG["area_wing"],
        aero_input=aero_input,
        wind_model=wind_model,
    )


def main(run_plots=False):
    system_model = create_system_model()
    reelout = ReeloutSimple(
        system_model=system_model,
        pattern_config=REELOUT_CONFIG,
        depower=0,
    )
    optimization_params = [
        "az_amp0",
        "beta_amp0",
        "beta0",
        # "beta_coeffs",
        # "kappa",
        # "slope",
        # "offset",
    ]
    solution = reelout.run_simulation_opti(optimization_params=optimization_params)
    reelout.run_simulation(solution=solution, run_plots=run_plots)
    return reelout


if __name__ == "__main__":
    main(run_plots=True)
