import json
from pathlib import Path

import numpy as np

from awetrim import SystemModel
from awetrim.environment.Wind import Wind
from awetrim.system.kite import Kite
from awetrim.system.tether import RigidLumpedTether
from awetrim.timeseries.reelin_phase import ReelinSimple

AGGREGATED_RESULTS = None

# ---------------------------------------------------------------------------
# Configuration knobs – tweak these values to experiment with the setup.
# ---------------------------------------------------------------------------
PHYSICAL_CONFIG = {
    "mass_wing": 61,
    "mass_kcu": 30,
    "area_wing": 46.85,
    "tether_diameter": 0.014,
}

REELIN_CONFIG = {
    "beta0_deg": 30,
    "distance_radial_start": 360,
    "distance_radial_end": 220,
    "depower": 1.0,
}

RADIAL_PARAMETERS = {
    "reeling_strategy": "force",
    "force_model": "quadratic",
    "reeling_speed": 1.0,
    "max_tether_force": 2e4,
    "min_tether_force": 5000.0,
    "softplus": True,
    "softplus_beta": 1e-4,
    "softminus": True,
    "softminus_beta": 1e-3,
    "slope": 2716,
    "offset": -3,
}

AERO_INPUT_FILE = Path("data/LEI-V9-KITE/v9_aero_input.json")


def load_aero_input(path: Path = AERO_INPUT_FILE):
    """Load aerodynamic input data from disk."""
    with path.open("r") as file:
        return json.load(file)


def build_wind_model(speed_wind_at_100=7.6374, z0=0.0002):
    """Create a logarithmic wind model using the supplied parameters."""
    wind_model = Wind(
        wind_model="logarithmic",
        z0=z0,
    )
    speed_friction = 0.41 * speed_wind_at_100 / np.log(100 / wind_model.z0)
    wind_model.speed_friction = speed_friction
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


def create_runner():
    """Create a ReelinSimple runner with the configured system and parameters."""
    system_model = create_system_model()
    beta0 = np.radians(REELIN_CONFIG["beta0_deg"])
    return ReelinSimple(
        system_model=system_model,
        beta0=beta0,
        distance_radial_start=REELIN_CONFIG["distance_radial_start"],
        distance_radial_end=REELIN_CONFIG["distance_radial_end"],
        radial_parameters=RADIAL_PARAMETERS.copy(),
        depower=REELIN_CONFIG["depower"],
    )


def main(run_plots=False):
    runner = create_runner()
    solution = runner.run_opti()
    runner.run_simulation(solution=solution, run_plots=run_plots)
    return runner


def get_results(run_if_needed=True):
    global AGGREGATED_RESULTS
    if run_if_needed and AGGREGATED_RESULTS is None:
        runner = create_runner()
        solution = runner.run_opti()
        AGGREGATED_RESULTS = runner.run_simulation(solution=solution)
    return AGGREGATED_RESULTS


if __name__ == "__main__":
    main(run_plots=True)
