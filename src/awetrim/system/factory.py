from pathlib import Path
import math
from typing import Union

import yaml

from awetrim.environment.Wind import Wind
from awetrim.system.kite import Kite
from awetrim.system.system_model import SystemModel
from awetrim.system.tether import RigidLumpedTether


def create_wind_model_from_config(wind_cfg):
    """Create a Wind model from a YAML wind section."""
    if not wind_cfg:
        return None

    wind_model = Wind(
        wind_model=wind_cfg.get("model", wind_cfg.get("model_type", "uniform")),
        z0=wind_cfg.get("z0", 0.01),
        tabulated_heights=wind_cfg.get("tabulated_heights"),
        tabulated_speeds=wind_cfg.get("tabulated_speeds"),
        direction_wind=wind_cfg.get("direction_wind", 0),
        speed_wind_ref=wind_cfg.get("speed_wind_ref"),
    )

    if "speed_friction" in wind_cfg:
        wind_model.speed_friction = wind_cfg["speed_friction"]
    elif "speed_wind_at_100" in wind_cfg:
        if wind_model.wind_model == "uniform":
            wind_model.speed_wind_ref = wind_cfg["speed_wind_at_100"]
        else:
            wind_model.speed_friction = (
                wind_model.kappa
                * wind_cfg["speed_wind_at_100"]
                / math.log(100 / wind_model.z0)
            )
    elif "speed_wind_at_ref" in wind_cfg:
        height_ref = wind_cfg.get("height_ref", wind_model.height_ref)
        wind_model.height_ref = height_ref
        if wind_model.wind_model == "uniform":
            wind_model.speed_wind_ref = wind_cfg["speed_wind_at_ref"]
        else:
            wind_model.speed_friction = (
                wind_model.kappa
                * wind_cfg["speed_wind_at_ref"]
                / math.log(height_ref / wind_model.z0)
            )

    return wind_model


def create_system_model_from_yaml(
    yaml_path: Union[str, Path], steering_control: str = "asymmetric"
):
    """Create a SystemModel from a YAML configuration.

    Expects a YAML file structured like `data/LEI-V3-KITE/lei_v3_system_config.yaml` with sections:
    - physical: { model_name, mass, area, span }
    - kcu: { mass, ... } (optional)
    - aerodynamics: { model, params, coefficients }
    - tether: { diameter, ... } (optional)
    - wind: { model, z0, speed_ref } (optional)
    """

    config_path = Path(yaml_path)
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    wing = cfg.get("wing", {})
    kcu_cfg = cfg.get("kcu", {})
    aero_cfg = wing.get("aerodynamics", {})
    tether_cfg = cfg.get("tether", {})
    wind_cfg = cfg.get("wind", {})

    tether_diameter = tether_cfg.get("diameter", 0.006)
    tether = RigidLumpedTether(diameter=tether_diameter)

    kite = Kite(
        mass_wing=wing.get("mass", 20),
        mass_kcu=kcu_cfg.get("mass", 0),
        area_wing=wing.get("area", 20),
        aero_input=aero_cfg,
        steering_control=steering_control,
    )

    return SystemModel(
        dof=3,
        kite=kite,
        tether=tether,
        wind_model=create_wind_model_from_config(wind_cfg),
    )
