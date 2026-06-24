import math

import casadi as ca

from awetrim.system import create_system_model_from_yaml


def _write_config(path, wind):
    path.write_text(
        f"""
wing:
  mass: 15
  area: 19.75
  aerodynamics:
    model: coeffs
    params:
      CD0: 0.1
      CL0: 0.0
      angle_pitch_depower_0: -0.1
      delta_pitch_depower: -0.2
    coefficients:
      CL: []
      CD: []
      CS:
        - var: u_s
          power: 1
          coef: 0.15
kcu:
  mass: 10
tether:
  diameter: 0.01
wind:
{wind}
""",
        encoding="utf-8",
    )


def test_factory_builds_logarithmic_wind_from_yaml(tmp_path):
    config_path = tmp_path / "kite.yaml"
    _write_config(
        config_path,
        """  model: logarithmic
  z0: 0.03
  direction_wind: 0.25
  speed_wind_at_100: 10
""",
    )

    model = create_system_model_from_yaml(config_path)

    assert model.wind.wind_model == "logarithmic"
    assert model.wind.z0 == 0.03
    assert model.wind.direction_wind == 0.25
    assert math.isclose(
        float(model.wind.speed_friction),
        0.41 * 10 / math.log(100 / 0.03),
    )


def test_factory_builds_uniform_wind_from_yaml(tmp_path):
    config_path = tmp_path / "kite.yaml"
    _write_config(
        config_path,
        """  model: uniform
  direction_wind: 0
  speed_wind_ref: 8
""",
    )

    model = create_system_model_from_yaml(config_path)

    assert model.wind.wind_model == "uniform"
    assert model.wind.speed_wind_ref == 8


def test_system_model_exposes_named_expressions(tmp_path):
    config_path = tmp_path / "kite.yaml"
    _write_config(
        config_path,
        """  model: uniform
  direction_wind: 0
  speed_wind_ref: 8
""",
    )

    model = create_system_model_from_yaml(config_path)

    assert "angle_of_attack" in model.available_expressions()
    func = model.extract_function("angle_of_attack")
    assert isinstance(func, ca.Function)


# --- hardware actuator limits (system.yaml -> optimizer) --------------------


def test_extract_hardware_limits_maps_kcu_and_tether():
    from awetrim.system.factory import _extract_hardware_limits

    cs_struct = {
        "steering": {"range": [-0.35, 0.35], "rate": [-0.29, 0.29]},
        "depower": {"range": [1.1, 2.3], "rate": [-0.2, 0.2]},
    }
    tether_struct = {"length": 350.0}

    hw = _extract_hardware_limits(cs_struct, tether_struct)

    assert hw["input_steering"] == (-0.35, 0.35)
    assert hw["steering_rate"] == (-0.29, 0.29)
    assert hw["input_depower"] == (1.1, 2.3)
    assert hw["depower_rate"] == (-0.2, 0.2)
    assert hw["_max_tether_length"] == 350.0


def test_extract_hardware_limits_omits_missing_fields():
    from awetrim.system.factory import _extract_hardware_limits

    assert _extract_hardware_limits({}, {}) == {}


def test_legacy_config_yields_empty_hardware_limits(tmp_path):
    # The legacy format has no actuator-limit block -> the optimizer falls back
    # entirely to DEFAULT_OPTI_LIMITS.
    config_path = tmp_path / "kite.yaml"
    _write_config(config_path, "  model: uniform\n  direction_wind: 0\n  speed_wind_ref: 8\n")
    model = create_system_model_from_yaml(config_path)
    assert model.hardware_limits == {}


def test_resolve_opti_limits_merges_hardware_over_defaults():
    from awetrim.timeseries.phase_parametrized import PhaseParameterized
    from awetrim.utils.defaults import DEFAULT_OPTI_LIMITS

    class _Stub:
        kite_model = type(
            "_KM",
            (),
            {
                "hardware_limits": {
                    "input_steering": (-0.4, 0.4),
                    "_max_tether_length": 350.0,
                }
            },
        )()
        _resolve_opti_limits = PhaseParameterized._resolve_opti_limits

    lim = _Stub()._resolve_opti_limits()

    # hardware value overrides the default
    assert lim["input_steering"] == (-0.4, 0.4)
    # distance_radial upper bound = tether length, lower bound kept from default
    assert lim["distance_radial"] == (DEFAULT_OPTI_LIMITS["distance_radial"][0], 350.0)
    # the sentinel is consumed, not exposed as a limit
    assert "_max_tether_length" not in lim
    # untouched keys fall through to the defaults
    assert lim["height"] == DEFAULT_OPTI_LIMITS["height"]


def test_resolve_opti_limits_empty_is_exactly_defaults():
    from awetrim.timeseries.phase_parametrized import PhaseParameterized
    from awetrim.utils.defaults import DEFAULT_OPTI_LIMITS

    class _Stub:
        kite_model = type("_KM", (), {"hardware_limits": {}})()
        _resolve_opti_limits = PhaseParameterized._resolve_opti_limits

    assert _Stub()._resolve_opti_limits() == dict(DEFAULT_OPTI_LIMITS)
