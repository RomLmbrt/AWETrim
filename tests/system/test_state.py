"""Unit tests for awetrim.system.state.State.

State is a plain dataclass whose ``to_dict`` is the serialization contract the
TimeSeries analysis layer and the HDF5 writers depend on.
"""

import numpy as np
import pytest

from awetrim.system.state import State


def test_defaults_are_none():
    s = State()
    assert s.distance_radial is None
    assert s.tension_tether_ground is None
    assert s.elevation_last_element is None  # Williams-only field


def test_to_dict_roundtrips_through_constructor():
    s = State(
        t=1.0,
        s=0.5,
        s_dot=2.0,
        input_steering=0.1,
        tension_tether_ground=50000.0,
        distance_radial=200.0,
        speed_radial=0.2,
    )
    d = s.to_dict()
    assert isinstance(d, dict)
    assert d["t"] == 1.0
    assert d["tension_tether_ground"] == 50000.0
    # to_dict must be a faithful, reversible representation
    assert State(**d) == s


def test_to_dict_includes_all_fields_even_when_unset():
    d = State(t=0.0).to_dict()
    # Every declared field is present so downstream DataFrame/HDF5 schemas are stable.
    for field in ("angle_course", "speed_tangential", "lift_coefficient", "s_ddot"):
        assert field in d
    assert d["t"] == 0.0
    assert d["angle_course"] is None


def test_optional_array_fields_accepted():
    geom = np.zeros((4, 3))
    s = State(loaded_geometry=geom, lift_distribution=np.ones(4))
    d = s.to_dict()
    # asdict deep-copies, so compare by value/shape rather than identity.
    assert d["loaded_geometry"].shape == (4, 3)
    assert np.asarray(d["lift_distribution"]) == pytest.approx(np.ones(4))
