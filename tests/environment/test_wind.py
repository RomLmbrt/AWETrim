"""Unit tests for awetrim.environment.Wind

Tests verify the three wind-profile models and the frame transforms:
- uniform / logarithmic / tabulated speed profiles
- the friction-velocity <-> reference-speed relationship (and its inverse)
- world-frame wind vector structure and direction handling

Regression coverage: ``velocity_wind_at_height_W`` must be direction-aware.
A previous version dropped ``direction_wind`` and put all speed on +x; the
test ``test_velocity_wind_at_height_W_is_direction_aware`` locks in the fix.

Per AGENTS.md @tester role:
- Test CasADi expression structure/shapes; assert numeric values only where
  the profile is deterministic.
"""

import math
import types

import casadi as ca
import numpy as np
import pytest

from awetrim.environment.Wind import Wind


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def log_wind():
    """Logarithmic profile, 10 m/s at the 6 m reference height, wind along +x."""
    return Wind(wind_model="logarithmic", z0=0.1, direction_wind=0.0, speed_wind_ref=10.0)


@pytest.fixture
def uniform_wind():
    return Wind(wind_model="uniform", direction_wind=0.0, speed_wind_ref=8.0)


@pytest.fixture
def tabulated_wind():
    return Wind(
        wind_model="tabulated",
        tabulated_heights=[10.0, 100.0],
        tabulated_speeds=[5.0, 15.0],
        direction_wind=0.0,
    )


# ============================================================================
# SPEED PROFILES
# ============================================================================


class TestUniformProfile:
    def test_speed_independent_of_height(self, uniform_wind):
        assert float(uniform_wind.speed_wind(10.0)) == pytest.approx(8.0)
        assert float(uniform_wind.speed_wind(250.0)) == pytest.approx(8.0)


class TestLogarithmicProfile:
    def test_speed_equals_reference_at_reference_height(self, log_wind):
        # By construction the log law returns speed_wind_ref at height_ref (6 m).
        assert float(log_wind.speed_wind(log_wind.height_ref)) == pytest.approx(10.0)

    def test_speed_increases_with_height(self, log_wind):
        assert float(log_wind.speed_wind(50.0)) > float(log_wind.speed_wind(10.0))

    def test_speed_matches_log_law(self, log_wind):
        height = 80.0
        expected = (
            float(log_wind.speed_friction) / log_wind.kappa
        ) * math.log(height / log_wind.z0)
        assert float(log_wind.speed_wind(height)) == pytest.approx(expected)


class TestTabulatedProfile:
    def test_interpolates_at_nodes(self, tabulated_wind):
        assert float(tabulated_wind.speed_wind(10.0)) == pytest.approx(5.0)
        assert float(tabulated_wind.speed_wind(100.0)) == pytest.approx(15.0)

    def test_interpolates_at_midpoint(self, tabulated_wind):
        # Linear interpolant: halfway in height -> halfway in speed.
        assert float(tabulated_wind.speed_wind(55.0)) == pytest.approx(10.0)

    def test_requires_heights_and_speeds(self):
        with pytest.raises(ValueError, match="Tabulated wind model requires"):
            Wind(wind_model="tabulated", tabulated_heights=[10.0, 100.0])


# ============================================================================
# FRICTION VELOCITY <-> REFERENCE SPEED
# ============================================================================


class TestFrictionVelocityRelationship:
    def test_reference_speed_sets_friction_velocity(self, log_wind):
        expected = 10.0 * log_wind.kappa / math.log(log_wind.height_ref / log_wind.z0)
        assert float(log_wind.speed_friction) == pytest.approx(expected)

    def test_friction_setter_inverts_reference_speed(self):
        forward = Wind(wind_model="logarithmic", z0=0.1, direction_wind=0.0)
        forward.speed_wind_ref = 10.0
        friction = float(forward.speed_friction)

        inverse = Wind(wind_model="logarithmic", z0=0.1, direction_wind=0.0)
        inverse.speed_friction = friction
        assert float(inverse.speed_wind_ref_value) == pytest.approx(10.0)

    def test_reference_speed_value_numeric_when_set(self, log_wind):
        assert log_wind.speed_wind_ref_value == pytest.approx(10.0)

    def test_reference_speed_symbolic_when_unset(self):
        wind = Wind(wind_model="logarithmic", z0=0.1, direction_wind=0.0)
        assert wind.speed_wind_ref_value is None
        assert isinstance(wind.speed_wind_ref, ca.MX)


# ============================================================================
# WORLD-FRAME WIND VECTOR
# ============================================================================


class TestWorldFrameWindVector:
    def test_velocity_wind_W_shape_and_vertical_component(self, log_wind):
        vec = log_wind.velocity_wind_W(50.0)
        assert vec.shape == (3, 1)
        assert float(vec[2]) == pytest.approx(0.0)

    def test_velocity_wind_W_aligned_with_plus_x_for_zero_direction(self, log_wind):
        speed = float(log_wind.speed_wind(50.0))
        vec = log_wind.velocity_wind_W(50.0)
        assert float(vec[0]) == pytest.approx(speed)
        assert float(vec[1]) == pytest.approx(0.0)

    def test_velocity_wind_at_height_W_is_direction_aware(self):
        """Regression: a 90 deg wind direction must rotate speed onto +y.

        A prior bug dropped ``direction_wind`` here and put all speed on +x.
        """
        wind = Wind(
            wind_model="logarithmic",
            z0=0.1,
            direction_wind=math.pi / 2,
            speed_wind_ref=10.0,
        )
        speed = float(wind.speed_wind(50.0))
        vec = wind.velocity_wind_at_height_W(50.0)
        assert float(vec[0]) == pytest.approx(0.0, abs=1e-9)
        assert float(vec[1]) == pytest.approx(speed)
        assert float(vec[2]) == pytest.approx(0.0)

    def test_velocity_wind_at_height_W_matches_velocity_wind_W(self, log_wind):
        a = log_wind.velocity_wind_at_height_W(50.0)
        b = log_wind.velocity_wind_W(50.0)
        assert float(a[0]) == pytest.approx(float(b[0]))
        assert float(a[1]) == pytest.approx(float(b[1]))


# ============================================================================
# BODY-FRAME WIND VECTOR
# ============================================================================


class TestBodyFrameWindVector:
    def test_velocity_wind_returns_3x1(self, uniform_wind):
        model = types.SimpleNamespace(
            angle_azimuth=0.1, angle_elevation=0.5, angle_course=0.2, z=100.0
        )
        vec = uniform_wind.velocity_wind(model)
        assert vec.shape == (3, 1)
        assert np.all(np.isfinite(np.array(vec).ravel()))

    def test_velocity_wind_preserves_magnitude(self, uniform_wind):
        # The C<-W transform is a rotation, so it preserves the wind speed.
        model = types.SimpleNamespace(
            angle_azimuth=0.3, angle_elevation=0.7, angle_course=-0.4, z=120.0
        )
        vec = np.array(uniform_wind.velocity_wind(model)).ravel()
        assert np.linalg.norm(vec) == pytest.approx(8.0)
