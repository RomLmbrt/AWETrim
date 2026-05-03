"""Tests for force and acceleration computations on Kite.

Covers properties tested here that are not in test_kite.py:
  - acceleration_rotation_course  (Coriolis/centripetal cross-product)
"""

import casadi as ca

from awetrim.system.kite import Kite


_AERO_INPUT = {"model": "coeffs", "params": {"CD0": 0.05}, "coefficients": {}}


def _make_kite() -> Kite:
    return Kite(mass_wing=10.0, area_wing=5.0, aero_input=_AERO_INPUT)


def test_acceleration_rotation_course_equals_cross_product():
    """acceleration_rotation_course = cross(velocity_rotation_course_frame, velocity_kite).

    When velocity_rotation_course_frame and velocity_kite are set to constant
    vectors, the property must match the cross product of those same vectors.
    """
    kite = _make_kite()

    omega = ca.vertcat(0.1, 0.2, 0.3)
    vk = ca.vertcat(1.0, 0.5, 0.2)

    kite.velocity_rotation_course_frame = omega
    kite.velocity_kite = vk

    expected = ca.cross(omega, vk)
    residual = ca.norm_2(kite.acceleration_rotation_course - expected)

    f = ca.Function("residual", [], [residual])
    result = f()
    value = list(result.values())[0] if isinstance(result, dict) else result[0]
    assert float(value) < 1e-10
