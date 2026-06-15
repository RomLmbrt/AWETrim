"""Unit tests for awetrim.utils.talmar_equations.

Pure numpy power formulas (no solver). Each expected value is computed
independently from the formula in the docstring so the test pins the maths,
not just "runs without error".

Physics reference: Cayon, van Deursen, Schmehl (2026) WES 11, 1097 (ROM).
"""

import numpy as np
import pytest

from awetrim.utils.talmar_equations import (
    compute_Rmin,
    compute_power,
    compute_power_analytical_talmar,
)


def test_compute_Rmin_matches_formula():
    m, rho, S, CL = 20.0, 1.225, 19.75, 0.8
    assert compute_Rmin(m, rho, S, CL) == pytest.approx(2 * m / (rho * S * CL))


def test_compute_Rmin_scales_inversely_with_CL():
    base = compute_Rmin(20.0, 1.225, 19.75, 0.8)
    half_cl = compute_Rmin(20.0, 1.225, 19.75, 0.4)
    assert half_cl == pytest.approx(2 * base)


def test_compute_power_large_radius_matches_formula():
    m, rho, S, CL, E, vw = 20.0, 1.225, 19.75, 0.8, 5.0, 10.0
    R = 1000.0  # >> Rmin, so the clamp term is ~1
    Rmin = compute_Rmin(m, rho, S, CL)
    term = max(1 - (Rmin / R) ** 2, 0.0)
    expected = (4 / 27) * CL * E**2 * term ** 1.5 * 0.5 * rho * S * vw**3
    assert compute_power(m, rho, S, CL, E, R, vw) == pytest.approx(expected)


def test_compute_power_clamps_to_zero_below_Rmin():
    """A radius below Rmin must give exactly zero power, not a complex number."""
    m, rho, S, CL, E, vw = 20.0, 1.225, 19.75, 0.8, 5.0, 10.0
    Rmin = compute_Rmin(m, rho, S, CL)
    P = compute_power(m, rho, S, CL, E, R=0.5 * Rmin, vw=vw)
    assert P == 0.0
    assert np.isreal(P)


def test_compute_power_scales_with_windspeed_cubed():
    args = dict(m=20.0, rho=1.225, S=19.75, CL=0.8, E=5.0, R=1000.0)
    p1 = compute_power(**args, vw=5.0)
    p2 = compute_power(**args, vw=10.0)
    assert p2 / p1 == pytest.approx(8.0)


def test_compute_power_analytical_talmar_matches_formula():
    f, gamma, rho, S, CL, E, vw, lambd = 0.6, 0.0, 1.225, 19.75, 0.8, 5.0, 10.0, 4.0
    fx = f / np.cos(gamma)
    prefactor = fx / (1 - fx) * (rho * S * CL) / (2 * E)
    bracket = (lambd**-2 + (1 - fx) ** 2) ** 1.5
    expected = prefactor * vw**3 * bracket
    got = compute_power_analytical_talmar(f, gamma, rho, S, CL, E, vw, lambd)
    assert got == pytest.approx(expected)


def test_compute_power_analytical_talmar_elevation_increases_fx():
    """Non-zero elevation gamma raises fx = f / cos(gamma) vs the gamma=0 case."""
    base = compute_power_analytical_talmar(0.6, 0.0, 1.225, 19.75, 0.8, 5.0, 10.0, 4.0)
    tilted = compute_power_analytical_talmar(
        0.6, np.radians(30), 1.225, 19.75, 0.8, 5.0, 10.0, 4.0
    )
    assert tilted != pytest.approx(base)
