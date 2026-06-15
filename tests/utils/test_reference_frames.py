"""Unit tests for awetrim.utils.reference_frames

Every public transform builds a proper rotation matrix (CasADi 3x3). Tests
verify, for each transform:
- shape (3, 3)
- orthonormality (R @ R.T == I) and proper-rotation determinant (+1)
- composition / inverse relationships between the named frames
- a few exact values at zero angles
- symbolic MX structure is preserved

Per AGENTS.md @tester role: assert symbolic structure/shape, plus the
deterministic algebraic identities that define these matrices.
"""

import math

import casadi as ca
import numpy as np
import pytest

from awetrim.utils.reference_frames import (
    transformation_AZR_from_W,
    transformation_C_from_AZR,
    transformation_C_from_A,
    transformation_C_from_K,
    transformation_C_from_W,
    transformation_Wind_from_W,
    transformation_Wind_from_C,
    transformation_C_from_Wind,
)

I3 = np.eye(3)

# Sample angle triples (radians), incl. negatives and values past pi/2.
SAMPLE_ANGLES = [
    (0.0, 0.0, 0.0),
    (0.3, 0.7, -0.4),
    (-1.2, 0.5, 2.1),
    (math.pi / 2, -math.pi / 3, math.pi / 6),
    (2.5, -2.0, 1.0),
]


def _np(mat):
    """Evaluate a numeric CasADi matrix to a NumPy array."""
    return np.array(ca.DM(mat))


# ============================================================================
# SHAPES
# ============================================================================


class TestShapes:
    def test_azr_from_w_is_3x3(self):
        assert transformation_AZR_from_W(0.2, 0.3).shape == (3, 3)

    def test_c_from_azr_is_3x3(self):
        assert transformation_C_from_AZR(0.4).shape == (3, 3)

    def test_c_from_a_is_3x3(self):
        assert transformation_C_from_A(0.1, 0.2, 0.3).shape == (3, 3)

    def test_c_from_k_is_3x3(self):
        assert transformation_C_from_K(0.1, 0.2, 0.3).shape == (3, 3)

    def test_c_from_w_is_3x3(self):
        assert transformation_C_from_W(0.1, 0.2, 0.3).shape == (3, 3)

    def test_wind_from_w_is_3x3(self):
        assert transformation_Wind_from_W(0.5).shape == (3, 3)

    def test_wind_from_c_is_3x3(self):
        assert transformation_Wind_from_C(0.1, 0.2, 0.3, 0.4).shape == (3, 3)

    def test_c_from_wind_is_3x3(self):
        assert transformation_C_from_Wind(0.1, 0.2, 0.3, 0.4).shape == (3, 3)


# ============================================================================
# ORTHONORMALITY & PROPER-ROTATION DETERMINANT
# ============================================================================


class TestOrthonormality:
    """Each transform is a proper rotation: R R^T = I and det(R) = +1."""

    @pytest.mark.parametrize("a, b, c", SAMPLE_ANGLES)
    def test_azr_from_w_orthonormal(self, a, b, c):
        R = _np(transformation_AZR_from_W(a, b))
        assert np.allclose(R @ R.T, I3, atol=1e-12)
        assert np.linalg.det(R) == pytest.approx(1.0)

    @pytest.mark.parametrize("a, b, c", SAMPLE_ANGLES)
    def test_c_from_azr_orthonormal(self, a, b, c):
        R = _np(transformation_C_from_AZR(c))
        assert np.allclose(R @ R.T, I3, atol=1e-12)
        assert np.linalg.det(R) == pytest.approx(1.0)

    @pytest.mark.parametrize("a, b, c", SAMPLE_ANGLES)
    def test_c_from_a_orthonormal(self, a, b, c):
        R = _np(transformation_C_from_A(a, b, c))
        assert np.allclose(R @ R.T, I3, atol=1e-12)
        assert np.linalg.det(R) == pytest.approx(1.0)

    @pytest.mark.parametrize("a, b, c", SAMPLE_ANGLES)
    def test_c_from_k_orthonormal(self, a, b, c):
        R = _np(transformation_C_from_K(a, b, c))
        assert np.allclose(R @ R.T, I3, atol=1e-12)
        assert np.linalg.det(R) == pytest.approx(1.0)

    @pytest.mark.parametrize("a, b, c", SAMPLE_ANGLES)
    def test_c_from_w_orthonormal(self, a, b, c):
        R = _np(transformation_C_from_W(a, b, c))
        assert np.allclose(R @ R.T, I3, atol=1e-12)
        assert np.linalg.det(R) == pytest.approx(1.0)

    @pytest.mark.parametrize("a, b, c", SAMPLE_ANGLES)
    def test_wind_from_w_orthonormal(self, a, b, c):
        R = _np(transformation_Wind_from_W(a))
        assert np.allclose(R @ R.T, I3, atol=1e-12)
        assert np.linalg.det(R) == pytest.approx(1.0)


# ============================================================================
# COMPOSITION & INVERSE RELATIONSHIPS
# ============================================================================


class TestComposition:
    @pytest.mark.parametrize("a, b, c", SAMPLE_ANGLES)
    def test_c_from_w_is_product_of_pieces(self, a, b, c):
        # transformation_C_from_W = C_from_AZR(course) @ AZR_from_W(az, el)
        composed = _np(transformation_C_from_AZR(c) @ transformation_AZR_from_W(a, b))
        direct = _np(transformation_C_from_W(a, b, c))
        assert np.allclose(composed, direct, atol=1e-12)

    @pytest.mark.parametrize("a, b, c", SAMPLE_ANGLES)
    def test_wind_from_c_definition(self, a, b, c):
        d = 0.35
        expected = _np(
            transformation_Wind_from_W(d) @ _np(transformation_C_from_W(a, b, c)).T
        )
        direct = _np(transformation_Wind_from_C(a, b, c, d))
        assert np.allclose(expected, direct, atol=1e-12)

    @pytest.mark.parametrize("a, b, c", SAMPLE_ANGLES)
    def test_c_from_wind_is_transpose_of_wind_from_c(self, a, b, c):
        d = -0.6
        cw = _np(transformation_C_from_Wind(a, b, c, d))
        wc = _np(transformation_Wind_from_C(a, b, c, d))
        assert np.allclose(cw, wc.T, atol=1e-12)

    @pytest.mark.parametrize("a, b, c", SAMPLE_ANGLES)
    def test_c_from_wind_inverts_wind_from_c(self, a, b, c):
        d = 1.1
        cw = _np(transformation_C_from_Wind(a, b, c, d))
        wc = _np(transformation_Wind_from_C(a, b, c, d))
        assert np.allclose(cw @ wc, I3, atol=1e-12)


# ============================================================================
# EXACT VALUES AT ZERO ANGLES
# ============================================================================


class TestKnownValues:
    def test_wind_from_w_zero_is_identity(self):
        assert np.allclose(_np(transformation_Wind_from_W(0.0)), I3, atol=1e-15)

    def test_c_from_a_zero_is_identity(self):
        assert np.allclose(_np(transformation_C_from_A(0.0, 0.0, 0.0)), I3, atol=1e-15)

    def test_c_from_k_zero_is_identity(self):
        assert np.allclose(_np(transformation_C_from_K(0.0, 0.0, 0.0)), I3, atol=1e-15)

    def test_c_from_k_default_yaw(self):
        # yaw defaults to 0 -> identical to the explicit-zero-yaw call.
        default = _np(transformation_C_from_K(0.2, 0.3))
        explicit = _np(transformation_C_from_K(0.2, 0.3, 0.0))
        assert np.allclose(default, explicit, atol=1e-15)

    def test_c_from_azr_zero(self):
        expected = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        assert np.allclose(_np(transformation_C_from_AZR(0.0)), expected, atol=1e-15)

    def test_azr_from_w_zero(self):
        expected = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
        assert np.allclose(_np(transformation_AZR_from_W(0.0, 0.0)), expected, atol=1e-15)


# ============================================================================
# SYMBOLIC STRUCTURE
# ============================================================================


class TestSymbolicStructure:
    def test_azr_from_w_symbolic_is_mx(self):
        R = transformation_AZR_from_W(ca.MX.sym("az"), ca.MX.sym("el"))
        assert isinstance(R, ca.MX)
        assert R.shape == (3, 3)

    def test_c_from_w_symbolic_is_mx(self):
        R = transformation_C_from_W(
            ca.MX.sym("az"), ca.MX.sym("el"), ca.MX.sym("chi")
        )
        assert isinstance(R, ca.MX)
        assert R.shape == (3, 3)

    def test_wind_from_c_symbolic_is_mx(self):
        R = transformation_Wind_from_C(
            ca.MX.sym("az"),
            ca.MX.sym("el"),
            ca.MX.sym("chi"),
            ca.MX.sym("dir"),
        )
        assert isinstance(R, ca.MX)
        assert R.shape == (3, 3)
