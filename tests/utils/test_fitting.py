"""Unit tests for awetrim.utils.fitting.

Weighted least-squares helpers are pure numpy. Tests use exactly-linear data
so the recovered coefficients and residual error are known in closed form.
"""

import numpy as np
import pytest

from awetrim.utils.fitting import (
    compute_weighted_least_squares,
    construct_A_matrix,
    fit_and_evaluate_model,
)


def test_wls_recovers_exact_coefficients():
    A = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    x_true = np.array([2.0, 3.0])
    y = A @ x_true
    x_hat = compute_weighted_least_squares(y, A)
    assert x_hat == pytest.approx(x_true)


def test_wls_with_identity_weights_equals_unweighted():
    A = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    y = np.array([2.0, 3.0, 5.0])
    W = np.eye(3)
    assert compute_weighted_least_squares(y, A, W) == pytest.approx(
        compute_weighted_least_squares(y, A)
    )


def test_wls_weights_bias_toward_emphasised_row():
    # Inconsistent system: heavily weighting the third row pulls the fit toward it.
    A = np.array([[1.0], [1.0], [1.0]])
    y = np.array([0.0, 0.0, 10.0])
    unweighted = compute_weighted_least_squares(y, A)[0]
    W = np.diag([1.0, 1.0, 1000.0])
    weighted = compute_weighted_least_squares(y, A, W)[0]
    assert unweighted == pytest.approx(10.0 / 3.0)
    assert weighted > 9.0


def test_construct_A_matrix_shape_and_values():
    alpha = np.array([1.0, 2.0, 3.0])
    A = construct_A_matrix(["alpha", "alpha**2", "np.ones_like(alpha)"], alpha=alpha)
    assert A.shape == (3, 3)
    assert A[:, 0] == pytest.approx(alpha)
    assert A[:, 1] == pytest.approx(alpha**2)
    assert A[:, 2] == pytest.approx(np.ones(3))


def test_fit_and_evaluate_model_exact_fit_zero_mse():
    alpha = np.linspace(0.0, 1.0, 10)
    data = 2.0 + 3.0 * alpha  # exactly representable by the dependencies below
    result = fit_and_evaluate_model(
        data, ["np.ones_like(alpha)", "alpha"], alpha=alpha
    )
    assert result["coeffs"] == pytest.approx([2.0, 3.0])
    assert result["MSE"] == pytest.approx(0.0, abs=1e-20)
    assert result["data_est"] == pytest.approx(data)
