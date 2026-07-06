"""Unit tests for ``reduced_hessian_metrics`` in
:mod:`awetrim.utils.sparsity_analysis`.

The reduced Hessian (Lagrangian Hessian projected onto the active-constraint
nullspace) is the convergence-relevant curvature for a trajectory NLP, where
the full Hessian is singular by construction. These tests pin its behaviour on
small problems with a hand-computable answer -- no solver involved.
"""

import numpy as np

from awetrim.utils.sparsity_analysis import reduced_hessian_metrics


def test_single_active_equality_projects_out_one_direction():
    # H = diag(1, 10, 100); one active constraint A = [1, 0, 0] removes the
    # x0 direction, leaving reduced curvature diag(10, 100): cond = 10, SPD.
    HLag = np.diag([1.0, 10.0, 100.0])
    Jg = np.array([[1.0, 0.0, 0.0]])
    lam = np.array([1.0])

    m = reduced_hessian_metrics(Jg, HLag, lam)

    assert m["n_active"] == 1
    assert m["rank"] == 1
    assert m["n_free"] == 2
    assert np.isclose(m["cond"], 10.0)
    assert np.isclose(m["eig_min"], 10.0)
    assert np.isclose(m["eig_max"], 100.0)
    assert (m["n_neg"], m["n_zero"], m["n_pos"]) == (0, 0, 2)


def test_inactive_multiplier_drops_constraint_from_active_set():
    # Same constraint but with a ~0 multiplier -> inactive -> nullspace is the
    # full space, so the reduced Hessian is the full H (cond = 100).
    HLag = np.diag([1.0, 10.0, 100.0])
    Jg = np.array([[1.0, 0.0, 0.0]])
    lam = np.array([0.0])

    m = reduced_hessian_metrics(Jg, HLag, lam)

    assert m["n_active"] == 0
    assert m["n_free"] == 3
    assert np.isclose(m["cond"], 100.0)


def test_negative_curvature_shows_in_inertia():
    # A saddle: reduced Hessian has a negative eigenvalue -> n_neg = 1.
    HLag = np.diag([-5.0, 2.0, 100.0])
    Jg = np.array([[0.0, 0.0, 1.0]])  # active row removes x2
    lam = np.array([3.0])

    m = reduced_hessian_metrics(Jg, HLag, lam)

    assert m["n_free"] == 2
    assert m["n_neg"] == 1
    assert m["n_pos"] == 1
    assert m["n_zero"] == 0


def test_flat_direction_counts_as_zero_eigenvalue():
    # Reduced curvature diag(0, 50): one flat direction -> n_zero = 1, and the
    # reported nonzero eig range is just {50}.
    HLag = np.diag([7.0, 0.0, 50.0])
    Jg = np.array([[1.0, 0.0, 0.0]])  # removes x0 (the curved-but-constrained dir)
    lam = np.array([2.0])

    m = reduced_hessian_metrics(Jg, HLag, lam)

    assert m["n_free"] == 2
    assert m["n_zero"] == 1
    assert m["n_pos"] == 1
    assert np.isclose(m["eig_max"], 50.0)


def test_dependent_active_rows_flag_licq_failure():
    # Two identical active rows -> rank 1 < n_active 2 (LICQ fails) and the
    # smallest active singular value is ~0.
    HLag = np.diag([1.0, 4.0, 9.0])
    Jg = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    lam = np.array([1.0, 1.0])

    m = reduced_hessian_metrics(Jg, HLag, lam)

    assert m["n_active"] == 2
    assert m["rank"] == 1
    assert m["sigma_min_active"] < 1e-9
    # Only x0 is constrained, so x1, x2 remain free: curvature diag(4, 9).
    assert m["n_free"] == 2
    assert np.isclose(m["cond"], 9.0 / 4.0)


def test_simple_bound_rows_are_counted():
    # Two active rows: one simple bound (single nonzero) and one coupling row
    # (two nonzeros). Only the first counts as a simple variable bound.
    HLag = np.diag([1.0, 2.0, 3.0])
    Jg = np.array(
        [
            [1.0, 0.0, 0.0],  # simple bound on x0
            [0.0, 1.0, 1.0],  # coupling between x1, x2
        ]
    )
    lam = np.array([1.0, 1.0])

    m = reduced_hessian_metrics(Jg, HLag, lam)

    assert m["n_active"] == 2
    assert m["n_simple_bounds"] == 1


def test_fully_determined_returns_zero_free_directions():
    # Three independent active rows in R^3 -> nullspace empty.
    HLag = np.diag([1.0, 2.0, 3.0])
    Jg = np.eye(3)
    lam = np.array([1.0, 1.0, 1.0])

    m = reduced_hessian_metrics(Jg, HLag, lam)

    assert m["n_active"] == 3
    assert m["rank"] == 3
    assert m["n_free"] == 0
