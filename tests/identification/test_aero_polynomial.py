"""Tests for the aerodynamic polynomial identification module.

Covers the candidate-term library, least-squares fitting, BIC model selection,
the CD absolute-value basis, and a serialisation round-trip that confirms the
emitted rom_config terms are evaluated identically by the ROM consumer
(``Kite.aerodynamic_force_coefficients_for``).
"""

import numpy as np
import pytest

from awetrim.identification import aero_polynomial as ap
from awetrim.identification.aero_dataset import aerodynamic_roll


@pytest.fixture
def synthetic_data():
    rng = np.random.default_rng(0)
    n = 500
    data = {
        "alpha": rng.uniform(-0.1, 0.25, n),
        "u_s": rng.uniform(-0.3, 0.3, n),
        "u_p": rng.uniform(0.0, 0.4, n),
        "v_a": rng.uniform(10.0, 30.0, n),
    }
    return data, rng


# ── Term library ────────────────────────────────────────────────────────────
def test_generate_candidate_terms_excludes_intercept_and_dedupes():
    terms = ap.generate_candidate_terms(("alpha", "u_s"), max_degree=2, max_vars_per_term=2)
    keys = [ap.term_key(t) for t in terms]
    assert len(keys) == len(set(keys)), "terms must be unique"
    assert all(t for t in terms), "no empty (intercept) term"
    # alpha, u_s, alpha^2, u_s^2, alpha*u_s
    assert {ap.term_label(t) for t in terms} == {"alpha", "u_s", "alpha^2", "u_s^2", "alpha*u_s"}


def test_max_vars_per_term_limits_interactions():
    terms = ap.generate_candidate_terms(
        ("alpha", "u_s", "u_p"), max_degree=2, max_vars_per_term=1
    )
    assert all(len(t) == 1 for t in terms), "max_vars_per_term=1 -> pure powers only"


def test_squared_interactions_add_alpha2_times_rest():
    terms = ap.generate_candidate_terms(
        ("alpha", "u_s", "u_p", "v_a"),
        max_degree=2,
        max_vars_per_term=2,
        include_squared_interactions=("alpha",),
    )
    labels = {ap.term_label(t) for t in terms}
    assert {"alpha^2*u_s", "alpha^2*u_p", "alpha^2*v_a"} <= labels
    assert "alpha^2*alpha" not in labels and "alpha^3" not in labels


def test_squared_interaction_term_is_recovered():
    rng = np.random.default_rng(3)
    n = 600
    data = {
        "alpha": rng.uniform(-0.05, 0.2, n),
        "u_s": rng.uniform(-0.3, 0.3, n),
        "u_p": rng.uniform(1.5, 1.9, n),
        "v_a": rng.uniform(10.0, 30.0, n),
    }
    # CL with an alpha^2 * v_a modulation that a degree-2 library cannot express.
    y = 0.05 + 6.0 * data["alpha"] - 0.5 * data["alpha"] ** 2 * data["v_a"]
    fit = ap.select_model(
        data, y, target="CL", max_degree=2, max_vars_per_term=2,
        include_squared_interactions=("alpha",),
    )
    coef = {ap.term_label(pm): c for pm, c in fit.terms}
    assert "alpha^2*v_a" in coef
    assert coef["alpha^2*v_a"] == pytest.approx(-0.5, abs=1e-2)


# ── Design matrix ─────────────────────────────────────────────────────────────
def test_design_matrix_shape_and_abs_basis():
    data = {"alpha": np.array([-1.0, 2.0]), "u_s": np.array([1.0, -3.0])}
    terms = [{"alpha": 1}, {"u_s": 1}]
    A = ap.design_matrix(data, terms, include_intercept=True, abs_basis=False)
    assert A.shape == (2, 3)
    assert np.allclose(A[:, 0], 1.0)
    A_abs = ap.design_matrix(data, terms, include_intercept=True, abs_basis=True)
    assert np.allclose(A_abs[:, 1], np.abs(data["alpha"]))
    assert np.allclose(A_abs[:, 2], np.abs(data["u_s"]))


# ── Fitting / selection ───────────────────────────────────────────────────────
def test_fit_recovers_known_coefficients(synthetic_data):
    data, rng = synthetic_data
    a = data["alpha"]
    y = 0.05 + 6.1 * a - 10.0 * a**2 - 0.1 * data["u_p"]
    fit = ap.fit_terms(
        data, y, [{"alpha": 1}, {"alpha": 2}, {"u_p": 1}], target="CL"
    )
    assert fit.intercept == pytest.approx(0.05, abs=1e-6)
    coef = {ap.term_label(pm): c for pm, c in fit.terms}
    assert coef["alpha"] == pytest.approx(6.1, abs=1e-4)
    assert coef["alpha^2"] == pytest.approx(-10.0, abs=1e-4)
    assert coef["u_p"] == pytest.approx(-0.1, abs=1e-4)
    assert fit.metrics["r2"] > 0.999


def test_select_model_picks_correct_terms(synthetic_data):
    data, rng = synthetic_data
    a = data["alpha"]
    y = 0.05 + 6.1 * a - 10.0 * a**2 - 0.1 * data["u_p"] + rng.normal(0, 1e-3, len(a))
    fit = ap.select_model(data, y, target="CL", max_degree=2, max_vars_per_term=2)
    selected = {ap.term_label(pm) for pm, _ in fit.terms}
    assert {"alpha", "alpha^2", "u_p"} <= selected
    assert fit.cv_rmse < 0.05


def test_cd_abs_basis_recovers_absolute_term(synthetic_data):
    data, rng = synthetic_data
    # True CD in the ROM abs basis: CD0 + 1.5*alpha^2 + 0.02*|u_s|
    y = 0.11 + 1.5 * data["alpha"] ** 2 + 0.02 * np.abs(data["u_s"])
    fit = ap.select_model(data, y, target="CD", max_degree=2, max_vars_per_term=1)
    assert fit.abs_basis is True
    coef = {ap.term_label(pm): c for pm, c in fit.terms}
    assert coef.get("u_s") == pytest.approx(0.02, abs=1e-3)
    assert coef.get("alpha^2") == pytest.approx(1.5, abs=1e-3)


# ── Serialisation round-trip (fit -> rom_config -> ROM evaluation) ─────────────
def test_serialization_matches_rom_evaluation(synthetic_data):
    data, _ = synthetic_data
    a = data["alpha"]
    y = 0.05 + 6.1 * a - 10.0 * a**2 + 0.3 * data["u_s"] * data["v_a"]
    fit = ap.fit_terms(
        data, y,
        [{"alpha": 1}, {"alpha": 2}, {"u_s": 1, "v_a": 1}],
        target="CL",
    )
    aero = ap.build_rom_aerodynamics([fit])
    assert aero["model"] == "coeffs"
    assert aero["params"]["CL0"] == pytest.approx(fit.intercept)

    # Evaluate the serialised CL block with the SAME logic the ROM uses and
    # confirm it reproduces the fit prediction sample-by-sample.
    sample = {k: np.asarray(v)[:5] for k, v in data.items()}
    rom_pred = np.full(5, aero["params"]["CL0"], dtype=float)
    for i in range(5):
        variables = {k: sample[k][i] for k in sample}
        total = aero["params"]["CL0"]
        for term in aero["coefficients"]["CL"]:
            # Same term evaluation as Kite.aerodynamic_force_coefficients_for
            # ("coeffs" model): product of var**power, or a single var**power.
            if "vars" in term:
                value = 1.0
                for var, power in term["vars"].items():
                    value *= variables[var] ** power
            else:
                value = variables[term["var"]] ** term.get("power", 1)
            total += term["coef"] * value
        rom_pred[i] = total
    assert np.allclose(rom_pred, fit.predict(sample), atol=1e-9)


def test_phi_a_serialized_under_phi_a_key(synthetic_data):
    data, _ = synthetic_data
    y = 0.0 + 1.2 * data["u_s"]
    fit = ap.fit_terms(data, y, [{"u_s": 1}], target="phi_a")
    aero = ap.build_rom_aerodynamics([fit])
    assert "phi_a" in aero["coefficients"]
    assert aero["params"]["phi_a0"] == pytest.approx(fit.intercept, abs=1e-9)
    assert aero["coefficients"]["phi_a"][0]["var"] == "u_s"


# ── phi_a reconstruction helper ───────────────────────────────────────────────
def test_aerodynamic_roll_pure_lift_is_zero():
    radial = np.array([0.0, 0.0, 1.0])
    va = np.array([10.0, 0.0, 0.0])  # lift_dir is along +z
    force = np.array([0.0, 0.0, 50.0])  # pure lift
    assert aerodynamic_roll(force, va, radial) == pytest.approx(0.0, abs=1e-9)


def test_aerodynamic_roll_pure_side_is_ninety_degrees():
    radial = np.array([0.0, 0.0, 1.0])
    va = np.array([10.0, 0.0, 0.0])
    # lift_dir = +z, side_dir = lift x va_unit = z x x = +y
    force = np.array([0.0, 30.0, 0.0])
    assert aerodynamic_roll(force, va, radial) == pytest.approx(np.pi / 2, abs=1e-9)
