# Copyright (c) 2023-2026 Oriol Cayon, Delft University of Technology
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""Polynomial identification and model selection for ROM aerodynamic coefficients.

This module fits the ROM ``model: "coeffs"`` aerodynamic polynomials (see
``data/<kite>/rom_config.yaml`` and ``awetrim.system.kite``) from a tidy dataset
of aerodynamic samples.  The same entry points are reused for two data sources:

* the high-fidelity quasi-steady aerostructural (QSM) solver
  (anchors + per-anchor frozen-geometry alpha sweeps), and
* EKF flight-data reconstructions,

so that an AS-identified model can be compared directly against an
experiment-identified one.

Targets and regressors (see the module plan):

    target  in {CL, CD, phi_a}     regressors: {alpha, u_s, u_p, v_a}

Conventions
-----------
* ``alpha`` and ``phi_a`` are in **radians** in the tidy dataset.
* A *term* is a monomial in the regressors, represented as a power map, e.g.
  ``{"alpha": 2}`` or ``{"alpha": 1, "u_s": 1}``.  The intercept (constant) is
  handled separately and serialised to ``params`` (``CL0`` / ``CD0`` / ``phi_a0``).
* ``CD`` uses an **absolute-value basis** because the ROM evaluates each CD term
  as ``coef * sqrt(value**2 + eps)`` (``kite.py``).  Fitting in the same basis
  makes the identified coefficients transfer 1:1 into ``rom_config.yaml``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations_with_replacement
from typing import Iterable, Mapping, Sequence

import numpy as np

# Canonical regressor names — must match the keys exposed by the ROM evaluation
# (`variables` dict in ``awetrim.system.kite.aerodynamic_force_coefficients_for``).
DEFAULT_REGRESSORS: tuple[str, ...] = ("alpha", "u_s", "u_p", "v_a")

# Map each target to its ROM intercept parameter name and whether it is fit in
# the absolute-value basis (CD only).
TARGET_INTERCEPT_PARAM = {"CL": "CL0", "CD": "CD0", "phi_a": "phi_a0"}
TARGET_ABS_BASIS = {"CL": False, "CD": True, "phi_a": False}

# Power map for a monomial term: variable name -> non-negative integer power.
PowerMap = Mapping[str, int]


# ──────────────────────────────────────────────────────────────────────────────
# Term library
# ──────────────────────────────────────────────────────────────────────────────
def _normalise_term(power_map: PowerMap) -> dict[str, int]:
    """Drop zero powers and return a clean ``{var: power}`` dict."""
    return {var: int(p) for var, p in power_map.items() if int(p) != 0}


def term_key(power_map: PowerMap) -> tuple[tuple[str, int], ...]:
    """Hashable canonical key for a term (sorted ``(var, power)`` pairs)."""
    return tuple(sorted(_normalise_term(power_map).items()))


def term_label(power_map: PowerMap) -> str:
    """Human-readable label, e.g. ``alpha^2*u_s``; ``"1"`` for the intercept."""
    norm = _normalise_term(power_map)
    if not norm:
        return "1"
    parts = []
    for var, p in sorted(norm.items()):
        parts.append(var if p == 1 else f"{var}^{p}")
    return "*".join(parts)


def generate_candidate_terms(
    regressors: Sequence[str] = DEFAULT_REGRESSORS,
    *,
    max_degree: int = 2,
    max_vars_per_term: int = 2,
    include_squared_interactions: Sequence[str] = (),
) -> list[dict[str, int]]:
    """Generate candidate monomial terms (excluding the intercept).

    Args:
        regressors: regressor variable names.
        max_degree: maximum total polynomial degree of a term.
        max_vars_per_term: maximum number of *distinct* variables in a term
            (1 = pure powers only; 2 = allow pairwise interactions; etc.).
        include_squared_interactions: for each named variable ``v`` (typically
            ``"alpha"``), also add the degree-3 terms ``v^2 * w`` for every other
            regressor ``w`` (i.e. ``alpha^2`` modulated by the rest). These are
            added regardless of ``max_degree`` so the dominant quadratic AoA
            response can be made actuation/speed dependent without inflating the
            whole degree-3 library.

    Returns:
        A de-duplicated list of power maps, ordered by ascending total degree
        then by name, with the empty (constant) term excluded.
    """
    seen: set[tuple[tuple[str, int], ...]] = set()
    terms: list[dict[str, int]] = []

    def _add(power_map: dict[str, int]) -> None:
        key = term_key(power_map)
        if key not in seen:
            seen.add(key)
            terms.append(_normalise_term(power_map))

    for degree in range(1, max_degree + 1):
        for combo in combinations_with_replacement(regressors, degree):
            power_map: dict[str, int] = {}
            for var in combo:
                power_map[var] = power_map.get(var, 0) + 1
            if len(power_map) > max_vars_per_term:
                continue
            _add(power_map)

    for sq_var in include_squared_interactions:
        if sq_var not in regressors:
            continue
        for other in regressors:
            if other == sq_var:
                continue
            _add({sq_var: 2, other: 1})

    terms.sort(key=lambda pm: (sum(pm.values()), term_label(pm)))
    return terms


# ──────────────────────────────────────────────────────────────────────────────
# Design matrix and least squares
# ──────────────────────────────────────────────────────────────────────────────
def term_column(data: Mapping[str, np.ndarray], power_map: PowerMap) -> np.ndarray:
    """Evaluate a single monomial column ``prod(var**power)`` over the dataset."""
    norm = _normalise_term(power_map)
    n = len(next(iter(data.values())))
    col = np.ones(n, dtype=float)
    for var, p in norm.items():
        col = col * np.asarray(data[var], dtype=float) ** p
    return col


def design_matrix(
    data: Mapping[str, np.ndarray],
    terms: Sequence[PowerMap],
    *,
    include_intercept: bool = True,
    abs_basis: bool = False,
) -> np.ndarray:
    """Assemble the regression design matrix.

    Args:
        data: mapping of regressor name -> 1-D array of samples.
        terms: monomial terms (power maps) for the non-intercept columns.
        include_intercept: prepend a constant column of ones.
        abs_basis: if True, take the absolute value of each *non-intercept*
            column (matches the ROM CD evaluation ``coef*|value|``).

    Returns:
        Array of shape ``(n_samples, n_columns)``; column 0 is the intercept
        when ``include_intercept`` is True.
    """
    cols: list[np.ndarray] = []
    n = len(next(iter(data.values())))
    if include_intercept:
        cols.append(np.ones(n, dtype=float))
    for power_map in terms:
        col = term_column(data, power_map)
        if abs_basis:
            col = np.abs(col)
        cols.append(col)
    if not cols:
        return np.empty((n, 0), dtype=float)
    return np.column_stack(cols)


def _solve_scaled(A: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Least-squares solve with per-column scaling for conditioning.

    Polynomial regressors span very different magnitudes (``alpha ~ 0.1`` rad,
    ``v_a ~ 20`` m/s, ``v_a**2 ~ 400``).  Scaling each column to unit RMS before
    the solve avoids an ill-conditioned normal-equations system; coefficients
    are mapped back to the raw (unscaled) basis so they serialise directly.
    """
    scale = np.sqrt(np.mean(A**2, axis=0))
    scale[scale == 0] = 1.0
    A_scaled = A / scale
    coeffs_scaled, *_ = np.linalg.lstsq(A_scaled, y, rcond=None)
    return coeffs_scaled / scale


def _metrics(y: np.ndarray, y_hat: np.ndarray, n_params: int) -> dict[str, float]:
    """RMSE, R^2 and BIC for a fit with ``n_params`` free parameters."""
    resid = y - y_hat
    n = len(y)
    sse = float(resid @ resid)
    rmse = float(np.sqrt(sse / n)) if n else float("nan")
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - sse / ss_tot if ss_tot > 0 else float("nan")
    # Gaussian-likelihood BIC: n*ln(SSE/n) + k*ln(n).
    sigma2 = sse / n if n else float("nan")
    bic = n * np.log(sigma2) + n_params * np.log(n) if (n and sigma2 > 0) else -np.inf
    return {"rmse": rmse, "r2": r2, "bic": float(bic), "sse": sse, "n": float(n)}


# ──────────────────────────────────────────────────────────────────────────────
# Fit result
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class PolynomialFit:
    """A fitted ROM coefficient polynomial.

    Attributes:
        target: ``"CL"``, ``"CD"`` or ``"phi_a"``.
        regressors: regressor names used.
        intercept: constant term (serialised to ``params`` as e.g. ``CL0``).
        terms: list of ``(power_map, coef)`` for the non-intercept monomials.
        abs_basis: whether columns are evaluated as ``|monomial|`` (CD).
        metrics: in-sample metrics (rmse, r2, bic, ...).
        cv_rmse: mean k-fold cross-validation RMSE (NaN if not computed).
    """

    target: str
    regressors: tuple[str, ...]
    intercept: float
    terms: list[tuple[dict[str, int], float]]
    abs_basis: bool
    metrics: dict[str, float] = field(default_factory=dict)
    cv_rmse: float = float("nan")

    def predict(self, data: Mapping[str, np.ndarray]) -> np.ndarray:
        """Evaluate the fitted polynomial, mirroring the ROM evaluation."""
        n = len(next(iter(data.values())))
        out = np.full(n, float(self.intercept), dtype=float)
        for power_map, coef in self.terms:
            col = term_column(data, power_map)
            if self.abs_basis:
                col = np.abs(col)
            out = out + coef * col
        return out


def fit_terms(
    data: Mapping[str, np.ndarray],
    y: np.ndarray,
    terms: Sequence[PowerMap],
    *,
    target: str,
    regressors: Sequence[str] = DEFAULT_REGRESSORS,
    abs_basis: bool | None = None,
) -> PolynomialFit:
    """Fit a fixed set of terms (plus intercept) by least squares."""
    if abs_basis is None:
        abs_basis = TARGET_ABS_BASIS.get(target, False)
    y = np.asarray(y, dtype=float)
    A = design_matrix(data, terms, include_intercept=True, abs_basis=abs_basis)
    coeffs = _solve_scaled(A, y)
    intercept = float(coeffs[0])
    fitted_terms = [
        (_normalise_term(pm), float(c)) for pm, c in zip(terms, coeffs[1:])
    ]
    fit = PolynomialFit(
        target=target,
        regressors=tuple(regressors),
        intercept=intercept,
        terms=fitted_terms,
        abs_basis=abs_basis,
        metrics=_metrics(y, A @ coeffs, A.shape[1]),
    )
    return fit


def _cv_rmse(
    data: Mapping[str, np.ndarray],
    y: np.ndarray,
    terms: Sequence[PowerMap],
    *,
    abs_basis: bool,
    folds: int,
    seed: int = 0,
) -> float:
    """K-fold cross-validation RMSE for a fixed term set."""
    n = len(y)
    if folds < 2 or n < folds:
        return float("nan")
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    fold_id = np.array_split(order, folds)
    errors: list[float] = []
    for k in range(folds):
        test_idx = fold_id[k]
        train_idx = np.concatenate([fold_id[j] for j in range(folds) if j != k])
        train = {var: np.asarray(data[var])[train_idx] for var in data}
        test = {var: np.asarray(data[var])[test_idx] for var in data}
        A_tr = design_matrix(train, terms, include_intercept=True, abs_basis=abs_basis)
        coeffs = _solve_scaled(A_tr, y[train_idx])
        A_te = design_matrix(test, terms, include_intercept=True, abs_basis=abs_basis)
        resid = y[test_idx] - A_te @ coeffs
        errors.append(float(np.sqrt(np.mean(resid**2))))
    return float(np.mean(errors))


def select_model(
    data: Mapping[str, np.ndarray],
    y: np.ndarray,
    *,
    target: str,
    candidate_terms: Sequence[PowerMap] | None = None,
    regressors: Sequence[str] = DEFAULT_REGRESSORS,
    max_degree: int = 2,
    max_vars_per_term: int = 2,
    include_squared_interactions: Sequence[str] = (),
    cv_folds: int = 5,
    cv_seed: int = 0,
    verbose: bool = False,
) -> PolynomialFit:
    """Forward-stepwise model selection minimising BIC.

    Starts from the intercept-only model and greedily adds the candidate term
    that most reduces the BIC, stopping when no remaining term improves it.
    The selected model's k-fold CV RMSE is recorded on the returned fit.

    Args:
        data: mapping regressor name -> 1-D array.
        y: target samples.
        target: ``"CL"``, ``"CD"`` or ``"phi_a"`` (sets the abs-basis flag).
        candidate_terms: explicit candidate library; if None, generated from
            ``regressors``/``max_degree``/``max_vars_per_term``.
        cv_folds: folds for the reported cross-validation RMSE.

    Returns:
        The selected :class:`PolynomialFit`.
    """
    abs_basis = TARGET_ABS_BASIS.get(target, False)
    y = np.asarray(y, dtype=float)
    if candidate_terms is None:
        candidate_terms = generate_candidate_terms(
            regressors,
            max_degree=max_degree,
            max_vars_per_term=max_vars_per_term,
            include_squared_interactions=include_squared_interactions,
        )
    remaining = [dict(t) for t in candidate_terms]
    selected: list[dict[str, int]] = []

    def bic_of(terms: Sequence[PowerMap]) -> float:
        A = design_matrix(data, terms, include_intercept=True, abs_basis=abs_basis)
        coeffs = _solve_scaled(A, y)
        return _metrics(y, A @ coeffs, A.shape[1])["bic"]

    best_bic = bic_of(selected)
    improved = True
    while improved and remaining:
        improved = False
        best_candidate = None
        best_candidate_bic = best_bic
        for term in remaining:
            trial_bic = bic_of(selected + [term])
            if trial_bic < best_candidate_bic - 1e-9:
                best_candidate_bic = trial_bic
                best_candidate = term
        if best_candidate is not None:
            selected.append(best_candidate)
            remaining = [t for t in remaining if term_key(t) != term_key(best_candidate)]
            best_bic = best_candidate_bic
            improved = True
            if verbose:
                print(
                    f"[{target}] + {term_label(best_candidate):<14} BIC={best_bic:.3f}"
                )

    fit = fit_terms(
        data, y, selected, target=target, regressors=regressors, abs_basis=abs_basis
    )
    fit.cv_rmse = _cv_rmse(
        data, y, selected, abs_basis=abs_basis, folds=cv_folds, seed=cv_seed
    )
    return fit


# ──────────────────────────────────────────────────────────────────────────────
# Serialisation to the rom_config "coeffs" format
# ──────────────────────────────────────────────────────────────────────────────
def _term_to_rom_entry(power_map: dict[str, int], coef: float) -> dict:
    """Serialise one monomial to a rom_config coefficient entry.

    Single-variable terms use the compact ``{var, power, coef}`` form; multi-
    variable terms use the ``{vars: {var: power, ...}, coef}`` product form.
    Both are supported by ``kite.aerodynamic_force_coefficients_for``.
    """
    norm = _normalise_term(power_map)
    if len(norm) == 1:
        (var, power), = norm.items()
        return {"var": var, "power": int(power), "coef": float(coef)}
    return {"vars": {var: int(p) for var, p in norm.items()}, "coef": float(coef)}


def build_rom_aerodynamics(
    fits: Iterable[PolynomialFit],
    *,
    extra_params: Mapping[str, float] | None = None,
) -> dict:
    """Assemble a ROM ``aerodynamics`` block from fitted polynomials.

    Args:
        fits: fitted polynomials for some/all of {CL, CD, phi_a}.
        extra_params: additional ``params`` entries to merge (e.g. legacy
            ``angle_pitch_depower_0``).

    Returns:
        A dict with ``{model, params, coefficients}`` ready to dump under the
        ``aerodynamics:`` key of a rom_config YAML.
    """
    params: dict[str, float] = {}
    coefficients: dict[str, list[dict]] = {}
    for fit in fits:
        intercept_name = TARGET_INTERCEPT_PARAM.get(fit.target, f"{fit.target}0")
        params[intercept_name] = float(fit.intercept)
        coefficients[fit.target] = [
            _term_to_rom_entry(pm, coef) for pm, coef in fit.terms
        ]
    if extra_params:
        params.update({k: float(v) for k, v in extra_params.items()})
    return {"model": "coeffs", "params": params, "coefficients": coefficients}
