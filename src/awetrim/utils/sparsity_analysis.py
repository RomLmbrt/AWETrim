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

import numpy as np
import casadi as ca


def reduced_hessian_metrics(Jg0, HLag0, lam0, active_rtol=1e-6, zero_rtol=1e-8):
    """Curvature of the Lagrangian on the active-constraint nullspace.

    The full Hessian of a trajectory NLP is singular by construction: the
    objective depends explicitly on only a few variables, so every per-node
    state/control direction has zero objective curvature and enters only through
    the constraints. That makes ``cond(H_f)`` / ``cond(H_Lag)`` trivially
    infinite and non-diagnostic. The quantity that governs convergence is the
    Hessian projected onto the nullspace of the *active* constraint Jacobian --
    the curvature IPOPT actually sees along its free directions (the reduced
    Hessian). A strict local minimum has a positive-definite reduced Hessian;
    flat (near-zero) reduced directions are exactly what produce a slow,
    grind-to-``max_iter`` solve on a flat objective.

    Active set: a constraint row is treated as active when ``|lambda_i|`` exceeds
    ``active_rtol * max(|lambda|, 1)``. By complementarity, inactive inequality
    multipliers sit at ~0 while equalities and active inequalities carry a
    nonzero multiplier. Caveat: a weakly-active constraint with ``lambda ~ 0`` is
    classified inactive, which widens the reported nullspace.

    Parameters
    ----------
    Jg0 : (ng, nx) array
        Constraint Jacobian evaluated at the point of interest.
    HLag0 : (nx, nx) array
        Exact Hessian of the Lagrangian at the same point.
    lam0 : (ng,) array
        Constraint multipliers (from a solved point; zeros at the initial guess
        make the active set empty and the result non-diagnostic).
    active_rtol, zero_rtol : float
        Relative thresholds for the active-set and eigenvalue zero tests.

    Returns
    -------
    dict
        ``n_active`` (rows kept), ``rank`` (independent active rows; ``rank <
        n_active`` means LICQ fails), ``sigma_min_active`` (smallest singular
        value of the active Jacobian; ~0 => dependent active constraints),
        ``n_simple_bounds`` (active rows that are simple variable bounds, i.e.
        candidates to move to ``lbx``/``ubx``), ``n_free`` (reduced-Hessian
        dimension), ``cond`` (reduced condition number), ``eig_min``/``eig_max``
        (nonzero |eig| range), and the inertia triple
        ``n_neg``/``n_zero``/``n_pos``.
    """
    Jg0 = np.asarray(Jg0, dtype=float)
    HLag0 = np.asarray(HLag0, dtype=float)
    lam0 = np.asarray(lam0, dtype=float).reshape(-1)
    nx = HLag0.shape[0]

    lam_scale = max(float(np.max(np.abs(lam0))) if lam0.size else 0.0, 1.0)
    active = np.abs(lam0) > active_rtol * lam_scale
    A = Jg0[active, :]
    n_active = int(A.shape[0])

    if n_active == 0:
        # No active constraints -> the whole space is "free"; the reduced
        # Hessian is just H_Lag (typically singular here, hence non-diagnostic).
        Z = np.eye(nx)
        rank = 0
        sigma_min_active = float("nan")
        n_simple_bounds = 0
    else:
        U, s, Vt = np.linalg.svd(A, full_matrices=True)
        rank_tol = s.max() * max(A.shape) * np.finfo(float).eps if s.size else 0.0
        rank = int(np.sum(s > rank_tol))
        Z = Vt[rank:, :].T  # (nx, nx - rank): orthonormal nullspace basis
        sigma_min_active = float(s[-1]) if s.size else float("nan")
        # Active rows that touch a single variable are simple variable bounds
        # (Jacobian row = +-e_i). Coded as general ``subject_to`` inequalities
        # they bloat ``g`` and the KKT; routed to ``lbx``/``ubx`` they would be
        # handled by IPOPT's bound framework instead. This counts how many of
        # the active rows could move.
        nnz_per_row = np.sum(np.abs(A) > 1e-12, axis=1)
        n_simple_bounds = int(np.sum(nnz_per_row == 1))

    n_free = int(Z.shape[1])
    result = {
        "n_active": n_active,
        "rank": rank,
        "sigma_min_active": sigma_min_active,
        "n_simple_bounds": n_simple_bounds,
        "n_free": n_free,
        "cond": float("inf"),
        "eig_min": 0.0,
        "eig_max": 0.0,
        "n_neg": 0,
        "n_zero": 0,
        "n_pos": 0,
    }
    if n_free == 0:
        return result  # solution fully determined by the active constraints

    Hred = Z.T @ HLag0 @ Z
    Hred = 0.5 * (Hred + Hred.T)  # symmetrize away numerical noise
    w = np.linalg.eigvalsh(Hred)  # ascending real eigenvalues
    aw = np.abs(w)
    zthr = zero_rtol * max(float(aw.max()), 1.0)
    aw_nz = aw[aw > zthr]
    result["eig_max"] = float(aw.max())
    result["eig_min"] = float(aw_nz.min()) if aw_nz.size else 0.0
    result["cond"] = (
        float(aw_nz.max() / aw_nz.min()) if aw_nz.size else float("inf")
    )
    result["n_neg"] = int(np.sum(w < -zthr))
    result["n_zero"] = int(np.sum(aw <= zthr))
    result["n_pos"] = int(np.sum(w > zthr))
    return result


def stiffness_report(opti, sol=None, name="NLP"):
    # Objects
    x = opti.x
    g = opti.g
    f = opti.f
    lam = opti.lam_g

    # If no solution passed, evaluate at the initial guess. opti.value(expr, args)
    # evaluates a symbolic expression given value assignments without requiring a
    # solve; opti.initial() returns the list of warm-start assignments for x.
    if sol is None:
        x0 = np.asarray(opti.value(x, opti.initial())).squeeze()
        lam0 = np.zeros(opti.ng)
    else:
        x0 = sol.value(x)
        # lam_g is available only after a solve
        try:
            lam0 = sol.value(lam)
        except RuntimeError:
            lam0 = np.zeros(opti.ng)

    # Build functions
    Jg = ca.jacobian(g, x)
    Hf = ca.hessian(f, x)[0]
    # Hessian of the Lagrangian (H_f + sum_i lam_i * H_gi); cheap if many linear/affine g
    Lag = f + ca.dot(lam, g)
    HLag = ca.hessian(Lag, x)[0]

    fJg = ca.Function("fJg", [x], [Jg])
    fHf = ca.Function("fHf", [x], [Hf])
    fHLag = ca.Function("fHLag", [x, lam], [HLag])

    Jg0 = np.array(fJg(x0))
    Hf0 = np.array(fHf(x0))
    HLag0 = np.array(fHLag(x0, lam0))

    # KKT (symmetric indefinite)
    # [ HLag  Jg.T ]
    # [  Jg    0  ]
    KKT = np.block([[HLag0, Jg0.T], [Jg0, np.zeros((Jg0.shape[0], Jg0.shape[0]))]])

    # Safe condition numbers (fall back to SVD if needed)
    def cond(A):
        try:
            return np.linalg.cond(A)
        except np.linalg.LinAlgError:
            s = np.linalg.svd(A, compute_uv=False)
            return s[0] / s[-1] if s[-1] > 0 else np.inf

    print(f"=== {name} stiffness report ===")
    print("cond(Jg)          :", cond(Jg0))
    print("cond(Jg Jg^T)     :", cond(Jg0 @ Jg0.T))
    print("cond(H_f)         :", cond(Hf0))
    print("cond(H_Lagrangian):", cond(HLag0))
    print("cond(KKT)         :", cond(KKT))

    # Eigenvalue spread (magnitude range)
    def eig_spread(A):
        w = np.linalg.eigvals(A)
        aw = np.sort(np.abs(w))
        aw = aw[aw > 0]  # drop exact zeros
        if aw.size == 0:
            return (0.0, 0.0, np.inf)
        return (aw[0], aw[-1], aw[-1] / aw[0])

    emin, emax, ratio = eig_spread(HLag0)
    print("HLag |eig|min,max,ratio:", emin, emax, ratio)

    # --- Reduced Hessian: curvature on the active-constraint nullspace --------
    # This is the convergence-relevant metric; cond(H_f)/cond(H_Lag) above are
    # trivially infinite because the objective curvature is zero in the per-node
    # directions. Needs solution multipliers to define the active set, so at the
    # initial guess (lam = 0) it is skipped.
    if sol is not None and np.any(np.abs(lam0) > 0):
        rh = reduced_hessian_metrics(Jg0, HLag0, lam0)
        print(
            f"active constraints: {rh['n_active']} (rank {rh['rank']}"
            f"{'  <-- LICQ fails' if rh['rank'] < rh['n_active'] else ''}), "
            f"free directions: {rh['n_free']}"
        )
        if rh["n_active"]:
            print(
                "sigma_min(active Jg):",
                rh["sigma_min_active"],
                " (LICQ margin; ~0 => dependent active constraints)",
            )
            print(
                f"  of which simple variable bounds: {rh['n_simple_bounds']} "
                "(candidates for lbx/ubx instead of general constraints)"
            )
        if rh["n_free"] == 0:
            print(
                "reduced Hessian   : 0x0 "
                "(solution fully determined by active constraints)"
            )
        else:
            print(f"cond(reduced HLag): {rh['cond']}")
            print(
                "reduced |eig|min,max:",
                rh["eig_min"],
                rh["eig_max"],
            )
            print(
                f"reduced inertia (n-,n0,n+): "
                f"{rh['n_neg']}, {rh['n_zero']}, {rh['n_pos']}  "
                "(want n-=n0=0 for a strict local min)"
            )
    else:
        print(
            "reduced Hessian   : skipped (needs solution multipliers; "
            "re-run with sol= for the convergence-relevant curvature)"
        )

    # Simple magnitude/scale hints
    print("||x||_inf:", np.max(np.abs(x0)))
    if sol is not None:
        g0 = sol.value(g)
    else:
        g0 = ca.Function("fg", [x], [g])(x0).full().squeeze()
    print("||g||_inf:", np.max(np.abs(g0)))
