# Level 3 Aerodynamic Coefficient Model from Quasi-Steady VSM Aero-Structural Solves

## Purpose

This document describes an implementation strategy for building a fast 6-DOF rigid-body aerodynamic model for a soft kite using coefficient maps generated from quasi-steady aero-structural VSM simulations.

The target model is a **Level 3 aerodynamic model**, where aerodynamic force and moment coefficients depend on:

\[
C_i = C_i\left(
\alpha,\beta,\delta_d,\delta_s,v_a,
\hat p,\hat q,\hat r
\right)
\]

with:

\[
\hat p = \frac{b p}{2 v_a}, \qquad
\hat q = \frac{c q}{2 v_a}, \qquad
\hat r = \frac{b r}{2 v_a}
\]

and:

\[
C_i \in \{C_X,C_Y,C_Z,C_l,C_m,C_n\}
\]

or, if preferred in wind axes:

\[
C_i \in \{C_D,C_Y,C_L,C_l,C_m,C_n\}
\]

The model is inspired by conventional rigid-body aircraft aerodynamic models, but the coefficient data is generated from a soft-kite VSM model including aero-structural deformation and bridle effects.

---

## Conceptual Overview

For a rigid aircraft, aerodynamic coefficients can often be represented as functions of angle of attack, sideslip, angular rates and control-surface deflections.

For a soft kite, this is not sufficient because the aerodynamic shape depends on loading and actuation. Therefore, the coefficient database should be generated in two layers:

1. **Anchor aero-structural states**

   These are converged quasi-steady aerostructural solutions that define the deformed kite geometry for a selected operating condition. Each will be solved for a certain wind speed and position in the wind window.

2. **Local aerodynamic sweeps around each anchor state**

   Around each anchor geometry, perturb angle of attack, sideslip and rotational rates to estimate local aerodynamic response and derivatives.


## Generating Anchor States

Anchor states should be generated over a sparse but meaningful grid:

```text
vw        
elevation
azimuth
course
depower   : symmetric actuation levels
steering  : asymmetric actuation levels
```

However, a full tensor product may become too expensive. Prefer:

- start with sparse anchors;
- use adaptive refinement near nonlinear regions;
- use continuation between nearby operating points.

---

## Local Sweep Around Each Anchor State

For each converged anchor state, define local perturbations:

```text
Delta alpha : [-4, -2, 0, 2, 4] deg
Delta beta  : [-4, -2, 0, 2, 4] deg
Delta p_hat : [-0.05, 0.0, 0.05]
Delta q_hat : [-0.05, 0.0, 0.05]
Delta r_hat : [-0.05, 0.0, 0.05]
```

A complete tensor product can still be large. For derivative identification, a structured design is often better:

1. Baseline case.
2. One-at-a-time perturbations for each variable.
3. Selected cross terms if needed.
4. Extra samples near nonlinear regions.

Example:

```text
baseline:
  Delta alpha = 0, Delta beta = 0, Delta p = 0, Delta q = 0, Delta r = 0

alpha sweep:
  Delta alpha != 0, others = 0

beta sweep:
  Delta beta != 0, others = 0

rate sweeps:
  Delta p != 0, others = 0
  Delta q != 0, others = 0
  Delta r != 0, others = 0

optional coupling sweeps:
  Delta alpha and Delta beta
  Delta beta and Delta r
  Delta alpha and Delta q
```

---

## Frozen Geometry vs Re-Solved Geometry

There are two possible interpretations of the local sweep.

### Option A: Frozen-geometry aerodynamic sweep

1. Solve the full aero-structural equilibrium at the anchor state.
2. Freeze the geometry.
3. Evaluate aerodynamic forces and moments for perturbed \(\alpha,\beta,p,q,r\).

This is fast and useful for estimating local aerodynamic derivatives.

### Option B: Re-solved aero-structural perturbation

1. Solve the full aero-structural equilibrium at the anchor state.
2. For each perturbation, re-solve the aero-structural equilibrium.
3. Extract forces and moments.

This is more physical but more expensive.

### Recommended approach

Use a hybrid strategy:

- use frozen-geometry sweeps for most local derivative identification;
- re-solve the full aero-structural problem for selected perturbations;
- compare both results to quantify deformation-induced derivative changes;
- if the difference is large, include more full re-solves or add geometry/load-state parameters to the surrogate.

---

## Local Model Fitting

Around each anchor state, fit a local model for each coefficient:

\[
C_i \approx C_{i,0}
+ C_{i,\alpha}\Delta\alpha
+ C_{i,\beta}\Delta\beta
+ C_{i,p}\Delta\hat p
+ C_{i,q}\Delta\hat q
+ C_{i,r}\Delta\hat r
\]

Optionally include quadratic and coupling terms:

\[
C_i \approx C_{i,0}
+ \mathbf{g}_i^T \Delta \mathbf{x}
+ \frac{1}{2}\Delta \mathbf{x}^T H_i \Delta \mathbf{x}
\]

where:

\[
\Delta \mathbf{x} =
\begin{bmatrix}
\Delta\alpha & \Delta\beta & \Delta\hat p & \Delta\hat q & \Delta\hat r
\end{bmatrix}^T
\]

The fitted local parameters are then functions of the anchor coordinates:

\[
C_{i,0}, C_{i,\alpha}, C_{i,\beta}, C_{i,p}, C_{i,q}, C_{i,r}
=
F_i(v_a,\alpha_0,\beta_0,\delta_d,\delta_s,\hat p_0,\hat q_0,\hat r_0)
\]

---

## Global Interpolation Strategy

The final fast aerodynamic model should interpolate between anchor states.

Recommended options:

1. **Regular grid interpolation**

   Use if the anchor grid is structured.

   Good tools:

   - `scipy.interpolate.RegularGridInterpolator`
   - tensor-product splines

2. **Scattered interpolation**

   Use if adaptive sampling produces irregular anchor locations.

   Good tools:

   - radial basis functions
   - Gaussian processes
   - local regression
   - Delaunay-based interpolation for low dimension

3. **Surrogate model**

   Use if the dimension becomes too high.

   Good options:

   - sparse polynomial regression
   - neural network surrogate
   - gradient-boosted trees for analysis, but less ideal for smooth dynamics
   - differentiable MLP if gradients are needed for optimization

For control and trajectory optimization, smoothness matters. Avoid discontinuous interpolation if derivatives are required.

---

## Runtime Aerodynamic Model

At runtime, the rigid-body model should:

1. Compute apparent velocity in body frame.
2. Compute \(v_a\), \(\alpha\), \(\beta\).
3. Compute normalized rates \(\hat p,\hat q,\hat r\).
4. Query the aerodynamic coefficient model.
5. Convert coefficients to dimensional force and moment.

Pseudo-code:

```python
def aerodynamic_force_moment(state, controls, aero_db, params):
    va_body = compute_apparent_velocity_body(state, params.wind)
    va = np.linalg.norm(va_body)

    alpha = np.arctan2(va_body[2], va_body[0])
    beta = np.arcsin(np.clip(va_body[1] / va, -1.0, 1.0))

    p, q, r = state.angular_velocity_body

    p_hat = params.b_ref * p / (2.0 * va)
    q_hat = params.c_ref * q / (2.0 * va)
    r_hat = params.b_ref * r / (2.0 * va)

    query = np.array([
        va,
        alpha,
        beta,
        controls.depower,
        controls.steering,
        p_hat,
        q_hat,
        r_hat,
    ])

    CX, CY, CZ, Cl, Cm, Cn = aero_db.evaluate(query)

    q_dyn = 0.5 * params.rho * va**2

    force_body = q_dyn * params.S_ref * np.array([CX, CY, CZ])
    moment_body = q_dyn * params.S_ref * np.array([
        params.b_ref * Cl,
        params.c_ref * Cm,
        params.b_ref * Cn,
    ])

    return force_body, moment_body
```

---

## Symmetry Considerations

For a symmetric kite and symmetric bridle, useful approximate symmetries may exist.

At zero steering and zero sideslip:

- \(C_Y \approx 0\)
- \(C_l \approx 0\)
- \(C_n \approx 0\)

Under sign reversal of sideslip:

- \(C_X, C_Z, C_m\) are often approximately even in \(\beta\)
- \(C_Y, C_l, C_n\) are often approximately odd in \(\beta\)

Under sign reversal of steering:

- symmetric coefficients may remain approximately even;
- lateral and yaw/roll coefficients may change sign.

These symmetries should not be forced blindly, but they are useful for validation and data-quality checks.

---

## Validation Checks

### 1. Coefficient sanity checks

Plot coefficients versus:

- \(\alpha\)
- \(\beta\)
- depower
- steering
- normalized rates

Look for discontinuities, sign errors, nonphysical jumps or failed VSM solutions.

### 2. Symmetry checks

Compare:

```text
C(alpha, beta, steering)
C(alpha, -beta, -steering)
```

where applicable.

### 3. Finite-difference derivative checks

Verify fitted derivatives against central finite differences:

\[
C_{i,\alpha} \approx
\frac{C_i(\alpha_0 + h) - C_i(\alpha_0 - h)}{2h}
\]

and similarly for \(\beta,p,q,r\).

### 4. Dynamic validation

Run the 6-DOF rigid-body model with the fitted aerodynamic database and compare against:

- VSM quasi-steady reference cases;
- known turn-rate behaviour;
- measured flight data, if available;
- existing 4-point model trajectories.

---

## Practical Notes

### Apparent-speed dependence

In a rigid aircraft model, \(v_a\) mainly appears through dynamic pressure and rate normalization.

For a soft kite, \(v_a\) should also be part of the coefficient database because it changes loading and therefore deformation.

Therefore, keep \(v_a\) as an explicit database coordinate.

### Sideslip in the anchor state

Because the anchor state includes \(\beta_0\), do not treat sideslip only as a local perturbation. The database should support both:

```text
anchor beta = beta_0
local perturbation = Delta beta
absolute beta = beta_0 + Delta beta
```

This is important for turning or steered equilibria where the natural quasi-steady state may have nonzero sideslip.

### Angular rates in the anchor state

If anchor states include nonzero \(p,q,r\), the VSM solve should reflect the apparent velocity field induced by rotation.

For a point at body-frame position \(x_b\), the local apparent velocity should include:

\[
\mathbf{v}_{a,local} = \mathbf{v}_{a,CoM} + \boldsymbol{\omega} \times \mathbf{x}_b
\]

depending on the sign convention used in the code.

This is essential for identifying damping-like derivatives consistently.

---

## Suggested Development Roadmap

### Step 1: Implement body-axis coefficient extraction

Input:

- VSM force distribution;
- reference point;
- reference area/span/chord.

Output:

- \([C_X,C_Y,C_Z,C_l,C_m,C_n]\).

### Step 2: Generate a small pilot database

Use a very small grid, for example:

```text
va        : 2 values
alpha     : 3 values
beta      : 3 values
depower   : 2 values
steering  : 3 values
rates     : zero only for anchors
local sweeps: alpha, beta, p, q, r one-at-a-time
```

### Step 3: Fit local derivatives

Fit local linear models around each anchor.

### Step 4: Build runtime interpolator

Interpolate anchor coefficients and derivatives.

### Step 5: Validate in dynamic simulation

Compare against existing quasi-steady or 4-point model results.

### Step 6: Add nonlinear terms and adaptive refinement

Refine where errors are largest:

- near stall;
- strong depower;
- strong steering;
- large sideslip;
- high rotation rates.

---

## Open Design Decisions

Before finalizing the implementation, decide:

1. Store coefficients in body axes or wind axes?
2. Use frozen-geometry local sweeps, full re-solved sweeps, or hybrid?
3. Use structured grids or adaptive scattered sampling?
4. Fit local linear derivatives or global surrogate directly?
5. Include cross terms such as \(\beta r\), \(\alpha q\), steering-rate coupling?
6. Which reference point should define the moments: CoM, bridle point or aerodynamic reference point?
7. Should tether/bridle forces be included separately from aerodynamic forces, or should the coefficient model represent only canopy aerodynamic loads?

---

## Recommended Initial Choice

For the first implementation, use:

```text
coefficients: body-axis [CX, CY, CZ, Cl, Cm, Cn]
anchors: [va, alpha, beta, depower, steering, p_hat, q_hat, r_hat]
local model: linear in [Delta alpha, Delta beta, Delta p_hat, Delta q_hat, Delta r_hat]
geometry: frozen for local sweeps, with selected re-solved checks
interpolation: regular-grid interpolation if possible
validation: finite differences, symmetry, dynamic simulation
```

This gives a practical and extensible path from VSM data to a fast 6-DOF rigid-body model.
