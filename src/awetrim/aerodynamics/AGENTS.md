# AWETrim Aerodynamics Module

## Scope

This module owns aerodynamic analysis and trim functionality that is not part of
the point-mass `system/` equations themselves.

The first accepted interface is the **VSM aerodynamic quasi-steady trim** surface
transferred from `Vortex-Step-Method/src/VSM/quasi_steady_state.py`. It covers:

- rigid aerodynamic VSM trim,
- aerodynamic force and moment residuals,
- aerodynamic stability derivatives around a trim state,
- parameter sweeps and plotting/dataframe helpers.

This module is not the aerostructural coupling module. Do not put PSM, structural
deformation, FSI iteration, or ASKITE coupling code here.

## Boundary

- AWETrim does not vendor or reimplement `VSM.core` solver internals.
- VSM bodies and solvers enter through protocols or optional runtime imports.
- AWETrim `SystemModel` supplies course-frame kinematics, apparent wind,
  inertial force, gravity force, and wind/kite velocity.
- Cross-module data uses dataclasses or plain dictionaries.
- No CasADi symbolic objects cross this module boundary; values are numerical
  at VSM trim solve time.

## Public Source Layout

```text
src/awetrim/aerodynamics/
  __init__.py
  AGENTS.md
  protocols.py
  vsm_quasi_steady.py
```

If the implementation grows, split internal helpers into:

```text
frames.py
attitude.py
stability_derivatives.py
sweeps.py
```

Keep the top-level public import path stable through `vsm_quasi_steady.py`.

## Public Script Layout

```text
scripts/aerodynamics/vsm_quasi_steady/
  solve_single_state.py
  run_sweep.py
  profile_single_state.py
  compute_stability_derivatives.py
```

Case-specific scripts may live one level deeper, for example:

```text
scripts/aerodynamics/vsm_quasi_steady/tudelft_v3/
```

Use snake_case Python filenames. Do not use hyphenated script names for new
scripts.

## Naming

Use `vsm_quasi_steady` for the VSM aerodynamic trim adapter. Avoid the generic
name `quasi_steady_state` because AWETrim already has a point-mass
quasi-steady residual solver in `SystemModel`.

Public functions should use these names:

- `solve_vsm_quasi_steady_trim`
- `compute_vsm_trim_stability_derivatives`
- `run_vsm_quasi_steady_sweep`
- `vsm_quasi_steady_sweep_to_dataframe`
- `plot_vsm_quasi_steady_sweep`

## Trim State Convention

The VSM trim unknown vector is ordered as:

```text
[speed_tangential, angle_roll_body_deg, angle_pitch_body_deg,
 angle_yaw_body_deg, timeder_angle_course_body]
```

Do not expose this as `kite_speed` in AWETrim-facing APIs. Use
`speed_tangential` to match the root symbol table.

## Frame Convention

The default course-frame basis is:

```text
course = [1, 0, 0]
normal = [0, 1, 0]
radial = [0, 0, 1]
```

The default transform from AWETrim course-frame values to VSM values is:

```text
[[-1,  0, 0],
 [ 0, -1, 0],
 [ 0,  0, 1]]
```

Any implementation must make this transform configurable through the public
interface.

## Required Developer Checks

Before implementing:

- Read this file and the root `AGENTS.md`.
- Keep the VSM dependency optional or protocol-based at import time.
- Preserve the trim unknown ordering above.
- Preserve warm-start behaviour for sweep cases.
- Keep plotting and dataframe conversion separate from the core solve.
- Add bounds defaults to `awetrim.utils.defaults` if they become package-level
  defaults rather than call arguments.

## Required Tester Checks

Tests for this module should check:

- Public signatures and dataclass fields.
- Shape validation for trim state, bounds, frame transforms, Jacobians, and
  stability outputs.
- That no `VSM.core` import is required merely to import `awetrim`.
- That `SystemModel(quasi_steady=True)` enforces
  `timeder_speed_tangential = 0`.
- Numerical VSM solver tests may be marked `slow` and skipped when VSM is not
  installed.
