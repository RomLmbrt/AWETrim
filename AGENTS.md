# AWETrim — Agent Context

## What this repo is

AWETrim is a Python library for the design and optimisation of soft-kite Airborne
Wind Energy Systems (AWES). It models the kite as a point mass in a course-aligned
spherical reference frame, provides CasADi-based trajectory optimisation, and
implements the winch/tether physics for pumping-cycle simulation.

---

## Module status

```
src/awetrim/
  system/            ✅  Wing, Kite, SystemModel, Tether, Winch
  aerodynamics/      🟡  VSM aerodynamic quasi-steady trim interface
  kinematics/        ✅  course-frame kinematics, B-spline path patterns
  timeseries/        ✅  PhaseParameterized, ReeloutSimple, ReelinSimple, Cycle
  environment/       ✅  Wind (logarithmic / uniform / tabulated)
  utils/             ✅  fitting, defaults, reference frames

  aerostructural/    🔴  NOT YET BUILT — see src/awetrim/aerostructural/AGENTS.md
  identification/    🔴  NOT YET BUILT — see src/awetrim/identification/AGENTS.md
```

**Do not create files in `aerostructural/` or `identification/` without first
reading their `AGENTS.md` and following the architect → developer → tester
workflow below.**

---

## Three-repo ecosystem

| Repo | Role |
|------|------|
| `awegroup/Vortex-Step-Method` | Standalone VSM solver. AWETrim does not contain VSM code — polars enter as `aero_input` dicts. |
| **AWETrim** (this repo) | Aerodynamic trim, ROM, identification, trajectory optimisation. |
| `ASKITE` | PSM–VSM aerostructural FSI solver. Imports AWETrim's `aerostructural/` module. |

---

## Roles

Every session operates in exactly one role. State it explicitly at the start,
e.g. *"acting as @architect, design the identification module interface"*.
Default if unspecified: **@developer**.

### @architect
Design module interfaces before any implementation exists.
- Output: Python `Protocol` or `ABC` with type signatures, data-flow description,
  and public method docstrings. No implementation code.
- Update the module's `AGENTS.md` with the agreed interface before any `.py`
  file is created.
- Cross-module data uses plain dicts or `dataclass` — no CasADi symbolics
  crossing module boundaries unless explicitly decided.

### @developer
Implement against a spec the architect has already written.
- Read the module's `AGENTS.md` before writing any code.
- Use CasADi (`ca.MX`, `ca.Function`, `ca.Opti`) for all symbolic quantities.
  Do not substitute numeric values until solve time.
- Follow existing patterns: properties for symbolic quantities, `DEFAULT_*`
  constants in `utils/defaults.py`, YAML-driven configs.
- Do not change an architect-defined interface unilaterally.

### @reviewer
Check that an implementation matches the paper physics. No new code.
- Every equation must trace back to one of:
  - **Aerostructural / VSM:** Cayon, Gaunaa, Schmehl (2023) *Energies* 16, 3061
  - **Identification / ROM:** Cayon, van Deursen, Schmehl (2026) *WES* 11, 1097
  - **Trajectory optimisation:** Cayon & Schmehl (2026) Torque extended abstract
- Flag any mismatch between code variable names and the symbol table below.
- Confirm `timeder_speed_tangential = 0` is enforced in `SystemModel(quasi_steady=True)`.

### @tester
Write and run `pytest` tests. One test file per source module under `tests/`.
- Test CasADi expression structure and symbolic shapes, not numeric solver values.
- Use fixtures in `tests/conftest.py` for shared kite/system setups.
- Tests for `aerostructural/` and `identification/` are required before those
  modules are considered complete.

---

## Symbol ↔ code name table

| Symbol | Code name | Unit |
|--------|-----------|------|
| r | `distance_radial` | m |
| β | `angle_elevation` | rad |
| φ | `angle_azimuth` | rad |
| χ | `angle_course` | rad |
| vτ | `speed_tangential` | m/s |
| vr | `speed_radial` | m/s |
| s | `s` | — |
| ṡ | `s_dot` | 1/s |
| Ft | `tension_tether_ground` | N |
| uₛ | `input_steering` | — |
| uₚ | `input_depower` | — |
| α | `angle_of_attack` | rad |
| θb | `angle_pitch_tether` | rad |
| ΔCD0 | `cd0` | — |

---

## Core conventions

- **CasADi** throughout. All state variables are `ca.MX.sym`. Never replace
  symbolic variables with NumPy scalars inside module code.
- **YAML** configs drive kite parameters and pattern settings.
  See `data/LEI-V3-KITE/` for reference examples.
- **IPOPT** is the default NLP solver: `opti.solver("ipopt", {...})`.
- Optimisation variable bounds live in `utils/defaults.py` (`DEFAULT_OPTI_LIMITS`).
  Add new variables there, not inline in module code.
