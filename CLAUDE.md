# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install (editable, with dev extras)
pip install -e .[dev]

# Run all tests
pytest

# Run tests for one module
pytest tests/aerostructural/
pytest tests/system/test_kite.py

# Run a single test
pytest tests/aerostructural/test_pss.py::test_pulley_rest_lengths

# Run with coverage
pytest --cov=src --cov-report=term-missing

# Skip slow tests
pytest -m "not slow"
```

Scripts live in `scripts/` and are run directly from the project root:
```bash
python scripts/aerostructural/run_simulation_level_qsm.py
python scripts/aerodynamics/solve_single_state.py
python scripts/experimental/run_analysis_ekf.py
```

Each script directory has a `common.py` with shared CLI helpers; `scripts/aerostructural/common.py` additionally defines `CONFIG_DEFAULTS` (the single source of truth for all config key defaults) and shared helpers (`format_length_tag`, `build_actuation_case_folder`, `configure_system_model_from_config`, etc.).

## Architecture

### Reference frames

AWETrim uses a **course frame** (C) throughout: x = tangential (direction of kite flight), y = normal (spanwise), z = radial (along tether). The VSM package uses its own frame where x and y are negated relative to C; the transformation matrix `DEFAULT_TRANSFORMATION_C_FROM_VSM = [[-1,0,0],[0,-1,0],[0,0,1]]` in `aerodynamics/vsm_quasi_steady.py` converts between them.

### `system/` — the CasADi system model

`SystemModel` (extends `KiteKinematics`) is the central object. It holds symbolic CasADi expressions for all kinematic quantities: `velocity_kite`, `velocity_apparent_wind`, `acceleration`, etc. Concrete values are obtained by setting numeric attributes (`speed_tangential`, `angle_elevation`, …) which override the symbolic variables via `_speed_wind_ref_value` and similar numeric caches. **No CasADi symbolics cross module boundaries** — downstream code always evaluates them to numpy arrays first via `_as_numeric_3vector`.

`Kite`, `Tether` (several variants), `Wind`, and `Winch` are component models that `SystemModel` composes. `SystemModel` delegates aerodynamic force expressions to `Kite` and tether drag/tension to the chosen `Tether` subclass.

### `aerodynamics/` — VSM quasi-steady trim

`solve_vsm_quasi_steady_trim` in `vsm_quasi_steady.py` is the core function. It wraps the external VSM `Solver` in a `scipy.optimize.least_squares` loop that finds `[speed_tangential, roll_deg, pitch_deg, yaw_deg, course_rate_body]` such that the five moment/force residuals (cmx, cmy, cmz, cfx, cfy) vanish. The body attitude is applied each iteration via `_set_body_attitude_from_baseline` which rotates the baseline panel geometry and rebuilds VSM panels. `protocols.py` defines the `VsmBodyAerodynamics`, `VsmSolver`, and `AWETrimSystemModel` structural-subtyping protocols so that the VSM package can be swapped without changes to `vsm_quasi_steady.py`.

### `aerostructural/` — PSS/QSM coupled structural solver

Fixed-point iteration between a structural particle system (PSS) and the QSM trim:

1. `structural_geometry_io.py` parses `struc_geometry.yaml` into arrays (nodes, connectivity, rest lengths, stiffness, damping, pulley dict).  
   **Critical**: pulley arm rest lengths in the YAML store the *total* rope length; `structural_pss.instantiate` must override each arm with its individual length (from `pulley_line_to_other_node_pair[idx][3]`) — the PSS will diverge without this.
2. `structural_pss.py` wraps the PSS `ParticleSystem` — calls `kin_damp_sim` in a loop with external nodal forces until kinetic energy converges.
3. `aerodynamic_vsm.py` wraps `solve_vsm_quasi_steady_trim` — updates the VSM body from structural LE/TE points, runs the trim, returns panel forces in the course frame.
4. `aerostructural_coupled_solver_qsm.py` orchestrates the outer fixed-point loop: structural solve → aero trim → load mapping → actuation → convergence check. Aitken relaxation is applied to node positions.
5. `mapping.py` provides `LinearStructuralToAeroMapper` (LE/TE interpolation) and `BilinearAeroToStructuralLoadMapper` (panel force → nodal force with moment preservation).

### Data layout

```
data/<kite_name>/
  aerostructural_configs/config.yaml   ← solver settings (dt, tolerances, actuation)
  kite_geometries/powered_geometry/
    aero_geometry.yaml                 ← VSM panel sections and airfoil polars
    struc_geometry.yaml                ← structural node positions, spring connectivity
  ekf_config/<model>_config.yaml       ← EKF simulation/tuning parameters
  flight_logs/                         ← raw CSV flight data

results/<kite_name>/<analysis_type>/
  sim_output.h5                        ← aerostructural coupled-solver output
  ekf/<model>_<YYYY>-<MM>-<DD>.h5     ← EKF analysis output
```

### `experimental/` — EKF flight-data analysis

`experimental/settings.py` replaces `awes_ekf.setup.settings` for AWETrim. `load_config()` searches `data/` for YAML files with EKF keys instead of the hard-coded `data/config/` path, and injects `_awetrim_kite_name` into the returned dict. `save_ekf_results()` saves to `results/<kite_name>/ekf/` (absolute from project root) instead of CWD-relative `results/<model>/`.  
Import these from `awetrim.experimental.settings`, not from `awes_ekf.setup.settings`.

### External dependencies

| Package | Role | Install note |
|---------|------|-------------|
| `Vortex-Step-Method` | VSM aerodynamic solver | Installed from GitHub `@main` via `pyproject.toml`; pinned to origin, not a local clone |
| `PSS` | Particle System Simulator (structural solver) | Installed from GitHub via `pyproject.toml` |
| `awes-ekf` | Extended Kalman Filter for flight data | Installed from GitHub; path-related functions are overridden by `awetrim.experimental.settings` |
| `CasADi` | Symbolic computation in `system/` | Required; no CasADi in `aerostructural/` or `aerodynamics/` |

### `scripts/aerostructural/common.py` — shared defaults

`CONFIG_DEFAULTS` is the single source of truth for all `config.get(key, default)` fallback values. Any new aerostructural script should import from here rather than repeating literals. The sweep scripts (`run_sweep_wind_steering.py`, `run_sweep_course_steering_depower.py`) produce output under `results/<kite_name>/` using `PROJECT_DIR = Path(__file__).resolve().parents[2]` (not CWD-relative).
