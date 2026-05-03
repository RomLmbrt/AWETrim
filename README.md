# AWETrim

AWETrim is a Python library for the modelling, trim analysis, and aerostructural simulation of soft-kite Airborne Wind Energy Systems (AWES). It provides:

- A CasADi-based **quasi-steady system model** (kite, tether, winch, wind field) for trajectory optimisation and reduced-order simulation.
- A **VSM aerodynamic trim interface** that wraps the Vortex Step Method solver to find the trim state (kite speed, roll, pitch, yaw) at any flight condition.
- A **PSS/QSM aerostructural coupled solver** that iterates between a deformable structural particle system and the VSM quasi-steady trim to compute the loaded wing shape and forces.
- An **EKF flight-data analysis** pipeline for post-processing experimental flight logs.

## Installation

```bash
git clone https://github.com/awegroup/AWETrim.git
cd AWETrim
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -e .[dev]
```

## Package structure

```
src/awetrim/
  system/          CasADi system model: SystemModel, Kite, Tether variants, Wind, Winch
  aerodynamics/    VSM quasi-steady trim: solve_vsm_quasi_steady_trim
  aerostructural/  PSS/QSM coupled structural-aerodynamic solver
  kinematics/      Course-frame kinematics and parametrised path patterns (B-spline, helix, …)
  timeseries/      Simulation helpers: PhaseParameterized, ReeloutSimple, ReelinSimple, Cycle
  environment/     Wind models: uniform, logarithmic, tabulated
  experimental/    EKF flight-data analysis (wraps awes-ekf with AWETrim's data layout)
  utils/           Reference-frame transforms, plotting palettes, default limits
```

## Data layout

```
data/<kite_name>/
  aerostructural_configs/config.yaml      solver settings (dt, tolerances, actuation)
  kite_geometries/powered_geometry/
    aero_geometry.yaml                    VSM panel sections and airfoil polars
    struc_geometry.yaml                   structural node positions and spring connectivity
  ekf_config/<model>_config.yaml          EKF simulation/tuning parameters
  flight_logs/                            raw CSV flight data

results/<kite_name>/
  ekf/<model>_<YYYY>-<MM>-<DD>.h5        EKF analysis output
  <case_folder>/sim_output.h5             aerostructural coupled-solver output
```

## Scripts

Each script directory has a `common.py` with shared helpers and defaults.

**`scripts/aerodynamics/`** — standalone aerodynamic analysis

| Script | Purpose |
|--------|---------|
| `solve_single_state.py` | Single VSM trim solve at a given flight condition |
| `run_sweep.py` | Parameter sweep over wind speed / flight condition |
| `compute_stability_derivatives.py` | Numerical aerodynamic stability derivatives |
| `calculate_max_roll.py` | Maximum roll angle from trim constraints |
| `plot_polars.py` | Visualise airfoil polar data |

**`scripts/aerostructural/`** — coupled structural-aerodynamic simulation

| Script | Purpose |
|--------|---------|
| `run_simulation_level_qsm.py` | Single PSS/QSM aerostructural solve with optional steering sweep |
| `run_sweep_wind_steering.py` | 2-D sweep: wind speed × steering extension |
| `run_sweep_course_steering_depower.py` | 3-D sweep: course angle × steering × depower |

**`scripts/reduced-order-model/`** — reduced-order model validation and path optimisation

| Subdirectory | Purpose |
|-------------|---------|
| `optimization/reelout/` | Optimise uploop / downloop / helix path parameters |
| `validation/` | Validate quasi-steady state and spline approximations |
| `wind_estimation/` | Inverse wind estimation from trajectory data |

**`scripts/experimental/`** — EKF flight-data analysis

| Script | Purpose |
|--------|---------|
| `run_analysis_ekf.py` | Interactive pipeline: select flight log → pre-process → run EKF → save results |
| `plot_analysis_ekf.py` | Visualise saved EKF output |

## Running scripts

All scripts are run from the **project root**:

```bash
python scripts/aerodynamics/solve_single_state.py
python scripts/aerostructural/run_simulation_level_qsm.py
python scripts/experimental/run_analysis_ekf.py
```

## Tests

```bash
pytest                          # all tests
pytest tests/aerostructural/    # one module
pytest -m "not slow"            # skip slow integration tests
pytest --cov=src --cov-report=term-missing
```

## Key dependencies

| Package | Role |
|---------|------|
| [Vortex-Step-Method](https://github.com/awegroup/Vortex-Step-Method) | VSM aerodynamic solver (pinned to `@main` via `pyproject.toml`) |
| [Particle_System_Simulator](https://github.com/awegroup/Particle_System_Simulator) | Structural particle system (PSS) |
| [EKF-AWE](https://github.com/ocayon/EKF-AWE) | Extended Kalman Filter for flight-data analysis |
| CasADi | Symbolic computation in the system model |

## License

MIT — see `LICENSE`.
