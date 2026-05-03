# VSM Aerodynamic Quasi-Steady Scripts

Use this folder for runnable scripts that exercise
`awetrim.aerodynamics.vsm_quasi_steady`.

- `solve_single_state.py`
- `run_sweep.py`
- `profile_single_state.py`
- `compute_stability_derivatives.py`

Case-specific scripts may live in subfolders such as `tudelft_v3/`.

Each script accepts `--vsm-src` when Vortex-Step-Method is available locally but
not installed as a package.
