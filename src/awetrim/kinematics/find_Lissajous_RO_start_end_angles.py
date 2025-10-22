from awetrim.kinematics.parametrized_patterns import create_pattern_from_dict
import numpy as np
import pickle

# ---------- Load precomputed Lissajous fit data ----------
segment_name = "LISSAJOUS"

filename = f"fit_results_{segment_name}.pkl"
with open(filename, "rb") as f:
    fit_data = pickle.load(f)

r0 = fit_data["r0"]
duration = fit_data["duration"]
az_amp0 = fit_data["best_params"]["az_amp0"]
beta_amp0 = fit_data["best_params"]["beta_amp0"]
beta_coeffs = fit_data["best_params"]["beta_coeffs"]
az_coeffs = fit_data["best_params"]["az_coeffs"]
beta0 = fit_data["best_params"]["beta0"]

# Recreating the starting and ending point of the path used to validate the model
s_start_init = 1.36 * np.pi
range_init = 1.45 * np.pi
cycles = 1


trial_pattern = create_pattern_from_dict(
    pattern_type = "cst_lissajous",
    parameters = {
        "omega": 1.0,
        "r0": r0,
        "az_amp0": az_amp0,
        "beta_amp0": beta_amp0,
        "width_phi": 0.5,
        "width_beta": 0.5,
        "left_first": True,
        "normalize_bumps": False,
        "repeat_phi": True,
        "repeat_beta": True,
        "beta_coeffs": np.array(beta_coeffs),
        "az_coeffs": az_coeffs,
        "kbeta": 0,
        "beta0": beta0,
        "kappa": 0,
    })

az0 = trial_pattern.azimuth(1, s_start_init)
el0 = trial_pattern.elevation(1, s_start_init)

azf = trial_pattern.azimuth(1, s_start_init + range_init + cycles * (2*np.pi))
elf = trial_pattern.elevation(1, s_start_init + range_init + cycles * (2*np.pi))

def residuals(params):
    s_start, range = params
    az_start = trial_pattern.azimuth(1, s_start)
    el_start = trial_pattern.elevation(1, s_start)
    az_end = trial_pattern.azimuth(1, s_start + range + cycles * (2*np.pi))
    el_end = trial_pattern.elevation(1, s_start + range + cycles * (2*np.pi))

    res = np.array([
        float(az_start - az0),
        float(el_start - el0),
        float(az_end - azf),
        float(el_end - elf),
    ])

    return res

from scipy.optimize import least_squares

initial_guess = [s_start_init, range_init]
result = least_squares(residuals, initial_guess)
s_start_opt, range_opt = result.x

print(f"Optimized start angle: {s_start_opt}")
print(f"Optimized range: {range_opt}")