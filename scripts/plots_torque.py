import numpy as np
import matplotlib.pyplot as plt
from awetrim.system.winch import Winch
from awetrim.utils.color_palette import set_plot_style, get_color_list

set_plot_style()
# Example usage and test of Winch class and tension_curve plotting
example_pattern_config = {
    "reeling_strategy": "force",  # "force" or "constant"
    "force_model": "linear",  # "linear" or "quadratic"
    "reeling_speed": 0,  # m/s, only for constant reeling
    "max_tether_force": 8400,  # N, only for force reeling
    "min_tether_force": 1500.0,  # N, only for force reeling
    "softplus": False,
    "softplus_beta": 1e-3,  # bigger is sharper
    "softminus": False,
    "softminus_beta": 1e-3,  # bigger is sharper
    "slope_winch_force": 5555.55,  # N/(m/s)^2 for quadratic, N/(m/s) for linear
    "offset_winch_force": 0.58,  # m/s
}

winch = Winch(pattern_config=example_pattern_config)
fig, ax = plt.subplots(figsize=(5, 4))
# winch.plot_tension_curve(vr_min=-2, vr_max=6, n_points=400, show=False, ax=ax)
v_r = np.linspace(-2, 6, 400)
T_fun = winch.tension_curve
T_vals = np.array([float(T_fun(v)) for v in v_r])

ax.plot(
    v_r,
    T_vals,
    label="Identified linear relation",
    color=get_color_list()[0],
    linestyle="--",
)
# Example usage and test of Winch class and tension_curve plotting
example_pattern_config = {
    "reeling_strategy": "force",  # "force" or "constant"
    "force_model": "linear",  # "linear" or "quadratic"
    "reeling_speed": 0,  # m/s, only for constant reeling
    "max_tether_force": 8400,  # N, only for force reeling
    "min_tether_force": 1500.0,  # N, only for force reeling
    "softplus": True,
    "softplus_beta": 1e-3,  # bigger is sharper
    "softminus": True,
    "softminus_beta": 1e-3,  # bigger is sharper
    "slope_winch_force": 5555.55,  # N/(m/s)^2 for quadratic, N/(m/s) for linear
    "offset_winch_force": 0.58,  # m/s
}
winch = Winch(pattern_config=example_pattern_config)
# winch.plot_tension_curve(vr_min=-2, vr_max=6, n_points=400, show=True, ax=ax)
T_fun = winch.tension_curve
T_vals = np.array([float(T_fun(v)) for v in v_r])

ax.plot(
    v_r,
    T_vals,
    label="Tension curve with softplus/softminus",
    color=get_color_list()[0],
    linestyle="-",
)
ax.set_xlabel(r"Reeling speed $v_r$ (m/s)")
ax.set_ylabel("Tension (N)")
ax.set_ylim(0, 9000)
ax.legend()
ax.set_xlim(-2, 6)
plt.savefig("results/figures/torque2026/winch_tension_curve.pdf")
plt.show()


def bernstein2of4(x):
    """p(x) = 6 x^2 (1-x)^2"""
    return 6.0 * (x**2) * ((1.0 - x) ** 2)


def gate01(x):
    """Hard gate: 1 if 0<=x<=1 else 0"""
    return ((x >= 0.0) & (x <= 1.0)).astype(float)


def bump_pair(u, s0, wp, normalize=False):
    """
    Periodic paired bump:
      x_right = (u - s0)/wp
      x_left  = (u - s0 + 1)/wp   (shifted by 1 to wrap around)
      p_pair  = gate(x_right)*p(x_right) + gate(x_left)*p(x_left)

    Interpretation:
      - 'wp' controls the bump width in u-space
      - 's0' is where the RIGHT bump starts (becomes non-zero)
    """
    x_right = (u - s0) / wp
    x_left = (u - s0 + 1.0) / wp

    pright = gate01(x_right) * bernstein2of4(x_right)
    pleft = gate01(x_left) * bernstein2of4(x_left)

    b = pright + pleft
    return b / wp if normalize else b


def plot_bumps_and_total(az_coeffs, wp, repeat_phi=True, normalize_bumps=False, n=2000):
    az_coeffs = np.asarray(az_coeffs, dtype=float)
    P = len(az_coeffs)
    K = 2 * P if repeat_phi else P

    u = np.linspace(0, 1, n, endpoint=False)

    baseline = np.ones_like(u)
    contributions = []
    labels = []

    for k in range(K):
        wk = az_coeffs[k % P]

        # Choose where each bump "starts being non-zero" (s0 in your quote).
        # If you want bumps evenly distributed, this is the natural choice:
        s0 = k / K

        bk = bump_pair(u, s0=s0, wp=wp, normalize=normalize_bumps)
        contrib = wk * bk

        contributions.append(contrib)
        labels.append(f"k={k}, s0={s0:.3f}, w={wk:+.3f}")

    contributions = np.array(contributions)
    Nphi = baseline + contributions.sum(axis=0)

    # ---- Plot 1: individual contributions ----
    plt.figure(figsize=(5, 4))
    for k in range(K):
        plt.plot(
            u * 2 * np.pi,
            contributions[k] + 1,
            alpha=0.7,
            linestyle="--",
            color="black",
        )
    plt.plot(u * 2 * np.pi, baseline, label="baseline = 1")
    plt.plot(u * 2 * np.pi, Nphi, label=r"$S(s,\mathbf{w}) = 1 + \sum(w * p_{pair})$")
    # plt.axhline(0.0)
    plt.xlabel("s (rad)")
    plt.ylabel("Shape function value")
    # plt.title("Individual paired Bernstein bump contributions")
    plt.grid(True)

    plt.plot(
        [], [], linestyle="--", color="black", label="individual Bernstein polynomials"
    )
    plt.legend()
    plt.savefig("results/figures/torque2026/bernstein_bumps_contributions.pdf")
    # ---- Plot 2: total N_phi ----
    plt.figure(figsize=(5, 4))

    plt.xlabel("u (unit phase)")
    plt.ylabel("value")
    plt.title("Total shaping function N_phi (paired Bernstein bumps)")
    plt.grid(True)
    plt.legend()

    plt.show()


# -----------------------------
# Example usage
# -----------------------------
az_coeffs = np.random.uniform(-1, 1, 10)  # your az_coeffs
# az_coeffs[2] = 0.5
# az_coeffs[3] = -0.3
wp = 0.4  # width parameter (wp in the text)
repeat_phi = False  # repeat bumps?
normalize_bumps = False  # divide by wp?

plot_bumps_and_total(
    az_coeffs, wp, repeat_phi=repeat_phi, normalize_bumps=normalize_bumps
)
