import numpy as np
import pytest

from awetrim.kinematics.parametrized_patterns import (
    create_pattern_from_dict,
    full_cycle_angles,
    make_bspline_path_parameters_from_named_curve,
    named_curve_angles,
    reelin_control_point_mask,
)


def _bump_path_parameters(M, bump_indices, base=0.3, peak=1.0):
    """Periodic-spline params with a flat elevation except a bump (reel-in arc)."""
    C_beta = np.full(M, base)
    C_beta[list(bump_indices)] = peak
    return {
        "M": M,
        "C_phi": np.zeros(M),
        "C_beta": C_beta,
        "s_init": 0.0,
        "s_final": 1.0,
        "downloops": True,
    }


def test_reelin_control_point_mask_flags_elevation_peak_support():
    """The mask flags the control points shaping the elevation peak (the
    reel-in arc of a full-cycle spline) plus their B-spline support, and
    leaves the far figure-eight points untouched."""
    M = 12
    mask = reelin_control_point_mask(_bump_path_parameters(M, [5, 6]))
    assert mask.shape == (M,) and mask.dtype == bool
    assert mask[5] and mask[6]  # the peak itself
    assert not mask[0] and not mask[11]  # far from the arc
    assert 0 < mask.sum() < M  # neither empty nor all-flagged


def test_reelin_control_point_mask_wraps_at_the_periodic_seam():
    """A peak at the s=0 seam flags neighbours on BOTH sides of the wrap."""
    M = 12
    mask = reelin_control_point_mask(_bump_path_parameters(M, [0]))
    assert mask[0] and mask[1] and mask[11]
    assert not mask[6]


def test_full_cycle_angles_psi0_shifts_phase_and_preserves_periodicity():
    """psi0 offsets the figure phase without breaking exact periodicity.

    The curvature tuner in fit_periodic_cycle_config relies on psi0 to realign
    the reel-in fade with a fast part of the figure-eight; a constant phase
    offset must keep the curve 1-periodic (psi advances by 2*pi*n_loops over
    the period regardless of psi0) and reduce to the psi0=0 curve when zero.
    """
    s = np.linspace(0.0, 1.0, 400, endpoint=False)
    kwargs = dict(n_loops=3, reelout_fraction=0.65, beta0=0.35, beta_amp0=0.14,
                  az_amp0=0.36, beta_reelin_peak=1.2, az_reelin_amp=-0.5,
                  ramp_fraction=0.45, downloops=True)

    az_default, el_default = full_cycle_angles(s, **kwargs)
    az_zero, el_zero = full_cycle_angles(s, psi0=0.0, **kwargs)
    assert np.allclose(az_default, az_zero) and np.allclose(el_default, el_zero)

    az, el = full_cycle_angles(np.array([0.0, 1.0]), psi0=1.3, **kwargs)
    assert np.isclose(az[0], az[1]) and np.isclose(el[0], el[1])

    az_shift, _ = full_cycle_angles(s, psi0=1.3, **kwargs)
    assert not np.allclose(az_shift, az_zero)


def test_create_pattern_from_dict_rejects_unsupported_type():
    """A type with no constructor (e.g. cst_helix, the cycle-config default)
    must raise a clear ValueError listing supported types, not a KeyError."""
    with pytest.raises(ValueError, match="Unknown or unsupported pattern type"):
        create_pattern_from_dict("cst_helix", {})


def test_create_pattern_from_dict_reports_missing_params():
    """A supported type with missing params reports them explicitly."""
    with pytest.raises(ValueError, match="Missing required parameters"):
        create_pattern_from_dict("spline_open", {"M": 6})


def test_named_curve_angles_support_lissajous_and_helix():
    s = np.linspace(0.0, 2.0 * np.pi, 5)

    phi_lissajous, beta_lissajous = named_curve_angles(
        s,
        curve_type="lissajous",
        az_amp0=0.32,
        beta0=0.3,
        beta_amp0=0.15,
    )
    phi_helix, beta_helix = named_curve_angles(
        s,
        curve_type="helix",
        az_amp0=0.32,
        beta0=0.3,
        beta_amp0=0.15,
    )

    assert phi_lissajous.shape == s.shape
    assert beta_lissajous.shape == s.shape
    assert phi_helix.shape == s.shape
    assert beta_helix.shape == s.shape
    assert not np.allclose(beta_lissajous, beta_helix)


def test_named_curve_angles_lemniscate_is_amplitude_matched_figure_eight():
    """The Bernoulli lemniscate is a drop-in for the lissajous eight: same
    azimuth/elevation amplitudes, starting at the centre crossing."""
    s = np.linspace(0.0, 2.0 * np.pi, 2001)
    az_amp0, beta0, beta_amp0 = 0.32, 0.3, 0.15

    phi, beta = named_curve_angles(
        s,
        curve_type="lemniscate",
        az_amp0=az_amp0,
        beta0=beta0,
        beta_amp0=beta_amp0,
    )

    assert np.isclose(np.max(np.abs(phi)), az_amp0, atol=1e-6)
    assert np.isclose(np.max(np.abs(beta - beta0)), beta_amp0, atol=1e-6)
    # starts at the centre crossing and closes periodically there
    assert np.isclose(phi[0], 0.0, atol=1e-9)
    assert np.isclose(beta[0], beta0, atol=1e-9)
    assert np.isclose(phi[-1], 0.0, atol=1e-6)

    phi_liss, _ = named_curve_angles(
        s, curve_type="lissajous", az_amp0=az_amp0, beta0=beta0, beta_amp0=beta_amp0
    )
    assert not np.allclose(phi, phi_liss)


def test_named_curve_angles_rejects_unknown_curve():
    with pytest.raises(ValueError, match="lissajous"):
        named_curve_angles(np.array([0.0, 1.0]), curve_type="spiral")


@pytest.mark.parametrize("curve_type", ["lissajous", "lemniscate"])
def test_periodic_uploop_fit_matches_runtime(curve_type):
    """A reversed (uploop) fit must reproduce the curve once rebuilt the way
    create_pattern_from_dict does at sim time. Regression for the singular
    reversed-grid (u in [-1, 0]) basis matrix and the missing ``downloops``
    passthrough that left uploops evaluated in the downloop sense."""
    az_amp0, beta0, beta_amp0 = 0.32, 0.3, 0.15
    path_parameters = make_bspline_path_parameters_from_named_curve(
        spline_type="periodic",
        M=10,
        r0=230.0,
        s_init=0.0,
        s_final=2.0 * np.pi,
        n_fit=200,
        curve_type=curve_type,
        az_amp0=az_amp0,
        beta0=beta0,
        beta_amp0=beta_amp0,
        downloops=False,
    )
    assert path_parameters["downloops"] is False

    pattern = create_pattern_from_dict("spline_periodic", path_parameters)

    s = np.linspace(0.0, 2.0 * np.pi, 200, endpoint=True)
    az_target, el_target = named_curve_angles(
        s,
        curve_type=curve_type,
        az_amp0=az_amp0,
        beta0=beta0,
        beta_amp0=beta_amp0,
        downloops=False,
    )
    az_fit = np.array([float(pattern.azimuth(230.0, sv)) for sv in s])
    el_fit = np.array([float(pattern.elevation(230.0, sv)) for sv in s])

    assert np.max(np.abs(az_fit - az_target)) < 5e-2
    assert np.max(np.abs(el_fit - el_target)) < 5e-2


def test_make_periodic_bspline_path_parameters_are_pattern_ready():
    path_parameters = make_bspline_path_parameters_from_named_curve(
        spline_type="periodic",
        M=10,
        r0=230.0,
        s_init=0.0,
        s_final=2.0 * np.pi,
        n_fit=80,
        curve_type="helix",
        az_amp0=0.32,
        beta0=0.3,
        beta_amp0=0.15,
    )

    assert path_parameters["M"] == 10
    assert len(path_parameters["C_phi"]) == 10
    assert len(path_parameters["C_beta"]) == 10

    pattern = create_pattern_from_dict("spline_periodic", path_parameters)

    assert pattern.M == 10


def test_make_open_bspline_path_parameters_are_pattern_ready():
    path_parameters = make_bspline_path_parameters_from_named_curve(
        spline_type="open",
        M=6,
        r0=230.0,
        s_init=0.0,
        s_final=1.0,
        n_fit=40,
        curve_type="lissajous",
        az_amp0=0.32,
        beta0=0.3,
        beta_amp0=0.15,
    )

    assert path_parameters["M"] == 6
    assert len(path_parameters["C_phi"]) == 6
    assert len(path_parameters["C_beta"]) == 6

    pattern = create_pattern_from_dict("spline_open", path_parameters)

    assert pattern.M == 6
