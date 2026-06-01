import numpy as np


def setup_tracking_arrays(n_pts, t_vector, n_panels=0):
    """
    Initialize tracking arrays for simulation results.

    Args:
        n_pts (int): Number of nodes/particles.
        t_vector (np.ndarray): Array of time steps.
        n_panels (int): Number of aerodynamic panels (0 skips aero tracking arrays).

    Returns:
        dict: Dictionary with preallocated arrays for positions, forces, and tracking metrics.
    """
    nt = len(t_vector)
    arrays = {
        "positions": np.zeros((nt, n_pts, 3)),
        "f_ext": np.zeros((nt, n_pts, 3)),
        "f_int": np.zeros((nt, n_pts, 3)),
        "residual_norm": np.zeros(nt),
        "max_residual": np.zeros(nt),
    }
    if n_panels > 0:
        arrays["alpha_at_ac"] = np.full((nt, n_panels), np.nan)
        arrays["stall_mask"] = np.zeros((nt, n_panels), dtype=bool)
    return arrays


def update_aero_tracking(tracking_data, idx, alpha_at_ac, stall_mask):
    """Store per-panel aerodynamic state for one iteration.

    Args:
        tracking_data: Tracking dict (must contain 'alpha_at_ac' and 'stall_mask').
        idx: Current iteration index.
        alpha_at_ac: Per-panel local AoA [rad], shape (n_panels,) or None.
        stall_mask: Boolean stall flag per panel, shape (n_panels,) or None.
    """
    if "alpha_at_ac" not in tracking_data or alpha_at_ac is None:
        return
    alpha = np.ravel(alpha_at_ac)
    n = tracking_data["alpha_at_ac"].shape[1]
    tracking_data["alpha_at_ac"][idx, : min(len(alpha), n)] = alpha[: n]
    if stall_mask is not None:
        mask = np.ravel(stall_mask).astype(bool)
        tracking_data["stall_mask"][idx, : min(len(mask), n)] = mask[: n]


def update_tracking_arrays(
    tracking_data,
    idx,
    struc_nodes,
    f_ext_flat,
    f_int_flat,
):
    """
    Update tracking arrays with simulation results for a single time step.

    Args:
        tracking_data (dict): Tracking arrays to update.
        idx (int): Current time step index.
        pos3d (np.ndarray): Current 3D positions (n_nodes, 3).
        f_ext_flat (np.ndarray): Flattened external force vector (n_nodes*3,).
        f_int_flat (np.ndarray): Flattened internal force vector (n_nodes*3,).

    Returns:
        None. Updates tracking_data in place.
    """
    # Unpack 3D storage
    pos3d = tracking_data["positions"]
    ext3d = tracking_data["f_ext"]
    int3d = tracking_data["f_int"]

    n_pts = pos3d.shape[1]

    # 1) Positions
    pos3d[idx] = struc_nodes

    # 2) External & internal forces: reshape before storing
    ext3d[idx] = f_ext_flat.reshape(n_pts, 3)
    int3d[idx] = f_int_flat.reshape(n_pts, 3)

    # 3) Norms
    tracking_data["residual_norm"][idx] = np.linalg.norm(f_int_flat)
    tracking_data["max_residual"][idx] = np.max(np.abs(f_int_flat))
