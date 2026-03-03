import numpy as np
from awetrim.kinematics.reelin_parametrization import ParametrizedPatternsAngles
import casadi as ca


def periodic_cubic_bspline_matrices(u, M):
    """
    Periodic uniform cubic B-spline basis matrices on u in [0,1).

    u: (N,) numpy array in [0,1)
    M: number of periodic control points

    Returns DM matrices: B, dB/du, d2B/du2 with shape (N, M)
    """
    u = np.asarray(u).ravel()
    N = u.size

    B = np.zeros((N, M))
    dB = np.zeros((N, M))
    d2B = np.zeros((N, M))

    t = u * M  # in [0, M)
    i = np.floor(t).astype(int)
    l = t - i  # local coordinate in [0,1)

    # Cubic uniform B-spline blending functions (over 4 consecutive control points)
    b0 = ((1 - l) ** 3) / 6.0
    b1 = (3 * l**3 - 6 * l**2 + 4) / 6.0
    b2 = (-3 * l**3 + 3 * l**2 + 3 * l + 1) / 6.0
    b3 = (l**3) / 6.0

    # Derivatives wrt l
    db0 = -0.5 * (1 - l) ** 2
    db1 = 1.5 * l**2 - 2.0 * l
    db2 = -1.5 * l**2 + 1.0 * l + 0.5
    db3 = 0.5 * l**2

    d2b0 = 1 - l
    d2b1 = 3.0 * l - 2.0
    d2b2 = -3.0 * l + 1.0
    d2b3 = l

    # Chain rule: l = M*u - floor(M*u) -> dl/du = M (a.e.)
    dl_du = M

    j0 = (i - 1) % M
    j1 = (i + 0) % M
    j2 = (i + 1) % M
    j3 = (i + 2) % M

    for n in range(N):
        B[n, j0[n]] += b0[n]
        B[n, j1[n]] += b1[n]
        B[n, j2[n]] += b2[n]
        B[n, j3[n]] += b3[n]
        dB[n, j0[n]] += db0[n] * dl_du
        dB[n, j1[n]] += db1[n] * dl_du
        dB[n, j2[n]] += db2[n] * dl_du
        dB[n, j3[n]] += db3[n] * dl_du
        d2B[n, j0[n]] += d2b0[n] * (dl_du**2)
        d2B[n, j1[n]] += d2b1[n] * (dl_du**2)
        d2B[n, j2[n]] += d2b2[n] * (dl_du**2)
        d2B[n, j3[n]] += d2b3[n] * (dl_du**2)

    return ca.DM(B), ca.DM(dB), ca.DM(d2B)


class BSplineOffsets_Lissajous:
    def __init__(
        self,
        az_amp0,
        beta_amp0,
        beta0,
        beta_coeffs,  # length M_beta (CasADi SX/MX/DM)
        az_coeffs,  # length M_phi  (CasADi SX/MX/DM)
        s_grid,  # numpy array in [0, 2*pi)
        kappa=0.0,
        kbeta=0.0,
        left_first=True,
        downloops=True,
        zero_mean_offsets=True,  # keep your preference (optional)
        **kwargs,
    ):
        # super().__init__(
        #     az_amp0=az_amp0,
        #     beta_amp0=beta_amp0,
        #     beta0=beta0,
        #     kappa=kappa,
        #     kbeta=kbeta,
        #     beta_coeffs=beta_coeffs,
        #     az_coeffs=az_coeffs,
        #     **kwargs,
        # )
        self.az_amp0 = az_amp0
        self.beta_amp0 = beta_amp0
        self.r0 = 200
        self.kappa = kappa
        self.kbeta = kbeta
        self.beta0 = beta0
        self.omega = 1.0 if downloops else -1.0
        self.sgn = -1.0 if left_first else +1.0
        self.zero_mean_offsets = bool(zero_mean_offsets)

        self.az_coeffs = ca.vertcat(az_coeffs)
        self.beta_coeffs = ca.vertcat(beta_coeffs)

        self.M_phi = int(self.az_coeffs.numel())
        self.M_beta = int(self.beta_coeffs.numel())

        self.s_grid = np.asarray(s_grid).ravel()
        self.u_grid = (self.omega * self.s_grid / (2.0 * np.pi)) % 1.0

        # Basis matrices on the chosen grid
        self.B_phi, self.dB_phi, self.d2B_phi = periodic_cubic_bspline_matrices(
            self.u_grid, self.M_phi
        )
        self.B_beta, self.dB_beta, self.d2B_beta = periodic_cubic_bspline_matrices(
            self.u_grid, self.M_beta
        )

        # Chain rule u = omega*s/(2pi) mod 1
        self.du_ds = self.omega / (2.0 * np.pi)

        # Optional: enforce zero-mean offset on the grid (matches your "zero mean" preference)
        if self.zero_mean_offsets:
            self.B_phi = self.B_phi - ca.repmat(
                ca.mean1(self.B_phi), self.B_phi.size1(), 1
            )
            self.B_beta = self.B_beta - ca.repmat(
                ca.mean1(self.B_beta), self.B_beta.size1(), 1
            )
            # For derivatives, mean is already ~0, but no harm keeping as-is.

    def beta_center(self, r):
        return self.beta0 * (self.r0 / (self.r0 + (r - self.r0) * self.kbeta))

    def az_amp(self, r):
        return self.az_amp0 * (self.r0 / (self.r0 + (r - self.r0) * self.kappa))

    def beta_amp(self, r):
        return self.beta_amp0 * (self.r0 / (self.r0 + (r - self.r0) * self.kappa))

    # --- Grid outputs (vectors) ---
    def phi_class_grid(self, r):
        s = ca.DM(self.s_grid)
        a = self.az_amp(r)
        return self.sgn * a * ca.sin(self.omega * s)

    def beta_class_grid(self, r):
        s = ca.DM(self.s_grid)
        c = self.beta_center(r)
        b = self.beta_amp(r)
        return c + b * ca.sin(2.0 * self.omega * s)

    def dphi_spline_du_grid(self):
        return self.dB_phi @ self.az_coeffs

    def d2phi_spline_du2_grid(self):
        return self.d2B_phi @ self.az_coeffs

    def dbeta_spline_du_grid(self):
        return self.dB_beta @ self.beta_coeffs

    def d2beta_spline_du2_grid(self):
        return self.d2B_beta @ self.beta_coeffs

    def phi_grid(self, r):
        dphi = self.B_phi @ self.az_coeffs
        return self.phi_class_grid(r) + dphi

    def beta_grid(self, r):
        db = self.B_beta @ self.beta_coeffs
        return self.beta_class_grid(r) + db

    # --- Derivatives wrt s (nice for constraints) ---
    def dphi_ds_grid(self, r):
        # d/ds(phi_class) + (d/ds)(spline) = dphi_class/ds + du/ds * dphi_spline/du
        s = ca.DM(self.s_grid)
        a = self.az_amp(r)
        dphi_class_ds = self.sgn * a * self.omega * ca.cos(self.omega * s)
        dphi_spline_ds = self.du_ds * (self.dB_phi @ self.az_coeffs)
        return dphi_class_ds + dphi_spline_ds

    def d2phi_ds2_grid(self, r):
        s = ca.DM(self.s_grid)
        a = self.az_amp(r)
        d2phi_class_ds2 = self.sgn * a * (-(self.omega**2)) * ca.sin(self.omega * s)
        d2phi_spline_ds2 = (self.du_ds**2) * (self.d2B_phi @ self.az_coeffs)
        return d2phi_class_ds2 + d2phi_spline_ds2

    def dbeta_ds_grid(self, r):
        s = ca.DM(self.s_grid)
        b = self.beta_amp(r)
        dbeta_class_ds = b * (2.0 * self.omega) * ca.cos(2.0 * self.omega * s)
        dbeta_spline_ds = self.du_ds * (self.dB_beta @ self.beta_coeffs)
        return dbeta_class_ds + dbeta_spline_ds

    def d2beta_ds2_grid(self, r):
        s = ca.DM(self.s_grid)
        b = self.beta_amp(r)
        d2beta_class_ds2 = (
            b * (-((2.0 * self.omega) ** 2)) * ca.sin(2.0 * self.omega * s)
        )
        d2beta_spline_ds2 = (self.du_ds**2) * (self.d2B_beta @ self.beta_coeffs)
        return d2beta_class_ds2 + d2beta_spline_ds2

    import numpy as np


import matplotlib.pyplot as plt
import casadi as ca


def plot_pattern():
    N = 600
    s_grid = np.linspace(0, 2 * np.pi, N, endpoint=False)

    az_amp0 = 0.3
    beta_amp0 = 0.1
    beta0 = 0.45

    Mphi = 5
    Mbeta = 5

    # np.random.seed(2)
    az_c = ca.DM(0.1 * np.random.randn(Mphi))
    be_c = ca.DM(0.1 * np.random.randn(Mbeta))

    r = ca.DM(200.0)

    pat = BSplineOffsets_Lissajous(
        az_amp0=az_amp0,
        beta_amp0=beta_amp0,
        beta0=beta0,
        az_coeffs=az_c,
        beta_coeffs=be_c,
        s_grid=s_grid,
        left_first=True,
        downloops=True,
        zero_mean_offsets=False,
        kappa=0.0,
        kbeta=0.0,
    )

    phi = np.array(pat.phi_grid(r)).astype(float).ravel()
    beta = np.array(pat.beta_grid(r)).astype(float).ravel()

    dphi = np.array(pat.dphi_ds_grid(r)).astype(float).ravel()
    dbeta = np.array(pat.dbeta_ds_grid(r)).astype(float).ravel()

    plt.figure(figsize=(5, 4))
    plt.plot(phi, beta, linewidth=1.5)
    plt.xlabel("Azimuth φ (rad)")
    plt.ylabel("Elevation β (rad)")
    plt.grid(True)
    plt.title("Lissajous + periodic cubic spline offsets")
    plt.tight_layout()

    plt.figure(figsize=(5, 3.6))
    plt.plot(s_grid, dphi, label="dφ/ds")
    plt.plot(s_grid, dbeta, label="dβ/ds")
    plt.xlabel("s (rad)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.show()


plot_pattern()
