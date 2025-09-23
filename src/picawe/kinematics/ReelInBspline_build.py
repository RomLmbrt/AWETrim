import numpy as np
import casadi as ca

# -------------------------------
# Spline Building
# -------------------------------
class ReelInBspline_build():

    # something0 or somethingf means the start or end 0 for start and f for final
    # p - point eg. p0 start point
    # v - velocity
    # crs - course
    # idx - index
    # cyc - cycle
    # ri - reel-ina
    # ro - reel-out
    # sph - spherical
    # cart - cartesian

    def build_bspline(self, C, p, U, u, return_derivative=False):
        n_ctrl = C.shape[0]

        def N(i, k, u_val):
            if k == 0:
                return ca.if_else(
                    ca.logic_and(U[i] <= u_val, u_val <= U[i+1]),
                    1.0,
                    0.0
                )
            left = 0
            right = 0
            if U[i+k] > U[i]:
                left = (u_val - U[i])/(U[i+k]-U[i]) * N(i, k-1, u_val)
            if U[i+k+1] > U[i+1]:
                right = (U[i+k+1]-u_val)/(U[i+k+1]-U[i+1]) * N(i+1, k-1, u_val)
            return left + right

        Nvec = [N(i, p, u) for i in range(n_ctrl)]
        Nvec = ca.vertcat(*Nvec).T
        S = ca.mtimes(Nvec, C)

        dS = None
        if return_derivative:
            dS = ca.jacobian(S, u)

        # Nmat = ca.jacobian(S, C)
        # print(f"Nmat shape: {Nmat.shape}")

        return S, dS, Nvec

    # -------------------------------
    # Basis matrix construction
    # -------------------------------

    def build_Nmat(self, U, p, u_vals):
        """
        Compute B-spline basis function matrix Nmat for all u_vals.
        
        Parameters
        ----------
        U : array-like
            Knot vector
        p : int
            Degree
        u_vals : array-like
            Parameter values (len(u_vals) = n_points)
        
        Returns
        -------
        Nmat : np.ndarray
            Shape (len(u_vals), n_ctrl)
        """
        n_ctrl = len(U) - p - 1
        Nmat_list = []
        for u in u_vals:
            _, _, Nvec = self.build_bspline(np.zeros((n_ctrl,1)), p, U, u, return_derivative=False)
            Nmat_list.append(Nvec.full())
        Nmat = np.array(Nmat_list).squeeze()
        return Nmat  # shape (len(u_vals), n_ctrl)



    # -------------------------------
    # Evaluation with CasADi function
    # -------------------------------
    def eval_spline(self, spline_func, C_val, u_val):
        """
        Evaluate a CasADi spline function.

        Parameters
        ----------
        spline_func : casadi.Function
            CasADi function representing the spline
        C_val : numpy array
            Control points
        u_val : float or array-like
            Parameter value(s) at which to evaluate the spline

        Returns
        -------
        numpy array
            Spline evaluated at u_val
        """
        # Make sure u_val is an array for vectorized evaluation
        u_val = np.atleast_1d(u_val)
        S_eval = np.array([spline_func(C=C_val, u=ui)["S"].full().flatten() for ui in u_val])
        dS_eval = np.array([spline_func(C=C_val, u=ui)["dS"].full().flatten() for ui in u_val])

        if S_eval.shape[0] == 1:
            return S_eval[0], dS_eval[0]
        return S_eval, dS_eval
