import casadi as ca

from awetrim.kinematics.Kinematics import ParametrizedKinematics


class DummyPattern:
    def elevation(self, r, s):
        return 0.2 + 0.001 * r + 0.01 * s

    def azimuth(self, r, s):
        return 0.1 + 0.002 * r + 0.02 * s


class DummyKiteModel:
    def __init__(self, distance_radial, speed_radial):
        self.distance_radial = distance_radial
        self.speed_radial = speed_radial


class DummyPhase:
    def __init__(self, s, kite_model, s_dot, s_ddot):
        self.s = s
        self.kite_model = kite_model
        self.s_dot = s_dot
        self.s_ddot = s_ddot


def test_vtau_relation():
    # Create CasADi symbols for inputs
    s = ca.MX.sym("s")
    r = ca.MX.sym("r")
    vr = ca.MX.sym("vr")
    s_dot = ca.MX.sym("s_dot")
    s_ddot = ca.MX.sym("s_ddot")

    pattern = DummyPattern()
    kite_model = DummyKiteModel(r, vr)
    phase = DummyPhase(s, kite_model, s_dot, s_ddot)

    pk = ParametrizedKinematics(pattern, phase)

    expr = pk.vk**2 - pk.vr**2 - pk.vtau**2

    # Create a CasADi function and evaluate at representative numeric values
    f = ca.Function("check", [s, r, vr, s_dot, s_ddot], [expr])

    val = float(f(0.5, 50.0, 1.0, 0.2, 0.0)[0])  # choose values where vtau is real

    assert abs(val) < 1e-6
