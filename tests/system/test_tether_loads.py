"""Regression tests for tether load transfer to the kite.

Link tethers model the tether as a massless/dragless rigid or elastic link, so
their drag and gravity at the kite are zero. Defining these (rather than
leaving them missing) keeps the ``drag_tether_at_kite`` /
``force_gravity_tether_at_kite`` expressions and ``tension_tether_equation``
valid for every tether type; lumped/distributed tethers override them.
"""

import casadi as ca
import numpy as np

from awetrim.system.system_model import SystemModel
from awetrim.system.tether import (
    FlexibleLinkTether,
    RigidLinkTether,
    RigidLumpedTether,
)


def _is_zero_expr(expr) -> bool:
    """True if a free-variable-free CasADi expression is identically zero."""
    return np.allclose(np.array(ca.DM(ca.evalf(expr))).ravel(), 0.0)


def test_link_tethers_have_zero_drag_and_gravity_at_kite():
    model = SystemModel()  # default FlexibleLinkTether
    for tether in (FlexibleLinkTether(), RigidLinkTether()):
        assert _is_zero_expr(tether.drag_tether_at_kite_for(model))
        assert _is_zero_expr(tether.force_gravity_tether_at_kite_for(model))


def test_tension_tether_equation_builds_with_default_tether():
    """tension_tether_equation reads the drag/gravity-at-kite expressions; with
    the default link tether these are zero rather than missing (regression for
    the AttributeError that previously broke this with FlexibleLinkTether)."""
    model = SystemModel()
    equation = model.tension_tether_equation
    assert isinstance(equation, ca.MX)
    assert _is_zero_expr(model.expression("drag_tether_at_kite"))
    assert _is_zero_expr(model.expression("force_gravity_tether_at_kite"))


def test_lumped_tether_contributes_nonzero_drag_at_kite():
    """Sanity check the override path: a lumped tether's drag is a real
    (state-dependent) expression, not the zero default."""
    model = SystemModel(tether=RigidLumpedTether())
    drag = model.expression("drag_tether_at_kite")
    assert len(ca.symvar(drag)) > 0
