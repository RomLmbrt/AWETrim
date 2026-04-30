"""Named symbolic outputs for SystemModel.

These expressions are derived from the model context and are intended for
post-processing, constraints, and CasADi function extraction.
"""


def build_expression_registry(model):
    return {
        "force_aerodynamic": lambda: model.kite.force_aerodynamic(model),
        "force_gravity": lambda: model.kite.force_gravity_for(model),
        "force_gravity_wing": lambda: model.kite.force_gravity_wing_for(model),
        "force_gravity_kcu": lambda: model.kite.force_gravity_kcu_for(model),
        "force_tether_at_kite": lambda: model.tether.force_tether_at_kite_for(model),
        "drag_tether_at_kite": lambda: model.tether.drag_tether_at_kite_for(model),
        "force_gravity_tether_at_kite": lambda: (
            model.tether.force_gravity_tether_at_kite_for(model)
        ),
        "mass_tether": lambda: model.tether.mass_tether_for(model),
        "tension_kite": lambda: model.tether.tension_kite_for(model),
        "angle_of_attack": lambda: model.kite.angle_of_attack_for(model),
        "pitch_bridle": lambda: model.kite.pitch_bridle_for(model),
        "roll_bridle": lambda: model.kite.roll_bridle_for(model),
        "angle_roll_aerodynamic": lambda: model.kite.angle_roll_aerodynamic_for(model),
        "lift_coefficient": lambda: model.kite.lift_coefficient_for(model),
        "drag_coefficient": lambda: model.kite.drag_coefficient_for(model),
    }
