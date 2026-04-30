from __future__ import annotations

from typing import Protocol, Union

import casadi as ca

Symbolic = Union[ca.MX, ca.SX, ca.DM]


class KiteModel(Protocol):
    """Interface required by SystemModel for kite component equations."""

    mass_wing: float
    mass_kcu: float
    input_steering: Symbolic
    input_depower: Symbolic
    g: float
    rho: float
    steering_control: str

    def force_aerodynamic(self, model: "SystemModelProtocol") -> Symbolic: ...

    def force_gravity_for(self, model: "SystemModelProtocol") -> Symbolic: ...

    def force_gravity_wing_for(self, model: "SystemModelProtocol") -> Symbolic: ...

    def force_gravity_kcu_for(self, model: "SystemModelProtocol") -> Symbolic: ...

    def angle_of_attack_for(self, model: "SystemModelProtocol") -> Symbolic: ...

    def pitch_bridle_for(self, model: "SystemModelProtocol") -> Symbolic: ...

    def roll_bridle_for(self, model: "SystemModelProtocol") -> Symbolic: ...

    def angle_roll_aerodynamic_for(
        self, model: "SystemModelProtocol"
    ) -> Symbolic: ...

    def lift_coefficient_for(self, model: "SystemModelProtocol") -> Symbolic: ...

    def drag_coefficient_for(self, model: "SystemModelProtocol") -> Symbolic: ...


class TetherModel(Protocol):
    """Interface required by SystemModel for tether component equations."""

    is_tether_rigid: bool

    def mass_tether_for(self, model: "SystemModelProtocol") -> Symbolic: ...

    def force_tether_at_kite_for(self, model: "SystemModelProtocol") -> Symbolic: ...

    def tension_kite_for(self, model: "SystemModelProtocol") -> Symbolic: ...

    def drag_tether_at_kite_for(self, model: "SystemModelProtocol") -> Symbolic: ...

    def force_gravity_tether_at_kite_for(
        self, model: "SystemModelProtocol"
    ) -> Symbolic: ...


class WindModel(Protocol):
    """Interface required by SystemModel for wind component equations."""

    speed_wind_ref: Symbolic

    def velocity_wind(self, model: "SystemModelProtocol") -> Symbolic: ...

    def velocity_wind_at_height(
        self, model: "SystemModelProtocol", height: Symbolic
    ) -> Symbolic: ...


class SystemModelProtocol(Protocol):
    """Structural protocol for the symbolic model context passed to components."""

    kite: KiteModel
    tether: TetherModel
    wind: WindModel
    distance_radial: Symbolic
    angle_elevation: Symbolic
    angle_azimuth: Symbolic
    angle_course: Symbolic
    speed_tangential: Symbolic
    speed_radial: Symbolic
    input_steering: Symbolic
    input_depower: Symbolic
    tension_tether_ground: Symbolic
    velocity_apparent_wind: Symbolic
    acceleration: Symbolic
    force_gravity_kcu: Symbolic
    g: float
    rho: float
