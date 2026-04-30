from dataclasses import asdict, dataclass
from typing import Optional


@dataclass
class State:
    distance_radial: float = None
    angle_elevation: float = None
    angle_azimuth: float = None
    angle_course: float = None
    speed_radial: float = None
    speed_tangential: float = None
    input_depower: float = None
    input_steering: float = None
    timeder_angle_course: float = None
    length_tether: float = None
    tension_tether_ground: float = None
    timeder_speed_tangential: Optional[float] = None
    timeder_speed_radial: Optional[float] = None
    # Optional inputs
    angle_roll: Optional[float] = None
    angle_pitch: Optional[float] = None
    angle_yaw: Optional[float] = None

    # Optional outputs
    angle_of_attack: Optional[float] = None
    lift_coefficient: Optional[float] = None
    drag_coefficient: Optional[float] = None
    speed_apparent_wind: Optional[float] = None
    # Parametrization
    s: Optional[float] = None
    s_dot: Optional[float] = None
    s_ddot: Optional[float] = None
    t: Optional[float] = None

    def to_dict(self):
        return asdict(self)
