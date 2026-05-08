from dataclasses import asdict, dataclass
from typing import Any, Optional


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

    # Optional outputs — ROM and VSM
    angle_of_attack: Optional[float] = None
    lift_coefficient: Optional[float] = None
    drag_coefficient: Optional[float] = None
    side_force_coefficient: Optional[float] = None
    speed_apparent_wind: Optional[float] = None
    # Parametrization
    s: Optional[float] = None
    s_dot: Optional[float] = None
    s_ddot: Optional[float] = None
    t: Optional[float] = None
    # VSM / aerostructural outputs — None when filled by ROM solver
    lift_distribution: Optional[Any] = None           # np.ndarray (n_panels,)
    drag_distribution: Optional[Any] = None           # np.ndarray (n_panels,)
    circulation_distribution: Optional[Any] = None    # np.ndarray (n_panels,)
    angle_of_attack_sections: Optional[Any] = None    # np.ndarray (n_panels,)
    loaded_geometry: Optional[Any] = None             # np.ndarray (n_nodes, 3), PSS output

    def to_dict(self):
        return asdict(self)
