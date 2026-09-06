from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SensorState:
    t: float
    yaw_rate: float = 0.0
    surge_velocity: float = 0.0
    sway_velocity: float = 0.0
    ego_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    ego_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    image_stamp: float | None = None


@dataclass
class PerceptionTick:
    t: float
    entities: np.ndarray
    raw_entities: np.ndarray
    action: tuple[float, float]
    collision: bool
    image_t: float
    detection_t: float | None
