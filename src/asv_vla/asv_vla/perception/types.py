"""Perception / tracker types. No Isaac, no ROS."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Detection:
    x: float
    y: float
    rho: float = 0.0
    appearance: np.ndarray | None = None
    stamp: float = 0.0
    # Timestamp of the source image. ``stamp`` may be translated to the
    # tracker's local clock, but this value is never rewritten.
    source_stamp: float | None = None


@dataclass
class Track:
    id: int
    x: float
    y: float
    vx: float
    vy: float
    P: np.ndarray
    rho: float = 0.0
    appearance: np.ndarray | None = None
    misses: int = 0
    hits: int = 1

    def copy(self) -> "Track":
        return Track(
            id=self.id,
            x=self.x,
            y=self.y,
            vx=self.vx,
            vy=self.vy,
            P=np.array(self.P, dtype=np.float64, copy=True),
            rho=self.rho,
            appearance=None if self.appearance is None else np.array(self.appearance, copy=True),
            misses=self.misses,
            hits=self.hits,
        )

    def state(self) -> np.ndarray:
        return np.array([self.x, self.y, self.vx, self.vy], dtype=np.float64)


@dataclass
class Snapshot:
    t: float
    yaw_rate: float
    ego_velocity: tuple[float, float] = (0.0, 0.0)
    tracks: list[Track] = field(default_factory=list)
