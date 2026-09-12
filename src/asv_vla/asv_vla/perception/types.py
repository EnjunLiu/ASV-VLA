from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# 单帧图像检测结果
@dataclass
class Detection:
    x: float
    y: float
    appearance: np.ndarray | None = None # 从 OWL-ViT 内部提取的 512 维图像特征
    stamp: float = 0.0

# 连续帧跟踪结果
@dataclass
class Track:
    id: int
    x: float
    y: float
    vx: float
    vy: float
    P: np.ndarray
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
