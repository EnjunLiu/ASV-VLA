from __future__ import annotations

import math

import numpy as np

from .perception.owl import owl_to_detections
from .perception.tracker import KalmanTracker
from .perception.types import Detection
from .types import SensorState

SAFETY_STOP_M = 1.35
SAFETY_FULL_M = 2.40


# 把期望位移的模长限制在 cap 以内
def _clip_action(ax: float, ay: float, cap: float) -> tuple[float, float]:
    norm = math.hypot(ax, ay)
    if norm <= cap or norm < 1.0e-12:
        return float(ax), float(ay)
    scale = float(cap) / norm
    return float(ax) * scale, float(ay) * scale


# 靠近目标时削掉朝目标走的位移
def _limit_inward_action(ax: float, ay: float, target_xy, stop=1.35, full=2.40):
    px, py = float(target_xy[0]), float(target_xy[1])
    distance = math.hypot(px, py)
    if distance < 1.0e-9 or full <= stop:
        return float(ax), float(ay)
    ux, uy = px / distance, py / distance
    inward = float(ax) * ux + float(ay) * uy
    if inward <= 0.0:
        return float(ax), float(ay)
    scale = float(np.clip((distance - stop) / (full - stop), 0.0, 1.0))
    removed = inward * (1.0 - scale)
    return float(ax) - removed * ux, float(ay) - removed * uy


# 感知-策略闭环
class PerceptionLoop:
    def __init__(self, actor) -> None:
        self.tracker = KalmanTracker()
        self.actor = actor

    # 单拍：更新航迹、求期望位移，再做近距内向限幅
    def step(self, sensors: SensorState, detections: list[Detection] | None):
        ego_velocity = (float(sensors.surge_velocity), float(sensors.sway_velocity))
        self.tracker.step(
            sensors.t,
            sensors.yaw_rate,
            detections,
            ego_velocity=ego_velocity,
        )
        raw_entities = self.tracker.raw_entity_matrix(ego_velocity=ego_velocity)
        # 航迹身份只给演员做锁定，不进网络输入
        self.actor.set_track_metadata(
            [track.id for track in self.tracker.tracks],
            [track.misses for track in self.tracker.tracks],
            [track.hits for track in self.tracker.tracks],
        )
        ax, ay = self.actor(raw_entities)
        probabilities = np.asarray(self.actor.last_target_probabilities, dtype=np.float64)
        actor_raw = np.asarray(self.actor.last_raw_entities, dtype=np.float64)
        # 选中真实目标后，近距削掉朝目标走的位移
        if probabilities.size and actor_raw.ndim == 2:
            selected = int(np.argmax(probabilities))
            if selected < len(actor_raw) and probabilities[selected] > self.actor.last_null_probability:
                ax, ay = _limit_inward_action(
                    ax,
                    ay,
                    actor_raw[selected, :2],
                    SAFETY_STOP_M,
                    SAFETY_FULL_M,
                )
        ax, ay = _clip_action(ax, ay, float(self.actor.max_action))
        return len(actor_raw), (ax, ay)


# 把 OWL 框转成船体坐标系 Detection
def detections_from_boxes(boxes, sensors: SensorState):
    return owl_to_detections(
        boxes,
        ego_pos=sensors.ego_pos,
        ego_quat=sensors.ego_quat,
        stamp=sensors.t,
    )
