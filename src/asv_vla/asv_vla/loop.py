"""Runtime perception and policy loop; deliberately contains no controller."""

from __future__ import annotations

import math

import numpy as np

from .perception import Detection, KalmanTracker, owl_to_detections
from .types import PerceptionTick, SensorState

COLLISION_M = 1.0


def _clip_action(ax: float, ay: float, cap: float) -> tuple[float, float]:
    norm = math.hypot(ax, ay)
    if norm <= cap or norm < 1.0e-12:
        return float(ax), float(ay)
    scale = float(cap) / norm
    return float(ax) * scale, float(ay) * scale


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


class PerceptionLoop:
    def __init__(self, actor) -> None:
        self.tracker = KalmanTracker()
        self.actor = actor

    def step(self, sensors: SensorState, detections: list[Detection] | None) -> PerceptionTick:
        ego_velocity = (float(sensors.surge_velocity), float(sensors.sway_velocity))
        self.tracker.step(
            sensors.t,
            sensors.yaw_rate,
            detections,
            ego_velocity=ego_velocity,
        )
        raw_entities = self.tracker.raw_entity_matrix(ego_velocity=ego_velocity)
        self.actor.set_track_metadata(
            [track.id for track in self.tracker.tracks],
            [track.misses for track in self.tracker.tracks],
            [track.hits for track in self.tracker.tracks],
        )
        ax, ay = self.actor(raw_entities, self.actor.task_embed)
        entities = np.asarray(self.actor.last_entities, dtype=np.float64)
        probabilities = np.asarray(self.actor.last_target_probabilities, dtype=np.float64)
        actor_raw = np.asarray(self.actor.last_raw_entities, dtype=np.float64)
        if probabilities.size and actor_raw.ndim == 2:
            selected = int(np.argmax(probabilities))
            if selected < len(actor_raw) and probabilities[selected] > self.actor.last_null_probability:
                ax, ay = _limit_inward_action(
                    ax,
                    ay,
                    actor_raw[selected, :2],
                    float(getattr(self.actor, "safety_stop_range", 1.35)),
                    float(getattr(self.actor, "safety_full_range", 2.40)),
                )
        ax, ay = _clip_action(ax, ay, float(self.actor.max_action))
        image_t = float(sensors.t if sensors.image_stamp is None else sensors.image_stamp)
        collision = any(math.hypot(float(row[0]), float(row[1])) < COLLISION_M for row in entities)
        return PerceptionTick(
            t=float(sensors.t),
            entities=entities,
            raw_entities=actor_raw,
            action=(ax, ay),
            collision=collision,
            image_t=image_t,
            detection_t=(
                min(float(d.stamp if d.source_stamp is None else d.source_stamp) for d in detections)
                if detections else None
            ),
        )


def detections_from_boxes(boxes, sensors: SensorState):
    return owl_to_detections(
        boxes,
        ego_pos=sensors.ego_pos,
        ego_quat=sensors.ego_quat,
        stamp=sensors.t,
    )
