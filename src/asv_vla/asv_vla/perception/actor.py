"""Actor stub for slice 3: always a=(0,0). Do not load AttentionActor."""

from __future__ import annotations


class ZeroActor:
    max_action = 0.5

    def __call__(self, entities, task_embedding=None) -> tuple[float, float]:
        return (0.0, 0.0)
