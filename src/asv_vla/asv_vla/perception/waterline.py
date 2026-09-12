"""Project a detection's waterline to body-frame position."""

from __future__ import annotations

import math

from .camera import (
    CAMERA_FOCAL_PX,
    CAMERA_HEIGHT,
    CAMERA_WIDTH,
    MAX_RANGE_M,
    camera_world_from_ego,
    yaw_from_wxyz,
)


def range_from_bbox(
    cx: float,
    cy: float,
    w: float,
    h: float,
    *,
    ego_pos=(0.0, 0.0, 0.0),
    ego_quat=(1.0, 0.0, 0.0, 0.0),
    width: int = CAMERA_WIDTH,
    height: int = CAMERA_HEIGHT,
    max_range: float = MAX_RANGE_M,
) -> tuple[float, float] | None:
    u, v = float(cx), float(cy) + 0.5 * float(h)
    cam_pos, (x_axis, y_axis, z_axis) = camera_world_from_ego(ego_pos, ego_quat)
    d_cam = (
        (u - 0.5 * float(width)) / CAMERA_FOCAL_PX,
        -(v - 0.5 * float(height)) / CAMERA_FOCAL_PX,
        -1.0,
    )
    d_w = (
        x_axis[0] * d_cam[0] + y_axis[0] * d_cam[1] + z_axis[0] * d_cam[2],
        x_axis[1] * d_cam[0] + y_axis[1] * d_cam[1] + z_axis[1] * d_cam[2],
        x_axis[2] * d_cam[0] + y_axis[2] * d_cam[1] + z_axis[2] * d_cam[2],
    )
    if abs(d_w[2]) < 1.0e-12:
        return None
    t = -float(cam_pos[2]) / float(d_w[2])
    if t <= 1.0e-4:
        return None
    hit = (
        float(cam_pos[0]) + t * d_w[0],
        float(cam_pos[1]) + t * d_w[1],
        0.0,
    )
    look = (-z_axis[0], -z_axis[1], -z_axis[2])
    if sum((hit[i] - cam_pos[i]) * look[i] for i in range(3)) <= 0.0:
        return None
    if math.sqrt(sum((hit[i] - cam_pos[i]) ** 2 for i in range(3))) > float(max_range):
        return None
    dx = hit[0] - float(ego_pos[0])
    dy = hit[1] - float(ego_pos[1])
    yaw = yaw_from_wxyz(ego_quat)
    c, s = math.cos(yaw), math.sin(yaw)
    return c * dx + s * dy, -s * dx + c * dy
