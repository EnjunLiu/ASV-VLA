"""Bbox bottom-center ∩ still water z=0 → body-frame XY (howto 1.3.1)."""

from __future__ import annotations

import math

from .camera import (
    CAMERA_HEIGHT,
    CAMERA_WIDTH,
    MAX_RANGE_M,
    WATER_Z,
    camera_fx_fy,
    camera_world_from_ego,
    yaw_from_wxyz,
)


def bbox_bottom_center(cx: float, cy: float, w: float, h: float) -> tuple[float, float]:
    """Image v grows downward; waterline is the bottom edge midpoint."""
    return float(cx), float(cy) + 0.5 * float(h)


def pixel_dir_camera(u: float, v: float, fx: float, fy: float, width: int, height: int):
    """Direction in USD camera frame: +X right, +Y up, -Z look."""
    cx = 0.5 * float(width)
    cy = 0.5 * float(height)
    xn = (float(u) - cx) / float(fx)
    yn = (float(v) - cy) / float(fy)
    return (xn, -yn, -1.0)


def _dir_world(d_cam, cam_axes):
    x_axis, y_axis, z_axis = cam_axes
    return (
        x_axis[0] * d_cam[0] + y_axis[0] * d_cam[1] + z_axis[0] * d_cam[2],
        x_axis[1] * d_cam[0] + y_axis[1] * d_cam[1] + z_axis[1] * d_cam[2],
        x_axis[2] * d_cam[0] + y_axis[2] * d_cam[1] + z_axis[2] * d_cam[2],
    )


def world_to_body_xy(hit_xy, ego_xy, yaw: float) -> tuple[float, float]:
    dx = float(hit_xy[0]) - float(ego_xy[0])
    dy = float(hit_xy[1]) - float(ego_xy[1])
    c = math.cos(float(yaw))
    s = math.sin(float(yaw))
    return c * dx + s * dy, -s * dx + c * dy


def range_from_pixel(
    u: float,
    v: float,
    *,
    ego_pos=(0.0, 0.0, 0.0),
    ego_quat=(1.0, 0.0, 0.0, 0.0),
    width: int = CAMERA_WIDTH,
    height: int = CAMERA_HEIGHT,
    max_range: float = MAX_RANGE_M,
) -> tuple[float, float] | None:
    """Body-frame (x forward, y left) from a pixel, or None if the ray misses still water."""
    fx, fy = camera_fx_fy()
    cam_pos, cam_axes = camera_world_from_ego(ego_pos, ego_quat)
    d_cam = pixel_dir_camera(u, v, fx, fy, width, height)
    d_w = _dir_world(d_cam, cam_axes)
    oz = float(cam_pos[2])
    dz = float(d_w[2])
    if abs(dz) < 1e-12:
        return None
    t = (WATER_Z - oz) / dz
    if t <= 1e-4:
        return None
    hit = (
        float(cam_pos[0]) + t * d_w[0],
        float(cam_pos[1]) + t * d_w[1],
        WATER_Z,
    )
    look = (-cam_axes[2][0], -cam_axes[2][1], -cam_axes[2][2])
    ahead = (
        (hit[0] - cam_pos[0]) * look[0]
        + (hit[1] - cam_pos[1]) * look[1]
        + (hit[2] - cam_pos[2]) * look[2]
    )
    if ahead <= 0.0:
        return None
    dist = math.sqrt(
        (hit[0] - cam_pos[0]) ** 2
        + (hit[1] - cam_pos[1]) ** 2
        + (hit[2] - cam_pos[2]) ** 2
    )
    if dist > float(max_range):
        return None
    yaw = yaw_from_wxyz(ego_quat)
    return world_to_body_xy(hit, ego_pos, yaw)


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
    """Body-frame (x forward, y left) or None if the ray misses still water."""
    u, v = bbox_bottom_center(cx, cy, w, h)
    return range_from_pixel(
        u,
        v,
        ego_pos=ego_pos,
        ego_quat=ego_quat,
        width=width,
        height=height,
        max_range=max_range,
    )
