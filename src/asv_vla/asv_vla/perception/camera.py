"""Ship-camera pinhole constants (howto 1.1.4 / 1.3.1). No Isaac import."""

from __future__ import annotations

import math

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_HFOV_DEG = 90.0
CAMERA_H_APERTURE = 16.0
CAMERA_V_APERTURE = 9.0
CAMERA_FOCAL = CAMERA_H_APERTURE / 2.0
CAMERA_MOUNT = (0.42, 0.0, 0.2)
CAMERA_PITCH_DOWN_DEG = 5.0
WATER_Z = 0.0
MAX_RANGE_M = 40.0


def camera_fx_fy() -> tuple[float, float]:
    fx = CAMERA_WIDTH * CAMERA_FOCAL / CAMERA_H_APERTURE
    fy = CAMERA_HEIGHT * CAMERA_FOCAL / CAMERA_V_APERTURE
    return fx, fy


def camera_usd_axes(pitch_down_deg: float = CAMERA_PITCH_DOWN_DEG):
    """USD camera basis in parent: +X right, +Y up, -Z look = body +X pitched down."""
    pitch = math.radians(float(pitch_down_deg))
    look = (math.cos(pitch), 0.0, -math.sin(pitch))
    z_axis = (-look[0], -look[1], -look[2])
    world_up = (0.0, 0.0, 1.0)
    x_axis = (
        world_up[1] * z_axis[2] - world_up[2] * z_axis[1],
        world_up[2] * z_axis[0] - world_up[0] * z_axis[2],
        world_up[0] * z_axis[1] - world_up[1] * z_axis[0],
    )
    x_len = math.sqrt(x_axis[0] ** 2 + x_axis[1] ** 2 + x_axis[2] ** 2)
    if x_len < 1e-8:
        x_axis = (0.0, -1.0, 0.0)
    else:
        x_axis = (x_axis[0] / x_len, x_axis[1] / x_len, x_axis[2] / x_len)
    y_axis = (
        z_axis[1] * x_axis[2] - z_axis[2] * x_axis[1],
        z_axis[2] * x_axis[0] - z_axis[0] * x_axis[2],
        z_axis[0] * x_axis[1] - z_axis[1] * x_axis[0],
    )
    y_len = math.sqrt(y_axis[0] ** 2 + y_axis[1] ** 2 + y_axis[2] ** 2)
    y_axis = (y_axis[0] / y_len, y_axis[1] / y_len, y_axis[2] / y_len)
    z_len = math.sqrt(z_axis[0] ** 2 + z_axis[1] ** 2 + z_axis[2] ** 2)
    z_axis = (z_axis[0] / z_len, z_axis[1] / z_len, z_axis[2] / z_len)
    return x_axis, y_axis, z_axis


def rot_from_wxyz(q) -> tuple[tuple[float, float, float], ...]:
    w, x, y, z = (float(v) for v in q)
    return (
        (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
        (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
        (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
    )


def _mul(r, p) -> tuple[float, float, float]:
    return (
        r[0][0] * p[0] + r[0][1] * p[1] + r[0][2] * p[2],
        r[1][0] * p[0] + r[1][1] * p[1] + r[1][2] * p[2],
        r[2][0] * p[0] + r[2][1] * p[1] + r[2][2] * p[2],
    )


def world_from_local(p_local, pos, quat_wxyz) -> tuple[float, float, float]:
    r = rot_from_wxyz(quat_wxyz)
    d = _mul(r, p_local)
    return (float(pos[0]) + d[0], float(pos[1]) + d[1], float(pos[2]) + d[2])


def camera_world_from_ego(pos, quat_wxyz, mount=CAMERA_MOUNT, pitch_down_deg: float = CAMERA_PITCH_DOWN_DEG):
    axes_body = camera_usd_axes(pitch_down_deg)
    cam_pos = world_from_local(mount, pos, quat_wxyz)
    r = rot_from_wxyz(quat_wxyz)
    axes_w = tuple(_mul(r, ax) for ax in axes_body)
    return cam_pos, axes_w


def yaw_from_wxyz(q) -> float:
    w, x, y, z = (float(v) for v in q)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def point_to_uv(
    p_world,
    cam_pos,
    cam_axes,
    fx: float,
    fy: float,
    width: int,
    height: int,
) -> tuple[float, float, float] | None:
    """USD camera: +X right, +Y up, -Z look. Returns (u, v, depth) or None if behind."""
    x_axis, y_axis, z_axis = cam_axes
    rel = (
        float(p_world[0]) - float(cam_pos[0]),
        float(p_world[1]) - float(cam_pos[1]),
        float(p_world[2]) - float(cam_pos[2]),
    )
    x_c = x_axis[0] * rel[0] + x_axis[1] * rel[1] + x_axis[2] * rel[2]
    y_c = y_axis[0] * rel[0] + y_axis[1] * rel[1] + y_axis[2] * rel[2]
    z_c = z_axis[0] * rel[0] + z_axis[1] * rel[1] + z_axis[2] * rel[2]
    depth = -z_c
    if depth < 1e-4:
        return None
    cx = 0.5 * float(width)
    cy = 0.5 * float(height)
    u = fx * x_c / depth + cx
    v = fy * (-y_c) / depth + cy
    return u, v, depth
