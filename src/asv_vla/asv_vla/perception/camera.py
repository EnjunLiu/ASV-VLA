"""Ship-camera pinhole constants (howto 1.1.4 / 1.3.1). No Isaac import."""

from __future__ import annotations

import math

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FOCAL_PX = 640.0
CAMERA_MOUNT = (0.42, 0.0, 0.2)
CAMERA_PITCH_DOWN_DEG = 5.0
MAX_RANGE_M = 40.0


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


def camera_world_from_ego(pos, quat_wxyz, mount=CAMERA_MOUNT, pitch_down_deg: float = CAMERA_PITCH_DOWN_DEG):
    axes_body = camera_usd_axes(pitch_down_deg)
    r = rot_from_wxyz(quat_wxyz)
    offset = _mul(r, mount)
    cam_pos = tuple(float(pos[i]) + offset[i] for i in range(3))
    axes_w = tuple(_mul(r, ax) for ax in axes_body)
    return cam_pos, axes_w


def yaw_from_wxyz(q) -> float:
    w, x, y, z = (float(v) for v in q)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
