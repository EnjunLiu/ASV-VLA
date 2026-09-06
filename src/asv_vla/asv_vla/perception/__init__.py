"""Waterline ranging, 10 Hz Kalman, zero actor. OWL/Qwen load only when constructed."""

from .actor import ZeroActor
from .camera import CAMERA_HEIGHT, CAMERA_WIDTH, camera_fx_fy
from .owl import letterbox_rgb, nms_xywh, owl_to_detections, range_bucket, unletterbox_cxcywh
from .qwen import format_query, truncate_unit
from .tracker import KalmanTracker
from .types import Detection
from .waterline import range_from_bbox, range_from_pixel

__all__ = [
    "CAMERA_HEIGHT",
    "CAMERA_WIDTH",
    "Detection",
    "KalmanTracker",
    "ZeroActor",
    "camera_fx_fy",
    "format_query",
    "letterbox_rgb",
    "nms_xywh",
    "owl_to_detections",
    "range_bucket",
    "range_from_bbox",
    "range_from_pixel",
    "truncate_unit",
    "unletterbox_cxcywh",
]
