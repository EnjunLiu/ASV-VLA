"""Camera + platform state -> visual-language action -> /espapp/input.

This process ends at the existing controller input boundary.  It neither embeds
EspApp nor publishes actuator commands.
"""

from __future__ import annotations

import argparse
import math
import os
import threading
import time
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")

import numpy as np

from .loop import PerceptionLoop, detections_from_boxes
from .policy import SemanticTorchActor
from .tasks import render_task
from .types import SensorState

DARK_FRAME_MEAN = 40.0
TARGET_FRAME_MEAN = 88.0
MAX_EXPOSURE_GAIN = 8.0


def _model_root() -> Path:
    return Path(os.environ.get("ASV_VLA_MODEL_DIR", Path.cwd() / "models"))


def _sensor_qos(reliable=False):
    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

    return QoSProfile(
        depth=10 if reliable else 5,
        reliability=(ReliabilityPolicy.RELIABLE if reliable else ReliabilityPolicy.BEST_EFFORT),
        history=HistoryPolicy.KEEP_LAST,
    )


def _recover_dark_frame(rgb):
    src = np.asarray(rgb)
    mean = float(np.mean(src))
    if mean >= DARK_FRAME_MEAN or mean <= 1.0e-6:
        return src, 1.0
    gain = min(MAX_EXPOSURE_GAIN, TARGET_FRAME_MEAN / mean)
    return np.clip(src.astype(np.float32) * gain, 0.0, 255.0).astype(np.uint8), gain


def _rgb8(msg):
    raw = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    image = raw.reshape(int(msg.height), int(msg.width), 3)
    if str(msg.encoding) == "bgr8":
        return image[:, :, ::-1].copy()
    if str(msg.encoding) != "rgb8":
        raise ValueError(f"unsupported image encoding: {msg.encoding}")
    return image


def _decode_ue_camera(msg):
    import cv2

    if not msg.valid:
        raise ValueError("invalid UE camera frame")
    encoded = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot decode UE camera payload ({msg.encoding})")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _yaw_quaternion(yaw: float):
    return (math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw))


def _target_missing(actor, count: int, stale_after: int = 6) -> bool:
    if count <= 0:
        return True
    misses = getattr(actor, "last_target_misses", None)
    if misses is not None and int(misses) >= stale_after:
        return True
    probabilities = np.asarray(getattr(actor, "last_target_probabilities", []))
    best = float(np.max(probabilities)) if probabilities.size else 0.0
    null = float(getattr(actor, "last_null_probability", 0.0))
    return null >= 0.70 and null > best


def main() -> None:
    models = _model_root()
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("isaac", "ue"), default="isaac")
    parser.add_argument("--color", choices=("red", "blue"), default="red")
    parser.add_argument("--standoff", type=float, default=4.0)
    parser.add_argument("--weights", default=str(models / "actor_ppo_semantic16_v14_isaaclab_deploysafe.pt"))
    parser.add_argument("--qwen-embed", default=str(models / "qwen_task_embed.npz"))
    parser.add_argument("--hf-home", default=os.environ.get("HF_HOME", str(models / "hf")))
    parser.add_argument("--owl-id", default="google/owlvit-base-patch32")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--detection-period", type=float, default=0.5)
    parser.add_argument("--sim-time", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    for required in (Path(args.weights), Path(args.qwen_embed), Path(args.hf_home)):
        if not required.exists():
            raise SystemExit(f"runtime model asset missing: {required}")

    blob = np.load(args.qwen_embed)
    key = f"{args.color}_{int(round(args.standoff))}"
    if key not in blob.files:
        raise SystemExit(f"task embedding missing: {key}")
    actor = SemanticTorchActor(args.weights, np.asarray(blob[key], dtype=np.float64), args.device)
    actor.standoff = float(args.standoff)
    loop = PerceptionLoop(actor)
    task = render_task(args.color, args.standoff)

    from .perception.owl import FrozenOwl

    owl = FrozenOwl(model_id=args.owl_id, device=args.device, cache_dir=args.hf_home)

    import rclpy
    from interfaces.msg import ASVState, CameraFrame, Input
    from nav_msgs.msg import Odometry
    from rclpy.executors import ExternalShutdownException
    from rclpy.parameter import Parameter
    from sensor_msgs.msg import Image, Imu

    rclpy.init()
    node = rclpy.create_node(
        "asv_vla",
        parameter_overrides=[Parameter("use_sim_time", Parameter.Type.BOOL, args.sim_time)],
    )
    publisher = node.create_publisher(Input, "/espapp/input", 10)
    lock = threading.Lock()
    stop = threading.Event()
    state = {
        "rgb": None,
        "image_t": None,
        "sensors": None,
        "imu_r": 0.0,
        "detections": None,
        "detection_t": -1.0,
        "consumed_t": -1.0,
        "scheduled_t": -1.0,
        "count": 0,
    }

    def on_image(msg: Image):
        try:
            stamp = float(msg.header.stamp.sec) + 1.0e-9 * float(msg.header.stamp.nanosec)
            with lock:
                state["rgb"], state["image_t"] = _rgb8(msg), stamp
        except Exception as exc:
            node.get_logger().warning(f"image rejected: {exc}")

    def on_imu(msg: Imu):
        with lock:
            state["imu_r"] = float(msg.angular_velocity.z)

    def on_odom(msg: Odometry):
        q = msg.pose.pose.orientation
        quat = (float(q.w), float(q.x), float(q.y), float(q.z))
        yaw = math.atan2(2.0 * (quat[0] * quat[3] + quat[1] * quat[2]), 1.0 - 2.0 * (quat[2] ** 2 + quat[3] ** 2))
        vx, vy = float(msg.twist.twist.linear.x), float(msg.twist.twist.linear.y)
        with lock:
            image_t = state["image_t"]
            r = float(msg.twist.twist.angular.z) or float(state["imu_r"])
            state["sensors"] = SensorState(
                t=float(image_t if image_t is not None else node.get_clock().now().nanoseconds * 1.0e-9),
                yaw_rate=r,
                surge_velocity=vx * math.cos(yaw) + vy * math.sin(yaw),
                sway_velocity=-vx * math.sin(yaw) + vy * math.cos(yaw),
                ego_pos=(float(msg.pose.pose.position.x), float(msg.pose.pose.position.y), float(msg.pose.pose.position.z)),
                ego_quat=quat,
                image_stamp=image_t,
            )

    def on_ue_camera(msg: CameraFrame):
        try:
            with lock:
                state["rgb"] = _decode_ue_camera(msg)
                state["image_t"] = float(msg.stamp_us) * 1.0e-6
        except Exception as exc:
            node.get_logger().warning(f"UE camera rejected: {exc}")

    def on_ue_state(msg: ASVState):
        if not msg.valid:
            return
        with lock:
            image_t = state["image_t"]
            state["sensors"] = SensorState(
                t=float(msg.stamp_us) * 1.0e-6,
                yaw_rate=float(msg.yaw_rate),
                surge_velocity=float(msg.surge_velocity),
                sway_velocity=0.0,
                ego_pos=(float(msg.position_x), float(msg.position_y), float(msg.position_z)),
                ego_quat=_yaw_quaternion(float(msg.yaw)),
                image_stamp=image_t,
            )

    if args.backend == "isaac":
        node.create_subscription(Image, "/asv/camera/image_raw", on_image, _sensor_qos(True))
        node.create_subscription(Imu, "/asv/imu", on_imu, _sensor_qos())
        node.create_subscription(Odometry, "/asv/state_estimate", on_odom, _sensor_qos())
    else:
        node.create_subscription(CameraFrame, "/ue/camera_frame", on_ue_camera, _sensor_qos())
        node.create_subscription(ASVState, "/ue/asv_state", on_ue_state, _sensor_qos(True))

    def detect_loop():
        while not stop.is_set():
            with lock:
                stamp, rgb, sensors = state["image_t"], state["rgb"], state["sensors"]
                scheduled = float(state["scheduled_t"])
            if stamp is None or rgb is None or sensors is None or float(stamp) - scheduled < args.detection_period:
                stop.wait(0.01)
                continue
            frame, gain = _recover_dark_frame(np.array(rgb, copy=True))
            with lock:
                state["scheduled_t"] = float(stamp)
            boxes = owl.detect_entities(frame)
            snapshot = SensorState(**{**sensors.__dict__, "t": float(stamp), "image_stamp": float(stamp)})
            detections = detections_from_boxes(boxes, snapshot)
            with lock:
                if float(stamp) > float(state["detection_t"]):
                    state["detections"], state["detection_t"] = detections, float(stamp)
            node.get_logger().info(
                f"OWL image_t={stamp:.2f} boxes={len(boxes)} detections={len(detections)} exposure_gain={gain:.2f}"
            )

    detector = threading.Thread(target=detect_loop, name="owl_detector", daemon=True)
    detector.start()

    def publish_invalid(stamp_us=0):
        message = Input()
        message.stamp_us = int(stamp_us)
        message.valid = False
        publisher.publish(message)

    def beat():
        with lock:
            sensors, image_t = state["sensors"], state["image_t"]
            detections = None
            if float(state["detection_t"]) > float(state["consumed_t"]):
                detections = list(state["detections"] or [])
                state["consumed_t"] = float(state["detection_t"])
        if sensors is None or image_t is None:
            publish_invalid()
            return
        now = node.get_clock().now().nanoseconds * 1.0e-9
        if now > 0.0 and now - float(image_t) > 0.25:
            publish_invalid(round(float(image_t) * 1.0e6))
            return
        sensors.t = float(image_t)
        sensors.image_stamp = float(image_t)
        tick = loop.step(sensors, detections)
        valid = not _target_missing(actor, len(tick.entities))
        action = tick.action if valid else (0.0, 0.0)
        message = Input()
        message.stamp_us = int(round(float(image_t) * 1.0e6))
        message.desired_x = float(action[0])
        message.desired_y = float(action[1])
        message.surge_velocity = float(sensors.surge_velocity)
        message.sway_velocity = float(sensors.sway_velocity)
        message.yaw_rate = float(sensors.yaw_rate)
        message.valid = bool(valid and all(math.isfinite(v) for v in (*action, sensors.surge_velocity, sensors.sway_velocity, sensors.yaw_rate)))
        publisher.publish(message)
        state["count"] = int(state["count"]) + 1
        if int(state["count"]) % 10 == 1:
            node.get_logger().info(
                f"input valid={message.valid} desired=({message.desired_x:.3f},{message.desired_y:.3f}) "
                f"velocity=({message.surge_velocity:.3f},{message.sway_velocity:.3f},{message.yaw_rate:.3f})"
            )

    node.create_timer(0.1, beat)
    print(f"ASV_VLA_READY backend={args.backend} task={key}", flush=True)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        stop.set()
        detector.join(timeout=2.0)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
