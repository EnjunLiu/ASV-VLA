from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

import numpy as np

OWL_ID = "google/owlvit-base-patch32"

# OWL-ViT 的输入尺寸
OWL_SIZE = 768

# 检测实体类别（OWL-ViT 使用 CLIP 风格文本）
RECALL_QUERIES = ["a photo of a boat", "a photo of a ship", "a photo of an obstacle"]

# OWL-ViT 检测阈值
RECALL_THRESH = 0.05

# 重叠阈值
NMS_IOU = 0.5

# 最多框数
MAX_DETS = 16

# 从模型内部提取出的图像特征的维度
APPEAR_DIM = 512

# 经验数值，用于测距。距离(米) ≈ 192 / 框高(像素)。
ISAAC_ASV_HEIGHT_RANGE_SCALE = 192.0

HF_HOME = os.environ.get("OWL_CACHE") or os.environ.get("HF_HOME") or str(
    Path(__file__).resolve().parents[4] / "models" / "hf"
)

# OWL-ViT 检测框
@dataclass
class OwlBox:
    cx: float
    cy: float
    w: float
    h: float
    appearance: np.ndarray | None = None # OWL-ViT 内部提取的 512 维图像特征

# 将原始捕获缩放到 OWL-ViT 的输入尺寸
def letterbox_rgb(rgb: np.ndarray, size: int = OWL_SIZE) -> tuple[np.ndarray, float, int, int]:

    from PIL import Image

    src = np.asarray(rgb)
    if src.ndim != 3 or src.shape[2] != 3:
        raise ValueError(f"rgb expected HWC uint8, got {src.shape}")
    h, w = int(src.shape[0]), int(src.shape[1])
    scale = float(size) / float(max(h, w))
    nh = max(1, int(round(h * scale)))
    nw = max(1, int(round(w * scale)))
    img = Image.fromarray(src.astype(np.uint8)).resize((nw, nh), Image.Resampling.BILINEAR)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    pad_y = (size - nh) // 2
    pad_x = (size - nw) // 2
    canvas[pad_y : pad_y + nh, pad_x : pad_x + nw] = np.asarray(img)
    return canvas, scale, pad_x, pad_y

# 将 OWL-ViT 的输出框映射回原始捕获的尺寸
def unletterbox_cxcywh(
    cx: float,
    cy: float,
    w: float,
    h: float,
    *,
    scale: float,
    pad_x: int,
    pad_y: int,
    src_w: int,
    src_h: int,
) -> tuple[float, float, float, float]:
    s = float(scale)
    cx_s = (float(cx) - float(pad_x)) / s
    cy_s = (float(cy) - float(pad_y)) / s
    w_s = float(w) / s
    h_s = float(h) / s
    cx_s = min(max(cx_s, 0.0), float(src_w))
    cy_s = min(max(cy_s, 0.0), float(src_h))
    w_s = min(max(w_s, 1.0), float(src_w))
    h_s = min(max(h_s, 1.0), float(src_h))
    return cx_s, cy_s, w_s, h_s

# 计算两个框的交并比
def iou_xywh(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax0, ay0, ax1, ay1 = ax - 0.5 * aw, ay - 0.5 * ah, ax + 0.5 * aw, ay + 0.5 * ah
    bx0, by0, bx1, by1 = bx - 0.5 * bw, by - 0.5 * bh, bx + 0.5 * bw, by + 0.5 * bh
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    if union <= 1e-12:
        return 0.0
    return inter / union

# 非极大值抑制，用于去除重叠框
def nms_xywh(boxes: list[tuple[float, float, float, float]], scores: list[float], iou: float = NMS_IOU, cap: int = MAX_DETS) -> list[int]:
    order = sorted(range(len(boxes)), key=lambda i: float(scores[i]), reverse=True)
    keep: list[int] = []
    for i in order:
        if any(iou_xywh(boxes[i], boxes[j]) >= float(iou) for j in keep):
            continue
        keep.append(i)
        if len(keep) >= int(cap):
            break
    return keep

# 将 OWL-ViT 的输出框转换为 Detection 类型（单帧图像检测结果）
def owl_to_detections(boxes, *, ego_pos=(0.0, 0.0, 0.0), ego_quat=(1.0, 0.0, 0.0, 0.0), stamp: float = 0.0):
    from .types import Detection
    from .waterline import range_from_bbox

    out = []
    for b in boxes:
        xy = range_from_bbox(b.cx, b.cy, b.w, b.h, ego_pos=ego_pos, ego_quat=ego_quat)
        if xy is None:
            continue
        ray_range = float(np.hypot(xy[0], xy[1]))
        if float(b.h) > 1.0 and ray_range > 1.0e-6:
            size_range = float(ISAAC_ASV_HEIGHT_RANGE_SCALE / float(b.h))
            size_range = float(np.clip(size_range, 0.5, 30.0))
            xy = (float(xy[0]) * size_range / ray_range, float(xy[1]) * size_range / ray_range)
        out.append(
            Detection(
                x=float(xy[0]),
                y=float(xy[1]),
                appearance=b.appearance,
                stamp=float(stamp),
            )
        )
    return out

# 封装的 OWL-ViT 检测器
class FrozenOwl:

    def __init__(
        self,
        model_id: str = OWL_ID,
        device: str = "cpu",
        recall_thresh: float = RECALL_THRESH,
        cache_dir: str = HF_HOME,
    ) -> None:
        os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
        os.environ.setdefault("USE_TF", "0")
        os.environ.setdefault("HF_HOME", cache_dir)
        import torch
        from transformers import OwlViTForObjectDetection, OwlViTProcessor

        self.device = torch.device(device)
        self.recall_thresh = float(recall_thresh)
        self.processor = OwlViTProcessor.from_pretrained(
            model_id, cache_dir=cache_dir, local_files_only=True
        )
        self.model = OwlViTForObjectDetection.from_pretrained(
            model_id, cache_dir=cache_dir, local_files_only=True
        )
        self.model.to(self.device)
        self.model_dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        if self.model_dtype == torch.float16:
            self.model.half()
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    def detect(self, rgb: np.ndarray) -> list[OwlBox]:
        import torch

        src = np.asarray(rgb)
        src_h, src_w = int(src.shape[0]), int(src.shape[1])
        canvas, scale, pad_x, pad_y = letterbox_rgb(src, OWL_SIZE)
        inputs = self.processor(text=[RECALL_QUERIES], images=canvas, return_tensors="pt", do_resize=False)
        inputs = {
            k: (
                v.to(self.device, dtype=self.model_dtype)
                if hasattr(v, "is_floating_point") and v.is_floating_point()
                else v.to(self.device)
                if hasattr(v, "to")
                else v
            )
            for k, v in inputs.items()
        }
        with torch.no_grad():
            out = self.model(**inputs)
        logits = out.logits[0].float().cpu().numpy()
        pred = out.pred_boxes[0].float().cpu().numpy()
        appear = None
        if hasattr(out, "class_embeds") and out.class_embeds is not None:
            appear = out.class_embeds[0].float().cpu().numpy()
        scores = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
        recall = scores.max(axis=1)
        raw_boxes: list[tuple[float, float, float, float]] = []
        raw_score: list[float] = []
        raw_app: list[np.ndarray | None] = []
        for i, r in enumerate(recall):
            if float(r) < self.recall_thresh:
                continue
            cx, cy, bw, bh = pred[i]
            raw_boxes.append((float(cx) * OWL_SIZE, float(cy) * OWL_SIZE, float(bw) * OWL_SIZE, float(bh) * OWL_SIZE))
            raw_score.append(float(r))
            if appear is not None:
                vec = np.asarray(appear[i], dtype=np.float64).reshape(-1)
                if vec.size >= APPEAR_DIM:
                    vec = vec[:APPEAR_DIM]
                nrm = float(np.linalg.norm(vec))
                raw_app.append(vec / nrm if nrm > 1e-12 else vec)
            else:
                raw_app.append(None)
        keep = nms_xywh(raw_boxes, raw_score)
        boxes: list[OwlBox] = []
        for i in keep:
            cx, cy, bw, bh = unletterbox_cxcywh(
                *raw_boxes[i],
                scale=scale,
                pad_x=pad_x,
                pad_y=pad_y,
                src_w=src_w,
                src_h=src_h,
            )
            boxes.append(
                OwlBox(
                    cx=cx,
                    cy=cy,
                    w=bw,
                    h=bh,
                    appearance=raw_app[i],
                )
            )
        return boxes
