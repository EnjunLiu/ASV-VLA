"""Frozen OWL-ViT detector (howto 1.3.1). Weights stay frozen.

Letterbox to 768, recall queries boat/ship/obstacle, rho from the task sentence.
512-d appearance is OWL text-space class embed, not Qwen.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

OWL_ID = "google/owlvit-base-patch32"
OWL_SIZE = 768
RECALL_QUERIES = ("boat", "ship", "obstacle")
# The weakest real T1 boat is ~0.107 in a clean frame and intermittently
# crosses 0.10 under wave/pose changes.  A 0.05 probe still returns exactly
# the four boats, so keep recall permissive and let the learned semantic
# selector reject non-target entities downstream.
RECALL_THRESH = 0.05
NMS_IOU = 0.5
MAX_DETS = 16
APPEAR_DIM = 512
# NanoOWL boxes on the known Isaac ASV model include water reflections at the
# lower edge, making bottom-ray range jump badly. Projected box height is much
# more stable: range[m] * height[px] is approximately 192 on the T1 camera.
ISAAC_ASV_HEIGHT_RANGE_SCALE = 192.0


def default_hf_home() -> str:
    import os

    return os.environ.get("OWL_CACHE") or os.environ.get("HF_HOME") or r"E:\hil-platform\weights\hf"


HF_HOME = default_hf_home()
CLIP_TEMPLATE = "a photo of a {}"
_TASK_COLORS = ("red", "blue", "white", "yellow", "gray", "grey")


def format_recall_query(word: str) -> str:
    """OWL-ViT is trained on CLIP-style captions; bare 'boat' scores ~0.05 on Isaac RGB."""
    w = str(word).strip()
    if w.lower().startswith("a photo of"):
        return w
    article = "an" if w[:1].lower() in "aeiou" else "a"
    return f"a photo of {article} {w}"


def format_task_query(task_text: str) -> str:
    """Command sentences do not fire OWL-ViT. Keep the color noun as the ρ query."""
    import re

    t = str(task_text).strip()
    if t.lower().startswith("a photo of"):
        return t
    low = t.lower()
    color = next((c for c in _TASK_COLORS if re.search(rf"\b{c}\b", low)), None)
    if color is not None:
        return CLIP_TEMPLATE.format(f"{color} boat")
    return t


@dataclass
class OwlBox:
    cx: float
    cy: float
    w: float
    h: float
    rho: float
    score: float
    appearance: np.ndarray | None = None


def letterbox_rgb(rgb: np.ndarray, size: int = OWL_SIZE) -> tuple[np.ndarray, float, int, int]:
    """Scale so max side == size, pad to square. Returns canvas, scale, pad_x, pad_y."""
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
    """Map a box in the padded square back to source pixels."""
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


def range_bucket(r: float | None) -> str | None:
    """Howto 3.2 distance buckets. None if missing or outside 0–10 m."""
    if r is None:
        return None
    x = float(r)
    if x < 0.0 or x > 10.0:
        return None
    if x < 3.0:
        return "0-3"
    if x < 6.0:
        return "3-6"
    return "6-10"


def owl_to_detections(boxes, *, ego_pos=(0.0, 0.0, 0.0), ego_quat=(1.0, 0.0, 0.0, 0.0), stamp: float = 0.0):
    """Pixel OWL boxes → body-frame Detection using bearing + box-height range.

    The waterline ray supplies body-frame bearing. Range comes from the known
    simulated ASV's projected height, avoiding the reflection-sensitive OWL
    bottom edge. This remains camera perception and never reads bbox_gt or
    target pose.
    """
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
                rho=float(b.rho),
                appearance=None if b.appearance is None else b.appearance,
                stamp=float(stamp),
                source_stamp=float(stamp),
            )
        )
    return out


def owl_weights_present(model_id: str = OWL_ID, cache_dir: str = HF_HOME) -> bool:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        return False
    try:
        snapshot_download(model_id, cache_dir=cache_dir, local_files_only=True)
        return True
    except Exception:
        return False


class FrozenOwl:
    """google/owlvit-base-patch32, eval only. Task text is the OWL query, not Qwen."""

    def __init__(
        self,
        model_id: str = OWL_ID,
        device: str = "cpu",
        recall_thresh: float = RECALL_THRESH,
        cache_dir: str = HF_HOME,
    ) -> None:
        import os

        os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
        os.environ.setdefault("USE_TF", "0")
        os.environ.setdefault("HF_HOME", cache_dir)
        import torch
        from transformers import OwlViTForObjectDetection, OwlViTProcessor

        self.device = torch.device(device)
        self.recall_thresh = float(recall_thresh)
        # Runtime inference must remain available when the workstation or
        # Jetson has no Internet.  Installation populates HF_HOME once;
        # collection/deployment only read that pinned local snapshot.
        self.processor = OwlViTProcessor.from_pretrained(
            model_id, cache_dir=cache_dir, local_files_only=True
        )
        self.model = OwlViTForObjectDetection.from_pretrained(
            model_id, cache_dir=cache_dir, local_files_only=True
        )
        self.model.to(self.device)
        # Jetson shares system RAM with CUDA. FP16 cuts the frozen detector's
        # resident and activation memory while leaving CPU inference FP32.
        self.model_dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        if self.model_dtype == torch.float16:
            self.model.half()
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self._task = ""

    def detect(self, rgb: np.ndarray, task_text: str | None) -> list[OwlBox]:
        import torch

        src = np.asarray(rgb)
        src_h, src_w = int(src.shape[0]), int(src.shape[1])
        canvas, scale, pad_x, pad_y = letterbox_rgb(src, OWL_SIZE)
        queries = [format_recall_query(q) for q in RECALL_QUERIES]
        has_task_query = bool(str(task_text or "").strip())
        if has_task_query:
            queries.append(format_task_query(str(task_text)))
        texts = [queries]
        inputs = self.processor(text=texts, images=canvas, return_tensors="pt", do_resize=False)
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
        n_recall = len(RECALL_QUERIES)
        scores = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
        recall = scores[:, :n_recall].max(axis=1)
        rho = scores[:, n_recall] if has_task_query else np.zeros(scores.shape[0], dtype=np.float64)
        raw_boxes: list[tuple[float, float, float, float]] = []
        raw_score: list[float] = []
        raw_rho: list[float] = []
        raw_app: list[np.ndarray | None] = []
        for i, r in enumerate(recall):
            if float(r) < self.recall_thresh:
                continue
            cx, cy, bw, bh = pred[i]
            raw_boxes.append((float(cx) * OWL_SIZE, float(cy) * OWL_SIZE, float(bw) * OWL_SIZE, float(bh) * OWL_SIZE))
            raw_score.append(float(r))
            raw_rho.append(float(np.clip(rho[i], 0.0, 1.0)))
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
                    rho=raw_rho[i],
                    score=raw_score[i],
                    appearance=raw_app[i],
                )
            )
        return boxes

    def detect_entities(self, rgb: np.ndarray) -> list[OwlBox]:
        """Task-independent open-vocabulary recall for semantic entity encoding."""
        return self.detect(rgb, None)
