from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn

# 最大实体数
E_MAX = 16

# 实体的运动学维度（二维相对位置、二维相对速度）
KIN_DIM = 4

# 实体的图像特征维度
OWL_DIM = 512

# 实体的语义特征维度
SEMANTIC_DIM = 16

# 实体的原始特征维度（运动学 + 图像特征）
RAW_DIM = KIN_DIM + OWL_DIM

# 实体的最终特征维度（运动学 + 语义特征）
ENTITY_DIM = KIN_DIM + SEMANTIC_DIM

# 任务编码维度
TASK_DIM = 64

# 包装实体原始特征矩阵，确保其维度正确
def pack_raw_entities(raw_entities) -> tuple[np.ndarray, np.ndarray]:
    raw = np.asarray(raw_entities, dtype=np.float64)
    if raw.size == 0:
        raw = np.zeros((0, RAW_DIM), dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != RAW_DIM:
        raise ValueError(f"raw entities expected (n,{RAW_DIM}), got {raw.shape}")
    n = min(int(raw.shape[0]), E_MAX)
    padded = np.zeros((E_MAX, RAW_DIM), dtype=np.float64)
    mask = np.zeros(E_MAX, dtype=np.bool_)
    if n:
        padded[:n] = raw[:n]
        mask[:n] = True
    return padded, mask

# 将 512 维图像特征投影到 16 维语义特征
class SemanticProjector(nn.Module):
    def __init__(self, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(OWL_DIM, int(hidden)),
            nn.ReLU(),
            nn.Linear(int(hidden), SEMANTIC_DIM),
        )

    def forward(self, appearance: torch.Tensor) -> torch.Tensor:
        feature = self.net(appearance)
        return torch.nn.functional.normalize(feature, dim=-1, eps=1e-8)

# 对位置/速度进行规范化；语义列保持不变
class SemanticCanonization(nn.Module):
    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = float(eps)

    def forward(self, state: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        m = mask.unsqueeze(-1).to(state.dtype)
        p = state[..., :2] * m
        gram = p.transpose(-2, -1) @ p
        values, vectors = torch.linalg.eigh(gram)
        axis = vectors[..., -1]
        anchor = p.sum(dim=-2)
        alignment = (axis * anchor).sum(dim=-1)
        axis = torch.where((alignment < 0).unsqueeze(-1), -axis, axis)
        theta = torch.atan2(axis[..., 1], axis[..., 0])
        gap = values[..., -1] - values[..., -2]
        theta = torch.where(gap < self.eps, torch.zeros_like(theta), theta)
        c = torch.cos(theta).unsqueeze(-1)
        s = torch.sin(theta).unsqueeze(-1)
        out = state.clone()
        px, py = state[..., 0], state[..., 1]
        vx, vy = state[..., 2], state[..., 3]
        out[..., 0] = px * c + py * s
        out[..., 1] = -px * s + py * c
        out[..., 2] = vx * c + vy * s
        out[..., 3] = -vx * s + vy * c
        return out * m

# 由任务特征生成两组乘性增益，分别调制 V 和输出权重
class TaskHyperNetwork(nn.Module):
    def __init__(self, task_dim: int = TASK_DIM, value_dim: int = 16, hidden: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(task_dim), hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2 * int(value_dim)),
        )
        self.value_dim = int(value_dim)

    def forward(self, task: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gains = self.net(task)
        return gains[..., : self.value_dim], gains[..., self.value_dim :]

# 任务嵌入调制的旋转等变决策网络
class SemanticAttentionActor(nn.Module):
    def __init__(
        self,
        task_dim: int = TASK_DIM,
        width: int = 16,
        max_action: float = 0.5,
        task_center=None,
        task_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.projector = SemanticProjector()
        self.canonize = SemanticCanonization()
        self.w_q = nn.Linear(ENTITY_DIM, int(width), bias=False)
        self.w_k = nn.Linear(ENTITY_DIM, int(width), bias=False)
        self.w_v = nn.Linear(ENTITY_DIM, int(width), bias=False)
        self.w_o = nn.Linear(int(width), 1, bias=False)
        self.hypernet = TaskHyperNetwork(task_dim=task_dim, value_dim=width)
        self.w_select = nn.Linear(ENTITY_DIM, int(width), bias=False)
        self.selector_hypernet = nn.Sequential(
            nn.Linear(int(task_dim), 32),
            nn.ReLU(),
            nn.Linear(32, int(width)),
        )
        self.selector_null = nn.Sequential(
            nn.Linear(int(task_dim), 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )
        self.task_dim = int(task_dim)
        self.width = int(width)
        self.max_action = float(max_action)
        center = (
            torch.zeros(self.task_dim, dtype=torch.float32)
            if task_center is None
            else torch.as_tensor(task_center, dtype=torch.float32).reshape(self.task_dim)
        )
        self.register_buffer("task_center", center)
        self.register_buffer("task_scale", torch.tensor(float(task_scale), dtype=torch.float32))

    def condition_task(self, task: torch.Tensor) -> torch.Tensor:
        """Expose small Qwen embedding differences to the hypernetworks."""
        return (task - self.task_center) * self.task_scale

    def entity_matrix(self, raw: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        semantic = self.projector(raw[..., KIN_DIM:])
        entities = torch.cat((raw[..., :KIN_DIM], semantic), dim=-1)
        return entities * mask.unsqueeze(-1).to(entities.dtype)

    def _encode(self, raw, task, mask):
        entities = self.entity_matrix(raw, mask)
        return entities, self.canonize(entities, mask), self.condition_task(task)

    def _target_logits(self, canonical, conditioned_task, mask):
        key = self.w_select(canonical)
        query = self.selector_hypernet(conditioned_task)
        logits = (key * query.unsqueeze(-2)).sum(dim=-1) / math.sqrt(float(self.width))
        logits = logits.masked_fill(~mask, -1.0e9)
        return torch.cat((logits, self.selector_null(conditioned_task)), dim=-1)

    def forward(
        self,
        raw: torch.Tensor,
        task: torch.Tensor,
        mask: torch.Tensor,
        selector_override: torch.Tensor | None = None,
    ) -> torch.Tensor:
        no_batch = raw.dim() == 2
        if no_batch:
            raw = raw.unsqueeze(0)
        if task.dim() == 1:
            task = task.unsqueeze(0)
        if mask.dim() == 1:
            mask = mask.unsqueeze(0)
        if selector_override is not None and selector_override.dim() == 1:
            selector_override = selector_override.unsqueeze(0)
        mask = mask.bool()
        entities, canonical, conditioned_task = self._encode(raw, task, mask)
        if selector_override is None:
            logits = self._target_logits(canonical, conditioned_task, mask)
            selector = torch.nan_to_num(torch.softmax(logits, dim=-1), nan=0.0)[..., :E_MAX]
        else:
            selector = selector_override.to(raw.dtype) * mask.to(raw.dtype)
            selector = selector / selector.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
        action = self._action(entities, canonical, conditioned_task, mask, selector)
        return action.squeeze(0) if no_batch else action

    def _action(self, entities, canonical, conditioned_task, mask, selector):
        q = self.w_q(canonical)
        k = self.w_k(canonical)
        value_gain, output_gain = self.hypernet(conditioned_task)
        value = self.w_v(canonical) * value_gain.unsqueeze(-2)
        scores = q @ k.transpose(-2, -1) / math.sqrt(float(self.width))
        scores = scores.masked_fill(~mask.unsqueeze(-2), -1.0e9)
        scores = scores.masked_fill(~mask.unsqueeze(-1), -1.0e9)
        attention = torch.nan_to_num(torch.softmax(scores, dim=-1), nan=0.0)
        context = attention @ value
        output_weight = self.w_o.weight * output_gain.unsqueeze(-2)
        weights = (context * output_weight).sum(dim=-1) * mask.to(entities.dtype)
        weights = weights * selector * mask.sum(dim=-1, keepdim=True).to(entities.dtype)
        action = (weights.unsqueeze(-2) @ entities[..., :2]).squeeze(-2)
        action = torch.where(mask.any(dim=-1, keepdim=True), action, torch.zeros_like(action))
        norm = torch.linalg.norm(action, dim=-1, keepdim=True)
        return self.max_action * torch.tanh(norm) * action / (norm + 1.0e-8)

# 加载决策模型
def load_semantic_actor(path, map_location=None) -> SemanticAttentionActor:
    blob = torch.load(path, map_location=map_location, weights_only=True)
    if blob.get("architecture") != "semantic16_task_modulated_attention":
        raise ValueError("expected a semantic16_task_modulated_attention checkpoint")
    model = SemanticAttentionActor(
        task_dim=int(blob["task_dim"]),
        width=int(blob["width"]),
        max_action=float(blob["max_action"]),
        task_center=blob.get("task_center"),
        task_scale=float(blob.get("task_scale", 1.0)),
    )
    model.load_state_dict(blob["state_dict"], strict=True)
    return model

# 决策网络包装类
class SemanticTorchActor:
    """Runtime wrapper accepting task-independent raw tracker observations."""

    def __init__(self, path, task_embed, device: str = "cpu") -> None:
        self.device = torch.device(device)
        self.net = load_semantic_actor(path, map_location=self.device).to(self.device).eval()
        task = np.asarray(task_embed, dtype=np.float64).reshape(-1)
        if task.size != self.net.task_dim:
            raise ValueError(f"task embedding {task.size} != {self.net.task_dim}")
        self.max_action = self.net.max_action
        self._task = torch.as_tensor(task, dtype=torch.float32, device=self.device)
        self.last_raw_entities = np.zeros((0, RAW_DIM), dtype=np.float64)
        self.last_target_probabilities = np.zeros((0,), dtype=np.float64)
        self.last_null_probability = 0.0
        self.last_target_misses = None
        self._locked_track_id = None
        self._track_ids = None
        self._track_misses = None
        self._track_hits = None
        self._lock_candidate_id = None
        self._lock_candidate_hit = None
        self._lock_candidate_confirmations = 0

    def set_track_metadata(self, track_ids, track_misses, track_hits) -> None:
        """Attach tracker identity/freshness without adding them to model input."""
        self._track_ids = np.asarray(track_ids, dtype=np.int64).reshape(-1)
        self._track_misses = np.asarray(track_misses, dtype=np.int64).reshape(-1)
        self._track_hits = np.asarray(track_hits, dtype=np.int64).reshape(-1)

    def _clear_lock_candidate(self) -> None:
        self._lock_candidate_id = None
        self._lock_candidate_hit = None
        self._lock_candidate_confirmations = 0

    def _confirm_lock_candidate(self, track_id: int, hit: int) -> int:
        """Count distinct detector updates, never repeated 10 Hz prediction ticks."""
        track_id, hit = int(track_id), int(hit)
        if self._lock_candidate_id != track_id:
            self._lock_candidate_id = track_id
            self._lock_candidate_hit = hit
            self._lock_candidate_confirmations = 1
        elif self._lock_candidate_hit != hit:
            self._lock_candidate_hit = hit
            self._lock_candidate_confirmations += 1
        return int(self._lock_candidate_confirmations)

    def __call__(self, raw_entities) -> tuple[float, float]:
        padded, mask = pack_raw_entities(raw_entities)
        raw = torch.as_tensor(padded, dtype=torch.float32, device=self.device)
        valid = torch.as_tensor(mask, dtype=torch.bool, device=self.device)
        task = self._task
        with torch.no_grad():
            batch_mask = valid.unsqueeze(0)
            entities, canonical, conditioned_task = self.net._encode(
                raw.unsqueeze(0), task.unsqueeze(0) if task.dim() == 1 else task, batch_mask
            )
            target_probabilities = torch.softmax(
                self.net._target_logits(canonical, conditioned_task, batch_mask), dim=-1
            ).squeeze(0)
            probabilities = target_probabilities.detach().cpu().numpy().astype(np.float64)
            n_valid = int(mask.sum())
            null_probability = (
                float(probabilities[-1])
                if probabilities.size
                else 0.0
            )
            forced_index = None
            track_ids = self._track_ids
            track_misses = self._track_misses
            track_hits = self._track_hits
            metadata_valid = (
                track_ids is not None
                and track_misses is not None
                and track_hits is not None
                and len(track_ids) == n_valid
                and len(track_misses) == n_valid
                and len(track_hits) == n_valid
            )
            if self._locked_track_id is not None and metadata_valid:
                matches = np.flatnonzero(track_ids == int(self._locked_track_id))
                if matches.size:
                    forced_index = int(matches[0])
                else:
                    # The tracker has retired that identity.  Do not silently
                    # transfer the lock to a visually similar distractor.
                    self._locked_track_id = None
                    self._clear_lock_candidate()
            if forced_index is not None and metadata_valid and probabilities.size:
                best = int(np.argmax(probabilities[:n_valid]))
                current_probability = float(probabilities[forced_index])
                challenger_probability = float(probabilities[best])
                challenger_is_fresh = int(track_misses[best]) == 0
                if (
                    best != forced_index
                    and challenger_is_fresh
                    and challenger_probability >= 0.85
                    and current_probability <= 0.15
                ):
                    confirmations = self._confirm_lock_candidate(
                        int(track_ids[best]), int(track_hits[best])
                    )
                    if confirmations >= 3:
                        self._locked_track_id = int(track_ids[best])
                        forced_index = best
                        self._clear_lock_candidate()
                else:
                    self._clear_lock_candidate()
            if self._locked_track_id is None and n_valid and probabilities.size:
                best = int(np.argmax(probabilities[:n_valid]))
                fresh = not metadata_valid or int(track_misses[best]) == 0
                if (
                    fresh
                    and float(probabilities[best]) >= 0.70
                    and float(probabilities[best]) > null_probability
                ):
                    if metadata_valid:
                        self._locked_track_id = int(track_ids[best])
                    forced_index = best
                    self._clear_lock_candidate()
                else:
                    self._clear_lock_candidate()
            selector = torch.nan_to_num(target_probabilities, nan=0.0)[:E_MAX]
            if forced_index is not None:
                selector = torch.zeros(E_MAX, dtype=raw.dtype, device=self.device)
                selector[forced_index] = 1.0
                selector = selector * valid.to(raw.dtype)
                selector = selector / selector.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
            action = self.net._action(
                entities, canonical, conditioned_task, batch_mask, selector.unsqueeze(0)
            ).squeeze(0)
        self.last_raw_entities = raw[valid].detach().cpu().numpy().astype(np.float64)
        self.last_target_misses = (
            int(track_misses[forced_index])
            if forced_index is not None and metadata_valid
            else None
        )
        if probabilities.size:
            if forced_index is not None:
                effective = np.zeros_like(probabilities[:-1])
                effective[forced_index] = 1.0
                self.last_target_probabilities = effective
                self.last_null_probability = 0.0
            else:
                self.last_target_probabilities = probabilities[:-1]
                self.last_null_probability = float(probabilities[-1])
        else:
            self.last_target_probabilities = probabilities
            self.last_null_probability = 0.0
        return float(action[0]), float(action[1])
