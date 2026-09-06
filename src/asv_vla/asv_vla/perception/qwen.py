"""Frozen Qwen3 embedding (howto 1.2 / 3.1). First 64 dims, L2-normalized. Not OWL."""

from __future__ import annotations

import numpy as np

QWEN_ID = "Qwen/Qwen3-Embedding-0.6B"
INSTRUCT = "Represent the navigation task for an unmanned surface vehicle."
EMBED_DIM = 64  # probe 3.1: 32-d R^2=0.75, 64-d R^2=0.91
HF_HOME = r"E:\hil-platform\weights\hf"


def format_query(task_text: str) -> str:
    return f"Instruct: {INSTRUCT}\nQuery: {task_text}"


def qwen_weights_present(model_id: str = QWEN_ID, cache_dir: str = HF_HOME) -> bool:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        return False
    try:
        snapshot_download(model_id, cache_dir=cache_dir, local_files_only=True)
        return True
    except Exception:
        return False


def truncate_unit(vec, dim: int = EMBED_DIM) -> np.ndarray:
    z = np.asarray(vec, dtype=np.float64).reshape(-1)
    if z.size < int(dim):
        pad = np.zeros(int(dim), dtype=np.float64)
        pad[: z.size] = z
        z = pad
    else:
        z = z[: int(dim)].copy()
    n = float(np.linalg.norm(z))
    if n < 1e-12:
        z[0] = 1.0
        n = 1.0
    return z / n


class FrozenQwen:
    def __init__(self, model_id: str = QWEN_ID, device: str = "cpu", cache_dir: str = HF_HOME) -> None:
        import os

        os.environ.setdefault("HF_HOME", cache_dir)
        from sentence_transformers import SentenceTransformer

        self.dim = EMBED_DIM
        self.model = SentenceTransformer(model_id, cache_folder=cache_dir, device=device)
        self.model.eval()

    def embed(self, task_text: str) -> np.ndarray:
        q = format_query(task_text)
        vec = self.model.encode(q, normalize_embeddings=False, show_progress_bar=False)
        return truncate_unit(vec, self.dim)

    def embed_many(self, texts) -> np.ndarray:
        qs = [format_query(t) for t in texts]
        vecs = self.model.encode(qs, normalize_embeddings=False, show_progress_bar=False)
        return np.stack([truncate_unit(v, self.dim) for v in vecs], axis=0)
