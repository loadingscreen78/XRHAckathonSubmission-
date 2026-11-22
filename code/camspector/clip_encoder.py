"""CLIP ViT-B/16 image/text encoder (Hugging Face transformers).

Weights are loaded from `weights/clip-vit-base-patch16/` (created once by
`python code/scripts/download_weights.py`) so run.sh works offline in the organisers'
isolated re-run. If that folder is missing it falls back to the Hub id (needs internet).
CLIP weights: MIT licence (OpenAI).
"""
from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import numpy as np

log = logging.getLogger("camspector.clip")

HUB_ID = "openai/clip-vit-base-patch16"
LOCAL_DIR = Path(__file__).resolve().parents[2] / "weights" / "clip-vit-base-patch16"


class ClipEncoder:
    def __init__(self, source: str | Path | None = None, device: str | None = None):
        import torch
        from transformers import CLIPModel, CLIPProcessor

        src = str(source or (LOCAL_DIR if LOCAL_DIR.exists() else HUB_ID))
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = CLIPModel.from_pretrained(src).to(self.device, self.dtype).eval()
        self.proc = CLIPProcessor.from_pretrained(src)
        log.info("CLIP loaded from %s on %s", src, self.device)

    def _norm(self, x) -> np.ndarray:
        x = x.float()
        x = x / x.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return x.cpu().numpy().astype(np.float32)

    def images(self, frames: list[np.ndarray], batch: int = 64) -> np.ndarray:
        out = []
        with self.torch.inference_mode():
            for i in range(0, len(frames), batch):
                px = self.proc(images=frames[i:i + batch], return_tensors="pt")["pixel_values"]
                pooled = self.model.vision_model(pixel_values=px.to(self.device, self.dtype))[1]
                f = self.model.visual_projection(pooled)
                out.append(self._norm(f))
        return np.concatenate(out) if out else np.zeros((0, self.model.config.projection_dim), np.float32)

    def texts(self, prompts: list[str], batch: int = 256) -> np.ndarray:
        out = []
        with self.torch.inference_mode():
            for i in range(0, len(prompts), batch):
                tok = self.proc(text=prompts[i:i + batch], return_tensors="pt", padding=True, truncation=True)
                pooled = self.model.text_model(input_ids=tok["input_ids"].to(self.device),
                                                attention_mask=tok["attention_mask"].to(self.device))[1]
                f = self.model.text_projection(pooled)
                out.append(self._norm(f))
        return np.concatenate(out)

    def prompt_sets(self, sets: list[list[str]]) -> np.ndarray:
        """One embedding per class: mean of its prompt templates, re-normalised."""
        flat = [p for s in sets for p in s]
        emb = self.texts(flat)
        out, k = [], 0
        for s in sets:
            v = emb[k:k + len(s)].mean(axis=0)
            out.append(v / max(np.linalg.norm(v), 1e-8))
            k += len(s)
        return np.stack(out).astype(np.float32)


@lru_cache(maxsize=1)
def get_encoder() -> ClipEncoder:
    return ClipEncoder()
