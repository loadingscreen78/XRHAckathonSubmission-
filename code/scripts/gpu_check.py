"""Phase 0.5a GPU check: CUDA available, fp16 numerically sane, conv3d fp16 works.

Usage: python code/scripts/gpu_check.py [--out experiments/gpu_check.json]
Exit code 0 = all checks passed, 1 = a check failed (details in the JSON / stderr).
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="experiments/gpu_check.json")
    a = ap.parse_args()
    r = {"python": sys.version.split()[0], "platform": platform.platform(),
         "torch": torch.__version__, "torch_cuda": torch.version.cuda,
         "cudnn": torch.backends.cudnn.version(), "cuda_available": torch.cuda.is_available(),
         "checks": {}}
    ok = True
    if not torch.cuda.is_available():
        r["error"] = "torch.cuda.is_available() is False (CPU-only torch build or driver issue)"
        ok = False
    else:
        p = torch.cuda.get_device_properties(0)
        r.update(gpu=p.name, vram_gb=round(p.total_memory / 2**30, 2),
                 capability=f"{p.major}.{p.minor}", bf16=torch.cuda.is_bf16_supported())
        torch.manual_seed(0)
        a32 = torch.randn(2048, 2048, device="cuda")
        b32 = torch.randn(2048, 2048, device="cuda")
        ref = a32 @ b32
        h = (a32.half() @ b32.half()).float()
        rel = ((h - ref).norm() / ref.norm()).item()
        r["checks"]["fp16_matmul_rel_err"] = rel
        r["checks"]["fp16_matmul_ok"] = bool(rel < 1e-2 and torch.isfinite(h).all().item())
        x = torch.randn(2, 3, 16, 112, 112, device="cuda")
        conv = torch.nn.Conv3d(3, 32, 3, padding=1).cuda()
        y32 = conv(x)
        y16 = conv.half()(x.half()).float()
        crel = ((y16 - y32).norm() / y32.norm()).item()
        r["checks"]["fp16_conv3d_rel_err"] = crel
        r["checks"]["fp16_conv3d_ok"] = bool(crel < 1e-2)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(20):
            a32.half() @ b32.half()
        torch.cuda.synchronize()
        r["checks"]["fp16_matmul_tflops"] = round(20 * 2 * 2048**3 / (time.perf_counter() - t0) / 1e12, 2)
        ok = r["checks"]["fp16_matmul_ok"] and r["checks"]["fp16_conv3d_ok"]
    r["all_ok"] = ok
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(r, indent=2))
    print(json.dumps(r, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
