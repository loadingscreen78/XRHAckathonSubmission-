"""Encoder feasibility benchmark (Phase 0.5f).

Measures forward-pass throughput and peak VRAM of candidate encoders on this machine.
Weights are RANDOMLY INITIALISED from the official architecture configs: compute cost is
identical to the pretrained checkpoints, and nothing has to be downloaded. Accuracy is NOT
measured here (that needs real weights; Phase 3).

Also measures video DECODE cost (PyAV) on a synthetic 1080p60 clip, since EPIC is 1080p.

Usage: python code/scripts/bench_encoders.py [--cpu] [--out experiments/bench_encoders.json]
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import torch

# sampling plan per encoder -> forward units per minute of video
PLANS = {
    "clip_vitb16": dict(unit="frame", per_min=120, desc="frames @2 fps, 224px"),
    "siglip_b16_224": dict(unit="frame", per_min=120, desc="frames @2 fps, 224px"),
    "videomae_v2_b": dict(unit="clip", per_min=60, desc="16-frame clip, 1 s stride, 224px"),
    "videomae_v2_b_0.5s": dict(unit="clip", per_min=120, desc="16-frame clip, 0.5 s stride"),
    "slowfast_r50_8x8": dict(unit="clip", per_min=112.5,
                             desc="32-frame clip (8 slow / 32 fast), 16-frame stride @30 fps "
                                  "(ActionFormer EPIC setting), 224px"),
}


def build(name: str):
    if name == "clip_vitb16":
        from transformers import CLIPVisionConfig, CLIPVisionModel
        m = CLIPVisionModel(CLIPVisionConfig(hidden_size=768, intermediate_size=3072,
                                             num_hidden_layers=12, num_attention_heads=12,
                                             image_size=224, patch_size=16))
        return m, lambda b: (torch.randn(b, 3, 224, 224),)
    if name == "siglip_b16_224":
        from transformers import SiglipVisionConfig, SiglipVisionModel
        m = SiglipVisionModel(SiglipVisionConfig(hidden_size=768, intermediate_size=3072,
                                                 num_hidden_layers=12, num_attention_heads=12,
                                                 image_size=224, patch_size=16))
        return m, lambda b: (torch.randn(b, 3, 224, 224),)
    if name.startswith("videomae_v2_b"):
        # VideoMAE V2-B (distilled) == ViT-B, 16 frames, tubelet 2, patch 16, 224px.
        from transformers import VideoMAEConfig, VideoMAEModel
        m = VideoMAEModel(VideoMAEConfig(num_frames=16, tubelet_size=2, image_size=224,
                                         patch_size=16, hidden_size=768, num_hidden_layers=12,
                                         num_attention_heads=12, intermediate_size=3072,
                                         use_mean_pooling=True))
        return m, lambda b: (torch.randn(b, 16, 3, 224, 224),)
    if name == "slowfast_r50_8x8":
        from pytorchvideo.models.slowfast import create_slowfast
        m = create_slowfast(model_num_class=None, model_depth=50,
                            slowfast_fusion_conv_stride=(4, 1, 1),
                            input_channels=(3, 3), head=None)
        return m, lambda b: ([torch.randn(b, 3, 8, 224, 224), torch.randn(b, 3, 32, 224, 224)],)
    raise KeyError(name)


def to(x, dev, dtype):
    if isinstance(x, (list, tuple)):
        return type(x)(to(t, dev, dtype) for t in x)
    return x.to(dev, dtype)


def bench_one(name, dev, dtype, batches, iters=5):
    m, mk = build(name)
    m = m.to(dev, dtype).eval()
    best = None
    for b in batches:
        try:
            x = to(mk(b), dev, dtype)
            if dev == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            # pytorchvideo's SlowFast overwrites its input list in place -> fresh list per call
            def call():
                return m(*[list(v) if isinstance(v, list) else v for v in x])
            with torch.inference_mode():
                for _ in range(2):  # warm-up
                    call()
                if dev == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(iters):
                    call()
                if dev == "cuda":
                    torch.cuda.synchronize()
                dt = (time.perf_counter() - t0) / iters
            peak = torch.cuda.max_memory_allocated() / 2**30 if dev == "cuda" else None
            r = dict(batch=b, units_per_s=b / dt, peak_vram_gb=peak)
            print(f"  {name} {dev} {dtype} bs={b}: {b/dt:.1f} units/s, peak {peak}", flush=True)
            if best is None or r["units_per_s"] > best["units_per_s"]:
                best = r
            del x
        except torch.OutOfMemoryError:
            print(f"  {name} bs={b}: OOM", flush=True)
            torch.cuda.empty_cache()
            break
    del m
    if dev == "cuda":
        torch.cuda.empty_cache()
    return best


def bench_decode(seconds=20):
    """Decode cost for 1080p60 H.264 (EPIC-like), full decode and 2-fps sampled."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.make_synthetic_videos import cfr, write_video
    from camspector.video_io import iter_frames
    p = Path("data/synthetic/bench_1080p60.mp4")
    if not p.exists():
        write_video(p, cfr(seconds, 60), 1920, 1080, fps_hint=60)
    out = {}
    for label, fps, size in (("full_decode_rgb_1080p", None, None),
                             ("2fps_resize_224", 2.0, (224, 224)),
                             ("all_frames_resize_224", None, (224, 224))):
        t0 = time.perf_counter()
        n = sum(1 for _ in iter_frames(p, target_fps=fps, size=size))
        dt = time.perf_counter() - t0
        out[label] = dict(frames=n, sec_per_video_min=dt * 60 / seconds)
        print(f"  decode {label}: {n} frames, {dt*60/seconds:.1f} s per video-minute", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu", action="store_true", help="also benchmark CPU fp32")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--out", default="experiments/bench_encoders.json")
    a = ap.parse_args()
    torch.manual_seed(0)
    res = {"env": dict(torch=torch.__version__, cuda=torch.version.cuda,
                       gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                       cpu=platform.processor(), threads=torch.get_num_threads()),
           "plans": PLANS, "gpu_fp16": {}, "cpu_fp32": {}, "decode": {}}
    names = a.only or [n for n in PLANS if n != "videomae_v2_b_0.5s"]
    for n in names:
        if torch.cuda.is_available():
            r = bench_one(n, "cuda", torch.float16, [1, 2, 4, 8, 16, 32, 64])
            res["gpu_fp16"][n] = r
        if a.cpu:
            res["cpu_fp32"][n] = bench_one(n, "cpu", torch.float32, [1, 4], iters=2)
    res["decode"] = bench_decode()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2))
    # summary table
    print("\n| encoder | plan | GPU fp16 best batch | units/s | peak VRAM GB | GPU s/video-min |"
          " CPU units/s | CPU s/video-min |")
    print("|---|---|---:|---:|---:|---:|---:|---:|")
    for n in names:
        g, c = res["gpu_fp16"].get(n), res["cpu_fp32"].get(n)
        pm = PLANS[n]["per_min"]
        gs = f"{g['batch']} | {g['units_per_s']:.1f} | {g['peak_vram_gb']:.2f} | {pm/g['units_per_s']:.1f}" if g else "- | - | - | -"
        cs = f"{c['units_per_s']:.2f} | {pm/c['units_per_s']:.0f}" if c else "- | -"
        print(f"| {n} | {PLANS[n]['desc']} | {gs} | {cs} |")


if __name__ == "__main__":
    main()
