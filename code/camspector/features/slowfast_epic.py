"""SlowFast-R50 EPIC-100 feature extractor for parity test.

Extracts 2304-d features (2048 slow + 256 fast pathway, global-avg-pooled) from video
using the SlowFast EPIC-100 checkpoint. Uses 32-frame clips with stride 16 at 30fps.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

try:
    from camspector.video_io import iter_frames
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parents[2]))
    from camspector.video_io import iter_frames

log = logging.getLogger("camspector.features.slowfast_epic")


def map_pyslowfast_to_pytorchvideo(state_dict):
    """Map PySlowFast checkpoint keys to pytorchvideo model keys.
    
    PySlowFast format: s1.pathway0_stem.conv.weight
    pytorchvideo format: blocks.0.multipathway_blocks.0.conv.weight
    """
    new_state = {}
    
    for key, value in state_dict.items():
        # Skip head keys - they'll be random init since we extract features before head
        if key.startswith('head.'):
            continue
            
        new_key = key
        
        # Map stem: s1.pathway0_stem -> blocks.0.multipathway_blocks.0
        if 's1.pathway0_stem' in key:
            new_key = key.replace('s1.pathway0_stem', 'blocks.0.multipathway_blocks.0')
        elif 's1.pathway1_stem' in key:
            new_key = key.replace('s1.pathway1_stem', 'blocks.0.multipathway_blocks.1')
        
        # Map fusion: s1.pathway0_pathway1_fuse -> blocks.0.multipathway_fusion
        elif 's1.pathway0_pathway1_fuse' in key:
            new_key = key.replace('s1.pathway0_pathway1_fuse', 'blocks.0.multipathway_fusion')
            new_key = new_key.replace('.lateral', '.conv_fast_to_slow')
        
        # Map res stages: s2.pathway0_res0 -> blocks.1.multipathway_blocks.0.res_blocks
        # s2 -> blocks.1, s3 -> blocks.2, s4 -> blocks.3, s5 -> blocks.4
        elif key.startswith('s2.'):
            new_key = key.replace('s2.', 'blocks.1.')
            new_key = new_key.replace('pathway0_res', 'multipathway_blocks.0.res_blocks.')
            new_key = new_key.replace('pathway1_res', 'multipathway_blocks.1.res_blocks.')
        elif key.startswith('s3.'):
            new_key = key.replace('s3.', 'blocks.2.')
            new_key = new_key.replace('pathway0_res', 'multipathway_blocks.0.res_blocks.')
            new_key = new_key.replace('pathway1_res', 'multipathway_blocks.1.res_blocks.')
            new_key = new_key.replace('pathway0_pathway1_fuse', 'multipathway_fusion')
            new_key = new_key.replace('.lateral', '.conv_fast_to_slow')
        elif key.startswith('s4.'):
            new_key = key.replace('s4.', 'blocks.3.')
            new_key = new_key.replace('pathway0_res', 'multipathway_blocks.0.res_blocks.')
            new_key = new_key.replace('pathway1_res', 'multipathway_blocks.1.res_blocks.')
            new_key = new_key.replace('pathway0_pathway1_fuse', 'multipathway_fusion')
            new_key = new_key.replace('.lateral', '.conv_fast_to_slow')
        elif key.startswith('s5.'):
            new_key = key.replace('s5.', 'blocks.4.')
            new_key = new_key.replace('pathway0_res', 'multipathway_blocks.0.res_blocks.')
            new_key = new_key.replace('pathway1_res', 'multipathway_blocks.1.res_blocks.')
        
        # Map bn -> norm
        new_key = new_key.replace('.bn.', '.norm.')
        new_key = new_key.replace('_bn.', '_norm.')
        
        # Map branch layers
        new_key = new_key.replace('.branch1.', '.branch1_conv.')
        new_key = new_key.replace('.branch1_bn.', '.branch1_norm.')
        new_key = new_key.replace('.branch2.a.', '.branch2.conv_a.')
        new_key = new_key.replace('.branch2.a_bn.', '.branch2.norm_a.')
        new_key = new_key.replace('.branch2.b.', '.branch2.conv_b.')
        new_key = new_key.replace('.branch2.b_bn.', '.branch2.norm_b.')
        new_key = new_key.replace('.branch2.c.', '.branch2.conv_c.')
        new_key = new_key.replace('.branch2.c_bn.', '.branch2.norm_c.')
        
        new_state[new_key] = value
    
    return new_state


def load_slowfast_model(checkpoint_path: str | Path, device: str = "cuda"):
    """Load SlowFast EPIC-100 checkpoint with proper key mapping."""
    from pytorchvideo.models import slowfast
    
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Create SlowFast R50 model
    model = slowfast.create_slowfast(
        model_depth=50,
        model_num_class=400,
        input_channels=(3, 3),
        dropout_rate=0.5,
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    # The checkpoint contains 'model_state' key
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        state_dict = checkpoint["model_state"]
    else:
        state_dict = checkpoint
    
    # Map keys from PySlowFast to pytorchvideo format
    log.info("Mapping checkpoint keys from PySlowFast to pytorchvideo format...")
    mapped_state = map_pyslowfast_to_pytorchvideo(state_dict)
    
    # Load state dict
    missing, unexpected = model.load_state_dict(mapped_state, strict=False)
    log.info(f"Loaded checkpoint from {checkpoint_path}")
    if missing:
        log.info(f"Missing keys: {len(missing)} (head keys expected to be missing)")
    if unexpected:
        log.warning(f"Unexpected keys: {len(unexpected)}")
    
    model = model.to(device)
    model = model.half()  # fp16
    model.eval()
    
    return model


def extract_features_from_model(model, slow_input, fast_input):
    """Extract 2304-d features before the classification head.
    
    The model structure is: blocks[0-4] -> blocks[5] (PoolConcatPathway) -> blocks[6] (ResNetBasicHead)
    We want the pooled features before the head's projection layer.
    """
    # Hook to capture features before projection
    features_out = {}
    
    def hook_fn(module, input, output):
        # Capture input to the projection layer (after dropout, before proj)
        if isinstance(input, tuple):
            features_out['features'] = input[0]
        else:
            features_out['features'] = input
    
    # Register hook on the projection layer of the head (blocks[6].proj)
    hook = model.blocks[6].proj.register_forward_hook(hook_fn)
    
    try:
        # Forward pass
        with torch.no_grad():
            _ = model([slow_input, fast_input])
        
        # Extract features from hook
        feat = features_out.get('features')
        if feat is not None:
            # Should already be [B, 2304]
            if feat.ndim > 2:
                feat = feat.flatten(1)
            return feat
        else:
            raise RuntimeError("Failed to extract features")
    finally:
        hook.remove()


def extract(
    video_path: str | Path,
    *,
    variant: dict[str, Any],
    device: str = "cuda",
    checkpoint_path: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract SlowFast EPIC-100 features from video.
    
    Args:
        video_path: Path to input video
        variant: Dictionary with extraction parameters:
            - resize_mode: "short256_center224" | "resize224" | "short224"
            - downscale_first: bool - downsample to 456x256 before resize (mimic JPEGs)
            - start_offset: int - first clip start offset in frames (0 or 8)
            - pad_last: bool - pad incomplete last clip by repeating last frame
        device: "cuda" or "cpu"
        checkpoint_path: Path to SlowFast.pyth checkpoint
        
    Returns:
        features: [N, 2304] float32 feature vectors
        timestamps: [N] clip center times in seconds
    """
    if checkpoint_path is None:
        checkpoint_path = Path(__file__).parents[3] / "weights" / "slowfast_epic100" / "SlowFast.pyth"
    
    # Load model
    model = load_slowfast_model(checkpoint_path, device=device)
    
    # Extract variant parameters
    resize_mode = variant.get("resize_mode", "short256_center224")
    downscale_first = variant.get("downscale_first", False)
    start_offset = variant.get("start_offset", 0)
    pad_last = variant.get("pad_last", True)
    
    # SlowFast parameters
    TARGET_FPS = 30.0
    CLIP_FRAMES = 32
    STRIDE_FRAMES = 16
    ALPHA = 4  # slow pathway uses every 4th frame
    
    # Normalization (from SlowFast config)
    mean = torch.tensor([0.45, 0.45, 0.45], device=device).view(1, 3, 1, 1, 1)
    std = torch.tensor([0.225, 0.225, 0.225], device=device).view(1, 3, 1, 1, 1)
    
    # Decode video at 30fps
    log.info(f"Decoding {video_path} at {TARGET_FPS} fps")
    frames_list = []
    timestamps_list = []
    
    import cv2
    
    for t, frame in iter_frames(video_path, target_fps=TARGET_FPS):
        timestamps_list.append(t)
        
        # Apply downscaling first if requested
        if downscale_first:
            # Resize to 456x256 to mimic JPEG extraction
            h, w = frame.shape[:2]
            scale = 456 / min(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            frame = cv2.resize(frame, (new_w, new_h))
        
        # Apply resize mode
        h, w = frame.shape[:2]
        if resize_mode == "short256_center224":
            # Short side 256, then center crop 224x224
            scale = 256 / min(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            frame = cv2.resize(frame, (new_w, new_h))
            # Center crop to 224x224
            h, w = frame.shape[:2]
            y = (h - 224) // 2
            x = (w - 224) // 2
            frame = frame[y:y+224, x:x+224]
        elif resize_mode == "resize224":
            # Direct resize to 224x224
            frame = cv2.resize(frame, (224, 224))
        elif resize_mode == "short224":
            # Short side 224, no crop
            scale = 224 / min(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            frame = cv2.resize(frame, (new_w, new_h))
        else:
            raise ValueError(f"Unknown resize_mode: {resize_mode}")
        
        frames_list.append(frame)
    
    if len(frames_list) < CLIP_FRAMES:
        log.warning(f"Video too short: {len(frames_list)} frames < {CLIP_FRAMES}")
        return np.zeros((0, 2304), dtype=np.float32), np.array([], dtype=np.float32)
    
    frames_array = np.stack(frames_list)  # [T, H, W, 3]
    timestamps_array = np.array(timestamps_list, dtype=np.float32)
    
    log.info(f"Decoded {len(frames_array)} frames, shape: {frames_array.shape}")
    
    # Extract clips with sliding window
    features = []
    clip_times = []
    
    start_frame = start_offset
    with torch.no_grad():
        while start_frame + CLIP_FRAMES <= len(frames_array):
            end_frame = start_frame + CLIP_FRAMES
            clip = frames_array[start_frame:end_frame]  # [32, H, W, 3]
            
            # Center timestamp
            clip_center_time = (timestamps_array[start_frame] + timestamps_array[end_frame - 1]) / 2
            
            # Convert to torch tensor: [32, H, W, 3] -> [3, 32, H, W]
            clip_tensor = torch.from_numpy(clip).permute(3, 0, 1, 2).float() / 255.0
            clip_tensor = clip_tensor.unsqueeze(0).to(device)  # [1, 3, 32, H, W]
            
            # Normalize
            clip_tensor = (clip_tensor - mean) / std
            
            # Create slow and fast pathways
            # Slow: every 4th frame (alpha=4) -> 8 frames
            # Fast: all frames -> 32 frames
            slow_input = clip_tensor[:, :, ::ALPHA, :, :]  # [1, 3, 8, H, W]
            fast_input = clip_tensor  # [1, 3, 32, H, W]
            
            # Convert to fp16 for efficiency
            slow_input = slow_input.half()
            fast_input = fast_input.half()
            
            # Extract features
            feat = extract_features_from_model(model, slow_input, fast_input)  # [1, 2304]
            
            features.append(feat.cpu().float().numpy()[0])
            clip_times.append(clip_center_time)
            
            start_frame += STRIDE_FRAMES
        
        # Handle last incomplete clip if pad_last is True
        if pad_last and start_frame < len(frames_array):
            remaining = len(frames_array) - start_frame
            if remaining > 0:
                clip = frames_array[start_frame:]
                # Pad by repeating last frame
                last_frame = clip[-1]
                pad_count = CLIP_FRAMES - len(clip)
                clip = np.concatenate([clip, np.tile(last_frame[None], (pad_count, 1, 1, 1))], axis=0)
                
                clip_center_time = (timestamps_array[start_frame] + timestamps_array[-1]) / 2
                
                clip_tensor = torch.from_numpy(clip).permute(3, 0, 1, 2).float() / 255.0
                clip_tensor = clip_tensor.unsqueeze(0).to(device)
                clip_tensor = (clip_tensor - mean) / std
                
                slow_input = clip_tensor[:, :, ::ALPHA, :, :].half()
                fast_input = clip_tensor.half()
                
                feat = extract_features_from_model(model, slow_input, fast_input)
                features.append(feat.cpu().float().numpy()[0])
                clip_times.append(clip_center_time)
    
    features_array = np.array(features, dtype=np.float32)
    clip_times_array = np.array(clip_times, dtype=np.float32)
    
    log.info(f"Extracted {len(features_array)} feature vectors")
    
    return features_array, clip_times_array


if __name__ == "__main__":
    # Test extraction
    import sys
    logging.basicConfig(level=logging.INFO)
    
    if len(sys.argv) < 2:
        print("Usage: python slowfast_epic.py <video_path>")
        sys.exit(1)
    
    video_path = sys.argv[1]
    variant = {
        "resize_mode": "short256_center224",
        "downscale_first": False,
        "start_offset": 0,
        "pad_last": True,
    }
    
    features, times = extract(video_path, variant=variant)
    print(f"Extracted {len(features)} features, shape: {features.shape}")
    print(f"First 3 feature norms: {np.linalg.norm(features[:3], axis=1)}")
    print(f"First 3 clip times: {times[:3]}")
