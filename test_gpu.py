import torch
import json

print("=" * 60)
print("GPU CHECK")
print("=" * 60)

result = {
    "pytorch_version": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
    "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
}

if torch.cuda.is_available():
    result["gpu_properties"] = {
        "total_memory_gb": torch.cuda.get_device_properties(0).total_memory / (1024**3),
        "compute_capability": f"{torch.cuda.get_device_properties(0).major}.{torch.cuda.get_device_properties(0).minor}",
    }
    
    # Test fp16 math
    print("\nTesting FP16 math on GPU...")
    a = torch.randn(1000, 1000, dtype=torch.float16, device='cuda')
    b = torch.randn(1000, 1000, dtype=torch.float16, device='cuda')
    c = torch.matmul(a, b)
    result["fp16_test"] = "PASS"
    print("✓ FP16 math test passed")
    
    # Quick speed test
    print("\nRunning speed test...")
    import time
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        c = torch.matmul(a, b)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    result["speed_test_100_matmul_ms"] = elapsed * 1000
    print(f"✓ 100 matrix multiplies (1000x1000 fp16): {elapsed*1000:.1f} ms")
    
    result["all_ok"] = True
else:
    result["all_ok"] = False
    print("ERROR: CUDA not available!")

print("\n" + "=" * 60)
print("RESULT:")
print(json.dumps(result, indent=2))
print("=" * 60)
