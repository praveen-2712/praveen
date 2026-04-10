import torch
import torch_directml
import time

import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from utils.gpu_config import get_nvidia_device

def check_gpu():
    print("="*60)
    print("PyTorch DirectML Hardware Accelerator Test")
    print("="*60)
    print("Enumerating Available Graphics Processors:")
    for i in range(torch_directml.device_count()):
        print(f" - GPU {i}: {torch_directml.device_name(i)}")
        
    device = get_nvidia_device()
    print(f"\nActive PyTorch Device: {device}")
    print(f"Hardware Name: {torch_directml.device_name(torch_directml.default_device())}")
    
    print("\nRunning matrix multiplication benchmark (CPU vs GPU)...")
    
    # Warmup
    _ = torch.randn(10, 10) @ torch.randn(10, 10)
    _ = torch.randn(10, 10).to(device) @ torch.randn(10, 10).to(device)
    
    # CPU
    cpu_tensor = torch.randn(5000, 5000)
    start = time.time()
    cpu_result = cpu_tensor @ cpu_tensor
    cpu_time = time.time() - start
    print(f"CPU Computation Time: {cpu_time:.4f} seconds")
    
    # GPU
    gpu_tensor = torch.randn(5000, 5000).to(device)
    start = time.time()
    gpu_result = gpu_tensor @ gpu_tensor
    # Force synchronization to wait for GPU to finish
    _ = gpu_result.cpu()
    gpu_time = time.time() - start
    print(f"GPU Computation Time: {gpu_time:.4f} seconds")
    
    if gpu_time < cpu_time:
        print(f"\n✅ SUCCESS! Your pipeline is using the RTX 5050 via DirectML.")
        print(f"Your RTX 5050 was {cpu_time/gpu_time:.1f}x faster than your CPU.")
    else:
        print("\nHmm, something is wrong.")

if __name__ == "__main__":
    check_gpu()
