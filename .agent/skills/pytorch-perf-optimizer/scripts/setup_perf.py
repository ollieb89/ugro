import torch
import os
from typing import Dict, Any

def setup_pytorch_performance(use_tf32: bool = True, cudnn_benchmark: bool = True):
    """
    Apply global PyTorch performance optimizations.
    """
    # 1. Enable cuDNN autotuner
    torch.backends.cudnn.benchmark = cudnn_benchmark
    
    # 2. TensorFloat-32 (TF32) for Ampere+ GPUs
    if use_tf32 and torch.cuda.is_available():
        # High precision matmul using TF32
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
    print(f"PyTorch performance optimizations applied: TF32={use_tf32}, cuDNN Benchmark={cudnn_benchmark}")

def get_optimized_dataloader_config(
    num_workers: int = None,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    prefetch_factor: int = 2
) -> Dict[str, Any]:
    """
    Get recommended DataLoader arguments for high throughput.
    """
    if num_workers is None:
        # Default to half of CPU count to avoid overloading
        num_workers = min(os.cpu_count() or 4, 8)
        
    return {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
        "prefetch_factor": prefetch_factor
    }

def optimize_model(model: torch.nn.Module, compile: bool = True) -> torch.nn.Module:
    """
    Apply model-level optimizations like torch.compile and memory format.
    """
    # 1. Channels Last (NHWC) is often faster for CV models on Tensor Cores
    model = model.to(memory_format=torch.channels_last)
    
    # 2. torch.compile (PyTorch 2.0+)
    if compile and hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print("Model optimized with torch.compile (mode='reduce-overhead')")
        except Exception as e:
            print(f"Warning: torch.compile failed: {e}")
            
    return model
