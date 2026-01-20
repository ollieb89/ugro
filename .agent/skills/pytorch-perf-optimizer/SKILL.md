---
name: pytorch-perf-optimizer
description: Optimal PyTorch settings for GPU acceleration, throughput maximization, and memory management. Use when creating or modifying PyTorch scripts (training/inference) to ensure correct use of TF32, cuDNN benchmarking, optimized DataLoaders, and torch.compile. Critical for RTX 30/40/50 series GPUs and high-throughput workloads.
---

# PyTorch Performance Optimizer

Optimize PyTorch training and inference for maximum GPU utilization.

## Quick Start (Automated)

Initialize optimizations at script start using the bundled helper:

```python
from scripts.setup_perf import setup_pytorch_performance, get_optimized_dataloader_config, optimize_model
from torch.utils.data import DataLoader

# 1. Global Setup (TF32, cuDNN Benchmark)
setup_pytorch_performance()

# 2. Optimized Model (Channels Last + torch.compile)
model = MyModel().to("cuda")
model = optimize_model(model)

# 3. Fast Data Loading
dl_args = get_optimized_dataloader_config()
loader = DataLoader(dataset, batch_size=64, **dl_args)
```

## Manual Optimization Patterns

### Backend Configuration
Apply these immediately after imports to configure the CUDA backend:

```python
import torch

# Enable cuDNN autotuner for optimal kernel selection
torch.backends.cudnn.benchmark = True

# Use TF32 for matrix multiplications (Ampere+ GPUs)
# Significantly faster with minimal precision loss
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

### Execution Efficiency
- **torch.compile**: Use `model = torch.compile(model)` for kernel fusion (~10-20% speedup).
- **Channels Last**: Use `model.to(memory_format=torch.channels_last)` for computer vision models to better utilize Tensor Cores.
- **Non-Blocking Transfers**: Use `data.to(device, non_blocking=True)` to overlap CPU-GPU data movement.

### Data Loading
Configure `DataLoader` to prevent CPU bottlenecks:

- `num_workers`: Set to `os.cpu_count() // 2`.
- `pin_memory=True`: Mandatory for fast host-to-device transfers.
- `persistent_workers=True`: Prevents recreating workers between epochs.
- `prefetch_factor=2`: Adjust based on preprocessing complexity.

### Memory Optimization
Maintain high GPU memory utilization without OOM:

- **AMP**: Use `torch.cuda.amp.autocast()` for automatic mixed precision.
- **Grad Scaling**: Use `torch.cuda.amp.GradScaler()` for stable training with FP16.
- **Empty Cache**: Avoid calling `torch.cuda.empty_cache()` inside loops as it causes performance-killing syncs.

## References & Tools
- **Advanced Optimization**: See [api_reference.md](references/api_reference.md) for multi-GPU and distributed patterns.
- **VRAM Debugging**: Use `scripts/setup_perf.py`'s built-in profilers for tracking memory fragmentation.