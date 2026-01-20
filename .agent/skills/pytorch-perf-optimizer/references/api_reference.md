# Advanced PyTorch Performance Patterns

Detailed guide for squeezing the last bits of performance out of CUDA-accelerated PyTorch.

## Table of Contents
1. [Memory Formats (Channels Last)](#1-memory-formats-channels-last)
2. [Mixed Precision (FP16/BF16)](#2-mixed-precision-f16bf16)
3. [Scaled Dot Product Attention (SDPA)](#3-scaled-dot-product-attention-sdpa)
4. [Memory Management & Defragmentation](#4-memory-management--defragmentation)
5. [Distributed Training Settings](#5-distributed-training-settings)

---

## 1. Memory Formats (Channels Last)
For Computer Vision models (CNNs), using `channels_last` (NHWC) layout can improve performance on Tensor Cores by up to 20%.

```python
model = model.to(memory_format=torch.channels_last)
# Ensure inputs and targets also use this format if applicable
images = images.to(device, memory_format=torch.channels_last)
```

## 2. Mixed Precision (FP16/BF16)
Use `torch.cuda.amp` to reduce memory bandwidth and increase compute throughput.

- **BF16 (Brain Float 16)**: Recommended for Ampere+ GPUs (RTX 30/40/50). No GradScaler needed.
- **FP16**: Legacy, requires `GradScaler`.

```python
# BF16 Example (Ampere+)
with torch.cuda.amp.autocast(dtype=torch.bfloat16):
    output = model(input)
    loss = criterion(output, target)
# No scaler needed for bfloat16 in most cases
```

## 3. Scaled Dot Product Attention (SDPA)
PyTorch 2.0+ includes optimized attention kernels (FlashAttention, Memory Efficient Attention). Use them automatically:

```python
import torch.nn.functional as F

# Most modern Transformers use this internally
# To force a specific backend:
with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False):
    F.scaled_dot_product_attention(...)
```

## 4. Memory Management & Defragmentation
If encountering OOM due to fragmentation:

1. **Set max_split_size**: Reduces fragmentation by preventing large blocks from being split.
   `export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"`
2. **Garbage Collection**: Manually trigger `gc.collect()` before `torch.cuda.empty_cache()` if absolutely necessary.

## 5. Distributed Training Settings
When using DDP (DistributedDataParallel):
- Use `gradient_as_bucket_view=True` in DDP constructor.
- Use `static_graph=True` if the model graph doesn't change.
