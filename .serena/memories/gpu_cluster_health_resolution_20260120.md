# GPU Cluster Health Resolution - Jan 20, 2026

## Incident Summary
`gpu2` (RTX 3070 Ti) and `gpu1` (RTX 4070 Laptop) were reporting "Python environment issues" during cluster health checks. `gpu1` was previously falling back to a simulated healthy state due to SSH connection flakiness, while `gpu2` was failing real health checks.

## Root Cause
Missing CUDA-related system libraries in the `pixi` `cuda` environment. Specifically, PyTorch 2.7.1+cu128 required:
- `libcusparseLt.so.0` (provided by `cusparselt`)
- `libnccl.so.2` (provided by `nccl`)
- `libcudnn.so` (provided by `cudnn`)

## Resolution
1.  **Dependency Updates**: Modified `pixi.toml` to explicitly include these libraries in the `cuda` feature:
    ```toml
    [feature.cuda.dependencies]
    libcusparse = ">=12.1"
    libcusparse-dev = ">=12.1"
    cusparselt = ">=0.6"
    cusparselt-dev = ">=0.6"
    nccl = ">=2.18"
    cudnn = ">=8.9"
    cuda-runtime = ">=12.1"
    ```
2.  **Environment Sync**:
    - Ran `pixi install` on master to update `pixi.lock`.
    - Used `rsync` to push updated configuration to worker nodes.
    - Executed remote `pixi install -e cuda` on both `gpu1` and `gpu2`.

## Current Status
- All nodes (master, gpu1, gpu2) are reporting healthy with real environment status.
- `gpu1` is no longer in "simulated" mode.
- Worker operation tests (remote command execution) are passing.

## Knowledge for Future Tasks
- When upgrading PyTorch or using newer CUDA indices (like `cu128`), ensure `cusparselt`, `nccl`, and `cudnn` are explicitly added to the `pixi` dependencies as they are not always bundled in the PyPI wheels.
- `cusparselt` (note the name) is the correct package for `libcusparseLt` in the `nvidia` channel.
