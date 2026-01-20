"""Training metrics emitter for UGRO worker nodes.

This module provides the MetricsEmitter class used by training scripts to
persist real-time metrics (loss, throughput, GPU stats) to disk.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from torch.utils.tensorboard import SummaryWriter

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@dataclass
class MetricsEmitter:
    """Emits training metrics to local disk for collection by the orchestrator.
    
    Features:
    - Atomic writes via temp file + rename
    - GPU status collection
    - Optional TensorBoard integration
    - Context manager support
    """
    job_id: str
    metrics_dir: Path
    rank: int = 0
    enable_tensorboard: bool = True
    
    _job_dir: Path = field(init=False)
    _metrics_file: Path = field(init=False)
    _tb_writer: Optional[SummaryWriter] = field(init=False, default=None)
    _logger: logging.Logger = field(init=False)

    def __post_init__(self) -> None:
        """Initialize directory structure and paths."""
        self.metrics_dir = Path(self.metrics_dir)
        self._job_dir = self.metrics_dir / self.job_id
        self._job_dir.mkdir(parents=True, exist_ok=True)
        
        # We use a rank-specific file name to avoid contention on multi-GPU nodes
        self._metrics_file = self._job_dir / f"metrics_rank{self.rank}.jsonl"
        self._logger = logging.getLogger(f"ugro.metrics_emitter.rank{self.rank}")
        
        if self.enable_tensorboard and HAS_TORCH:
            self._init_tensorboard()

    def _init_tensorboard(self) -> None:
        """Initialize TensorBoard SummaryWriter."""
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb_dir = self._job_dir / "tensorboard" / f"rank{self.rank}"
            tb_dir.mkdir(parents=True, exist_ok=True)
            self._tb_writer = SummaryWriter(log_dir=str(tb_dir))
        except Exception as e:
            self._logger.warning(f"Failed to initialize TensorBoard: {e}")
            self._tb_writer = None

    def emit_step(
        self, 
        step: int, 
        loss: float, 
        lr: float, 
        grad_norm: float, 
        throughput: float
    ) -> dict[str, Any]:
        """Emit a single training step's metrics.
        
        Args:
            step: Current training step/iteration
            loss: Current loss value
            lr: Current learning rate
            grad_norm: Gradient norm
            throughput: Tokens or samples per second
            
        Returns:
            The dict containing the emitted metrics
        """
        metrics = {
            "timestamp": time.time(),
            "job_id": self.job_id,
            "rank": self.rank,
            "step": step,
            "training_loss": float(loss),
            "learning_rate": float(lr),
            "gradient_norm": float(grad_norm),
            "throughput_tokens_sec": float(throughput),
        }
        
        # Add GPU stats
        metrics.update(self.emit_gpu_stats())
        
        # Atomic JSONL write: write to .tmp then rename
        # This ensures the reader never sees a partial line.
        # Note: This implementation keeps ONLY THE LATEST entry in the rank-file
        # which is sufficient for the orchestrator polling and very efficient.
        line = json.dumps(metrics) + "\n"
        tmp_file = self._metrics_file.with_suffix(".tmp")
        
        try:
            with open(tmp_file, "w") as f:
                f.write(line)
            os.rename(tmp_file, self._metrics_file)
        except Exception as e:
            self._logger.error(f"Failed to write metrics file: {e}")

        # TensorBoard logging
        if self._tb_writer:
            self._tb_writer.add_scalar("train/loss", loss, step)
            self._tb_writer.add_scalar("train/lr", lr, step)
            self._tb_writer.add_scalar("train/grad_norm", grad_norm, step)
            self._tb_writer.add_scalar("train/throughput", throughput, step)
            if "gpu_util" in metrics:
                self._tb_writer.add_scalar("gpu/utilization", metrics["gpu_util"], step)
            if "gpu_mem_used_gb" in metrics:
                self._tb_writer.add_scalar("gpu/memory_used_gb", metrics["gpu_mem_used_gb"], step)

        return metrics

    def emit_gpu_stats(self) -> dict[str, float]:
        """Collect current GPU utilization and memory stats.
        
        Prefers torch.cuda for memory info and nvidia-smi for utilization.
        """
        stats: dict[str, float] = {}
        
        if not HAS_TORCH or not torch.cuda.is_available():
            return stats

        try:
            # Current device for this rank
            device = torch.cuda.current_device()
            stats["gpu_mem_used_gb"] = torch.cuda.memory_allocated(device) / (1024**3)
            stats["gpu_mem_reserved_gb"] = torch.cuda.memory_reserved(device) / (1024**3)
            
            # Utilization via nvidia-smi
            res = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                encoding='utf-8',
                stderr=subprocess.DEVNULL
            )
            # Pick the line corresponding to our device index
            lines = res.strip().split("\n")
            if lines:
                if len(lines) > device:
                    stats["gpu_util"] = float(lines[device])
                else:
                    stats["gpu_util"] = float(lines[0])
        except Exception:
            # Fallback for systems where nvidia-smi is unavailable or fails
            pass
            
        return stats

    def get_tensorboard_writer(self) -> Optional[SummaryWriter]:
        """Direct access to the underlying TensorBoard SummaryWriter."""
        return self._tb_writer

    def __enter__(self) -> MetricsEmitter:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Close resources on exit."""
        if self._tb_writer:
            try:
                self._tb_writer.close()
            except Exception:
                pass
