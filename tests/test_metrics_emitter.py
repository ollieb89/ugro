import json
import os
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from ugro.metrics_emitter import MetricsEmitter


def test_metrics_emitter_init(tmp_path):
    """Test MetricsEmitter initialization and directory creation."""
    job_id = "test_job_123"
    emitter = MetricsEmitter(
        job_id=job_id,
        metrics_dir=tmp_path,
        rank=0,
        enable_tensorboard=False
    )
    
    assert emitter.job_id == job_id
    assert (tmp_path / job_id).exists()
    assert (tmp_path / job_id).is_dir()


def test_metrics_emitter_emit_step(tmp_path):
    """Test that emit_step correctly writes JSONL data."""
    job_id = "test_job_emit"
    emitter = MetricsEmitter(
        job_id=job_id,
        metrics_dir=tmp_path,
        rank=2,
        enable_tensorboard=False
    )
    
    # Mock GPU stats to avoid dependencies
    with patch.object(emitter, 'emit_gpu_stats', return_value={"gpu_util": 85.0}):
        metrics = emitter.emit_step(
            step=100,
            loss=0.5,
            lr=0.001,
            grad_norm=1.2,
            throughput=150.5
        )
        
    assert metrics["step"] == 100
    assert metrics["training_loss"] == 0.5
    assert metrics["gpu_util"] == 85.0
    assert metrics["rank"] == 2
    
    # Verify file content
    metrics_file = tmp_path / job_id / "metrics_rank2.jsonl"
    assert metrics_file.exists()
    
    with open(metrics_file, "r") as f:
        line = f.readline()
        data = json.loads(line)
        assert data["step"] == 100
        assert data["training_loss"] == 0.5
        assert data["rank"] == 2


def test_metrics_emitter_atomic_write(tmp_path):
    """Test atomic write by checking that no partial files are left and content is updated."""
    job_id = "test_job_atomic"
    emitter = MetricsEmitter(
        job_id=job_id,
        metrics_dir=tmp_path,
        rank=0,
        enable_tensorboard=False
    )
    
    metrics_file = tmp_path / job_id / "metrics_rank0.jsonl"
    
    # First write
    emitter.emit_step(1, 1.0, 0.1, 0.1, 10.0)
    with open(metrics_file, "r") as f:
        first_content = f.read()
    
    # Second write (overwrites in our atomic implementation)
    emitter.emit_step(2, 0.5, 0.1, 0.1, 20.0)
    with open(metrics_file, "r") as f:
        second_content = f.read()
    
    assert first_content != second_content
    assert json.loads(second_content)["step"] == 2
    # Ensure no .tmp file left behind
    assert not (tmp_path / job_id / "metrics_rank0.jsonl.tmp").exists()


def test_metrics_emitter_context_manager(tmp_path):
    """Test context manager support."""
    job_id = "test_job_ctx"
    
    # Use patch to avoid SummaryWriter init if torch is available
    with patch("ugro.metrics_emitter.HAS_TORCH", False):
        with MetricsEmitter(job_id, tmp_path, rank=0, enable_tensorboard=True) as emitter:
            assert isinstance(emitter, MetricsEmitter)
            emitter.emit_step(1, 1.0, 0.1, 0.1, 10.0)
    
    assert (tmp_path / job_id / "metrics_rank0.jsonl").exists()


@patch("subprocess.check_output")
@patch("torch.cuda.is_available", return_value=True)
@patch("torch.cuda.current_device", return_value=0)
@patch("torch.cuda.memory_allocated", return_value=1024**3)
@patch("torch.cuda.memory_reserved", return_value=2*1024**3)
def test_emit_gpu_stats(mock_reserved, mock_allocated, mock_device, mock_cuda_avail, mock_subprocess, tmp_path):
    """Test GPU statistics collection with mocks."""
    mock_subprocess.return_value = "75.0\n"
    
    # We need to make sure HAS_TORCH is True for this test
    with patch("ugro.metrics_emitter.HAS_TORCH", True):
        emitter = MetricsEmitter("job", tmp_path, rank=0, enable_tensorboard=False)
        stats = emitter.emit_gpu_stats()
        
        assert stats["gpu_mem_used_gb"] == 1.0
        assert stats["gpu_mem_reserved_gb"] == 2.0
        assert stats["gpu_util"] == 75.0
        
        mock_subprocess.assert_called_once()
