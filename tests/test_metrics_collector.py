#!/usr/bin/env python3
"""Tests for the TrainingMetricsCollector functionality."""

import asyncio
import json
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

from src.ugro.health_monitor import (
    TrainingMetrics,
    TrainingMetricsCollector,
    MetricsCollectorConfig,
    create_metrics_collector
)


class TestTrainingMetrics:
    """Test the TrainingMetrics dataclass."""
    
    def test_valid_metrics_creation(self):
        """Test creating valid training metrics."""
        metrics = TrainingMetrics(
            timestamp=datetime.now(),
            job_id="test_job",
            rank=0,
            gpu_util=85.5,
            gpu_mem_used_gb=12.3,
            training_loss=2.1,
            throughput_tokens_sec=156.7,
            gradient_norm=1.8,
            learning_rate=0.0002
        )
        
        assert metrics.job_id == "test_job"
        assert metrics.rank == 0
        assert metrics.gpu_util == 85.5
        assert 0 <= metrics.efficiency_score <= 100
    
    def test_invalid_gpu_utilization(self):
        """Test validation of invalid GPU utilization."""
        with pytest.raises(ValueError, match="GPU utilization must be 0-100"):
            TrainingMetrics(
                timestamp=datetime.now(),
                job_id="test_job",
                rank=0,
                gpu_util=150.0,  # Invalid: > 100
                gpu_mem_used_gb=12.3,
                training_loss=2.1,
                throughput_tokens_sec=156.7,
                gradient_norm=1.8,
                learning_rate=0.0002
            )
    
    def test_negative_values(self):
        """Test validation of negative values."""
        with pytest.raises(ValueError, match="cannot be negative"):
            TrainingMetrics(
                timestamp=datetime.now(),
                job_id="test_job",
                rank=0,
                gpu_util=85.5,
                gpu_mem_used_gb=-1.0,  # Invalid: negative
                gpu_memory_total=16.0,
                gpu_temperature=75.0,
                gpu_power_usage=250.0,
                cpu_utilization=45.0,
                memory_usage=60.0,
                disk_usage=40.0,
                network_latency=1.5,
                health_score=85.0
            )
    
    def test_efficiency_score_calculation(self):
        """Test efficiency score calculation."""
        # High efficiency metrics
        high_eff = TrainingMetrics(
            timestamp=datetime.now(),
            job_id="test_job",
            rank=0,
            gpu_util=95.0,
            gpu_mem_used_gb=12.3,
            training_loss=0.5,
            throughput_tokens_sec=200.0,
            gradient_norm=1.0,
            learning_rate=0.0002
        )
        
        # Low efficiency metrics
        low_eff = TrainingMetrics(
            timestamp=datetime.now(),
            job_id="test_job",
            rank=0,
            gpu_util=30.0,
            gpu_mem_used_gb=12.3,
            training_loss=10.0,
            throughput_tokens_sec=20.0,
            gradient_norm=5.0,
            learning_rate=0.0002
        )
        
        assert high_eff.efficiency_score > low_eff.efficiency_score
        assert 0 <= high_eff.efficiency_score <= 100
        assert 0 <= low_eff.efficiency_score <= 100


class TestMetricsCollectorConfig:
    """Test the MetricsCollectorConfig class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = MetricsCollectorConfig()
        
        assert config.collection_interval == 5.0
        assert config.max_history_size == 1000
        assert config.low_throughput_threshold == 50.0
        assert config.high_loss_threshold == 10.0
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = MetricsCollectorConfig(
            collection_interval=2.0,
            max_history_size=500,
            low_throughput_threshold=100.0
        )
        
        assert config.collection_interval == 2.0
        assert config.max_history_size == 500
        assert config.low_throughput_threshold == 100.0
    
    def test_invalid_config(self):
        """Test validation of invalid configuration."""
        with pytest.raises(ValueError, match="Collection interval must be positive"):
            MetricsCollectorConfig(collection_interval=-1.0)
        
        with pytest.raises(ValueError, match="Max history size must be positive"):
            MetricsCollectorConfig(max_history_size=0)


class TestTrainingMetricsCollector:
    """Test the TrainingMetricsCollector class."""
    
    @pytest.fixture
    def mock_cluster(self):
        """Create a mock cluster for testing."""
        cluster = MagicMock()
        cluster.get_all_workers.return_value = [
            {"name": "gpu0", "ip": "192.168.1.10"},
            {"name": "gpu1", "ip": "192.168.1.11"}
        ]
        cluster.execute_on_worker = AsyncMock()
        return cluster
    
    @pytest.fixture
    def collector(self, mock_cluster):
        """Create a metrics collector for testing."""
        config = MetricsCollectorConfig(
            collection_interval=0.1,  # Fast collection for tests
            max_history_size=10
        )
        return create_metrics_collector(mock_cluster, config)
    
    @pytest.mark.asyncio
    async def test_start_stop_collection(self, collector):
        """Test starting and stopping metrics collection."""
        job_id = "test_job"
        ranks = [0, 1]
        
        # Start collection
        await collector.start_collection(job_id, ranks)
        
        # Check that collection task was created
        assert job_id in collector._collection_tasks
        assert collector._stats["active_jobs"] == 1
        
        # Stop collection
        await collector.stop_collection(job_id)
        
        # Check that collection task was removed
        assert job_id not in collector._collection_tasks
        assert collector._stats["active_jobs"] == 0
    
    @pytest.mark.asyncio
    async def test_metrics_storage(self, collector):
        """Test metrics storage and retrieval."""
        job_id = "test_job"
        rank = 0
        
        # Manually add metrics
        metrics = TrainingMetrics(
            timestamp=datetime.now(),
            job_id=job_id,
            rank=rank,
            gpu_util=85.0,
            gpu_mem_used_gb=12.0,
            training_loss=2.0,
            throughput_tokens_sec=150.0,
            gradient_norm=1.5,
            learning_rate=0.0002
        )
        
        collector._metrics_storage[job_id] = {rank: [metrics]}
        
        # Test retrieval
        latest = collector.get_latest_metrics(job_id, rank)
        assert latest is not None
        assert latest.rank == rank
        assert latest.gpu_util == 85.0
        
        # Test history retrieval
        history = collector.get_metrics_history(job_id, rank)
        assert len(history) == 1
        assert history[0].rank == rank
    
    def test_get_job_statistics(self, collector):
        """Test job statistics calculation."""
        job_id = "test_job"
        
        # Add some metrics
        metrics1 = TrainingMetrics(
            timestamp=datetime.now(),
            job_id=job_id,
            rank=0,
            gpu_util=80.0,
            gpu_mem_used_gb=12.0,
            training_loss=2.0,
            throughput_tokens_sec=150.0,
            gradient_norm=1.5,
            learning_rate=0.0002
        )
        
        metrics2 = TrainingMetrics(
            timestamp=datetime.now(),
            job_id=job_id,
            rank=1,
            gpu_util=90.0,
            gpu_mem_used_gb=14.0,
            training_loss=1.5,
            throughput_tokens_sec=180.0,
            gradient_norm=1.2,
            learning_rate=0.0002
        )
        
        collector._metrics_storage[job_id] = {0: [metrics1], 1: [metrics2]}
        
        stats = collector.get_job_statistics(job_id)
        
        assert stats["status"] == "active"
        assert stats["ranks"] == 2
        assert stats["total_metrics"] == 2
        assert stats["avg_gpu_util"] == 85.0  # (80 + 90) / 2
        assert stats["avg_throughput"] == 165.0  # (150 + 180) / 2
    
    def test_get_statistics(self, collector):
        """Test collector statistics."""
        stats = collector.get_statistics()
        
        assert "total_collections" in stats
        assert "failed_collections" in stats
        assert "metrics_collected" in stats
        assert "active_jobs" in stats
        assert "circuit_breakers_open" in stats
        assert "stored_jobs" in stats
    
    @pytest.mark.asyncio
    async def test_circuit_breaker(self, collector):
        """Test circuit breaker functionality."""
        job_id = "test_job"
        rank = 0
        key = f"{job_id}_{rank}"
        
        # Get circuit breaker
        breaker = collector._get_circuit_breaker(key)
        assert not breaker.is_open()
        
        # Record failures
        breaker.record_failure()
        breaker.record_failure()
        assert not breaker.is_open()  # Should still be closed
        
        # Record third failure - should open
        breaker.record_failure()
        assert breaker.is_open()
        
        # Reset on success
        breaker.record_success()
        assert not breaker.is_open()
    
    @pytest.mark.asyncio
    async def test_cleanup_old_metrics(self, collector):
        """Test cleanup of old metrics."""
        job_id = "test_job"
        rank = 0
        
        # Create old metrics
        old_timestamp = datetime.now() - asyncio.sleep(25)  # 25 hours ago
        old_metrics = TrainingMetrics(
            timestamp=old_timestamp,
            job_id=job_id,
            rank=rank,
            gpu_util=85.0,
            gpu_mem_used_gb=12.0,
            training_loss=2.0,
            throughput_tokens_sec=150.0,
            gradient_norm=1.5,
            learning_rate=0.0002
        )
        
        # Create recent metrics
        recent_metrics = TrainingMetrics(
            timestamp=datetime.now(),
            job_id=job_id,
            rank=rank,
            gpu_util=90.0,
            gpu_mem_used_gb=14.0,
            training_loss=1.5,
            throughput_tokens_sec=180.0,
            gradient_norm=1.2,
            learning_rate=0.0002
        )
        
        collector._metrics_storage[job_id] = {rank: [old_metrics, recent_metrics]}
        
        # Run cleanup (24 hour threshold)
        collector._cleanup_old_metrics(job_id)
        
        # Should only have recent metrics
        remaining = collector._metrics_storage[job_id][rank]
        assert len(remaining) == 1
        assert remaining[0].gpu_util == 90.0  # Recent metric


class TestMetricsCollectionMethods:
    """Test the metrics collection methods."""
    
    @pytest.fixture
    def mock_cluster(self):
        """Create a mock cluster with specific responses."""
        cluster = MagicMock()
        cluster.get_all_workers.return_value = [
            {"name": "gpu0", "ip": "192.168.1.10"}
        ]
        
        # Mock successful metrics file read
        cluster.execute_on_worker = AsyncMock(return_value=(
            True,
            json.dumps({
                "gpu_util": 85.5,
                "gpu_mem_used_gb": 12.3,
                "training_loss": 2.1,
                "throughput_tokens_sec": 156.7,
                "gradient_norm": 1.8,
                "learning_rate": 0.0002
            }),
            ""
        ))
        
        return cluster
    
    @pytest.mark.asyncio
    async def test_read_metrics_from_file(self, mock_cluster):
        """Test reading metrics from JSON file."""
        config = MetricsCollectorConfig()
        collector = TrainingMetricsCollector(mock_cluster, config)
        
        metrics = await collector._read_metrics_from_file("test_job", 0, "gpu0")
        
        assert metrics is not None
        assert metrics.job_id == "test_job"
        assert metrics.rank == 0
        assert metrics.gpu_util == 85.5
        assert metrics.training_loss == 2.1
    
    @pytest.mark.asyncio
    async def test_estimate_metrics_from_gpu(self, mock_cluster):
        """Test estimating metrics from GPU utilization."""
        # Mock GPU query response
        mock_cluster.execute_on_worker.return_value = (
            True,
            "85.0,13312",  # 85% utilization, 13GB memory
            ""
        )
        
        config = MetricsCollectorConfig()
        collector = TrainingMetricsCollector(mock_cluster, config)
        
        metrics = await collector._estimate_metrics_from_gpu("test_job", 0, "gpu0")
        
        assert metrics is not None
        assert metrics.gpu_util == 85.0
        assert metrics.gpu_mem_used_gb == 13.0  # 13312 MB / 1024


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
