"""Comprehensive tests for the health monitoring system.

This module provides unit tests and integration tests for the health monitor,
following Python 3.12+ testing best practices with pytest and async testing.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ugro.health_monitor import (
    AdaptiveHealthMonitor,
    AlertLevel,
    CircuitBreaker,
    HealthMetrics,
    MonitoringConfig,
    NodeStatus,
    create_health_monitor,
)


class TestHealthMetrics:
    """Test cases for HealthMetrics dataclass."""
    
    def test_health_metrics_creation(self) -> None:
        """Test creating valid health metrics."""
        metrics = HealthMetrics(
            node_name="test-node",
            timestamp=datetime.now(),
            gpu_utilization=75.5,
            gpu_memory_used=8000.0,
            gpu_memory_total=12000.0,
            gpu_temperature=65.0,
            gpu_power_usage=250.0,
            cpu_utilization=45.0,
            memory_usage=60.0,
            disk_usage=30.0,
            network_latency=1.5,
            health_score=85.0,
        )
        
        assert metrics.node_name == "test-node"
        assert metrics.health_score == 85.0
        assert metrics.status == NodeStatus.HEALTHY

    def test_health_score_validation(self) -> None:
        """Test health score validation."""
        with pytest.raises(ValueError, match="Health score must be 0-100"):
            HealthMetrics(
                node_name="test",
                timestamp=datetime.now(),
                gpu_utilization=0,
                gpu_memory_used=0,
                gpu_memory_total=0,
                gpu_temperature=0,
                gpu_power_usage=0,
                cpu_utilization=0,
                memory_usage=0,
                disk_usage=0,
                network_latency=0,
                health_score=150.0,  # Invalid
            )

    def test_status_determination(self) -> None:
        """Test node status determination from health score."""
        healthy = HealthMetrics(
            node_name="test", timestamp=datetime.now(),
            gpu_utilization=0, gpu_memory_used=0, gpu_memory_total=0,
            gpu_temperature=0, gpu_power_usage=0, cpu_utilization=0,
            memory_usage=0, disk_usage=0, network_latency=0,
            health_score=90.0
        )
        assert healthy.status == NodeStatus.HEALTHY
        
        degraded = HealthMetrics(
            node_name="test", timestamp=datetime.now(),
            gpu_utilization=0, gpu_memory_used=0, gpu_memory_total=0,
            gpu_temperature=0, gpu_power_usage=0, cpu_utilization=0,
            memory_usage=0, disk_usage=0, network_latency=0,
            health_score=65.0
        )
        assert degraded.status == NodeStatus.DEGRADED
        
        unhealthy = HealthMetrics(
            node_name="test", timestamp=datetime.now(),
            gpu_utilization=0, gpu_memory_used=0, gpu_memory_total=0,
            gpu_temperature=0, gpu_power_usage=0, cpu_utilization=0,
            memory_usage=0, disk_usage=0, network_latency=0,
            health_score=30.0
        )
        assert unhealthy.status == NodeStatus.UNHEALTHY

    def test_alert_filtering(self) -> None:
        """Test alert filtering by severity."""
        alerts = [
            "CRITICAL: GPU temperature 90°C",
            "WARNING: Memory usage 92%",
            "INFO: Status change detected",
        ]
        
        metrics = HealthMetrics(
            node_name="test", timestamp=datetime.now(),
            gpu_utilization=0, gpu_memory_used=0, gpu_memory_total=0,
            gpu_temperature=0, gpu_power_usage=0, cpu_utilization=0,
            memory_usage=0, disk_usage=0, network_latency=0,
            health_score=30.0, alerts=alerts
        )
        
        assert len(metrics.critical_alerts) == 1
        assert len(metrics.warning_alerts) == 1
        assert metrics.critical_alerts[0] == alerts[0]


class TestCircuitBreaker:
    """Test cases for CircuitBreaker implementation."""
    
    def test_circuit_breaker_initial_state(self) -> None:
        """Test initial circuit breaker state."""
        breaker = CircuitBreaker()
        assert not breaker.is_open()
        assert breaker.failure_count == 0
        assert breaker.state.name == "CLOSED"

    def test_circuit_breaker_opens_after_failures(self) -> None:
        """Test circuit breaker opens after max failures."""
        breaker = CircuitBreaker()
        
        # Record failures up to max
        for _ in range(breaker.MAX_FAILURES):
            breaker.record_failure()
            assert not breaker.is_open()
        
        # One more failure should open it
        breaker.record_failure()
        assert breaker.is_open()
        assert breaker.state.name == "OPEN"

    def test_circuit_breaker_timeout(self) -> None:
        """Test circuit breaker timeout and half-open state."""
        breaker = CircuitBreaker()
        
        # Open the circuit breaker
        for _ in range(breaker.MAX_FAILURES + 1):
            breaker.record_failure()
        
        assert breaker.is_open()
        
        # Mock time passage beyond timeout
        breaker.last_failure_time = datetime.now() - timedelta(seconds=breaker.TIMEOUT_SECONDS + 10)
        
        # Should now be half-open
        assert not breaker.is_open()
        assert breaker.state.name == "HALF_OPEN"

    def test_circuit_breaker_reset_on_success(self) -> None:
        """Test circuit breaker resets on success."""
        breaker = CircuitBreaker()
        
        # Add some failures
        breaker.record_failure()
        breaker.record_failure()
        assert breaker.failure_count == 2
        
        # Success should reset
        breaker.record_success()
        assert breaker.failure_count == 0
        assert not breaker.is_open()


class TestMonitoringConfig:
    """Test cases for MonitoringConfig."""
    
    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = MonitoringConfig()
        
        assert config.base_interval == 10.0
        assert config.max_interval == 60.0
        assert config.min_interval == 5.0
        assert config.max_failures == 3
        assert config.max_workers == 8

    def test_config_validation_weights(self) -> None:
        """Test configuration weight validation."""
        with pytest.raises(ValueError, match="Weights must sum to 100"):
            MonitoringConfig(gpu_weight=50.0, system_weight=30.0, network_weight=20.0, process_weight=5.0)

    def test_config_validation_intervals(self) -> None:
        """Test configuration interval validation."""
        with pytest.raises(ValueError, match="Interval configuration invalid"):
            MonitoringConfig(min_interval=15.0, base_interval=10.0, max_interval=5.0)


class TestAdaptiveHealthMonitor:
    """Test cases for AdaptiveHealthMonitor."""
    
    @pytest.fixture
    def mock_cluster(self) -> MagicMock:
        """Create mock cluster for testing."""
        cluster = MagicMock()
        cluster.get_all_workers.return_value = [
            {"name": "worker-1", "ip": "192.168.1.10"},
            {"name": "worker-2", "ip": "192.168.1.11"},
        ]
        cluster.check_health.return_value = {
            "worker-1": {"healthy": True},
            "worker-2": {"healthy": True},
        }
        cluster.execute_on_worker.return_value = (True, "test output", "")
        return cluster

    @pytest.fixture
    def mock_state_manager(self) -> MagicMock:
        """Create mock state manager for testing."""
        state_manager = MagicMock()
        state_manager.update_node_status = MagicMock()
        return state_manager

    @pytest.fixture
    def health_monitor(self, mock_cluster: MagicMock, mock_state_manager: MagicMock) -> AdaptiveHealthMonitor:
        """Create health monitor for testing."""
        config = MonitoringConfig(base_interval=1.0, max_interval=5.0)  # Faster for testing
        return AdaptiveHealthMonitor(mock_cluster, mock_state_manager, config)

    def test_initialization(self, health_monitor: AdaptiveHealthMonitor) -> None:
        """Test health monitor initialization."""
        assert health_monitor.cluster is not None
        assert health_monitor.state_manager is not None
        assert health_monitor.config.base_interval == 1.0
        assert not health_monitor._running

    def test_adaptive_interval_calculation(self, health_monitor: AdaptiveHealthMonitor) -> None:
        """Test adaptive interval calculation."""
        # No active jobs - should use base interval
        interval = health_monitor._calculate_adaptive_interval()
        assert interval == health_monitor.config.base_interval
        
        # Active jobs - should use minimum interval
        health_monitor.register_job_activity("test-job")
        interval = health_monitor._calculate_adaptive_interval()
        assert interval == health_monitor.config.min_interval
        
        # Old activity - should use longer interval
        health_monitor._last_job_activity = datetime.now() - timedelta(hours=8)
        interval = health_monitor._calculate_adaptive_interval()
        assert interval > health_monitor.config.base_interval

    def test_health_score_calculation(self, health_monitor: AdaptiveHealthMonitor) -> None:
        """Test health score calculation."""
        # Perfect metrics
        score = health_monitor._calculate_health_score(
            {"temperature": 70, "memory_used": 4000, "memory_total": 8000},
            {"memory_usage": 50, "disk_usage": 30},
            {"latency": 10},
            {"1234": True}
        )
        assert score == 100.0
        
        # Poor metrics
        score = health_monitor._calculate_health_score(
            {"temperature": 90, "memory_used": 7600, "memory_total": 8000},
            {"memory_usage": 98, "disk_usage": 98},
            {"latency": 200},
            {}
        )
        assert score < 50  # Should be critical

    def test_alert_generation(self, health_monitor: AdaptiveHealthMonitor) -> None:
        """Test alert generation."""
        alerts = health_monitor._generate_alerts(
            "test-node",
            25.0,  # Critical health score
            {"temperature": 90},  # Critical temperature
            {"memory_usage": 98},  # Critical memory
            {}  # No processes
        )
        
        assert len(alerts) >= 3
        assert any(alert.startswith("CRITICAL:") for alert in alerts)
        assert any("health score 25.0" in alert for alert in alerts)
        assert any("temperature 90" in alert for alert in alerts)

    def test_circuit_breaker_integration(self, health_monitor: AdaptiveHealthMonitor) -> None:
        """Test circuit breaker integration."""
        node_name = "test-node"
        
        # Initially closed
        assert not health_monitor._is_circuit_breaker_open(node_name)
        
        # Record failures
        for _ in range(health_monitor.config.max_failures):
            health_monitor._handle_node_failure(node_name, Exception("Test error"))
        
        # Should be open now
        assert health_monitor._is_circuit_breaker_open(node_name)

    @pytest.mark.asyncio
    async def test_metric_collection_concurrent(self, health_monitor: AdaptiveHealthMonitor) -> None:
        """Test concurrent metric collection."""
        # Mock the metric collection methods
        health_monitor._get_detailed_gpu_metrics = AsyncMock(return_value={"utilization": 75.0})
        health_monitor._get_system_metrics = AsyncMock(return_value={"memory_usage": 50.0})
        health_monitor._get_network_metrics = AsyncMock(return_value={"latency": 10.0})
        health_monitor._get_process_metrics = AsyncMock(return_value={"1234": True})
        
        # Collect metrics
        metrics = await health_monitor._collect_metrics_concurrently()
        
        assert len(metrics) == 2  # Two workers
        assert all(isinstance(m, HealthMetrics) for m in metrics)

    @pytest.mark.asyncio
    async def test_gpu_metrics_collection(self, health_monitor: AdaptiveHealthMonitor) -> None:
        """Test GPU metrics collection."""
        worker = {"name": "test-worker", "ip": "192.168.1.10"}
        
        # Mock successful nvidia-smi output
        mock_output = "75,4096,8192,65,250"
        health_monitor.cluster.execute_on_worker.return_value = (True, mock_output, "")
        
        metrics = await health_monitor._get_detailed_gpu_metrics(worker)
        
        assert metrics["utilization"] == 75.0
        assert metrics["memory_used"] == 4096.0
        assert metrics["memory_total"] == 8192.0
        assert metrics["temperature"] == 65.0
        assert metrics["power_usage"] == 250.0

    @pytest.mark.asyncio
    async def test_system_metrics_collection(self, health_monitor: AdaptiveHealthMonitor) -> None:
        """Test system metrics collection."""
        worker = {"name": "test-worker", "ip": "192.168.1.10"}
        
        # Mock system command output
        mock_output = "CPU:45.5;MEM:67.2;DISK:32.1"
        health_monitor.cluster.execute_on_worker.return_value = (True, mock_output, "")
        
        metrics = await health_monitor._get_system_metrics(worker)
        
        assert metrics["cpu_util"] == 45.5
        assert metrics["memory_usage"] == 67.2
        assert metrics["disk_usage"] == 32.1

    @pytest.mark.asyncio
    async def test_metrics_processing(self, health_monitor: AdaptiveHealthMonitor) -> None:
        """Test metrics processing and state updates."""
        metrics = HealthMetrics(
            node_name="test-node",
            timestamp=datetime.now(),
            gpu_utilization=75.0,
            gpu_memory_used=4000.0,
            gpu_memory_total=8000.0,
            gpu_temperature=65.0,
            gpu_power_usage=200.0,
            cpu_utilization=45.0,
            memory_usage=60.0,
            disk_usage=30.0,
            network_latency=1.5,
            health_score=85.0,
        )
        
        await health_monitor._process_metrics([metrics])
        
        # Check metrics stored
        assert "test-node" in health_monitor._metrics_history
        assert len(health_monitor._metrics_history["test-node"]) == 1
        
        # Check health score stored
        assert health_monitor._health_scores["test-node"] == 85.0
        
        # Check state manager called
        health_monitor.state_manager.update_node_status.assert_called_once()

    def test_job_activity_tracking(self, health_monitor: AdaptiveHealthMonitor) -> None:
        """Test job activity tracking."""
        job_name = "test-job"
        
        # Register job activity
        health_monitor.register_job_activity(job_name)
        assert job_name in health_monitor._active_jobs
        
        # Unregister job activity
        health_monitor.unregister_job_activity(job_name)
        assert job_name not in health_monitor._active_jobs

    def test_statistics_collection(self, health_monitor: AdaptiveHealthMonitor) -> None:
        """Test statistics collection."""
        stats = health_monitor.get_statistics()
        
        assert "total_checks" in stats
        assert "failed_checks" in stats
        assert "critical_alerts" in stats
        assert "warning_alerts" in stats
        assert "active_jobs" in stats
        assert "monitored_nodes" in stats

    def test_cleanup_old_metrics(self, health_monitor: AdaptiveHealthMonitor) -> None:
        """Test cleanup of old metrics."""
        # Add old metrics
        old_time = datetime.now() - timedelta(hours=25)
        old_metrics = HealthMetrics(
            node_name="test-node", timestamp=old_time,
            gpu_utilization=0, gpu_memory_used=0, gpu_memory_total=0,
            gpu_temperature=0, gpu_power_usage=0, cpu_utilization=0,
            memory_usage=0, disk_usage=0, network_latency=0,
            health_score=50.0
        )
        
        health_monitor._metrics_history["test-node"] = [old_metrics]
        health_monitor._cleanup_old_metrics()
        
        # Should be cleaned up
        assert "test-node" not in health_monitor._metrics_history

    @pytest.mark.asyncio
    async def test_stop_graceful(self, health_monitor: AdaptiveHealthMonitor) -> None:
        """Test graceful shutdown."""
        # Start the monitor briefly
        health_monitor._running = True
        
        # Stop it
        await health_monitor.stop()
        
        assert not health_monitor._running
        assert health_monitor._executor._shutdown


class TestFactoryFunction:
    """Test cases for factory functions."""
    
    def test_create_health_monitor(self) -> None:
        """Test health monitor factory function."""
        mock_cluster = MagicMock()
        mock_state_manager = MagicMock()
        
        monitor = create_health_monitor(mock_cluster, mock_state_manager)
        
        assert isinstance(monitor, AdaptiveHealthMonitor)
        assert monitor.cluster is mock_cluster
        assert monitor.state_manager is mock_state_manager


class TestIntegration:
    """Integration tests for the health monitoring system."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_monitoring_cycle(self) -> None:
        """Test end-to-end monitoring cycle."""
        # Create mocks
        mock_cluster = MagicMock()
        mock_cluster.get_all_workers.return_value = [{"name": "test-worker", "ip": "192.168.1.10"}]
        mock_cluster.check_health.return_value = {"test-worker": {"healthy": True}}
        mock_cluster.execute_on_worker.return_value = (True, "75,4096,8192,65,250", "")
        
        mock_state_manager = MagicMock()
        
        # Create monitor with fast intervals for testing
        config = MonitoringConfig(base_interval=0.1, max_interval=0.5)
        monitor = AdaptiveHealthMonitor(mock_cluster, mock_state_manager, config)
        
        # Run one monitoring cycle
        with patch.object(monitor, '_calculate_adaptive_interval', return_value=0.1):
            # Mock metric collection to avoid actual SSH calls
            monitor._get_detailed_gpu_metrics = AsyncMock(return_value={
                "utilization": 75, "memory_used": 4096, "memory_total": 8192,
                "temperature": 65, "power_usage": 250
            })
            monitor._get_system_metrics = AsyncMock(return_value={
                "cpu_util": 45, "memory_usage": 60, "disk_usage": 30
            })
            monitor._get_network_metrics = AsyncMock(return_value={"latency": 10})
            monitor._get_process_metrics = AsyncMock(return_value={"1234": True})
            
            # Collect metrics
            metrics = await monitor._collect_metrics_concurrently()
            
            # Process metrics
            await monitor._process_metrics(metrics)
            
            # Check results
            assert len(metrics) == 1
            assert metrics[0].health_score > 80
            assert mock_state_manager.update_node_status.called
        
        # Cleanup
        await monitor.stop()


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
