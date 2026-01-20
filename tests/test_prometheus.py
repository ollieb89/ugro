import pytest
from unittest.mock import MagicMock, patch
from ugro.health_monitor import AdaptiveHealthMonitor, MonitoringConfig
from ugro.scheduler.resources import ResourceTracker
from datetime import datetime

# --- Health Monitor Tests ---

@patch("ugro.health_monitor.PROMETHEUS_AVAILABLE", True)
@patch("ugro.health_monitor.GPU_UTIL")
@patch("ugro.health_monitor.SYS_CPU")
@patch("ugro.health_monitor.NODE_HEALTH")
@pytest.mark.asyncio
async def test_health_monitor_updates_prometheus(mock_node_health, mock_sys_cpu, mock_gpu_util):
    """Test that health monitor updates prometheus gauges."""
    monitor = AdaptiveHealthMonitor(
        cluster=MagicMock(),
        state_manager=MagicMock(),
        config=MonitoringConfig()
    )
    
    # Mock mocks
    monitor.check_node_health = MagicMock(return_value=True) # Since it's await-ed but in this code base it seems check_node_health IS async. 
    # Wait, check_node_health IS async in the file view (line 457). So we need AsyncMock or a coroutine.
    # unittest.mock.AsyncMock is available in 3.8+
    from unittest.mock import AsyncMock

    monitor.check_node_health = AsyncMock(return_value=True)
    monitor.cluster.check_health.return_value = {"node1": {"healthy": True}}
    
    monitor._get_detailed_gpu_metrics = AsyncMock(return_value={
        "utilization": 80.0, "memory_used": 0, "memory_total": 0, "temperature": 0, "power_usage": 0
    })
    monitor._get_system_metrics = AsyncMock(return_value={
        "cpu_util": 50.0, "memory_usage": 0, "disk_usage": 0
    })
    monitor._get_network_metrics = AsyncMock(return_value={"latency": 0})
    monitor._get_process_metrics = AsyncMock(return_value={})

    # Call target method
    worker = {"name": "node1", "ip": "1.2.3.4"}
    await monitor._collect_node_metrics(worker)
    
    # Verify calls
    # Note: Gauges use labels(...).set(...)
    mock_sys_cpu.labels.assert_called_with(node="node1")
    mock_sys_cpu.labels.return_value.set.assert_called_with(50.0)
    
    mock_gpu_util.labels.assert_called_with(node="node1", gpu_index="0")
    mock_gpu_util.labels.return_value.set.assert_called_with(80.0)
    
    mock_node_health.labels.assert_called_with(node="node1")
    # Health score is calculated, just check it was set
    mock_node_health.labels.return_value.set.assert_called()

# --- Resource Tracker Tests ---

@patch("ugro.scheduler.resources.requests.get")
def test_resource_tracker_sync_prometheus(mock_get):
    """Test that resource tracker updates state from prometheus query."""
    tracker = ResourceTracker(prometheus_url="http://mock:9090")
    
    # Mock response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "status": "success",
        "data": {
            "result": [
                {
                    "metric": {"node": "node1", "gpu_index": "0"},
                    "value": [1234567890, "8589934592"] # 8GB in bytes
                }
            ]
        }
    }
    mock_get.return_value = mock_response
    
    tracker.sync_from_prometheus()
    
    assert "node1" in tracker.node_usage
    assert tracker.node_usage["node1"]["vram_gb"] == 8.0
    mock_get.assert_called_once() # Should call requests.get
