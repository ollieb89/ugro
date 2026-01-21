# tests/test_health_metrics.py
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_health_metrics_creation():
    """Test HealthMetrics dataclass creation"""
    from ugro.health_monitor import HealthMetrics
    
    metrics = HealthMetrics(
        node_name="gpu-master",
        timestamp=datetime.now(),
        gpu_utilization=85.5,
        gpu_memory_used=8.2,
        gpu_memory_total=12.0,
        gpu_temperature=72.0,
        gpu_power_usage=250.0,
        cpu_utilization=45.0,
        memory_usage=60.0,
        disk_usage=30.0,
        network_latency=0.5,
        process_status={"1234": True},
        health_score=85.0,
        alerts=[]
    )
    
    assert metrics.node_name == "gpu-master"
    assert metrics.health_score == 85.0
    assert len(metrics.alerts) == 0
    assert metrics.gpu_utilization == 85.5

def test_health_metrics_defaults():
    """Test HealthMetrics default values"""
    from ugro.health_monitor import HealthMetrics
    
    metrics = HealthMetrics(
        node_name="gpu1",
        timestamp=datetime.now(),
        gpu_utilization=0.0,
        gpu_memory_used=0.0,
        gpu_memory_total=8.0,
        gpu_temperature=0.0,
        gpu_power_usage=0.0,
        cpu_utilization=0.0,
        memory_usage=0.0,
        disk_usage=0.0,
        network_latency=0.0
    )
    
    # Should have defaults
    assert metrics.process_status == {}
    assert metrics.health_score == 0.0
    assert metrics.alerts == []
