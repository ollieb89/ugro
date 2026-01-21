# tests/test_adaptive_monitor.py
import sys
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import MagicMock

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_adaptive_interval_calculation():
    """Test adaptive polling interval calculation"""
    from ugro.health_monitor import AdaptiveHealthMonitor, MonitoringConfig
    
    # Create mock objects
    mock_cluster = MagicMock()
    mock_state_manager = MagicMock()
    
    # Create monitor with default config
    monitor = AdaptiveHealthMonitor(
        cluster=mock_cluster,
        state_manager=mock_state_manager,
        config=MonitoringConfig()
    )
    
    # Test default intervals
    assert monitor.config.base_interval == 10.0
    assert monitor.config.min_interval == 5.0
    assert monitor.config.max_interval == 60.0
    
    # Test activity tracking
    monitor.register_job_activity('test_job')
    assert 'test_job' in monitor.active_jobs
    
    monitor.unregister_job_activity('test_job')
    assert 'test_job' not in monitor.active_jobs

def test_monitor_initialization():
    """Test monitor initialization with config"""
    from ugro.health_monitor import AdaptiveHealthMonitor, MonitoringConfig
    
    mock_cluster = MagicMock()
    mock_state_manager = MagicMock()
    
    config = MonitoringConfig(
        base_interval=15.0,
        min_interval=7.5,
        max_interval=90.0
    )
    
    monitor = AdaptiveHealthMonitor(
        cluster=mock_cluster,
        state_manager=mock_state_manager,
        config=config
    )
    
    assert monitor.config.base_interval == 15.0
    assert monitor.config.min_interval == 7.5
    assert monitor.config.max_interval == 90.0
