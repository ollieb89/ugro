# tests/hpo/test_tracking_interface.py
import pytest
from ugro.hpo.tracking.interface import TrackingInterface, TrackingSystem

def test_tracking_interface():
    # Test that interface defines required methods
    assert hasattr(TrackingInterface, 'log_params')
    assert hasattr(TrackingInterface, 'log_metrics')
    assert hasattr(TrackingInterface, 'set_tags')
    assert hasattr(TrackingInterface, 'finish')
    
    # Test TrackingSystem enum
    assert TrackingSystem.MLFLOW.value == "mlflow"
    assert TrackingSystem.WANDB.value == "wandb"
    assert TrackingSystem.BOTH.value == "both"
