# tests/hpo/test_wandb_tracker.py
import pytest
from unittest.mock import patch
from ugro.hpo.tracking.wandb_tracker import WandbTracker

def test_wandb_tracker_logs():
    with patch('wandb.init') as mock_init, \
         patch('wandb.log') as mock_log, \
         patch('wandb.config') as mock_config:
        
        tracker = WandbTracker(project="test")
        tracker.log_metrics({"loss": 0.5})
        
        mock_log.assert_called_once_with({"loss": 0.5})
