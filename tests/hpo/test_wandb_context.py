# tests/hpo/test_wandb_context.py
import pytest
from ugro.hpo.tracking.wandb_context import WandbContextManager

def test_wandb_context_manager_cleanup():
    with patch('wandb.init') as mock_init, \
         patch('wandb.finish') as mock_finish:
        
        with WandbContextManager(project="test", config={"key": "value"}):
            pass
        
        mock_init.assert_called_once()
        mock_finish.assert_called_once()
