# tests/hpo/test_objective_abstraction.py
import pytest
from unittest.mock import patch
from ugro.hpo.objective import LoRAFinetuningObjective

def test_objective_uses_tracking_interface():
    objective = LoRAFinetuningObjective(
        model_name="test/model",
        dataset_path="test.csv",
        use_wandb=True,
        wandb_project="test-project"
    )
    
    # Should have tracking interface, not direct wandb
    assert hasattr(objective, '_trackers')
    assert not hasattr(objective, '_wandb_run')
