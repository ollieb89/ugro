# tests/hpo/test_objective_cleanup.py
import pytest
from unittest.mock import patch, MagicMock
from ugro.hpo.objective import LoRAFinetuningObjective

def test_wandb_cleanup_on_exception():
    objective = LoRAFinetuningObjective(
        model_name="test/model",
        dataset_path="test.csv",
        use_wandb=True,
        wandb_project="test-project"
    )
    
    with patch('wandb.init') as mock_init, \
         patch('wandb.finish') as mock_finish, \
         patch.object(objective, '_train_model', side_effect=Exception("Training failed")):
        
        with pytest.raises(Exception):
            objective({})
        
        # Verify wandb.finish() is called even on exception
        mock_finish.assert_called_once()
