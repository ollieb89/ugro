"""Comprehensive tests for W&B integration."""

import os
import sys
import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from ugro.hpo.wandb_tracker import WandbTracker
from ugro.hpo.tracking_interface import NoOpTracker, MultiTracker
from ugro.hpo.security import validate_wandb_api_key, mask_api_key, validate_project_name


class TestWandbTracker:
    """Test W&B tracker implementation."""
    
    def test_init_without_wandb(self):
        """Test initialization when W&B is not installed."""
        with patch.dict('sys.modules', {'wandb': None}):
            # Reload module to trigger import error
            import importlib
            from ugro.hpo import wandb_tracker
            importlib.reload(wandb_tracker)
            
            tracker = wandb_tracker.WandbTracker(project="test")
            assert not tracker.is_active()
    
    def test_init_with_validation(self):
        """Test initialization with input validation."""
        # Invalid project name
        with pytest.raises(ValueError, match="Invalid project name"):
            WandbTracker(project="invalid project name!")
        
        # Invalid API key
        with pytest.raises(ValueError, match="Invalid API key format"):
            WandbTracker(project="valid-project", api_key="short")
    
    def test_api_key_masking(self):
        """Test API key masking functionality."""
        # Normal API key
        key = "0123456789abcdef0123456789abcdef01234567"
        masked = mask_api_key(key)
        assert masked == "0123456789ab************4567"
        
        # Short key
        short_key = "1234"
        masked = mask_api_key(short_key)
        assert masked == "***"
        
        # Empty key
        assert mask_api_key("") == "***"
        assert mask_api_key(None) == "***"
    
    def test_project_name_validation(self):
        """Test project name validation."""
        # Valid names
        assert validate_project_name("valid-project")
        assert validate_project_name("valid_project")
        assert validate_project_name("validproject123")
        assert validate_project_name("a")  # Minimum length
        
        # Invalid names
        assert not validate_project_name("")
        assert not validate_project_name("invalid project name!")
        assert not validate_project_name("invalid@project")
        assert not validate_project_name("a" * 129)  # Too long
    
    def test_api_key_validation(self):
        """Test API key validation."""
        # Valid keys
        assert validate_wandb_api_key("0123456789abcdef0123456789abcdef01234567")
        assert validate_wandb_api_key("0123456789abcdef0123456789abcdef0123456789abcdef")
        
        # Invalid keys
        assert not validate_wandb_api_key("")
        assert not validate_wandb_api_key(None)
        assert not validate_wandb_api_key("short")
        assert not validate_wandb_api_key("invalid-key!")
    
    @patch('wandb.init')
    @patch('wandb.log')
    @patch('wandb.finish')
    @patch('wandb.config')
    def test_tracking_workflow(self, mock_config, mock_finish, mock_log, mock_init):
        """Test complete tracking workflow."""
        # Setup mock
        mock_run = Mock()
        mock_run.id = "test-run-id"
        mock_run.tags = {}
        mock_init.return_value = mock_run
        mock_config.update = Mock()
        
        # Create tracker
        tracker = WandbTracker(
            project="test-project",
            api_key="0123456789abcdef0123456789abcdef01234567",
            tags={"env": "test"}
        )
        
        # Test logging
        with tracker.start_run("test-run"):
            assert tracker.is_active()
            
            # Log params
            tracker.log_params({"learning_rate": 0.001, "batch_size": 32})
            mock_config.update.assert_called_with({"learning_rate": 0.001, "batch_size": 32}, allow_val_change=True)
            
            # Log metrics
            tracker.log_metrics({"loss": 0.5, "accuracy": 0.9})
            mock_log.assert_called_with({"loss": 0.5, "accuracy": 0.9}, step=None)
            
            # Log tags
            tracker.log_tags({"model": "test"})
            assert mock_run.tags == {"model": "test"}
        
        # Verify finish was called
        mock_finish.assert_called_once()
    
    @patch('wandb.init')
    @patch('wandb.finish')
    def test_error_handling(self, mock_finish, mock_init):
        """Test error handling in W&B operations."""
        # Setup mock to raise exception
        mock_init.side_effect = Exception("W&B error")
        
        tracker = WandbTracker(project="test-project")
        
        # Should not raise exception
        with tracker.start_run("test-run"):
            assert not tracker.is_active()
        
        # Finish should still be called
        mock_finish.assert_called_once()


class TestTrackingInterface:
    """Test tracking interface abstractions."""
    
    def test_noop_tracker(self):
        """Test NoOpTracker does nothing."""
        tracker = NoOpTracker()
        
        assert not tracker.is_active()
        
        # All operations should do nothing
        tracker.log_params({"test": "value"})
        tracker.log_metrics({"loss": 0.5})
        tracker.log_tags({"tag": "value"})
        tracker.set_context("key", "value")
        
        with tracker.start_run("test"):
            pass  # Should not raise
        
        tracker.finish_run()
    
    @patch('ugro.hpo.wandb_tracker.WandbTracker')
    @patch('ugro.hpo.mlflow_tracker.MLflowTracker')
    def test_multi_tracker(self, mock_mlflow, mock_wandb):
        """Test MultiTracker forwards to all trackers."""
        # Setup mocks
        mock_wandb_tracker = Mock()
        mock_mlflow_tracker = Mock()
        mock_wandb.return_value = mock_wandb_tracker
        mock_mlflow.return_value = mock_mlflow_tracker
        
        # Configure mocks
        mock_wandb_tracker.is_active.return_value = True
        mock_mlflow_tracker.is_active.return_value = False
        
        # Create multi-tracker
        multi = MultiTracker([mock_wandb_tracker, mock_mlflow_tracker])
        
        # Test is_active
        assert multi.is_active()
        
        # Test forwarding
        multi.log_params({"lr": 0.001})
        mock_wandb_tracker.log_params.assert_called_once_with({"lr": 0.001})
        mock_mlflow_tracker.log_params.assert_called_once_with({"lr": 0.001})
        
        multi.log_metrics({"loss": 0.5})
        mock_wandb_tracker.log_metrics.assert_called_once_with({"loss": 0.5}, step=None)
        mock_mlflow_tracker.log_metrics.assert_called_once_with({"loss": 0.5}, step=None)


class TestWandbIntegration:
    """Test W&B integration with objective function."""
    
    @patch.dict(os.environ, {"WANDB_PROJECT": "test-project"})
    @patch('ugro.hpo.objective_refactored.WandbTracker')
    @patch('ugro.hpo.objective_refactored.MLflowTracker')
    def test_objective_with_wandb(self, mock_mlflow, mock_wandb):
        """Test objective function with W&B tracking."""
        from ugro.hpo.objective_refactored import LoRAFinetuningObjectiveRefactored
        
        # Setup mocks
        mock_wandb_tracker = Mock()
        mock_mlflow_tracker = Mock()
        mock_wandb.return_value = mock_wandb_tracker
        mock_mlflow.return_value = mock_mlflow_tracker
        
        # Create objective
        objective = LoRAFinetuningObjectiveRefactored(
            model_id="test/model",
            dataset_name="test_dataset",
            use_mlflow=True,
            use_wandb=True,
            wandb_project="test-project"
        )
        
        # Verify trackers were created
        mock_wandb.assert_called_once()
        mock_mlflow.assert_called_once()
    
    def test_constraint_violation_logging(self):
        """Test that constraint violations are logged properly."""
        with patch('ugro.hpo.objective_refactored.WandbTracker') as mock_wandb:
            mock_tracker = Mock()
            mock_wandb.return_value = mock_tracker
            
            from ugro.hpo.objective_refactored import LoRAFinetuningObjectiveRefactored
            
            objective = LoRAFinetuningObjectiveRefactored(
                model_id="test/model",
                dataset_name="test_dataset",
                constraints=["lora_alpha >= lora_r"],
                use_wandb=True,
                wandb_project="test-project"
            )
            
            # Call with violating parameters
            params = {"lora_alpha": 8, "lora_r": 16}  # Violates lora_alpha >= lora_r
            result = objective(params)
            
            # Should return penalty metrics
            assert "eval_loss" in result
            assert result["eval_loss"] > 1e5  # Large penalty
            
            # Verify violation was logged
            mock_tracker.start_run.assert_called()
            mock_tracker.log_tags.assert_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
