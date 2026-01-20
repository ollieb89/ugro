"""Tests for HPO parameter constraints."""

import pytest
from ugro.hpo.constraints import (
    ParameterValidator,
    LORA_CONSTRAINTS,
    TRANSFORMER_CONSTRAINTS,
    create_validator_from_config,
)


class TestParameterValidator:
    """Tests for ParameterValidator class."""

    def test_validate_all_pass(self):
        """Test validation when all constraints pass."""
        validator = ParameterValidator([
            "x > 0",
            "y < 100",
            "x < y",
        ])
        
        assert validator.validate({"x": 10, "y": 50}) is True

    def test_validate_constraint_fails(self):
        """Test validation when a constraint fails."""
        validator = ParameterValidator(["x > y"])
        
        assert validator.validate({"x": 5, "y": 10}) is False

    def test_validate_empty_constraints(self):
        """Test validation with no constraints always passes."""
        validator = ParameterValidator([])
        assert validator.validate({"anything": 123}) is True

    def test_validate_missing_param_fails(self):
        """Test that missing parameter causes validation to fail."""
        validator = ParameterValidator(["missing_param > 0"])
        
        assert validator.validate({"other_param": 10}) is False

    def test_validate_with_errors_returns_violations(self):
        """Test validate_with_errors returns list of violations."""
        validator = ParameterValidator([
            "x > 0",  # Will pass
            "y > 100",  # Will fail
            "z == 5",  # Will fail
        ])
        
        is_valid, errors = validator.validate_with_errors({"x": 10, "y": 50, "z": 3})
        
        assert is_valid is False
        assert len(errors) == 2
        assert any("y > 100" in e for e in errors)
        assert any("z == 5" in e for e in errors)

    def test_validate_with_errors_no_violations(self):
        """Test validate_with_errors with no violations."""
        validator = ParameterValidator(["x > 0"])
        
        is_valid, errors = validator.validate_with_errors({"x": 10})
        
        assert is_valid is True
        assert errors == []


class TestLoRAConstraints:
    """Tests for predefined LoRA constraints."""

    def test_lora_constraints_valid(self):
        """Test LoRA constraints with valid parameters."""
        validator = ParameterValidator(LORA_CONSTRAINTS)
        
        valid_params = {
            "lora_r": 16,
            "lora_alpha": 32,  # >= lora_r
            "lora_dropout": 0.1,  # 0.0 <= x <= 0.5
            "learning_rate": 0.0005,  # <= 0.01
            "batch_size": 32,  # <= 128
        }
        
        assert validator.validate(valid_params) is True

    def test_lora_constraints_alpha_less_than_r_fails(self):
        """Test that lora_alpha < lora_r fails."""
        validator = ParameterValidator(LORA_CONSTRAINTS)
        
        invalid_params = {
            "lora_r": 32,
            "lora_alpha": 16,  # < lora_r, should fail
            "lora_dropout": 0.1,
            "learning_rate": 0.0005,
            "batch_size": 32,
        }
        
        assert validator.validate(invalid_params) is False

    def test_lora_constraints_high_learning_rate_fails(self):
        """Test that learning_rate > 0.01 fails."""
        validator = ParameterValidator(LORA_CONSTRAINTS)
        
        invalid_params = {
            "lora_r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.1,
            "learning_rate": 0.05,  # > 0.01, should fail
            "batch_size": 32,
        }
        
        assert validator.validate(invalid_params) is False

    def test_lora_constraints_high_dropout_fails(self):
        """Test that lora_dropout > 0.5 fails."""
        validator = ParameterValidator(LORA_CONSTRAINTS)
        
        invalid_params = {
            "lora_r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.7,  # > 0.5, should fail
            "learning_rate": 0.0005,
            "batch_size": 32,
        }
        
        assert validator.validate(invalid_params) is False


class TestTransformerConstraints:
    """Tests for predefined Transformer constraints."""

    def test_transformer_constraints_defined(self):
        """Test that transformer constraints are defined."""
        assert len(TRANSFORMER_CONSTRAINTS) >= 3
        assert any("learning_rate" in c for c in TRANSFORMER_CONSTRAINTS)
        assert any("warmup_ratio" in c for c in TRANSFORMER_CONSTRAINTS)


class TestCreateValidatorFromConfig:
    """Tests for create_validator_from_config factory."""

    def test_create_with_preset_lora(self):
        """Test creating validator with lora preset."""
        validator = create_validator_from_config(preset="lora")
        
        # Should include LoRA constraints
        assert len(validator.constraints) == len(LORA_CONSTRAINTS)

    def test_create_with_preset_transformer(self):
        """Test creating validator with transformer preset."""
        validator = create_validator_from_config(preset="transformer")
        
        assert len(validator.constraints) == len(TRANSFORMER_CONSTRAINTS)

    def test_create_with_custom_constraints(self):
        """Test creating validator with custom constraints."""
        custom = ["x > 0", "y < 100"]
        validator = create_validator_from_config(constraints=custom)
        
        assert len(validator.constraints) == 2

    def test_create_combines_preset_and_custom(self):
        """Test that preset and custom constraints are combined."""
        custom = ["custom_param > 0"]
        validator = create_validator_from_config(
            constraints=custom,
            preset="lora",
        )
        
        # Should have LoRA constraints + custom
        expected_count = len(LORA_CONSTRAINTS) + len(custom)
        assert len(validator.constraints) == expected_count

    def test_create_empty_config(self):
        """Test creating validator with no constraints."""
        validator = create_validator_from_config()
        assert validator.constraints == []
