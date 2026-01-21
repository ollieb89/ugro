"""Tests for conditional parameter functionality."""

import unittest
from ugro.hpo.config import ParameterBound
from ugro.hpo.search_space import (
    evaluate_condition,
    apply_conditional_parameters,
    parse_parameter_bounds,
)


class TestConditionEvaluation:
    """Test condition evaluation with safe AST parsing."""

    def test_simple_condition_true(self):
        """Test simple equality condition that evaluates to True."""
        config = {"optimizer_type": "adamw"}
        condition = "optimizer_type == 'adamw'"
        assert evaluate_condition(condition, config) is True

    def test_simple_condition_false(self):
        """Test simple equality condition that evaluates to False."""
        config = {"optimizer_type": "adam"}
        condition = "optimizer_type == 'adamw'"
        assert evaluate_condition(condition, config) is False

    def test_complex_condition(self):
        """Test condition with multiple operators."""
        config = {"batch_size": 32, "optimizer_type": "sgd"}
        condition = "batch_size >= 32 and optimizer_type == 'sgd'"
        assert evaluate_condition(condition, config) is True

    def test_in_condition(self):
        """Test 'in' operator for categorical values."""
        config = {"optimizer_type": "adam"}
        condition = "optimizer_type in ['adam', 'adamw']"
        assert evaluate_condition(condition, config) is True

    def test_empty_condition(self):
        """Test that empty condition returns True."""
        config = {"optimizer_type": "adam"}
        assert evaluate_condition("", config) is True
        assert evaluate_condition(None, config) is True

    def test_invalid_syntax(self):
        """Test that invalid syntax raises SyntaxError."""
        config = {"optimizer_type": "adam"}
        condition = "optimizer_type =="  # Missing value
        with self.assertRaises(SyntaxError):
            evaluate_condition(condition, config)

    def test_unsafe_operation(self):
        """Test that unsafe operations are rejected."""
        config = {"optimizer_type": "adam"}
        condition = "__import__('os').system('ls')"
        with self.assertRaises(ValueError):
            evaluate_condition(condition, config)

    def test_unknown_parameter(self):
        """Test that unknown parameters raise ValueError."""
        config = {"optimizer_type": "adam"}
        condition = "unknown_param == 'value'"
        with self.assertRaises(ValueError):
            evaluate_condition(condition, config)


class TestConditionalParameters:
    """Test conditional parameter application."""

    def test_apply_conditional_parameters_met(self):
        """Test parameters kept when condition is met."""
        bounds = [
            ParameterBound("optimizer_type", "categorical", choices=["adam", "adamw"]),
            ParameterBound("weight_decay", "float", min=0, max=0.1, default=0.01, 
                          condition="optimizer_type == 'adamw'"),
        ]
        config = {"optimizer_type": "adamw", "weight_decay": 0.05}
        
        filtered = apply_conditional_parameters(config, bounds)
        assert filtered == {"optimizer_type": "adamw", "weight_decay": 0.05}

    def test_apply_conditional_parameters_not_met_with_default(self):
        """Test parameter set to default when condition not met."""
        bounds = [
            ParameterBound("optimizer_type", "categorical", choices=["adam", "adamw"]),
            ParameterBound("weight_decay", "float", min=0, max=0.1, default=0.01,
                          condition="optimizer_type == 'adamw'"),
        ]
        config = {"optimizer_type": "adam", "weight_decay": 0.05}
        
        filtered = apply_conditional_parameters(config, bounds)
        assert filtered == {"optimizer_type": "adam", "weight_decay": 0.01}

    def test_apply_conditional_parameters_not_met_no_default(self):
        """Test parameter removed when condition not met and no default."""
        bounds = [
            ParameterBound("optimizer_type", "categorical", choices=["adam", "adamw"]),
            ParameterBound("weight_decay", "float", min=0, max=0.1,
                          condition="optimizer_type == 'adamw'"),
        ]
        config = {"optimizer_type": "adam", "weight_decay": 0.05}
        
        filtered = apply_conditional_parameters(config, bounds)
        assert filtered == {"optimizer_type": "adam"}

    def test_multiple_conditions(self):
        """Test multiple conditional parameters."""
        bounds = [
            ParameterBound("optimizer_type", "categorical", choices=["adam", "adamw", "sgd"]),
            ParameterBound("weight_decay", "float", min=0, max=0.1, default=0.01,
                          condition="optimizer_type == 'adamw'"),
            ParameterBound("momentum", "float", min=0.5, max=0.99, default=0.9,
                          condition="optimizer_type == 'sgd'"),
            ParameterBound("beta1", "float", min=0.8, max=0.999, default=0.9,
                          condition="optimizer_type in ['adam', 'adamw']"),
        ]
        
        # Test with AdamW
        config = {"optimizer_type": "adamw", "weight_decay": 0.05, "momentum": 0.95, "beta1": 0.95}
        filtered = apply_conditional_parameters(config, bounds)
        assert filtered == {"optimizer_type": "adamw", "weight_decay": 0.05, "beta1": 0.95}
        
        # Test with SGD
        config = {"optimizer_type": "sgd", "weight_decay": 0.05, "momentum": 0.95, "beta1": 0.95}
        filtered = apply_conditional_parameters(config, bounds)
        assert filtered == {"optimizer_type": "sgd", "momentum": 0.95}

    def test_no_conditions(self):
        """Test that parameters without conditions are unchanged."""
        bounds = [
            ParameterBound("learning_rate", "float", min=1e-5, max=1e-3),
            ParameterBound("batch_size", "categorical", choices=[8, 16, 32]),
        ]
        config = {"learning_rate": 0.001, "batch_size": 16}
        
        filtered = apply_conditional_parameters(config, bounds)
        assert filtered == config


class TestParseConditionalParameters:
    """Test parsing of conditional parameters from YAML."""

    def test_parse_with_conditions(self):
        """Test parsing YAML with conditional parameters."""
        config_dict = {
            "parameters": {
                "optimizer_type": {
                    "type": "categorical",
                    "choices": ["adam", "adamw"],
                    "default": "adamw",
                },
                "weight_decay": {
                    "type": "float",
                    "min": 0.0,
                    "max": 0.1,
                    "default": 0.01,
                    "condition": "optimizer_type == 'adamw'",
                },
            }
        }
        
        bounds = parse_parameter_bounds(config_dict)
        assert len(bounds) == 2
        
        # Check optimizer_type (no condition)
        opt_param = next(p for p in bounds if p.name == "optimizer_type")
        assert opt_param.condition is None
        
        # Check weight_decay (has condition)
        wd_param = next(p for p in bounds if p.name == "weight_decay")
        assert wd_param.condition == "optimizer_type == 'adamw'"

    def test_parse_without_conditions(self):
        """Test parsing YAML without conditional parameters."""
        config_dict = {
            "parameters": {
                "learning_rate": {
                    "type": "float",
                    "min": 1e-5,
                    "max": 1e-3,
                },
                "batch_size": {
                    "type": "categorical",
                    "choices": [8, 16, 32],
                },
            }
        }
        
        bounds = parse_parameter_bounds(config_dict)
        assert len(bounds) == 2
        for param in bounds:
            assert param.condition is None


if __name__ == "__main__":
    unittest.main()
