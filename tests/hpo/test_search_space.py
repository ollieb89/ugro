"""Tests for HPO search space parsing."""

import pytest
import tempfile
from pathlib import Path

from ugro.hpo.config import ParameterBound, Objective
from ugro.hpo.search_space import (
    load_search_space_yaml,
    parse_parameter_bounds,
    parse_objectives,
    parse_constraints,
    sample_defaults,
)


@pytest.fixture
def sample_yaml_config():
    """Create a temporary YAML config file."""
    yaml_content = """
parameters:
  lora_r:
    type: int
    min: 4
    max: 64
    step: 4
    default: 16
  
  lora_alpha:
    type: float
    min: 8.0
    max: 128.0
    log: true
    default: 32.0
  
  lora_dropout:
    type: float
    min: 0.0
    max: 0.3
    default: 0.1
  
  batch_size:
    type: categorical
    choices: [8, 16, 32, 64]
    default: 16

objectives:
  - name: eval_loss
    direction: minimize
    weight: 1.0

constraints:
  - "lora_alpha >= lora_r"
  - "learning_rate <= 0.01"
"""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False
    ) as f:
        f.write(yaml_content)
        return Path(f.name)


class TestLoadSearchSpaceYaml:
    """Tests for load_search_space_yaml function."""

    def test_load_valid_yaml(self, sample_yaml_config):
        """Test loading valid YAML config."""
        config = load_search_space_yaml(sample_yaml_config)
        
        assert "parameters" in config
        assert "objectives" in config
        assert "constraints" in config
        assert len(config["parameters"]) == 4

    def test_load_nonexistent_file_raises(self):
        """Test that loading nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_search_space_yaml("/nonexistent/path.yaml")


class TestParseParameterBounds:
    """Tests for parse_parameter_bounds function."""

    def test_parse_all_parameter_types(self, sample_yaml_config):
        """Test parsing all parameter types."""
        config = load_search_space_yaml(sample_yaml_config)
        bounds = parse_parameter_bounds(config)
        
        assert len(bounds) == 4
        
        # Check by name
        by_name = {b.name: b for b in bounds}
        
        # Int parameter
        assert by_name["lora_r"].type == "int"
        assert by_name["lora_r"].min == 4
        assert by_name["lora_r"].max == 64
        assert by_name["lora_r"].step == 4
        
        # Float with log scale
        assert by_name["lora_alpha"].type == "float"
        assert by_name["lora_alpha"].log is True
        
        # Float without log scale
        assert by_name["lora_dropout"].type == "float"
        assert by_name["lora_dropout"].log is False
        
        # Categorical
        assert by_name["batch_size"].type == "categorical"
        assert by_name["batch_size"].choices == [8, 16, 32, 64]

    def test_parse_empty_config_returns_empty(self):
        """Test that empty config returns empty list."""
        bounds = parse_parameter_bounds({})
        assert bounds == []

    def test_parse_invalid_type_raises(self):
        """Test that invalid parameter type raises ValueError."""
        config = {
            "parameters": {
                "bad_param": {"type": "invalid_type", "min": 0, "max": 1}
            }
        }
        with pytest.raises(ValueError, match="Invalid parameter type"):
            parse_parameter_bounds(config)


class TestParseObjectives:
    """Tests for parse_objectives function."""

    def test_parse_single_objective(self, sample_yaml_config):
        """Test parsing single objective."""
        config = load_search_space_yaml(sample_yaml_config)
        objectives = parse_objectives(config)
        
        assert len(objectives) == 1
        assert objectives[0].name == "eval_loss"
        assert objectives[0].direction == "minimize"
        assert objectives[0].weight == 1.0

    def test_parse_multi_objective(self):
        """Test parsing multiple objectives."""
        config = {
            "objectives": [
                {"name": "accuracy", "direction": "maximize", "weight": 0.7},
                {"name": "latency", "direction": "minimize", "weight": 0.3},
            ]
        }
        objectives = parse_objectives(config)
        
        assert len(objectives) == 2
        assert objectives[0].name == "accuracy"
        assert objectives[1].name == "latency"

    def test_parse_empty_objectives_returns_default(self):
        """Test that empty objectives returns default minimize loss."""
        objectives = parse_objectives({})
        
        assert len(objectives) == 1
        assert objectives[0].name == "eval_loss"
        assert objectives[0].direction == "minimize"

    def test_parse_invalid_direction_raises(self):
        """Test that invalid direction raises ValueError."""
        config = {
            "objectives": [{"name": "test", "direction": "invalid"}]
        }
        with pytest.raises(ValueError, match="Invalid direction"):
            parse_objectives(config)


class TestParseConstraints:
    """Tests for parse_constraints function."""

    def test_parse_constraints(self, sample_yaml_config):
        """Test parsing constraint expressions."""
        config = load_search_space_yaml(sample_yaml_config)
        constraints = parse_constraints(config)
        
        assert len(constraints) == 2
        assert "lora_alpha >= lora_r" in constraints
        assert "learning_rate <= 0.01" in constraints

    def test_parse_empty_constraints(self):
        """Test that empty config returns empty list."""
        constraints = parse_constraints({})
        assert constraints == []


class TestSampleDefaults:
    """Tests for sample_defaults function."""

    def test_sample_defaults(self, sample_yaml_config):
        """Test extracting default values."""
        config = load_search_space_yaml(sample_yaml_config)
        bounds = parse_parameter_bounds(config)
        defaults = sample_defaults(bounds)
        
        assert defaults["lora_r"] == 16
        assert defaults["lora_alpha"] == 32.0
        assert defaults["lora_dropout"] == 0.1
        assert defaults["batch_size"] == 16

    def test_sample_defaults_none_values_skipped(self):
        """Test that parameters without defaults are skipped."""
        bounds = [
            ParameterBound(name="with_default", type="int", min=0, max=10, default=5),
            ParameterBound(name="no_default", type="int", min=0, max=10),
        ]
        defaults = sample_defaults(bounds)
        
        assert "with_default" in defaults
        assert "no_default" not in defaults
