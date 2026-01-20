"""Tests for HPO configuration classes."""

import pytest
from ugro.hpo.config import (
    HPOConfig,
    ParameterBound,
    Objective,
    OptimizerAlgorithm,
    SamplerType,
)


class TestParameterBound:
    """Tests for ParameterBound dataclass."""

    def test_int_parameter_valid(self):
        """Test valid integer parameter."""
        param = ParameterBound(
            name="lora_r",
            type="int",
            min=4,
            max=64,
            step=4,
            default=16,
        )
        assert param.name == "lora_r"
        assert param.type == "int"
        assert param.min == 4
        assert param.max == 64
        assert param.step == 4
        assert param.log is False

    def test_float_parameter_log_scale(self):
        """Test float parameter with log scale."""
        param = ParameterBound(
            name="learning_rate",
            type="float",
            min=1e-5,
            max=1e-2,
            log=True,
            default=5e-4,
        )
        assert param.type == "float"
        assert param.log is True
        assert param.min == 1e-5
        assert param.max == 1e-2

    def test_categorical_parameter(self):
        """Test categorical parameter with choices."""
        param = ParameterBound(
            name="batch_size",
            type="categorical",
            choices=[8, 16, 32, 64],
            default=16,
        )
        assert param.type == "categorical"
        assert param.choices == [8, 16, 32, 64]

    def test_int_parameter_missing_bounds_raises(self):
        """Test that int/float without min/max raises ValueError."""
        with pytest.raises(ValueError, match="requires min and max"):
            ParameterBound(name="bad_param", type="int", min=None, max=None)

    def test_int_parameter_invalid_bounds_raises(self):
        """Test that min > max raises ValueError."""
        with pytest.raises(ValueError, match="min .* > max"):
            ParameterBound(name="bad_param", type="int", min=100, max=10)

    def test_categorical_missing_choices_raises(self):
        """Test that categorical without choices raises ValueError."""
        with pytest.raises(ValueError, match="requires choices"):
            ParameterBound(name="bad_param", type="categorical", choices=None)


class TestObjective:
    """Tests for Objective dataclass."""

    def test_minimize_objective(self):
        """Test minimize objective."""
        obj = Objective(name="eval_loss", direction="minimize", weight=1.0)
        assert obj.name == "eval_loss"
        assert obj.direction == "minimize"
        assert obj.weight == 1.0

    def test_maximize_objective(self):
        """Test maximize objective."""
        obj = Objective(name="accuracy", direction="maximize", weight=0.7)
        assert obj.direction == "maximize"
        assert obj.weight == 0.7

    def test_default_weight(self):
        """Test default weight is 1.0."""
        obj = Objective(name="loss", direction="minimize")
        assert obj.weight == 1.0


class TestOptimizerAlgorithm:
    """Tests for OptimizerAlgorithm enum."""

    def test_tpe_algorithm(self):
        """Test TPE algorithm value."""
        assert OptimizerAlgorithm.TPE.value == "tpe"

    def test_asha_algorithm(self):
        """Test ASHA algorithm value."""
        assert OptimizerAlgorithm.ASHA.value == "asha"

    def test_all_algorithms(self):
        """Test all algorithm values exist."""
        assert len(OptimizerAlgorithm) == 7
        expected = {"tpe", "asha", "hyperband", "pbt", "bohb", "grid", "random"}
        actual = {a.value for a in OptimizerAlgorithm}
        assert actual == expected


class TestHPOConfig:
    """Tests for HPOConfig dataclass."""

    def test_minimal_config(self):
        """Test minimal HPO config with defaults."""
        config = HPOConfig(study_name="test-study")
        
        assert config.study_name == "test-study"
        assert config.n_trials == 50
        assert config.parallel_jobs == 4
        assert config.algorithm == OptimizerAlgorithm.TPE
        assert config.sampler == SamplerType.OPTUNA
        assert config.seed == 42

    def test_config_with_search_space(self):
        """Test config with parameter bounds."""
        bounds = [
            ParameterBound(name="lr", type="float", min=1e-5, max=1e-2, log=True),
            ParameterBound(name="batch", type="categorical", choices=[8, 16, 32]),
        ]
        
        config = HPOConfig(
            study_name="test",
            search_space=bounds,
            n_trials=100,
            parallel_jobs=8,
        )
        
        assert len(config.search_space) == 2
        assert config.n_trials == 100
        assert config.parallel_jobs == 8

    def test_primary_objective(self):
        """Test primary_objective property."""
        config = HPOConfig(study_name="test")
        assert config.primary_objective.name == "eval_loss"
        assert config.primary_objective.direction == "minimize"

    def test_multi_objective(self):
        """Test multi-objective configuration."""
        objectives = [
            Objective("accuracy", "maximize", 0.7),
            Objective("latency", "minimize", 0.3),
        ]
        
        config = HPOConfig(study_name="test", objectives=objectives)
        
        assert config.is_multi_objective is True
        assert config.primary_objective.name == "accuracy"

    def test_ray_resources(self):
        """Test Ray resource computation."""
        config = HPOConfig(
            study_name="test",
            ray_gpu_per_trial=0.5,
            ray_cpu_per_trial=8,
        )
        
        resources = config.get_ray_resources()
        assert resources["gpu"] == 0.5
        assert resources["cpu"] == 8

    def test_storage_defaults(self):
        """Test storage backend defaults."""
        config = HPOConfig(study_name="test")
        assert config.storage_backend == "sqlite:///ugro_hpo.db"
        assert config.tracking_uri is None
        assert config.wandb_project is None
