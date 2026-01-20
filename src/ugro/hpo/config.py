"""HPO Configuration Classes.

Defines dataclasses and enums for hyperparameter optimization configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, List, Literal, Optional


class OptimizerAlgorithm(str, Enum):
    """Supported optimization algorithms."""

    TPE = "tpe"  # Multivariate Tree-structured Parzen Estimator
    ASHA = "asha"  # Asynchronous Successive Halving Algorithm
    HYPERBAND = "hyperband"  # HyperBand with scheduler
    PBT = "pbt"  # Population Based Training
    BOHB = "bohb"  # Bayesian Optimization and HyperBand
    GRID = "grid"  # Exhaustive grid search
    RANDOM = "random"  # Random search baseline


class SamplerType(str, Enum):
    """HPO sampler backend type."""

    OPTUNA = "optuna"
    RAY = "ray"
    CUSTOM = "custom"


@dataclass
class ParameterBound:
    """Single hyperparameter specification.

    Attributes:
        name: Parameter name (used as key in config dict)
        type: Parameter type ("int", "float", "categorical")
        min: Minimum value (for int/float types)
        max: Maximum value (for int/float types)
        choices: Valid choices (for categorical type)
        log: Use log scale sampling (for numeric types)
        step: Step size for discrete sampling
        default: Default value if not sampled
    """

    name: str
    type: Literal["int", "float", "categorical"]
    min: Optional[float] = None
    max: Optional[float] = None
    choices: Optional[List[Any]] = None
    log: bool = False
    step: Optional[float] = None
    default: Optional[Any] = None

    def __post_init__(self) -> None:
        """Validate parameter specifications."""
        if self.type in ("int", "float"):
            if self.min is None or self.max is None:
                raise ValueError(
                    f"Parameter '{self.name}' of type '{self.type}' requires min and max"
                )
            if self.min > self.max:
                raise ValueError(
                    f"Parameter '{self.name}': min ({self.min}) > max ({self.max})"
                )
        elif self.type == "categorical":
            if not self.choices:
                raise ValueError(
                    f"Parameter '{self.name}' of type 'categorical' requires choices"
                )


@dataclass
class Objective:
    """Optimization objective specification.

    Attributes:
        name: Metric name to optimize (e.g., "eval_loss", "accuracy")
        direction: Optimization direction ("maximize" or "minimize")
        weight: Weight for multi-objective scalarization (default 1.0)
    """

    name: str
    direction: Literal["maximize", "minimize"]
    weight: float = 1.0


@dataclass
class HPOConfig:
    """Complete HPO configuration.

    Attributes:
        study_name: Unique name for the HPO study
        search_space: List of parameter bounds to optimize
        objectives: List of optimization objectives (default: minimize loss)
        algorithm: Optimization algorithm to use
        sampler: Sampler backend (optuna, ray, custom)
        n_trials: Total number of trials to run
        parallel_jobs: Number of concurrent trials

        scheduler_type: Early stopping scheduler (asha, hyperband, pbt)
        grace_period: Minimum iterations before pruning
        reduction_factor: Halving factor for ASHA/HyperBand

        ray_address: Ray cluster address (auto, localhost, or remote)
        ray_gpu_per_trial: GPU fraction per trial (0.5 = 2 trials per GPU)
        ray_cpu_per_trial: CPU cores per trial
        ray_timeout: Timeout per trial in seconds

        storage_backend: Optuna storage URI (sqlite:///..., postgresql://...)
        tracking_uri: MLflow tracking server URI
        wandb_project: Weights & Biases project name
        mlflow_experiment: MLflow experiment name

        sample_size: Startup trials before TPE (exploration)
        seed: Random seed for reproducibility

        export_best: Path to export best hyperparameters (YAML)
        save_trials_csv: Path to export all trials (CSV)
    """

    study_name: str
    search_space: List[ParameterBound] = field(default_factory=list)
    objectives: List[Objective] = field(
        default_factory=lambda: [Objective("eval_loss", "minimize")]
    )

    # Algorithm settings
    algorithm: OptimizerAlgorithm = OptimizerAlgorithm.TPE
    sampler: SamplerType = SamplerType.OPTUNA
    n_trials: int = 50
    parallel_jobs: int = 4

    # Scheduler settings (for ASHA/HyperBand)
    scheduler_type: Optional[Literal["asha", "hyperband", "pbt"]] = None
    grace_period: int = 1
    reduction_factor: float = 2.0

    # Ray settings
    ray_address: Optional[str] = None
    ray_gpu_per_trial: float = 1.0
    ray_cpu_per_trial: int = 4
    ray_timeout: int = 3600

    # Storage & Tracking
    storage_backend: str = "sqlite:///ugro_hpo.db"
    tracking_uri: Optional[str] = None
    wandb_project: Optional[str] = None
    mlflow_experiment: str = "hpo"

    # Sampling strategy
    sample_size: int = 10
    seed: int = 42

    # Export
    export_best: Optional[str] = None
    save_trials_csv: Optional[str] = None

    @property
    def primary_objective(self) -> Objective:
        """Get the primary (first) objective."""
        return self.objectives[0] if self.objectives else Objective("loss", "minimize")

    @property
    def is_multi_objective(self) -> bool:
        """Check if this is a multi-objective optimization."""
        return len(self.objectives) > 1

    def get_ray_resources(self) -> dict[str, float]:
        """Get Ray resource requirements per trial."""
        resources: dict[str, float] = {"cpu": self.ray_cpu_per_trial}
        if self.ray_gpu_per_trial > 0:
            resources["gpu"] = self.ray_gpu_per_trial
        return resources
