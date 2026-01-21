"""UGRO HPO Optimizer.

Main orchestrator for distributed hyperparameter optimization using
Ray Tune with Optuna's multivariate TPE sampler.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

import yaml

from ugro.hpo.config import HPOConfig, OptimizerAlgorithm
from ugro.hpo.schedulers import create_scheduler
from ugro.hpo.search_space import build_ray_search_space

if TYPE_CHECKING:
    from ray.tune import ResultGrid

logger = logging.getLogger(__name__)


class UGROOptimizer:
    """Main HPO orchestrator with Ray Tune + Optuna.

    Provides distributed hyperparameter optimization with:
    - Multivariate TPE sampling (models parameter correlations)
    - ASHA/HyperBand/PBT early stopping
    - MLflow + W&B experiment tracking
    - Fault-tolerant checkpointing

    Example:
        config = HPOConfig(
            study_name="llama-lora",
            search_space=bounds,
            n_trials=100,
            parallel_jobs=4,
        )
        optimizer = UGROOptimizer(config, objective_fn)
        results = optimizer.optimize()
    """

    def __init__(
        self,
        config: HPOConfig,
        objective_fn: Callable[[Dict[str, Any]], Dict[str, float]],
    ):
        """Initialize optimizer.

        Args:
            config: HPO configuration
            objective_fn: Training objective function
        """
        self.config = config
        self.objective_fn = objective_fn

    def _setup_optuna_sampler(self) -> Any:
        """Create multivariate TPE sampler with Optuna.

        Returns:
            Configured TPESampler instance
        """
        try:
            from optuna.samplers import TPESampler
        except ImportError as e:
            raise ImportError(
                "Optuna is required. Install with: pip install optuna"
            ) from e

        logger.info(
            f"Creating multivariate TPE sampler with "
            f"n_startup={self.config.sample_size}"
        )

        return TPESampler(
            multivariate=True,  # Key: models parameter correlations
            group=True,  # Group conditionally dependent params
            n_startup_trials=self.config.sample_size,
            seed=self.config.seed,
            consider_prior=True,
            prior_weight=1.0,
        )

    def _create_optuna_search(self) -> Any:
        """Create Ray Tune search algorithm with Optuna backend.

        Returns:
            Configured OptunaSearch instance
        """
        try:
            from optuna.storages import RDBStorage
            from ray.tune.search.optuna import OptunaSearch
        except ImportError as e:
            raise ImportError(
                "Ray Tune and Optuna are required. Install with: "
                "pip install 'ray[tune]' optuna"
            ) from e

        # Setup persistent storage for study
        storage = RDBStorage(self.config.storage_backend)

        # Get optimization mode
        primary = self.config.primary_objective
        mode = "max" if primary.direction == "maximize" else "min"

        # Build Ray search space from parameter bounds
        search_space = build_ray_search_space(self.config.search_space)

        logger.info(
            f"Creating OptunaSearch: metric={primary.name}, mode={mode}, "
            f"storage={self.config.storage_backend}"
        )
        
        optuna_search = OptunaSearch(
            sampler=self._setup_optuna_sampler(),
            metric=primary.name,
            mode=mode,
            space=search_space,
            # Note: Storage is set via sampler's study, not directly
        )
        return optuna_search

    def _setup_tracking(self) -> None:
        """Configure experiment tracking (MLflow + W&B)."""
        # MLflow setup
        if self.config.tracking_uri:
            try:
                import mlflow

                mlflow.set_tracking_uri(self.config.tracking_uri)
                mlflow.set_experiment(self.config.mlflow_experiment)
                logger.info(
                    f"MLflow tracking: {self.config.tracking_uri}, "
                    f"experiment={self.config.mlflow_experiment}"
                )
            except ImportError:
                logger.warning("MLflow not installed, skipping tracking setup")

        # Weights & Biases setup
        if self.config.wandb_project:
            os.environ["WANDB_PROJECT"] = self.config.wandb_project
            logger.info(f"W&B project: {self.config.wandb_project}")

    def optimize(self) -> Dict[str, Any]:
        """Execute distributed HPO.

        Returns:
            Dictionary with best_config, best_metrics, and study_id
        """
        try:
            import ray
            from ray import tune
            from ray.tune import CLIReporter
        except ImportError as e:
            raise ImportError(
                "Ray Tune is required. Install with: pip install 'ray[tune]'"
            ) from e

        logger.info(f"Starting HPO study: {self.config.study_name}")

        # Initialize Ray
        if not ray.is_initialized():
            ray.init(address=self.config.ray_address)
            logger.info(f"Ray initialized: {ray.cluster_resources()}")

        # Setup tracking
        self._setup_tracking()

        # Create scheduler for early stopping
        primary = self.config.primary_objective
        scheduler = create_scheduler(
            self.config.scheduler_type,
            metric=primary.name,
            mode="min" if primary.direction == "minimize" else "max",
            grace_period=self.config.grace_period,
            reduction_factor=self.config.reduction_factor,
        )

        # Create run configuration with optional W&B callback
        storage_path = Path("./ray_results").resolve()
        
        # Setup callbacks
        callbacks = []
        if self.config.wandb_project:
            try:
                from ray.tune.integration.wandb import WandbLoggerCallback
                callbacks.append(
                    WandbLoggerCallback(
                        project=self.config.wandb_project,
                        group=self.config.study_name,
                        api_key=os.getenv("WANDB_API_KEY"),
                        log_config=True,
                    )
                )
                logger.info("W&B logger callback configured for Ray Tune")
            except ImportError:
                logger.warning("Ray Tune W&B integration not available")
        
        run_config = tune.RunConfig(
            name=self.config.study_name,
            storage_path=str(storage_path),
            progress_reporter=CLIReporter(
                metric_columns=[obj.name for obj in self.config.objectives]
            ),
            checkpoint_config=tune.CheckpointConfig(
                num_to_keep=3,
                checkpoint_score_attribute=primary.name,
                checkpoint_score_order="min" if primary.direction == "minimize" else "max",
            ),
            callbacks=callbacks,
        )

        # Create tune config with Optuna search
        tune_config = tune.TuneConfig(
            search_alg=self._create_optuna_search(),
            num_samples=self.config.n_trials,
            max_concurrent_trials=self.config.parallel_jobs,
        )

        # Wrap objective in trainable function for Ray Tune
        def trainable(config):
            metrics = self.objective_fn(config)
            tune.report(metrics)
        
        # Create tuner with resource specifications
        tuner = tune.Tuner(
            tune.with_resources(trainable, resources=self.config.get_ray_resources()),
            tune_config=tune_config,
            run_config=run_config,
        )

        # Execute optimization
        logger.info(
            f"Starting {self.config.n_trials} trials with "
            f"{self.config.parallel_jobs} concurrent workers"
        )
        results = tuner.fit()

        # Extract best trial
        best_result = results.get_best_result(
            metric=primary.name,
            mode="min" if primary.direction == "minimize" else "max",
        )

        best_config = best_result.config
        best_metrics = best_result.metrics

        logger.info(f"Best config: {best_config}")
        logger.info(f"Best metrics: {best_metrics}")

        # Export results
        if self.config.export_best:
            self._export_best_config(best_config)

        if self.config.save_trials_csv:
            self._export_trials_csv(results)

        return {
            "best_config": best_config,
            "best_metrics": best_metrics,
            "study_id": self.config.study_name,
            "n_trials": len(results),
        }

    def _export_best_config(self, config: Dict[str, Any]) -> None:
        """Export best hyperparameters to YAML.

        Args:
            config: Best hyperparameter configuration
        """
        output_path = Path(self.config.export_best)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Best config exported to {output_path}")

    def _export_trials_csv(self, results: "ResultGrid") -> None:
        """Export all trials to CSV for analysis.

        Args:
            results: Ray Tune ResultGrid
        """
        try:
            import pandas as pd
        except ImportError:
            logger.warning("pandas not installed, skipping CSV export")
            return

        trials_data = []
        for result in results:
            trial_dict = {
                "trial_id": result.path.name,
                **result.config,
                **result.metrics,
            }
            trials_data.append(trial_dict)

        df = pd.DataFrame(trials_data)
        output_path = Path(self.config.save_trials_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

        logger.info(f"All trials exported to {output_path}")


def run_hpo(
    study_name: str,
    search_space_path: str,
    objective_fn: Callable[[Dict[str, Any]], Dict[str, float]],
    n_trials: int = 50,
    parallel_jobs: int = 4,
    algorithm: str = "tpe",
    **kwargs: Any,
) -> Dict[str, Any]:
    """Convenience function to run HPO with minimal configuration.

    Args:
        study_name: Unique study name
        search_space_path: Path to search space YAML
        objective_fn: Training objective function
        n_trials: Number of trials
        parallel_jobs: Concurrent trials
        algorithm: HPO algorithm
        **kwargs: Additional HPOConfig arguments

    Returns:
        Optimization results
    """
    from ugro.hpo.search_space import (
        load_search_space_yaml,
        parse_objectives,
        parse_parameter_bounds,
    )

    # Load search space
    config_dict = load_search_space_yaml(search_space_path)
    bounds = parse_parameter_bounds(config_dict)
    objectives = parse_objectives(config_dict)

    # Create HPO config
    hpo_config = HPOConfig(
        study_name=study_name,
        search_space=bounds,
        objectives=objectives,
        algorithm=OptimizerAlgorithm(algorithm),
        n_trials=n_trials,
        parallel_jobs=parallel_jobs,
        **kwargs,
    )

    # Run optimization
    optimizer = UGROOptimizer(hpo_config, objective_fn)
    return optimizer.optimize()
