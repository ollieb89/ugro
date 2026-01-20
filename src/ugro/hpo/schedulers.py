"""HPO Scheduler Factories.

Pre-configured schedulers for early stopping and resource-efficient HPO.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, Literal, Optional, Tuple

if TYPE_CHECKING:
    from ray.tune.schedulers import TrialScheduler
    from ray.tune.stopper import Stopper

logger = logging.getLogger(__name__)
from ray.tune.stopper import CombinedStopper


class AdaptiveASHAScheduler:
    """Factory for ASHA (Asynchronous Successive Halving Algorithm) schedulers.

    ASHA is particularly effective for expensive training runs like LoRA
    fine-tuning, as it aggressively prunes underperforming trials early.

    How ASHA works:
    1. Run all trials for `grace_period` iterations
    2. Stop bottom 50% (reduction_factor=2)
    3. Continue survivors for 2x iterations
    4. Repeat until completion
    """

    @staticmethod
    def create_for_lora_tuning(
        metric: str = "eval_loss",
        mode: Literal["min", "max"] = "min",
        max_epochs: int = 3,
        grace_period: int = 1,
        reduction_factor: float = 2,
    ) -> "TrialScheduler":
        """Create ASHA scheduler optimized for LoRA fine-tuning.

        Args:
            metric: Metric to optimize
            mode: "min" to minimize, "max" to maximize
            max_epochs: Maximum training iterations
            grace_period: Minimum iterations before pruning
            reduction_factor: Halving factor (2 = keep top 50%)

        Returns:
            Configured ASHAScheduler instance
        """
        try:
            from ray.tune.schedulers import ASHAScheduler
        except ImportError as e:
            raise ImportError(
                "Ray Tune is required. Install with: pip install 'ray[tune]'"
            ) from e

        logger.info(
            f"Creating ASHA scheduler: metric={metric}, mode={mode}, "
            f"max_t={max_epochs}, grace={grace_period}"
        )

        return ASHAScheduler(
            time_attr="training_iteration",
            metric=metric,
            mode=mode,
            max_t=max_epochs,
            grace_period=grace_period,
            reduction_factor=reduction_factor,
        )


class HyperBandSchedulerFactory:
    """Factory for HyperBand schedulers."""

    @staticmethod
    def create(
        metric: str = "eval_loss",
        mode: Literal["min", "max"] = "min",
        max_t: int = 100,
    ) -> "TrialScheduler":
        """Create HyperBand scheduler.

        Args:
            metric: Metric to optimize
            mode: "min" to minimize, "max" to maximize
            max_t: Maximum training iterations

        Returns:
            Configured HyperBandScheduler instance
        """
        try:
            from ray.tune.schedulers import HyperBandScheduler
        except ImportError as e:
            raise ImportError(
                "Ray Tune is required. Install with: pip install 'ray[tune]'"
            ) from e

        logger.info(f"Creating HyperBand scheduler: metric={metric}, mode={mode}")

        return HyperBandScheduler(
            time_attr="training_iteration",
            metric=metric,
            mode=mode,
            max_t=max_t,
        )


class PBTSchedulerFactory:
    """Factory for Population Based Training schedulers."""

    @staticmethod
    def create(
        metric: str = "eval_loss",
        mode: Literal["min", "max"] = "min",
        perturbation_interval: int = 10,
        hyperparam_mutations: Optional[Dict[str, Any]] = None,
    ) -> "TrialScheduler":
        """Create PBT scheduler for transfer learning.

        Args:
            metric: Metric to optimize
            mode: "min" to minimize, "max" to maximize
            perturbation_interval: Steps between perturbations
            hyperparam_mutations: Parameter mutation specs

        Returns:
            Configured PopulationBasedTraining instance
        """
        try:
            from ray.tune.schedulers import PopulationBasedTraining
        except ImportError as e:
            raise ImportError(
                "Ray Tune is required. Install with: pip install 'ray[tune]'"
            ) from e

        import random

        # Default mutations for common hyperparameters
        default_mutations = {
            "learning_rate": lambda: 10 ** (-2 - 2 * random.random()),
            "lora_alpha": [8.0, 16.0, 32.0, 64.0],
        }

        mutations = hyperparam_mutations or default_mutations

        logger.info(
            f"Creating PBT scheduler: metric={metric}, mode={mode}, "
            f"interval={perturbation_interval}"
        )

        return PopulationBasedTraining(
            time_attr="training_iteration",
            metric=metric,
            mode=mode,
            perturbation_interval=perturbation_interval,
            hyperparam_mutations=mutations,
        )


class MaxTokensPerTrialStopper:
    """Stop trial if it exceeds token budget.

    Useful for controlling costs in LLM fine-tuning where token
    consumption directly impacts compute costs.
    """

    def __init__(self, max_tokens: int = 10_000_000):
        """Initialize stopper with token limit.

        Args:
            max_tokens: Maximum tokens per trial (default 10M)
        """
        self.max_tokens = max_tokens
        self.trial_tokens: Dict[str, int] = {}

    def __call__(self, trial_id: str, result: Dict[str, Any]) -> bool:
        """Check if trial should be stopped.

        Args:
            trial_id: Trial identifier
            result: Trial result dictionary

        Returns:
            True if trial should stop, False to continue
        """
        tokens = result.get("tokens_used", 0)
        self.trial_tokens[trial_id] = tokens

        should_stop = tokens >= self.max_tokens
        if should_stop:
            logger.warning(
                f"Trial {trial_id} exceeded token limit: "
                f"{tokens:,} >= {self.max_tokens:,}"
            )
        return should_stop

    def stop_all(self) -> bool:
        """Check if all trials should stop.

        Returns:
            Always False (individual trial stopping only)
        """
        return False


class TimeoutStopper:
    """Stop trial after maximum time elapsed."""

    def __init__(self, timeout_seconds: int = 3600):
        """Initialize stopper with timeout.

        Args:
            timeout_seconds: Maximum seconds per trial
        """
        self.timeout_seconds = timeout_seconds
        self.trial_start_times: Dict[str, float] = {}

    def __call__(self, trial_id: str, result: Dict[str, Any]) -> bool:
        """Check if trial should be stopped due to timeout.

        Args:
            trial_id: Trial identifier
            result: Trial result dictionary

        Returns:
            True if trial exceeded timeout
        """
        import time

        if trial_id not in self.trial_start_times:
            self.trial_start_times[trial_id] = time.time()

        elapsed = time.time() - self.trial_start_times[trial_id]
        should_stop = elapsed >= self.timeout_seconds

        if should_stop:
            logger.warning(
                f"Trial {trial_id} exceeded time limit: "
                f"{elapsed:.0f}s >= {self.timeout_seconds}s"
            )
        return should_stop

    def stop_all(self) -> bool:
        """Check if all trials should stop.

        Returns:
            Always False (individual trial stopping only)
        """
        return False


class ResourceAwareSchedulerFactory:
    """Factory to create a scheduler with optional resource-aware stop criteria.

    Combines a Ray Tune scheduler with MaxTokensPerTrialStopper and/or TimeoutStopper.
    """

    @staticmethod
    def create(
        scheduler_type: Literal["asha", "hyperband", "pbt"] | None,
        metric: str = "eval_loss",
        mode: Literal["min", "max"] = "min",
        max_tokens: Optional[int] = None,
        timeout_seconds: Optional[int] = None,
        **kwargs: Any,
    ) -> Tuple["TrialScheduler", Optional[Stopper]]:
        """Create scheduler and combined stopper.

        Args:
            scheduler_type: Type of scheduler.
            metric: Metric name.
            mode: Optimization mode.
            max_tokens: Optional token budget per trial.
            timeout_seconds: Optional time budget per trial.
            **kwargs: Scheduler specific args.
        """
        # Base scheduler
        if scheduler_type == "asha":
            scheduler = AdaptiveASHAScheduler.create_for_lora_tuning(
                metric=metric,
                mode=mode,
                grace_period=kwargs.get("grace_period", 1),
                reduction_factor=kwargs.get("reduction_factor", 2),
                max_epochs=kwargs.get("max_epochs", 3),
            )
        elif scheduler_type == "hyperband":
            scheduler = HyperBandSchedulerFactory.create(
                metric=metric,
                mode=mode,
                max_t=kwargs.get("max_t", 100),
            )
        elif scheduler_type == "pbt":
            scheduler = PBTSchedulerFactory.create(
                metric=metric,
                mode=mode,
                perturbation_interval=kwargs.get("perturbation_interval", 10),
                hyperparam_mutations=kwargs.get("hyperparam_mutations"),
            )
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")

        # Stop criteria
        stop_criteria: list[Stopper] = []
        if max_tokens is not None:
            stop_criteria.append(MaxTokensPerTrialStopper(max_tokens))
        if timeout_seconds is not None:
            stop_criteria.append(TimeoutStopper(timeout_seconds))
        stopper: Optional[Stopper] = None
        if stop_criteria:
            if len(stop_criteria) == 1:
                stopper = stop_criteria[0]
            else:
                stopper = CombinedStopper(*stop_criteria)
        return scheduler, stopper

def create_scheduler(
    scheduler_type: Literal["asha", "hyperband", "pbt"] | None,
    metric: str = "eval_loss",
    mode: Literal["min", "max"] = "min",
    max_tokens: Optional[int] = None,
    timeout_seconds: Optional[int] = None,
    **kwargs: Any,
) -> Tuple["TrialScheduler", Optional[Stopper]]:
    """Factory function to create scheduler and optional stopper.

    Maintains backward compatibility; returns (scheduler, stopper).
    """
    if scheduler_type is None:
        return None, None
    return ResourceAwareSchedulerFactory.create(
        scheduler_type,
        metric=metric,
        mode=mode,
        max_tokens=max_tokens,
        timeout_seconds=timeout_seconds,
        **kwargs,
    )
