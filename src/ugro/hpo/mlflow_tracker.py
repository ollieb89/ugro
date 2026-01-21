"""MLflow tracker implementation."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional
from contextlib import contextmanager

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    mlflow = None

from .tracking_interface import TrackingInterface

logger = logging.getLogger(__name__)


class MLflowTracker(TrackingInterface):
    """MLflow implementation of TrackingInterface."""
    
    def __init__(
        self,
        tracking_uri: Optional[str] = None,
        experiment_name: Optional[str] = None,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ):
        """Initialize MLflow tracker.
        
        Args:
            tracking_uri: MLflow tracking server URI
            experiment_name: MLflow experiment name
            run_name: Default run name
            tags: Default tags to apply
        """
        if not MLFLOW_AVAILABLE:
            logger.warning("MLflow not installed. Install with: pip install mlflow")
            self._active = False
            return
        
        self._active = True
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.default_run_name = run_name
        self.default_tags = tags or {}
        
        # Configure MLflow
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        
        if experiment_name:
            mlflow.set_experiment(experiment_name)
    
    def log_params(self, params: Dict[str, Any]) -> None:
        """Log hyperparameters."""
        if not self._active:
            return
        try:
            mlflow.log_params(params)
        except Exception as e:
            logger.warning(f"Failed to log params to MLflow: {e}")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics."""
        if not self._active:
            return
        try:
            mlflow.log_metrics(metrics, step=step)
        except Exception as e:
            logger.warning(f"Failed to log metrics to MLflow: {e}")
    
    def log_tags(self, tags: Dict[str, str]) -> None:
        """Log tags."""
        if not self._active:
            return
        try:
            mlflow.set_tags(tags)
        except Exception as e:
            logger.warning(f"Failed to log tags to MLflow: {e}")
    
    def set_context(self, key: str, value: Any) -> None:
        """Set context information."""
        if not self._active:
            return
        try:
            if key == "run_name":
                mlflow.set_tag("mlflow.runName", value)
            elif key == "tags":
                mlflow.set_tags(value)
            else:
                mlflow.set_tag(key, value)
        except Exception as e:
            logger.warning(f"Failed to set context in MLflow: {e}")
    
    @contextmanager
    def start_run(self, run_name: Optional[str] = None):
        """Context manager for a tracking run."""
        if not self._active:
            yield self
            return
        
        run_name = run_name or self.default_run_name
        try:
            if mlflow.active_run():
                # Use nested run
                with mlflow.start_run(run_name=run_name, nested=True) as run:
                    logger.debug(f"Started MLflow nested run: {run.info.run_id}")
                    yield self
            else:
                # Start new run
                with mlflow.start_run(run_name=run_name) as run:
                    logger.debug(f"Started MLflow run: {run.info.run_id}")
                    # Apply default tags
                    if self.default_tags:
                        mlflow.set_tags(self.default_tags)
                    yield self
        except Exception as e:
            logger.error(f"Failed to start MLflow run: {e}")
            yield self
    
    def finish_run(self) -> None:
        """Finish the current run."""
        if not self._active:
            return
        try:
            if mlflow.active_run():
                mlflow.end_run()
                logger.debug("Finished MLflow run")
        except Exception as e:
            logger.warning(f"Failed to finish MLflow run: {e}")
    
    def is_active(self) -> bool:
        """Check if tracking is active."""
        return self._active and MLFLOW_AVAILABLE
