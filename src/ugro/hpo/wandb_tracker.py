"""W&B tracker implementation with security features."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional
from contextlib import contextmanager

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

from .tracking_interface import TrackingInterface
from .security import validate_wandb_api_key, mask_api_key, validate_project_name

logger = logging.getLogger(__name__)


class WandbTracker(TrackingInterface):
    """W&B implementation of TrackingInterface with security validation."""
    
    def __init__(
        self,
        project: Optional[str] = None,
        entity: Optional[str] = None,
        group: Optional[str] = None,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        api_key: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize W&B tracker with security validation.
        
        Args:
            project: W&B project name (validated)
            entity: W&B entity/team name
            group: Group name for runs
            run_name: Default run name
            tags: Default tags
            api_key: W&B API key (validated and masked)
            config: Default config dictionary
        """
        if not WANDB_AVAILABLE:
            logger.warning("W&B not installed. Install with: pip install wandb")
            self._active = False
            return
        
        # Validate inputs
        if project and not validate_project_name(project):
            raise ValueError(f"Invalid project name: {project}")
        
        if api_key and not validate_wandb_api_key(api_key):
            raise ValueError(f"Invalid API key format")
        
        self._active = True
        self.project = project
        self.entity = entity
        self.group = group
        self.default_run_name = run_name
        self.default_tags = tags or {}
        self.api_key = api_key
        self.default_config = config or {}
        self._run = None
        
        # Log masked API key for debugging
        if api_key:
            logger.debug(f"Using W&B API key: {mask_api_key(api_key)}")
    
    def log_params(self, params: Dict[str, Any]) -> None:
        """Log hyperparameters."""
        if not self._active or not self._run:
            return
        try:
            wandb.config.update(params, allow_val_change=True)
        except Exception as e:
            logger.warning(f"Failed to log params to W&B: {e}")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics."""
        if not self._active or not self._run:
            return
        try:
            wandb.log(metrics, step=step)
        except Exception as e:
            logger.warning(f"Failed to log metrics to W&B: {e}")
    
    def log_tags(self, tags: Dict[str, str]) -> None:
        """Log tags."""
        if not self._active or not self._run:
            return
        try:
            wandb.tags.update(tags)
            # Update run tags
            self._run.tags = {**self._run.tags, **tags}
        except Exception as e:
            logger.warning(f"Failed to log tags to W&B: {e}")
    
    def set_context(self, key: str, value: Any) -> None:
        """Set context information."""
        if not self._active or not self._run:
            return
        try:
            if key == "project":
                # Can't change project after init
                logger.warning("Cannot change W&B project after initialization")
            elif key == "group":
                wandb.config.group = value
            elif key == "tags":
                self.log_tags(value)
            else:
                wandb.config[key] = value
        except Exception as e:
            logger.warning(f"Failed to set context in W&B: {e}")
    
    @contextmanager
    def start_run(self, run_name: Optional[str] = None):
        """Context manager for a tracking run."""
        if not self._active:
            yield self
            return
        
        # Set API key if provided
        if self.api_key:
            os.environ["WANDB_API_KEY"] = self.api_key
        
        run_name = run_name or self.default_run_name
        
        try:
            # Initialize W&B run
            self._run = wandb.init(
                project=self.project,
                entity=self.entity,
                group=self.group,
                name=run_name,
                tags=self.default_tags,
                config=self.default_config,
                reinit=True  # Allow multiple runs
            )
            logger.debug(f"Started W&B run: {self._run.id}")
            yield self
        except Exception as e:
            logger.error(f"Failed to start W&B run: {e}")
            self._run = None
            yield self
        finally:
            self.finish_run()
    
    def finish_run(self) -> None:
        """Finish the current run."""
        if not self._active or not self._run:
            return
        try:
            wandb.finish()
            logger.debug("Finished W&B run")
            self._run = None
        except Exception as e:
            logger.warning(f"Failed to finish W&B run: {e}")
            self._run = None
    
    def is_active(self) -> bool:
        """Check if tracking is active."""
        return self._active and WANDB_AVAILABLE and self._run is not None
