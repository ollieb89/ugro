import wandb
from typing import Dict, Any, Optional
from .interface import TrackingInterface
from ..security import validate_wandb_api_key, mask_api_key
import logging

logger = logging.getLogger(__name__)

class WandbTracker(TrackingInterface):
    """W&B implementation of tracking interface.
    
    Provides a secure wrapper around W&B tracking with validation
    and proper resource management.
    """
    
    def __init__(self, project: str, config: Optional[Dict[str, Any]] = None, **kwargs):
        """Initialize W&B tracker.
        
        Args:
            project: W&B project name
            config: Optional configuration dictionary
            **kwargs: Additional arguments for wandb.init()
            
        Raises:
            ValueError: If API key validation fails
        """
        # Validate inputs
        if not validate_wandb_api_key(wandb.api.api_key):
            raise ValueError("Invalid W&B API key")
        
        self.project = project
        self.config = config or {}
        self.kwargs = kwargs
        self._run = None
        self._initialized = False
    
    def _ensure_run(self):
        """Ensure W&B run is initialized."""
        if not self._initialized and wandb.run is None:
            self._run = wandb.init(
                project=self.project,
                config=self.config,
                **self.kwargs
            )
            self._initialized = True
    
    def log_params(self, params: Dict[str, Any]) -> None:
        """Log parameters to W&B."""
        self._ensure_run()
        wandb.config.update(params)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics to W&B."""
        self._ensure_run()
        wandb.log(metrics, step=step)
    
    def set_tags(self, tags: Dict[str, str]) -> None:
        """Set tags in W&B."""
        self._ensure_run()
        if wandb.run:
            wandb.run.tags = tags
    
    def finish(self) -> None:
        """Finish W&B run."""
        if wandb.run is not None:
            wandb.finish()
        self._run = None
        self._initialized = False
