from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from enum import Enum

class TrackingSystem(Enum):
    """Available tracking systems."""
    MLFLOW = "mlflow"
    WANDB = "wandb"
    BOTH = "both"

class TrackingInterface(ABC):
    """Abstract interface for tracking systems.
    
    This interface provides a common API for different tracking systems
    (MLflow, W&B, etc.) to enable easy switching and testing.
    """
    
    @abstractmethod
    def log_params(self, params: Dict[str, Any]) -> None:
        """Log parameters to the tracking system.
        
        Args:
            params: Dictionary of parameter names and values
        """
        pass
    
    @abstractmethod
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics to the tracking system.
        
        Args:
            metrics: Dictionary of metric names and values
            step: Optional step number for the metrics
        """
        pass
    
    @abstractmethod
    def set_tags(self, tags: Dict[str, str]) -> None:
        """Set tags for the run.
        
        Args:
            tags: Dictionary of tag names and values
        """
        pass
    
    @abstractmethod
    def finish(self) -> None:
        """Finish the tracking run and cleanup resources."""
        pass
