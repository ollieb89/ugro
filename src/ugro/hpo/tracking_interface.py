"""Tracking interface abstraction for HPO experiment tracking.

This module provides a unified interface for different tracking backends
(MLflow, W&B, etc.) to reduce coupling and enable easy switching between
tracking systems.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List
from contextlib import contextmanager


class TrackingInterface(ABC):
    """Abstract interface for experiment tracking systems."""
    
    @abstractmethod
    def log_params(self, params: Dict[str, Any]) -> None:
        """Log hyperparameters."""
        pass
    
    @abstractmethod
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics."""
        pass
    
    @abstractmethod
    def log_tags(self, tags: Dict[str, str]) -> None:
        """Log tags."""
        pass
    
    @abstractmethod
    def set_context(self, key: str, value: Any) -> None:
        """Set context information (e.g., run name, group)."""
        pass
    
    @contextmanager
    @abstractmethod
    def start_run(self, run_name: Optional[str] = None):
        """Context manager for a tracking run."""
        pass
    
    @abstractmethod
    def finish_run(self) -> None:
        """Finish the current run."""
        pass
    
    @abstractmethod
    def is_active(self) -> bool:
        """Check if tracking is active."""
        pass


class NoOpTracker(TrackingInterface):
    """No-op tracker that does nothing. Used when tracking is disabled."""
    
    def log_params(self, params: Dict[str, Any]) -> None:
        pass
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        pass
    
    def log_tags(self, tags: Dict[str, str]) -> None:
        pass
    
    def set_context(self, key: str, value: Any) -> None:
        pass
    
    @contextmanager
    def start_run(self, run_name: Optional[str] = None):
        yield self
        self.finish_run()
    
    def finish_run(self) -> None:
        pass
    
    def is_active(self) -> bool:
        return False


class MultiTracker(TrackingInterface):
    """Composite tracker that forwards to multiple trackers."""
    
    def __init__(self, trackers: List[TrackingInterface]):
        self.trackers = trackers
    
    def log_params(self, params: Dict[str, Any]) -> None:
        for tracker in self.trackers:
            tracker.log_params(params)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        for tracker in self.trackers:
            tracker.log_metrics(metrics, step=step)
    
    def log_tags(self, tags: Dict[str, str]) -> None:
        for tracker in self.trackers:
            tracker.log_tags(tags)
    
    def set_context(self, key: str, value: Any) -> None:
        for tracker in self.trackers:
            tracker.set_context(key, value)
    
    @contextmanager
    def start_run(self, run_name: Optional[str] = None):
        contexts = []
        try:
            for tracker in self.trackers:
                ctx = tracker.start_run(run_name)
                contexts.append(ctx)
                # Enter context
                if hasattr(ctx, '__enter__'):
                    ctx.__enter__()
            yield self
        finally:
            # Exit contexts in reverse order
            for ctx in reversed(contexts):
                if hasattr(ctx, '__exit__'):
                    ctx.__exit__(None, None, None)
            self.finish_run()
    
    def finish_run(self) -> None:
        for tracker in self.trackers:
            tracker.finish_run()
    
    def is_active(self) -> bool:
        return any(tracker.is_active() for tracker in self.trackers)
