import wandb
from typing import Dict, Any, Optional
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)

@contextmanager
def WandbContextManager(project: str, config: Optional[Dict[str, Any]] = None, **kwargs):
    """Context manager for W&B runs with guaranteed cleanup.
    
    Args:
        project: W&B project name
        config: Configuration dictionary for the run
        **kwargs: Additional arguments passed to wandb.init()
        
    Yields:
        The W&B run object or None if not initialized
    """
    run = None
    try:
        if project:
            run = wandb.init(project=project, config=config or {}, **kwargs)
        yield run
    except Exception as e:
        logger.error(f"W&B context error: {e}")
        raise
    finally:
        if run is not None or wandb.run is not None:
            try:
                wandb.finish()
            except Exception as e:
                logger.warning(f"Failed to finish W&B run: {e}")
