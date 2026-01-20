"""
UGRO: Unified GPU Resource Orchestrator

Personal-scale GPU cluster management and distributed training orchestration.
"""

__version__ = "0.1.0"
__author__ = "Oliver Buitelaar"
__email__ = "buitelaar@gmail.com"

# Import main components from src package
try:
    from src.config import load_config, expand_paths, get_config_dir
    from src.agent import UGROAgent, JobStatus
    
    __all__ = [
        "__version__",
        "load_config",
        "expand_paths", 
        "get_config_dir",
        "UGROAgent",
        "JobStatus"
    ]
except ImportError:
    # Fallback for development when package is not installed
    __all__ = ["__version__"]