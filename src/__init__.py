"""
UGRO: Unified GPU Resource Orchestrator

Personal-scale GPU cluster management and distributed training orchestration.
"""

__version__ = "0.1.0"
__author__ = "Oliver Buitelaar"
__email__ = "buitelaar@gmail.com"

from .config import load_config, expand_paths, get_config_dir
from .agent import UGROAgent, JobStatus

__all__ = [
    "load_config",
    "expand_paths", 
    "get_config_dir",
    "UGROAgent",
    "JobStatus"
]