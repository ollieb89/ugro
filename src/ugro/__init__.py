"""UGRO: Unified GPU Resource Orchestrator.

Personal-scale GPU cluster management and distributed training orchestration.
"""

from __future__ import annotations

# Package metadata
__version__ = "0.1.0"
__author__ = "Oliver Buitelaar"
__email__ = "buitelaar@gmail.com"

# Import main components using relative imports
from .agent import UGROAgent
from .cluster import Cluster
from .config import expand_paths, get_config_dir, load_config
from .job import Job, JobStatus
from .ssh_utils import SSHClient

__all__ = [
    # Metadata
    "__version__",
    "__author__",
    "__email__",
    # Configuration
    "load_config",
    "expand_paths",
    "get_config_dir",
    # Core classes
    "UGROAgent",
    "Cluster",
    "Job",
    "JobStatus",
    "SSHClient",
]