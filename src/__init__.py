"""UGRO: Unified GPU Resource Orchestrator.

Personal-scale GPU cluster management and distributed training orchestration.

This module re-exports from the ugro package for backward compatibility.
"""

from __future__ import annotations

# Re-export from ugro package
from ugro import (
    Cluster,
    Job,
    JobStatus,
    SSHClient,
    UGROAgent,
    __author__,
    __email__,
    __version__,
    expand_paths,
    get_config_dir,
    load_config,
)

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