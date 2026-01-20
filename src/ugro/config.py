"""UGRO Configuration Management.

Handles loading and validation of cluster and training configurations.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from typing import Any


def get_config_dir() -> Path:
    """Get configuration directory"""
    return Path(__file__).parent.parent / "config"


def load_config(config_name: str = "cluster.yaml") -> dict[str, Any]:
    """Load YAML configuration"""
    config_path = get_config_dir() / config_name
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    
    with open(config_path) as f:
        return yaml.safe_load(f)


def expand_paths(config: dict[str, Any]) -> dict[str, Any]:
    """Expand ~ and environment variables in paths"""
    if 'paths' in config:
        for key, value in config['paths'].items():
            if isinstance(value, str):
                config['paths'][key] = os.path.expanduser(value)
                config['paths'][key] = os.path.expandvars(config['paths'][key])
    
    return config