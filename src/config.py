#!/usr/bin/env python3
"""
UGRO Configuration Management

Handles loading and validation of cluster and training configurations.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any


def get_config_dir() -> Path:
    """Get configuration directory"""
    return Path(__file__).parent.parent / "config"


def load_config(config_name: str = "cluster.yaml") -> Dict[str, Any]:
    """Load YAML configuration"""
    config_path = get_config_dir() / config_name
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    
    with open(config_path) as f:
        return yaml.safe_load(f)


def expand_paths(config: Dict[str, Any]) -> Dict[str, Any]:
    """Expand ~ and environment variables in paths"""
    if 'paths' in config:
        for key, value in config['paths'].items():
            if isinstance(value, str):
                config['paths'][key] = os.path.expanduser(value)
                config['paths'][key] = os.path.expandvars(config['paths'][key])
    
    return config