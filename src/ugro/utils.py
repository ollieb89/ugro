"""Utility functions for UGRO"""
import yaml
from pathlib import Path
from typing import Dict, Any

def load_cluster_config(config_path: str = None) -> Dict[str, Any]:
    """Load cluster configuration from YAML file"""
    if config_path is None:
        config_path = Path(__file__).parent.parent / "config" / "cluster.yaml"
    
    with open(config_path) as f:
        return yaml.safe_load(f)
