from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Optional, List, Dict, Union
import yaml
from pydantic import BaseModel, Field, field_validator

if TYPE_CHECKING:
    from typing import Any


def get_config_dir() -> Path:
    """Get configuration directory."""
    return Path(__file__).parent.parent.parent / "config"


class PathsConfig(BaseModel):
    """Path configuration for workers."""
    home: str
    project: str
    scripts: Optional[str] = None


class HardwareConfig(BaseModel):
    """Hardware configuration for a node."""
    gpu_model: str
    vram_gb: int


class NodeConfig(BaseModel):
    """Configuration for a single cluster node."""
    name: str
    ip: str  # Kept as str to allow hostnames, could use IPvAnyAddress
    user: str
    role: str = "worker"
    ssh_port: int = 22
    hardware: Optional[HardwareConfig] = None
    paths: Optional[PathsConfig] = None


class MasterConfig(BaseModel):
    """Configuration for the master node."""
    hostname: Optional[str] = None
    ip: str
    port: int = 22
    user: str = "ob"


class CommConfig(BaseModel):
    """Communication settings."""
    backend: str = "c10d"
    master_port: int = 29500
    timeout_seconds: int = 300


class ClusterConfig(BaseModel):
    """Root cluster configuration."""
    name: str = "UGRO Cluster"
    location: Optional[str] = None
    description: Optional[str] = None
    master: MasterConfig
    communication: Optional[CommConfig] = Field(default_factory=CommConfig)
    nodes: Dict[str, NodeConfig] = Field(default_factory=dict)
    
    @property
    def master_ip(self) -> str:
        return self.master.ip
        
    @property
    def master_port(self) -> int:
        return self.communication.master_port if self.communication else 29500

    # Backward compatibility with list-based 'workers'
    @field_validator("nodes", mode="before")
    @classmethod
    def parse_nodes(cls, v: Any) -> Any:
        # If input is already a dict, return it
        if isinstance(v, dict):
            return v
        return {}

class AppConfig(BaseModel):
    """Application level config wrapper."""
    cluster: ClusterConfig
    
    @classmethod
    def from_yaml(cls, path: Path) -> "AppConfig":
        with open(path) as f:
            raw = yaml.safe_load(f)
            
        # Transform raw yaml to match model structure if needed
        # The existing yaml structure is a bit mixed, let's normalize it
        if "cluster" in raw and isinstance(raw["cluster"], dict):
            cluster_data = raw["cluster"]
            nodes = {}
            
            # Handle 'nodes' dict format
            if "nodes" in raw:
                nodes.update(raw["nodes"])
                
            # Handle 'workers' list format (legacy)
            if "workers" in raw and isinstance(raw["workers"], list):
                for w in raw["workers"]:
                    nodes[w["name"]] = w
            
            cluster_data["nodes"] = nodes
            return cls(cluster=ClusterConfig(**cluster_data))
            
        # Fallback/Direct mapping if structure matches
        return cls(**raw)


def load_config(config_name: str = "cluster.yaml") -> dict[str, Any]:
    """Load YAML configuration (Legacy helper).
    
    Returns raw dict for backward compatibility, but conceptually we move to models.
    """
    config_path = get_config_dir() / config_name
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
        
    return expand_paths(config)


def expand_paths(config: dict[str, Any]) -> dict[str, Any]:
    """Expand ~ and environment variables in paths"""
    # Recursive expansion logic could go here
    if 'paths' in config:
        for key, value in config['paths'].items():
            if isinstance(value, str):
                config['paths'][key] = os.path.expanduser(value)
                config['paths'][key] = os.path.expandvars(config['paths'][key])
    
    return config