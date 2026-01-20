#!/usr/bin/env python3
"""
UGRO Configuration Management

Handles loading and validation of cluster and training configurations.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field, validator
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class WorkerConfig(BaseModel):
    """Configuration for a worker node"""
    name: str
    hostname: str
    ip: str
    user: str
    ssh_port: int = 22
    rank: int
    
    hardware: Dict[str, Any]
    paths: Dict[str, str]


class ClusterConfig(BaseModel):
    """Cluster configuration"""
    name: str
    location: str
    description: str
    
    master: Dict[str, Any]
    communication: Dict[str, Any]
    workers: list[WorkerConfig]
    paths: Dict[str, str]
    training: Dict[str, Any]
    environment: Dict[str, Any]
    logging: Dict[str, Any]


class TrainingConfig(BaseModel):
    """Training configuration defaults"""
    model: Dict[str, Any]
    dataset: Dict[str, Any]
    training: Dict[str, Any]
    optimizer: Dict[str, Any]
    lora: Dict[str, Any]
    quantization: Dict[str, Any]
    logging: Dict[str, Any]


class UGROConfig:
    """Main configuration loader and validator"""
    
    def __init__(self, config_dir: Optional[Path] = None):
        """Initialize configuration loader
        
        Args:
            config_dir: Path to configuration directory. Defaults to ./config
        """
        if config_dir is None:
            # Default to config/ relative to project root
            project_root = Path(__file__).parent.parent
            config_dir = project_root / "config"
        
        self.config_dir = Path(config_dir)
        self._cluster_config: Optional[ClusterConfig] = None
        self._training_config: Optional[TrainingConfig] = None
    
    @property
    def cluster_config(self) -> ClusterConfig:
        """Get cluster configuration, loading if necessary"""
        if self._cluster_config is None:
            self._cluster_config = self._load_cluster_config()
        return self._cluster_config
    
    @property
    def training_config(self) -> TrainingConfig:
        """Get training configuration, loading if necessary"""
        if self._training_config is None:
            self._training_config = self._load_training_config()
        return self._training_config
    
    def _load_cluster_config(self) -> ClusterConfig:
        """Load cluster configuration from YAML file"""
        cluster_file = self.config_dir / "cluster.yaml"
        
        if not cluster_file.exists():
            raise FileNotFoundError(f"Cluster config not found: {cluster_file}")
        
        with open(cluster_file, 'r') as f:
            raw_data = yaml.safe_load(f)
        
        # Handle cluster.yaml structure - merge cluster section with root level fields
        config_data = {}
        
        # Add fields from cluster section
        if 'cluster' in raw_data:
            cluster_fields = raw_data['cluster']
            config_data.update(cluster_fields)
        
        # Add root level fields (workers, paths, training, environment, logging)
        root_fields = ['workers', 'paths', 'training', 'environment', 'logging']
        for field in root_fields:
            if field in raw_data:
                config_data[field] = raw_data[field]
        
        # Expand environment variables
        config_data = self._expand_env_vars(config_data)
        
        return ClusterConfig(**config_data)
    
    def _load_training_config(self) -> TrainingConfig:
        """Load training configuration from YAML file"""
        training_file = self.config_dir / "training_defaults.yaml"
        
        if not training_file.exists():
            raise FileNotFoundError(f"Training config not found: {training_file}")
        
        with open(training_file, 'r') as f:
            config_data = yaml.safe_load(f)
        
        # Expand environment variables
        config_data = self._expand_env_vars(config_data)
        
        return TrainingConfig(**config_data)
    
    def _expand_env_vars(self, data: Any) -> Any:
        """Recursively expand environment variables in configuration data"""
        if isinstance(data, str):
            # Expand ${VAR} and $VAR patterns
            return os.path.expandvars(data)
        elif isinstance(data, dict):
            return {k: self._expand_env_vars(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._expand_env_vars(item) for item in data]
        else:
            return data
    
    def get_worker_by_name(self, name: str) -> Optional[WorkerConfig]:
        """Get worker configuration by name"""
        for worker in self.cluster_config.workers:
            if worker.name == name:
                return worker
        return None
    
    def get_worker_by_rank(self, rank: int) -> Optional[WorkerConfig]:
        """Get worker configuration by rank"""
        for worker in self.cluster_config.workers:
            if worker.rank == rank:
                return worker
        return None
    
    def get_all_workers(self) -> list[WorkerConfig]:
        """Get all worker configurations"""
        return self.cluster_config.workers
    
    def reload(self) -> None:
        """Reload configurations from disk"""
        self._cluster_config = None
        self._training_config = None


def load_config(config_dir: Optional[Path] = None) -> UGROConfig:
    """Load UGRO configuration
    
    Args:
        config_dir: Path to configuration directory
        
    Returns:
        UGROConfig instance with loaded configurations
    """
    return UGROConfig(config_dir)


# Default configuration paths
DEFAULT_CONFIG_DIR = Path(__file__).parent.parent / "config"
DEFAULT_CLUSTER_CONFIG = DEFAULT_CONFIG_DIR / "cluster.yaml"
DEFAULT_TRAINING_CONFIG = DEFAULT_CONFIG_DIR / "training_defaults.yaml"