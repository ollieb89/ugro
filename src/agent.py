#!/usr/bin/env python3
"""
UGRO Agent: Core GPU Cluster Orchestrator

Handles distributed training coordination, health monitoring, and job management
across a personal-scale GPU cluster.
"""

import os
import sys
import json
import time
import uuid
import signal
import logging
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

from src.config import UGROConfig, load_config


class JobStatus:
    """Job status constants"""
    PENDING = "pending"
    RUNNING = "running" 
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class UGROAgent:
    """Main UGRO agent for cluster orchestration"""
    
    def __init__(self, config_dir: Optional[Path] = None):
        """Initialize UGRO agent
        
        Args:
            config_dir: Path to configuration directory
        """
        self.config = load_config(config_dir)
        self.logger = self._setup_logging()
        
        # Job tracking
        self.jobs_dir = Path(self.config.cluster_config.paths["experiments"])
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize job registry
        self.job_registry_file = self.jobs_dir / "job_registry.json"
        self.job_registry = self._load_job_registry()
        
        self.logger.info("UGRO Agent initialized")
        self.logger.info(f"Cluster: {self.config.cluster_config.name}")
        self.logger.info(f"Workers: {len(self.config.cluster_config.workers)} nodes")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        log_config = self.config.cluster_config.logging
        log_level = getattr(logging, log_config.get("level", "INFO"))
        
        # Create logger
        logger = logging.getLogger("ugro")
        logger.setLevel(log_level)
        
        # Create formatter
        formatter = logging.Formatter(
            log_config.get("format", "[%(asctime)s] %(name)s - %(levelname)s - %(message)s")
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler
        log_file = Path(log_config.get("file", "logs/agent.log"))
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        return logger
    
    def _load_job_registry(self) -> Dict[str, Any]:
        """Load job registry from disk"""
        if self.job_registry_file.exists():
            try:
                with open(self.job_registry_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load job registry: {e}")
        
        return {"jobs": {}, "last_updated": str(datetime.now())}
    
    def _save_job_registry(self) -> None:
        """Save job registry to disk"""
        self.job_registry["last_updated"] = str(datetime.now())
        
        try:
            with open(self.job_registry_file, 'w') as f:
                json.dump(self.job_registry, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save job registry: {e}")
    
    def _create_job_dir(self, job_name: str) -> Path:
        """Create job directory"""
        job_dir = self.jobs_dir / job_name
        job_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (job_dir / "logs").mkdir(exist_ok=True)
        (job_dir / "checkpoints").mkdir(exist_ok=True)
        (job_dir / "tensorboard").mkdir(exist_ok=True)
        
        return job_dir
    
    def _simulate_gpu_check(self, worker_name: str) -> Tuple[bool, str]:
        """Simulate GPU health check (placeholder for real implementation)"""
        # In real implementation, this would SSH to worker and check GPU status
        import random
        
        # Simulate 90% success rate for demo
        if random.random() < 0.9:
            gpu_model = "RTX 4070" if "1" in worker_name else "RTX 3070 Ti"
            return True, f"GPU ({gpu_model}) healthy, memory available"
        else:
            return False, "GPU not responding or CUDA error"
    
    def _simulate_training_launch(self, job_params: Dict[str, Any]) -> bool:
        """Simulate training launch (placeholder for real implementation)"""
        job_name = job_params["job_name"]
        
        # Create job directory
        job_dir = self._create_job_dir(job_name)
        
        # Register job
        job_id = str(uuid.uuid4())
        self.job_registry["jobs"][job_name] = {
            "id": job_id,
            "status": JobStatus.RUNNING,
            "created_at": str(datetime.now()),
            "started_at": str(datetime.now()),
            "parameters": job_params,
            "directory": str(job_dir),
            "workers": [w.name for w in self.config.cluster_config.workers]
        }
        self._save_job_registry()
        
        # Simulate training process
        self.logger.info(f"Launching training job '{job_name}' on {len(self.config.cluster_config.workers)} workers")
        
        # In real implementation, this would:
        # 1. SSH to each worker
        # 2. Start distributed training processes
        # 3. Monitor progress
        
        return True
    
    def check_cluster_health(self) -> Dict[str, Dict[str, Any]]:
        """Check health status of all cluster nodes
        
        Returns:
            Dict mapping node names to health status
        """
        health_status = {}
        
        # Check master node
        health_status["master"] = {
            "healthy": True,
            "message": "Master node healthy",
            "timestamp": str(datetime.now())
        }
        
        # Check worker nodes
        for worker in self.config.cluster_config.workers:
            is_healthy, message = self._simulate_gpu_check(worker.name)
            
            health_status[worker.name] = {
                "healthy": is_healthy,
                "message": message,
                "gpu_model": worker.hardware["gpu_model"],
                "vram_gb": worker.hardware["vram_gb"],
                "timestamp": str(datetime.now())
            }
        
        return health_status
    
    def launch_training(
        self,
        job_name: str,
        model: str,
        dataset: str,
        epochs: int,
        learning_rate: float,
        verbose: bool = False
    ) -> bool:
        """Launch distributed training job
        
        Args:
            job_name: Unique job identifier
            model: Model name/path
            dataset: Dataset name
            epochs: Number of training epochs
            learning_rate: Learning rate
            verbose: Enable verbose logging
            
        Returns:
            True if launch successful, False otherwise
        """
        self.logger.info(f"Launching training job: {job_name}")
        
        # Validate job name uniqueness
        if job_name in self.job_registry["jobs"]:
            self.logger.error(f"Job '{job_name}' already exists")
            return False
        
        # Prepare job parameters
        job_params = {
            "job_name": job_name,
            "model": model,
            "dataset": dataset,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "verbose": verbose,
            "num_workers": len(self.config.cluster_config.workers)
        }
        
        # Launch training
        success = self._simulate_training_launch(job_params)
        
        if success:
            self.logger.info(f"Training job '{job_name}' launched successfully")
            
            # Create a mock log file for demonstration
            job_dir = self.jobs_dir / job_name
            log_file = job_dir / "logs" / "training.log"
            
            with open(log_file, 'w') as f:
                f.write(f"# Training Log for Job: {job_name}\n")
                f.write(f"# Started: {datetime.now()}\n")
                f.write(f"# Model: {model}\n")
                f.write(f"# Dataset: {dataset}\n")
                f.write(f"# Epochs: {epochs}\n")
                f.write(f"# Learning Rate: {learning_rate}\n")
                f.write(f"# Workers: {len(self.config.cluster_config.workers)}\n\n")
                f.write("Training started...\n")
                f.write("Epoch 1/1 - Step 1/100 - Loss: 2.5\n")
                f.write("Epoch 1/1 - Step 2/100 - Loss: 2.3\n")
                f.write("Epoch 1/1 - Step 3/100 - Loss: 2.1\n")
        else:
            self.logger.error(f"Failed to launch training job '{job_name}'")
        
        return success
    
    def display_logs(self, job_name: str, rank: Optional[int] = None) -> None:
        """Display training logs for a job
        
        Args:
            job_name: Job name
            rank: Specific worker rank to display logs for
        """
        if job_name not in self.job_registry["jobs"]:
            print(f"❌ Job '{job_name}' not found")
            return
        
        job_info = self.job_registry["jobs"][job_name]
        job_dir = Path(job_info["directory"])
        log_file = job_dir / "logs" / "training.log"
        
        if not log_file.exists():
            print(f"❌ No logs found for job '{job_name}'")
            return
        
        print(f"\n📋 Training Logs: {job_name}")
        print("=" * 60)
        print(f"Status: {job_info['status']}")
        print(f"Created: {job_info['created_at']}")
        print(f"Workers: {', '.join(job_info['workers'])}")
        
        if rank is not None:
            print(f"Showing logs for rank {rank}")
        else:
            print("Showing logs for all ranks")
        
        print("-" * 60)
        
        try:
            with open(log_file, 'r') as f:
                print(f.read())
        except Exception as e:
            print(f"❌ Error reading logs: {e}")
    
    def display_results(self, job_name: str) -> None:
        """Display results summary for a job
        
        Args:
            job_name: Job name
        """
        if job_name not in self.job_registry["jobs"]:
            print(f"❌ Job '{job_name}' not found")
            return
        
        job_info = self.job_registry["jobs"][job_name]
        
        print(f"\n📊 Results Summary: {job_name}")
        print("=" * 60)
        print(f"Status: {job_info['status']}")
        print(f"Created: {job_info['created_at']}")
        print(f"Model: {job_info['parameters']['model']}")
        print(f"Dataset: {job_info['parameters']['dataset']}")
        print(f"Epochs: {job_info['parameters']['epochs']}")
        print(f"Learning Rate: {job_info['parameters']['learning_rate']}")
        print(f"Workers: {len(job_info['workers'])}")
        
        # Simulate results (in real implementation, this would read from training output)
        if job_info['status'] == JobStatus.COMPLETED:
            print(f"\n✅ Training completed successfully!")
            print(f"Final Loss: 0.85")
            print(f"Training Time: 2h 15m")
            print(f"GPU Utilization: 87%")
            print(f"Samples Processed: 10,000")
        elif job_info['status'] == JobStatus.RUNNING:
            print(f"\n🔄 Training in progress...")
            print(f"Current Epoch: 1/1")
            print(f"Current Loss: 1.2")
            print(f"Elapsed Time: 45m")
        else:
            print(f"\n❌ Training failed or was cancelled")
        
        job_dir = Path(job_info["directory"])
        print(f"\n📁 Output Directory: {job_dir}")
        print(f"📊 TensorBoard: {job_dir / 'tensorboard'}")
        print(f"💾 Checkpoints: {job_dir / 'checkpoints'}")
    
    def display_status(self) -> None:
        """Display current cluster status"""
        print(f"\n🖥️  Cluster Status: {self.config.cluster_config.name}")
        print("=" * 60)
        print(f"Location: {self.config.cluster_config.location}")
        print(f"Description: {self.config.cluster_config.description}")
        
        # Worker status
        print(f"\n🔧 Workers ({len(self.config.cluster_config.workers)} nodes):")
        for worker in self.config.cluster_config.workers:
            print(f"  • {worker.name}: {worker.hardware['gpu_model']} ({worker.hardware['vram_gb']}GB VRAM)")
        
        # Active jobs
        active_jobs = [name for name, job in self.job_registry["jobs"].items() 
                      if job["status"] in [JobStatus.RUNNING, JobStatus.PENDING]]
        
        print(f"\n🚀 Active Jobs: {len(active_jobs)}")
        for job_name in active_jobs:
            job_info = self.job_registry["jobs"][job_name]
            print(f"  • {job_name}: {job_info['status']} (started {job_info['started_at']})")
        
        # Recent jobs
        recent_jobs = list(self.job_registry["jobs"].keys())[-5:] if self.job_registry["jobs"] else []
        
        print(f"\n📋 Recent Jobs:")
        for job_name in recent_jobs:
            job_info = self.job_registry["jobs"][job_name]
            status_emoji = "✅" if job_info["status"] == JobStatus.COMPLETED else "❌" if job_info["status"] == JobStatus.FAILED else "🔄"
            print(f"  • {job_name} {status_emoji} {job_info['status']}")
        
        # Storage info
        print(f"\n💾 Storage:")
        print(f"  • Experiments: {self.jobs_dir}")
        print(f"  • Total Jobs: {len(self.job_registry['jobs'])}")
    
    def get_job_status(self, job_name: str) -> Optional[str]:
        """Get status of a specific job
        
        Args:
            job_name: Job name
            
        Returns:
            Job status string or None if not found
        """
        if job_name in self.job_registry["jobs"]:
            return self.job_registry["jobs"][job_name]["status"]
        return None
    
    def cancel_job(self, job_name: str) -> bool:
        """Cancel a running job
        
        Args:
            job_name: Job name
            
        Returns:
            True if cancelled successfully, False otherwise
        """
        if job_name not in self.job_registry["jobs"]:
            self.logger.error(f"Job '{job_name}' not found")
            return False
        
        job_info = self.job_registry["jobs"][job_name]
        
        if job_info["status"] not in [JobStatus.RUNNING, JobStatus.PENDING]:
            self.logger.warning(f"Job '{job_name}' is not running")
            return False
        
        # Update job status
        job_info["status"] = JobStatus.CANCELLED
        job_info["completed_at"] = str(datetime.now())
        self._save_job_registry()
        
        self.logger.info(f"Job '{job_name}' cancelled")
        return True