"""Main UGRO orchestration agent."""

from __future__ import annotations

import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from .cluster import Cluster
from .config import expand_paths, load_config
from .job import Job, JobStatus
from .ssh_utils import SSHClient

if TYPE_CHECKING:
    from typing import Any

class UGROAgent:
    """Main orchestrator"""
    
    def __init__(self):
        # Load configuration using simplified approach
        config = load_config("cluster.yaml")
        config = expand_paths(config)
        
        # Handle cluster.yaml structure - merge cluster section with root level fields
        if 'cluster' in config:
            cluster_fields = config['cluster']
            config.update(cluster_fields)
        
        self.config = config
        self.cluster = Cluster(self.config)
        self.results_dir = Path.home() / "projects" / "UGRO" / "data" / "experiments"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Job tracking
        self.jobs_dir = self.results_dir
        self.job_registry_file = self.jobs_dir / "job_registry.json"
        self.job_registry = self._load_job_registry()
    
    def _load_job_registry(self) -> dict[str, Any]:
        """Load job registry from disk"""
        if self.job_registry_file.exists():
            try:
                with open(self.job_registry_file, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        
        return {"jobs": {}, "last_updated": str(datetime.now())}
    
    def _save_job_registry(self):
        """Save job registry to disk"""
        self.job_registry["last_updated"] = str(datetime.now())
        
        try:
            with open(self.job_registry_file, 'w') as f:
                json.dump(self.job_registry, f, indent=2)
        except Exception:
            pass
    
    def check_cluster_health(self) -> dict[str, dict[str, Any]]:
        """Check health of all nodes"""
        return self.cluster.check_health()
    
    def launch_training(
        self,
        job_name: str,
        model: str,
        dataset: str,
        epochs: int = 1,
        learning_rate: float = 0.0002,
        verbose: bool = False,
    ) -> bool:
        """Launch distributed training"""
        
        print(f"\n{'='*60}")
        print(f"UGRO: Launching Distributed Training")
        print(f"{'='*60}")
        print(f"Job: {job_name}")
        print(f"Model: {model}")
        print(f"Dataset: {dataset}")
        print(f"Epochs: {epochs}\n")
        
        # Validate cluster
        print("🔍 Checking cluster...")
        health = self.check_cluster_health()
        
        if not all(h['healthy'] for h in health.values()):
            print("❌ Cluster health check failed!")
            return False
        
        print("✓ All nodes healthy\n")
        
        # Check if job already exists
        if job_name in self.job_registry["jobs"]:
            print(f"❌ Job '{job_name}' already exists")
            return False
        
        # Create job
        job = Job(
            name=job_name,
            model=model,
            dataset=dataset,
            epochs=epochs,
            learning_rate=learning_rate,
            results_dir=self.results_dir
        )
        
        # Get worker names
        worker_names = [worker['name'] for worker in self.cluster.get_all_workers()]
        
        # Start job
        if not job.start(worker_names):
            print(f"❌ Failed to start job '{job_name}'")
            return False
        
        # Register job
        self.job_registry["jobs"][job_name] = {
            "id": job.id,
            "status": job.status,
            "created_at": job.created_at.isoformat(),
            "started_at": job.started_at.isoformat() if job.started_at else None,
            "parameters": {
                "model": model,
                "dataset": dataset,
                "epochs": epochs,
                "learning_rate": learning_rate,
                "verbose": verbose
            },
            "directory": str(job.result_dir),
            "workers": worker_names
        }
        self._save_job_registry()
        
        print(f"🚀 Launching {len(worker_names)} ranks...\n")
        
        # Launch ranks
        success = self._launch_ranks(job, verbose)
        
        if success:
            print(f"\n✅ Job {job_name} launched successfully!")
            print(f"📁 Results: {job.result_dir}")
        else:
            job.complete(success=False)
            print(f"\n❌ Job {job_name} failed!")
        
        # Update job registry
        if job_name in self.job_registry["jobs"]:
            self.job_registry["jobs"][job_name]["status"] = job.status
            if job.completed_at:
                self.job_registry["jobs"][job_name]["completed_at"] = job.completed_at.isoformat()
        self._save_job_registry()
        
        return success
    
    def _launch_ranks(self, job: Job, verbose: bool = False) -> bool:
        """Launch training on all nodes"""
        workers = self.cluster.get_all_workers()
        
        # Simulate distributed training launch
        # In real implementation, this would:
        # 1. Copy training scripts to all workers
        # 2. Start training processes with proper rank assignments
        # 3. Monitor progress and handle failures
        
        print("📋 Launching training processes:")
        
        for i, worker in enumerate(workers):
            worker_name = worker['name']
            rank = worker['rank']
            
            print(f"  • Rank {rank} on {worker_name} ({worker['hardware']['gpu_model']})")
            
            # Update worker status
            job.update_worker_status(worker_name, JobStatus.RUNNING, f"Training started as rank {rank}")
            
            # Simulate some training progress
            if verbose:
                print(f"    - Copying training scripts...")
                print(f"    - Starting process with rank {rank}...")
                print(f"    - Process started successfully")
        
        # Simulate training progress
        print("\n🔄 Training progress:")
        for epoch in range(1, job.epochs + 1):
            # Simulate epoch training time
            time.sleep(0.5)  # Simulate training time
            
            # Generate mock metrics
            loss = 2.5 - (epoch * 0.3)  # Decreasing loss
            accuracy = 0.6 + (epoch * 0.1)  # Increasing accuracy
            epoch_time = 45.0 + (epoch * 2.0)  # Increasing time per epoch
            
            job.add_metric(epoch, loss, accuracy, epoch_time)
            
            print(f"  Epoch {epoch}/{job.epochs}: Loss={loss:.3f}, Accuracy={accuracy:.3f}, Time={epoch_time:.1f}s")
        
        # Mark job as completed
        job.complete(success=True)
        
        return True
    
    def display_logs(self, job_name: str, rank: int | None = None):
        """Display logs for a job"""
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
    
    def display_results(self, job_name: str):
        """Display results for a job"""
        if job_name not in self.job_registry["jobs"]:
            print(f"❌ Job '{job_name}' not found")
            return
        
        job_info = self.job_registry["jobs"][job_name]
        
        print(f"\n📊 Results Summary: {job_name}")
        print("=" * 60)
        print(f"Status: {job_info['status']}")
        print(f"Created: {job_info['created_at']}")
        
        params = job_info['parameters']
        print(f"Model: {params['model']}")
        print(f"Dataset: {params['dataset']}")
        print(f"Epochs: {params['epochs']}")
        print(f"Learning Rate: {params['learning_rate']}")
        print(f"Workers: {len(job_info['workers'])}")
        
        # Try to load job for detailed metrics
        job_dir = Path(job_info["directory"])
        metadata_file = job_dir / "metadata.json"
        
        if metadata_file.exists():
            try:
                job = Job.load_from_metadata(metadata_file)
                
                if job.status == JobStatus.COMPLETED:
                    print(f"\n✅ Training completed successfully!")
                    if job.metrics['loss']:
                        print(f"Final Loss: {job.metrics['loss'][-1]:.4f}")
                    if job.metrics['accuracy']:
                        print(f"Final Accuracy: {job.metrics['accuracy'][-1]:.4f}")
                    if job.metrics['epoch_times']:
                        total_time = sum(job.metrics['epoch_times'])
                        print(f"Total Training Time: {total_time:.1f}s ({total_time/60:.1f}m)")
                elif job.status == JobStatus.RUNNING:
                    print(f"\n🔄 Training in progress...")
                    if job.metrics['loss']:
                        print(f"Current Loss: {job.metrics['loss'][-1]:.4f}")
                else:
                    print(f"\n❌ Training failed or was cancelled")
                    if job.errors:
                        print("Errors:")
                        for error in job.errors[-3:]:  # Show last 3 errors
                            print(f"  • {error}")
                
            except Exception:
                pass
        
        print(f"\n📁 Output Directory: {job_dir}")
        print(f"📊 TensorBoard: {job_dir / 'tensorboard'}")
        print(f"💾 Checkpoints: {job_dir / 'checkpoints'}")
    
    def display_status(self):
        """Display cluster status"""
        cluster_info = self.cluster.get_cluster_info()
        
        print(f"\n🖥️  Cluster Status: {cluster_info['name']}")
        print("=" * 60)
        print(f"Location: {cluster_info['location']}")
        print(f"Description: {cluster_info['description']}")
        
        # Worker status
        workers = cluster_info['workers']
        print(f"\n🔧 Workers ({len(workers)} nodes):")
        for worker in workers:
            print(f"  • {worker['name']}: {worker['hardware']['gpu_model']} ({worker['hardware']['vram_gb']}GB VRAM)")
        
        # Active jobs
        active_jobs = [name for name, job in self.job_registry["jobs"].items() 
                      if job["status"] in [JobStatus.RUNNING, JobStatus.PENDING]]
        
        print(f"\n🚀 Active Jobs: {len(active_jobs)}")
        for job_name in active_jobs:
            job_info = self.job_registry["jobs"][job_name]
            started_at = job_info.get('started_at', 'Unknown')
            print(f"  • {job_name}: {job_info['status']} (started {started_at})")
        
        # Recent jobs
        recent_jobs = list(self.job_registry["jobs"].keys())[-5:] if self.job_registry["jobs"] else []
        
        print(f"\n📋 Recent Jobs:")
        for job_name in recent_jobs:
            job_info = self.job_registry["jobs"][job_name]
            status = job_info['status']
            emoji = "✅" if status == JobStatus.COMPLETED else "❌" if status == JobStatus.FAILED else "🔄"
            print(f"  • {job_name} {emoji} {status}")
        
        # Storage info
        print(f"\n💾 Storage:")
        print(f"  • Experiments: {self.results_dir}")
        print(f"  • Total Jobs: {len(self.job_registry['jobs'])}")
    
    def get_job_status(self, job_name: str) -> str | None:
        """Get status of a specific job"""
        if job_name in self.job_registry["jobs"]:
            return self.job_registry["jobs"][job_name]["status"]
        return None