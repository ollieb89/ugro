"""Main UGRO orchestration agent."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from .cluster import Cluster
from .cluster_state import ClusterStateManager, NodeState
from .config import AppConfig, get_config_dir
from .job import Job, JobStatus
from .launch_coordinator import LaunchCoordinator
from .result_aggregator import ResultAggregator
from .result_aggregator import ResultAggregator
from .result_aggregator import ResultAggregator
from .health_monitor import TrainingMetricsCollector
from .database import Database
from .queue import JobQueue
from .database import Database
from .queue import JobQueue

if TYPE_CHECKING:
    from typing import Any

class UGROAgent:
    """Main orchestrator"""
 
    def __init__(
        self,
        config_name: str = "cluster.yaml",
        config: dict[str, Any] | None = None,
    ):
        logger = logging.getLogger(__name__)
        self.logger = logger

        if config:
            # If config provided as dict, attempt to validate it
            try:
                self.app_config = AppConfig(cluster=config.get("cluster", config))
            except Exception as e:
                logger.warning(f"Could not validate provided config dict: {e}")
                self.app_config = None
            self.config = config
        else:
            config_path = get_config_dir() / config_name
            try:
                self.app_config = AppConfig.from_yaml(config_path)
                # Export to dict for parts of the system still using dicts
                self.config = self.app_config.cluster.model_dump()
                # Merge root level fields if they exist in yaml but not in cluster model
                # (handled by AppConfig.from_yaml already)
            except Exception as e:
                logger.error(f"Failed to load/validate config from {config_path}: {e}")
                # Fallback to old loading if Pydantic fails (graceful degradation)
                from .config import load_config
                self.config = load_config(config_name)
                self.app_config = None
        self.cluster = Cluster(self.config)
        
        # Ensure workers are loaded (handle both 'workers' and 'nodes' fields)
        if not self.cluster.workers and 'nodes' in self.config:
            # Convert nodes dict to workers list
            nodes = self.config['nodes']
            self.cluster.workers = list(nodes.values())
            # Re-initialize SSH clients now that we have workers
            self.cluster._initialize_ssh_clients(self.cluster.env_command)

        # Database and Queue
        self.database = Database()
        self.queue = JobQueue(self.database)

        # Database and Queue
        self.database = Database()
        self.queue = JobQueue(self.database)

        self._result_aggregator = ResultAggregator()
        try:
            self._result_aggregator.base_dir.mkdir(parents=True, exist_ok=True)
            self._result_aggregator.jobs_dir.mkdir(parents=True, exist_ok=True)
            self._result_aggregator.experiments_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            logger.exception("Failed to create UGRO data directory: %s", self._result_aggregator.base_dir)
            raise
        
        # Job tracking
        self.jobs_dir = self._result_aggregator.jobs_dir
        self.job_registry_file = self.jobs_dir / "job_registry.json"
        self.job_registry = self._load_job_registry()

        self.cluster_state_manager = ClusterStateManager()
        self._load_cluster_state()
        self._sync_cluster_state_nodes()
        
        # Initialize metrics collector
        self.metrics_collector = TrainingMetricsCollector(cluster=self.cluster)

        # Initialize launch coordinator
        self.launch_coordinator = LaunchCoordinator(
            cluster=self.cluster,
            sync_window_seconds=30,
            monitoring_interval_seconds=5.0,
            max_startup_timeout_seconds=300,
            metrics_collector=self.metrics_collector,
        )

    def _load_cluster_state(self) -> None:
        """Load cluster state from disk, logging failures."""
        try:
            self.cluster_state_manager.load()
        except RuntimeError:
            self.logger.exception("Failed to load cluster state")

    def _persist_cluster_state(self) -> None:
        """Persist cluster state to disk, logging failures."""
        try:
            self.cluster_state_manager.save()
        except RuntimeError:
            self.logger.exception("Failed to persist cluster state")

    def _sync_cluster_state_nodes(self) -> None:
        """Ensure cluster nodes are represented in state storage."""
        state = self.cluster_state_manager.get_state()

        master = self.config.get("master", {}) if isinstance(self.config, dict) else {}
        master_name = master.get("hostname", "gpu-master")
        if master_name not in state.nodes:
            state.nodes[master_name] = NodeState(
                ip=str(master.get("ip", "")),
                gpu=str(master.get("gpu", "unknown")),
                vram_gb=int(master.get("vram_gb", 0) or 0),
                status="available",
                running_job_id=None,
            )

        for worker in self.cluster.get_all_workers():
            worker_name = worker.get("name")
            if not worker_name:
                continue
            if worker_name in state.nodes:
                continue
            hardware = worker.get("hardware", {}) if isinstance(worker, dict) else {}
            state.nodes[worker_name] = NodeState(
                ip=str(worker.get("ip", "")),
                gpu=str(hardware.get("gpu_model", "unknown")),
                vram_gb=int(hardware.get("vram_gb", 0) or 0),
                status="available",
                running_job_id=None,
            )

        self._persist_cluster_state()

    def _update_state_for_job_start(
        self,
        job_name: str,
        model: str,
        worker_names: list[str],
        worker_ranks: list[int],
    ) -> None:
        """Record job start and mark nodes as busy."""
        self.cluster_state_manager.set_job(
            job_name,
            self.cluster_state_manager.build_job_state(
                status="running",
                ranks=worker_ranks,
                model=model,
                gpu_nodes=worker_names,
            ),
        )

        for worker_name in worker_names:
            try:
                self.cluster_state_manager.update_node_status(
                    worker_name,
                    status="busy",
                    running_job_id=job_name,
                )
            except KeyError:
                continue

    def _update_state_for_job_end(self, job_name: str, status: str) -> None:
        """Update job status and release nodes."""
        try:
            self.cluster_state_manager.update_job_status(job_name, status)
        except KeyError:
            return

        job_state = self.cluster_state_manager.get_state().jobs.get(job_name)
        if not job_state:
            return

        for worker_name in job_state.gpu_nodes:
            try:
                self.cluster_state_manager.update_node_status(
                    worker_name,
                    status="available",
                    running_job_id=None,
                )
            except KeyError:
                continue

    def get_live_metrics(self, job_name: str) -> dict[str, Any] | None:
        """Get latest training metrics for dashboard."""
        if not self.metrics_collector:
            # We check if it exists on the agent
            return self._result_aggregator.get_job_summary(job_name)
        return self._result_aggregator.get_job_summary(job_name)
    
    def _load_job_registry(self) -> dict[str, Any]:
        """Load job registry from disk"""
        if self.job_registry_file.exists():
            try:
                with open(self.job_registry_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (OSError, json.JSONDecodeError):
                logging.getLogger(__name__).exception(
                    "Failed to load job registry: %s", self.job_registry_file
                )
        
        return {"jobs": {}, "last_updated": str(datetime.now())}
     
    def _save_job_registry(self) -> None:
        """Save job registry to disk"""
        self.job_registry["last_updated"] = str(datetime.now())
        
        try:
            with open(self.job_registry_file, "w", encoding="utf-8") as f:
                json.dump(self.job_registry, f, indent=2)
        except OSError:
            logging.getLogger(__name__).exception(
                "Failed to save job registry: %s", self.job_registry_file
            )
    
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
            results_dir=self._result_aggregator.base_dir,
            result_aggregator=self._result_aggregator,
        )
        
        # Get worker names
        workers = self.cluster.get_all_workers()
        worker_names = [worker['name'] for worker in workers]
        worker_ranks = [worker.get("rank", index) for index, worker in enumerate(workers)]
        
        # Start job
        if not job.start(worker_names):
            print(f"❌ Failed to start job '{job_name}'")
            return False

        self._update_state_for_job_start(job_name, model, worker_names, worker_ranks)
        
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

        self._update_state_for_job_end(job_name, job.status)
        
        # Update job registry
        if job_name in self.job_registry["jobs"]:
            self.job_registry["jobs"][job_name]["status"] = job.status
            if job.completed_at:
                self.job_registry["jobs"][job_name]["completed_at"] = job.completed_at.isoformat()
        self._save_job_registry()
        
        return success
    
    def _launch_ranks(self, job: Job, verbose: bool = False) -> bool:
        """Launch training on all nodes using LaunchCoordinator"""
        return asyncio.run(self._launch_ranks_async(job, verbose))
    
    async def _launch_ranks_async(self, job: Job, verbose: bool = False) -> bool:
        """Async implementation of rank launching using LaunchCoordinator"""
        workers = self.cluster.get_all_workers()
        
        print("📋 Launch Coordinator: Distributed Training Orchestration")
        print("=" * 60)
        
        try:
            # Step 1: Validate cluster state
            print("🔍 Step 1: Validating cluster state...")
            validation_success, validation_results = await self.launch_coordinator.validate_cluster_state()
            
            if not validation_success:
                print("❌ Cluster validation failed!")
                print(f"   Unhealthy nodes: {len(validation_results['unhealthy_nodes'])}")
                for node in validation_results['unhealthy_nodes']:
                    print(f"     • {node['name']}: {node['error']}")
                return False
            
            print(f"✓ All {len(validation_results['healthy_nodes'])} nodes validated successfully")
            
            # Step 2: Allocate resources
            print("\n🎯 Step 2: Allocating GPU resources...")
            allocation_plan = await self.launch_coordinator.allocate_resources(workers)
            
            # Step 3: Launch distributed training
            print("\n🚀 Step 3: Launching distributed training...")
            launch_success = await self.launch_coordinator.launch_distributed_training(
                job=job,
                allocation_plan=allocation_plan,
                training_script="scripts/train_production.py",
            )
            
            if not launch_success:
                print("❌ Failed to launch distributed training")
                return False
            
            print("✓ Distributed training launched successfully")
            
            # Step 4: Monitor training progress
            print("\n📊 Step 4: Monitoring training progress...")
            print("   Training is now running on all workers...")
            print("   Use 'ugro logs {}' to view real-time logs".format(job.name))
            print("   Use 'ugro results {}' to view progress".format(job.name))
            
            # Wait for training to complete (in a real implementation, this would be non-blocking)
            # For now, we'll simulate the monitoring
            if verbose:
                print("\n🔄 Training monitoring (simulated for demo):")
                for epoch in range(1, min(job.epochs + 1, 4)):  # Show first 3 epochs max
                    await asyncio.sleep(1.0)  # Simulate monitoring interval
                    
                    # Generate mock metrics for demonstration
                    loss = 2.5 - (epoch * 0.3)
                    accuracy = 0.6 + (epoch * 0.1)
                    epoch_time = 45.0 + (epoch * 2.0)
                    
                    job.add_metric(epoch, loss, accuracy, epoch_time)
                    
                    print(f"  Epoch {epoch}/{job.epochs}: Loss={loss:.3f}, Accuracy={accuracy:.3f}, Time={epoch_time:.1f}s")
                
                if job.epochs > 3:
                    print(f"  ... (training continues for {job.epochs - 3} more epochs)")
            
            # In a real implementation, the LaunchCoordinator would handle completion
            # For now, we'll mark as successful to demonstrate the flow
            print("\n✅ Training orchestration completed successfully!")
            return True
            
        except Exception as e:
            self.logger.error(f"Launch coordination error: {e}")
            job.add_error(f"Launch coordination error: {e}")
            print(f"❌ Launch coordination failed: {e}")
            
            # Clean up on failure
            try:
                await self.launch_coordinator.stop_training()
            except Exception as cleanup_error:
                self.logger.error(f"Cleanup error: {cleanup_error}")
            
            return False
    
    def display_logs(self, job_name: str, rank: int | None = None):
        """Display logs for a job"""
        if job_name not in self.job_registry["jobs"]:
            print(f"❌ Job '{job_name}' not found")
            return
        
        job_info = self.job_registry["jobs"][job_name]
        job_dir = Path(job_info["directory"])
        
        if rank is not None:
            log_file = job_dir / "logs" / f"rank_{rank}.log"
        else:
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
        print(f"  • UGRO Data: {self._result_aggregator.base_dir}")
        print(f"  • Total Jobs: {len(self.job_registry['jobs'])}")
    
    def get_job_status(self, job_name: str) -> str | None:
        """Get status of a specific job"""
        if job_name in self.job_registry["jobs"]:
            return self.job_registry["jobs"][job_name]["status"]
        return None

    def process_queue(self, loop_interval: float = 5.0):
        """Infinite loop to process jobs from the queue"""
        self.logger.info("Starting queue processor...")
        print(f"🔄 Queue Processor started. Polling every {loop_interval}s...")
        
        while True:
            try:
                # 1. Peek next job
                job_data = self.queue.peek_next_job()
                if not job_data:
                    time.sleep(loop_interval)
                    continue
                
                print(f"\\n📢 Found pending job: {job_data['id']} (Priority: {job_data['priority']})")
                
                # 2. Check resources
                # In a real system, we'd check if specific requested GPUs are free.
                # Here, we do a simple cluster availability check.
                # If any job is running (tracked in memory/file), we wait.
                
                active_job = None
                for name, j in self.job_registry["jobs"].items():
                     if j["status"] in [JobStatus.RUNNING]:
                         active_job = name
                         break
                
                if active_job:
                    print(f"⏳ Cluster busy with '{active_job}'. Waiting...")
                    time.sleep(loop_interval * 2)
                    continue

                # 3. Launch
                job_id = job_data['id']
                model = job_data['model_name']
                dataset = job_data['dataset_name']
                config = job_data['config']
                
                self.queue.update_status(job_id, 'running')
                
                # We reuse the job_id as the job_name for simplicity in the registry
                launch_name = job_id 
                
                print(f"🚀 Processing Job {job_id}...")
                
                success = self.launch_training(
                    job_name=launch_name,
                    model=model,
                    dataset=dataset,
                    epochs=config.get('epochs', 1),
                    learning_rate=config.get('learning_rate', 2e-4),
                    verbose=config.get('verbose', False)
                )
                
                final_status = 'completed' if success else 'failed'
                self.queue.update_status(job_id, final_status)
                
                print(f"🏁 Job {job_id} finished: {final_status}")
                
            except KeyboardInterrupt:
                print("\\n🛑 Queue processor stopping...")
                break
            except Exception as e:
                self.logger.error(f"Queue processing error: {e}")
                print(f"❌ Error in queue loop: {e}")
                time.sleep(loop_interval)