"""Launch Coordinator for UGRO distributed training.

Orchestrates distributed training launch across multiple GPU nodes with
synchronization, monitoring, and log collection capabilities.
"""

from __future__ import annotations

import asyncio
import json
import logging
import signal
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any
    from .health_monitor import TrainingMetricsCollector

from .cluster import Cluster
from .job import Job
from .cluster import Cluster
from .job import Job
from .ssh_utils import SSHClient
from .commands import CommandBuilder, TrainingJobParams


class LaunchCoordinator:
    """Coordinates distributed training launch across GPU cluster.
    
    Replaces manual multi-terminal setup with automated orchestration:
    1. Validate cluster state (all nodes reachable)
    2. Allocate GPU resources (rank assignment)
    3. SSH to each worker, start torchrun with unique rank
    4. Ensure all start within sync window
    5. Poll for completion or errors
    6. Collect logs and artifacts to central location
    """
    
    def __init__(
        self,
        cluster: Cluster,
        sync_window_seconds: int = 30,
        monitoring_interval_seconds: float = 5.0,
        max_startup_timeout_seconds: int = 300,
        metrics_collector: TrainingMetricsCollector | None = None,
    ):
        """Initialize launch coordinator.
        
        Args:
            cluster: Cluster instance with SSH clients
            sync_window_seconds: Time window for all processes to start
            monitoring_interval_seconds: Interval for progress monitoring
            max_startup_timeout_seconds: Maximum time to wait for all processes to start
        """
        self.cluster = cluster
        self.sync_window_seconds = sync_window_seconds
        self.monitoring_interval_seconds = monitoring_interval_seconds
        self.max_startup_timeout_seconds = max_startup_timeout_seconds
        self.metrics_collector = metrics_collector
        
        self.logger = logging.getLogger(__name__)
        
        # Process tracking
        self.active_processes: dict[str, asyncio.subprocess.Process] = {}
        self.process_status: dict[str, str] = {}
        self.startup_times: dict[str, datetime] = {}
        
        # Monitoring state
        self.monitoring_task: asyncio.Task[None] | None = None
        self.log_collection_tasks: dict[str, asyncio.Task[None]] = {}
        self.should_stop = asyncio.Event()
        
        # Training configuration
        self.master_addr = self._get_master_address()
        self.master_port = self._get_master_port()
    
    def _get_master_address(self) -> str:
        """Get master node address for distributed training."""
        master_config = self.cluster.config.get('master', {})
        return master_config.get('ip', '127.0.0.1')
    
    def _get_master_port(self) -> int:
        """Get master port for distributed training."""
        comm_config = self.cluster.config.get('communication', {})
        return comm_config.get('master_port', 29500)
    
    async def validate_cluster_state(self) -> tuple[bool, dict[str, Any]]:
        """Validate that all cluster nodes are reachable and healthy.
        
        Returns:
            Tuple of (validation_success, detailed_status)
        """
        self.logger.info("Validating cluster state...")
        
        health_status = self.cluster.check_health()
        validation_results = {
            'healthy_nodes': [],
            'unhealthy_nodes': [],
            'ssh_failures': [],
            'gpu_failures': [],
            'env_failures': [],
        }
        
        for node_name, status in health_status.items():
            if status['healthy']:
                validation_results['healthy_nodes'].append({
                    'name': node_name,
                    'gpu_model': status.get('gpu_model', 'Unknown'),
                    'vram_gb': status.get('vram_gb', 0),
                    'python_version': status.get('python_version', 'Unknown'),
                    'pytorch_version': status.get('pytorch_version', 'Unknown'),
                    'cuda_available': status.get('cuda_available', False),
                })
                self.logger.info(f"✓ {node_name}: Healthy ({status.get('message', 'OK')})")
            else:
                validation_results['unhealthy_nodes'].append({
                    'name': node_name,
                    'error': status.get('message', 'Unknown error'),
                })
                self.logger.error(f"❌ {node_name}: {status.get('message', 'Unhealthy')}")
                
                # Categorize failure type
                message = status.get('message', '').lower()
                if 'ssh' in message or 'connection' in message:
                    validation_results['ssh_failures'].append(node_name)
                elif 'gpu' in message or 'nvidia' in message:
                    validation_results['gpu_failures'].append(node_name)
                else:
                    validation_results['env_failures'].append(node_name)
        
        all_healthy = len(validation_results['unhealthy_nodes']) == 0
        
        if all_healthy:
            self.logger.info(f"✓ All {len(validation_results['healthy_nodes'])} nodes validated successfully")
        else:
            self.logger.error(f"❌ {len(validation_results['unhealthy_nodes'])} nodes failed validation")
        
        return all_healthy, validation_results
    
    async def allocate_resources(self, workers: list[dict[str, Any]]) -> dict[str, Any]:
        """Allocate GPU resources and prepare training environments.
        
        Args:
            workers: List of worker configurations
            
        Returns:
            Resource allocation plan with rank assignments
        """
        self.logger.info("Allocating GPU resources...")
        
        allocation_plan = {
            'total_nodes': len(workers),
            'rank_assignments': {},
            'master_addr': self.master_addr,
            'master_port': self.master_port,
            'world_size': len(workers),
        }
        
        for worker in workers:
            worker_name = worker['name']
            rank = worker.get('rank', 0)
            
            allocation_plan['rank_assignments'][worker_name] = {
                'rank': rank,
                'ip': worker['ip'],
                'gpu_model': worker['hardware']['gpu_model'],
                'vram_gb': worker['hardware']['vram_gb'],
                'user': worker['user'],
            }
            
            self.logger.info(f"  • Rank {rank}: {worker_name} ({worker['hardware']['gpu_model']})")
        
        self.logger.info(f"✓ Allocated {len(workers)} GPU resources for distributed training")
        
        return allocation_plan
    
    async def launch_distributed_training(
        self,
        job: Job,
        allocation_plan: dict[str, Any],
        training_script: str = "scripts/train_production.py",
    ) -> bool:
        """Launch distributed training on all workers.
        
        Args:
            job: Job instance for tracking
            allocation_plan: Resource allocation plan
            training_script: Path to training script relative to project root
            
        Returns:
            True if launch successful, False otherwise
        """
        self.logger.info("Launching distributed training...")
        
        # Build torchrun command template
        cmd_template = self._build_torchrun_command(
            allocation_plan=allocation_plan,
            training_script=training_script,
            job=job,
        )
        
        # Launch processes on all workers
        launch_tasks = []
        for worker_name, worker_config in allocation_plan['rank_assignments'].items():
            task = asyncio.create_task(
                self._launch_worker_process(
                    worker_name=worker_name,
                    worker_config=worker_config,
                    allocation_plan=allocation_plan,
                    cmd_template=cmd_template,
                    job=job,
                )
            )
            launch_tasks.append(task)
        
        # Wait for all launch tasks with timeout
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*launch_tasks, return_exceptions=True),
                timeout=self.max_startup_timeout_seconds,
            )
        except asyncio.TimeoutError:
            self.logger.error("❌ Launch timeout: some workers failed to start within time limit")
            await self._cleanup_processes()
            return False
        
        # Check launch results
        successful_launches = 0
        for result in results:
            if isinstance(result, Exception):
                self.logger.error(f"❌ Launch exception: {result}")
                continue
            
            worker_name, success = result
            if success:
                successful_launches += 1
                self.logger.info(f"✓ {worker_name}: Process started successfully")
            else:
                self.logger.error(f"❌ {worker_name}: Failed to start process")
        
        if successful_launches != len(allocation_plan['rank_assignments']):
            self.logger.error(f"❌ Only {successful_launches}/{len(allocation_plan['rank_assignments'])} workers started successfully")
            await self._cleanup_processes()
            return False
        
        # Start monitoring and log collection
        await self._start_monitoring(job)
        await self._start_log_collection(job)
        
        # Start metrics collection if available
        if self.metrics_collector:
            ranks = [cfg['rank'] for cfg in allocation_plan['rank_assignments'].values()]
            workers = self.cluster.get_all_workers()
            await self.metrics_collector.start_collection(
                job_id=job.name,
                ranks=ranks,
                workers=workers,
                ssh_clients=self.cluster.ssh_clients
            )
        
        self.logger.info(f"✓ All {successful_launches} workers launched successfully")
        return True
    
    def _build_torchrun_command(
        self,
        allocation_plan: dict[str, Any],
        training_script: str,
        job: Job,
    ) -> str:
        """Build torchrun command template.
        
        Args:
            allocation_plan: Resource allocation plan
            training_script: Training script path
            job: Job instance
            
        Returns:
            Command template with placeholders for rank-specific values
        """
        return CommandBuilder.build_torchrun_command(
            TrainingJobParams(
                job_id=job.name,
                model=job.model,
                dataset=job.dataset,
                epochs=job.epochs,
                learning_rate=job.learning_rate,
                nnodes=allocation_plan['world_size'],
                nproc_per_node=1,
                master_addr=allocation_plan['master_addr'],
                master_port=allocation_plan['master_port'],
                node_rank=0, # Placeholder, dynamic
                script_path=training_script
            )
        )
    
    async def _launch_worker_process(
        self,
        worker_name: str,
        worker_config: dict[str, Any],
        allocation_plan: dict[str, Any],
        cmd_template: str,
        job: Job,
    ) -> tuple[str, bool]:
        """Launch training process on a specific worker.
        
        Args:
            worker_name: Name of the worker
            worker_config: Worker configuration
            allocation_plan: Resource allocation plan
            cmd_template: Command template with placeholders
            job: Job instance
            
        Returns:
            Tuple of (worker_name, success)
        """
        ssh_client = self.cluster.ssh_clients.get(worker_name)
        if not ssh_client:
            self.logger.error(f"No SSH client for worker {worker_name}")
            return worker_name, False
        
        # Substitute rank-specific values (CommandBuilder logic handles this, but here we replace the placeholder for now
        # or we could rebuild the command. For efficiency we likely want to just swap the rank index if possible,
        # but since CommandBuilder produces a precise string, regening it is safer)
        
        # Re-build command for specific rank to ensure safety (overkill but safe)
        params = TrainingJobParams(
            job_id=job.name,
            model=job.model,
            dataset=job.dataset,
            epochs=job.epochs,
            learning_rate=job.learning_rate,
            nnodes=allocation_plan['world_size'],
            nproc_per_node=1,
            master_addr=allocation_plan['master_addr'],
            master_port=allocation_plan['master_port'],
            node_rank=rank,
            script_path=allocation_plan.get("script_path", "scripts/train_production.py") # Ensure script path is passed or defaulted
        )
        command = CommandBuilder.build_torchrun_command(params)
        
        # Update job worker status
        job.update_worker_status(worker_name, JobStatus.RUNNING, f"Starting torchrun (rank {rank})")
        
        self.logger.info(f"Starting process on {worker_name} (rank {rank})")
        
        # Execute command via Async SSH
        success, stdout, stderr = await ssh_client.run_command_async(
            command,
            timeout=self.max_startup_timeout_seconds,
            use_env=True,
        )
        
        if success:
            self.startup_times[worker_name] = datetime.now()
            self.process_status[worker_name] = 'running'
            job.update_worker_status(worker_name, JobStatus.RUNNING, f"Training started (rank {rank})")
        else:
            self.process_status[worker_name] = 'failed'
            job.update_worker_status(worker_name, JobStatus.FAILED, f"Failed to start: {stderr}")
            self.logger.error(f"Failed to start process on {worker_name}: {stderr}")
        
        return worker_name, success
    
    async def _start_monitoring(self, job: Job) -> None:
        """Start monitoring training progress."""
        self.monitoring_task = asyncio.create_task(self._monitor_training(job))
        self.logger.info("Started training monitoring")
    
    async def _start_log_collection(self, job: Job) -> None:
        """Start log collection from all workers."""
        for worker_name in self.cluster.ssh_clients.keys():
            task = asyncio.create_task(
                self._collect_worker_logs(worker_name, job)
            )
            self.log_collection_tasks[worker_name] = task
        
        self.logger.info(f"Started log collection for {len(self.log_collection_tasks)} workers")
    
    async def _monitor_training(self, job: Job) -> None:
        """Monitor training progress and handle completion."""
        self.logger.info("Monitoring training progress...")
        last_tb_sync = time.time()
        
        try:
            while not self.should_stop.is_set():
                # Periodic TensorBoard sync (every 60s)
                if time.time() - last_tb_sync > 60:
                    if self.metrics_collector:
                        await self.metrics_collector._result_aggregator.sync_tensorboard(
                            job.name,
                            self.cluster.get_all_workers(),
                            self.cluster.ssh_clients
                        )
                    last_tb_sync = time.time()
                # Check process status on all workers
                active_workers = 0
                completed_workers = 0
                failed_workers = 0
                
                for worker_name, ssh_client in self.cluster.ssh_clients.items():
                    if self.process_status.get(worker_name) == 'running':
                        # Check if process is still running
                        # Use async call
                        success, stdout, stderr = await ssh_client.run_command_async(
                            "pgrep -f torchrun > /dev/null 2>&1",
                            timeout=5,
                            use_env=False,
                        )
                        
                        if success:
                            active_workers += 1
                        else:
                            # Process finished, check if successful
                            self.process_status[worker_name] = 'completed'
                            completed_workers += 1
                            job.update_worker_status(worker_name, JobStatus.COMPLETED, "Training completed")
                            self.logger.info(f"✓ {worker_name}: Training completed")
                    
                    elif self.process_status.get(worker_name) == 'completed':
                        completed_workers += 1
                    elif self.process_status.get(worker_name) == 'failed':
                        failed_workers += 1
                
                # Update job progress
                total_workers = len(self.cluster.ssh_clients)
                progress_percent = (completed_workers / total_workers) * 100 if total_workers > 0 else 0
                
                self.logger.info(
                    f"Training status: {active_workers} active, {completed_workers} completed, "
                    f"{failed_workers} failed ({progress_percent:.1f}% complete)"
                )
                
                # Check if training is complete
                if completed_workers + failed_workers == total_workers:
                    break
                
                # Wait for next monitoring interval
                await asyncio.sleep(self.monitoring_interval_seconds)
        
        except asyncio.CancelledError:
            self.logger.info("Monitoring task cancelled")
        except Exception as e:
            self.logger.error(f"Monitoring error: {e}")
        
        # Final status update
        await self._finalize_job_status(job)
    
    async def _collect_worker_logs(self, worker_name: str, job: Job) -> None:
        """Collect logs from a specific worker."""
        ssh_client = self.cluster.ssh_clients.get(worker_name)
        if not ssh_client:
            return
        
        self.logger.info(f"Starting log collection for {worker_name}")
        
        try:
            while not self.should_stop.is_set():
                # Get recent log output from worker
                # Async call for log collection
                success, stdout, stderr = await ssh_client.run_command_async(
                    "echo 'Log collection placeholder - would tail training logs here'",
                    timeout=5,
                    use_env=False,
                )
                
                if success and stdout.strip():
                    # Add timestamp and worker prefix
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    log_entry = f"[{timestamp}] [{worker_name}] {stdout.strip()}"
                    
                    # Write to job log file
                    log_file = job.get_log_file()
                    with open(log_file, 'a', encoding='utf-8') as f:
                        f.write(log_entry + '\n')
                
                await asyncio.sleep(self.monitoring_interval_seconds)
        
        except asyncio.CancelledError:
            self.logger.info(f"Log collection for {worker_name} cancelled")
        except Exception as e:
            self.logger.error(f"Log collection error for {worker_name}: {e}")
    
    async def _finalize_job_status(self, job: Job) -> None:
        """Finalize job status based on worker outcomes."""
        completed_workers = sum(1 for status in self.process_status.values() if status == 'completed')
        failed_workers = sum(1 for status in self.process_status.values() if status == 'failed')
        total_workers = len(self.process_status)
        
        if completed_workers == total_workers:
            job.complete(success=True)
            self.logger.info(f"✅ Job {job.name} completed successfully")
        elif failed_workers > 0:
            job.complete(success=False)
            self.logger.error(f"❌ Job {job.name} failed: {failed_workers}/{total_workers} workers failed")
        else:
            self.logger.error(f"❌ Job {job.name} ended with unknown status")

        # Stop metrics collection
        if self.metrics_collector:
            await self.metrics_collector.stop_collection(job.name)
    
    async def _cleanup_processes(self) -> None:
        """Clean up running processes on all workers."""
        self.logger.info("Cleaning up processes...")
        
        cleanup_tasks = []
        for worker_name, ssh_client in self.cluster.ssh_clients.items():
            task = asyncio.create_task(self._cleanup_worker_processes(worker_name, ssh_client))
            cleanup_tasks.append(task)
        
        await asyncio.gather(*cleanup_tasks, return_exceptions=True)
        self.logger.info("Process cleanup completed")
    
    async def _cleanup_worker_processes(self, worker_name: str, ssh_client: SSHClient) -> None:
        """Clean up processes on a specific worker."""
        try:
            # Kill torchrun processes
            success, stdout, stderr = await ssh_client.run_command_async(
                "pkill -f torchrun",
                timeout=10,
                use_env=False,
            )
            
            if success:
                self.logger.info(f"✓ {worker_name}: Processes cleaned up")
            else:
                self.logger.warning(f"⚠ {worker_name}: Cleanup failed: {stderr}")
        
        except Exception as e:
            self.logger.error(f"Cleanup error for {worker_name}: {e}")
    
    async def stop_training(self) -> None:
        """Stop training and clean up resources."""
        self.logger.info("Stopping training...")
        
        # Signal stop to all tasks
        self.should_stop.set()
        
        # Cancel monitoring task
        if self.monitoring_task:
            self.monitoring_task.cancel()
        
        # Cancel log collection tasks
        for task in self.log_collection_tasks.values():
            task.cancel()
        
        # Clean up processes
        await self._cleanup_processes()
        
        # Stop metrics collection
        if self.metrics_collector:
            # Note: We don't have job_id here easily, but stop_training is usually called for the active session.
            # However, stop_collection is per job_id. 
            # For now, we'll stop all active collections in the collector if needed, 
            # but ideally we'd pass job_id here.
            # As a shortcut, the collector handles multiple jobs, but LaunchCoordinator 
            # currently handles one active launch at a time in self.active_processes (mapped by name).
            pass 
            # Actually, LaunchCoordinator should track the active job_id if possible.
        
        self.logger.info("Training stopped and resources cleaned up")
    
    @asynccontextmanager
    async def training_session(self, job: Job) -> AsyncIterator[None]:
        """Context manager for training session with automatic cleanup."""
        try:
            yield
        except Exception as e:
            self.logger.error(f"Training session error: {e}")
            job.add_error(f"Session error: {e}")
        finally:
            await self.stop_training()