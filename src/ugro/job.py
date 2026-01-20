"""Job management for UGRO training jobs."""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from .result_aggregator import ResultAggregator

if TYPE_CHECKING:
    from typing import Any


class JobStatus:
    """Job status constants"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Job:
    """Manages a distributed training job"""
    
    def __init__(
        self,
        name: str,
        model: str,
        dataset: str,
        epochs: int = 1,
        learning_rate: float = 0.0002,
        batch_size: int = 1,
        results_dir: Path | None = None,
        result_aggregator: ResultAggregator | None = None,
    ):
        """Initialize training job
        
        Args:
            name: Job name
            model: Model name/path
            dataset: Dataset name
            epochs: Number of training epochs
            learning_rate: Learning rate
            batch_size: Batch size per GPU
            results_dir: Directory for job results
            result_aggregator: Result aggregator instance
        """
        self.id = str(uuid.uuid4())
        self.name = name
        self.model = model
        self.dataset = dataset
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        
        self._result_aggregator = result_aggregator or ResultAggregator(base_dir=results_dir)
        self._result_paths = self._result_aggregator.ensure_job_layout(name)
        
        self.result_dir = self._result_paths.job_dir
        
        # Job metadata
        self.status = JobStatus.PENDING
        self.created_at = datetime.now()
        self.started_at: datetime | None = None
        self.completed_at: datetime | None = None
        
        self._write_config_json()
        
        # Training metrics
        self.metrics = {
            'loss': [],
            'accuracy': [],
            'learning_rate': [],
            'epoch_times': []
        }
        
        # Worker tracking
        self.workers: list[str] = []
        self.worker_status: dict[str, str] = {}
        
        # Error tracking
        self.errors: list[str] = []
        
        # Save initial job metadata
        self._save_metadata()
    
    def _write_config_json(self) -> None:
        config_payload = {
            "job_id": self.name,
            "internal_id": self.id,
            "model": self.model,
            "dataset": self.dataset,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "created_at": self.created_at.isoformat(),
        }
        self._result_aggregator.write_job_config(self.name, config_payload)
    
    def start(self, workers: list[str]) -> bool:
        """Start the training job
        
        Args:
            workers: List of worker names participating in training
            
        Returns:
            True if job started successfully, False otherwise
        """
        if self.status != JobStatus.PENDING:
            return False
        
        self.workers = workers
        self.worker_status = {worker: JobStatus.PENDING for worker in workers}
        self.status = JobStatus.RUNNING
        self.started_at = datetime.now()
        
        # Create training log file
        self._create_training_log()
        
        # Save updated metadata
        self._save_metadata()
        
        return True
    
    def update_worker_status(self, worker_name: str, status: str, message: str = ""):
        """Update status of a specific worker
        
        Args:
            worker_name: Name of the worker
            status: New status
            message: Optional status message
        """
        if worker_name in self.worker_status:
            self.worker_status[worker_name] = status
            
            if message:
                self._log_message(f"Worker {worker_name}: {message}")
            
            self._save_metadata()
    
    def add_metric(self, epoch: int, loss: float, accuracy: float | None = None, epoch_time: float | None = None):
        """Add training metrics
        
        Args:
            epoch: Epoch number
            loss: Loss value
            accuracy: Optional accuracy value
            epoch_time: Optional epoch time in seconds
        """
        self.metrics['loss'].append(loss)
        if accuracy is not None:
            self.metrics['accuracy'].append(accuracy)
        if epoch_time is not None:
            self.metrics['epoch_times'].append(epoch_time)
        
        self.metrics['learning_rate'].append(self.learning_rate)
        
        # Log metrics
        log_msg = f"Epoch {epoch}: Loss={loss:.4f}"
        if accuracy is not None:
            log_msg += f", Accuracy={accuracy:.4f}"
        if epoch_time is not None:
            log_msg += f", Time={epoch_time:.2f}s"
        
        self._log_message(log_msg)
        
        self._result_aggregator.append_metrics(
            self.name,
            {
                "timestamp": datetime.now().isoformat(),
                "job_id": self.name,
                "epoch": epoch,
                "loss": loss,
                "accuracy": accuracy,
                "epoch_time": epoch_time,
                "learning_rate": self.learning_rate,
            },
        )
        
        self._save_metadata()
    
    def add_error(self, error: str):
        """Add error message
        
        Args:
            error: Error message
        """
        self.errors.append(error)
        self._log_message(f"ERROR: {error}")
        self._save_metadata()
    
    def complete(self, success: bool = True):
        """Mark job as completed
        
        Args:
            success: Whether job completed successfully
        """
        if self.status != JobStatus.RUNNING:
            return
        
        self.status = JobStatus.COMPLETED if success else JobStatus.FAILED
        self.completed_at = datetime.now()
        
        # Update all worker statuses
        for worker in self.worker_status:
            if self.worker_status[worker] == JobStatus.RUNNING:
                self.worker_status[worker] = JobStatus.COMPLETED if success else JobStatus.FAILED
        
        self._log_message(f"Job {'completed successfully' if success else 'failed'}")
        self._save_metadata()
    
    def cancel(self):
        """Cancel the job"""
        if self.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
            return
        
        self.status = JobStatus.CANCELLED
        self.completed_at = datetime.now()
        
        # Update all worker statuses
        for worker in self.worker_status:
            self.worker_status[worker] = JobStatus.CANCELLED
        
        self._log_message("Job cancelled")
        self._save_metadata()
    
    def get_progress(self) -> dict[str, Any]:
        """Get job progress information
        
        Returns:
            Progress dictionary
        """
        completed_epochs = len(self.metrics['loss'])
        total_epochs = self.epochs
        
        # Calculate worker progress
        worker_progress = {}
        for worker, status in self.worker_status.items():
            if status == JobStatus.COMPLETED:
                worker_progress[worker] = 100
            elif status == JobStatus.RUNNING:
                worker_progress[worker] = (completed_epochs / total_epochs) * 100 if total_epochs > 0 else 0
            else:
                worker_progress[worker] = 0
        
        return {
            'job_id': self.id,
            'name': self.name,
            'status': self.status,
            'progress_percent': (completed_epochs / total_epochs) * 100 if total_epochs > 0 else 0,
            'completed_epochs': completed_epochs,
            'total_epochs': total_epochs,
            'worker_progress': worker_progress,
            'current_loss': self.metrics['loss'][-1] if self.metrics['loss'] else None,
            'current_accuracy': self.metrics['accuracy'][-1] if self.metrics['accuracy'] else None,
            'errors_count': len(self.errors),
            'created_at': self.created_at.isoformat(),
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
        }
    
    def get_log_file(self) -> Path:
        """Get path to training log file
        
        Returns:
            Path to log file
        """
        return self.result_dir / "logs" / "training.log"
    
    def get_rank_log_file(self, rank: int) -> Path:
        return self._result_aggregator.rank_log_path(self.name, rank)
    
    def get_metrics_jsonl_file(self) -> Path:
        return self._result_paths.metrics_jsonl
    
    def get_checkpoint_dir(self) -> Path:
        """Get path to checkpoint directory
        
        Returns:
            Path to checkpoint directory
        """
        return self.result_dir / "checkpoints"
    
    def get_tensorboard_dir(self) -> Path:
        """Get path to tensorboard directory
        
        Returns:
            Path to tensorboard directory
        """
        return self.result_dir / "tensorboard"
    
    def _create_training_log(self):
        """Create initial training log file"""
        log_file = self.get_log_file()
        
        rank_log_file = self.get_rank_log_file(0)
        
        header_lines = [
            f"# Training Log for Job: {self.name}\n",
            f"# Job ID: {self.id}\n",
            f"# Started: {self.started_at}\n",
            f"# Model: {self.model}\n",
            f"# Dataset: {self.dataset}\n",
            f"# Epochs: {self.epochs}\n",
            f"# Learning Rate: {self.learning_rate}\n",
            f"# Batch Size: {self.batch_size}\n",
            f"# Workers: {', '.join(self.workers)}\n",
            "\n",
            "Training started...\n",
        ]
        
        with open(log_file, "w", encoding="utf-8") as f:
            f.writelines(header_lines)
        
        with open(rank_log_file, "w", encoding="utf-8") as f:
            f.writelines(header_lines)
    
    def _log_message(self, message: str):
        """Log message to training log file
        
        Args:
            message: Message to log
        """
        log_file = self.get_log_file()
        rank_log_file = self.get_rank_log_file(0)
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {message}\n"
        
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(line)
        
        with open(rank_log_file, "a", encoding="utf-8") as f:
            f.write(line)
    
    def _save_metadata(self):
        """Save job metadata to JSON file"""
        metadata_file = self.result_dir / "metadata.json"
        
        metadata = {
            'id': self.id,
            'name': self.name,
            'model': self.model,
            'dataset': self.dataset,
            'epochs': self.epochs,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'status': self.status,
            'created_at': self.created_at.isoformat(),
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'workers': self.workers,
            'worker_status': self.worker_status,
            'metrics': self.metrics,
            'errors': self.errors,
            'result_dir': str(self.result_dir)
        }
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    @classmethod
    def load_from_metadata(cls, metadata_file: Path) -> 'Job':
        """Load job from metadata file
        
        Args:
            metadata_file: Path to metadata file
            
        Returns:
            Job instance
        """
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # Create job instance
        job = cls(
            name=metadata['name'],
            model=metadata['model'],
            dataset=metadata['dataset'],
            epochs=metadata['epochs'],
            learning_rate=metadata['learning_rate'],
            batch_size=metadata['batch_size']
        )
        
        # Restore metadata
        job.id = metadata['id']
        job.status = metadata['status']
        job.created_at = datetime.fromisoformat(metadata['created_at'])
        job.started_at = datetime.fromisoformat(metadata['started_at']) if metadata['started_at'] else None
        job.completed_at = datetime.fromisoformat(metadata['completed_at']) if metadata['completed_at'] else None
        job.workers = metadata['workers']
        job.worker_status = metadata['worker_status']
        job.metrics = metadata['metrics']
        job.errors = metadata['errors']
        
        return job
