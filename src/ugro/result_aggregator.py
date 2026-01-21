from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .ssh_utils import SSHClient


@dataclass(frozen=True, slots=True)
class ResultPaths:
    job_dir: Path
    config_json: Path
    metrics_jsonl: Path
    logs_dir: Path
    checkpoints_dir: Path
    tensorboard_dir: Path


class ResultAggregator:
    def __init__(self, base_dir: Path | None = None) -> None:
        env_base = os.getenv("UGRO_DATA_DIR")
        if base_dir is not None:
            resolved = base_dir
        elif env_base:
            resolved = Path(env_base)
        else:
            resolved = Path(f"{os.getenv('HOME')}/Development/Tools/ugro_data")

        self.base_dir = resolved
        self.jobs_dir = self.base_dir / "jobs"
        self.experiments_dir = self.base_dir / "experiments"

    def ensure_job_layout(self, job_id: str) -> ResultPaths:
        job_dir = self.jobs_dir / job_id
        logs_dir = job_dir / "logs"
        checkpoints_dir = job_dir / "checkpoints"
        tensorboard_dir = job_dir / "tensorboard"

        job_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        tensorboard_dir.mkdir(parents=True, exist_ok=True)

        config_json = job_dir / "config.json"
        metrics_jsonl = job_dir / "metrics.jsonl"

        if not metrics_jsonl.exists():
            metrics_jsonl.touch()

        return ResultPaths(
            job_dir=job_dir,
            config_json=config_json,
            metrics_jsonl=metrics_jsonl,
            logs_dir=logs_dir,
            checkpoints_dir=checkpoints_dir,
            tensorboard_dir=tensorboard_dir,
        )

    def write_job_config(self, job_id: str, config: dict[str, Any]) -> Path:
        paths = self.ensure_job_layout(job_id)
        with open(paths.config_json, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        return paths.config_json

    def append_metrics(self, job_id: str, payload: dict[str, Any]) -> Path:
        paths = self.ensure_job_layout(job_id)
        with open(paths.metrics_jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")
        return paths.metrics_jsonl

    def rank_log_path(self, job_id: str, rank: int) -> Path:
        paths = self.ensure_job_layout(job_id)
        return paths.logs_dir / f"rank_{rank}.log"

    def rank_tb_dir(self, job_id: str, rank: int) -> Path:
        paths = self.ensure_job_layout(job_id)
        tb_dir = paths.tensorboard_dir / f"rank{rank}"
        tb_dir.mkdir(parents=True, exist_ok=True)
        return tb_dir

    async def sync_tensorboard(
        self, 
        job_id: str, 
        workers: list[dict[str, Any]], 
        ssh_clients: dict[str, SSHClient]
    ) -> bool:
        """Collect TensorBoard events from all workers."""
        paths = self.ensure_job_layout(job_id)
        success = True
        
        for worker in workers:
            node_name = worker.get("node_name", worker.get("host"))
            if not node_name or node_name not in ssh_clients:
                continue
                
            client = ssh_clients[node_name]
            # Assumes MetricsEmitter path structure: ~/ugro_data/jobs/{job_id}/tensorboard/
            remote_tb_base = f"{worker.get('home_dir', '~')}/ugro_data/jobs/{job_id}/tensorboard/"
            local_tb_base = paths.tensorboard_dir
            
            # Pull everything under tensorboard/ for this job
            res = await client.pull_dir_async(
                remote_path=remote_tb_base, 
                local_path=str(local_tb_base),
                include_pattern="*.tfevents*"
            )
            if not res:
                success = False
        return success

    async def sync_logs(
        self,
        job_id: str,
        workers: list[dict[str, Any]],
        ssh_clients: dict[str, SSHClient]
    ) -> bool:
        """Collect training logs (rank logs) from all workers."""
        paths = self.ensure_job_layout(job_id)
        success = True
        
        for worker in workers:
            node_name = worker.get("node_name", worker.get("host"))
            if not node_name or node_name not in ssh_clients:
                continue
                
            client = ssh_clients[node_name]
            remote_logs_dir = f"{worker.get('home_dir', '~')}/ugro_data/jobs/{job_id}/logs/"
            local_logs_dir = paths.logs_dir
            
            res = await client.pull_dir_async(
                remote_path=remote_logs_dir,
                local_path=str(local_logs_dir)
            )
            if not res:
                success = False
        return success

    async def sync_rank_metrics(
        self,
        job_id: str,
        workers: list[dict[str, Any]],
        ssh_clients: dict[str, SSHClient]
    ) -> bool:
        """Poll and pull rank-specific metrics files from all workers."""
        paths = self.ensure_job_layout(job_id)
        success = True
        
        for worker in workers:
            node_name = worker.get("node_name", worker.get("host"))
            if not node_name or node_name not in ssh_clients:
                continue
                
            client = ssh_clients[node_name]
            # Pull rank-specific metrics from worker's ugro_data/jobs/{job_id}/
            remote_job_dir = f"{worker.get('home_dir', '~')}/ugro_data/jobs/{job_id}/"
            local_job_dir = paths.job_dir
            
            res = await client.pull_dir_async(
                remote_path=remote_job_dir,
                local_path=str(local_job_dir),
                include_pattern="metrics_rank*.jsonl"
            )
            if not res:
                success = False
        
        # Consolidation is now handled by the TrainingMetricsCollector to avoid duplication
        return success

    def _consolidate_rank_metrics(self, job_id: str, latest_entries: list[dict[str, Any]]) -> None:
        """Append latest entries from rank-specific files to the main metrics.jsonl."""
        if not latest_entries:
            return
            
        # Append to main metrics.jsonl
        for entry in latest_entries:
            self.append_metrics(job_id, entry)

    def get_job_summary(self, job_id: str) -> dict[str, Any]:
        """Parse metrics.jsonl and return aggregated stats."""
        paths = self.ensure_job_layout(job_id)
        summary: dict[str, Any] = {
            "job_id": job_id,
            "status": "unknown",
            "total_steps": 0,
            "final_loss": None,
            "avg_throughput": 0.0,
            "duration": 0.0,
            "per_rank_stats": {}
        }
        
        if not paths.metrics_jsonl.exists():
            return summary
            
        try:
            with open(paths.metrics_jsonl, "r", encoding="utf-8") as f:
                lines = [json.loads(line) for line in f if line.strip()]
                
            if not lines:
                return summary
                
            # Sort by step
            lines.sort(key=lambda x: x.get("step", 0))
            
            summary["total_steps"] = lines[-1].get("step", 0)
            summary["final_loss"] = lines[-1].get("training_loss")
            
            # Throughput average
            throughputs = [l.get("throughput_tokens_sec", 0) for l in lines if "throughput_tokens_sec" in l]
            if throughputs:
                summary["avg_throughput"] = sum(throughputs) / len(throughputs)
                
            # Duration
            timestamps = [l.get("timestamp", 0) for l in lines if "timestamp" in l]
            if timestamps:
                summary["duration"] = max(timestamps) - min(timestamps)
                
            # Per rank (latest)
            for line in lines:
                rank = line.get("rank", 0)
                summary["per_rank_stats"][f"rank{rank}"] = {
                    "step": line.get("step"),
                    "loss": line.get("training_loss"),
                    "gpu_util": line.get("gpu_util"),
                    "gpu_mem_used_gb": line.get("gpu_mem_used_gb")
                }
                
            summary["status"] = "running"
            
        except Exception:
            pass
            
        return summary
