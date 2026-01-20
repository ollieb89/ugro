import json
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

from ugro.result_aggregator import ResultAggregator
from ugro.cluster_state import ClusterStateManager
from apps.api.schemas.job import JobSummary, MetricPoint

logger = logging.getLogger(__name__)

class JobService:
    def __init__(self):
        self.aggregator = ResultAggregator()
        self.state_manager = ClusterStateManager()

    async def list_jobs(self, limit: int = 50, offset: int = 0) -> List[JobSummary]:
        """List all jobs merging cluster state and on-disk results."""
        # 1. Get state from ClusterManager (source of truth for status)
        state = self.state_manager.refresh()
        jobs_state = state.jobs

        # 2. Convert to JobSummary list
        summaries = []
        
        # Sort by started_at desc
        sorted_job_ids = sorted(
            jobs_state.keys(),
            key=lambda jid: jobs_state[jid].started_at,
            reverse=True
        )
        
        # Apply pagination on IDs
        paged_ids = sorted_job_ids[offset : offset + limit]

        for job_id in paged_ids:
            job_state = jobs_state[job_id]
            
            # Get aggregated metrics summary from disk
            # ResultAggregator.get_job_summary parses the JSONL file
            disk_summary = self.aggregator.get_job_summary(job_id)
            
            # Combine info
            # Handle timestamps carefully
            try:
                started_at = datetime.fromisoformat(job_state.started_at)
            except ValueError:
                started_at = datetime.now() # Fallback

            summary = JobSummary(
                job_id=job_id,
                status=job_state.status,
                model=job_state.model,
                # We need to fetch dataset from config.json if not in state, 
                # but state is faster. For now, use placeholder or fetch if critical.
                # Simplification: we might need to update ClusterState to include dataset.
                dataset="unknown", 
                started_at=started_at,
                duration_seconds=disk_summary.get("duration"),
                total_steps=disk_summary.get("total_steps", 0),
                final_loss=disk_summary.get("final_loss"),
                avg_throughput=disk_summary.get("avg_throughput"),
                gpu_nodes=job_state.gpu_nodes
            )
            summaries.append(summary)

        return summaries

    async def get_job_details(self, job_id: str) -> Optional[JobSummary]:
        """Get detailed summary for a single job."""
        state = self.state_manager.refresh()
        if job_id not in state.jobs:
            return None
            
        job_state = state.jobs[job_id]
        disk_summary = self.aggregator.get_job_summary(job_id)
        
        # Determine dataset from config if possible
        dataset = "unknown"
        paths = self.aggregator.ensure_job_layout(job_id)
        if paths.config_json.exists():
            try:
                with open(paths.config_json) as f:
                    config = json.load(f)
                    dataset = config.get("dataset_name", "unknown")
            except Exception:
                pass

        try:
            started_at = datetime.fromisoformat(job_state.started_at)
        except ValueError:
            started_at = datetime.now()

        return JobSummary(
            job_id=job_id,
            status=job_state.status,
            model=job_state.model,
            dataset=dataset,
            started_at=started_at,
            duration_seconds=disk_summary.get("duration"),
            total_steps=disk_summary.get("total_steps", 0),
            final_loss=disk_summary.get("final_loss"),
            avg_throughput=disk_summary.get("avg_throughput"),
            gpu_nodes=job_state.gpu_nodes,
            metrics_summary=disk_summary.get("per_rank_stats")
        )

    async def get_job_metrics(self, job_id: str) -> List[MetricPoint]:
        """Get full time-series metrics for a job."""
        paths = self.aggregator.ensure_job_layout(job_id)
        if not paths.metrics_jsonl.exists():
            return []
            
        metrics = []
        try:
            with open(paths.metrics_jsonl, "r") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)
                        # We only want one entry per step usually, or we organize by rank?
                        # For simple graphing, we might just aggregate or filter by rank 0.
                        # Let's filter for rank 0 for the main graph for now to avoid noise.
                        if data.get("rank") == 0:
                            metrics.append(MetricPoint(
                                timestamp=data.get("timestamp"),
                                step=data.get("step"),
                                loss=data.get("training_loss"),
                                learning_rate=data.get("learning_rate"),
                                throughput=data.get("throughput_tokens_sec"),
                                gradient_norm=data.get("gradient_norm"),
                                gpu_stats={
                                    "util": data.get("gpu_util", 0),
                                    "mem": data.get("gpu_mem_used_gb", 0)
                                }
                            ))
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logger.error(f"Error reading metrics for {job_id}: {e}")
            
        return metrics
