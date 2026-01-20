"""Cluster state management for UGRO."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_STATE_FILE = Path.home() / ".ugro" / "cluster_state.json"
ENV_STATE_FILE = "UGRO_STATE_FILE"


@dataclass(slots=True)
class NodeState:
    """State for a single cluster node."""

    ip: str
    gpu: str
    vram_gb: int
    status: str = "available"
    running_job_id: str | None = None
    health_score: float = 100.0
    last_check: str | None = None


@dataclass(slots=True)
class JobState:
    """State for a training job."""

    status: str
    ranks: list[int]
    model: str
    started_at: str
    gpu_nodes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ClusterState:
    """Persisted cluster state."""

    nodes: dict[str, NodeState] = field(default_factory=dict)
    jobs: dict[str, JobState] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize cluster state to a JSON-ready dictionary."""
        return {
            "nodes": {name: asdict(node) for name, node in self.nodes.items()},
            "jobs": {job_id: asdict(job) for job_id, job in self.jobs.items()},
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ClusterState":
        """Load cluster state from a dictionary."""
        nodes = {
            name: NodeState(**node_payload)
            for name, node_payload in payload.get("nodes", {}).items()
        }
        jobs = {
            job_id: JobState(**job_payload)
            for job_id, job_payload in payload.get("jobs", {}).items()
        }
        return cls(nodes=nodes, jobs=jobs)


class ClusterStateManager:
    """Manages persisted cluster and job state."""

    def __init__(self, state_file: Path | None = None):
        resolved_state_file = state_file or Path(
            os.environ.get(ENV_STATE_FILE, DEFAULT_STATE_FILE)
        )
        self.state_file = Path(resolved_state_file)
        self.state: ClusterState = ClusterState()

    def load(self) -> ClusterState:
        """Load cluster state from disk."""
        if not self.state_file.exists():
            self.state = ClusterState()
            return self.state

        try:
            with self.state_file.open("r", encoding="utf-8") as state_file:
                payload = json.load(state_file)
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Failed to load cluster state from {self.state_file}"
            ) from exc

        self.state = ClusterState.from_dict(payload)
        return self.state

    def save(self) -> None:
        """Persist cluster state to disk."""
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        payload = self.state.to_dict()
        temp_path = self.state_file.with_suffix(".tmp")
        try:
            with temp_path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
            temp_path.replace(self.state_file)
        except OSError as exc:
            raise RuntimeError(
                f"Failed to write cluster state to {self.state_file}"
            ) from exc

    def refresh(self) -> ClusterState:
        """Load the latest state from disk and return it."""
        return self.load()

    def get_state(self) -> ClusterState:
        """Return the in-memory state."""
        return self.state

    def set_node(self, name: str, node_state: NodeState) -> None:
        """Set or update a node state."""
        self.state.nodes[name] = node_state
        self.save()

    def update_node_status(
        self,
        name: str,
        status: str,
        running_job_id: str | None = None,
        health_score: float | None = None,
        last_check: str | None = None,
    ) -> None:
        """Update node status and health metrics."""
        node = self.state.nodes.get(name)
        if node is None:
            raise KeyError(f"Node {name} not found")
        node.status = status
        node.running_job_id = running_job_id
        if health_score is not None:
            node.health_score = health_score
        if last_check is not None:
            node.last_check = last_check
        self.save()

    def remove_node(self, name: str) -> None:
        """Remove a node from state."""
        if name in self.state.nodes:
            self.state.nodes.pop(name)
            self.save()

    def set_job(self, job_id: str, job_state: JobState) -> None:
        """Set or update a job state."""
        self.state.jobs[job_id] = job_state
        self.save()

    def update_job_status(self, job_id: str, status: str) -> None:
        """Update job status."""
        job = self.state.jobs.get(job_id)
        if job is None:
            raise KeyError(f"Job {job_id} not found")
        job.status = status
        self.save()

    def remove_job(self, job_id: str) -> None:
        """Remove job from state."""
        if job_id in self.state.jobs:
            self.state.jobs.pop(job_id)
            self.save()

    @staticmethod
    def build_job_state(
        status: str,
        ranks: list[int],
        model: str,
        gpu_nodes: list[str],
    ) -> JobState:
        """Create a job state with an ISO timestamp."""
        return JobState(
            status=status,
            ranks=ranks,
            model=model,
            started_at=datetime.now().isoformat(),
            gpu_nodes=gpu_nodes,
        )
