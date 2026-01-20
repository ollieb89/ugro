"""
Tests for Gang Scheduling.

Tests cover:
- Multi-node allocation with can_fit_gang()
- Scheduler gang scheduling path
- Resource tracking across gang nodes
"""

import pytest
from unittest.mock import MagicMock
from ugro.scheduler import Scheduler, ResourceTracker
from ugro.queues import Job, JobResources, JobStatus, JobPriority


class MockQueueWithClaim:
    """Mock queue supporting list_jobs and claim."""
    
    def __init__(self):
        self.jobs = []
    
    def list_jobs(self, status=None, limit=100):
        if status == JobStatus.PENDING:
            return [j for j in self.jobs if j.status == JobStatus.PENDING][:limit]
        return self.jobs[:limit]
    
    def claim(self, job_id):
        for i, job in enumerate(self.jobs):
            if job.id == job_id and job.status == JobStatus.PENDING:
                job.status = JobStatus.RUNNING
                return job
        return None
    
    def update_job(self, job):
        pass


@pytest.fixture
def cluster_config_3_nodes():
    """Cluster config with 3 nodes."""
    return {
        "nodes": {
            "node1": {"vram_gb": 8, "gpu_count": 1},
            "node2": {"vram_gb": 8, "gpu_count": 1},
            "node3": {"vram_gb": 8, "gpu_count": 1},
        }
    }


@pytest.fixture
def cluster_config_5_nodes():
    """Cluster config with 5 nodes of varying sizes."""
    return {
        "nodes": {
            "small_1": {"vram_gb": 8, "gpu_count": 1},
            "small_2": {"vram_gb": 8, "gpu_count": 1},
            "medium": {"vram_gb": 16, "gpu_count": 1},
            "large_1": {"vram_gb": 24, "gpu_count": 2},
            "large_2": {"vram_gb": 24, "gpu_count": 2},
        }
    }


class TestCanFitGang:
    """Tests for ResourceTracker.can_fit_gang()"""
    
    def test_gang_3_nodes_success(self, cluster_config_3_nodes):
        """Test gang allocation when exactly 3 nodes are available."""
        tracker = ResourceTracker(cluster_config=cluster_config_3_nodes)
        
        job_res = JobResources(gpu_count=1, min_vram_gb=4, nnodes=3)
        nodes = tracker.can_fit_gang(job_res)
        
        assert nodes is not None
        assert len(nodes) == 3
        assert set(nodes) == {"node1", "node2", "node3"}
    
    def test_gang_insufficient_nodes(self, cluster_config_3_nodes):
        """Test gang allocation fails when not enough nodes."""
        tracker = ResourceTracker(cluster_config=cluster_config_3_nodes)
        
        # Request 5 nodes when only 3 exist
        job_res = JobResources(gpu_count=1, min_vram_gb=4, nnodes=5)
        nodes = tracker.can_fit_gang(job_res)
        
        assert nodes is None
    
    def test_gang_partial_resources(self, cluster_config_3_nodes):
        """Test gang fails when some nodes don't have resources."""
        tracker = ResourceTracker(cluster_config=cluster_config_3_nodes)
        
        # Use up node1
        tracker.update_usage("node1", used_vram=8, used_gpu_count=1)
        
        # Request 3 nodes - should fail since only 2 available
        job_res = JobResources(gpu_count=1, min_vram_gb=4, nnodes=3)
        nodes = tracker.can_fit_gang(job_res)
        
        assert nodes is None
    
    def test_gang_single_node_fallback(self, cluster_config_3_nodes):
        """Test nnodes=1 falls back to single-node scheduling."""
        tracker = ResourceTracker(cluster_config=cluster_config_3_nodes)
        
        job_res = JobResources(gpu_count=1, min_vram_gb=4, nnodes=1)
        nodes = tracker.can_fit_gang(job_res)
        
        assert nodes is not None
        assert len(nodes) == 1
    
    def test_gang_bfd_ordering(self, cluster_config_5_nodes):
        """Test gang uses BFD to select best-fit nodes."""
        tracker = ResourceTracker(cluster_config=cluster_config_5_nodes)
        
        # Request 2 nodes needing 6GB each
        job_res = JobResources(gpu_count=1, min_vram_gb=6, nnodes=2)
        nodes = tracker.can_fit_gang(job_res)
        
        assert nodes is not None
        assert len(nodes) == 2
        # Should select small nodes (tightest fit: 8-6=2GB remaining)
        assert "small_1" in nodes or "small_2" in nodes


class TestSchedulerGangScheduling:
    """Tests for Scheduler.schedule_next() with gang scheduling."""
    
    def test_gang_job_allocates_all_nodes(self, cluster_config_3_nodes):
        """Test scheduler allocates all nodes for gang job."""
        queue = MockQueueWithClaim()
        tracker = ResourceTracker(cluster_config=cluster_config_3_nodes)
        scheduler = Scheduler(queue, tracker)
        
        # Create gang job requiring 3 nodes
        job_res = JobResources(gpu_count=1, min_vram_gb=4, nnodes=3)
        job = Job(name="gang_job", command="torchrun --nnodes=3", resources=job_res)
        queue.jobs.append(job)
        
        # Schedule
        scheduled = scheduler.schedule_next()
        
        assert scheduled is not None
        assert scheduled.id == job.id
        assert scheduled.worker_ids is not None
        assert len(scheduled.worker_ids) == 3
        assert scheduled.worker_id == scheduled.worker_ids[0]  # Primary node
    
    def test_gang_job_reserves_all_resources(self, cluster_config_3_nodes):
        """Test gang scheduling reserves resources on all gang nodes."""
        queue = MockQueueWithClaim()
        tracker = ResourceTracker(cluster_config=cluster_config_3_nodes)
        scheduler = Scheduler(queue, tracker)
        
        job_res = JobResources(gpu_count=1, min_vram_gb=4, nnodes=3)
        job = Job(name="gang_job", command="train.py", resources=job_res)
        queue.jobs.append(job)
        
        scheduler.schedule_next()
        
        # Check all nodes have usage recorded
        for node in ["node1", "node2", "node3"]:
            assert node in tracker.node_usage
            assert tracker.node_usage[node]["vram_gb"] == 4
            assert tracker.node_usage[node]["gpu_count"] == 1
    
    def test_gang_backfilling_smaller_job(self, cluster_config_3_nodes):
        """Test smaller job gets scheduled when gang can't fit."""
        queue = MockQueueWithClaim()
        tracker = ResourceTracker(cluster_config=cluster_config_3_nodes)
        
        # Use 2 nodes, leaving only 1 free
        tracker.update_usage("node1", used_vram=8, used_gpu_count=1)
        tracker.update_usage("node2", used_vram=8, used_gpu_count=1)
        
        scheduler = Scheduler(queue, tracker)
        
        # Gang job can't fit (needs 3)
        gang_res = JobResources(gpu_count=1, min_vram_gb=4, nnodes=3)
        gang_job = Job(name="gang_job", command="torchrun", resources=gang_res)
        
        # Single job CAN fit
        single_res = JobResources(gpu_count=1, min_vram_gb=4, nnodes=1)
        single_job = Job(name="single_job", command="python train.py", resources=single_res)
        
        queue.jobs.append(gang_job)  # Gang first in queue
        queue.jobs.append(single_job)  # Single second
        
        # Backfilling should skip gang and schedule single
        scheduled = scheduler.schedule_next()
        
        assert scheduled is not None
        assert scheduled.id == single_job.id
        assert scheduled.worker_id == "node3"


class TestReleaseUsage:
    """Tests for ResourceTracker.release_usage()"""
    
    def test_release_frees_resources(self, cluster_config_3_nodes):
        """Test release_usage frees resources."""
        tracker = ResourceTracker(cluster_config=cluster_config_3_nodes)
        
        tracker.update_usage("node1", used_vram=4, used_gpu_count=1)
        assert tracker.node_usage["node1"]["vram_gb"] == 4
        
        tracker.release_usage("node1", vram_gb=4, gpu_count=1)
        assert tracker.node_usage["node1"]["vram_gb"] == 0
        assert tracker.node_usage["node1"]["gpu_count"] == 0
    
    def test_release_clamps_to_zero(self, cluster_config_3_nodes):
        """Test release doesn't go negative."""
        tracker = ResourceTracker(cluster_config=cluster_config_3_nodes)
        
        tracker.update_usage("node1", used_vram=2, used_gpu_count=1)
        
        # Release more than used
        tracker.release_usage("node1", vram_gb=10, gpu_count=5)
        
        assert tracker.node_usage["node1"]["vram_gb"] == 0
        assert tracker.node_usage["node1"]["gpu_count"] == 0
