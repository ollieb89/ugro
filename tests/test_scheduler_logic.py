
import pytest
from ugro.scheduler.resources import ResourceTracker
from ugro.queues.models import JobResources

# Mock the requests library to avoid actual network calls
import requests
from unittest.mock import Mock, patch

@pytest.fixture
def mock_cluster_config():
    return {
        "nodes": {
            "node_small": {"vram_gb": 8, "gpu_count": 1},
            "node_med": {"vram_gb": 16, "gpu_count": 1},
            "node_large": {"vram_gb": 24, "gpu_count": 2},
        }
    }

def test_best_fit_bin_packing(mock_cluster_config):
    tracker = ResourceTracker(cluster_config=mock_cluster_config)
    
    # Simulate usage
    # node_small: 2GB used (6 free)
    # node_med: 2GB used (14 free)
    # node_large: 2GB used (22 free -> or 46 if total? Tracker treats single pool per node currently)
    
    tracker.update_usage("node_small", 2, 0)
    tracker.update_usage("node_med", 2, 0)
    tracker.update_usage("node_large", 2, 0)
    
    # Case 1: Job needs 4GB.
    # node_small free: 6GB -> Remaining 2GB (Tight fit!)
    # node_med free: 14GB -> Remaining 10GB
    # node_large free: ~22GB/46GB -> Huge
    # Best fit should return node_small first.
    
    job_res = JobResources(min_vram_gb=4, gpu_count=1)
    
    nodes = tracker.can_fit(job_res)
    assert len(nodes) > 0
    assert nodes[0] == "node_small" # Best fit (tightest)
    
    # Case 2: Job needs 10GB.
    # node_small free 6GB -> Can't fit
    # node_med free 14GB -> 4GB remaining
    # node_large free 22GB+ -> lots remaining
    # Best fit should be node_med.
    
    job_res_big = JobResources(min_vram_gb=10, gpu_count=1)
    nodes = tracker.can_fit(job_res_big)
    assert nodes[0] == "node_med"
    assert "node_small" not in nodes

def test_gpu_count_constraint(mock_cluster_config):
    tracker = ResourceTracker(cluster_config=mock_cluster_config)
    
    # node_large has 2 GPUs. Others 1.
    # If job needs 2 GPUs, only node_large should fit (assuming it's free).
    
    job_res_dual = JobResources(min_vram_gb=4, gpu_count=2)
    nodes = tracker.can_fit(job_res_dual)
    
    assert nodes == ["node_large"]

