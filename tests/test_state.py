# tests/test_state.py
import sys
from pathlib import Path
import tempfile
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_cluster_state_manager():
    """Test cluster state persistence and updates"""
    from ugro.cluster_state import ClusterStateManager, NodeState, JobState
    
    # Use a temporary file for testing
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_file = Path(f.name)
    
    try:
        manager = ClusterStateManager(state_file=temp_file)
        
        # Test initial state
        state = manager.load()
        assert hasattr(state, 'nodes')
        assert hasattr(state, 'jobs')
        assert len(state.nodes) == 0
        assert len(state.jobs) == 0
        
        # Test node registration
        manager.register_node('gpu-master', NodeState(
            ip='192.168.1.100',
            gpu='RTX 5070 Ti',
            vram_gb=12
        ))
        
        # Test job registration
        manager.register_job('test_job', JobState(
            status='running',
            ranks=[0, 1, 2],
            model='test_model',
            started_at='2026-01-21T12:00:00Z',
            gpu_nodes=['gpu-master', 'gpu1', 'gpu2']
        ))
        
        # Verify state was saved
        updated_state = manager.load()
        assert 'gpu-master' in updated_state.nodes
        assert updated_state.nodes['gpu-master'].gpu == 'RTX 5070 Ti'
        assert 'test_job' in updated_state.jobs
        assert updated_state.jobs['test_job'].status == 'running'
        
    finally:
        # Clean up
        if temp_file.exists():
            temp_file.unlink()
