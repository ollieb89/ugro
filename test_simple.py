"""Simple test for Launch Coordinator without pytest dependency."""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

# Add src to path
import sys
sys.path.append('src')

from ugro.launch_coordinator import LaunchCoordinator
from ugro.job import Job


async def test_launch_coordinator():
    """Simple integration test for LaunchCoordinator."""
    print("🧪 Testing Launch Coordinator...")
    
    # Create mock cluster
    cluster = MagicMock()
    cluster.config = {
        'master': {'ip': '192.168.1.100'},
        'communication': {'master_port': 29500},
    }
    cluster.ssh_clients = {
        'gpu1': MagicMock(),
        'gpu2': MagicMock(),
    }
    cluster.check_health.return_value = {
        'gpu1': {'healthy': True, 'message': 'Healthy', 'gpu_model': 'RTX 4070'},
        'gpu2': {'healthy': True, 'message': 'Healthy', 'gpu_model': 'RTX 3070 Ti'},
    }
    
    # Mock SSH responses
    for ssh_client in cluster.ssh_clients.values():
        ssh_client.run_command.return_value = (True, "torchrun started", "")
    
    # Create coordinator
    coordinator = LaunchCoordinator(cluster)
    
    # Test 1: Cluster validation
    print("📋 Test 1: Cluster validation")
    success, results = await coordinator.validate_cluster_state()
    assert success is True, "Cluster validation should succeed"
    assert len(results['healthy_nodes']) == 2, "Should have 2 healthy nodes"
    print("✓ Cluster validation passed")
    
    # Test 2: Resource allocation
    print("📋 Test 2: Resource allocation")
    workers = [
        {'name': 'gpu1', 'rank': 1, 'ip': '192.168.1.101', 'hardware': {'gpu_model': 'RTX 4070', 'vram_gb': 8}, 'user': 'ob'},
        {'name': 'gpu2', 'rank': 2, 'ip': '192.168.1.102', 'hardware': {'gpu_model': 'RTX 3070 Ti', 'vram_gb': 8}, 'user': 'ollie'},
    ]
    allocation_plan = await coordinator.allocate_resources(workers)
    assert allocation_plan['total_nodes'] == 2, "Should allocate 2 nodes"
    assert allocation_plan['world_size'] == 2, "World size should be 2"
    assert len(allocation_plan['rank_assignments']) == 2, "Should have 2 rank assignments"
    print("✓ Resource allocation passed")
    
    # Test 3: Command building
    print("📋 Test 3: Command building")
    with tempfile.TemporaryDirectory() as temp_dir:
        job = Job(
            name="test_job",
            model="tinyllama",
            dataset="wikitext",
            epochs=1,
            learning_rate=0.0002,
            results_dir=Path(temp_dir),
        )
        
        command = coordinator._build_torchrun_command(
            allocation_plan=allocation_plan,
            training_script="scripts/train_production.py",
            job=job,
        )
        
        # Verify command components
        assert 'torchrun' in command, "Command should contain torchrun"
        assert '--nproc_per_node=1' in command, "Should set nproc_per_node to 1"
        assert '--nnodes=2' in command, "Should set nnodes to 2"
        assert '--node_rank=${RANK}' in command, "Should have rank placeholder"
        assert '--master_addr=192.168.1.100' in command, "Should set master address"
        assert '--master_port=29500' in command, "Should set master port"
        assert 'scripts/train_production.py' in command, "Should include training script"
        assert '--model tinyllama' in command, "Should include model parameter"
        assert '--dataset wikitext' in command, "Should include dataset parameter"
        assert '--epochs 1' in command, "Should include epochs parameter"
        assert '--learning-rate 0.0002' in command, "Should include learning rate parameter"
        
        print("✓ Command building passed")
        print(f"   Generated command: {command[:100]}...")
    
    # Test 4: Master address and port
    print("📋 Test 4: Configuration retrieval")
    master_addr = coordinator._get_master_address()
    master_port = coordinator._get_master_port()
    assert master_addr == '192.168.1.100', "Master address should match config"
    assert master_port == 29500, "Master port should match config"
    print("✓ Configuration retrieval passed")
    
    print("\n✅ All tests passed!")
    print("🎉 Launch Coordinator implementation is working correctly!")


if __name__ == "__main__":
    asyncio.run(test_launch_coordinator())