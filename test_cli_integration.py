#!/usr/bin/env python3
"""Test CLI integration with Launch Coordinator."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add src to path
sys.path.append('src')

from ugro.agent import UGROAgent
from ugro.config import load_config


def test_cli_integration():
    """Test CLI integration with Launch Coordinator."""
    print("🧪 Testing CLI Integration with Launch Coordinator...")
    
    # Create a mock config for testing
    mock_config = {
        'master': {'ip': '192.168.1.100', 'hostname': 'gpu-master'},
        'communication': {'master_port': 29500},
        'workers': [
            {
                'name': 'gpu1',
                'hostname': 'gpu1',
                'ip': '192.168.1.101',
                'user': 'ob',
                'ssh_port': 22,
                'rank': 1,
                'hardware': {'gpu_model': 'RTX 4070', 'vram_gb': 8},
                'paths': {'home': '/home/ob', 'project': '${HOME}/Development/Tools/ugro'}
            },
            {
                'name': 'gpu2',
                'hostname': 'gpu2',
                'ip': '192.168.1.102',
                'user': 'ollie',
                'ssh_port': 22,
                'rank': 2,
                'hardware': {'gpu_model': 'RTX 3070 Ti', 'vram_gb': 8},
                'paths': {'home': '${HOME}', 'project': '${HOME}/Development/Tools/ugro'}
            }
        ]
    }
    
    # Test 1: Agent initialization
    print("📋 Test 1: Agent initialization with Launch Coordinator")
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            agent = UGROAgent(config=mock_config)
            
            # Verify LaunchCoordinator is initialized
            assert hasattr(agent, 'launch_coordinator'), "Agent should have launch_coordinator"
            assert agent.launch_coordinator is not None, "LaunchCoordinator should be initialized"
            print("✓ Agent initialization passed")
            
            # Test 2: Cluster health check
            print("📋 Test 2: Cluster health check")
            health = agent.check_cluster_health()
            assert 'master' in health, "Should have master node in health check"
            assert 'gpu1' in health, "Should have gpu1 in health check"
            assert 'gpu2' in health, "Should have gpu2 in health check"
            print("✓ Cluster health check passed")
            
            # Test 3: Launch training (dry run with mocked SSH)
            print("📋 Test 3: Launch training with Launch Coordinator")
            
            # Mock SSH clients to return success
            for ssh_client in agent.cluster.ssh_clients.values():
                ssh_client.test_connection.return_value = True
                ssh_client.get_gpu_info.return_value = (True, {
                    'name': 'RTX 4070',
                    'memory_total': 8192,
                    'memory_used': 0,
                    'utilization': 0
                })
                ssh_client.check_python_environment.return_value = (True, {
                    'python_version': '3.11',
                    'pytorch_version': '2.1.0',
                    'cuda': True
                })
                ssh_client.run_command.return_value = (True, "torchrun started", "")
            
            # Mock the launch coordinator methods to avoid actual SSH calls
            with patch.object(agent.launch_coordinator, 'validate_cluster_state') as mock_validate, \
                 patch.object(agent.launch_coordinator, 'allocate_resources') as mock_allocate, \
                 patch.object(agent.launch_coordinator, 'launch_distributed_training') as mock_launch:
                
                # Setup mock returns
                mock_validate.return_value = (True, {
                    'healthy_nodes': [
                        {'name': 'gpu1', 'gpu_model': 'RTX 4070'},
                        {'name': 'gpu2', 'gpu_model': 'RTX 3070 Ti'}
                    ],
                    'unhealthy_nodes': []
                })
                
                mock_allocate.return_value = {
                    'total_nodes': 2,
                    'world_size': 2,
                    'rank_assignments': {
                        'gpu1': {'rank': 1, 'ip': '192.168.1.101'},
                        'gpu2': {'rank': 2, 'ip': '192.168.1.102'}
                    },
                    'master_addr': '192.168.1.100',
                    'master_port': 29500
                }
                
                mock_launch.return_value = True
                
                # Test launch
                success = agent.launch_training(
                    job_name="test_job",
                    model="tinyllama",
                    dataset="wikitext",
                    epochs=1,
                    learning_rate=0.0002,
                    verbose=True
                )
                
                assert success is True, "Launch should succeed"
                print("✓ Launch training passed")
                
                # Verify methods were called
                mock_validate.assert_called_once()
                mock_allocate.assert_called_once()
                mock_launch.assert_called_once()
                print("✓ Launch Coordinator methods called correctly")
    
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n✅ All CLI integration tests passed!")
    print("🎉 Launch Coordinator is properly integrated with CLI!")
    return True


if __name__ == "__main__":
    success = test_cli_integration()
    sys.exit(0 if success else 1)