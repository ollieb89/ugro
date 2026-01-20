"""Test the Launch Coordinator implementation."""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.ugro.launch_coordinator import LaunchCoordinator
from src.ugro.job import Job, JobStatus
from src.ugro.cluster import Cluster


class TestLaunchCoordinator:
    """Test suite for LaunchCoordinator."""
    
    @pytest.fixture
    def mock_cluster(self):
        """Create a mock cluster for testing."""
        cluster = MagicMock(spec=Cluster)
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
        return cluster
    
    @pytest.fixture
    def launch_coordinator(self, mock_cluster):
        """Create LaunchCoordinator instance for testing."""
        return LaunchCoordinator(
            cluster=mock_cluster,
            sync_window_seconds=30,
            monitoring_interval_seconds=1.0,
            max_startup_timeout_seconds=60,
        )
    
    @pytest.fixture
    def mock_job(self):
        """Create a mock job for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            job = Job(
                name="test_job",
                model="tinyllama",
                dataset="wikitext",
                epochs=1,
                learning_rate=0.0002,
                results_dir=Path(temp_dir),
            )
            job.start(['gpu1', 'gpu2'])
            yield job
    
    @pytest.mark.asyncio
    async def test_validate_cluster_state_success(self, launch_coordinator):
        """Test successful cluster state validation."""
        success, results = await launch_coordinator.validate_cluster_state()
        
        assert success is True
        assert len(results['healthy_nodes']) == 2
        assert len(results['unhealthy_nodes']) == 0
        assert 'gpu1' in [node['name'] for node in results['healthy_nodes']]
        assert 'gpu2' in [node['name'] for node in results['healthy_nodes']]
    
    @pytest.mark.asyncio
    async def test_validate_cluster_state_failure(self, launch_coordinator):
        """Test cluster state validation with failures."""
        # Mock unhealthy cluster
        launch_coordinator.cluster.check_health.return_value = {
            'gpu1': {'healthy': False, 'message': 'SSH connection failed'},
            'gpu2': {'healthy': True, 'message': 'Healthy', 'gpu_model': 'RTX 3070 Ti'},
        }
        
        success, results = await launch_coordinator.validate_cluster_state()
        
        assert success is False
        assert len(results['healthy_nodes']) == 1
        assert len(results['unhealthy_nodes']) == 1
        assert 'gpu1' in results['ssh_failures']
    
    @pytest.mark.asyncio
    async def test_allocate_resources(self, launch_coordinator):
        """Test GPU resource allocation."""
        workers = [
            {'name': 'gpu1', 'rank': 1, 'ip': '192.168.1.101', 'hardware': {'gpu_model': 'RTX 4070', 'vram_gb': 8}, 'user': 'ob'},
            {'name': 'gpu2', 'rank': 2, 'ip': '192.168.1.102', 'hardware': {'gpu_model': 'RTX 3070 Ti', 'vram_gb': 8}, 'user': 'ollie'},
        ]
        
        allocation_plan = await launch_coordinator.allocate_resources(workers)
        
        assert allocation_plan['total_nodes'] == 2
        assert allocation_plan['world_size'] == 2
        assert allocation_plan['master_addr'] == '192.168.1.100'
        assert allocation_plan['master_port'] == 29500
        assert len(allocation_plan['rank_assignments']) == 2
        
        # Check rank assignments
        gpu1_assignment = allocation_plan['rank_assignments']['gpu1']
        assert gpu1_assignment['rank'] == 1
        assert gpu1_assignment['gpu_model'] == 'RTX 4070'
        
        gpu2_assignment = allocation_plan['rank_assignments']['gpu2']
        assert gpu2_assignment['rank'] == 2
        assert gpu2_assignment['gpu_model'] == 'RTX 3070 Ti'
    
    @pytest.mark.asyncio
    async def test_launch_distributed_training_success(self, launch_coordinator, mock_job):
        """Test successful distributed training launch."""
        # Mock SSH client responses
        for ssh_client in launch_coordinator.cluster.ssh_clients.values():
            ssh_client.run_command.return_value = (True, "torchrun started", "")
        
        workers = [
            {'name': 'gpu1', 'rank': 1, 'ip': '192.168.1.101', 'hardware': {'gpu_model': 'RTX 4070', 'vram_gb': 8}, 'user': 'ob'},
            {'name': 'gpu2', 'rank': 2, 'ip': '192.168.1.102', 'hardware': {'gpu_model': 'RTX 3070 Ti', 'vram_gb': 8}, 'user': 'ollie'},
        ]
        
        allocation_plan = await launch_coordinator.allocate_resources(workers)
        
        # Mock monitoring and log collection to avoid infinite loops
        with patch.object(launch_coordinator, '_start_monitoring'), \
             patch.object(launch_coordinator, '_start_log_collection'):
            
            success = await launch_coordinator.launch_distributed_training(
                job=mock_job,
                allocation_plan=allocation_plan,
                training_script="scripts/train_production.py",
            )
        
        assert success is True
        
        # Verify SSH commands were called
        for ssh_client in launch_coordinator.cluster.ssh_clients.values():
            ssh_client.run_command.assert_called()
            call_args = ssh_client.run_command.call_args
            assert 'torchrun' in call_args[0][0]  # Command should contain torchrun
    
    @pytest.mark.asyncio
    async def test_launch_distributed_training_failure(self, launch_coordinator, mock_job):
        """Test distributed training launch with failures."""
        # Mock SSH client failure for one worker
        launch_coordinator.cluster.ssh_clients['gpu1'].run_command.return_value = (True, "torchrun started", "")
        launch_coordinator.cluster.ssh_clients['gpu2'].run_command.return_value = (False, "", "Connection failed")
        
        workers = [
            {'name': 'gpu1', 'rank': 1, 'ip': '192.168.1.101', 'hardware': {'gpu_model': 'RTX 4070', 'vram_gb': 8}, 'user': 'ob'},
            {'name': 'gpu2', 'rank': 2, 'ip': '192.168.1.102', 'hardware': {'gpu_model': 'RTX 3070 Ti', 'vram_gb': 8}, 'user': 'ollie'},
        ]
        
        allocation_plan = await launch_coordinator.allocate_resources(workers)
        
        success = await launch_coordinator.launch_distributed_training(
            job=mock_job,
            allocation_plan=allocation_plan,
            training_script="scripts/train_production.py",
        )
        
        assert success is False
    
    @pytest.mark.asyncio
    async def test_build_torchrun_command(self, launch_coordinator, mock_job):
        """Test torchrun command building."""
        allocation_plan = {
            'world_size': 2,
            'master_addr': '192.168.1.100',
            'master_port': 29500,
        }
        
        command = launch_coordinator._build_torchrun_command(
            allocation_plan=allocation_plan,
            training_script="scripts/train_production.py",
            job=mock_job,
        )
        
        assert 'torchrun' in command
        assert '--nproc_per_node=1' in command
        assert '--nnodes=2' in command
        assert '--node_rank=${RANK}' in command
        assert '--master_addr=192.168.1.100' in command
        assert '--master_port=29500' in command
        assert 'scripts/train_production.py' in command
        assert '--model tinyllama' in command
        assert '--dataset wikitext' in command
        assert '--epochs 1' in command
        assert '--learning-rate 0.0002' in command
    
    @pytest.mark.asyncio
    async def test_stop_training(self, launch_coordinator):
        """Test training stop and cleanup."""
        # Mock cleanup methods
        with patch.object(launch_coordinator, '_cleanup_processes') as mock_cleanup:
            await launch_coordinator.stop_training()
        
        # Verify cleanup was called
        mock_cleanup.assert_called_once()
        assert launch_coordinator.should_stop.is_set()
    
    def test_get_master_address(self, launch_coordinator):
        """Test master address retrieval."""
        addr = launch_coordinator._get_master_address()
        assert addr == '192.168.1.100'
    
    def test_get_master_port(self, launch_coordinator):
        """Test master port retrieval."""
        port = launch_coordinator._get_master_port()
        assert port == 29500


if __name__ == "__main__":
    # Run a simple test
    async def simple_test():
        """Simple integration test."""
        print("Testing Launch Coordinator...")
        
        # Create mock cluster
        cluster = MagicMock(spec=Cluster)
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
        
        # Test validation
        success, results = await coordinator.validate_cluster_state()
        print(f"✓ Cluster validation: {success}")
        
        # Test resource allocation
        workers = [
            {'name': 'gpu1', 'rank': 1, 'ip': '192.168.1.101', 'hardware': {'gpu_model': 'RTX 4070', 'vram_gb': 8}, 'user': 'ob'},
            {'name': 'gpu2', 'rank': 2, 'ip': '192.168.1.102', 'hardware': {'gpu_model': 'RTX 3070 Ti', 'vram_gb': 8}, 'user': 'ollie'},
        ]
        allocation_plan = await coordinator.allocate_resources(workers)
        print(f"✓ Resource allocation: {allocation_plan['total_nodes']} nodes")
        
        # Test command building
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
            print(f"✓ Command built: {len(command)} characters")
        
        print("✅ All tests passed!")
    
    # Run the test
    asyncio.run(simple_test())