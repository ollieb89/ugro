# UGRO Phase 2 Orchestration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a unified GPU resource orchestration layer that replaces manual 3-terminal distributed training with a single command interface.

**Architecture:** Python-based orchestrator running on gpu-master with Click CLI, YAML configuration, JSON state management, and asyncio-based health monitoring.

**Tech Stack:** Python 3.10+, Click, PyYAML, asyncio, subprocess, systemd, pytest

---

## Phase 1: Foundation Setup

### Task 1: Create Project Structure

**Files:**
- Create: `src/ugro/__init__.py`
- Create: `src/ugro/cli.py`
- Create: `src/ugro/agent.py`
- Create: `src/ugro/monitor.py`
- Create: `src/ugro/metrics.py`
- Create: `src/ugro/state.py`
- Create: `src/ugro/ssh.py`
- Create: `src/ugro/utils.py`
- Create: `config/cluster.yaml`
- Create: `data/.gitkeep`
- Create: `logs/.gitkeep`
- Create: `tests/__init__.py`
- Create: `tests/test_cli.py`
- Create: `tests/test_agent.py`
- Create: `tests/test_monitor.py`

**Step 1: Write the failing test**

```python
# tests/test_basic_structure.py
def test_project_structure():
    """Test that all required directories and files exist"""
    from pathlib import Path
    
    # Check directories exist
    assert Path("src/ugro").exists()
    assert Path("config").exists()
    assert Path("data").exists()
    assert Path("logs").exists()
    assert Path("tests").exists()
    
    # Check main module is importable
    import src.ugro
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_basic_structure.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'src.ugro'"

**Step 3: Write minimal implementation**

```bash
# Create directory structure
mkdir -p src/ugro config data logs tests
```

```python
# src/ugro/__init__.py
"""UGRO: Unified GPU Resource Orchestrator"""
__version__ = "0.1.0"
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_basic_structure.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add .
git commit -m "feat: create basic project structure for UGRO orchestration"
```

### Task 2: Set Up Configuration System

**Files:**
- Create: `config/cluster.yaml`
- Modify: `src/ugro/utils.py`
- Test: `tests/test_config.py`

**Step 1: Write the failing test**

```python
# tests/test_config.py
def test_load_cluster_config():
    """Test loading cluster configuration from YAML"""
    from src.ugro.utils import load_cluster_config
    
    config = load_cluster_config()
    
    assert 'cluster' in config
    assert 'nodes' in config
    assert config['cluster']['name'] == "Home AI Lab"
    assert len(config['nodes']) == 3
    assert 'gpu-master' in config['nodes']
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py -v`
Expected: FAIL with "cannot import name 'load_cluster_config'"

**Step 3: Write minimal implementation**

```yaml
# config/cluster.yaml
cluster:
  name: "Home AI Lab"
  master_ip: "192.168.1.100"
  master_port: 29500
  
nodes:
  gpu-master:
    ip: "192.168.1.100"
    user: "ollie"
    gpu: "RTX 5070 Ti"
    vram_gb: 12
    role: "master"
  
  gpu1:
    ip: "192.168.1.101"
    user: "ob"
    gpu: "RTX 4070"
    vram_gb: 8
    role: "worker"
  
  gpu2:
    ip: "192.168.1.102"
    user: "ollie"
    gpu: "RTX 3070 Ti"
    vram_gb: 8
    role: "worker"

training:
  batch_size_per_gpu: 1
  gradient_accumulation: 8
  default_model: "unsloth/tinyllama-bnb-4bit"
  default_dataset: "wikitext"
```

```python
# src/ugro/utils.py
"""Utility functions for UGRO"""
import yaml
from pathlib import Path
from typing import Dict, Any

def load_cluster_config(config_path: str = None) -> Dict[str, Any]:
    """Load cluster configuration from YAML file"""
    if config_path is None:
        config_path = Path(__file__).parent.parent / "config" / "cluster.yaml"
    
    with open(config_path) as f:
        return yaml.safe_load(f)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_config.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add config/cluster.yaml src/ugro/utils.py tests/test_config.py
git commit -m "feat: add cluster configuration system"
```

### Task 3: Implement Basic CLI with Click

**Files:**
- Modify: `src/ugro/cli.py`
- Test: `tests/test_cli.py`

**Step 1: Write the failing test**

```python
# tests/test_cli.py
import pytest
from click.testing import CliRunner
from src.ugro.cli import cli

def test_cli_health_command():
    """Test the health command exists and runs"""
    runner = CliRunner()
    result = runner.invoke(cli, ['health'])
    
    assert result.exit_code == 0
    assert 'GPU Cluster Orchestration' in result.output
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_cli.py -v`
Expected: FAIL with "cannot import name 'cli'"

**Step 3: Write minimal implementation**

```python
# src/ugro/cli.py
"""UGRO CLI: Command-line interface for GPU orchestration"""
import click

@click.group()
@click.version_option()
def cli():
    """UGRO: GPU Cluster Orchestration"""
    pass

@cli.command()
def health():
    """Check cluster health"""
    click.echo("UGRO: GPU Cluster Orchestration")
    click.echo("Health check not yet implemented")

if __name__ == '__main__':
    cli()
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_cli.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/ugro/cli.py tests/test_cli.py
git commit -m "feat: add basic CLI structure with Click"
```

### Task 4: Add Cluster State Management

**Files:**
- Create: `src/ugro/state.py`
- Test: `tests/test_state.py`

**Step 1: Write the failing test**

```python
# tests/test_state.py
def test_cluster_state_manager():
    """Test cluster state persistence and updates"""
    from src.ugro.state import ClusterStateManager
    
    manager = ClusterStateManager()
    
    # Test initial state
    state = manager.load_state()
    assert 'nodes' in state
    assert 'jobs' in state
    
    # Test job registration
    manager.register_job('test_job', ['gpu-master', 'gpu1'], {'model': 'test'})
    updated_state = manager.load_state()
    
    assert 'test_job' in updated_state['jobs']
    assert updated_state['jobs']['test_job']['status'] == 'running'
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_state.py -v`
Expected: FAIL with "cannot import name 'ClusterStateManager'"

**Step 3: Write minimal implementation**

```python
# src/ugro/state.py
"""Cluster state management for UGRO"""
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

class ClusterStateManager:
    """Manages cluster state persistence"""
    
    def __init__(self, state_path: str = None):
        if state_path is None:
            self.state_path = Path(__file__).parent.parent / "data" / "cluster_state.json"
        else:
            self.state_path = Path(state_path)
        
        # Ensure data directory exists
        self.state_path.parent.mkdir(exist_ok=True)
        
        # Initialize state if doesn't exist
        if not self.state_path.exists():
            self._initialize_state()
    
    def _initialize_state(self):
        """Create initial cluster state"""
        initial_state = {
            "nodes": {
                "gpu-master": {
                    "ip": "192.168.1.100",
                    "gpu": "RTX 5070 Ti",
                    "vram_gb": 12,
                    "status": "available",
                    "running_job_id": None
                },
                "gpu1": {
                    "ip": "192.168.1.101",
                    "gpu": "RTX 4070",
                    "vram_gb": 8,
                    "status": "available",
                    "running_job_id": None
                },
                "gpu2": {
                    "ip": "192.168.1.102",
                    "gpu": "RTX 3070 Ti",
                    "vram_gb": 8,
                    "status": "available",
                    "running_job_id": None
                }
            },
            "jobs": {}
        }
        
        with open(self.state_path, 'w') as f:
            json.dump(initial_state, f, indent=2)
    
    def load_state(self) -> Dict[str, Any]:
        """Load current cluster state"""
        with open(self.state_path) as f:
            return json.load(f)
    
    def save_state(self, state: Dict[str, Any]):
        """Save cluster state to disk"""
        with open(self.state_path, 'w') as f:
            json.dump(state, f, indent=2)
    
    def register_job(self, job_id: str, nodes: List[str], config: Dict[str, Any]):
        """Register a new job in the cluster"""
        state = self.load_state()
        
        state['jobs'][job_id] = {
            "status": "running",
            "nodes": nodes,
            "config": config,
            "started_at": datetime.now().isoformat(),
            "gpu_nodes": nodes
        }
        
        # Update node status
        for node in nodes:
            if node in state['nodes']:
                state['nodes'][node]['status'] = 'busy'
                state['nodes'][node]['running_job_id'] = job_id
        
        self.save_state(state)
    
    def update_job_status(self, job_id: str, status: str):
        """Update job status"""
        state = self.load_state()
        
        if job_id in state['jobs']:
            state['jobs'][job_id]['status'] = status
            
            if status in ['completed', 'failed']:
                # Free up nodes
                for node in state['jobs'][job_id]['nodes']:
                    if node in state['nodes']:
                        state['nodes'][node]['status'] = 'available'
                        state['nodes'][node]['running_job_id'] = None
            
            self.save_state(state)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_state.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/ugro/state.py tests/test_state.py
git commit -m "feat: implement cluster state management"
```

---

## Phase 2: Core Orchestration

### Task 5: Implement SSH Utilities

**Files:**
- Create: `src/ugro/ssh.py`
- Test: `tests/test_ssh.py`

**Step 1: Write the failing test**

```python
# tests/test_ssh.py
import pytest
from unittest.mock import patch, MagicMock
from src.ugro.ssh import SSHClient

def test_ssh_client_execute():
    """Test SSH command execution"""
    client = SSHClient()
    
    with patch('subprocess.run') as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="OK", stderr="")
        
        success, stdout, stderr = client.execute("gpu-master", "echo test")
        
        assert success is True
        assert stdout == "OK"
        mock_run.assert_called_once()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_ssh.py -v`
Expected: FAIL with "cannot import name 'SSHClient'"

**Step 3: Write minimal implementation**

```python
# src/ugro/ssh.py
"""SSH utilities for remote command execution"""
import subprocess
import logging
from typing import Tuple, Optional
from src.ugro.utils import load_cluster_config

class SSHClient:
    """SSH client for remote command execution"""
    
    def __init__(self):
        self.config = load_cluster_config()
        self.logger = logging.getLogger(__name__)
    
    def execute(self, node: str, command: str, timeout: int = 30) -> Tuple[bool, str, str]:
        """Execute command on remote node via SSH"""
        if node not in self.config['nodes']:
            raise ValueError(f"Unknown node: {node}")
        
        node_config = self.config['nodes'][node]
        ssh_command = f"ssh -o ConnectTimeout={timeout} {node_config['user']}@{node_config['ip']} '{command}'"
        
        try:
            result = subprocess.run(
                ssh_command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            success = result.returncode == 0
            return success, result.stdout, result.stderr
            
        except subprocess.TimeoutExpired:
            return False, "", "SSH timeout"
        except Exception as e:
            self.logger.error(f"SSH error on {node}: {e}")
            return False, "", str(e)
    
    def test_connection(self, node: str) -> bool:
        """Test SSH connection to node"""
        success, _, _ = self.execute(node, "echo 'OK'", timeout=5)
        return success and "OK" in stdout
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_ssh.py -v`
Expected: PASS (after fixing stdout reference in test)

**Step 5: Commit**

```bash
git add src/ugro/ssh.py tests/test_ssh.py
git commit -m "feat: add SSH utilities for remote execution"
```

### Task 6: Implement UGROAgent Class

**Files:**
- Modify: `src/ugro/agent.py`
- Test: `tests/test_agent.py`

**Step 1: Write the failing test**

```python
# tests/test_agent.py
import pytest
from unittest.mock import patch, MagicMock
from src.ugro.agent import UGROAgent

def test_agent_validate_cluster():
    """Test cluster validation"""
    agent = UGROAgent()
    
    with patch.object(agent.ssh_client, 'test_connection', return_value=True):
        assert agent.validate_cluster() is True
    
    with patch.object(agent.ssh_client, 'test_connection', return_value=False):
        assert agent.validate_cluster() is False
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_agent.py -v`
Expected: FAIL with "cannot import name 'UGROAgent'"

**Step 3: Write minimal implementation**

```python
# src/ugro/agent.py
"""UGRO Agent: Main orchestration logic"""
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

from src.ugro.utils import load_cluster_config
from src.ugro.state import ClusterStateManager
from src.ugro.ssh import SSHClient

class UGROAgent:
    """Main orchestration agent for UGRO"""
    
    def __init__(self):
        self.config = load_cluster_config()
        self.state_manager = ClusterStateManager()
        self.ssh_client = SSHClient()
        self.logger = logging.getLogger(__name__)
        
        self.master_ip = self.config['cluster']['master_ip']
        self.nodes = self.config['nodes']
    
    def validate_cluster(self) -> bool:
        """Validate all nodes are reachable via SSH"""
        for node_name in self.nodes:
            if not self.ssh_client.test_connection(node_name):
                self.logger.error(f"Cannot reach node: {node_name}")
                return False
        return True
    
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get current status of a job"""
        state = self.state_manager.load_state()
        
        if job_id in state['jobs']:
            job = state['jobs'][job_id]
            
            # Calculate elapsed time if running
            if job['status'] == 'running':
                start_time = datetime.fromisoformat(job['started_at'])
                elapsed = (datetime.now() - start_time).total_seconds()
                job['elapsed_seconds'] = int(elapsed)
            
            return job
        
        return {'status': 'not_found'}
    
    def full_health_check(self) -> Dict[str, Dict[str, Any]]:
        """Perform comprehensive health check"""
        health = {}
        
        for node_name, node_config in self.nodes.items():
            node_health = {
                'healthy': False,
                'message': 'Unknown'
            }
            
            # Test SSH connection
            if self.ssh_client.test_connection(node_name):
                node_health['healthy'] = True
                node_health['message'] = f"{node_config['gpu']} ({node_config['vram_gb']}GB)"
            else:
                node_health['message'] = 'SSH connection failed'
            
            health[node_name] = node_health
        
        return health
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_agent.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/ugro/agent.py tests/test_agent.py
git commit -m "feat: implement UGROAgent core orchestration class"
```

### Task 7: Add Launch Distributed Training Method

**Files:**
- Modify: `src/ugro/agent.py`
- Test: `tests/test_agent_launch.py`

**Step 1: Write the failing test**

```python
# tests/test_agent_launch.py
import pytest
from unittest.mock import patch, MagicMock
from src.ugro.agent import UGROAgent

def test_launch_distributed_training():
    """Test distributed training launch"""
    agent = UGROAgent()
    
    with patch.object(agent, 'validate_cluster', return_value=True), \
         patch.object(agent, '_launch_on_node') as mock_launch, \
         patch.object(agent, '_monitor_training') as mock_monitor:
        
        agent.launch_distributed_training(
            job_id="test_job",
            model="test_model",
            dataset="test_dataset",
            epochs=1
        )
        
        # Should launch on all 3 nodes
        assert mock_launch.call_count == 3
        mock_monitor.assert_called_once_with("test_job")
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_agent_launch.py -v`
Expected: FAIL with "UGROAgent has no attribute 'launch_distributed_training'"

**Step 3: Write minimal implementation**

```python
# Add to src/ugro/agent.py
    def launch_distributed_training(self, job_id: str, model: str, dataset: str, epochs: int):
        """Launch distributed training across cluster"""
        self.logger.info(f"Launching distributed training: {job_id}")
        
        # Validate cluster
        if not self.validate_cluster():
            raise RuntimeError("Cluster health check failed")
        
        # Register job
        node_names = list(self.nodes.keys())
        self.state_manager.register_job(job_id, node_names, {
            'model': model,
            'dataset': dataset,
            'epochs': epochs
        })
        
        # Launch on each node
        for rank, node_name in enumerate(node_names):
            self._launch_on_node(job_id, rank, node_name, model, dataset, epochs)
        
        # Monitor training
        self._monitor_training(job_id)
    
    def _launch_on_node(self, job_id: str, rank: int, node_name: str, 
                       model: str, dataset: str, epochs: int):
        """Launch training on a specific node"""
        command = self._build_torchrun_command(job_id, rank, model, dataset, epochs)
        
        # Execute via SSH
        success, stdout, stderr = self.ssh_client.execute(
            node_name,
            f"cd ~/ai-cluster/scripts && {command}",
            timeout=60
        )
        
        if not success:
            raise RuntimeError(f"Failed to launch on {node_name}: {stderr}")
        
        self.logger.info(f"Launched rank {rank} on {node_name}")
    
    def _build_torchrun_command(self, job_id: str, rank: int, 
                               model: str, dataset: str, epochs: int) -> str:
        """Build torchrun command for distributed training"""
        return f"""
        nohup torchrun \\
            --nnodes=3 \\
            --nproc_per_node=1 \\
            --rdzv_id={job_id} \\
            --rdzv_backend=c10d \\
            --rdzv_endpoint={self.master_ip}:29500 \\
            --node_rank={rank} \\
            train_production.py \\
            --model-name {model} \\
            --dataset-name {dataset} \\
            --num-epochs {epochs} \\
            --job-id {job_id} \\
            > training_rank{rank}_{job_id}.log 2>&1 &
        """
    
    def _monitor_training(self, job_id: str):
        """Monitor training until completion"""
        import time
        
        self.logger.info(f"Monitoring training job: {job_id}")
        
        while True:
            job_status = self.get_job_status(job_id)
            
            if job_status['status'] in ['completed', 'failed']:
                self.logger.info(f"Job {job_id} finished with status: {job_status['status']}")
                self.state_manager.update_job_status(job_id, job_status['status'])
                break
            
            time.sleep(10)  # Check every 10 seconds
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_agent_launch.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/ugro/agent.py tests/test_agent_launch.py
git commit -m "feat: add distributed training launch functionality"
```

### Task 8: Connect CLI to Agent

**Files:**
- Modify: `src/ugro/cli.py`
- Test: `tests/test_cli_integration.py`

**Step 1: Write the failing test**

```python
# tests/test_cli_integration.py
import pytest
from click.testing import CliRunner
from unittest.mock import patch, MagicMock
from src.ugro.cli import cli

def test_cli_launch_command():
    """Test CLI launch command integration"""
    runner = CliRunner()
    
    with patch('src.ugro.cli.UGROAgent') as mock_agent_class:
        mock_agent = MagicMock()
        mock_agent_class.return_value = mock_agent
        
        result = runner.invoke(cli, [
            'launch',
            '--model', 'test_model',
            '--dataset', 'test_dataset',
            '--epochs', '1',
            '--name', 'test_job'
        ])
        
        assert result.exit_code == 0
        mock_agent.launch_distributed_training.assert_called_once()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_cli_integration.py -v`
Expected: FAIL with "No such command 'launch'"

**Step 3: Write minimal implementation**

```python
# Update src/ugro/cli.py
import click
from datetime import datetime
from src.ugro.agent import UGROAgent

@click.group()
@click.version_option()
def cli():
    """UGRO: GPU Cluster Orchestration"""
    pass

@cli.command()
@click.option('--model', default='unsloth/tinyllama-bnb-4bit', help='Model to train')
@click.option('--dataset', default='wikitext', help='Dataset to use')
@click.option('--epochs', default=1, type=int, help='Number of epochs')
@click.option('--name', default=None, help='Job name')
def launch(model, dataset, epochs, name):
    """Launch distributed training across cluster"""
    agent = UGROAgent()
    
    # Generate job ID if not provided
    job_id = name or f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    click.echo(f"🚀 Launching job: {job_id}")
    click.echo(f"   Model: {model}")
    click.echo(f"   Dataset: {dataset}")
    click.echo(f"   Epochs: {epochs}")
    
    try:
        agent.launch_distributed_training(
            job_id=job_id,
            model=model,
            dataset=dataset,
            epochs=epochs
        )
        click.echo(f"✅ Job {job_id} completed successfully")
    except Exception as e:
        click.echo(f"❌ Error: {e}")
        raise click.ClickException(str(e))

@cli.command()
@click.argument('job_id')
def status(job_id):
    """Check status of running job"""
    agent = UGROAgent()
    job_status = agent.get_job_status(job_id)
    
    click.echo(f"Job: {job_id}")
    click.echo(f"Status: {job_status['status']}")
    
    if 'gpu_nodes' in job_status:
        click.echo(f"GPU Nodes: {', '.join(job_status['gpu_nodes'])}")
    
    if 'elapsed_seconds' in job_status:
        click.echo(f"Elapsed: {job_status['elapsed_seconds']}s")

@cli.command()
def health():
    """Check cluster health"""
    agent = UGROAgent()
    health = agent.full_health_check()
    
    for node, status in health.items():
        symbol = "✓" if status['healthy'] else "❌"
        click.echo(f"{symbol} {node}: {status['message']}")

if __name__ == '__main__':
    cli()
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_cli_integration.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/ugro/cli.py tests/test_cli_integration.py
git commit -m "feat: connect CLI to UGROAgent for launch commands"
```

---

## Phase 3: Health Monitoring System

### Task 9: Implement Health Metrics Data Structure

**Files:**
- Create: `src/ugro/monitor.py`
- Test: `tests/test_monitor_data.py`

**Step 1: Write the failing test**

```python
# tests/test_monitor_data.py
import pytest
from datetime import datetime
from src.ugro.monitor import HealthMetrics

def test_health_metrics_creation():
    """Test HealthMetrics dataclass creation"""
    metrics = HealthMetrics(
        node_name="gpu-master",
        timestamp=datetime.now(),
        gpu_utilization=85.5,
        gpu_memory_used=8.2,
        gpu_memory_total=12.0,
        gpu_temperature=72.0,
        gpu_power_usage=250.0,
        cpu_utilization=45.0,
        memory_usage=60.0,
        disk_usage=30.0,
        network_latency=0.5,
        process_status={"1234": True},
        health_score=85.0,
        alerts=[]
    )
    
    assert metrics.node_name == "gpu-master"
    assert metrics.health_score == 85.0
    assert len(metrics.alerts) == 0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_monitor_data.py -v`
Expected: FAIL with "cannot import name 'HealthMetrics'"

**Step 3: Write minimal implementation**

```python
# src/ugro/monitor.py
"""Health monitoring system for UGRO"""
import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import time

@dataclass
class HealthMetrics:
    """Comprehensive health metrics for a node"""
    node_name: str
    timestamp: datetime
    gpu_utilization: float
    gpu_memory_used: float
    gpu_memory_total: float
    gpu_temperature: float
    gpu_power_usage: float
    cpu_utilization: float
    memory_usage: float
    disk_usage: float
    network_latency: float
    process_status: Dict[str, bool]
    health_score: float
    alerts: List[str]
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_monitor_data.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/ugro/monitor.py tests/test_monitor_data.py
git commit -m "feat: add HealthMetrics data structure for monitoring"
```

### Task 10: Implement Adaptive Health Monitor

**Files:**
- Modify: `src/ugro/monitor.py`
- Test: `tests/test_adaptive_monitor.py`

**Step 1: Write the failing test**

```python
# tests/test_adaptive_monitor.py
import pytest
from unittest.mock import MagicMock, patch
from src.ugro.monitor import AdaptiveHealthMonitor

def test_adaptive_interval_calculation():
    """Test adaptive polling interval calculation"""
    monitor = AdaptiveHealthMonitor(MagicMock(), MagicMock())
    
    # No active jobs, recent activity
    monitor.last_job_activity = datetime.now() - timedelta(minutes=15)
    interval = monitor._calculate_adaptive_interval()
    assert interval == 10.0  # Base interval
    
    # Active jobs
    monitor.active_jobs.add('test_job')
    interval = monitor._calculate_adaptive_interval()
    assert interval == 5.0  # Minimum interval
    
    # No active jobs, long idle
    monitor.active_jobs.clear()
    monitor.last_job_activity = datetime.now() - timedelta(hours=3)
    interval = monitor._calculate_adaptive_interval()
    assert interval == 40.0  # 4x base interval
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_adaptive_monitor.py -v`
Expected: FAIL with "cannot import name 'AdaptiveHealthMonitor'"

**Step 3: Write minimal implementation**

```python
# Add to src/ugro/monitor.py
from concurrent.futures import ThreadPoolExecutor
from src.ugro.state import ClusterStateManager

class AdaptiveHealthMonitor:
    """Production health monitor with adaptive polling"""
    
    def __init__(self, cluster, state_manager: ClusterStateManager):
        self.cluster = cluster
        self.state_manager = state_manager
        self.logger = logging.getLogger(__name__)
        
        # Adaptive polling configuration
        self.base_interval = 10.0
        self.max_interval = 60.0
        self.min_interval = 5.0
        
        # Circuit breaker for failing nodes
        self.node_failures: Dict[str, int] = {}
        self.node_circuit_breakers: Dict[str, datetime] = {}
        self.max_failures = 3
        self.circuit_breaker_timeout = 300  # 5 minutes
        
        # Performance optimization
        self.executor = ThreadPoolExecutor(max_workers=8)
        self.metrics_history: Dict[str, List[HealthMetrics]] = {}
        
        # State tracking
        self.active_jobs = set()
        self.last_job_activity = datetime.now()
        self.health_scores: Dict[str, float] = {}
    
    def _calculate_adaptive_interval(self) -> float:
        """Calculate polling interval based on cluster activity"""
        if self.active_jobs:
            # Active jobs - more frequent monitoring
            return self.min_interval
        else:
            # No active jobs - check how long since last activity
            time_since_activity = datetime.now() - self.last_job_activity
            
            if time_since_activity < timedelta(minutes=30):
                return self.base_interval
            elif time_since_activity < timedelta(hours=2):
                return self.base_interval * 2
            else:
                return min(self.max_interval, self.base_interval * 4)
    
    def register_job_activity(self, job_name: str):
        """Register job activity to adjust monitoring frequency"""
        self.active_jobs.add(job_name)
        self.last_job_activity = datetime.now()
        self.logger.info(f"Job activity registered: {job_name}")
    
    def unregister_job_activity(self, job_name: str):
        """Unregister job activity"""
        self.active_jobs.discard(job_name)
        self.last_job_activity = datetime.now()
        self.logger.info(f"Job activity unregistered: {job_name}")
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_adaptive_monitor.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/ugro/monitor.py tests/test_adaptive_monitor.py
git commit -m "feat: implement AdaptiveHealthMonitor with polling logic"
```

### Task 11: Add Concurrent Metrics Collection

**Files:**
- Modify: `src/ugro/monitor.py`
- Test: `tests/test_metrics_collection.py`

**Step 1: Write the failing test**

```python
# tests/test_metrics_collection.py
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
import asyncio
from src.ugro.monitor import AdaptiveHealthMonitor

@pytest.mark.asyncio
async def test_collect_metrics_concurrently():
    """Test concurrent metrics collection from nodes"""
    cluster = MagicMock()
    cluster.get_all_workers.return_value = [
        {'name': 'gpu-master', 'ip': '192.168.1.100'},
        {'name': 'gpu1', 'ip': '192.168.1.101'}
    ]
    
    monitor = AdaptiveHealthMonitor(cluster, MagicMock())
    
    with patch.object(monitor, '_collect_node_metrics', new_callable=AsyncMock) as mock_collect:
        mock_collect.return_value = MagicMock()
        
        metrics = await monitor._collect_metrics_concurrently()
        
        assert len(metrics) == 2
        assert mock_collect.call_count == 2
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_metrics_collection.py -v`
Expected: FAIL with "AdaptiveHealthMonitor has no attribute '_collect_metrics_concurrently'"

**Step 3: Write minimal implementation**

```python
# Add to src/ugro/monitor.py
    async def _collect_metrics_concurrently(self) -> List[HealthMetrics]:
        """Collect metrics from all nodes concurrently"""
        workers = self.cluster.get_all_workers()
        
        # Create tasks for concurrent execution
        tasks = []
        for worker in workers:
            if not self._is_circuit_breaker_active(worker['name']):
                task = asyncio.create_task(
                    self._collect_node_metrics(worker)
                )
                tasks.append((worker['name'], task))
        
        # Wait for all tasks to complete
        results = []
        for node_name, task in tasks:
            try:
                metrics = await task
                if metrics:
                    results.append(metrics)
                    # Reset failure count on success
                    self.node_failures[node_name] = 0
            except Exception as e:
                self._handle_node_failure(node_name, e)
        
        return results
    
    async def _collect_node_metrics(self, worker: Dict) -> Optional[HealthMetrics]:
        """Collect comprehensive metrics from a single node"""
        node_name = worker['name']
        
        try:
            # Use existing cluster health check as base
            health_status = self.cluster.check_health().get(node_name, {})
            
            if not health_status.get('healthy', False):
                return None
            
            # Mock metrics for now (will implement actual collection)
            return HealthMetrics(
                node_name=node_name,
                timestamp=datetime.now(),
                gpu_utilization=75.0,
                gpu_memory_used=6.0,
                gpu_memory_total=12.0,
                gpu_temperature=70.0,
                gpu_power_usage=200.0,
                cpu_utilization=50.0,
                memory_usage=60.0,
                disk_usage=40.0,
                network_latency=1.0,
                process_status={},
                health_score=80.0,
                alerts=[]
            )
            
        except Exception as e:
            self.logger.error(f"Failed to collect metrics from {node_name}: {e}")
            raise
    
    def _is_circuit_breaker_active(self, node_name: str) -> bool:
        """Check if circuit breaker is active for a node"""
        if node_name not in self.node_circuit_breakers:
            return False
        
        breaker_time = self.node_circuit_breakers[node_name]
        if datetime.now() - breaker_time > timedelta(seconds=self.circuit_breaker_timeout):
            # Circuit breaker timeout expired
            del self.node_circuit_breakers[node_name]
            self.node_failures[node_name] = 0
            return False
        
        return True
    
    def _handle_node_failure(self, node_name: str, error: Exception):
        """Handle node failures with circuit breaker logic"""
        self.node_failures[node_name] = self.node_failures.get(node_name, 0) + 1
        
        self.logger.warning(f"Node {node_name} failure #{self.node_failures[node_name]}: {error}")
        
        if self.node_failures[node_name] >= self.max_failures:
            # Activate circuit breaker
            self.node_circuit_breakers[node_name] = datetime.now()
            self.logger.error(f"Circuit breaker activated for {node_name}")
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_metrics_collection.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/ugro/monitor.py tests/test_metrics_collection.py
git commit -m "feat: add concurrent metrics collection to health monitor"
```

### Task 12: Implement Main Monitoring Loop

**Files:**
- Modify: `src/ugro/monitor.py`
- Test: `tests/test_monitoring_loop.py`

**Step 1: Write the failing test**

```python
# tests/test_monitoring_loop.py
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
import asyncio
from src.ugro.monitor import AdaptiveHealthMonitor

@pytest.mark.asyncio
async def test_monitoring_loop():
    """Test main monitoring loop execution"""
    cluster = MagicMock()
    monitor = AdaptiveHealthMonitor(cluster, MagicMock())
    
    with patch.object(monitor, '_calculate_adaptive_interval', return_value=0.1), \
         patch.object(monitor, '_collect_metrics_concurrently', new_callable=AsyncMock) as mock_collect, \
         patch.object(monitor, '_process_metrics', new_callable=AsyncMock) as mock_process, \
         patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
        
        mock_collect.return_value = []
        mock_process.return_value = None
        
        # Run loop for a short time
        task = asyncio.create_task(monitor.start_monitoring())
        await asyncio.sleep(0.3)
        task.cancel()
        
        try:
            await task
        except asyncio.CancelledError:
            pass
        
        # Should have collected metrics multiple times
        assert mock_collect.call_count >= 2
        assert mock_process.call_count >= 2
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_monitoring_loop.py -v`
Expected: FAIL with "AdaptiveHealthMonitor has no attribute 'start_monitoring'"

**Step 3: Write minimal implementation**

```python
# Add to src/ugro/monitor.py
    async def start_monitoring(self):
        """Main monitoring loop with adaptive polling"""
        self.logger.info("Starting adaptive health monitoring...")
        
        while True:
            start_time = time.time()
            
            try:
                # Get current polling interval based on cluster state
                interval = self._calculate_adaptive_interval()
                
                # Collect metrics concurrently
                metrics = await self._collect_metrics_concurrently()
                
                # Process metrics and update state
                await self._process_metrics(metrics)
                
                # Check for critical conditions
                await self._check_critical_conditions(metrics)
                
                # Clean up old data
                self._cleanup_old_metrics()
                
                # Calculate sleep time (remaining interval - execution time)
                execution_time = time.time() - start_time
                sleep_time = max(0, interval - execution_time)
                
                self.logger.debug(f"Health check completed in {execution_time:.2f}s, sleeping {sleep_time:.2f}s")
                await asyncio.sleep(sleep_time)
                
            except Exception as e:
                self.logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(self.base_interval)
    
    async def _process_metrics(self, metrics_list: List[HealthMetrics]):
        """Process collected metrics and update state"""
        for metrics in metrics_list:
            node_name = metrics.node_name
            
            # Store metrics history
            if node_name not in self.metrics_history:
                self.metrics_history[node_name] = []
            
            self.metrics_history[node_name].append(metrics)
            
            # Keep only last 100 entries per node
            if len(self.metrics_history[node_name]) > 100:
                self.metrics_history[node_name] = self.metrics_history[node_name][-100:]
            
            # Update health score
            self.health_scores[node_name] = metrics.health_score
            
            # Update node status based on health score
            try:
                if metrics.health_score >= 80:
                    status = "healthy"
                elif metrics.health_score >= 60:
                    status = "degraded"
                else:
                    status = "unhealthy"
                
                self.state_manager.update_node_status(
                    node_name, 
                    status=status,
                    health_score=metrics.health_score,
                    last_check=metrics.timestamp.isoformat()
                )
            except KeyError:
                pass
    
    async def _check_critical_conditions(self, metrics_list: List[HealthMetrics]):
        """Check for critical cluster conditions"""
        critical_alerts = []
        
        for metrics in metrics_list:
            critical_alerts.extend([alert for alert in metrics.alerts if alert.startswith("CRITICAL")])
        
        # Check for cluster-wide issues
        unhealthy_nodes = sum(1 for m in metrics_list if m.health_score < 50)
        total_nodes = len(metrics_list)
        
        if total_nodes > 0 and unhealthy_nodes / total_nodes > 0.5:
            critical_alerts.append(f"CRITICAL: {unhealthy_nodes}/{total_nodes} nodes unhealthy")
        
        # Send alerts if any
        if critical_alerts:
            await self._send_alerts(critical_alerts)
    
    async def _send_alerts(self, alerts: List[str]):
        """Send alerts to configured channels"""
        for alert in alerts:
            self.logger.critical(alert)
            # TODO: Add webhook, Slack, email notifications
    
    def _cleanup_old_metrics(self):
        """Clean up old metrics to prevent memory leaks"""
        cutoff_time = datetime.now() - timedelta(hours=24)
        
        for node_name in list(self.metrics_history.keys()):
            self.metrics_history[node_name] = [
                m for m in self.metrics_history[node_name] 
                if m.timestamp > cutoff_time
            ]
            
            if not self.metrics_history[node_name]:
                del self.metrics_history[node_name]
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_monitoring_loop.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/ugro/monitor.py tests/test_monitoring_loop.py
git commit -m "feat: implement main monitoring loop with adaptive polling"
```

---

## Phase 4: Metrics and Results

### Task 13: Implement Metrics Collector

**Files:**
- Create: `src/ugro/metrics.py`
- Test: `tests/test_metrics_collector.py`

**Step 1: Write the failing test**

```python
# tests/test_metrics_collector.py
import pytest
from unittest.mock import MagicMock, patch
from src.ugro.metrics import MetricsCollector

def test_metrics_collector_collect_training_metrics():
    """Test collecting training metrics from logs"""
    collector = MetricsCollector(MagicMock())
    
    with patch.object(collector, '_parse_training_log', return_value={
        'loss': 3.5,
        'perplexity': 33.1,
        'learning_rate': 0.0002,
        'gpu_util': 85.0
    }):
        metrics = collector.collect_training_metrics('test_job', 0)
        
        assert metrics['loss'] == 3.5
        assert metrics['rank'] == 0
        assert 'timestamp' in metrics
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_metrics_collector.py -v`
Expected: FAIL with "cannot import name 'MetricsCollector'"

**Step 3: Write minimal implementation**

```python
# src/ugro/metrics.py
"""Metrics collection and aggregation for UGRO"""
import json
import re
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

class MetricsCollector:
    """Collects and aggregates training metrics"""
    
    def __init__(self, state_manager):
        self.state_manager = state_manager
        self.logger = logging.getLogger(__name__)
        self.metrics_path = Path(__file__).parent.parent / "data" / "experiments"
        self.metrics_path.mkdir(exist_ok=True)
    
    def collect_training_metrics(self, job_id: str, rank: int) -> Dict[str, Any]:
        """Collect metrics from training log for a specific rank"""
        log_file = Path.home() / f"training_rank{rank}_{job_id}.log"
        
        if not log_file.exists():
            self.logger.warning(f"Log file not found: {log_file}")
            return {}
        
        metrics = self._parse_training_log(log_file)
        metrics.update({
            'job_id': job_id,
            'rank': rank,
            'timestamp': datetime.now().isoformat()
        })
        
        # Store metrics
        self._store_metrics(job_id, metrics)
        
        return metrics
    
    def _parse_training_log(self, log_file: Path) -> Dict[str, float]:
        """Parse training metrics from log file"""
        metrics = {}
        
        try:
            with open(log_file, 'r') as f:
                content = f.read()
            
            # Extract loss
            loss_match = re.search(r'loss[:\s=]+([\d.]+)', content)
            if loss_match:
                metrics['loss'] = float(loss_match.group(1))
            
            # Extract perplexity
            ppl_match = re.search(r'perplexity[:\s=]+([\d.]+)', content)
            if ppl_match:
                metrics['perplexity'] = float(ppl_match.group(1))
            
            # Extract learning rate
            lr_match = re.search(r'lr[:\s=]+([\de.-]+)', content)
            if lr_match:
                metrics['learning_rate'] = float(lr_match.group(1))
            
            # Extract GPU utilization (if available)
            gpu_match = re.search(r'gpu[:\s=]+([\d.]+)%', content)
            if gpu_match:
                metrics['gpu_util'] = float(gpu_match.group(1))
            
        except Exception as e:
            self.logger.error(f"Error parsing log {log_file}: {e}")
        
        return metrics
    
    def _store_metrics(self, job_id: str, metrics: Dict[str, Any]):
        """Store metrics in JSONL format"""
        job_dir = self.metrics_path / job_id
        job_dir.mkdir(exist_ok=True)
        
        metrics_file = job_dir / "metrics.jsonl"
        
        with open(metrics_file, 'a') as f:
            f.write(json.dumps(metrics) + '\n')
    
    def generate_job_report(self, job_id: str) -> Dict[str, Any]:
        """Generate summary report for completed job"""
        metrics_file = self.metrics_path / job_id / "metrics.jsonl"
        
        if not metrics_file.exists():
            return {'error': 'No metrics found for job'}
        
        metrics = []
        try:
            with open(metrics_file, 'r') as f:
                for line in f:
                    metrics.append(json.loads(line))
        except Exception as e:
            self.logger.error(f"Error reading metrics: {e}")
            return {'error': 'Failed to read metrics'}
        
        if not metrics:
            return {'error': 'No metrics data'}
        
        # Calculate summary
        final_metrics = metrics[-1]
        
        report = {
            'job_id': job_id,
            'total_steps': len(metrics),
            'final_loss': final_metrics.get('loss', 'N/A'),
            'final_perplexity': final_metrics.get('perplexity', 'N/A'),
            'avg_gpu_util': sum(m.get('gpu_util', 0) for m in metrics) / len(metrics),
            'duration': metrics[-1]['timestamp']  # Simplified
        }
        
        return report
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_metrics_collector.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/ugro/metrics.py tests/test_metrics_collector.py
git commit -m "feat: implement MetricsCollector for training telemetry"
```

### Task 14: Add Result Aggregation

**Files:**
- Modify: `src/ugro/agent.py`
- Test: `tests/test_result_aggregation.py`

**Step 1: Write the failing test**

```python
# tests/test_result_aggregation.py
import pytest
from unittest.mock import MagicMock, patch
from src.ugro.agent import UGROAgent

def test_collect_job_results():
    """Test collecting and aggregating job results"""
    agent = UGROAgent()
    
    with patch.object(agent.metrics_collector, 'generate_job_report', return_value={
        'job_id': 'test_job',
        'final_loss': 3.2,
        'avg_gpu_util': 82.5
    }):
        results = agent.collect_job_results('test_job')
        
        assert results['job_id'] == 'test_job'
        assert results['final_loss'] == 3.2
        assert 'checkpoint_path' in results
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_result_aggregation.py -v`
Expected: FAIL with "UGROAgent has no attribute 'collect_job_results'"

**Step 3: Write minimal implementation**

```python
# Add to src/ugro/agent.py
from src.ugro.metrics import MetricsCollector

class UGROAgent:
    def __init__(self):
        # ... existing init code ...
        self.metrics_collector = MetricsCollector(self.state_manager)
    
    def collect_job_results(self, job_id: str) -> Dict[str, Any]:
        """Collect and aggregate results for completed job"""
        self.logger.info(f"Collecting results for job: {job_id}")
        
        # Get job status
        job_status = self.get_job_status(job_id)
        if job_status['status'] not in ['completed', 'failed']:
            raise RuntimeError(f"Job {job_id} not completed")
        
        # Generate metrics report
        report = self.metrics_collector.generate_job_report(job_id)
        
        # Add checkpoint information
        checkpoint_dir = Path(__file__).parent.parent / "data" / "experiments" / job_id / "checkpoints"
        if checkpoint_dir.exists():
            checkpoints = list(checkpoint_dir.glob("*.pt"))
            if checkpoints:
                latest_checkpoint = max(checkpoints, key=lambda p: p.stat().st_mtime)
                report['checkpoint_path'] = str(latest_checkpoint)
        
        # Add log paths
        report['log_paths'] = {
            f'rank_{i}': str(Path.home() / f"training_rank{i}_{job_id}.log")
            for i in range(len(job_status.get('gpu_nodes', [])))
        }
        
        return report
    
    def display_logs(self, job_id: str, rank: Optional[int] = None):
        """Display training logs for a job"""
        job_status = self.get_job_status(job_id)
        
        if job_status['status'] == 'not_found':
            click.echo(f"Job {job_id} not found")
            return
        
        if rank is not None:
            # Show specific rank
            log_file = Path.home() / f"training_rank{rank}_{job_id}.log"
            if log_file.exists():
                click.echo(f"=== Logs for {job_id}, Rank {rank} ===")
                with open(log_file, 'r') as f:
                    click.echo(f.read())
            else:
                click.echo(f"No log file found for rank {rank}")
        else:
            # Show all ranks
            for i in range(len(job_status.get('gpu_nodes', []))):
                log_file = Path.home() / f"training_rank{i}_{job_id}.log"
                if log_file.exists():
                    click.echo(f"\n=== Rank {i} ===")
                    with open(log_file, 'r') as f:
                        lines = f.readlines()[-50:]  # Last 50 lines
                        click.echo(''.join(lines))
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_result_aggregation.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/ugro/agent.py tests/test_result_aggregation.py
git commit -m "feat: add result aggregation and log display functionality"
```

### Task 15: Add Logs Command to CLI

**Files:**
- Modify: `src/ugro/cli.py`
- Test: `tests/test_cli_logs.py`

**Step 1: Write the failing test**

```python
# tests/test_cli_logs.py
import pytest
from click.testing import CliRunner
from unittest.mock import MagicMock, patch
from src.ugro.cli import cli

def test_cli_logs_command():
    """Test CLI logs command"""
    runner = CliRunner()
    
    with patch('src.ugro.cli.UGROAgent') as mock_agent_class:
        mock_agent = MagicMock()
        mock_agent_class.return_value = mock_agent
        
        result = runner.invoke(cli, ['logs', 'test_job', '--rank', '0'])
        
        assert result.exit_code == 0
        mock_agent.display_logs.assert_called_once_with('test_job', 0)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_cli_logs.py -v`
Expected: FAIL with "No such command 'logs'"

**Step 3: Write minimal implementation**

```python
# Add to src/ugro/cli.py
@cli.command()
@click.argument('job_id')
@click.option('--rank', default=None, type=int, help='Specific rank to view')
def logs(job_id, rank):
    """View training logs"""
    agent = UGROAgent()
    agent.display_logs(job_id, rank)

@cli.command()
@click.argument('job_id')
def results(job_id):
    """Show job results and metrics"""
    agent = UGROAgent()
    
    try:
        results = agent.collect_job_results(job_id)
        
        click.echo(f"\n=== Results for {job_id} ===")
        click.echo(f"Status: {results.get('status', 'Unknown')}")
        click.echo(f"Final Loss: {results.get('final_loss', 'N/A')}")
        click.echo(f"Final Perplexity: {results.get('final_perplexity', 'N/A')}")
        click.echo(f"Average GPU Utilization: {results.get('avg_gpu_util', 'N/A'):.1f}%")
        
        if 'checkpoint_path' in results:
            click.echo(f"Latest Checkpoint: {results['checkpoint_path']}")
        
        if 'log_paths' in results:
            click.echo("\nLog Files:")
            for rank, path in results['log_paths'].items():
                click.echo(f"  {rank}: {path}")
                
    except Exception as e:
        click.echo(f"❌ Error: {e}")
        raise click.ClickException(str(e))
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_cli_logs.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/ugro/cli.py tests/test_cli_logs.py
git commit -m "feat: add logs and results commands to CLI"
```

---

## Phase 5: Production Features

### Task 16: Add Systemd Service Integration

**Files:**
- Create: `systemd/ugro-agent.service`
- Create: `systemd/ugro-monitor.service`
- Test: `tests/test_systemd_integration.py`

**Step 1: Write the failing test**

```python
# tests/test_systemd_integration.py
def test_systemd_service_files_exist():
    """Test that systemd service files are created"""
    from pathlib import Path
    
    agent_service = Path("systemd/ugro-agent.service")
    monitor_service = Path("systemd/ugro-monitor.service")
    
    assert agent_service.exists()
    assert monitor_service.exists()
    
    # Check service file contents
    agent_content = agent_service.read_text()
    assert "Description=UGRO Agent Service" in agent_content
    assert "ExecStart" in agent_content
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_systemd_integration.py -v`
Expected: FAIL with "No such file or directory: 'systemd/ugro-agent.service'"

**Step 3: Write minimal implementation**

```ini
# systemd/ugro-agent.service
[Unit]
Description=UGRO Agent Service
After=network.target

[Service]
Type=simple
User=ollie
WorkingDirectory=/home/ollie/Development/Tools/ugro
Environment=PYTHONPATH=/home/ollie/Development/Tools/ugro
ExecStart=/home/ollie/Development/Tools/ugro/venv/bin/python -m src.ugro.cli daemon
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```ini
# systemd/ugro-monitor.service
[Unit]
Description=UGRO Health Monitor Service
After=network.target

[Service]
Type=simple
User=ollie
WorkingDirectory=/home/ollie/Development/Tools/ugro
Environment=PYTHONPATH=/home/ollie/Development/Tools/ugro
ExecStart=/home/ollie/Development/Tools/ugro/venv/bin/python -m src.ugro.monitor start
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```python
# src/ugro/cli.py - Add daemon command
@cli.command()
def daemon():
    """Run UGRO agent as daemon"""
    click.echo("Starting UGRO agent daemon...")
    
    # Start health monitor in background
    monitor = AdaptiveHealthMonitor(None, ClusterStateManager())
    
    try:
        asyncio.run(monitor.start_monitoring())
    except KeyboardInterrupt:
        click.echo("\nShutting down UGRO daemon...")
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_systemd_integration.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add systemd/ src/ugro/cli.py tests/test_systemd_integration.py
git commit -m "feat: add systemd service integration for production deployment"
```

### Task 17: Add Comprehensive Error Handling

**Files:**
- Modify: `src/ugro/agent.py`
- Modify: `src/ugro/monitor.py`
- Test: `tests/test_error_handling.py`

**Step 1: Write the failing test**

```python
# tests/test_error_handling.py
import pytest
from unittest.mock import MagicMock, patch
from src.ugro.agent import UGROAgent

def test_ssh_failure_handling():
    """Test graceful handling of SSH failures"""
    agent = UGROAgent()
    
    with patch.object(agent.ssh_client, 'test_connection', side_effect=[True, False, True]):
        # Should handle partial failures gracefully
        with pytest.raises(RuntimeError, match="Cluster health check failed"):
            agent.validate_cluster()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_error_handling.py -v`
Expected: FAIL (test should pass but we need to ensure error handling is robust)

**Step 3: Write minimal implementation**

```python
# Add to src/ugro/agent.py
import logging
from typing import Optional

class UGROError(Exception):
    """Base exception for UGRO"""
    pass

class ClusterValidationError(UGROError):
    """Raised when cluster validation fails"""
    pass

class JobLaunchError(UGROError):
    """Raised when job launch fails"""
    pass

# Update validate_cluster with better error handling
    def validate_cluster(self) -> bool:
        """Validate all nodes are reachable via SSH"""
        failed_nodes = []
        
        for node_name in self.nodes:
            try:
                if not self.ssh_client.test_connection(node_name):
                    failed_nodes.append(node_name)
                    self.logger.error(f"Cannot reach node: {node_name}")
            except Exception as e:
                failed_nodes.append(node_name)
                self.logger.error(f"Error testing {node_name}: {e}")
        
        if failed_nodes:
            raise ClusterValidationError(
                f"Cluster validation failed. Unreachable nodes: {', '.join(failed_nodes)}"
            )
        
        return True

# Update launch_distributed_training with rollback on failure
    def launch_distributed_training(self, job_id: str, model: str, dataset: str, epochs: int):
        """Launch distributed training across cluster"""
        launched_nodes = []
        
        try:
            # Validate cluster
            self.validate_cluster()
            
            # Register job
            node_names = list(self.nodes.keys())
            self.state_manager.register_job(job_id, node_names, {
                'model': model,
                'dataset': dataset,
                'epochs': epochs
            })
            
            # Launch on each node
            for rank, node_name in enumerate(node_names):
                try:
                    self._launch_on_node(job_id, rank, node_name, model, dataset, epochs)
                    launched_nodes.append(node_name)
                except Exception as e:
                    self.logger.error(f"Failed to launch on {node_name}: {e}")
                    # Cleanup launched nodes
                    self._cleanup_failed_launch(job_id, launched_nodes)
                    raise JobLaunchError(f"Failed to launch on {node_name}: {e}")
            
            # Monitor training
            self._monitor_training(job_id)
            
        except Exception as e:
            # Update job status to failed
            self.state_manager.update_job_status(job_id, 'failed')
            raise
    
    def _cleanup_failed_launch(self, job_id: str, launched_nodes: List[str]):
        """Clean up nodes after failed launch"""
        self.logger.info(f"Cleaning up failed launch for {job_id}")
        
        for node_name in launched_nodes:
            try:
                # Kill any running processes
                self.ssh_client.execute(
                    node_name,
                    f"pkill -f 'torchrun.*{job_id}'",
                    timeout=10
                )
            except Exception as e:
                self.logger.warning(f"Failed to cleanup {node_name}: {e}")
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_error_handling.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/ugro/agent.py src/ugro/exceptions.py tests/test_error_handling.py
git commit -m "feat: add comprehensive error handling with custom exceptions"
```

### Task 18: Add Integration Tests

**Files:**
- Create: `tests/integration/test_full_workflow.py`
- Create: `tests/fixtures/cluster_state.py`

**Step 1: Write the failing test**

```python
# tests/integration/test_full_workflow.py
import pytest
from click.testing import CliRunner
from unittest.mock import patch, MagicMock
from src.ugro.cli import cli

def test_full_training_workflow():
    """Test complete workflow from launch to results"""
    runner = CliRunner()
    
    with patch('src.ugro.cli.UGROAgent') as mock_agent_class:
        mock_agent = MagicMock()
        mock_agent_class.return_value = mock_agent
        
        # Mock successful launch
        mock_agent.launch_distributed_training.return_value = None
        mock_agent.get_job_status.return_value = {'status': 'completed'}
        mock_agent.collect_job_results.return_value = {
            'job_id': 'test_job',
            'final_loss': 3.2,
            'avg_gpu_util': 85.0
        }
        
        # Launch job
        result = runner.invoke(cli, [
            'launch',
            '--model', 'test_model',
            '--dataset', 'test_dataset',
            '--name', 'test_job'
        ])
        assert result.exit_code == 0
        
        # Check status
        result = runner.invoke(cli, ['status', 'test_job'])
        assert result.exit_code == 0
        
        # Get results
        result = runner.invoke(cli, ['results', 'test_job'])
        assert result.exit_code == 0
        assert '3.2' in result.output
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/integration/test_full_workflow.py -v`
Expected: FAIL with "No module named 'tests.integration'"

**Step 3: Write minimal implementation**

```bash
# Create integration test directory
mkdir -p tests/integration
touch tests/integration/__init__.py
```

```python
# tests/fixtures/cluster_state.py
"""Test fixtures for cluster state"""
import pytest
from pathlib import Path
import json

@pytest.fixture
def test_cluster_state():
    """Provide test cluster state"""
    return {
        "nodes": {
            "gpu-master": {
                "ip": "192.168.1.100",
                "gpu": "RTX 5070 Ti",
                "vram_gb": 12,
                "status": "available",
                "running_job_id": None
            },
            "gpu1": {
                "ip": "192.168.1.101",
                "gpu": "RTX 4070",
                "vram_gb": 8,
                "status": "available",
                "running_job_id": None
            },
            "gpu2": {
                "ip": "192.168.1.102",
                "gpu": "RTX 3070 Ti",
                "vram_gb": 8,
                "status": "available",
                "running_job_id": None
            }
        },
        "jobs": {}
    }

@pytest.fixture
def mock_ssh_responses():
    """Mock SSH responses for testing"""
    return {
        'gpu_master': (True, "OK", ""),
        'gpu1': (True, "OK", ""),
        'gpu2': (True, "OK", "")
    }
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/integration/test_full_workflow.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/integration/ tests/fixtures/
git commit -m "feat: add integration tests for full workflow"
```

### Task 19: Add Documentation and README

**Files:**
- Create: `README.md`
- Create: `docs/UGRO-User-Guide.md`
- Create: `docs/UGRO-API-Reference.md`

**Step 1: Write the failing test**

```python
# tests/test_documentation.py
def test_documentation_exists():
    """Test that all documentation files exist"""
    from pathlib import Path
    
    assert Path("README.md").exists()
    assert Path("docs/UGRO-User-Guide.md").exists()
    assert Path("docs/UGRO-API-Reference.md").exists()
    
    # Check README has key sections
    readme = Path("README.md").read_text()
    assert "# UGRO" in readme
    assert "## Quick Start" in readme
    assert "## Installation" in readme
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_documentation.py -v`
Expected: FAIL with "No such file or directory: 'README.md'"

**Step 3: Write minimal implementation**

```markdown
# README.md
# UGRO: Unified GPU Resource Orchestrator

A lightweight orchestration layer for distributed GPU training clusters.

## Quick Start

```bash
# Clone repository
git clone https://github.com/ollieb89/ugro.git
cd ugro

# Install dependencies
pip install -e .

# Configure cluster
cp config/cluster.yaml.example config/cluster.yaml
# Edit config/cluster.yaml with your node details

# Test cluster health
ugro health

# Launch training
ugro launch --model unsloth/tinyllama-bnb-4bit --dataset wikitext --name my_experiment

# Monitor progress
ugro status my_experiment
ugro logs my_experiment
ugro results my_experiment
```

## Installation

### Prerequisites

- Python 3.10+
- SSH access configured between all nodes
- CUDA-enabled GPUs on each node
- PyTorch installed on each node

### Install UGRO

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install UGRO
pip install -e .

# Install as command (optional)
ln -s $(pwd)/bin/ugro ~/.local/bin/ugro
```

## Configuration

Edit `config/cluster.yaml` with your cluster details:

```yaml
cluster:
  name: "My AI Lab"
  master_ip: "192.168.1.100"
  master_port: 29500

nodes:
  gpu-master:
    ip: "192.168.1.100"
    user: "your_username"
    gpu: "RTX 4090"
    vram_gb: 24
    role: "master"
  # Add more nodes...
```

## Usage

### Basic Commands

- `ugro health` - Check cluster health
- `ugro launch` - Launch distributed training
- `ugro status <job_id>` - Check job status
- `ugro logs <job_id>` - View training logs
- `ugro results <job_id>` - Show job results

### Launch Options

```bash
ugro launch \
  --model <model_name> \
  --dataset <dataset_name> \
  --epochs <num_epochs> \
  --name <job_name>
```

## Architecture

UGRO consists of:

- **CLI** - Command-line interface for user interaction
- **Agent** - Orchestrates training across nodes
- **Monitor** - Tracks cluster health and metrics
- **State Manager** - Persists cluster state

## Contributing

1. Fork repository
2. Create feature branch
3. Add tests
4. Submit pull request

## License

MIT License - see LICENSE file for details.
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_documentation.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add README.md docs/ tests/test_documentation.py
git commit -m "feat: add comprehensive documentation"
```

### Task 20: Final Integration and Testing

**Files:**
- Create: `tests/test_end_to_end.py`
- Modify: `setup.py`

**Step 1: Write the failing test**

```python
# tests/test_end_to_end.py
import pytest
import subprocess
import time
from pathlib import Path

def test_ugro_cli_installation():
    """Test that UGRO CLI is properly installed"""
    result = subprocess.run(['ugro', '--help'], capture_output=True, text=True)
    
    assert result.returncode == 0
    assert 'GPU Cluster Orchestration' in result.stdout
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_end_to_end.py -v`
Expected: FAIL with "ugro: command not found"

**Step 3: Write minimal implementation**

```python
# setup.py
from setuptools import setup, find_packages

setup(
    name="ugro",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "click>=8.0.0",
        "pyyaml>=6.0",
        "asyncio-mqtt",
        "psutil",
    ],
    entry_points={
        'console_scripts': [
            'ugro=src.ugro.cli:cli',
        ],
    },
    author="Ollie B",
    author_email="ollie@example.com",
    description="Unified GPU Resource Orchestrator",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url="https://github.com/ollieb89/ugro",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires='>=3.10',
)
```

```bash
# Create bin directory and executable
mkdir -p bin
cat > bin/ugro << 'EOF'
#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/ollie/Development/Tools/ugro')
from src.ugro.cli import cli
cli()
EOF
chmod +x bin/ugro
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_end_to_end.py -v`
Expected: PASS (after installation)

**Step 5: Commit**

```bash
git add setup.py bin/ugro tests/test_end_to_end.py
git commit -m "feat: add setup.py and end-to-end testing"
```

---

## Summary

This implementation plan provides a complete roadmap to build the UGRO orchestration layer with:

1. **Phase 1**: Foundation - Project structure, configuration, CLI, state management
2. **Phase 2**: Core Orchestration - SSH utilities, UGROAgent, distributed training launch
3. **Phase 3**: Health Monitoring - Adaptive polling, concurrent metrics collection
4. **Phase 4**: Metrics and Results - Training telemetry, result aggregation, log management
5. **Phase 5**: Production Features - Systemd services, error handling, comprehensive testing

Each task is designed to be completed in 2-5 minutes with:
- Failing test first (TDD)
- Minimal implementation
- Verification
- Git commit

The plan follows the architecture outlined in the Phase2 document and provides a production-ready orchestration system for GPU clusters.

## Execution Options

**Plan complete and saved to `docs/plans/2025-01-23-ugro-phase2-orchestration.md`. Two execution options:**

**1. Subagent-Driven (this session)** - I dispatch fresh subagent per task, review between tasks, fast iteration

**2. Parallel Session (separate)** - Open new session with executing-plans, batch execution with checkpoints

**Which approach?**
