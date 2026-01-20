# UGRO: Complete Project Setup Guide
## Fresh Start with SSH + Pixi Environments

**Status:** Building fresh project from proven infrastructure
- ✅ SSH passwordless configured
- ✅ Pixi environments installed
- ✅ Ready for clean project initialization

---

## Table of Contents

1. [Project Architecture](#project-architecture)
2. [Directory Structure](#directory-structure)
3. [Setup Instructions](#setup-instructions)
4. [Configuration](#configuration)
5. [Implementation](#implementation)
6. [Testing & Validation](#testing--validation)

---

## Project Architecture

### Your 3-Node Cluster

```
┌─────────────────────────────────────────────────────────┐
│              GPU Cluster (LAN 192.168.1.x)              │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  gpu-master (192.168.1.100)  ← Control Plane           │
│  ├─ RTX 5070 Ti (12GB)                                 │
│  ├─ Pixi env: main                                     │
│  └─ UGRO orchestrator runs here                        │
│                                                         │
│  gpu1 (192.168.1.101, user: ob)  ← Worker             │
│  ├─ RTX 4070 (8GB)                                     │
│  ├─ Pixi env: main (synced)                            │
│  └─ SSH passwordless ✓                                 │
│                                                         │
│  gpu2 (192.168.1.102, user: ollie)  ← Worker          │
│  ├─ RTX 3070 Ti (8GB)                                  │
│  ├─ Pixi env: main (synced)                            │
│  └─ SSH passwordless ✓                                 │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Project Layers

```
UGRO Project
├── Layer 1: Infrastructure (your existing setup)
│   ├─ SSH passwordless auth ✓
│   ├─ Pixi environments ✓
│   └─ Network connectivity ✓
│
├── Layer 2: Orchestration (new - building)
│   ├─ Cluster discovery & health
│   ├─ Job scheduling & launching
│   └─ Result collection & tracking
│
├── Layer 3: Training (reuse existing scripts)
│   ├─ train_production.py
│   ├─ train_single_test.py
│   └─ Model/dataset configs
│
├── Layer 4: Monitoring (future)
│   ├─ Metrics collection
│   ├─ Web dashboard
│   └─ Experiment comparison
│
└── Layer 5: Serving (future)
    ├─ Model deployment
    └─ Inference API
```

---

## Directory Structure

### On gpu-master (Control Node)

Create this complete structure:

```
~/projects/UGRO/                          ← Main project root
├── .gitignore
├── README.md
├── pyproject.toml                        ← Pixi/Poetry config
├── pixi.lock
│
├── src/                                  ← Orchestration source code
│   ├── __init__.py
│   ├── __main__.py                       ← Entry point
│   ├── cli.py                            ← CLI commands
│   ├── agent.py                          ← Main orchestrator
│   ├── config.py                         ← Configuration management
│   ├── ssh_utils.py                      ← SSH operations
│   ├── cluster.py                        ← Cluster state
│   ├── job.py                            ← Job management
│   ├── metrics.py                        ← Metrics collection
│   └── utils.py                          ← Helpers
│
├── config/                               ← Configuration files
│   ├── cluster.yaml                      ← Your 3-node definition
│   ├── training_defaults.yaml            ← Training parameters
│   └── logging.yaml                      ← Logging config
│
├── scripts/                              ← Reused training scripts
│   ├── train_production.py               ← From existing setup
│   ├── train_single_test.py              ← From existing setup
│   └── requirements.txt                  ← Training dependencies
│
├── data/                                 ← Runtime data
│   ├── experiments/                      ← Job results
│   │   └── exp_001_20260120_120000/
│   │       ├── config.json
│   │       ├── metrics.jsonl
│   │       ├── logs/
│   │       ├── checkpoints/
│   │       └── artifacts/
│   ├── cluster_state.json               ← Current cluster state
│   └── job_history.jsonl                ← All past jobs
│
├── logs/                                 ← Application logs
│   ├── agent.log                         ← UGRO agent logs
│   ├── launcher.log                      ← Job launch logs
│   └── health.log                        ← Health check logs
│
├── tests/                                ← Unit tests
│   ├── __init__.py
│   ├── test_ssh_utils.py
│   ├── test_cluster.py
│   └── test_agent.py
│
├── docs/                                 ← Documentation
│   ├── ARCHITECTURE.md
│   ├── SETUP.md
│   ├── API.md
│   ├── TROUBLESHOOTING.md
│   └── examples/
│
└── tools/                                ← Utility scripts
    ├── init_cluster.sh                   ← First-time setup
    ├── sync_envs.sh                      ← Sync pixi to workers
    ├── health_check.sh                   ← Quick health check
    └── reset_cluster.sh                  ← Emergency reset

```

### On Workers (gpu1, gpu2)

Keep lightweight - just training scripts and environments:

```
~/projects/UGRO/                          ← Same paths
├── scripts/
│   ├── train_production.py               ← Synced from master
│   └── train_single_test.py              ← Synced from master
│
└── .pixi/                                ← Pixi environment
    └── (managed by pixi via sync)
```

---

## Setup Instructions

### Step 1: Create Project Root (5 min)

Run **on gpu-master**:

```bash
# Create project directory
mkdir -p ~/projects/UGRO
cd ~/projects/UGRO

# Create all subdirectories
mkdir -p src config scripts data/{experiments,jobs} logs tests docs/examples tools

# Create __init__.py files
touch src/__init__.py tests/__init__.py

# Create .gitignore
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Pixi
.pixi/
pixi.lock

# IDE
.vscode/
.idea/
*.swp
*.swo

# Project
.env
.env.local
*.log
data/experiments/*/logs/
data/experiments/*/checkpoints/
data/*.json
data/*.jsonl

# OS
.DS_Store
Thumbs.db
EOF

# Initialize git (optional but recommended)
git init
git add .gitignore
git commit -m "Initial project structure"

# Verify structure
tree -L 2 ~/projects/UGRO
```

### Step 2: Create Pixi Configuration (5 min)

**File: `~/projects/UGRO/pyproject.toml`**

```toml
[project]
name = "ugro"
version = "0.1.0"
description = "Unified GPU Resource Orchestrator - Personal scale GPU cluster management"
authors = [{name = "Your Name", email = "your.email@example.com"}]
requires-python = ">=3.10"
dependencies = [
    "click>=8.1.0",
    "pyyaml>=6.0",
    "pydantic>=2.0",
    "paramiko>=3.0",
    "psutil>=5.9.0",
    "requests>=2.31.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-cov>=4.0",
    "black>=23.0",
    "ruff>=0.1.0",
    "mypy>=1.0",
]
monitor = [
    "prometheus-client>=0.17.0",
    "flask>=2.3.0",
]

[project.scripts]
ugro = "src.cli:main"

[build-system]
requires = ["setuptools>=65.0", "wheel"]
build-backend = "setuptools.build_meta"

[tool.black]
line-length = 100
target-version = ["py310", "py311"]

[tool.ruff]
line-length = 100
target-version = "py310"

[tool.mypy]
python_version = "3.10"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = false
```

### Step 3: Initialize Pixi Environment (5 min)

```bash
cd ~/projects/UGRO

# Install pixi (if not already done)
curl -fsSL https://pixi.sh/install.sh | bash

# Create pixi environment
pixi init --format pyproject

# Verify
pixi env list

# You should see:
# default (~/projects/UGRO/.pixi/envs/default)
```

### Step 4: Create Configuration Files (10 min)

**File: `~/projects/UGRO/config/cluster.yaml`**

```yaml
# Cluster Configuration
# This defines your 3-node GPU cluster

cluster:
  name: "Home AI Lab"
  location: "Trondheim, Norway"
  description: "Personal-scale GPU cluster for LLM training"
  
  # Master node (orchestrator)
  master:
    hostname: "gpu-master"
    ip: "192.168.1.100"
    port: 22
    user: "${USER}"  # Current user on master
  
  # Communication settings
  communication:
    backend: "c10d"  # PyTorch distributed backend
    master_port: 29500
    timeout_seconds: 300

# Worker nodes
workers:
  - name: "gpu1"
    hostname: "gpu1"
    ip: "192.168.1.101"
    user: "ob"
    ssh_port: 22
    rank: 1
    
    hardware:
      gpu_model: "RTX 4070"
      vram_gb: 8
      cpu_cores: 22
      ram_gb: 16
    
    paths:
      home: "/home/ob"
      project: "/home/ob/projects/UGRO"
      scripts: "/home/ob/projects/UGRO/scripts"
  
  - name: "gpu2"
    hostname: "gpu2"
    ip: "192.168.1.102"
    user: "ollie"
    ssh_port: 22
    rank: 2
    
    hardware:
      gpu_model: "RTX 3070 Ti"
      vram_gb: 8
      cpu_cores: 16
      ram_gb: 32
    
    paths:
      home: "/home/ollie"
      project: "/home/ollie/projects/UGRO"
      scripts: "/home/ollie/projects/UGRO/scripts"

# Local paths (on master)
paths:
  project_root: "~/projects/UGRO"
  scripts: "~/projects/UGRO/scripts"
  config: "~/projects/UGRO/config"
  data: "~/projects/UGRO/data"
  logs: "~/projects/UGRO/logs"
  experiments: "~/projects/UGRO/data/experiments"

# Training defaults
training:
  default_model: "unsloth/tinyllama-bnb-4bit"
  default_dataset: "wikitext"
  default_epochs: 1
  default_learning_rate: 0.0002
  
  batch_size_per_gpu: 1
  gradient_accumulation_steps: 8
  max_seq_length: 2048
  
  lora:
    rank: 16
    alpha: 32
    dropout: 0.05
    target_modules: ["q_proj", "v_proj"]

# Environment settings
environment:
  conda_env: "main"  # Your pixi environment name
  python_version: "3.11"
  torch_version: "2.1.0"
  cuda_version: "12.1"

# Logging
logging:
  level: "INFO"
  format: "[%(asctime)s] %(name)s - %(levelname)s - %(message)s"
  file: "~/projects/UGRO/logs/agent.log"
  keep_days: 7
```

**File: `~/projects/UGRO/config/training_defaults.yaml`**

```yaml
# Default training configuration

model:
  name: "unsloth/tinyllama-bnb-4bit"
  max_seq_length: 2048
  dtype: "float16"
  load_in_4bit: true

dataset:
  name: "wikitext"
  split: "train"
  cache_dir: "~/.cache/huggingface/datasets"

training:
  num_epochs: 1
  learning_rate: 0.0002
  warmup_steps: 100
  weight_decay: 0.01
  
  batch_size_per_gpu: 1
  gradient_accumulation_steps: 8
  max_grad_norm: 1.0
  
  save_strategy: "epoch"
  evaluation_strategy: "epoch"
  eval_steps: 100

optimizer:
  type: "adamw"
  betas: [0.9, 0.999]
  eps: 1e-8

lora:
  enabled: true
  rank: 16
  alpha: 32
  dropout: 0.05
  target_modules:
    - "q_proj"
    - "v_proj"
  bias: "none"

quantization:
  enabled: true
  bits: 4
  compute_dtype: "float16"

logging:
  log_level: "INFO"
  steps: 10
  use_tensorboard: true
  tensorboard_dir: "~/projects/UGRO/data/experiments/{job_id}/tensorboard"
```

### Step 5: Create Entry Point (5 min)

**File: `~/projects/UGRO/src/__main__.py`**

```python
"""UGRO entry point"""

from src.cli import main

if __name__ == "__main__":
    main()
```

**File: `~/projects/UGRO/src/cli.py`**

```python
#!/usr/bin/env python3
"""
UGRO CLI: Main command interface

Usage:
    ugro health
    ugro launch --name exp1 --epochs 3
    ugro logs exp1
    ugro results exp1
    ugro status
"""

import click
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent import UGROAgent
from src.config import load_config

@click.group()
@click.pass_context
def cli(ctx):
    """UGRO: Unified GPU Resource Orchestrator
    
    Personal-scale GPU cluster orchestration tool.
    
    Quick Start:
      ugro health          # Check cluster
      ugro launch          # Start training
      ugro logs <name>     # View logs
      ugro results <name>  # See results
    """
    ctx.ensure_object(dict)
    ctx.obj['agent'] = UGROAgent()
    ctx.obj['config'] = load_config()

@cli.command()
@click.pass_context
def health(ctx):
    """Check cluster health status"""
    agent = ctx.obj['agent']
    
    print("\n🔍 Cluster Health Check")
    print("=" * 60)
    
    health_status = agent.check_cluster_health()
    
    for node_name, status in health_status.items():
        symbol = "✓" if status['healthy'] else "❌"
        print(f"{symbol} {node_name:15} {status['message']}")
    
    print()

@cli.command()
@click.option('--name', required=True, help='Job name')
@click.option('--model', default='unsloth/tinyllama-bnb-4bit', help='Model name')
@click.option('--dataset', default='wikitext', help='Dataset name')
@click.option('--epochs', default=1, type=int, help='Number of epochs')
@click.option('--lr', default=0.0002, type=float, help='Learning rate')
@click.option('--verbose', is_flag=True, help='Verbose output')
@click.pass_context
def launch(ctx, name, model, dataset, epochs, lr, verbose):
    """Launch distributed training across cluster"""
    agent = ctx.obj['agent']
    
    success = agent.launch_training(
        job_name=name,
        model=model,
        dataset=dataset,
        epochs=epochs,
        learning_rate=lr,
        verbose=verbose,
    )
    
    sys.exit(0 if success else 1)

@cli.command()
@click.argument('job_name')
@click.option('--rank', default=None, type=int, help='Specific rank')
@click.pass_context
def logs(ctx, job_name, rank):
    """View training logs for a job"""
    agent = ctx.obj['agent']
    agent.display_logs(job_name, rank)

@cli.command()
@click.argument('job_name')
@click.pass_context
def results(ctx, job_name):
    """Show results summary for a job"""
    agent = ctx.obj['agent']
    agent.display_results(job_name)

@cli.command()
@click.pass_context
def status(ctx):
    """Show current cluster status"""
    agent = ctx.obj['agent']
    agent.display_status()

def main():
    """Main entry point"""
    cli(obj={})

if __name__ == '__main__':
    main()
```

### Step 6: Create Core Modules (20 min)

**File: `~/projects/UGRO/src/config.py`**

```python
"""Configuration management"""

from pathlib import Path
from typing import Dict, Any
import yaml

def get_config_dir() -> Path:
    """Get configuration directory"""
    return Path(__file__).parent.parent / "config"

def load_config(config_name: str = "cluster.yaml") -> Dict[str, Any]:
    """Load YAML configuration"""
    config_path = get_config_dir() / config_name
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    
    with open(config_path) as f:
        return yaml.safe_load(f)

def expand_paths(config: Dict[str, Any]) -> Dict[str, Any]:
    """Expand ~ and environment variables in paths"""
    import os
    
    if 'paths' in config:
        for key, value in config['paths'].items():
            if isinstance(value, str):
                config['paths'][key] = os.path.expanduser(value)
                config['paths'][key] = os.path.expandvars(config['paths'][key])
    
    return config
```

**File: `~/projects/UGRO/src/agent.py`**

```python
"""Main UGRO orchestration agent"""

from typing import Dict, List, Optional
from pathlib import Path
import json
from datetime import datetime
import subprocess
import time

from src.config import load_config, expand_paths
from src.ssh_utils import SSHClient
from src.cluster import Cluster
from src.job import Job

class UGROAgent:
    """Main orchestrator"""
    
    def __init__(self):
        self.config = expand_paths(load_config())
        self.cluster = Cluster(self.config)
        self.results_dir = Path.home() / "projects" / "UGRO" / "data" / "experiments"
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def check_cluster_health(self) -> Dict[str, Dict]:
        """Check health of all nodes"""
        return self.cluster.check_health()
    
    def launch_training(
        self,
        job_name: str,
        model: str,
        dataset: str,
        epochs: int = 1,
        learning_rate: float = 0.0002,
        verbose: bool = False,
    ) -> bool:
        """Launch distributed training"""
        
        print(f"\n{'='*60}")
        print(f"UGRO: Launching Distributed Training")
        print(f"{'='*60}")
        print(f"Job: {job_name}")
        print(f"Model: {model}")
        print(f"Dataset: {dataset}")
        print(f"Epochs: {epochs}\n")
        
        # Validate cluster
        print("🔍 Checking cluster...")
        health = self.check_cluster_health()
        
        if not all(h['healthy'] for h in health.values()):
            print("❌ Cluster health check failed!")
            return False
        
        print("✓ All nodes healthy\n")
        
        # Create job
        job = Job(
            name=job_name,
            model=model,
            dataset=dataset,
            epochs=epochs,
            learning_rate=learning_rate,
        )
        
        print(f"🚀 Launching {len(self.config['workers']) + 1} ranks...\n")
        
        # Launch ranks
        success = self._launch_ranks(job, verbose)
        
        if success:
            print(f"\n✅ Job {job_name} completed!")
            print(f"📁 Results: {job.result_dir}")
        
        return success
    
    def _launch_ranks(self, job: Job, verbose: bool = False) -> bool:
        """Launch training on all nodes"""
        # Implementation will go in next section
        pass
    
    def display_logs(self, job_name: str, rank: Optional[int] = None):
        """Display logs for a job"""
        print(f"Logs for {job_name}")
        # Implementation will go in next section
    
    def display_results(self, job_name: str):
        """Display results for a job"""
        print(f"Results for {job_name}")
        # Implementation will go in next section
    
    def display_status(self):
        """Display cluster status"""
        print("Cluster Status")
        # Implementation will go in next section
```

**File: `~/projects/UGRO/src/ssh_utils.py`**

```python
"""SSH utilities for remote execution"""

import subprocess
from typing import Tuple, Optional

class SSHClient:
    """Simple SSH client wrapper"""
    
    def __init__(self, host: str, user: str, port: int = 22):
        self.host = host
        self.user = user
        self.port = port
    
    def test_connection(self, timeout: int = 5) -> bool:
        """Test if SSH connection works"""
        cmd = f"ssh -o ConnectTimeout={timeout} {self.user}@{self.host} 'echo OK' > /dev/null 2>&1"
        return subprocess.call(cmd, shell=True) == 0
    
    def execute(self, command: str, timeout: Optional[int] = None) -> Tuple[int, str, str]:
        """Execute command on remote host"""
        ssh_cmd = f"ssh {self.user}@{self.host} '{command}'"
        
        try:
            result = subprocess.run(
                ssh_cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return -1, "", "Timeout"
```

**File: `~/projects/UGRO/src/cluster.py`**

```python
"""Cluster management"""

from typing import Dict, List, Any
from src.ssh_utils import SSHClient

class Cluster:
    """Cluster state and operations"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.master_config = config['cluster']['master']
        self.worker_configs = config['workers']
    
    def check_health(self) -> Dict[str, Dict]:
        """Check health of all nodes"""
        results = {}
        
        # Check master
        results['master'] = self._check_node_health(
            'master',
            self.master_config,
        )
        
        # Check workers
        for worker in self.worker_configs:
            results[worker['name']] = self._check_node_health(
                worker['name'],
                worker,
            )
        
        return results
    
    def _check_node_health(self, name: str, config: Dict) -> Dict:
        """Check single node health"""
        
        ssh = SSHClient(
            host=config['ip'],
            user=config.get('user', config.get('username')),
            port=config.get('ssh_port', 22),
        )
        
        # Test SSH
        ssh_ok = ssh.test_connection()
        
        if not ssh_ok:
            return {
                'healthy': False,
                'message': 'SSH unreachable',
                'node_name': name,
            }
        
        # Test GPU
        code, out, err = ssh.execute('nvidia-smi -L')
        gpu_ok = code == 0
        
        # Test PyTorch
        code, out, err = ssh.execute('python -c "import torch; print(torch.cuda.is_available())"')
        torch_ok = code == 0 and 'True' in out
        
        all_ok = ssh_ok and gpu_ok and torch_ok
        
        gpu_model = config.get('hardware', {}).get('gpu_model', 'Unknown')
        vram_gb = config.get('hardware', {}).get('vram_gb', '?')
        
        return {
            'healthy': all_ok,
            'message': f"{gpu_model} ({vram_gb}GB) - {'Ready' if all_ok else 'Unhealthy'}",
            'node_name': name,
            'ssh': ssh_ok,
            'gpu': gpu_ok,
            'torch': torch_ok,
        }
```

**File: `~/projects/UGRO/src/job.py`**

```python
"""Job management"""

from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
import json
from typing import Dict, Any

@dataclass
class Job:
    """Training job"""
    
    name: str
    model: str
    dataset: str
    epochs: int
    learning_rate: float
    created_at: str = None
    job_id: str = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
        
        if self.job_id is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.job_id = f"{self.name}_{timestamp}"
    
    @property
    def result_dir(self) -> Path:
        """Get job result directory"""
        return Path.home() / "projects" / "UGRO" / "data" / "experiments" / self.job_id
    
    def save_config(self):
        """Save job configuration"""
        self.result_dir.mkdir(parents=True, exist_ok=True)
        
        config_file = self.result_dir / "config.json"
        with open(config_file, 'w') as f:
            json.dump(asdict(self), f, indent=2)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
```

---

## Configuration

Your `cluster.yaml` is **production-ready** and defines:

✅ **Master node** (gpu-master, 192.168.1.100)
✅ **Worker nodes** (gpu1, gpu2 with users ob, ollie)
✅ **Hardware specs** (GPU models, VRAM, CPU cores)
✅ **Path mappings** (project directories on each machine)
✅ **Training defaults** (model, dataset, hyperparameters)
✅ **Communication settings** (PyTorch DDP backend, port)

---

## Testing & Validation

### Test 1: Verify Project Structure

```bash
cd ~/projects/UGRO

# Check directory layout
tree -L 2

# Should show complete structure created above
```

### Test 2: Verify Pixi Environment

```bash
cd ~/projects/UGRO

# List environments
pixi env list

# Activate environment
pixi shell

# Test Python
python --version

# Test imports
python -c "import click; import yaml; print('✓ Dependencies OK')"

# Exit
exit
```

### Test 3: Verify Configuration

```bash
cd ~/projects/UGRO

# Test config loading
pixi run python -c "
from src.config import load_config
config = load_config()
print('Cluster:', config['cluster']['name'])
print('Master:', config['cluster']['master']['hostname'])
print('Workers:', len(config['workers']))
"
```

### Test 4: Test SSH Connectivity

```bash
cd ~/projects/UGRO

# Test cluster health
pixi run python -c "
from src.agent import UGROAgent
agent = UGROAgent()
health = agent.check_cluster_health()
for name, status in health.items():
    print(f\"{'✓' if status['healthy'] else '❌'} {name}: {status['message']}\")
"
```

---

## Next Steps

### Immediate (Next 30 min)

1. ✅ Copy project structure to `~/projects/UGRO/`
2. ✅ Create all Python files from this guide
3. ✅ Create `cluster.yaml` with your node definitions
4. ✅ Run tests above to validate

### Short-term (This week)

1. Complete `agent.py` with full launch implementation
2. Add `display_logs()`, `display_results()`, `display_status()`
3. Test `ugro launch --name test1` end-to-end
4. Verify results saved to `~/projects/UGRO/data/experiments/`

### Medium-term (Next 2 weeks)

1. Add job queueing (multiple experiments in sequence)
2. Add metrics collection and visualization
3. Add health monitoring daemon
4. Create web dashboard

---

## Project Ready

You now have:
- ✅ Clean, organized project structure
- ✅ Pixi environment configuration
- ✅ Cluster definition as YAML
- ✅ Modular Python codebase
- ✅ SSH integration layer
- ✅ CLI interface skeleton
- ✅ Job management system

**This is a production-grade foundation ready to extend with features.**

Next document will show complete implementation of core functions.
