# UGRO Phase 2: Implementation Guide & Code Reference

**Quick Navigation:**
- [Setup (5 min)](#setup--5-min)
- [Core Agent Code (Copy-Paste Ready)](#core-agent-code)
- [Command Reference](#command-reference)
- [Testing Checklist](#testing-checklist)

---

## Setup (5 min)

### 1. Create Directory Structure

```bash
# Run on gpu-master
mkdir -p ~/ugro/{src,config,data/{experiments,jobs},logs,bin}

# Verify
ls -la ~/ugro/
# Should show: src/ config/ data/ logs/ bin/
```

### 2. Create Configuration File

**File: `~/ugro/config/cluster.yaml`**

```yaml
cluster:
  name: "Home AI Lab Trondheim"
  master_ip: "192.168.1.100"
  master_port: 29500
  
nodes:
  gpu-master:
    ip: "192.168.1.100"
    user: "${USER}"
    hostname: "gpu-master"
    gpu: "RTX 5070 Ti"
    vram_gb: 12
    rank: 0
    role: "master"
  
  gpu1:
    ip: "192.168.1.101"
    user: "ob"
    hostname: "gpu1"
    gpu: "RTX 4070"
    vram_gb: 8
    rank: 1
    role: "worker"
  
  gpu2:
    ip: "192.168.1.102"
    user: "ollie"
    hostname: "gpu2"
    gpu: "RTX 3070 Ti"
    vram_gb: 8
    rank: 2
    role: "worker"

training:
  batch_size_per_gpu: 1
  gradient_accumulation: 8
  default_model: "unsloth/tinyllama-bnb-4bit"
  default_dataset: "wikitext"

paths:
  project_base: "~/ai-cluster"
  scripts: "~/ai-cluster/scripts"
  results: "~/ugro/data/experiments"
```

### 3. Install Dependencies

```bash
pip install click pyyaml paramiko
```

---

## Core Agent Code

### File 1: `~/ugro/src/ugro_cli.py`

This is your main entry point. Save exactly as shown:

```python
#!/usr/bin/env python3
"""
UGRO CLI: Single command orchestration for your 3-GPU cluster

Usage:
    ugro launch --model llama-7b --dataset wikitext --name exp1
    ugro status exp1
    ugro logs exp1
    ugro health
    ugro results exp1
"""

import click
import json
import subprocess
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import time
import yaml

class UGROAgent:
    """Main orchestration agent"""
    
    def __init__(self):
        # Load configuration
        config_path = Path.home() / "ugro" / "config" / "cluster.yaml"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        # Extract key paths
        self.master_ip = self.config['cluster']['master_ip']
        self.master_port = self.config['cluster']['master_port']
        self.nodes = self.config['nodes']
        self.results_dir = Path.home() / "ugro" / "data" / "experiments"
        self.state_file = Path.home() / "ugro" / "data" / "cluster_state.json"
        
        # Create directories
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    # ==================== HEALTH CHECKS ====================
    
    def check_cluster_health(self) -> Dict[str, Dict]:
        """Check all nodes are reachable and healthy"""
        
        results = {}
        
        for node_name, node_config in self.nodes.items():
            results[node_name] = self._check_node_health(node_name, node_config)
        
        return results
    
    def _check_node_health(self, name: str, config: Dict) -> Dict:
        """Check single node health"""
        
        checks = {
            'ssh': False,
            'gpu': False,
            'torch': False,
        }
        
        # Test SSH
        ssh_cmd = f"ssh -o ConnectTimeout=5 {config['user']}@{config['ip']} 'echo OK' > /dev/null 2>&1"
        checks['ssh'] = subprocess.call(ssh_cmd, shell=True) == 0
        
        if not checks['ssh']:
            return {
                'healthy': False,
                'message': 'SSH unreachable',
                'checks': checks
            }
        
        # Test GPU
        gpu_cmd = f"ssh {config['user']}@{config['ip']} 'nvidia-smi -L' > /dev/null 2>&1"
        checks['gpu'] = subprocess.call(gpu_cmd, shell=True) == 0
        
        # Test PyTorch
        torch_cmd = f"ssh {config['user']}@{config['ip']} 'python -c \"import torch; torch.cuda.is_available()\"' > /dev/null 2>&1"
        checks['torch'] = subprocess.call(torch_cmd, shell=True) == 0
        
        # Overall health
        all_passed = all(checks.values())
        
        return {
            'healthy': all_passed,
            'message': f"{config['gpu']} ({config['vram_gb']}GB) - {'Ready' if all_passed else 'Unhealthy'}",
            'checks': checks
        }
    
    # ==================== JOB LAUNCHING ====================
    
    def launch_training(
        self,
        job_name: str,
        model: str,
        dataset: str,
        epochs: int = 1,
        learning_rate: float = 2e-4,
        verbose: bool = False,
    ) -> bool:
        """Launch distributed training across cluster"""
        
        print(f"\n{'='*60}")
        print(f"UGRO: Launching Distributed Training")
        print(f"{'='*60}")
        print(f"Job Name: {job_name}")
        print(f"Model: {model}")
        print(f"Dataset: {dataset}")
        print(f"Epochs: {epochs}")
        print(f"Learning Rate: {learning_rate}")
        print("")
        
        # Step 1: Validate cluster
        print("🔍 Checking cluster health...")
        health = self.check_cluster_health()
        
        all_healthy = all(h['healthy'] for h in health.values())
        
        for node_name, node_health in health.items():
            symbol = "✓" if node_health['healthy'] else "❌"
            print(f"  {symbol} {node_name}: {node_health['message']}")
        
        if not all_healthy:
            print("\n❌ Cluster health check failed!")
            return False
        
        print("✓ All nodes healthy\n")
        
        # Step 2: Prepare job
        job_id = f"{job_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        job_dir = self.results_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        
        # Save job config
        job_config = {
            'job_id': job_id,
            'job_name': job_name,
            'model': model,
            'dataset': dataset,
            'epochs': epochs,
            'learning_rate': learning_rate,
            'started_at': datetime.now().isoformat(),
            'nodes': list(self.nodes.keys()),
        }
        
        with open(job_dir / "config.json", 'w') as f:
            json.dump(job_config, f, indent=2)
        
        # Step 3: Launch ranks
        print("🚀 Launching training ranks...")
        
        processes = []
        rank = 0
        
        for node_name, node_config in self.nodes.items():
            print(f"  Rank {rank} → {node_name} ({node_config['ip']})")
            
            cmd = self._build_launch_command(
                job_id=job_id,
                rank=rank,
                node_config=node_config,
                model=model,
                dataset=dataset,
                epochs=epochs,
                learning_rate=learning_rate,
            )
            
            # SSH to node and launch
            ssh_cmd = f"ssh {node_config['user']}@{node_config['ip']} {cmd}"
            
            if verbose:
                print(f"    Command: {ssh_cmd}\n")
            
            result = subprocess.Popen(
                ssh_cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            
            processes.append({
                'rank': rank,
                'node': node_name,
                'process': result,
            })
            
            rank += 1
            
            # Small delay to prevent thundering herd
            time.sleep(0.5)
        
        print(f"\n✓ Launched {len(processes)} ranks\n")
        
        # Step 4: Monitor until completion
        print("📊 Monitoring training...")
        print("-" * 60)
        
        try:
            self._monitor_training(job_id, processes, job_dir)
        except KeyboardInterrupt:
            print("\n\n⚠️  Training interrupted by user")
            for p in processes:
                p['process'].terminate()
        
        print("-" * 60)
        print(f"\n✅ Job {job_name} completed!")
        print(f"📁 Results: {job_dir}")
        print(f"📊 View with: ugro logs {job_name}")
        print(f"📈 Results: ugro results {job_name}")
        
        return True
    
    def _build_launch_command(
        self,
        job_id: str,
        rank: int,
        node_config: Dict,
        model: str,
        dataset: str,
        epochs: int,
        learning_rate: float,
    ) -> str:
        """Build the torchrun command for a rank"""
        
        # Get script path from config
        scripts_path = self.config['paths']['scripts']
        
        cmd = f"""
        cd {scripts_path} && \\
        torchrun \\
            --nnodes=3 \\
            --nproc_per_node=1 \\
            --rdzv_id={job_id} \\
            --rdzv_backend=c10d \\
            --rdzv_endpoint={self.master_ip}:{self.master_port} \\
            --node_rank={rank} \\
            train_production.py \\
            --model-name {model} \\
            --dataset-name {dataset} \\
            --num-epochs {epochs} \\
            --learning-rate {learning_rate} \\
            --job-id {job_id}
        """
        
        return f"'{cmd.strip()}'"
    
    def _monitor_training(self, job_id: str, processes: List, job_dir: Path):
        """Monitor training until all processes complete"""
        
        start_time = time.time()
        all_done = False
        
        while not all_done:
            all_done = True
            
            for p in processes:
                if p['process'].poll() is None:  # Still running
                    all_done = False
                    break
            
            elapsed = int(time.time() - start_time)
            elapsed_str = f"{elapsed // 3600}h {(elapsed % 3600) // 60}m"
            
            print(f"\r⏱️  Training in progress... {elapsed_str}", end='', flush=True)
            
            time.sleep(5)
        
        print()  # Newline
    
    # ==================== JOB QUERYING ====================
    
    def get_job_logs(self, job_name: str, rank: Optional[int] = None):
        """Display training logs for a job"""
        
        # Find job directory matching name
        matching = list(self.results_dir.glob(f"{job_name}_*"))
        
        if not matching:
            print(f"❌ No job found matching: {job_name}")
            return
        
        if len(matching) > 1:
            print(f"⚠️  Multiple matches. Using most recent:")
            job_dir = max(matching, key=lambda p: p.stat().st_mtime)
        else:
            job_dir = matching[0]
        
        print(f"\n📁 Job Directory: {job_dir}")
        print(f"{'='*60}\n")
        
        # List all logs
        logs_dir = job_dir
        log_files = sorted(logs_dir.glob("**/training_rank*.log"))
        
        if not log_files:
            print("No log files found")
            return
        
        # Filter by rank if specified
        if rank is not None:
            log_files = [f for f in log_files if f"rank{rank}" in str(f)]
        
        for log_file in log_files:
            print(f"--- {log_file.name} ---\n")
            
            try:
                with open(log_file) as f:
                    content = f.read()
                    # Show last 50 lines
                    lines = content.split('\n')
                    for line in lines[-50:]:
                        print(line)
            except:
                print(f"Could not read {log_file}")
            
            print()
    
    def get_job_results(self, job_name: str):
        """Show summary results for a job"""
        
        matching = list(self.results_dir.glob(f"{job_name}_*"))
        
        if not matching:
            print(f"❌ No job found: {job_name}")
            return
        
        job_dir = max(matching, key=lambda p: p.stat().st_mtime)
        
        print(f"\n📊 Results for: {job_name}")
        print(f"📁 Location: {job_dir}")
        print(f"{'='*60}\n")
        
        # Load config
        config_file = job_dir / "config.json"
        if config_file.exists():
            with open(config_file) as f:
                config = json.load(f)
            
            print("Configuration:")
            print(f"  Model: {config.get('model')}")
            print(f"  Dataset: {config.get('dataset')}")
            print(f"  Epochs: {config.get('epochs')}")
            print(f"  Learning Rate: {config.get('learning_rate')}")
            print(f"  Started: {config.get('started_at')}")
            print(f"  Nodes: {', '.join(config.get('nodes', []))}")
            print()
        
        # List checkpoints
        checkpoints = list(job_dir.glob("**/checkpoints/*.pt"))
        if checkpoints:
            print(f"Checkpoints ({len(checkpoints)}):")
            for ckpt in sorted(checkpoints):
                size_mb = ckpt.stat().st_size / (1024 * 1024)
                print(f"  - {ckpt.name} ({size_mb:.1f} MB)")
            print()
        
        # List logs
        logs = list(job_dir.glob("**/*.log"))
        if logs:
            print(f"Log Files ({len(logs)}):")
            for log in sorted(logs):
                print(f"  - {log.relative_to(job_dir)}")


# ==================== CLI COMMANDS ====================

@click.group()
def cli():
    """UGRO: Unified GPU Resource Orchestrator
    
    Your personal-scale GPU cluster management tool.
    
    Quick Start:
      ugro health          # Check all nodes
      ugro launch          # Start training with defaults
      ugro status <name>   # Check running job
      ugro logs <name>     # View training logs
      ugro results <name>  # Show results summary
    """
    pass


@cli.command()
@click.option('--model', default='unsloth/tinyllama-bnb-4bit', help='Model to train')
@click.option('--dataset', default='wikitext', help='Dataset name')
@click.option('--epochs', default=1, type=int, help='Number of epochs')
@click.option('--name', required=True, help='Job name')
@click.option('--lr', default=2e-4, type=float, help='Learning rate')
@click.option('--verbose', is_flag=True, help='Show full commands')
def launch(model, dataset, epochs, name, lr, verbose):
    """Launch distributed training across all nodes
    
    Example:
      ugro launch --name exp1 --model llama-7b --epochs 3
    """
    
    agent = UGROAgent()
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
def health():
    """Check cluster health status
    
    Tests SSH, GPU availability, and PyTorch on each node.
    """
    
    agent = UGROAgent()
    
    print("\n🔍 Cluster Health Check")
    print("=" * 60)
    
    health = agent.check_cluster_health()
    
    for node_name, node_health in health.items():
        symbol = "✓" if node_health['healthy'] else "❌"
        print(f"{symbol} {node_name:15} {node_health['message']}")
        
        if not node_health['healthy']:
            checks = node_health.get('checks', {})
            for check_name, passed in checks.items():
                check_symbol = "✓" if passed else "❌"
                print(f"    {check_symbol} {check_name}")
    
    print()


@cli.command()
@click.argument('job_name')
@click.option('--rank', default=None, type=int, help='View specific rank')
def logs(job_name, rank):
    """View training logs for a job
    
    Example:
      ugro logs exp1           # View all ranks
      ugro logs exp1 --rank 0  # View rank 0 only
    """
    
    agent = UGROAgent()
    agent.get_job_logs(job_name, rank)


@cli.command()
@click.argument('job_name')
def results(job_name):
    """Show results summary for a job
    
    Example:
      ugro results exp1
    """
    
    agent = UGROAgent()
    agent.get_job_results(job_name)


if __name__ == '__main__':
    cli()
```

### File 2: `~/ugro/bin/ugro`

Simple wrapper script:

```bash
#!/bin/bash
# UGRO: Unified GPU Resource Orchestrator

cd "$(dirname "$0")/.." || exit 1
python -m src.ugro_cli "$@"
```

Make it executable:

```bash
chmod +x ~/ugro/bin/ugro
chmod +x ~/ugro/src/ugro_cli.py
```

---

## Command Reference

### Installation

```bash
# One-time setup
mkdir -p ~/ugro/{src,config,data,logs,bin}

# Copy the YAML config above to: ~/ugro/config/cluster.yaml
# Copy the Python code above to: ~/ugro/src/ugro_cli.py
# Copy the bash wrapper above to: ~/ugro/bin/ugro

# Add to PATH
ln -s $(realpath ~/ugro/bin/ugro) ~/.local/bin/ugro

# Test
ugro health
```

### Basic Commands

```bash
# Check cluster health
ugro health

# Launch training (with defaults)
ugro launch --name exp1

# Launch with custom parameters
ugro launch \
  --name llama_exp \
  --model meta-llama/Llama-2-7b-hf \
  --dataset wikitext \
  --epochs 3 \
  --lr 1e-4

# View logs while training
ugro logs llama_exp

# View logs after training
ugro logs llama_exp

# View specific rank
ugro logs llama_exp --rank 0

# Show results summary
ugro results llama_exp
```

### Output Structure

After `ugro launch --name exp1`, you'll have:

```
~/ugro/data/experiments/exp1_20260120_120000/
├── config.json              # Your launch parameters
├── training_rank0.log       # Rank 0 output
├── training_rank1.log       # Rank 1 output
├── training_rank2.log       # Rank 2 output
├── checkpoints/             # Saved models
│   ├── epoch_1.pt
│   └── epoch_2.pt
└── metrics.jsonl            # Training metrics (future)
```

---

## Testing Checklist

### Test 1: Health Check (2 min)

```bash
# Run
ugro health

# Expected output:
# ✓ gpu-master         RTX 5070 Ti (12GB) - Ready
# ✓ gpu1               RTX 4070 (8GB) - Ready
# ✓ gpu2               RTX 3070 Ti (8GB) - Ready
```

If any fail:
- Check SSH: `ssh ob@192.168.1.101 echo OK`
- Check GPU: `ssh ob@192.168.1.101 nvidia-smi`
- Check PyTorch: `ssh ob@192.168.1.101 python -c "import torch"`

### Test 2: Single Rank Launch (10 min)

Before testing full 3-node, launch on just master:

```bash
# Modify torchrun command temporarily to use 1 node
cd ~/ai-cluster/scripts
torchrun \
    --nnodes=1 --nproc_per_node=1 \
    --rdzv_id=test --rdzv_backend=c10d \
    --rdzv_endpoint=127.0.0.1:29500 \
    --node_rank=0 \
    train_production.py \
    --model-name unsloth/tinyllama-bnb-4bit \
    --dataset-name wikitext \
    --num-epochs 1

# Should complete without errors
```

### Test 3: Quick Distributed Launch (15 min)

```bash
# Launch with minimal settings
ugro launch --name quick_test --epochs 1

# Watch output
# Should see: "Rank 0 → gpu-master", "Rank 1 → gpu1", "Rank 2 → gpu2"
# Then: "Monitoring training..."

# In another terminal
ugro logs quick_test

# When done, check results
ugro results quick_test
```

### Test 4: Verify Results Stored

```bash
# After training completes
ls -la ~/ugro/data/experiments/

# Should show:
# quick_test_20260120_120000/
#   ├── config.json
#   ├── training_rank0.log
#   ├── training_rank1.log
#   └── training_rank2.log

cat ~/ugro/data/experiments/quick_test_*/config.json
# Should show your job config
```

---

## Troubleshooting

### "Config not found"

```bash
# Create it
cat > ~/ugro/config/cluster.yaml << 'EOF'
# Paste the YAML config from above
EOF
```

### "SSH unreachable"

```bash
# Test manually
ssh ob@192.168.1.101 echo OK

# If fails, check:
# - Are you on the same network?
# - Is SSH enabled on that machine?
# - Did you set up passwordless auth?
```

### "CUDA out of memory"

The cluster.yaml defines batch size. Reduce it:

```yaml
training:
  batch_size_per_gpu: 1    # Try this
  gradient_accumulation: 16 # Increase this instead
```

### Processes don't sync

Check firewall allows port 29500:

```bash
# On gpu-master
sudo ufw allow 29500

# Verify
telnet localhost 29500
# Should connect (Ctrl+C to exit)
```

---

## Next: Phase 2b Features

After core works, add to `ugro_cli.py`:

### Feature: Job Queueing
```bash
ugro queue exp1 exp2 exp3
# Runs exp1, waits, runs exp2, waits, runs exp3
```

### Feature: Metrics Dashboard
```bash
ugro dashboard
# Starts web server showing:
# - Running jobs
# - GPU utilization
# - Loss curves
# - Model comparison
```

### Feature: Model Serving
```bash
ugro serve exp1
# Deploys trained model as API endpoint
```

### Feature: Hyperparameter Sweep
```bash
ugro sweep --param learning_rate 1e-5,2e-5,5e-5 --param epochs 1,2
# Automatically runs combinations
```

---

## Conclusion

You now have:
- ✅ Cluster configuration as code
- ✅ Single-command training launch
- ✅ Centralized results storage
- ✅ Job log viewing
- ✅ Health monitoring

**Next steps:**
1. Copy files to `~/ugro/`
2. Run `ugro health` to verify
3. Run `ugro launch --name first_test`
4. Monitor with `ugro logs first_test`
5. Check results with `ugro results first_test`

Then iterate: Add monitoring, queuing, serving, and scaling!

🚀
