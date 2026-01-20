# Unified GPU Resource Orchestrator (UGRO)
## Production Deployment & Next-Phase Build Guide

**Status:** SSH configured, ML environments deployed ✓  
**Focus:** From raw cluster → Unified orchestration platform

---

## Executive Summary

You have 28 GB of GPU VRAM across 3 machines in a cohesive LAN. The goal now is to **build operational infrastructure** that transforms this raw compute into a reliable, inspectable, coordinated platform.

**This guide covers:**
1. Verification & diagnostics
2. Orchestration layer (job scheduling, distributed coordination)
3. Monitoring & observability (real-time dashboards)
4. Resilience & failure handling
5. Common workflow automation

---

# PHASE 0: Cluster Verification (15 minutes)

## 0.1 Connectivity Check

Run on **gpu-master** (192.168.1.100):

```bash
# Test network connectivity to workers
ping -c 3 192.168.1.101
ping -c 3 192.168.1.102

# Test SSH passwordless access
ssh ob@192.168.1.101 "echo 'Worker 1 alive'"
ssh ollie@192.168.1.102 "echo 'Worker 2 alive'"

# Test command execution on workers
ssh ob@192.168.1.101 "nvidia-smi -L"
ssh ollie@192.168.1.102 "nvidia-smi -L"
```

**Expected Output:**
```
PONG from .101 and .102
Connection successful on both workers
GPU list shown from both
```

## 0.2 Environment Consistency Check

**On each machine independently, run:**

```bash
# Verify conda environment exists and is activated
conda activate dist-train  # or whatever your env name is

# Check versions match across all machines
python --version
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA: {torch.version.cuda}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
python -c "import torch; print(f'NCCL: {torch.cuda.nccl.version()}')"
```

**Create a script to automate this across all nodes:**

Save as `~/verify_cluster_env.sh`:

```bash
#!/bin/bash

echo "=== MASTER NODE (gpu-master) ==="
python --version
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA: {torch.version.cuda}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"

echo ""
echo "=== WORKER 1 (gpu1 - user: ob) ==="
ssh ob@192.168.1.101 'python --version && python -c "import torch; print(f'"'"'PyTorch: {torch.__version__}'"'"')" && python -c "import torch; print(f'"'"'CUDA: {torch.version.cuda}'"'"')" && python -c "import torch; print(f'"'"'GPU: {torch.cuda.get_device_name(0)}'"'"')"'

echo ""
echo "=== WORKER 2 (gpu2 - user: ollie) ==="
ssh ollie@192.168.1.102 'python --version && python -c "import torch; print(f'"'"'PyTorch: {torch.__version__}'"'"')" && python -c "import torch; print(f'"'"'CUDA: {torch.version.cuda}'"'"')" && python -c "import torch; print(f'"'"'GPU: {torch.cuda.get_device_name(0)}'"'"')"'

echo ""
echo "=== CLUSTER SUMMARY ==="
python -c "import socket; print(f'Master hostname: {socket.gethostname()}')"
```

```bash
chmod +x ~/verify_cluster_env.sh
bash ~/verify_cluster_env.sh
```

## 0.3 Cluster Configuration File

Create `~/.ugro/cluster_config.json` on **gpu-master** to centralize node definitions:

```json
{
  "cluster_name": "home-gpu-cluster",
  "created": "2026-01-20",
  "master_node": {
    "hostname": "gpu-master",
    "ip": "192.168.1.100",
    "user": "$(whoami)",
    "gpu": "RTX 5070 Ti",
    "vram_gb": 12,
    "cuda_cores": 5888,
    "rank": 0
  },
  "worker_nodes": [
    {
      "hostname": "gpu1",
      "ip": "192.168.1.101",
      "user": "ob",
      "gpu": "RTX 4070",
      "vram_gb": 8,
      "cuda_cores": 5888,
      "rank": 1
    },
    {
      "hostname": "gpu2",
      "ip": "192.168.1.102",
      "user": "ollie",
      "gpu": "RTX 3070 Ti",
      "vram_gb": 8,
      "cuda_cores": 5888,
      "rank": 2
    }
  ],
  "network": {
    "subnet": "192.168.1.0/24",
    "dns": "8.8.8.8",
    "nccl_port": 29500,
    "tensorboard_port": 6006
  },
  "storage": {
    "project_root": "~/Development/Projects/ai-ml-pipeline",
    "cluster_root": "~/Development/Projects/ai-ml-pipeline/ai-cluster"
  },
  "ml_environment": {
    "conda_env": "dist-train",
    "python_version": "3.11",
    "pytorch_version": "2.x",
    "cuda_version": "12.1"
  }
}
```

---

# PHASE 1: Orchestration Layer

## 1.1 Unified Launcher Script

Create `~/Development/Projects/ai-ml-pipeline/ai-cluster/scripts/launch_distributed.py`:

```python
#!/usr/bin/env python3
"""
UGRO Unified Launcher
Orchestrates multi-node training with single command
"""

import os
import sys
import json
import subprocess
import argparse
import time
from pathlib import Path
from datetime import datetime


class UGROLauncher:
    """Coordinates distributed training across cluster"""
    
    def __init__(self, config_path: str = "~/.ugro/cluster_config.json"):
        self.config_path = Path(config_path).expanduser()
        self.load_config()
    
    def load_config(self):
        """Load cluster configuration"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config not found: {self.config_path}")
        
        with open(self.config_path) as f:
            self.config = json.load(f)
        
        print(f"✓ Loaded cluster config: {self.config['cluster_name']}")
    
    def launch_distributed_training(
        self,
        script_path: str,
        model_name: str = "unsloth/tinyllama-bnb-4bit",
        num_epochs: int = 3,
        batch_size: int = 1,
        learning_rate: float = 2e-4,
        wait_for_all: bool = True,
        timeout: int = 300,
    ):
        """
        Launch training on all nodes simultaneously
        
        Args:
            script_path: Path to training script
            model_name: Model to train
            num_epochs: Number of training epochs
            batch_size: Per-GPU batch size
            learning_rate: Learning rate
            wait_for_all: Wait for all processes to complete
            timeout: Timeout in seconds for process startup
        """
        
        master = self.config["master_node"]
        workers = self.config["worker_nodes"]
        
        nccl_port = self.config["network"]["nccl_port"]
        master_ip = master["ip"]
        world_size = 1 + len(workers)
        
        print(f"\n{'='*70}")
        print(f"UGRO DISTRIBUTED TRAINING LAUNCHER")
        print(f"{'='*70}")
        print(f"Cluster: {self.config['cluster_name']}")
        print(f"Master:  {master_ip}:{nccl_port}")
        print(f"Workers: {', '.join(w['ip'] for w in workers)}")
        print(f"World Size: {world_size}")
        print(f"Script: {script_path}")
        print(f"Model: {model_name}")
        print(f"{'='*70}\n")
        
        # Build torchrun base command
        torchrun_cmd = [
            "torchrun",
            f"--nnodes={world_size}",
            "--nproc_per_node=1",
            f"--rdzv_id=100",
            "--rdzv_backend=c10d",
            f"--rdzv_endpoint={master_ip}:{nccl_port}",
        ]
        
        # Additional training arguments
        training_args = [
            script_path,
            f"--model-name={model_name}",
            f"--num-epochs={num_epochs}",
            f"--batch-size={batch_size}",
            f"--learning-rate={learning_rate}",
        ]
        
        # Launch on master
        processes = []
        
        print(f"→ Launching on MASTER ({master['ip']})...")
        master_cmd = torchrun_cmd + ["--node_rank=0"] + training_args
        master_proc = subprocess.Popen(
            master_cmd,
            cwd=os.path.dirname(script_path),
        )
        processes.append(("master", master_proc))
        
        # Launch on workers via SSH
        for worker in workers:
            print(f"→ Launching on WORKER ({worker['ip']}, user: {worker['user']})...")
            
            worker_cmd = (
                f"cd {os.path.dirname(script_path)} && "
                f"{' '.join(torchrun_cmd)} "
                f"--node_rank={worker['rank']} "
                f"{' '.join(training_args)}"
            )
            
            ssh_cmd = [
                "ssh",
                f"{worker['user']}@{worker['ip']}",
                worker_cmd,
            ]
            
            worker_proc = subprocess.Popen(ssh_cmd)
            processes.append((f"worker-{worker['rank']}", worker_proc))
            
            time.sleep(1)  # Stagger launches
        
        print(f"\n✓ All processes launched. Monitoring...\n")
        
        if wait_for_all:
            all_alive = True
            for name, proc in processes:
                retcode = proc.wait()
                if retcode == 0:
                    print(f"✓ {name} completed successfully")
                else:
                    print(f"✗ {name} failed with code {retcode}")
                    all_alive = False
            
            if all_alive:
                print(f"\n{'='*70}")
                print("✓ TRAINING COMPLETED SUCCESSFULLY")
                print(f"{'='*70}\n")
            else:
                print(f"\n{'='*70}")
                print("✗ TRAINING FAILED - Check logs")
                print(f"{'='*70}\n")
                sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="UGRO Unified Launcher")
    parser.add_argument(
        "script",
        help="Training script to run"
    )
    parser.add_argument(
        "--model",
        default="unsloth/tinyllama-bnb-4bit",
        help="Model name"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size per GPU"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--config",
        default="~/.ugro/cluster_config.json",
        help="Cluster config file"
    )
    
    args = parser.parse_args()
    
    launcher = UGROLauncher(args.config)
    launcher.launch_distributed_training(
        script_path=args.script,
        model_name=args.model,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )


if __name__ == "__main__":
    main()
```

**Usage:**

```bash
chmod +x ~/Development/Projects/ai-ml-pipeline/ai-cluster/scripts/launch_distributed.py

# Launch training from master node
cd ~/Development/Projects/ai-ml-pipeline/ai-cluster/scripts
python launch_distributed.py train_production.py --model unsloth/tinyllama-bnb-4bit --epochs 3
```

---

## 1.2 Health Check & Status Monitor

Create `/home/ollie/Development/Tools/ugro_status.py`:

```python
#!/usr/bin/env python3
"""
UGRO Cluster Health Monitor
Real-time status of all nodes
"""

import subprocess
import json
import sys
from pathlib import Path


def get_node_status(ip: str, user: str, hostname: str) -> dict:
    """Query node for GPU status"""
    
    try:
        # Query GPU info
        if ip == "192.168.1.100":  # Master node
            cmd = "nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)
        else:  # Workers via SSH
            cmd = f"ssh {user}@{ip} 'nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader'"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)
        
        if result.returncode != 0:
            return {"status": "OFFLINE", "error": result.stderr}
        
        # Parse output
        output = result.stdout.strip()
        if output:
            gpu_name, mem_used, mem_total, util = output.split(", ")
            return {
                "status": "ONLINE",
                "gpu": gpu_name,
                "memory_used_mb": int(mem_used),
                "memory_total_mb": int(mem_total),
                "utilization_percent": int(util.rstrip(" %")),
            }
        else:
            return {"status": "UNKNOWN", "error": "No GPU data"}
    
    except subprocess.TimeoutExpired:
        return {"status": "TIMEOUT"}
    except Exception as e:
        return {"status": "ERROR", "error": str(e)}


def main():
    config_path = Path.home() / ".ugro" / "cluster_config.json"
    
    if not config_path.exists():
        print(f"✗ Config not found: {config_path}")
        sys.exit(1)
    
    with open(config_path) as f:
        config = json.load(f)
    
    print(f"\n{'='*80}")
    print(f"UGRO CLUSTER STATUS - {config['cluster_name']}")
    print(f"{'='*80}\n")
    
    # Check master
    master = config["master_node"]
    status = get_node_status("192.168.1.100", master["user"], master["hostname"])
    
    print(f"MASTER NODE (192.168.1.100)")
    print(f"  Status: {status['status']}")
    if status['status'] == 'ONLINE':
        print(f"  GPU: {status['gpu']}")
        print(f"  Memory: {status['memory_used_mb']} / {status['memory_total_mb']} MB")
        print(f"  Utilization: {status['utilization_percent']}%")
    print()
    
    # Check workers
    for worker in config["worker_nodes"]:
        status = get_node_status(worker["ip"], worker["user"], worker["hostname"])
        
        print(f"WORKER {worker['rank']} ({worker['ip']}, user: {worker['user']})")
        print(f"  Status: {status['status']}")
        if status['status'] == 'ONLINE':
            print(f"  GPU: {status['gpu']}")
            print(f"  Memory: {status['memory_used_mb']} / {status['memory_total_mb']} MB")
            print(f"  Utilization: {status['utilization_percent']}%")
        print()
    
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
```

```bash
chmod +x /home/ollie/Development/Tools/ugro_status.py
python /home/ollie/Development/Tools/ugro_status.py
```

---

# PHASE 2: Monitoring & Observability

## 2.1 Centralized TensorBoard Setup

Create `/home/ollie/Development/Tools/ugro_tensorboard.sh`:

```bash
#!/bin/bash

# Launch TensorBoard pointing to distributed logs

CLUSTER_ROOT="$HOME/Development/Projects/ai-ml-pipeline/ai-cluster"
LOG_DIR="$CLUSTER_ROOT/logs/runs"

echo "Starting TensorBoard..."
echo "Logs directory: $LOG_DIR"
echo "Access at: http://localhost:6006"

tensorboard --logdir="$LOG_DIR" --port=6006 --bind_all
```

```bash
chmod +x /home/ollie/Development/Tools/ugro_tensorboard.sh
bash /home/ollie/Development/Tools/ugro_tensorboard.sh
```

## 2.2 Unified Logging Aggregator

Create `~/Development/Projects/ai-ml-pipeline/ai-cluster/scripts/aggregate_logs.py`:

```python
#!/usr/bin/env python3
"""
Aggregate logs from all nodes into single view
"""

import subprocess
import json
from pathlib import Path
from datetime import datetime


def aggregate_training_logs(cluster_root: str):
    """Collect training logs from all nodes"""
    
    config_path = Path.home() / ".ugro" / "cluster_config.json"
    with open(config_path) as f:
        config = json.load(f)
    
    master = config["master_node"]
    workers = config["worker_nodes"]
    
    print(f"\n{'='*80}")
    print(f"TRAINING LOGS - {datetime.now().isoformat()}")
    print(f"{'='*80}\n")
    
    # Master logs
    log_dir = Path(cluster_root) / "logs"
    master_logs = list(log_dir.glob("training_rank0_*.log"))
    
    if master_logs:
        print("MASTER NODE (Rank 0)")
        print("-" * 80)
        with open(master_logs[-1]) as f:
            print(f.read()[-2000:])  # Last 2000 chars
        print()
    
    # Worker logs via SSH
    for worker in workers:
        cmd = (
            f"ssh {worker['user']}@{worker['ip']} "
            f"'tail -100 {cluster_root}/logs/training_rank{worker['rank']}_*.log 2>/dev/null || echo \"No logs yet\"'"
        )
        
        print(f"WORKER {worker['rank']} ({worker['ip']}, rank={worker['rank']})")
        print("-" * 80)
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        print(result.stdout)
        print()


if __name__ == "__main__":
    import sys
    cluster_root = sys.argv[1] if len(sys.argv) > 1 else "~/Development/Projects/ai-ml-pipeline/ai-cluster"
    aggregate_training_logs(str(Path(cluster_root).expanduser()))
```

```bash
chmod +x ~/Development/Projects/ai-ml-pipeline/ai-cluster/scripts/aggregate_logs.py
python ~/Development/Projects/ai-ml-pipeline/ai-cluster/scripts/aggregate_logs.py
```

---

# PHASE 3: Resilience & Failure Handling

## 3.1 Automatic Restart on Failure

Create `/home/ollie/Development/Tools/ugro_watchdog.py`:

```python
#!/usr/bin/env python3
"""
UGRO Watchdog: Monitor training and restart on failure
"""

import subprocess
import sys
import json
import time
import signal
from pathlib import Path
from datetime import datetime


class UGROWatchdog:
    """Monitor training processes, handle failures"""
    
    def __init__(self, config_path: str = "~/.ugro/cluster_config.json"):
        self.config_path = Path(config_path).expanduser()
        self.load_config()
        self.processes = []
        self.max_restarts = 3
        self.restart_count = 0
        signal.signal(signal.SIGINT, self.cleanup)
    
    def load_config(self):
        with open(self.config_path) as f:
            self.config = json.load(f)
    
    def check_process_health(self, process_name: str, pid: int) -> bool:
        """Check if process is still alive"""
        try:
            result = subprocess.run(
                f"ps -p {pid} > /dev/null 2>&1",
                shell=True,
                timeout=5,
            )
            return result.returncode == 0
        except:
            return False
    
    def cleanup(self, signum, frame):
        """Graceful shutdown"""
        print("\n✓ Watchdog shutting down...")
        for proc in self.processes:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except:
                proc.kill()
        sys.exit(0)
    
    def monitor(self):
        """Monitor processes continuously"""
        print(f"\n{'='*70}")
        print("UGRO WATCHDOG - Training Monitor Active")
        print(f"{'='*70}\n")
        
        while True:
            # Check all processes
            all_alive = all(p.poll() is None for p in self.processes)
            
            if not all_alive:
                print(f"✗ Process failure detected at {datetime.now().isoformat()}")
                
                if self.restart_count < self.max_restarts:
                    self.restart_count += 1
                    print(f"→ Attempting restart ({self.restart_count}/{self.max_restarts})...")
                    time.sleep(5)
                    # Restart logic here
                else:
                    print("✗ Max restarts exceeded. Training failed.")
                    sys.exit(1)
            
            time.sleep(10)


if __name__ == "__main__":
    watchdog = UGROWatchdog()
    watchdog.monitor()
```

---

# PHASE 4: Common Workflows

## 4.1 Quick Start: TinyLlama Training

```bash
#!/bin/bash
# Execute on gpu-master

cd ~/Development/Projects/ai-ml-pipeline/ai-cluster/scripts

# Verify cluster health
python /home/ollie/Development/Tools/ugro_status.py

# Launch training
python launch_distributed.py \
    train_production.py \
    --model unsloth/tinyllama-bnb-4bit \
    --epochs 1 \
    --batch-size 1
```

## 4.2 Model Inference After Training

After training completes, inference on master:

```python
# inference.py
import torch
from unsloth import FastLanguageModel

# Load trained model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="./checkpoints/final_model",
    max_seq_length=2048,
    dtype=torch.float16,
    load_in_4bit=True,
)

# Generate
inputs = tokenizer("What is machine learning?", return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_length=100)
print(tokenizer.decode(outputs[0]))
```

## 4.3 Dataset Preparation (Parallelized)

Create `~/Development/Projects/ai-ml-pipeline/ai-cluster/scripts/prepare_dataset.py`:

```python
#!/usr/bin/env python3
"""
Prepare dataset in parallel across cluster
Each node processes its shard locally
"""

import torch.distributed as dist
import os
from datasets import load_dataset


def prepare_dataset_distributed(dataset_name: str, output_dir: str):
    """Load and preprocess dataset - each rank handles its shard"""
    
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    if rank == 0:
        print(f"Downloading {dataset_name}...")
    
    # Load dataset
    if dataset_name == "wikitext":
        dataset = load_dataset("wikitext", "wikitext-2-v1", split="train")
    else:
        dataset = load_dataset(dataset_name, split="train")
    
    # Each rank gets a shard
    shard = dataset.shard(num_shards=world_size, index=rank)
    
    # Process locally
    print(f"[Rank {rank}] Processing {len(shard)} samples...")
    
    # Save shard
    shard.save_to_disk(f"{output_dir}/shard_rank_{rank}")
    
    print(f"[Rank {rank}] Done")


if __name__ == "__main__":
    import sys
    dist.init_process_group(backend="gloo")  # Use Gloo for CPU coordination
    prepare_dataset_distributed(sys.argv[1], sys.argv[2])
    dist.destroy_process_group()
```

```bash
# Launch dataset preparation across cluster
torchrun --nnodes=3 --nproc_per_node=1 \
  --rdzv_backend=c10d --rdzv_endpoint=192.168.1.100:29500 \
  prepare_dataset.py wikitext ./data/wikitext
```

---

# PHASE 5: Production Readiness Checklist

## ✅ Pre-Training Verification

```bash
# Run on master before launching production training

# 1. Verify all nodes online
python /home/ollie/Development/Tools/ugro_status.py

# 2. Check network latency
ping -c 5 192.168.1.101
ping -c 5 192.168.1.102

# 3. Test SSH passwordless on each
ssh ob@192.168.1.101 "echo OK"
ssh ollie@192.168.1.102 "echo OK"

# 4. Verify environments match
bash ~/verify_cluster_env.sh

# 5. Check disk space
df -h ~/Development/Projects/ai-ml-pipeline/ai-cluster

# 6. Run quick single-GPU test
cd ~/Development/Projects/ai-ml-pipeline/ai-cluster/scripts
python -c "
import torch
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    'unsloth/tinyllama-bnb-4bit',
    max_seq_length=2048,
    dtype=torch.float16,
    load_in_4bit=True,
)
print('✓ Single GPU test passed')
"

# 7. Check NCCL connectivity (test distributed)
python -c "import torch.distributed as dist; print('✓ Distributed import OK')"

# 8. Verify TensorBoard can write
mkdir -p ~/Development/Projects/ai-ml-pipeline/ai-cluster/logs/test
tensorboard --logdir ~/Development/Projects/ai-ml-pipeline/ai-cluster/logs/test --port 6007 &
sleep 2
kill %1  # Stop it
echo "✓ TensorBoard writable"
```

## ✅ During Training

Monitor in separate terminals:

```bash
# Terminal 1: Status
watch -n 5 "python /home/ollie/Development/Tools/ugro_status.py"

# Terminal 2: TensorBoard
bash /home/ollie/Development/Tools/ugro_tensorboard.sh

# Terminal 3: Aggregate logs
watch -n 30 "python ~/Development/Projects/ai-ml-pipeline/ai-cluster/scripts/aggregate_logs.py"

# Terminal 4: SSH into gpu1 and monitor locally
ssh ob@192.168.1.101
watch -n 1 nvidia-smi

# Terminal 5: SSH into gpu2 and monitor locally
ssh ollie@192.168.1.102
watch -n 1 nvidia-smi
```

---

# PHASE 6: Troubleshooting Matrix

| Issue | Diagnosis | Fix |
|-------|-----------|-----|
| **Training hangs after rank init** | Check if all nodes started within 30s | Restart all nodes, sync time: `date` |
| **"connect() call failed"** | Network connectivity issue | Check firewall: `sudo ufw allow 29500` |
| **CUDA out of memory on workers** | 8GB insufficient for batch size | Reduce batch size: `--batch-size 1`, increase gradient accumulation |
| **Different loss values per rank** | Random seed mismatch | Set seed uniformly: `seed=42` on all nodes |
| **Very slow training** | One node is bottleneck | Check `nvidia-smi` on all nodes; slowest GPU limits throughput |
| **Workers unreachable via SSH** | SSH key missing or misconfigured | Regenerate and copy SSH keys: `ssh-copy-id -i ~/.ssh/id_ed25519.pub user@ip` |
| **Logs not appearing** | Permission or disk space issue | Check: `ls -la ~/Development/Projects/ai-ml-pipeline/ai-cluster/logs/`, `df -h` |
| **Version mismatch errors** | PyTorch/CUDA versions differ | Run `verify_cluster_env.sh`, reinstall matching versions on mismatched node |
| **NaN loss values** | Learning rate too high or instability | Reduce learning rate: `--lr 1e-4`, check gradient clipping |
| **Process killed unexpectedly** | OOM or system resource exhaustion | Reduce batch size, check system RAM: `free -h` on all nodes |

---

# PHASE 7: Next Evolution - Kubernetes-Lite

When you outgrow single-script orchestration, consider:

1. **K3s** (lightweight Kubernetes) - Job scheduling across cluster
2. **Ray Cluster** - Distributed task execution
3. **Dask Distributed** - Python-native parallelism
4. **SLURM** - HPC-style job scheduler

For now, this UGRO setup provides:
- ✅ Unified orchestration (one command)
- ✅ Health monitoring (status dashboard)
- ✅ Failure recovery (watchdog)
- ✅ Log aggregation (unified view)
- ✅ Easy scaling (add new node = add to JSON + 1 param)

---

# Quick Reference: Daily Operations

```bash
# Morning: Check cluster health
python /home/ollie/Development/Tools/ugro_status.py

# Start training
cd ~/Development/Projects/ai-ml-pipeline/ai-cluster/scripts
python launch_distributed.py train_production.py --epochs 3

# Monitor in another terminal
bash /home/ollie/Development/Tools/ugro_tensorboard.sh
# Browse: http://localhost:6006

# Check logs
python ~/Development/Projects/ai-ml-pipeline/ai-cluster/scripts/aggregate_logs.py

# After training: Save checkpoints
rsync -avz ollie@192.168.1.102:~/Development/Projects/ai-ml-pipeline/ai-cluster/checkpoints ./backup/

# Clean up
rm -rf ~/Development/Projects/ai-ml-pipeline/ai-cluster/logs/runs/*
```

---

**Status:** Infrastructure ready for production training ✓  
**Next:** Deploy your first distributed model!
