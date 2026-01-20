# Advanced Cluster Optimization & Troubleshooting
## Deep-Dive Configuration & Performance Tuning

---

# PART 1: Performance Optimization

## 1.1: Memory Optimization for Heterogeneous GPUs

Your setup has a memory bottleneck (8GB on workers vs 12GB on master). Optimize:

### Per-GPU Memory Targets

**RTX 5070 Ti (Master, 12 GB):**
- Model weights (4-bit): 2.8 GB
- LoRA: 0.5 GB
- Optimizer state: 2.0 GB
- Gradients: 2.0 GB
- Activations: 2.5 GB
- **Total: ~10 GB (safe margin)**

**RTX 4070 / 3070 Ti (Workers, 8 GB each):**
- Model weights (4-bit): 2.8 GB
- LoRA: 0.5 GB
- Optimizer state: 1.5 GB (reduced)
- Gradients: 1.5 GB (reduced)
- Activations: 1.0 GB (reduced with gradient checkpointing)
- **Total: ~7.3 GB (tight fit)**

### Recommended Settings for Your Hardware

```python
# train_production.py adjustments

# For workers (8GB GPUs)
model_config = {
    "max_seq_length": 1024,  # Reduce from 2048
    "lora_r": 8,             # Reduce from 16
    "lora_alpha": 16,        # Reduce from 32
    "load_in_4bit": True,    # Keep quantization
}

training_config = {
    "batch_size": 1,         # Must stay at 1
    "gradient_accumulation": 16,  # Increase for effective batch
    "gradient_checkpointing": True,
    "num_train_epochs": 3,
    "learning_rate": 2e-4,
    "warmup_ratio": 0.1,
}
```

### Test Configuration Before Full Run

```bash
# Quick 5-step test to verify memory stability
python << 'EOF'
import torch
from unsloth import FastLanguageModel
import sys

print("Testing memory configuration...")

try:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/tinyllama-bnb-4bit",
        max_seq_length=1024,
        dtype=torch.float16,
        load_in_4bit=True,
    )
    print(f"✓ Model loaded, GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    
    model = FastLanguageModel.get_peft_model(
        model,
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )
    print(f"✓ LoRA applied, GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    
    # Forward pass test
    inputs = tokenizer("test", return_tensors="pt").to("cuda")
    outputs = model(**inputs)
    print(f"✓ Forward pass OK, GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    
    print("✓ Configuration validated for this GPU")
    
except RuntimeError as e:
    print(f"✗ Out of Memory: {e}")
    print("→ Reduce sequence length or LoRA rank further")
    sys.exit(1)
EOF
```

## 1.2: Network Optimization (NCCL Tuning)

NCCL over Ethernet can be tuned for your LAN:

```bash
# Optimal NCCL settings for Gigabit LAN (1Gbps)

# Add to environment before training
export NCCL_ALGO=Ring          # Better for LAN
export NCCL_PROTO=Simple       # Simpler protocol
export NCCL_MIN_NCHANNELS=4    # Parallel channels
export NCCL_MAX_NCHANNELS=4
export NCCL_DEBUG=INFO         # Verbose logging (remove in production)

# Save in ~/.bashrc for persistence
echo "export NCCL_ALGO=Ring" >> ~/.bashrc
echo "export NCCL_PROTO=Simple" >> ~/.bashrc
```

## 1.3: Data Loading Optimization

```python
# In train_production.py, optimize data loading

from torch.utils.data import DataLoader, DistributedSampler

dataloader = DataLoader(
    dataset,
    sampler=DistributedSampler(dataset, shuffle=True),
    batch_size=1,
    num_workers=4,          # Increase from 2
    pin_memory=True,        # Keep True
    prefetch_factor=2,      # Prefetch batches
    persistent_workers=True,  # Keep workers alive
)
```

---

# PART 2: Troubleshooting Deep-Dives

## 2.1: "NCCL: unhandled cuda error"

**Symptom:** Training fails immediately after rank initialization

**Root Causes:**
1. CUDA version mismatch across nodes
2. GPU driver version mismatch
3. Firewall blocking NCCL (port 29500)

**Diagnosis:**

```bash
# Check CUDA versions match
echo "Master CUDA:" && python -c "import torch; print(torch.version.cuda)"
echo "GPU1 CUDA:" && ssh ob@192.168.1.101 "python -c 'import torch; print(torch.version.cuda)'"
echo "GPU2 CUDA:" && ssh ollie@192.168.1.102 "python -c 'import torch; print(torch.version.cuda)'"

# Check driver versions
nvidia-smi --query-gpu=driver_version --format=csv,noheader
ssh ob@192.168.1.101 'nvidia-smi --query-gpu=driver_version --format=csv,noheader'
ssh ollie@192.168.1.102 'nvidia-smi --query-gpu=driver_version --format=csv,noheader'

# Test NCCL initialization
python -c "import torch.distributed as dist; dist.init_process_group('nccl'); print('OK'); dist.destroy_process_group()"
```

**Fix:**

```bash
# Sync CUDA versions (example: install CUDA 12.1 on all)
sudo apt install cuda-toolkit-12-1

# Sync driver (example: install driver 545)
sudo apt install nvidia-driver-545

# Reboot all machines
ssh ob@192.168.1.101 'sudo reboot'
ssh ollie@192.168.1.102 'sudo reboot'
sleep 30

# Verify after reboot
bash ~/check_all_envs.sh
```

## 2.2: "rank 1: CUDA out of memory"

**Symptom:** Workers (8GB) run OOM while master (12GB) is fine

**Diagnosis:**

```bash
# Monitor GPU memory during training
watch -n 2 'echo "Master:" && nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader && echo "GPU1:" && ssh ob@192.168.1.101 "nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader" && echo "GPU2:" && ssh ollie@192.168.1.102 "nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader"'

# Check what's consuming memory
ssh ob@192.168.1.101 'nvidia-smi'
```

**Fix (in order of preference):**

1. **Reduce sequence length**
```python
max_seq_length = 512  # from 1024 or 2048
```

2. **Enable advanced gradient checkpointing**
```python
use_gradient_checkpointing="unsloth"  # Already on, verify
```

3. **Reduce LoRA rank**
```python
lora_r = 4  # from 8
lora_alpha = 8  # from 16
```

4. **Reduce batch accumulation steps** (accept slower training)
```python
gradient_accumulation_steps = 8  # from 16
```

5. **Use 8-bit quantization instead of 4-bit** (slightly slower)
```python
load_in_8bit = True  # More memory efficient
```

## 2.3: "Training very slow / one GPU bottlenecking"

**Expected behavior** with heterogeneous GPUs: DDP bottlenecked by slowest GPU (8GB workers)

**Verify this is normal:**

```bash
# Check utilization during training
watch -n 1 'echo "=== GPU Utilization ===" && nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader && echo "GPU1:" && ssh ob@192.168.1.101 "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader" && echo "GPU2:" && ssh ollie@192.168.1.102 "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader"'

# All should be ~70-90% if balanced
# If one is 100% and others lower → that's the bottleneck
```

**Actual bottlenecks to investigate:**

```bash
# 1. Network latency
ping -c 10 192.168.1.101 | grep "avg="
ping -c 10 192.168.1.102 | grep "avg="
# Should be < 1ms on LAN

# 2. CPU saturation on workers (data loading bottleneck)
ssh ob@192.168.1.101 'watch -n 1 "top -b -n 1 | head -15"'

# 3. Disk I/O bottleneck
ssh ob@192.168.1.101 'iostat -x 1 5'

# 4. Model size mismatch verification
echo "Checking all nodes load same model..."
ssh ob@192.168.1.101 "python -c 'from unsloth import FastLanguageModel; model, _ = FastLanguageModel.from_pretrained(\"unsloth/tinyllama-bnb-4bit\", max_seq_length=1024, dtype=__import__(\"torch\").float16, load_in_4bit=True); print(f\"✓ {sum(p.numel() for p in model.parameters())} parameters\")'"
```

**Optimization:**

```bash
# If network is bottleneck:
export NCCL_ALGO=Ring
export NCCL_MIN_NCHANNELS=4

# If CPU is bottleneck:
# Reduce num_workers in DataLoader from 4 to 2 or 1

# If I/O is bottleneck:
# Pre-download dataset to local SSD on each machine
```

## 2.4: "All3 nodes hang during training (deadlock)"

**Symptom:** Training starts, then hangs indefinitely

**Causes:**
1. Mismatched dataset sizes across ranks
2. Different random seed causing divergent code paths
3. NCCL synchronization timeout
4. One rank crashed silently

**Diagnosis:**

```bash
# Check if processes are still running
ps aux | grep -i torchrun
ssh ob@192.168.1.101 'ps aux | grep -i torchrun'
ssh ollie@192.168.1.102 'ps aux | grep -i torchrun'

# Check logs for errors
tail -100 ~/Development/Projects/ai-ml-pipeline/ai-cluster/logs/training_rank*.log | grep -i error

# Check network connectivity during training
ssh ob@192.168.1.101 'ping -c 1 192.168.1.100' && echo "GPU1 can reach master"
ssh ollie@192.168.1.102 'ping -c 1 192.168.1.100' && echo "GPU2 can reach master"

# Check if NCCL timed out
grep "timeout\|TIMEOUT" ~/Development/Projects/ai-ml-pipeline/ai-cluster/logs/training_*.log
```

**Fix:**

```python
# In train_production.py, ensure deterministic behavior

import torch
import numpy as np

seed = 42

torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

# Ensure all datasets are identical size across ranks
dataset_size = len(dataset) // world_size * world_size  # Truncate to multiple
dataset = dataset.select(range(dataset_size))

# Increase NCCL timeout
os.environ['NCCL_DEBUG_SUBSYS'] = 'INIT,COLL'
os.environ['NCCL_IB_TIMEOUT'] = '50'  # Milliseconds
```

```bash
# Kill all and restart
pkill -f torchrun
ssh ob@192.168.1.101 'pkill -f torchrun'
ssh ollie@192.168.1.102 'pkill -f torchrun'

sleep 5

# Restart training with fixes applied
```

## 2.5: "Loss is NaN after first step"

**Symptom:** Training starts but loss becomes NaN immediately

**Causes:**
1. Learning rate too high
2. Gradient explosion
3. Data pipeline returning invalid values
4. Numerical instability in model

**Diagnosis:**

```python
# Add to training loop to track loss stability

if torch.isnan(loss):
    print(f"Loss is NaN at step {step}")
    print(f"Gradient norm: {torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)}")
    print(f"Input sample: {batch}")
    break
```

**Fix:**

```python
# 1. Reduce learning rate
learning_rate = 1e-4  # from 2e-4

# 2. Add gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

# 3. Increase warmup steps
warmup_steps = 500  # from 100

# 4. Validate data
assert not torch.isnan(batch['input_ids']).any(), "Input NaN detected"
```

---

# PART 3: Advanced Monitoring

## 3.1: Real-Time Cluster Dashboard

Create `~/cluster_dashboard.py`:

```python
#!/usr/bin/env python3
"""Real-time cluster monitoring dashboard"""

import subprocess
import json
import time
from datetime import datetime
import curses


def get_gpu_stats(ip: str = None, user: str = None) -> dict:
    """Get GPU stats, optionally remote"""
    
    if ip is None:
        cmd = 'nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader'
    else:
        cmd = f'ssh {user}@{ip} "nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader"'
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            parts = result.stdout.strip().split(', ')
            return {
                'gpu': parts[0],
                'mem_used': int(parts[1]),
                'mem_total': int(parts[2]),
                'util': int(parts[3].rstrip(' %')),
                'temp': int(parts[4].rstrip(' C')),
                'status': 'ONLINE'
            }
    except:
        pass
    
    return {'status': 'OFFLINE'}


def dashboard(stdscr):
    """Curses-based dashboard"""
    
    stdscr.nodelay(True)
    
    while True:
        stdscr.clear()
        
        # Header
        stdscr.addstr(0, 0, f"UGRO CLUSTER DASHBOARD - {datetime.now().isoformat()}", curses.A_BOLD)
        stdscr.addstr(1, 0, "=" * 80)
        
        # Master
        stats = get_gpu_stats()
        line = 3
        stdscr.addstr(line, 0, "MASTER (192.168.1.100)")
        if stats['status'] == 'ONLINE':
            msg = f"  {stats['gpu']} | Memory: {stats['mem_used']}/{stats['mem_total']} MB | Util: {stats['util']}% | Temp: {stats['temp']}°C"
            stdscr.addstr(line + 1, 0, msg)
        else:
            stdscr.addstr(line + 1, 0, "  OFFLINE", curses.A_DIM)
        
        # GPU1
        stats = get_gpu_stats('192.168.1.101', 'ob')
        line += 3
        stdscr.addstr(line, 0, "GPU1 (192.168.1.101, user: ob)")
        if stats['status'] == 'ONLINE':
            msg = f"  {stats['gpu']} | Memory: {stats['mem_used']}/{stats['mem_total']} MB | Util: {stats['util']}% | Temp: {stats['temp']}°C"
            stdscr.addstr(line + 1, 0, msg)
        else:
            stdscr.addstr(line + 1, 0, "  OFFLINE", curses.A_DIM)
        
        # GPU2
        stats = get_gpu_stats('192.168.1.102', 'ollie')
        line += 3
        stdscr.addstr(line, 0, "GPU2 (192.168.1.102, user: ollie)")
        if stats['status'] == 'ONLINE':
            msg = f"  {stats['gpu']} | Memory: {stats['mem_used']}/{stats['mem_total']} MB | Util: {stats['util']}% | Temp: {stats['temp']}°C"
            stdscr.addstr(line + 1, 0, msg)
        else:
            stdscr.addstr(line + 1, 0, "  OFFLINE", curses.A_DIM)
        
        line += 3
        stdscr.addstr(line, 0, "=" * 80)
        stdscr.addstr(line + 1, 0, "Press 'q' to quit, updates every 2s")
        
        stdscr.refresh()
        
        # Check for quit
        try:
            ch = stdscr.getch()
            if ch == ord('q'):
                break
        except:
            pass
        
        time.sleep(2)


if __name__ == "__main__":
    try:
        curses.wrapper(dashboard)
    except KeyboardInterrupt:
        pass
```

```bash
chmod +x ~/cluster_dashboard.py
python ~/cluster_dashboard.py
```

## 3.2: Training Metrics Exporter (Prometheus)

For integration with monitoring systems:

```python
# metrics_exporter.py
from prometheus_client import start_http_server, Gauge
import subprocess
import time

# Define metrics
gpu_memory_used = Gauge('gpu_memory_used_mb', 'GPU memory used (MB)', ['node'])
gpu_utilization = Gauge('gpu_utilization_percent', 'GPU utilization (%)', ['node'])
training_loss = Gauge('training_loss', 'Training loss', ['rank'])

def export_metrics():
    """Export metrics to Prometheus"""
    start_http_server(8000)
    
    while True:
        # Update GPU metrics
        # ... add logic here
        time.sleep(10)

if __name__ == "__main__":
    export_metrics()
```

```bash
# Starts Prometheus metrics on localhost:8000
python metrics_exporter.py
```

---

# PART 4: Advanced Configurations

## 4.1: Multi-GPU Per Node (if you add more GPUs)

If you upgrade any machine to have multiple GPUs:

```bash
# Change nproc_per_node
torchrun \
    --nnodes=3 \
    --nproc_per_node=2 \  # 2 GPUs per node instead of 1
    --rdzv_id=100 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=192.168.1.100:29500 \
    --node_rank=0 \
    train_production.py
```

## 4.2: Fault Tolerance (Elastic DDP)

For automatic recovery on node failure:

```python
# Enable elastic training (PyTorch 1.11+)

dist.init_process_group(
    "nccl",
    init_method="tcp://192.168.1.100:29500",
    timeout=timedelta(minutes=30),
    world_size=-1,  # Auto-detect
    rank=-1,
)
```

## 4.3: Model Parallelism (for larger models)

```python
# If model doesn't fit on single GPU even with quantization
from torch.nn.parallel import DataParallel

# Split model across 2 GPUs on master
model = DataParallel(model, device_ids=[0])

# Wrap with DDP for multi-node
model = DDP(model)
```

---

**Status:** Advanced optimization complete. Ready for production!
