# GPU Cluster Operations Manual
## Quick-Reference Command Library (Tailored to Your Setup)

**Cluster Config:**
- Master: `gpu-master` at `192.168.1.100`
- Worker 1: `gpu1` at `192.168.1.101` (user: `ob`)
- Worker 2: `gpu2` at `192.168.1.102` (user: `ollie`)

---

# SECTION A: Cluster Connectivity

## A.1: Test Basic Connectivity (Master)

```bash
# From gpu-master
ping -c 3 192.168.1.101 && echo "✓ gpu1 reachable"
ping -c 3 192.168.1.102 && echo "✓ gpu2 reachable"
```

## A.2: Test SSH Access (Master)

```bash
# Verify passwordless SSH works
ssh ob@192.168.1.101 "hostname" && echo "✓ SSH to gpu1 works"
ssh ollie@192.168.1.102 "hostname" && echo "✓ SSH to gpu2 works"

# One-liner to test all at once
for ip in 192.168.1.101 192.168.1.102; do ssh ${ip%.*.*.*}@$ip "echo ✓ $(hostname)"; done
```

## A.3: One-Time SSH Key Setup (if needed)

```bash
# On gpu-master, generate keys
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -N ""

# Copy to workers
ssh-copy-id -i ~/.ssh/id_ed25519.pub ob@192.168.1.101
ssh-copy-id -i ~/.ssh/id_ed25519.pub ollie@192.168.1.102

# Verify
ssh ob@192.168.1.101 "echo connection_test" | grep connection_test
```

---

# SECTION B: GPU Status & Monitoring

## B.1: Local GPU Status (on any machine)

```bash
# Quick check
nvidia-smi -L

# Detailed status
nvidia-smi

# Compact format
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader

# Watch live (updates every second)
watch -n 1 nvidia-smi

# JSON format (useful for parsing)
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,nounits
```

## B.2: Check All Nodes Simultaneously (Master)

```bash
echo "=== GPU STATUS - ALL NODES ==="
echo ""
echo "MASTER NODE:"
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader
echo ""
echo "GPU1 (user: ob):"
ssh ob@192.168.1.101 'nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader'
echo ""
echo "GPU2 (user: ollie):"
ssh ollie@192.168.1.102 'nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader'
```

## B.3: Watch GPU Memory in Real-Time (Master)

```bash
# Master + Workers, updates every 2 seconds
watch -n 2 'echo "=== MASTER ===" && nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader && echo "" && echo "=== GPU1 ===" && ssh ob@192.168.1.101 "nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader" && echo "" && echo "=== GPU2 ===" && ssh ollie@192.168.1.102 "nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader"'
```

---

# SECTION C: Environment & Package Management

## C.1: Check Python Environment (All Machines)

```bash
# Local check
python --version
python -c "import sys; print(sys.prefix)"  # Show conda env path
conda list | grep torch  # Show installed torch

# Check on workers
ssh ob@192.168.1.101 'python -c "import torch; print(torch.__version__)"'
ssh ollie@192.168.1.102 'python -c "import torch; print(torch.__version__)"'
```

## C.2: Verify CUDA (All Machines)

```bash
# Local
nvcc --version
python -c "import torch; print(f'Torch CUDA: {torch.version.cuda}')"

# Workers
ssh ob@192.168.1.101 'python -c "import torch; print(torch.version.cuda)"'
ssh ollie@192.168.1.102 'python -c "import torch; print(torch.version.cuda)"'
```

## C.3: Unified Environment Check Script

```bash
#!/bin/bash
# Save as: ~/check_all_envs.sh

echo "====== PYTHON VERSIONS ======"
echo "Master:" && python --version
echo "GPU1:" && ssh ob@192.168.1.101 "python --version"
echo "GPU2:" && ssh ollie@192.168.1.102 "python --version"

echo ""
echo "====== PYTORCH VERSIONS ======"
echo "Master:" && python -c "import torch; print(torch.__version__)"
echo "GPU1:" && ssh ob@192.168.1.101 "python -c 'import torch; print(torch.__version__)'"
echo "GPU2:" && ssh ollie@192.168.1.102 "python -c 'import torch; print(torch.__version__)'"

echo ""
echo "====== CUDA VERSIONS ======"
echo "Master:" && python -c "import torch; print(f'CUDA: {torch.version.cuda}')"
echo "GPU1:" && ssh ob@192.168.1.101 "python -c 'import torch; print(f\"CUDA: {torch.version.cuda}\")'"
echo "GPU2:" && ssh ollie@192.168.1.102 "python -c 'import torch; print(f\"CUDA: {torch.version.cuda}\")'"
```

```bash
chmod +x ~/check_all_envs.sh
bash ~/check_all_envs.sh
```

---

# SECTION D: File Transfer & Synchronization

## D.1: Copy Files to Workers

```bash
# Copy file from master to gpu1
scp ~/some_file.py ob@192.168.1.101:~/

# Copy directory
scp -r ~/scripts ob@192.168.1.101:~/

# From gpu2 back to master
scp ollie@192.168.1.102:~/results.json ~/

# Sync entire directory to both workers
for user_ip in ob@192.168.1.101 ollie@192.168.1.102; do
  rsync -avz ~/Development/Projects/ai-ml-pipeline $user_ip:~/Development/Projects/
done
```

## D.2: Collect Results from Workers (Master)

```bash
# Create collection directory
mkdir -p ~/cluster_results/{gpu1,gpu2}

# Copy from gpu1
scp -r ob@192.168.1.101:~/Development/Projects/ai-ml-pipeline/ai-cluster/checkpoints ~/cluster_results/gpu1/

# Copy from gpu2
scp -r ollie@192.168.1.102:~/Development/Projects/ai-ml-pipeline/ai-cluster/checkpoints ~/cluster_results/gpu2/

# Consolidate logs
mkdir -p ~/cluster_results/logs
scp ob@192.168.1.101:~/Development/Projects/ai-ml-pipeline/ai-cluster/logs/* ~/cluster_results/logs/gpu1/
scp ollie@192.168.1.102:~/Development/Projects/ai-ml-pipeline/ai-cluster/logs/* ~/cluster_results/logs/gpu2/
```

---

# SECTION E: Training Launch Commands

## E.1: Standard DDP Training (3 GPUs)

```bash
# Terminal 1: Master node
cd ~/Development/Projects/ai-ml-pipeline/ai-cluster/scripts

torchrun \
    --nnodes=3 \
    --nproc_per_node=1 \
    --rdzv_id=100 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=192.168.1.100:29500 \
    --node_rank=0 \
    train_production.py \
    --model-name unsloth/tinyllama-bnb-4bit \
    --num-epochs 3 \
    --learning-rate 2e-4

# Terminal 2: GPU1 (SSH)
ssh ob@192.168.1.101
cd ~/Development/Projects/ai-ml-pipeline/ai-cluster/scripts

torchrun \
    --nnodes=3 \
    --nproc_per_node=1 \
    --rdzv_id=100 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=192.168.1.100:29500 \
    --node_rank=1 \
    train_production.py \
    --model-name unsloth/tinyllama-bnb-4bit \
    --num-epochs 3 \
    --learning-rate 2e-4

# Terminal 3: GPU2 (SSH)
ssh ollie@192.168.1.102
cd ~/Development/Projects/ai-ml-pipeline/ai-cluster/scripts

torchrun \
    --nnodes=3 \
    --nproc_per_node=1 \
    --rdzv_id=100 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=192.168.1.100:29500 \
    --node_rank=2 \
    train_production.py \
    --model-name unsloth/tinyllama-bnb-4bit \
    --num-epochs 3 \
    --learning-rate 2e-4
```

## E.2: Quick Validation Test (Single Epoch)

```bash
# Run same commands as above but with --num-epochs 1
# Expected time: ~5-10 minutes for full pass
```

## E.3: Larger Model Training

```bash
# Change to Llama-2-7B
--model-name meta-llama/Llama-2-7b-hf

# Or Code Llama
--model-name meta-llama/CodeLlama-7b-hf

# Reduce batch size if OOM
--batch-size 1

# Increase gradient accumulation
--gradient-accumulation-steps 16
```

---

# SECTION F: Monitoring & Logging

## F.1: Live GPU Monitoring (Master)

```bash
# Terminal dedicated to monitoring
watch -n 1 nvidia-smi

# Or more compact
watch -n 2 'nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv'
```

## F.2: TensorBoard (Master)

```bash
# Start TensorBoard
tensorboard --logdir ~/Development/Projects/ai-ml-pipeline/ai-cluster/logs/runs/ --port 6006

# Access in browser: http://localhost:6006
# Or remote: http://<master-ip>:6006
```

## F.3: Watch Training Logs (Master)

```bash
# Real-time tail of master's logs
tail -f ~/Development/Projects/ai-ml-pipeline/ai-cluster/logs/training_rank0_*.log

# Search for errors across all logs
grep "ERROR" ~/Development/Projects/ai-ml-pipeline/ai-cluster/logs/training_rank*.log

# Count loss updates
grep "loss:" ~/Development/Projects/ai-ml-pipeline/ai-cluster/logs/training_rank0_*.log | tail -10
```

## F.4: Aggregate Logs from All Nodes

```bash
#!/bin/bash
# Save as: ~/show_all_logs.sh

echo "=== MASTER LOGS (Rank 0) ==="
tail -20 ~/Development/Projects/ai-ml-pipeline/ai-cluster/logs/training_rank0_*.log 2>/dev/null

echo ""
echo "=== GPU1 LOGS (Rank 1) ==="
ssh ob@192.168.1.101 "tail -20 ~/Development/Projects/ai-ml-pipeline/ai-cluster/logs/training_rank1_*.log 2>/dev/null"

echo ""
echo "=== GPU2 LOGS (Rank 2) ==="
ssh ollie@192.168.1.102 "tail -20 ~/Development/Projects/ai-ml-pipeline/ai-cluster/logs/training_rank2_*.log 2>/dev/null"
```

```bash
chmod +x ~/show_all_logs.sh
bash ~/show_all_logs.sh
```

---

# SECTION G: Emergency/Troubleshooting

## G.1: Kill All Training Processes

```bash
# Master
pkill -f "torchrun\|train_production"

# GPU1
ssh ob@192.168.1.101 'pkill -f "torchrun\|train_production"'

# GPU2
ssh ollie@192.168.1.102 'pkill -f "torchrun\|train_production"'

# Or more forcefully
pkill -9 python
ssh ob@192.168.1.101 'pkill -9 python'
ssh ollie@192.168.1.102 'pkill -9 python'
```

## G.2: Free GPU Memory

```bash
# Per machine
nvidia-smi --gpu-reset

# All machines
nvidia-smi --gpu-reset
ssh ob@192.168.1.101 'nvidia-smi --gpu-reset'
ssh ollie@192.168.1.102 'nvidia-smi --gpu-reset'
```

## G.3: Test Network Connectivity

```bash
# Check latency to workers
ping -c 10 192.168.1.101 | tail -1
ping -c 10 192.168.1.102 | tail -1

# Test port 29500 (NCCL)
telnet 192.168.1.100 29500

# Check firewall
sudo ufw status
sudo ufw allow 29500
```

## G.4: Debug Process Issues

```bash
# What's running on GPU?
ps aux | grep -i gpu

# Check which process owns GPU memory
nvidia-smi | grep python

# On workers
ssh ob@192.168.1.101 'ps aux | grep -i torch'
ssh ollie@192.168.1.102 'ps aux | grep -i torch'
```

## G.5: Disk Space Check

```bash
# Master
df -h ~/Development/Projects/ai-ml-pipeline/ai-cluster

# Workers
ssh ob@192.168.1.101 'df -h ~/Development/Projects/ai-ml-pipeline/ai-cluster'
ssh ollie@192.168.1.102 'df -h ~/Development/Projects/ai-ml-pipeline/ai-cluster'

# If space is low, clean logs
rm -f ~/Development/Projects/ai-ml-pipeline/ai-cluster/logs/training_rank*.log
```

---

# SECTION H: Pre-Training Checklist

Run this before every training session:

```bash
#!/bin/bash
# Save as: ~/pre_training_check.sh

echo "1. Cluster connectivity..."
ping -c 1 192.168.1.101 > /dev/null && echo "  ✓ GPU1" || echo "  ✗ GPU1"
ping -c 1 192.168.1.102 > /dev/null && echo "  ✓ GPU2" || echo "  ✗ GPU2"

echo ""
echo "2. SSH access..."
ssh ob@192.168.1.101 "echo ✓" > /dev/null && echo "  ✓ GPU1" || echo "  ✗ GPU1"
ssh ollie@192.168.1.102 "echo ✓" > /dev/null && echo "  ✓ GPU2" || echo "  ✗ GPU2"

echo ""
echo "3. GPU availability..."
echo "  Master: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "  GPU1: $(ssh ob@192.168.1.101 'nvidia-smi --query-gpu=name --format=csv,noheader')"
echo "  GPU2: $(ssh ollie@192.168.1.102 'nvidia-smi --query-gpu=name --format=csv,noheader')"

echo ""
echo "4. Disk space..."
du -sh ~/Development/Projects/ai-ml-pipeline/ai-cluster
ssh ob@192.168.1.101 "du -sh ~/Development/Projects/ai-ml-pipeline/ai-cluster"
ssh ollie@192.168.1.102 "du -sh ~/Development/Projects/ai-ml-pipeline/ai-cluster"

echo ""
echo "5. Environment consistency..."
bash ~/check_all_envs.sh | tail -6

echo ""
echo "✓ PRE-TRAINING CHECK COMPLETE"
```

```bash
chmod +x ~/pre_training_check.sh
bash ~/pre_training_check.sh
```

---

# SECTION I: Common One-Liners

```bash
# Show all GPU memory usage
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,nounits | column -t -s,

# Show GPU utilization only
nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader

# Find processes using GPU
lsof /dev/nvidia*

# Check NCCL version
python -c "import torch; print(torch.cuda.nccl.version())"

# Verify DDP setup
python -c "import torch.distributed as dist; print('DDP OK')"

# Get cluster summary
echo "Total VRAM: $((12 + 8 + 8)) GB"
echo "Effective with checkpointing: ~45 GB"

# SSH into gpu1 without password
ssh ob@192.168.1.101

# SSH into gpu2 without password
ssh ollie@192.168.1.102

# Run command on gpu1
ssh ob@192.168.1.101 "cd ~/Development/Projects/ai-ml-pipeline/ai-cluster && pwd"

# Copy result from gpu2 to master
scp ollie@192.168.1.102:~/checkpoint.pt ./
```

---

# Quick Reference Card

| Task | Command |
|------|---------|
| Check all GPUs | `bash ~/check_all_envs.sh` |
| Watch GPU memory | `watch -n 2 nvidia-smi` |
| SSH to GPU1 | `ssh ob@192.168.1.101` |
| SSH to GPU2 | `ssh ollie@192.168.1.102` |
| Kill training | `pkill -f torchrun` |
| View master logs | `tail -f ~/Development/.../logs/training_rank0_*.log` |
| Start TensorBoard | `tensorboard --logdir ~/Development/.../logs/runs/ --port 6006` |
| Copy to GPU1 | `scp file.py ob@192.168.1.101:~/` |
| Pre-flight check | `bash ~/pre_training_check.sh` |
| Emergency kill all | `for ip in 192.168.1.10{1,2}; do ssh ${ip%.*.*.*}@$ip 'pkill -9 python'; done` |

---

**Keep this file open while operating the cluster!**
