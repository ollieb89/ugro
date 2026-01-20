# UGRO Setup Status & Next Steps

**Project:** Unified GPU Resource Orchestrator (UGRO)  
**Status:** Foundation Complete ✓ | **Phase:** Advanced Development  
**Last Updated:** 2026-01-20

---

## Current Infrastructure State ✓

### Machines Configured

| Machine | IP | GPU | User | SSH | Env | Status |
|---------|----|----|------|-----|-----|--------|
| **gpu-master** | `192.168.1.100` | RTX 5070 Ti (12GB) | default | ✅ | ✅ | Ready |
| **gpu1** | `192.168.1.101` | RTX 4070 (8GB) | `ob` | ✅ | ✅ | Ready |
| **gpu2** | `192.168.1.102` | RTX 3070 Ti (8GB) | `ollie` | ✅ | ✅ | Ready |

### What's Complete ✓

- ✅ Network configured (static IPs, LAN connectivity)
- ✅ SSH passwordless auth between machines
- ✅ ML environments installed (PyTorch, transformers, unsloth, etc.)
- ✅ CUDA 12.1 + GPU drivers verified
- ✅ GPUs accessible and verified

### What We're Building Next 🚀

**Phase 1: Control Plane & Orchestration** (Next 2 weeks)
- Central task scheduler
- Job queue management
- Resource allocation API
- Status monitoring dashboard

**Phase 2: Advanced Features** (Weeks 3-4)
- Distributed model serving
- Checkpoint management
- Hyperparameter search
- Data pipeline optimization

**Phase 3: Production Hardening** (Ongoing)
- Failure recovery
- Auto-scaling (add machines)
- Performance optimization
- Security & auth

---

## Immediate Next Steps (This Week)

### 1. **Test Distributed Training End-to-End** (30 min)

Verify the full stack works before building control plane:

```bash
# Terminal 1: Master (192.168.1.100)
cd ~/ai-cluster/scripts
torchrun \
    --nnodes=3 --nproc_per_node=1 \
    --rdzv_id=100 --rdzv_backend=c10d \
    --rdzv_endpoint=192.168.1.100:29500 \
    --node_rank=0 \
    train_production.py \
    --model-name unsloth/tinyllama-bnb-4bit \
    --num-epochs 1

# Terminal 2: Worker 1 (ssh ob@192.168.1.101)
torchrun \
    --nnodes=3 --nproc_per_node=1 \
    --rdzv_id=100 --rdzv_backend=c10d \
    --rdzv_endpoint=192.168.1.100:29500 \
    --node_rank=1 \
    train_production.py \
    --model-name unsloth/tinyllama-bnb-4bit \
    --num-epochs 1

# Terminal 3: Worker 2 (ssh ollie@192.168.1.102)
torchrun \
    --nnodes=3 --nproc_per_node=1 \
    --rdzv_id=100 --rdzv_backend=c10d \
    --rdzv_endpoint=192.168.1.100:29500 \
    --node_rank=2 \
    train_production.py \
    --model-name unsloth/tinyllama-bnb-4bit \
    --num-epochs 1
```

**Success Indicators:**
- All 3 processes initialize without error
- GPU memory increases on all 3 GPUs
- Training loss decreases
- No NCCL timeout errors
- Completes without hanging

### 2. **Create Cluster Control Plane** (Next: Central management interface)

Build a Python service that manages cluster operations:

```bash
# Create new directory for orchestrator
mkdir -p ~/ugro-orchestrator/{scheduler,monitor,api,config}

# Structure:
# ugro-orchestrator/
# ├── scheduler/           # Job scheduling logic
# ├── monitor/             # Resource monitoring
# ├── api/                 # REST API for job submission
# ├── config/              # Configuration files
# └── README.md
```

### 3. **Design Job Queue System** (This week)

Decide on architecture:

```
Job Queue Architecture Options:
├─ Option A: Simple file-based (easiest)
│  └─ JSON files in shared directory
│  └─ Good for: small clusters, testing
│
├─ Option B: Redis-based (recommended)
│  └─ Central job queue
│  └─ Good for: scalability, real-time updates
│
└─ Option C: Full orchestration (k8s-style)
   └─ etcd, distributed consensus
   └─ Good for: enterprise, many machines
```

**Recommendation:** Start with **Option B (Redis)** for your use case.

---

## Architecture You're Building

```
┌─────────────────────────────────────────────────────────────────┐
│                    UGRO Control Plane                           │
│  (Central orchestrator on gpu-master at 192.168.1.100)          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────┐  │
│  │   Job Queue      │  │  Resource        │  │  API Server  │  │
│  │   (Redis)        │  │  Monitor         │  │  (REST)      │  │
│  │                  │  │                  │  │              │  │
│  │  - Submit jobs   │  │  - GPU utilization   │  - Submit job    │\n│  │  - Priority queue│  │  - Memory tracking   │  - Check status  │  │\n│  │  - Status track │  │  - Network I/O       │  - Get metrics   │  │\n│  │  - Checkpoint   │  │  - CPU usage         │  - List jobs     │  │\n│  │    management   │  │                  │  │              │  │\n│  └────────┬─────────┘  └────────┬─────────┘  └──────────┬───┘  │\n│           │                     │                       │       │\n│           └─────────────────────┼───────────────────────┘       │\n│                                 │                                │\n│                        Distributed RPC Layer                     │\n│                        (SSH + torch.distributed)                │\n└─────────────────────────────────────────────────────────────────┘\n         │\n         │\n    ┌────┴──────┬─────────────┬─────────────┐\n    │            │             │             │\n    ▼            ▼             ▼             ▼\n ┌──────┐   ┌──────┐      ┌──────┐     ┌──────┐\n │Worker│   │Worker│      │Worker│     │Worker│\n │ Node │   │ Node │      │ Node │     │ Node │\n │(101) │   │(102) │      │(103) │     │(104) │\n │      │   │      │      │      │     │      │\n │ RTX  │   │ RTX  │      │ RTX  │     │ RTX  │\n │ 4070 │   │ 3070 │      │ 4090 │     │ 6000 │\n └──────┘   └──────┘      └──────┘     └──────┘\n    │          │             │             │\n    └──────────┴─────────────┴─────────────┘\n         NCCL Collective Communication\n         (Distributed Training Pipeline)\n```

---

## Key Files to Create This Week

### 1. `ugro-scheduler.py` - Central Job Scheduler

```python
\"\"\"\nUGRO Scheduler: Manages job queue and worker allocation\n\"\"\"\nimport redis\nimport json\nimport logging\nfrom datetime import datetime\nfrom dataclasses import dataclass, asdict\nfrom enum import Enum\n\nclass JobStatus(Enum):\n    QUEUED = \"queued\"\n    RUNNING = \"running\"\n    COMPLETED = \"completed\"\n    FAILED = \"failed\"\n    CANCELLED = \"cancelled\"\n\n@dataclass\nclass Job:\n    job_id: str\n    name: str\n    model: str\n    dataset: str\n    epochs: int\n    learning_rate: float\n    batch_size: int\n    status: JobStatus = JobStatus.QUEUED\n    created_at: str = None\n    started_at: str = None\n    completed_at: str = None\n    \n    def __post_init__(self):\n        if self.created_at is None:\n            self.created_at = datetime.now().isoformat()\n\nclass UGROScheduler:\n    def __init__(self, redis_host='localhost', redis_port=6379):\n        self.redis = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)\n        self.logger = logging.getLogger(__name__)\n    \n    def submit_job(self, job: Job) -> str:\n        \"\"\"Add job to queue\"\"\"\n        job_dict = asdict(job)\n        job_dict['status'] = job_dict['status'].value\n        self.redis.lpush('job_queue', json.dumps(job_dict))\n        self.logger.info(f\"Job submitted: {job.job_id}\")\n        return job.job_id\n    \n    def get_next_job(self) -> Job:\n        \"\"\"Get next job from queue\"\"\"\n        job_data = self.redis.rpop('job_queue')\n        if job_data:\n            return Job(**json.loads(job_data))\n        return None\n    \n    def update_job_status(self, job_id: str, status: JobStatus):\n        \"\"\"Update job status\"\"\"\n        self.redis.hset(f\"job:{job_id}\", \"status\", status.value)\n\n# Usage\nif __name__ == \"__main__\":\n    scheduler = UGROScheduler()\n    \n    # Submit job\n    job = Job(\n        job_id=\"job_001\",\n        name=\"Llama-7B Fine-tune\",\n        model=\"meta-llama/Llama-2-7b-hf\",\n        dataset=\"wikitext\",\n        epochs=3,\n        learning_rate=2e-4,\n        batch_size=1\n    )\n    scheduler.submit_job(job)\n    \n    # Get next job\n    next_job = scheduler.get_next_job()\n    print(f\"Next job: {next_job.name}\")\n```

### 2. `ugro-monitor.py` - Resource Monitoring

```python\n\"\"\"\nUGRO Monitor: Real-time cluster resource monitoring\n\"\"\"\nimport subprocess\nimport json\nfrom dataclasses import dataclass\nfrom typing import List\nimport paramiko\n\n@dataclass\nclass GPUMetrics:\n    gpu_id: int\n    name: str\n    memory_used: float  # GB\n    memory_total: float  # GB\n    utilization: float  # %\n    temperature: float  # °C\n\nclass ClusterMonitor:\n    def __init__(self, nodes: dict):\n        \"\"\"\n        nodes: {\"gpu-master\": \"192.168.1.100\", \"gpu1\": \"192.168.1.101\", ...}\n        \"\"\"\n        self.nodes = nodes\n    \n    def get_local_gpu_metrics(self) -> List[GPUMetrics]:\n        \"\"\"Get GPU metrics on current machine\"\"\"\n        cmd = (\n            \"nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu \"\n            \"--format=csv,nounits,noheader\"\n        )\n        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)\n        \n        metrics = []\n        for line in result.stdout.strip().split('\\n'):\n            parts = [p.strip() for p in line.split(',')]\n            metrics.append(GPUMetrics(\n                gpu_id=int(parts[0]),\n                name=parts[1],\n                memory_used=float(parts[2]),\n                memory_total=float(parts[3]),\n                utilization=float(parts[4]),\n                temperature=float(parts[5])\n            ))\n        return metrics\n    \n    def get_remote_gpu_metrics(self, node_name: str, host: str, user: str) -> List[GPUMetrics]:\n        \"\"\"Get GPU metrics from remote machine via SSH\"\"\"\n        ssh = paramiko.SSHClient()\n        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())\n        ssh.connect(host, username=user)\n        \n        cmd = (\n            \"nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu \"\n            \"--format=csv,nounits,noheader\"\n        )\n        _, stdout, _ = ssh.exec_command(cmd)\n        \n        metrics = []\n        for line in stdout.read().decode().strip().split('\\n'):\n            if not line:\n                continue\n            parts = [p.strip() for p in line.split(',')]\n            metrics.append(GPUMetrics(\n                gpu_id=int(parts[0]),\n                name=parts[1],\n                memory_used=float(parts[2]),\n                memory_total=float(parts[3]),\n                utilization=float(parts[4]),\n                temperature=float(parts[5])\n            ))\n        \n        ssh.close()\n        return metrics\n    \n    def get_cluster_status(self) -> dict:\n        \"\"\"Get status of entire cluster\"\"\"\n        status = {}\n        \n        for node_name, host_ip in self.nodes.items():\n            try:\n                if host_ip == \"localhost\" or host_ip == \"127.0.0.1\":\n                    metrics = self.get_local_gpu_metrics()\n                else:\n                    # Determine username (extract from your setup)\n                    user = \"ob\" if \"101\" in host_ip else \"ollie\" if \"102\" in host_ip else \"default\"\n                    metrics = self.get_remote_gpu_metrics(node_name, host_ip, user)\n                \n                status[node_name] = {\n                    \"ip\": host_ip,\n                    \"gpus\": [asdict(m) for m in metrics]\n                }\n            except Exception as e:\n                status[node_name] = {\"error\": str(e)}\n        \n        return status\n\n# Usage\nif __name__ == \"__main__\":\n    nodes = {\n        \"gpu-master\": \"192.168.1.100\",\n        \"gpu1\": \"192.168.1.101\",\n        \"gpu2\": \"192.168.1.102\"\n    }\n    \n    monitor = ClusterMonitor(nodes)\n    status = monitor.get_cluster_status()\n    print(json.dumps(status, indent=2))\n```

### 3. `ugro-api.py` - REST API Server

```python\n\"\"\"\nUGRO API: REST interface for job submission and monitoring\n\"\"\"\nfrom flask import Flask, request, jsonify\nfrom ugro_scheduler import UGROScheduler, Job, JobStatus\nfrom ugro_monitor import ClusterMonitor\nimport uuid\n\napp = Flask(__name__)\nscheduler = UGROScheduler()\nmonitor = ClusterMonitor({\n    \"gpu-master\": \"192.168.1.100\",\n    \"gpu1\": \"192.168.1.101\",\n    \"gpu2\": \"192.168.1.102\"\n})\n\n@app.route('/api/jobs', methods=['POST'])\ndef submit_job():\n    \"\"\"Submit a new training job\"\"\"\n    data = request.json\n    \n    job = Job(\n        job_id=str(uuid.uuid4())[:8],\n        name=data.get('name', 'Untitled Job'),\n        model=data.get('model', 'meta-llama/Llama-2-7b-hf'),\n        dataset=data.get('dataset', 'wikitext'),\n        epochs=data.get('epochs', 3),\n        learning_rate=data.get('learning_rate', 2e-4),\n        batch_size=data.get('batch_size', 1)\n    )\n    \n    scheduler.submit_job(job)\n    return jsonify({\"job_id\": job.job_id, \"status\": \"queued\"}), 201\n\n@app.route('/api/jobs', methods=['GET'])\ndef list_jobs():\n    \"\"\"List all jobs\"\"\"\n    # TODO: Implement job listing from Redis\n    return jsonify({\"jobs\": []})\n\n@app.route('/api/jobs/<job_id>', methods=['GET'])\ndef get_job(job_id):\n    \"\"\"Get job details\"\"\"\n    # TODO: Implement job status retrieval\n    return jsonify({\"job_id\": job_id, \"status\": \"running\"})\n\n@app.route('/api/cluster/status', methods=['GET'])\ndef cluster_status():\n    \"\"\"Get cluster status\"\"\"\n    return jsonify(monitor.get_cluster_status())\n\n@app.route('/api/cluster/health', methods=['GET'])\ndef cluster_health():\n    \"\"\"Get cluster health summary\"\"\"\n    status = monitor.get_cluster_status()\n    \n    total_gpus = 0\n    total_memory = 0\n    used_memory = 0\n    healthy = 0\n    \n    for node in status.values():\n        if \"gpus\" in node:\n            healthy += 1\n            for gpu in node['gpus']:\n                total_gpus += 1\n                total_memory += gpu['memory_total']\n                used_memory += gpu['memory_used']\n    \n    return jsonify({\n        \"total_nodes\": len(status),\n        \"healthy_nodes\": healthy,\n        \"total_gpus\": total_gpus,\n        \"total_memory_gb\": total_memory,\n        \"used_memory_gb\": used_memory,\n        \"memory_utilization_percent\": (used_memory / total_memory * 100) if total_memory > 0 else 0\n    })\n\nif __name__ == '__main__':\n    app.run(host='0.0.0.0', port=5000, debug=False)\n```

---

## Quick Command Reference

### Current Infrastructure

```bash\n# Test SSH to all machines\nssh ob@192.168.1.101 \"nvidia-smi | head -5\"\nssh ollie@192.168.1.102 \"nvidia-smi | head -5\"\n\n# Check environment on all machines\nfor i in 100 101 102; do\n  echo \"=== 192.168.1.$i ===\"\n  ssh <user>@192.168.1.$i \"conda list | grep torch\"\ndone\n\n# View logs from distributed training\ntail -f ~/ai-cluster/logs/training_rank0_*.log\n```

### To Launch Services (Next Week)\n\n```bash\n# Start Redis (for job queue)\nredis-server --port 6379\n\n# Start UGRO monitor\npython ~/ugro-orchestrator/monitor/ugro-monitor.py\n\n# Start UGRO API\npython ~/ugro-orchestrator/api/ugro-api.py\n\n# Submit job via API\ncurl -X POST http://192.168.1.100:5000/api/jobs \\\n  -H \"Content-Type: application/json\" \\\n  -d '{\n    \"name\": \"Llama-7B Fine-tune\",\n    \"model\": \"meta-llama/Llama-2-7b-hf\",\n    \"dataset\": \"wikitext\",\n    \"epochs\": 3\n  }'\n\n# Check cluster status\ncurl http://192.168.1.100:5000/api/cluster/status\n```

---\n\n## Deliverables This Week\n\n- ✅ **Code:** Core orchestrator modules (scheduler, monitor, API)\n- ✅ **Config:** Job configuration templates\n- ✅ **Docs:** API documentation\n- ✅ **Test:** End-to-end distributed training verification\n\n---\n\n## Next Week's Goals\n\n1. **Distributed job launcher** - SSH into workers and launch training\n2. **Checkpoint manager** - Save/resume training across cluster\n3. **Dashboard** - Web UI for monitoring\n4. **Error handling** - Graceful failure recovery\n\n---\n\n## Questions for You\n\n1. **Job Queue Backend:** Redis vs simple file-based vs custom?\n2. **Authentication:** SSH key-based (current) vs token-based API auth?\n3. **Scaling:** Plan to add more machines soon? How many?\n4. **Data:** Where will training data live? (NFS mount, local copy, streaming?)\n5. **Checkpoint Strategy:** Central checkpoint directory or per-machine?\n\nResponses will shape the control plane architecture! 🚀\n