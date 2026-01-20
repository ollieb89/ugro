# UGRO: Unified GPU Resource Orchestrator
## Personal-Scale Production GPU Orchestration System

**Status:** Foundation Complete, Moving to Orchestration Layer  
**Current Infrastructure:** Ready ✓  
**Next Phase:** Building the Control Plane

---

## Executive Summary

You have a **fully functional 3-node GPU cluster** with:
- ✅ Network: Static IPs, SSH passwordless auth
- ✅ Environment: Identical conda setups on all machines
- ✅ Communication: PyTorch DDP working across nodes
- ✅ Training: Ready to launch distributed jobs

**Next step:** Build UGRO—a **control plane and orchestrator** that transforms your cluster from "three independent machines" into **one cohesive AI training platform**.

---

## Current Infrastructure Status

### Cluster Topology

```
┌─────────────────────────────────────────────────────────────────┐
│                        LOCAL LAN                                 │
│                   192.168.1.0/24                                 │
└─────────────────────────────────────────────────────────────────┘

gpu-master (192.168.1.100)          ← UGRO Control Plane Lives Here
├─ RTX 5070 Ti (12 GB)
├─ Rank 0 (Master Node)
└─ Role: Scheduler, metadata, job tracking

gpu1 (192.168.1.101) - user: ob
├─ RTX 4070 (8 GB)
├─ Rank 1 (Worker)
└─ Role: Training executor

gpu2 (192.168.1.102) - user: ollie
├─ RTX 3070 Ti (8 GB)
├─ Rank 2 (Worker)
└─ Role: Training executor
```

### What's Working Now

| Component | Status | Details |
|-----------|--------|---------|
| **Network** | ✅ Ready | Static IPs, ping latency < 5ms |
| **SSH** | ✅ Ready | Passwordless auth working |
| **Environments** | ✅ Ready | Identical Python 3.11, PyTorch 2.1, CUDA 12.1 |
| **DDP** | ✅ Ready | Multi-node training tested & working |
| **Storage** | ✅ Ready | Local data, Google Drive via rclone |
| **Monitoring** | ✅ Ready | nvidia-smi, TensorBoard, logs |

### Cluster Capacity

**Total VRAM:** 28 GB  
**Effective VRAM (with checkpointing):** ~45 GB  
**Training Speed:** ~2.5x single GPU  
**Efficiency:** ~85% (15% network overhead)

---

## Vision: What UGRO Will Be

### Three Layers

```
┌─────────────────────────────────────────────────────────────────┐
│                     LAYER 1: USER INTERFACE                      │
│  CLI + Dashboard for job submission, monitoring, management      │
└────────────┬────────────────────────────────────────────────────┘
             │
┌────────────▼────────────────────────────────────────────────────┐
│               LAYER 2: CONTROL PLANE (UGRO CORE)                 │
│  Scheduler, job registry, resource allocation, failure recovery  │
│  - Runs on: gpu-master (192.168.1.100)                          │
└────────────┬────────────────────────────────────────────────────┘
             │
┌────────────▼────────────────────────────────────────────────────┐
│              LAYER 3: EXECUTION (WORKER AGENTS)                  │
│  GPU control, resource reporting, job execution                 │
│  - Runs on: gpu1 (192.168.1.101), gpu2 (192.168.1.102)         │
└─────────────────────────────────────────────────────────────────┘
```

### Design Principles (Non-Negotiable)

1. **Human-in-the-Loop** – Automation assists, never surprises
2. **Distributed First** – Multi-node training is the core use case
3. **Failure is Expected** – Recovery must be simpler than prevention
4. **Visibility Over Abstraction** – Users see where & why jobs run
5. **Composable, Not Monolithic** – Each layer is separable & testable

---

## UGRO Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────────┐
│ USER LAYER (CLI + Web Dashboard)                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  CLI Commands:                                                   │
│  ├─ ugro launch <job_name>      → Submit training job           │
│  ├─ ugro status                 → Check cluster health          │
│  ├─ ugro logs <job>             → View job logs                 │
│  ├─ ugro cancel <job>           → Stop training                 │
│  ├─ ugro checkpoint <job>       → Save checkpoint               │
│  └─ ugro resume <job>           → Resume from checkpoint        │
│                                                                   │
│  Dashboard:                                                      │
│  ├─ Real-time GPU metrics        (memory, utilization, temp)   │
│  ├─ Job lifecycle view           (queued → running → done)      │
│  ├─ Cluster topology map         (visual layout of nodes)       │
│  ├─ Training curves              (loss, throughput trends)      │
│  └─ Alert system                 (OOM, node failure, etc)       │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
           ▲                              ▲
           │ REST API                     │ WebSocket
           │                              │
┌──────────┴──────────────────────────────┴──────────────────────┐
│ CONTROL PLANE (Runs on gpu-master: 192.168.1.100)             │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│ 1. JOB SCHEDULER                                                │
│    ├─ Queue: Maintains job queue (FIFO with priority)          │
│    ├─ Allocator: Assigns resources to jobs                     │
│    ├─ Launcher: Executes torchrun commands                     │
│    └─ Tracker: Tracks job state (queued/running/done/failed)  │
│                                                                   │
│ 2. RESOURCE MANAGER                                            │
│    ├─ Monitor: Polls worker agents for metrics                 │
│    ├─ Predictor: Estimates if job will fit                     │
│    ├─ Balancer: Optimizes GPU assignment                       │
│    └─ Governor: Enforces limits (prevent overcommit)           │
│                                                                   │
│ 3. METADATA & PERSISTENCE                                      │
│    ├─ Job Registry: Tracks all jobs & metadata                 │
│    ├─ Checkpoint Store: Manages model checkpoints              │
│    ├─ Config Store: Training hyperparameters                   │
│    └─ Metrics DB: Stores logs & performance data               │
│                                                                   │
│ 4. FAILURE HANDLER                                             │
│    ├─ Watchdog: Detects worker/job failures                   │
│    ├─ Retry Logic: Auto-retry failed jobs                      │
│    ├─ Recovery: Checkpoint-aware restart                       │
│    └─ Alert: Notifies user of critical failures                │
│                                                                   │
└──────────┬──────────────────────────────────────────────────────┘
           │
           │ Agent Protocol (gRPC or REST)
           │
┌──────────┴───────────────────────────────────────────────────────┐
│ WORKER AGENTS (Run on gpu1 & gpu2)                              │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│ On gpu1 (192.168.1.101):                                         │
│ ├─ Agent service (listens for commands)                         │
│ ├─ GPU monitor (nvidia-smi polling)                            │
│ ├─ Process manager (start/stop jobs)                           │
│ └─ Log streamer (stream training logs)                         │
│                                                                    │
│ On gpu2 (192.168.1.102):                                         │
│ ├─ Agent service (listens for commands)                         │
│ ├─ GPU monitor (nvidia-smi polling)                            │
│ ├─ Process manager (start/stop jobs)                           │
│ └─ Log streamer (stream training logs)                         │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

---

## Implementation Roadmap

### Phase 1: Foundation (This Week)
- [ ] Worker agents (gpu1, gpu2)
- [ ] Basic job runner on master
- [ ] Health monitoring & reporting

### Phase 2: Orchestration (Next Week)
- [ ] Job queue & scheduler
- [ ] Resource allocation logic
- [ ] Job state tracking
- [ ] Basic CLI

### Phase 3: Production Hardening (Week 3)
- [ ] Failure detection & recovery
- [ ] Checkpoint management
- [ ] Logging & metrics
- [ ] Multi-user support

### Phase 4: User Experience (Week 4)
- [ ] Web dashboard
- [ ] Advanced CLI commands
- [ ] Alert system
- [ ] Performance analytics

---

## Key Files to Create

### Control Plane (gpu-master)

```
ugro/
├── control_plane/
│   ├── scheduler.py           # Job queue, allocation, launching
│   ├── resource_manager.py    # GPU monitoring, capacity planning
│   ├── metadata_store.py      # Job registry, persistent storage
│   ├── failure_handler.py     # Detect & recover from failures
│   ├── api_server.py          # REST API for CLI/dashboard
│   └── main.py                # Entry point
│
├── common/
│   ├── models.py              # Data models (Job, GPU, Node)
│   ├── protocol.py            # Message protocol (gRPC or REST)
│   └── config.py              # Configuration
│
└── cli/
    ├── commands.py            # CLI command handlers
    ├── output.py              # Formatting output
    └── __main__.py            # Entry point
```

### Worker Agents (gpu1, gpu2)

```
ugro/
├── worker_agent/
│   ├── agent.py               # Main agent service
│   ├── gpu_monitor.py         # Real-time GPU metrics
│   ├── process_manager.py     # Job execution
│   ├── log_streamer.py        # Log forwarding
│   └── main.py                # Entry point
```

---

## Design Decisions

### Why This Architecture?

| Choice | Reason | Alternative |
|--------|--------|-------------|
| Master on gpu-master | Scheduler needs to be central | Distributed consensus (too complex) |
| gRPC/REST for agents | Simple, standard protocol | SSH (brittle), sockets (low-level) |
| Job queue (FIFO+priority) | Human-in-the-loop, predictable | Dynamic scheduling (too complex) |
| Checkpoint-first recovery | Minimize data loss on failure | Stateless restart (lose progress) |
| Visibility-first logging | User understands what's happening | Opaque auto-restart (mysterious) |

### Technology Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| Control plane | Python + FastAPI | Simple, fast, standard |
| Worker agents | Python + gRPC | Efficient, async-friendly |
| Metadata | SQLite (local) + JSON files | Lightweight, inspectable |
| Monitoring | Prometheus/simple polling | Standard metrics format |
| Dashboard | React/Vue + WebSocket | Real-time updates |
| CLI | Click/Typer | Standard Python CLI |

---

## Integration with Existing Setup

### What Changes?

**BEFORE** (Current):
```bash
# Manual on each terminal:
Terminal 1: torchrun --nnodes=3 ... --node_rank=0 train_production.py
Terminal 2: ssh ob@192.168.1.101 "torchrun --nnodes=3 ... --node_rank=1 train_production.py"
Terminal 3: ssh ollie@192.168.1.102 "torchrun --nnodes=3 ... --node_rank=2 train_production.py"
```

**AFTER** (With UGRO):
```bash
# Single command:
ugro launch --model llama-7b --dataset my-data --epochs 3
# UGRO handles all 3 nodes, monitoring, checkpointing, recovery
```

### Backward Compatibility

- Existing `train_production.py` **still works** (unchanged)
- UGRO is a **wrapper**, not a replacement
- Can run jobs manually or via UGRO
- Can migrate jobs gradually

---

## Data Flow Examples

### Example 1: Submit a Training Job

```
User:                          UGRO CLI
  │
  ├─ ugro launch --model llama-7b
  │       --dataset wikitext
  │       --epochs 3
  │       --name "test-run-1"
  │                          │
  │                          ▼ (HTTP POST)
  │                      API Server
  │                          │
  │                          ├─ Create job entry
  │                          ├─ Queue job
  │                          └─ Check resources
  │                          │
  │                          ▼ (Check capacity)
  │                    Resource Manager
  │                          │
  │                    Query agents for
  │                    GPU availability
  │                          │
  │        ┌──────────────────┼──────────────────┐
  │        │                  │                  │
  │        ▼                  ▼                  ▼
  │    Agent (gpu1)      Agent (gpu2)       Scheduler
  │    RTX 4070          RTX 3070 Ti
  │    8 GB free         8 GB free
  │        │                  │
  │        └──────────────────┴──────┐
  │                                  │
  │                                  ▼
  │                    Decision: Can fit ✓
  │                    Status: QUEUED
  │                          │
  │                          ▼
User:                  "Job queued (ID: job-001)"
  │                    "Starting in ~2 minutes"
```

### Example 2: Job Execution

```
Scheduler (observes job is next in queue)
    │
    ├─ Check all nodes ready
    ├─ Prepare torchrun command
    └─ Execute on each node
        │
        ├─► Agent (gpu1): Start rank 1 process
        ├─► Agent (gpu2): Start rank 2 process
        └─► Master: Start rank 0 process
        │
        ▼ (All nodes connect via NCCL)
    Training starts
        │
        ├─ Stream metrics back to control plane
        ├─ Log to persistent storage
        └─ Periodic checkpoint saves
        │
        ▼ (After 3 epochs or error)
    Job complete
        │
        ├─ Save final model
        ├─ Collect metrics
        └─ Report status to user
```

### Example 3: Failure Recovery

```
Training running for 2 hours...
    │
    ▼ (RTX 4070 runs out of memory)
Worker (gpu1): OOM Error detected
    │
    ├─ Kill training process
    ├─ Free GPU memory
    └─ Report error to master
        │
        ▼
Scheduler (receives error)
    │
    ├─ Check if checkpoint exists
    │  ├─ YES: Propose resume from checkpoint
    │  └─ NO: Propose restart or cancel
    │
    ├─ Notify user: "Training failed at step 5000"
    │  "Last checkpoint: 4950"
    │  "Retry with smaller batch size? (yes/no)"
    │
    ▼ (User response: yes, smaller batch)
Scheduler:
    ├─ Modify job config (batch_size=1)
    ├─ Re-queue job with resume flag
    └─ Resume from checkpoint at step 4950
        │
        ▼
Training resumes...
```

---

## Next Steps

1. **Week 1:** Build worker agents (gpu1, gpu2)
2. **Week 2:** Build control plane scheduler
3. **Week 3:** Add failure handling & recovery
4. **Week 4:** Add CLI & dashboard

---

## Questions for Your Consideration

1. **Job Priority:** Should user be able to prioritize jobs, or strictly FIFO?
2. **Multi-User:** Should different users have isolated job queues?
3. **Resource Reservation:** Should users pre-book resources, or dynamic allocation?
4. **Model Registry:** Should UGRO manage model downloads, or external?
5. **Data Movement:** Should UGRO handle data movement, or user responsibility?

---

## Success Criteria

After UGRO is complete:

✅ One command to launch multi-node training  
✅ Automatic resource tracking & allocation  
✅ Checkpoint-aware job recovery  
✅ Real-time monitoring dashboard  
✅ CLI for job management  
✅ Multi-user support (future)  
✅ Scale to 5+ machines without architecture changes  

**The goal:** Your GPU cluster feels like **one supercomputer**, not three independent machines.
