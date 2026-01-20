# UGRO Phase 2d Features: Fault Tolerance & Queue

**Status:** Implemented 2026-01-20

## 1. Fault Tolerance
- **Goal:** Resume training after interruptions.
- **Key File:** `scripts/train_production.py`
- **Logic:** `load_latest_checkpoint` finds latest `epoch_*.pt` in `checkpoints/`. Training loop resumes from `start_epoch`.
- **Elastic:** Compatible with `torch.distributed.elastic`.

## 2. Job Queue
- **Goal:** Decoupled, persistent job execution.
- **DB:** SQLite at `~/.ugro/ugro.db`.
- **Module:** `src/ugro/queue.py`.
- **CLI (`ugro`):**
    - `launch`: Enqueues by default. `--now` for instant run.
    - `queue list`: Show pending jobs.
    - `queue inspect <id>`: Job details.
    - `run-worker`: Polls queue, runs jobs via `UGROAgent`.

## 3. Intelligent Scheduling (Redis & Gang)
- **Redis Hardening:** Support for Sentinel and Cluster modes with automatic retry and hash-tagging.
- **Gang Scheduling:** Supports atomic allocation for multi-node jobs (`nnodes > 1`).
- **Policy:** Best-Fit Decreasing for optimized resource packing.
- **Implementation:** `src/ugro/queues/redis_client.py`, `src/ugro/scheduler/resources.py`.