# UGRO Redis Production Hardening & Gang Scheduling

**Status:** Implemented and Verified 2026-01-20

## 1. Redis Production Hardening
- **Objective:** Support high-availability Redis configurations (Sentinel, Cluster) for production deployments.
- **Factory:** `RedisClientFactory` in `src/ugro/queues/redis_client.py` abstracts connection logic.
- **Modes Supported:**
    - `STANDALONE`: Standard single-instance Redis.
    - `SENTINEL`: HA with master/replica and auto-failover.
    - `CLUSTER`: Sharded Redis for horizontal scaling.
- **Implementation Details:**
    - Integrated connection pooling for all modes.
    - Robust `@with_retry` decorator with exponential backoff.
    - Hash tags (`ugro:{job_id}:...`) used for Redis Cluster to ensure cross-key Lua operations stay within the same slot.
    - Two-phase fallback for operations like `next()` and `claim()` in Cluster mode to handle cross-slot limitations.

## 2. Gang Scheduling
- **Objective:** Atomic resource allocation for multi-node distributed training jobs.
- **Model:** `JobResources` now includes `nnodes`, and `Job` stores `worker_ids`.
- **Logic:** `ResourceTracker.can_fit_gang()` implements a Best-Fit Decreasing strategy across the cluster nodes.
- **Scheduler:** `Scheduler.schedule_next()` detects multi-node requirements and uses `can_fit_gang()` to reserve all required resources simultaneously (all-or-nothing).
- **Backfilling:** Effectively maintains FIFO order while allowing smaller jobs to fill gaps if a gang job is waiting for resources.

## 3. Verification
- 17/17 pytest cases pass, including comprehensive gang scheduling integration tests.
