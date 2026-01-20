# Phase 2b Implementation Plan: Results Aggregation & Advanced Monitoring

**Plan ID:** `plan_phase2b_results_aggregation_20260120`  
**Created:** 2026-01-20T17:28:00+01:00  
**Status:** Planning Complete → Ready for Execution

---

## Executive Summary

This plan implements Phase 2b of UGRO orchestration, focusing on:
1. **Real-time metrics streaming** from training scripts to the orchestrator
2. **TensorBoard event file centralization** via ResultAggregator
3. **CLI dashboard view** for live training progress tracking

### Current State Analysis

The Phase 2a foundation provides:
- ✅ `LaunchCoordinator` - Orchestrates distributed training launch
- ✅ `ResultAggregator` - Basic directory structure (jobs/, logs/, tensorboard/)
- ✅ `TrainingMetricsCollector` - Reads metrics from files, stores history
- ✅ `AdaptiveHealthMonitor` - System health with circuit breakers
- ❌ Training script does NOT emit structured metrics
- ❌ No TensorBoard synchronization from workers
- ❌ No live CLI dashboard

---

## Architecture Decision: File-Based Metrics Protocol

**Why file-based over webhooks:**
1. No firewall/network issues (SSH already established)
2. No additional dependencies on training nodes
3. Resilient to network blips (file persists on disk)
4. TrainingMetricsCollector already implements file reading
5. Simpler debugging (inspect JSON files directly)

**Data Flow:**
```
┌─────────────────────────────────────────────────────────────────┐
│                      TRAINING NODE (Worker)                      │
│                                                                  │
│  train_production.py                                             │
│       │                                                          │
│       ├──▶ MetricsEmitter.emit_step()                           │
│       │         │                                                │
│       │         ├──▶ ~/ugro_data/jobs/{job_id}/metrics.jsonl    │
│       │         └──▶ ~/ugro_data/jobs/{job_id}/tensorboard/     │
│       │                                                          │
└───────┼──────────────────────────────────────────────────────────┘
        │
        │ SSH Pull (every 5s)
        ▼
┌───────────────────────────────────────────────────────────────────┐
│                    ORCHESTRATOR (Master Node)                      │
│                                                                    │
│  TrainingMetricsCollector                                          │
│       │                                                            │
│       ├──▶ Read metrics.jsonl from each worker                    │
│       ├──▶ Store in memory (per-rank history)                     │
│       └──▶ Persist to central ResultAggregator                    │
│                                                                    │
│  ResultAggregator.sync_tensorboard()                              │
│       │                                                            │
│       └──▶ rsync tfevents from workers → central tensorboard/    │
│                                                                    │
│  CLI Dashboard (ugro watch)                                        │
│       │                                                            │
│       └──▶ Rich.Live renders metrics table in real-time          │
│                                                                    │
└───────────────────────────────────────────────────────────────────┘
```

---

## Implementation Phases

### 📦 Phase 1: Training Script Metrics Emission (Foundation)

**Priority:** CRITICAL - Blocks all other phases

#### Task 1.1: Create MetricsEmitter Module
**File:** `src/ugro/metrics_emitter.py`

```python
# Key features to implement:
class MetricsEmitter:
    def __init__(self, job_id: str, metrics_dir: Path, rank: int)
    def emit_step(self, step: int, loss: float, lr: float, grad_norm: float, throughput: float)
    def emit_gpu_stats(self) -> dict
    def get_tensorboard_writer(self) -> SummaryWriter

**Todos:**
- [x] Create MetricsEmitter dataclass with job_id, rank, paths
- [x] Implement atomic JSONL writes (write to .tmp, then rename)
- [x] Add GPU stats collection via `torch.cuda` or subprocess nvidia-smi
- [x] Add optional TensorBoard SummaryWriter wrapper
- [x] Add `__enter__`/`__exit__` for context manager pattern
- [x] Write unit tests with mock GPU

#### Task 1.2: Modify train_production.py
**File:** `scripts/train_production.py`

**Changes Required:**
1. Add `--job-id` argument (required)
2. Add `--metrics-dir` argument (default: ~/ugro_data/jobs)
3. Import and initialize MetricsEmitter in main()
4. Calculate throughput: `tokens_processed / time_elapsed`
5. Call `emitter.emit_step()` every 10 steps in train_epoch()
6. Log scalars to TensorBoard: loss, lr, throughput, gpu_util

**New Arguments:**
```python
parser.add_argument("--job-id", type=str, required=True)
parser.add_argument("--metrics-dir", type=str, default=os.path.expanduser("~/ugro_data/jobs"))
```

**Todos:**
- [x] Add new CLI arguments
- [x] Initialize MetricsEmitter with job_id, rank
- [x] Track step count and timestamp for throughput
- [x] Emit metrics every 10 steps
- [x] Add TensorBoard scalar logging
- [x] Test with single-node training

---

### 📦 Phase 2: ResultAggregator Enhancement

**Priority:** HIGH - Required for centralization

#### Task 2.1: Add sync_tensorboard() Method
**File:** `src/ugro/result_aggregator.py`

```python
async def sync_tensorboard(
    self, 
    job_id: str, 
    workers: list[dict], 
    ssh_clients: dict[str, SSHClient]
) -> bool:
    """Rsync TensorBoard events from all workers to central location."""
```

**Implementation:**
- Use `rsync -a --include='*.tfevents*' --exclude='*'` for efficient sync
- Merge into `{base_dir}/jobs/{job_id}/tensorboard/`
- Handle permission errors gracefully

#### Task 2.2: Add sync_logs() Method
```python
async def sync_logs(
    self,
    job_id: str,
    workers: list[dict],
    ssh_clients: dict[str, SSHClient]
) -> bool:
    """Collect rank logs from all workers."""
```

#### Task 2.3: Add get_job_summary() Method
```python
def get_job_summary(self, job_id: str) -> dict:
    """Parse metrics.jsonl and return aggregated stats."""
    # Returns: final_loss, avg_throughput, total_steps, duration, per_rank_stats
```

**Todos:**
- [x] Implement sync_tensorboard with subprocess rsync
- [x] Implement sync_logs with SCP/SFTP
- [x] Implement get_job_summary with metrics parsing
- [x] Add error handling for missing files
- [x] Add cleanup for old sync artifacts

---

### 📦 Phase 3: Real-time Metrics Streaming Integration

**Priority:** HIGH - Core functionality

#### Task 3.1: Integrate TrainingMetricsCollector with LaunchCoordinator

**File:** `src/ugro/launch_coordinator.py`

Modify `launch_distributed_training()`:
```python
# After successful launch
self._metrics_collector = TrainingMetricsCollector(self.cluster)
await self._metrics_collector.start_collection(job.job_id, ranks)
```

Modify `_finalize_job_status()`:
```python
# Before marking complete
await self._metrics_collector.stop_collection(job.job_id)
```

#### Task 3.2: Add Periodic TensorBoard Sync

In `_monitor_training()` loop:
```python
# Every 60 seconds
if time.time() - last_sync > 60:
    await self._result_aggregator.sync_tensorboard(job.job_id, ...)
    last_sync = time.time()
```

#### Task 3.3: Update Agent to Expose Metrics
**File:** `src/ugro/agent.py`

```python
def get_live_metrics(self, job_name: str) -> list[TrainingMetrics] | None:
    """Get latest training metrics for dashboard."""
    return self._metrics_collector.get_latest_metrics(job_name)
```

**Todos:**
- [x] Add _metrics_collector to LaunchCoordinator.__init__
- [x] Start collection after successful job launch
- [x] Stop collection on job completion/failure
- [x] Add periodic TensorBoard sync to monitoring loop
- [x] Expose get_live_metrics in Agent
- [x] Integration test with mock cluster

---

### 📦 Phase 4: CLI Dashboard View

**Priority:** MEDIUM - User-facing feature

#### Task 4.1: Create Dashboard Module
**File:** `src/ugro/dashboard.py`

```python
from rich.live import Live
from rich.layout import Layout
from rich.table import Table
from rich.panel import Panel

class TrainingDashboard:
    def __init__(self, agent: UGROAgent, job_name: str)
    async def run(self, refresh_interval: float = 2.0)
    def _build_layout(self) -> Layout
    def _build_metrics_table(self, metrics: list[TrainingMetrics]) -> Table
    def _build_alerts_panel(self, alerts: list[str]) -> Panel
```

**Dashboard Layout:**
```
┌─────────────────────────────────────────────────────────────────┐
│  🚀 UGRO Training Dashboard: job_20260120_120000                │
│  Status: RUNNING | Elapsed: 00:45:32 | ETA: ~01:15:00          │
├─────────────────────────────────────────────────────────────────┤
│ Rank │ Node        │ GPU%  │ Loss   │ LR       │ Tok/s │ Status│
│──────│─────────────│───────│────────│──────────│───────│───────│
│ 0    │ gpu-master  │ 87.5% │ 3.245  │ 2.00e-04 │ 152   │ ✓     │
│ 1    │ gpu1        │ 82.1% │ 3.312  │ 2.00e-04 │ 148   │ ✓     │
│ 2    │ gpu2        │ 79.8% │ 3.289  │ 2.00e-04 │ 145   │ ✓     │
├─────────────────────────────────────────────────────────────────┤
│ ⚠️ Alerts:                                                      │
│ • gpu2: GPU utilization below threshold (79.8% < 80%)          │
├─────────────────────────────────────────────────────────────────┤
│ Press Ctrl+C to exit                                            │
└─────────────────────────────────────────────────────────────────┘
```

#### Task 4.2: Add `watch` Command to CLI
**File:** `src/ugro/cli.py`

```python
@app.command()
def watch(
    ctx: typer.Context,
    job_name: Annotated[str, typer.Argument(help="Job name to watch")],
    refresh: Annotated[float, typer.Option("--refresh", "-r", help="Refresh interval")] = 2.0,
):
    """Watch live training progress."""
    agent: UGROAgent = ctx.obj["agent"]
    dashboard = TrainingDashboard(agent, job_name)
    asyncio.run(dashboard.run(refresh_interval=refresh))
```

**Todos:**
- [x] Create dashboard.py with LiveDashboard class
- [x] Implement Rich Layout with header, metrics, info, footer panels
- [x] Add graceful Ctrl+C handling
- [x] Add --refresh CLI option
- [x] Add `monitor` command to cli.py
- [x] Test with mock metrics data
- [x] Add color coding for status (green/yellow/red)

---

## Validation Gates

### Gate 1: After Phase 1 (Metrics Emission)
```bash
# Run training locally
cd scripts
python train_production.py --job-id test_001 --model-name unsloth/tinyllama-bnb-4bit

# Verify outputs
cat ~/ugro_data/jobs/test_001/metrics.jsonl  # Should have JSON lines
ls ~/ugro_data/jobs/test_001/tensorboard/    # Should have tfevents
```

### Gate 2: After Phase 2 (ResultAggregator)
```python
# Unit test
aggregator = ResultAggregator()
summary = aggregator.get_job_summary("test_001")
assert "final_loss" in summary
assert "throughput" in summary
```

### Gate 3: After Phase 3 (Integration)
```bash
# Full integration test
ugro launch --name integration_test --epochs 1

# Verify collector started
# Check logs for "Started metrics collection"
# Verify TensorBoard sync occurred
```

### Gate 4: After Phase 4 (Dashboard)
```bash
# Manual test
ugro watch integration_test

# Should see live updating table
# Ctrl+C should exit gracefully
```

---

## Risk Mitigations

| Risk | Mitigation |
|------|------------|
| Training script breaks | Add `--enable-metrics` flag to make optional |
| TensorBoard sync slow | Filter to only tfevents files, batch every 60s |
| SSH connection fails | TrainingMetricsCollector has circuit breakers |
| Dashboard blocks CLI | Use async with proper cancellation |
| Metrics file corruption | Atomic writes with temp file → rename |

---

## Estimated Effort

| Phase | Tasks | Effort | Dependencies |
|-------|-------|--------|--------------|
| 1     | 1.1, 1.2 | ~2h | None |
| 2     | 2.1, 2.2, 2.3 | ~1h | None |
| 3     | 3.1, 3.2, 3.3 | ~2h | Phase 1, 2 |
| 4     | 4.1, 4.2 | ~2h | Phase 3 |
| Testing | All | ~1h | All |
| **Total** | 10 tasks | **~8h** | |

---

## Files to Create/Modify

### New Files
- `src/ugro/metrics_emitter.py` - Training script metrics helper
- `src/ugro/dashboard.py` - CLI dashboard with Rich

### Modified Files
- `scripts/train_production.py` - Add metrics emission
- `src/ugro/result_aggregator.py` - Add sync methods
- `src/ugro/launch_coordinator.py` - Integrate metrics collection
- `src/ugro/agent.py` - Expose get_live_metrics()
- `src/ugro/cli.py` - Add `watch` command

---

## Success Criteria

- [ ] Real-time metrics visible during training via CLI
- [ ] TensorBoard files centralized automatically to master
- [ ] CLI dashboard shows live per-rank progress
- [ ] No regression in existing `launch`, `status`, `health` commands
- [ ] Documentation updated with new features

---

## Next Steps

1. **Start with Task 1.1** - Create `metrics_emitter.py` (lowest risk, new file)
2. **Then Task 1.2** - Modify training script to emit metrics
3. **Parallel: Task 2.x** - ResultAggregator enhancements
4. **Integration: Task 3.x** - Connect everything together
5. **Polish: Task 4.x** - Build the dashboard UI

**Ready for execution.** Proceed with implementation?
