# Cluster State Manager Implementation

## Overview
Implemented Cluster State Manager per Phase2 spec with file-backed JSON persistence and wired into UGROAgent for automatic state tracking during job lifecycle.

## Files Created/Modified

### src/ugro/cluster_state.py (new)
- ClusterStateManager class with load/save to /etc/ugro/cluster_state.json (configurable via UGRO_STATE_FILE)
- Dataclass models: NodeState, JobState, ClusterState
- Atomic writes via temp file + replace
- Node/job CRUD operations with validation

### src/ugro/__init__.py (modified)
- Exported ClusterState, ClusterStateManager, JobState, NodeState

### src/ugro/agent.py (modified)
- Added cluster_state_manager initialization
- Added _load_cluster_state(), _persist_cluster_state(), _sync_cluster_state_nodes()
- Added _update_state_for_job_start() and _update_state_for_job_end()
- Integrated state updates into launch_training() workflow

## Behavior

### Initialization
- Loads existing state from disk or creates empty state
- Syncs nodes from cluster.yaml into state (master defaults to gpu="unknown", vram_gb=0; workers use hardware fields)
- Persists initial node inventory

### Job Launch
- Creates job entry with status="running", ranks, model, gpu_nodes, ISO timestamp
- Marks participating nodes as status="busy" with running_job_id

### Job Completion/Failure
- Updates job status to completed/failed
- Releases nodes back to status="available", clears running_job_id

### Persistence
- All mutations trigger immediate save to disk
- Uses atomic write pattern (temp file + replace)
- Errors logged but do not crash agent

## Configuration
- Default state file: /etc/ugro/cluster_state.json
- Override via environment variable: UGRO_STATE_FILE
- File path configurable via ClusterStateManager(state_file=Path(...))

## Integration Points
- UGROAgent automatically manages state during training jobs
- State manager can be used standalone for other tools
- Exported via package for external access

## Validation
- State manager validates node existence before updates
- Job updates raise KeyError if job not found
- Load failures raise RuntimeError with context

## Next Steps (Optional)
- Add CLI command: `ugro state show` to display current state
- Add health monitoring daemon integration
- Add state cleanup for old completed jobs