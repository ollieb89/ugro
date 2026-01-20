# Launch Coordinator Implementation Plan

## Objective
Implement the Launch Coordinator feature for UGRO as described in the documentation, replacing manual 3-terminal work with a single command that orchestrates distributed training across multiple GPU nodes.

## Current State Analysis
- ✅ CLI structure exists with `launch` command
- ✅ UGROAgent.launch_training() method exists but simulates distributed training
- ✅ Job management system with comprehensive tracking
- ✅ Cluster management with SSHClient and health checks
- ✅ Configuration system with cluster.yaml
- ❌ Missing: Real torchrun execution via SSH
- ❌ Missing: Synchronization between processes
- ❌ Missing: Real-time monitoring and log collection

## Implementation Phases

### Phase 1: Create LaunchCoordinator Class
**File**: `src/ugro/launch_coordinator.py`

**Key Features**:
- Async-based coordination using asyncio
- Cluster state validation
- GPU resource allocation with rank assignment
- Synchronized torchrun execution across workers
- Real-time monitoring and progress tracking
- Log aggregation from all workers
- Comprehensive error handling and recovery

**Methods**:
- `validate_cluster_state()` - Check all nodes reachable and healthy
- `allocate_resources()` - Assign ranks and prepare environments
- `launch_distributed_training()` - Execute torchrun on all workers
- `monitor_training()` - Poll for completion and collect metrics
- `collect_logs()` - Aggregate logs from all workers
- `handle_failure()` - Graceful failure handling and cleanup

### Phase 2: Integration with UGROAgent
**File**: `src/ugro/agent.py` (modify existing)

**Changes**:
- Replace simulation in `_launch_ranks()` with LaunchCoordinator calls
- Maintain backward compatibility with existing Job/CLI structure
- Update job tracking to work with real distributed processes
- Add real-time progress updates

### Phase 3: Enhanced Error Handling
**Files**: Multiple files for comprehensive error handling

**Features**:
- SSH connection failure recovery
- torchrun process failure detection
- Network partition handling
- Resource cleanup on failures
- User-friendly error messages

### Phase 4: Testing and Validation
**Files**: `tests/test_launch_coordinator.py`

**Coverage**:
- Unit tests for LaunchCoordinator methods
- Integration tests with SSH clients
- Failure scenario testing
- Performance validation

## Technical Requirements

### Python 3.12+ Features
- Async/await patterns for concurrent operations
- Type hints with generics and protocols
- Context managers for resource management
- Structured pattern matching for error handling

### Distributed Training Requirements
```bash
# Target command to execute on each worker:
pixi run -e cuda torchrun \
  --nproc_per_node=1 \
  --nnodes=2 \
  --node_rank=$RANK \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  scripts/train_production.py \
  --model llama-7b \
  --dataset wikitext \
  --epochs 3
```

### Synchronization Strategy
- Use barrier synchronization via torchrun
- Implement startup window (30 seconds default)
- Monitor process health across all nodes
- Handle straggler detection and recovery

### Log Collection Strategy
- Real-time log streaming from all workers
- Centralized log aggregation with rank prefixes
- Structured logging with timestamps and metadata
- Error log highlighting and alerting

## Success Criteria
1. ✅ Single `ugro launch` command replaces manual 3-terminal setup
2. ✅ All workers start training within synchronization window
3. ✅ Real-time progress monitoring and log collection
4. ✅ Graceful handling of node failures and network issues
5. ✅ Comprehensive error reporting and debugging support
6. ✅ Backward compatibility with existing CLI and job tracking

## Implementation Timeline
- **Phase 1**: LaunchCoordinator class (2-3 hours)
- **Phase 2**: Agent integration (1 hour)
- **Phase 3**: Error handling enhancements (1 hour)
- **Phase 4**: Testing and validation (1 hour)

## Dependencies
- Existing SSHClient infrastructure
- Current cluster configuration
- Job tracking system
- Result aggregation system

## Risks and Mitigations
- **SSH connectivity issues**: Robust retry logic and connection pooling
- **torchrun failures**: Comprehensive error detection and reporting
- **Synchronization issues**: Timeout handling and straggler detection
- **Log collection failures**: Fallback mechanisms and local buffering