# Implementation Plan: UGRO Phase 2b - Health & Monitoring

## Objective
Implement a production-grade health monitoring system for UGRO clusters that provides real-time status, adaptive polling, and automated alerting.

## Phase 1: Core Monitoring Logic
- [ ] Task 1.1: Refactor `AdaptiveHealthMonitor` for better integration.
    - [ ] Add `check_node_health` method as per Phase 2b specs.
    - [ ] Implement robust `test_gpu`, `test_ssh`, `test_pytorch_import`, etc.
- [ ] Task 1.2: Implement `ugro-monitor` daemon script.
    - [ ] Create `src/ugro/monitor_daemon.py` that runs the `AdaptiveHealthMonitor` loop.
    - [ ] Ensure it logs to `logs/monitor.log`.

## Phase 2: System Integration
- [ ] Task 2.1: Implement systemd service units.
    - [ ] Create `systemd/ugro-monitor.service`.
    - [ ] Create automation script to install/enable service.
- [ ] Task 2.2: CLI Enhancements.
    - [ ] Add `ugro monitor --daemon` command (start/stop/status).
    - [ ] Ensure `ugro health` uses the cached status from `ClusterStateManager` updated by the daemon.

## Phase 3: Testing & Validation
- [ ] Task 3.1: Write integration tests for the monitor.
- [ ] Task 3.2: Verify circuit breaker and adaptive polling.

## Success Criteria
1. `ugro-monitor` service runs in background.
2. `ugro health` reflects real-time node status without blocking.
3. Critical GPU failures trigger alerts in logs.
