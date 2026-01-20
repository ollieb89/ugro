"""Integration of health monitor with UGRO agent and CLI.

This module provides integration points for the health monitor with the main UGRO
components, including agent integration and CLI commands.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from .agent import UGROAgent
from .health_monitor import (
    AdaptiveHealthMonitor, 
    MonitoringConfig, 
    TrainingMetricsCollector,
    MetricsCollectorConfig,
    create_health_monitor,
    create_metrics_collector
)


class HealthMonitorIntegration:
    """Integration layer for health monitor with UGRO agent."""
    
    def __init__(self, agent: UGROAgent) -> None:
        """Initialize health monitor integration.
        
        Args:
            agent: UGRO agent instance
        """
        self.agent = agent
        self.logger = logging.getLogger(__name__)
        self._monitor: AdaptiveHealthMonitor | None = None
        self._monitor_task: asyncio.Task | None = None
        self._metrics_collector: TrainingMetricsCollector | None = None

    async def start_health_monitoring(self, config: MonitoringConfig | None = None) -> None:
        """Start health monitoring with integration to UGRO agent.
        
        Args:
            config: Optional monitoring configuration
        """
        if self._monitor is not None:
            self.logger.warning("Health monitor already running")
            return
        
        # Create health monitor with existing cluster and state manager
        self._monitor = create_health_monitor(
            cluster=self.agent.cluster,
            state_manager=self.agent.cluster_state_manager,
            config=config
        )
        
        # Create metrics collector
        self._metrics_collector = create_metrics_collector(
            cluster=self.agent.cluster,
            config=None  # Use defaults
        )
        
        # Register job activity callbacks
        self._setup_job_activity_tracking()
        
        # Start monitoring in background
        self._monitor_task = asyncio.create_task(
            self._monitor.start_monitoring(),
            name="health_monitor"
        )
        
        self.logger.info("Health monitoring started with UGRO integration")

    async def stop_health_monitoring(self) -> None:
        """Stop health monitoring gracefully."""
        if self._monitor_task is None:
            self.logger.warning("Health monitor not running")
            return
        
        # Cancel monitoring task
        self._monitor_task.cancel()
        
        # Stop monitor
        if self._monitor:
            await self._monitor.stop()
            self._monitor = None
        
        # Stop metrics collector
        if self._metrics_collector:
            await self._metrics_collector.stop()
            self._metrics_collector = None
        
        # Wait for task to complete
        try:
            await self._monitor_task
        except asyncio.CancelledError:
            pass
        
        self._monitor_task = None
        self.logger.info("Health monitoring stopped")

    def _setup_job_activity_tracking(self) -> None:
        """Set up job activity tracking with UGRO agent."""
        # Hook into job lifecycle events
        # This would require extending UGROAgent to emit events
        # For now, we'll manually track in launch_training
        pass

    def register_job_start(self, job_name: str, ranks: list[int] | None = None) -> None:
        """Register job start for adaptive monitoring and metrics collection."""
        if self._monitor:
            self._monitor.register_job_activity(job_name)
        
        if self._metrics_collector and ranks:
            # Start collecting metrics for this job
            asyncio.create_task(
                self._metrics_collector.start_collection(job_name, ranks),
                name=f"start_metrics_{job_name}"
            )

    def register_job_end(self, job_name: str) -> None:
        """Register job end for adaptive monitoring."""
        if self._monitor:
            self._monitor.unregister_job_activity(job_name)
        
        if self._metrics_collector:
            # Stop collecting metrics for this job
            asyncio.create_task(
                self._metrics_collector.stop_collection(job_name),
                name=f"stop_metrics_{job_name}"
            )

    def get_training_metrics(self, job_id: str, rank: int | None = None):
        """Get training metrics for a specific job and optional rank."""
        if not self._metrics_collector:
            return None
        return self._metrics_collector.get_latest_metrics(job_id, rank)

    def get_job_metrics_statistics(self, job_id: str) -> dict[str, Any]:
        """Get statistics for a specific job's metrics."""
        if not self._metrics_collector:
            return {"status": "collector_not_running"}
        return self._metrics_collector.get_job_statistics(job_id)

    def get_monitoring_status(self) -> dict[str, Any]:
        """Get current monitoring status and statistics."""
        if not self._monitor:
            return {"status": "stopped"}
        
        status = {
            "status": "running",
            "statistics": self._monitor.get_statistics(),
            "config": {
                "base_interval": self._monitor.config.base_interval,
                "max_interval": self._monitor.config.max_interval,
                "min_interval": self._monitor.config.min_interval,
            }
        }
        
        # Add metrics collector statistics if available
        if self._metrics_collector:
            status["metrics_collector"] = {
                "statistics": self._metrics_collector.get_statistics(),
                "config": {
                    "collection_interval": self._metrics_collector.config.collection_interval,
                    "max_history_size": self._metrics_collector.config.max_history_size,
                }
            }
        
        return status


# Extend UGROAgent with health monitoring capabilities
class UGROAgentWithHealthMonitoring(UGROAgent):
    """UGRO Agent with integrated health monitoring."""
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._health_integration = HealthMonitorIntegration(self)

    async def start_health_monitoring(self, config: MonitoringConfig | None = None) -> None:
        """Start health monitoring."""
        await self._health_integration.start_health_monitoring(config)

    async def stop_health_monitoring(self) -> None:
        """Stop health monitoring."""
        await self._health_integration.stop_health_monitoring()

    def get_health_status(self) -> dict[str, Any]:
        """Get health monitoring status."""
        return self._health_integration.get_monitoring_status()

    def get_training_metrics(self, job_id: str, rank: int | None = None):
        """Get training metrics for a specific job and optional rank."""
        return self._health_integration.get_training_metrics(job_id, rank)

    def get_job_metrics_statistics(self, job_id: str) -> dict[str, Any]:
        """Get statistics for a specific job's metrics."""
        return self._health_integration.get_job_metrics_statistics(job_id)

    def launch_training(self, *args, **kwargs) -> bool:
        """Launch training with health monitoring and metrics collection integration."""
        job_name = kwargs.get("job_name", "unknown")
        ranks = kwargs.get("ranks", list(range(kwargs.get("num_gpus", 1))))
        
        # Register job start with ranks for metrics collection
        self._health_integration.register_job_start(job_name, ranks)
        
        try:
            # Launch training using parent method
            success = super().launch_training(*args, **kwargs)
            return success
        finally:
            # Register job end (regardless of success)
            self._health_integration.register_job_end(job_name)
