"""Example usage and demonstration of the health monitoring system.

This module provides practical examples of how to use the health monitor
in different scenarios, including standalone usage and integration with UGRO.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from ugro.config import load_config
from ugro.health_integration import UGROAgentWithHealthMonitoring
from ugro.health_monitor import (
    AdaptiveHealthMonitor, 
    MonitoringConfig, 
    TrainingMetricsCollector,
    MetricsCollectorConfig,
    create_health_monitor,
    create_metrics_collector
)


async def standalone_health_monitor_example() -> None:
    """Example of running health monitor standalone."""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Load configuration
    config = load_config()
    
    # Create cluster and state manager (mock for example)
    from ugro.cluster import Cluster
    from ugro.cluster_state import ClusterStateManager
    
    cluster = Cluster(config)
    state_manager = ClusterStateManager()
    
    # Create monitoring configuration
    monitor_config = MonitoringConfig(
        base_interval=5.0,  # Check every 5 seconds
        max_interval=30.0,  # Max 30 seconds when idle
        min_interval=2.0,  # Min 2 seconds during active jobs
        max_workers=4,     # 4 concurrent workers
        metrics_history_size=50,  # Keep 50 historical entries
    )
    
    # Create and start health monitor
    monitor = create_health_monitor(cluster, state_manager, monitor_config)
    
    print("Starting health monitor...")
    
    try:
        # Run for a limited time for demonstration
        monitor_task = asyncio.create_task(monitor.start_monitoring())
        
        # Simulate some job activity
        await asyncio.sleep(10)
        print("Registering job activity...")
        monitor.register_job_activity("demo-job")
        
        await asyncio.sleep(20)
        print("Unregistering job activity...")
        monitor.unregister_job_activity("demo-job")
        
        # Print statistics
        await asyncio.sleep(10)
        stats = monitor.get_statistics()
        print(f"Monitoring statistics: {stats}")
        
    except KeyboardInterrupt:
        print("Stopping health monitor...")
    finally:
        await monitor.stop()
        monitor_task.cancel()
        try:
            await monitor_task
        except asyncio.CancelledError:
            pass


async def integrated_ugro_example() -> None:
    """Example of using health monitor with UGRO agent."""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Create enhanced UGRO agent with health monitoring
    agent = UGROAgentWithHealthMonitoring()
    
    # Configure health monitoring
    monitor_config = MonitoringConfig(
        base_interval=10.0,
        max_interval=60.0,
        min_interval=5.0,
        critical_threshold=40.0,  # More sensitive
        warning_threshold=75.0,   # More sensitive
    )
    
    print("Starting UGRO agent with health monitoring...")
    
    try:
        # Start health monitoring
        await agent.start_health_monitoring(monitor_config)
        
        # Launch a training job (this will automatically register job activity)
        success = agent.launch_training(
            job_name="demo-training",
            model="resnet50",
            dataset="cifar10",
            epochs=2,
            learning_rate=0.001,
        )
        
        if success:
            print("Training launched successfully!")
            
            # Monitor health status
            for _ in range(30):  # Monitor for 30 seconds
                await asyncio.sleep(1)
                health_status = agent.get_health_status()
                
                if health_status["status"] == "running":
                    stats = health_status["statistics"]
                    if stats["critical_alerts"] > 0:
                        print(f"⚠️  {stats['critical_alerts']} critical alerts detected")
                    if stats["warning_alerts"] > 0:
                        print(f"⚠️  {stats['warning_alerts']} warning alerts detected")
                    
                    print(f"Health score: {stats['average_health_score']:.1f}, "
                          f"Active jobs: {stats['active_jobs']}")
        
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        await agent.stop_health_monitoring()


async def custom_alert_handler_example() -> None:
    """Example of custom alert handling."""
    from ugro.health_monitor import AlertLevel
    
    # Custom alert handler
    class CustomAlertHandler:
        def __init__(self) -> None:
            self.alerts = []
        
        async def handle_alert(self, alert: str) -> None:
            """Handle custom alert logic."""
            self.alerts.append(alert)
            
            if alert.startswith(AlertLevel.CRITICAL):
                print(f"🚨 CRITICAL: {alert}")
                # Send to PagerDuty, Slack, etc.
            elif alert.startswith(AlertLevel.WARNING):
                print(f"⚠️  WARNING: {alert}")
                # Send to monitoring dashboard
            else:
                print(f"ℹ️  INFO: {alert}")
    
    # Create monitor with custom alert handling
    config = load_config()
    cluster = Cluster(config)
    state_manager = ClusterStateManager()
    
    monitor = create_health_monitor(cluster, state_manager)
    alert_handler = CustomAlertHandler()
    
    # Override the alert sending method
    original_send_alerts = monitor._send_alerts
    
    async def custom_send_alerts(alerts: list[str]) -> None:
        for alert in alerts:
            await alert_handler.handle_alert(alert)
        # Still call original for logging
        await original_send_alerts(alerts)
    
    monitor._send_alerts = custom_send_alerts
    
    print("Running health monitor with custom alert handling...")
    
    try:
        # Run for demonstration
        monitor_task = asyncio.create_task(monitor.start_monitoring())
        await asyncio.sleep(30)
        
        print(f"Total alerts handled: {len(alert_handler.alerts)}")
        for alert in alert_handler.alerts:
            print(f"  - {alert}")
            
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        await monitor.stop()
        monitor_task.cancel()


async def performance_monitoring_example() -> None:
    """Example of performance-focused monitoring configuration."""
    # High-performance configuration for large clusters
    perf_config = MonitoringConfig(
        base_interval=2.0,      # Very frequent checks
        max_interval=15.0,     # Short max interval
        min_interval=1.0,      # Very frequent during jobs
        max_workers=16,         # High concurrency
        metrics_history_size=200,  # More history
        cleanup_interval_hours=12,  # More frequent cleanup
        
        # Adjusted thresholds for performance monitoring
        critical_threshold=30.0,
        warning_threshold=60.0,
        
        # Weights optimized for performance workloads
        gpu_weight=50.0,       # Focus on GPU health
        system_weight=25.0,    # Less system focus
        network_weight=15.0,   # Some network focus
        process_weight=10.0,   # Process monitoring
    )
    
    config = load_config()
    cluster = Cluster(config)
    state_manager = ClusterStateManager()
    
    monitor = create_health_monitor(cluster, state_manager, perf_config)
    
    print("Starting performance-focused health monitoring...")
    
    try:
        # Simulate high-activity scenario
        monitor.register_job_activity("perf-test-job-1")
        monitor.register_job_activity("perf-test-job-2")
        
        monitor_task = asyncio.create_task(monitor.start_monitoring())
        
        # Monitor performance metrics
        for i in range(60):  # 1 minute of monitoring
            await asyncio.sleep(1)
            
            if i % 10 == 0:  # Print stats every 10 seconds
                stats = monitor.get_statistics()
                print(f"Time: {i}s, Checks: {stats['total_checks']}, "
                      f"Avg Health: {stats['average_health_score']:.1f}, "
                      f"Failures: {stats['failed_checks']}")
        
        # Unregister jobs
        monitor.unregister_job_activity("perf-test-job-1")
        monitor.unregister_job_activity("perf-test-job-2")
        
        await asyncio.sleep(5)  # See adaptive polling in action
        
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        await monitor.stop()
        monitor_task.cancel()


def configuration_examples() -> None:
    """Examples of different monitoring configurations."""
    
    # Development configuration
    dev_config = MonitoringConfig(
        base_interval=5.0,
        max_interval=30.0,
        min_interval=2.0,
        max_workers=4,
        critical_threshold=40.0,
        warning_threshold=70.0,
    )
    print("Development configuration:")
    print(f"  Interval: {dev_config.min_interval}s - {dev_config.max_interval}s")
    print(f"  Thresholds: Critical < {dev_config.critical_threshold}, Warning < {dev_config.warning_threshold}")
    
    # Production configuration
    prod_config = MonitoringConfig(
        base_interval=15.0,
        max_interval=120.0,
        min_interval=5.0,
        max_workers=8,
        critical_threshold=30.0,
        warning_threshold=60.0,
        metrics_history_size=1000,
        cleanup_interval_hours=48,
    )
    print("\nProduction configuration:")
    print(f"  Interval: {prod_config.min_interval}s - {prod_config.max_interval}s")
    print(f"  History: {prod_config.metrics_history_size} entries")
    print(f"  Cleanup: Every {prod_config.cleanup_interval_hours} hours")
    
    # High-security configuration
    security_config = MonitoringConfig(
        base_interval=10.0,
        max_interval=60.0,
        min_interval=3.0,
        max_workers=6,
        critical_threshold=50.0,  # More sensitive
        warning_threshold=80.0,  # More sensitive
        gpu_weight=45.0,         # Focus on GPU
        system_weight=35.0,      # System health
        network_weight=15.0,    # Network monitoring
        process_weight=5.0,     # Less process focus
    )
    print("\nSecurity configuration:")
    print(f"  Sensitive thresholds: Critical < {security_config.critical_threshold}")
    print(f"  Weights: GPU {security_config.gpu_weight}%, System {security_config.system_weight}%")


async def metrics_collector_example() -> None:
    """Example of using the training metrics collector."""
    print("\n=== Training Metrics Collector Example ===")
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Load configuration
    config = load_config()
    
    # Create cluster
    from ugro.cluster import Cluster
    cluster = Cluster(config)
    
    # Create metrics collector configuration
    metrics_config = MetricsCollectorConfig(
        collection_interval=3.0,  # Collect every 3 seconds
        max_history_size=100,    # Keep 100 entries per rank
        low_throughput_threshold=100.0,  # Alert if < 100 tokens/sec
        high_loss_threshold=5.0,         # Alert if loss > 5.0
        low_gpu_util_threshold=40.0      # Alert if GPU util < 40%
    )
    
    # Create metrics collector
    collector = create_metrics_collector(cluster, metrics_config)
    
    print("Starting metrics collector...")
    
    try:
        # Simulate a training job
        job_id = "example_job_001"
        ranks = [0, 1, 2, 3]  # 4 GPU ranks
        
        print(f"Starting metrics collection for job {job_id}")
        await collector.start_collection(job_id, ranks)
        
        # Let it run for a bit to collect metrics
        print("Collecting metrics for 15 seconds...")
        await asyncio.sleep(15)
        
        # Get latest metrics
        latest_metrics = collector.get_latest_metrics(job_id)
        if latest_metrics:
            print(f"\nLatest metrics from {len(latest_metrics)} ranks:")
            for metrics in latest_metrics:
                print(f"  Rank {metrics.rank}:")
                print(f"    GPU Util: {metrics.gpu_util:.1f}%")
                print(f"    Memory: {metrics.gpu_mem_used_gb:.1f} GB")
                print(f"    Loss: {metrics.training_loss:.3f}")
                print(f"    Throughput: {metrics.throughput_tokens_sec:.1f} tokens/s")
                print(f"    Efficiency: {metrics.efficiency_score:.1f}%")
        
        # Get job statistics
        stats = collector.get_job_statistics(job_id)
        if stats.get("status") == "active":
            print(f"\nJob Statistics:")
            print(f"  Ranks: {stats['ranks']}")
            print(f"  Total metrics: {stats['total_metrics']}")
            print(f"  Avg GPU util: {stats['avg_gpu_util']:.1f}%")
            print(f"  Avg throughput: {stats['avg_throughput']:.1f} tokens/s")
            print(f"  Avg loss: {stats['avg_loss']:.3f}")
        
        # Get collector statistics
        collector_stats = collector.get_statistics()
        print(f"\nCollector Statistics:")
        print(f"  Active jobs: {collector_stats['active_jobs']}")
        print(f"  Metrics collected: {collector_stats['metrics_collected']}")
        print(f"  Failed collections: {collector_stats['failed_collections']}")
        
    finally:
        # Stop collection
        await collector.stop_collection(job_id)
        await collector.stop()
        print("\nMetrics collector stopped")


async def main() -> None:
    """Main function to run examples."""
    print("UGRO Health Monitor Examples")
    print("=" * 50)
    
    # Show configuration examples
    print("\n1. Configuration Examples:")
    configuration_examples()
    
    print("\n" + "=" * 50)
    print("Choose an example to run:")
    print("1. Standalone health monitor")
    print("2. Integrated UGRO agent")
    print("3. Custom alert handler")
    print("4. Performance monitoring")
    print("5. Training metrics collector")
    
    try:
        choice = input("\nEnter choice (1-5, or press Enter for standalone): ").strip()
        
        match choice:
            case "1" | "":
                print("\nRunning standalone health monitor example...")
                await standalone_health_monitor_example()
            case "2":
                print("\nRunning integrated UGRO example...")
                await integrated_ugro_example()
            case "3":
                print("\nRunning custom alert handler example...")
                await custom_alert_handler_example()
            case "4":
                print("\nRunning performance monitoring example...")
                await performance_monitoring_example()
            case "5":
                print("\nRunning training metrics collector example...")
                await metrics_collector_example()
            case _:
                print("Invalid choice, running standalone example...")
                await standalone_health_monitor_example()
                
    except KeyboardInterrupt:
        print("\nExample interrupted by user")
    except Exception as e:
        print(f"\nExample failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run the examples
    asyncio.run(main())
