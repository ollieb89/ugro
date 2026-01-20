#!/usr/bin/env python3
"""Examples demonstrating the Metrics Collector functionality for real-time training telemetry.

This example shows how to use the TrainingMetricsCollector to collect, store,
and analyze training metrics from distributed training jobs.
"""

import asyncio
import json
import logging
import random
import time
from datetime import datetime
from pathlib import Path

from src.ugro.health_monitor import (
    TrainingMetrics,
    TrainingMetricsCollector,
    MetricsCollectorConfig,
    create_metrics_collector
)
from src.ugro.cluster import Cluster
from src.ugro.cluster_state import ClusterStateManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_mock_cluster() -> Cluster:
    """Create a mock cluster for demonstration."""
    # This would normally load from config
    class MockCluster:
        def get_all_workers(self):
            return [
                {"name": f"gpu{i}", "ip": f"192.168.1.{10+i}"}
                for i in range(4)
            ]
        
        async def execute_on_worker(self, node_name: str, command: str, timeout: int = 10):
            # Mock execution with simulated metrics
            if "nvidia-smi" in command:
                gpu_util = random.uniform(70, 95)
                gpu_mem = random.uniform(8, 16)
                return True, f"{gpu_util:.1f},{gpu_mem:.1f}", ""
            elif "cat" in command and "training_metrics.json" in command:
                # Mock training metrics file
                metrics = {
                    "gpu_util": random.uniform(70, 95),
                    "gpu_mem_used_gb": random.uniform(8, 16),
                    "training_loss": random.uniform(0.1, 5.0),
                    "throughput_tokens_sec": random.uniform(100, 200),
                    "gradient_norm": random.uniform(0.5, 3.0),
                    "learning_rate": 0.0002
                }
                return True, json.dumps(metrics), ""
            else:
                return True, "mock output", ""
    
    return MockCluster()


def create_mock_state_manager() -> ClusterStateManager:
    """Create a mock state manager for demonstration."""
    class MockStateManager:
        def update_node_status(self, node_name: str, **kwargs):
            pass
    
    return MockStateManager()


async def example_1_basic_metrics_collection():
    """Example 1: Basic metrics collection setup and usage."""
    print("\n=== Example 1: Basic Metrics Collection ===")
    
    # Create cluster and collector
    cluster = create_mock_cluster()
    config = MetricsCollectorConfig(
        collection_interval=2.0,  # Collect every 2 seconds
        max_history_size=50,
        low_throughput_threshold=100.0,
        high_loss_threshold=3.0
    )
    
    collector = create_metrics_collector(cluster, config)
    
    try:
        # Start collecting metrics for a job
        job_id = "job_001"
        ranks = [0, 1, 2, 3]  # 4 GPU ranks
        
        print(f"Starting metrics collection for {job_id} with {len(ranks)} ranks...")
        await collector.start_collection(job_id, ranks)
        
        # Let it run for a bit
        await asyncio.sleep(10)
        
        # Get latest metrics
        latest_metrics = collector.get_latest_metrics(job_id)
        print(f"Latest metrics from {len(latest_metrics)} ranks:")
        
        for metrics in latest_metrics:
            print(f"  Rank {metrics.rank}: "
                  f"GPU={metrics.gpu_util:.1f}%, "
                  f"Loss={metrics.training_loss:.3f}, "
                  f"Throughput={metrics.throughput_tokens_sec:.1f} tokens/s")
        
        # Get job statistics
        stats = collector.get_job_statistics(job_id)
        print(f"\nJob Statistics:")
        print(f"  Total metrics collected: {stats['total_metrics']}")
        print(f"  Average GPU utilization: {stats['avg_gpu_util']:.1f}%")
        print(f"  Average throughput: {stats['avg_throughput']:.1f} tokens/s")
        print(f"  Average loss: {stats['avg_loss']:.3f}")
        print(f"  Average efficiency: {stats['avg_efficiency']:.1f}%")
        
    finally:
        # Stop collection
        await collector.stop_collection(job_id)
        await collector.stop()
        print("Metrics collection stopped")


async def example_2_historical_analysis():
    """Example 2: Historical metrics analysis."""
    print("\n=== Example 2: Historical Analysis ===")
    
    cluster = create_mock_cluster()
    config = MetricsCollectorConfig(
        collection_interval=1.0,
        max_history_size=100
    )
    
    collector = create_metrics_collector(cluster, config)
    
    try:
        job_id = "job_002"
        ranks = [0, 1]
        
        print(f"Starting collection for historical analysis...")
        await collector.start_collection(job_id, ranks)
        
        # Collect data for longer period
        await asyncio.sleep(15)
        
        # Analyze specific rank
        rank = 0
        history = collector.get_metrics_history(job_id, rank, limit=10)
        
        print(f"\nLast 10 metrics for rank {rank}:")
        for i, metrics in enumerate(history):
            print(f"  {i+1}: Loss={metrics.training_loss:.3f}, "
                  f"Efficiency={metrics.efficiency_score:.1f}%")
        
        # Show trends
        if len(history) >= 2:
            recent_loss = [m.training_loss for m in history[-5:]]
            avg_recent_loss = sum(recent_loss) / len(recent_loss)
            print(f"\nRecent average loss (last 5): {avg_recent_loss:.3f}")
            
            # Check if loss is decreasing
            if len(recent_loss) >= 2:
                trend = "decreasing" if recent_loss[-1] < recent_loss[0] else "increasing"
                print(f"Loss trend: {trend}")
        
    finally:
        await collector.stop_collection(job_id)
        await collector.stop()


async def example_3_performance_monitoring():
    """Example 3: Performance monitoring and alerts."""
    print("\n=== Example 3: Performance Monitoring ===")
    
    # Configure with performance thresholds
    config = MetricsCollectorConfig(
        collection_interval=1.5,
        low_throughput_threshold=120.0,
        high_loss_threshold=2.0,
        low_gpu_util_threshold=50.0
    )
    
    cluster = create_mock_cluster()
    collector = create_metrics_collector(cluster, config)
    
    try:
        job_id = "job_003"
        ranks = [0, 1, 2]
        
        print("Starting performance monitoring...")
        await collector.start_collection(job_id, ranks)
        
        # Monitor for performance issues
        for i in range(8):
            await asyncio.sleep(2)
            
            # Get current metrics
            latest = collector.get_latest_metrics(job_id)
            
            # Check for performance issues
            for metrics in latest:
                issues = []
                
                if metrics.throughput_tokens_sec < config.low_throughput_threshold:
                    issues.append(f"Low throughput: {metrics.throughput_tokens_sec:.1f}")
                
                if metrics.training_loss > config.high_loss_threshold:
                    issues.append(f"High loss: {metrics.training_loss:.3f}")
                
                if metrics.gpu_util < config.low_gpu_util_threshold:
                    issues.append(f"Low GPU util: {metrics.gpu_util:.1f}%")
                
                if issues:
                    print(f"⚠️  Rank {metrics.rank} performance issues: {', '.join(issues)}")
        
        # Get overall statistics
        stats = collector.get_job_statistics(job_id)
        print(f"\nFinal Performance Summary:")
        print(f"  Average efficiency: {stats['avg_efficiency']:.1f}%")
        print(f"  Total metrics: {stats['total_metrics']}")
        
    finally:
        await collector.stop_collection(job_id)
        await collector.stop()


async def example_4_integration_with_health_monitoring():
    """Example 4: Integration with health monitoring system."""
    print("\n=== Example 4: Integration with Health Monitoring ===")
    
    from src.ugro.health_integration import UGROAgentWithHealthMonitoring
    
    # This would normally be a real agent
    class MockUGROAgent:
        def __init__(self):
            self.cluster = create_mock_cluster()
            self.cluster_state_manager = create_mock_state_manager()
        
        def launch_training(self, *args, **kwargs):
            job_name = kwargs.get("job_name", "test_job")
            print(f"Launching training job: {job_name}")
            # Simulate training launch
            return True
    
    # Create agent with health monitoring
    agent = MockUGROAgent()
    health_agent = UGROAgentWithHealthMonitoring.__new__(UGROAgentWithHealthMonitoring)
    health_agent.__dict__.update(agent.__dict__)
    health_agent._health_integration = health_agent._health_integration.__class__(health_agent)
    
    try:
        # Start health monitoring
        await health_agent.start_health_monitoring()
        print("Health monitoring started")
        
        # Launch training with metrics collection
        job_id = "job_004"
        ranks = [0, 1, 2, 3]
        
        success = health_agent.launch_training(
            job_name=job_id,
            ranks=ranks,
            num_gpus=4
        )
        
        if success:
            print(f"Training launched for {job_id}")
            await asyncio.sleep(5)
            
            # Get health status
            status = health_agent.get_health_status()
            print(f"Health monitoring status: {status['status']}")
            
            if 'metrics_collector' in status:
                mc_stats = status['metrics_collector']['statistics']
                print(f"Metrics collector: {mc_stats['active_jobs']} active jobs, "
                      f"{mc_stats['metrics_collected']} metrics collected")
            
            # Get training metrics
            metrics = health_agent.get_training_metrics(job_id)
            if metrics:
                if isinstance(metrics, list):
                    print(f"Collected metrics from {len(metrics)} ranks")
                else:
                    print(f"Latest metrics: GPU={metrics.gpu_util:.1f}%, "
                          f"Loss={metrics.training_loss:.3f}")
        
    finally:
        await health_agent.stop_health_monitoring()
        print("Health monitoring stopped")


async def example_5_custom_metrics_validation():
    """Example 5: Custom metrics validation and efficiency analysis."""
    print("\n=== Example 5: Metrics Validation and Analysis ===")
    
    # Create custom metrics manually for validation
    custom_metrics = [
        TrainingMetrics(
            timestamp=datetime.now(),
            job_id="test_job",
            rank=0,
            gpu_util=85.5,
            gpu_mem_used_gb=12.3,
            training_loss=2.1,
            throughput_tokens_sec=156.7,
            gradient_norm=1.8,
            learning_rate=0.0002
        ),
        TrainingMetrics(
            timestamp=datetime.now(),
            job_id="test_job",
            rank=1,
            gpu_util=92.1,
            gpu_mem_used_gb=14.7,
            training_loss=1.8,
            throughput_tokens_sec=189.3,
            gradient_norm=1.5,
            learning_rate=0.0002
        )
    ]
    
    print("Custom metrics validation:")
    for metrics in custom_metrics:
        print(f"  Rank {metrics.rank}:")
        print(f"    GPU Utilization: {metrics.gpu_util}% (✓ Valid: 0-100)")
        print(f"    Memory Used: {metrics.gpu_mem_used_gb} GB (✓ Valid: >=0)")
        print(f"    Training Loss: {metrics.training_loss} (✓ Valid: >=0)")
        print(f"    Throughput: {metrics.throughput_tokens_sec} tokens/s (✓ Valid: >=0)")
        print(f"    Gradient Norm: {metrics.gradient_norm} (✓ Valid: >=0)")
        print(f"    Learning Rate: {metrics.learning_rate} (✓ Valid: >=0)")
        print(f"    Efficiency Score: {metrics.efficiency_score:.1f}%")
        print()
    
    # Demonstrate efficiency calculation
    print("Efficiency Analysis:")
    for metrics in custom_metrics:
        efficiency = metrics.efficiency_score
        if efficiency >= 80:
            rating = "Excellent"
        elif efficiency >= 60:
            rating = "Good"
        elif efficiency >= 40:
            rating = "Fair"
        else:
            rating = "Poor"
        
        print(f"  Rank {metrics.rank}: {efficiency:.1f}% ({rating})")


async def main():
    """Run all examples."""
    print("🚀 UGRO Metrics Collector Examples")
    print("=" * 50)
    
    try:
        await example_1_basic_metrics_collection()
        await example_2_historical_analysis()
        await example_3_performance_monitoring()
        await example_4_integration_with_health_monitoring()
        await example_5_custom_metrics_validation()
        
        print("\n✅ All examples completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⚠️  Examples interrupted by user")
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
