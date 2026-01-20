"""Optimized health monitoring for UGRO clusters with adaptive polling and comprehensive metrics.

This module implements a production-grade health monitoring system that provides:
- Adaptive polling intervals based on cluster activity
- Concurrent metric collection with async/await patterns
- Circuit breaker pattern for failure resilience
- Comprehensive health scoring and alerting
- Integration with existing UGRO infrastructure

Python 3.12+ features used:
- Modern type annotations (PEP 604, PEP 695)
- Structural pattern matching (match/case)
- Exception Groups for error handling
- Enhanced dataclasses and typing
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from collections.abc import AsyncGenerator, Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import IntEnum, StrEnum
from typing import Any, Final, NoReturn, TypeAlias, TYPE_CHECKING

if TYPE_CHECKING:
    from .ssh_utils import SSHClient

from .cluster import Cluster
from .cluster_state import ClusterStateManager, NodeState
from .result_aggregator import ResultAggregator

# Type aliases for better readability
HealthScore: TypeAlias = float
NodeName: TypeAlias = str
AlertMessage: TypeAlias = str
JobID: TypeAlias = str
RankID: TypeAlias = int


class AlertLevel(StrEnum):
    """Alert severity levels following Python 3.10+ StrEnum pattern."""
    CRITICAL = "CRITICAL"
    WARNING = "WARNING"
    INFO = "INFO"


class NodeStatus(StrEnum):
    """Node status enumeration."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class CircuitBreakerState(IntEnum):
    """Circuit breaker states for failure handling."""
    CLOSED = 0      # Normal operation
    OPEN = 1        # Failing, stop trying
    HALF_OPEN = 2   # Testing recovery


@dataclass(frozen=True, slots=True)
class TrainingMetrics:
    """Real-time training metrics from distributed training ranks.
    
    Uses frozen dataclass with slots for memory efficiency and immutability.
    Python 3.12+ features: dataclass_transform, slots, frozen.
    """
    timestamp: datetime
    job_id: JobID
    rank: RankID
    gpu_util: float
    gpu_mem_used_gb: float
    training_loss: float
    throughput_tokens_sec: float
    gradient_norm: float
    learning_rate: float
    
    def __post_init__(self) -> None:
        """Validate training metrics after initialization."""
        if self.gpu_util < 0 or self.gpu_util > 100:
            raise ValueError(f"GPU utilization must be 0-100, got {self.gpu_util}")
        
        if self.gpu_mem_used_gb < 0:
            raise ValueError(f"GPU memory used cannot be negative, got {self.gpu_mem_used_gb}")
        
        if self.training_loss < 0:
            raise ValueError(f"Training loss cannot be negative, got {self.training_loss}")
        
        if self.throughput_tokens_sec < 0:
            raise ValueError(f"Throughput cannot be negative, got {self.throughput_tokens_sec}")
        
        if self.gradient_norm < 0:
            raise ValueError(f"Gradient norm cannot be negative, got {self.gradient_norm}")
        
        if self.learning_rate < 0:
            raise ValueError(f"Learning rate cannot be negative, got {self.learning_rate}")
    
    @property
    def efficiency_score(self) -> float:
        """Calculate training efficiency score (0-100) based on metrics."""
        # Higher GPU utilization and throughput with reasonable loss = better efficiency
        util_score = self.gpu_util
        throughput_score = min(100, self.throughput_tokens_sec / 2)  # Scale throughput
        loss_penalty = max(0, 50 - self.training_loss) if self.training_loss > 1 else 0

        raw_score = (util_score + throughput_score) / 2 - loss_penalty
        return max(0.0, min(100.0, raw_score))


@dataclass(frozen=True, slots=True)
class HealthMetrics:
    """Comprehensive health metrics for a cluster node.
    
    Uses frozen dataclass with slots for memory efficiency and immutability.
    Python 3.12+ features: dataclass_transform, slots, frozen.
    """
    node_name: str
    timestamp: datetime
    gpu_utilization: float
    gpu_memory_used: float
    gpu_memory_total: float
    gpu_temperature: float
    gpu_power_usage: float
    cpu_utilization: float
    memory_usage: float
    disk_usage: float
    network_latency: float
    process_status: dict[str, bool] = field(default_factory=dict)
    health_score: HealthScore = 0.0
    alerts: list[AlertMessage] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate metrics after initialization."""
        if not 0 <= self.health_score <= 100:
            raise ValueError(f"Health score must be 0-100, got {self.health_score}")
        
        if self.gpu_utilization < 0 or self.gpu_temperature < 0:
            raise ValueError("GPU metrics cannot be negative")

    @property
    def status(self) -> NodeStatus:
        """Determine node status based on health score."""
        match self.health_score:
            case score if score >= 80:
                return NodeStatus.HEALTHY
            case score if score >= 60:
                return NodeStatus.DEGRADED
            case score if score >= 0:
                return NodeStatus.UNHEALTHY
            case _:
                return NodeStatus.UNKNOWN

    @property
    def critical_alerts(self) -> list[AlertMessage]:
        """Get only critical alerts."""
        return [alert for alert in self.alerts if alert.startswith(AlertLevel.CRITICAL)]

    @property
    def warning_alerts(self) -> list[AlertMessage]:
        """Get only warning alerts."""
        return [alert for alert in self.alerts if alert.startswith(AlertLevel.WARNING)]


@dataclass(slots=True)
class CircuitBreaker:
    """Circuit breaker implementation for failure resilience.
    
    Python 3.12+ features: dataclass slots, type annotations.
    """
    failure_count: int = 0
    last_failure_time: datetime | None = None
    state: CircuitBreakerState = CircuitBreakerState.CLOSED
    
    # Configuration constants
    MAX_FAILURES: Final[int] = 3
    TIMEOUT_SECONDS: Final[int] = 300  # 5 minutes
    
    def record_failure(self) -> None:
        """Record a failure and potentially open the circuit breaker."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.MAX_FAILURES:
            self.state = CircuitBreakerState.OPEN
    
    def record_success(self) -> None:
        """Record success and reset or close the circuit breaker."""
        self.failure_count = 0
        self.state = CircuitBreakerState.CLOSED
        self.last_failure_time = None
    
    def is_open(self) -> bool:
        """Check if circuit breaker is open."""
        if self.state != CircuitBreakerState.OPEN:
            return False
        
        # Check if timeout has expired
        if self.last_failure_time and (
            datetime.now() - self.last_failure_time
        ) > timedelta(seconds=self.TIMEOUT_SECONDS):
            self.state = CircuitBreakerState.HALF_OPEN
            return False
        
        return True


@dataclass(slots=True)
class MonitoringConfig:
    """Configuration for health monitoring with Python 3.12+ patterns."""
    base_interval: float = 10.0
    max_interval: float = 60.0
    min_interval: float = 5.0
    
    # Circuit breaker settings
    max_failures: int = 3
    circuit_breaker_timeout: int = 300
    
    # Performance settings
    max_workers: int = 8
    metrics_history_size: int = 100
    cleanup_interval_hours: int = 24
    
    # Health score weights (must sum to 100)
    gpu_weight: float = 40.0
    system_weight: float = 30.0
    network_weight: float = 20.0
    process_weight: float = 10.0
    
    # Thresholds
    critical_threshold: float = 50.0
    warning_threshold: float = 70.0
    
    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        total_weight = self.gpu_weight + self.system_weight + self.network_weight + self.process_weight
        if abs(total_weight - 100.0) > 0.1:
            raise ValueError(f"Weights must sum to 100, got {total_weight}")
        
        if self.min_interval >= self.base_interval >= self.max_interval:
            raise ValueError("Interval configuration invalid: min < base < max")


@dataclass(slots=True)
class MetricsCollectorConfig:
    """Configuration for training metrics collection."""
    collection_interval: float = 5.0  # Seconds between collections
    max_history_size: int = 1000  # Max metrics per job
    cleanup_interval_hours: int = 24
    collection_timeout: int = 10  # Seconds to wait for metrics
    
    # Metrics file paths (relative to training job directory)
    metrics_file: str = "training_metrics.json"
    log_file: str = "training.log"
    
    # Performance thresholds for alerts
    low_throughput_threshold: float = 50.0  # tokens/sec
    high_loss_threshold: float = 10.0
    low_gpu_util_threshold: float = 30.0  # percentage
    
    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.collection_interval <= 0:
            raise ValueError("Collection interval must be positive")
        
        if self.max_history_size <= 0:
            raise ValueError("Max history size must be positive")


class AdaptiveHealthMonitor:
    """Production health monitor with adaptive polling and smart error handling.
    
    This class implements the core health monitoring logic using modern Python 3.12+
    features including async/await, structural pattern matching, and enhanced typing.
    """
    
    def __init__(
        self,
        cluster: Cluster,
        state_manager: ClusterStateManager,
        config: MonitoringConfig | None = None,
    ) -> None:
        """Initialize the adaptive health monitor.
        
        Args:
            cluster: UGRO cluster instance for health checks
            state_manager: Cluster state manager for persistence
            config: Monitoring configuration, defaults to sensible values
        """
        self.cluster = cluster
        self.state_manager = state_manager
        self.config = config or MonitoringConfig()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Circuit breakers for each node
        self._circuit_breakers: dict[NodeName, CircuitBreaker] = {}
        
        # Performance optimization
        self._executor = ThreadPoolExecutor(max_workers=self.config.max_workers)
        self._metrics_history: dict[NodeName, list[HealthMetrics]] = {}
        
        # State tracking
        self._active_jobs: set[str] = set()
        self._last_job_activity = datetime.now()
        self._health_scores: dict[NodeName, HealthScore] = {}
        self._running = False
        
        # Statistics
        self._stats = {
            "total_checks": 0,
            "failed_checks": 0,
            "critical_alerts": 0,
            "warning_alerts": 0,
        }

    @asynccontextmanager
    async def _monitoring_lifecycle(self) -> AsyncGenerator[None, None]:
        """Context manager for monitoring lifecycle management."""
        try:
            self._running = True
            self.logger.info("Health monitoring started")
            yield
        finally:
            self._running = False
            self.logger.info("Health monitoring stopped")

    async def start_monitoring(self) -> NoReturn:
        """Main monitoring loop with adaptive polling.
        
        This method runs indefinitely and should be called as an async task.
        Uses Python 3.12+ exception groups for comprehensive error handling.
        """
        async with self._monitoring_lifecycle():
            while self._running:
                start_time = time.time()
                
                try:
                    # Get adaptive polling interval
                    interval = self._calculate_adaptive_interval()
                    
                    # Collect metrics concurrently
                    metrics = await self._collect_metrics_concurrently()
                    
                    # Process metrics and update state
                    await self._process_metrics(metrics)
                    
                    # Check for critical conditions
                    await self._check_critical_conditions(metrics)
                    
                    # Clean up old data periodically
                    if self._stats["total_checks"] % 100 == 0:
                        self._cleanup_old_metrics()
                    
                    # Calculate precise sleep time
                    execution_time = time.time() - start_time
                    sleep_time = max(0, interval - execution_time)
                    
                    self.logger.debug(
                        f"Health check completed in {execution_time:.2f}s, "
                        f"sleeping {sleep_time:.2f}s"
                    )
                    await asyncio.sleep(sleep_time)
                    
                    # Update statistics
                    self._stats["total_checks"] += 1
                    
                except* Exception as eg:
                    # Python 3.12+ exception groups for comprehensive error handling
                    self.logger.error(f"Health monitoring errors: {eg}")
                    self._stats["failed_checks"] += 1
                    await asyncio.sleep(self.config.base_interval)

    def _calculate_adaptive_interval(self) -> float:
        """Calculate polling interval based on cluster activity.
        
        Returns:
            Adaptive polling interval in seconds
        """
        if self._active_jobs:
            # Active jobs - more frequent monitoring
            return self.config.min_interval
        
        # No active jobs - check time since last activity
        time_since_activity = datetime.now() - self._last_job_activity
        
        match time_since_activity:
            case delta if delta < timedelta(minutes=30):
                return self.config.base_interval
            case delta if delta < timedelta(hours=2):
                return self.config.base_interval * 2
            case delta if delta < timedelta(hours=6):
                return self.config.base_interval * 3
            case _:
                return min(self.config.max_interval, self.config.base_interval * 4)

    async def _collect_metrics_concurrently(self) -> list[HealthMetrics]:
        """Collect metrics from all nodes concurrently.
        
        Returns:
            List of health metrics from responsive nodes
        """
        workers = self.cluster.get_all_workers()
        
        # Create concurrent tasks for each worker
        tasks = {
            worker["name"]: asyncio.create_task(
                self._collect_node_metrics(worker),
                name=f"collect_{worker['name']}"
            )
            for worker in workers
            if not self._is_circuit_breaker_open(worker["name"])
        }
        
        # Wait for all tasks with comprehensive error handling
        results = []
        exceptions = []
        
        for node_name, task in tasks.items():
            try:
                metrics = await task
                if metrics:
                    results.append(metrics)
                    # Reset circuit breaker on success
                    self._reset_circuit_breaker(node_name)
            except Exception as e:
                exceptions.append(e)
                self._handle_node_failure(node_name, e)
        
        # Log exceptions if any
        if exceptions:
            self.logger.warning(f"Failed to collect metrics from {len(exceptions)} nodes")
        
        return results
    async def check_node_health(self, worker: dict[str, Any]) -> bool:
        """Quick diagnostic of node as per Phase 2b specification."""
        node_name = worker["name"]
        
        # Concurrent checks for speed
        checks = {
            "ssh_reachable": self.test_ssh(worker),
            "gpu_available": self.test_gpu(worker),
            "pytorch_ready": self._test_pytorch_import(worker),
            "disk_space_ready": self._check_disk_space_ready(worker),
            "ping_latency_ready": self._ping_latency_ready(worker),
        }
        
        results = await asyncio.gather(*checks.values(), return_exceptions=True)
        
        # Map results back to names
        check_results = dict(zip(checks.keys(), results))
        
        # Log failures
        for name, result in check_results.items():
            if isinstance(result, Exception) or result is False:
                self.logger.warning(f"Node {node_name} check {name} failed: {result}")

        # Requirement: 90% pass rate
        success_count = sum(1 for r in results if r is True)
        pass_rate = success_count / len(results)
        
        return pass_rate > 0.9

    async def test_ssh(self, worker: dict[str, Any]) -> bool:
        """Test SSH is reachable."""
        node_name = worker["name"]
        success, _, _ = await asyncio.get_event_loop().run_in_executor(
            self._executor,
            self.cluster.execute_on_worker,
            node_name, "echo 'OK'", 5
        )
        return success

    async def test_gpu(self, worker: dict[str, Any]) -> bool:
        """Test GPU is working using nvidia-smi."""
        node_name = worker["name"]
        cmd = "nvidia-smi --query-gpu=count --format=csv,noheader"
        success, stdout, _ = await asyncio.get_event_loop().run_in_executor(
            self._executor,
            self.cluster.execute_on_worker,
            node_name, cmd, 5
        )
        try:
            return success and int(stdout.strip()) > 0
        except (ValueError, AttributeError):
            return False

    async def _test_pytorch_import(self, worker: dict[str, Any]) -> bool:
        """Test if PyTorch can be imported and CUDA is available."""
        node_name = worker["name"]
        cmd = 'python3 -c "import torch; print(torch.cuda.is_available())"'
        success, stdout, _ = await asyncio.get_event_loop().run_in_executor(
            self._executor,
            self.cluster.execute_on_worker,
            node_name, cmd, 10
        )
        return success and "True" in stdout

    async def _check_disk_space_ready(self, worker: dict[str, Any], min_gb: float = 1.0) -> bool:
        """Check if there's enough disk space."""
        node_name = worker["name"]
        cmd = "df -BG / | tail -1 | awk '{print $4}' | sed 's/G//'"
        success, stdout, _ = await asyncio.get_event_loop().run_in_executor(
            self._executor,
            self.cluster.execute_on_worker,
            node_name, cmd, 5
        )
        try:
            return success and float(stdout.strip()) > min_gb
        except (ValueError, AttributeError):
            return False

    async def _ping_latency_ready(self, worker: dict[str, Any], max_ms: float = 100.0) -> bool:
        """Check if network latency is within limits."""
        node_name = worker["name"]
        # Use simple ping from local to worker
        cmd = f"ping -c 1 -W 1 {worker['ip']} | grep 'time=' | awk -F'time=' '{{print $2}}' | awk '{{print $1}}'"
        
        # Ping is local, run in executor
        def run_ping():
            try:
                result = subprocess.check_output(cmd, shell=True, text=True)
                return float(result.strip())
            except:
                return 999.0

        latency = await asyncio.get_event_loop().run_in_executor(self._executor, run_ping)
        return latency < max_ms


    async def _collect_node_metrics(self, worker: dict[str, Any]) -> HealthMetrics | None:
        """Collect comprehensive metrics from a single node.
        
        Args:
            worker: Worker configuration dictionary
            
        Returns:
            HealthMetrics object or None if collection failed
            
        Raises:
            Exception: If metric collection fails
        """
        node_name = worker["name"]
        
        try:
            # Phase 2b: Detailed health check
            is_healthy = await self.check_node_health(worker)
            
            if not is_healthy:
                self.logger.warning(f"Node {node_name} failed Phase 2b health checks")
            
            # Use existing cluster health check as baseline
            health_status = self.cluster.check_health().get(node_name, {})
            
            if not health_status.get("healthy", False) and not is_healthy:
                return None
            
            # Collect detailed metrics concurrently
            gpu_task = self._get_detailed_gpu_metrics(worker)
            system_task = self._get_system_metrics(worker)
            network_task = self._get_network_metrics(worker)
            process_task = self._get_process_metrics(worker)
            
            # Wait for all metric collections with exception handling
            gpu_metrics, system_metrics, network_metrics, process_metrics = await asyncio.gather(
                gpu_task, system_task, network_task, process_task,
                return_exceptions=True
            )
            
            # Calculate health score using safe metric extraction
            health_score = self._calculate_health_score(
                self._safe_extract_metrics(gpu_metrics),
                self._safe_extract_metrics(system_metrics),
                self._safe_extract_metrics(network_metrics),
                self._safe_extract_metrics(process_metrics)
            )
            
            # Generate alerts
            alerts = self._generate_alerts(
                node_name, health_score,
                self._safe_extract_metrics(gpu_metrics),
                self._safe_extract_metrics(system_metrics),
                self._safe_extract_metrics(process_metrics)
            )
            
            return HealthMetrics(
                node_name=node_name,
                timestamp=datetime.now(),
                gpu_utilization=self._safe_extract_metric(gpu_metrics, "utilization", 0.0),
                gpu_memory_used=self._safe_extract_metric(gpu_metrics, "memory_used", 0.0),
                gpu_memory_total=self._safe_extract_metric(gpu_metrics, "memory_total", 0.0),
                gpu_temperature=self._safe_extract_metric(gpu_metrics, "temperature", 0.0),
                gpu_power_usage=self._safe_extract_metric(gpu_metrics, "power_usage", 0.0),
                cpu_utilization=self._safe_extract_metric(system_metrics, "cpu_util", 0.0),
                memory_usage=self._safe_extract_metric(system_metrics, "memory_usage", 0.0),
                disk_usage=self._safe_extract_metric(system_metrics, "disk_usage", 0.0),
                network_latency=self._safe_extract_metric(network_metrics, "latency", 999.0),
                process_status=self._safe_extract_metrics(process_metrics),
                health_score=health_score,
                alerts=alerts
            )
            
        except Exception as e:
            self.logger.error(f"Failed to collect metrics from {node_name}: {e}")
            raise

    def _safe_extract_metrics(self, metrics_result: Any) -> dict[str, Any]:
        """Safely extract metrics from result, handling exceptions."""
        if isinstance(metrics_result, Exception):
            return {}
        return metrics_result if isinstance(metrics_result, dict) else {}

    def _safe_extract_metric(self, metrics_result: Any, key: str, default: float) -> float:
        """Safely extract a single metric value."""
        metrics = self._safe_extract_metrics(metrics_result)
        return float(metrics.get(key, default))

    async def _get_detailed_gpu_metrics(self, worker: dict[str, Any]) -> dict[str, Any]:
        """Get detailed GPU metrics using nvidia-smi."""
        node_name = worker["name"]
        
        # Optimized nvidia-smi query
        gpu_query = (
            "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,"
            "temperature.gpu,power.draw --format=csv,noheader,nounits"
        )
        
        success, stdout, stderr = await asyncio.get_event_loop().run_in_executor(
            self._executor,
            self.cluster.execute_on_worker,
            node_name, gpu_query, 15
        )
        
        if not success:
            raise Exception(f"GPU query failed: {stderr}")
        
        # Parse GPU metrics with error handling
        try:
            parts = stdout.strip().split(",")
            if len(parts) != 5:
                raise ValueError(f"Expected 5 GPU metrics, got {len(parts)}")
            
            return {
                "utilization": float(parts[0]),
                "memory_used": float(parts[1]),
                "memory_total": float(parts[2]),
                "temperature": float(parts[3]),
                "power_usage": float(parts[4])
            }
        except (ValueError, IndexError) as e:
            raise Exception(f"Failed to parse GPU metrics: {e}")

    async def _get_system_metrics(self, worker: dict[str, Any]) -> dict[str, Any]:
        """Get system metrics (CPU, memory, disk) efficiently."""
        node_name = worker["name"]
        
        # Combined system query for efficiency
        system_query = (
            "echo \"CPU:$(top -bn1 | grep 'Cpu(s)' | awk '{print $2}' | cut -d'%' -f1);"
            "MEM:$(free | grep Mem | awk '{printf \"%.1f\", $3/$2 * 100.0}');"
            "DISK:$(df -h / | tail -1 | awk '{print $5}' | cut -d'%' -f1)\""
        )
        
        success, stdout, stderr = await asyncio.get_event_loop().run_in_executor(
            self._executor,
            self.cluster.execute_on_worker,
            node_name, system_query, 10
        )
        
        if not success:
            raise Exception(f"System query failed: {stderr}")
        
        # Parse system metrics with pattern matching
        metrics = {}
        for line in stdout.strip().split(";"):
            if ":" in line:
                key, value = line.split(":", 1)
                match key:
                    case "CPU":
                        metrics["cpu_util"] = float(value)
                    case "MEM":
                        metrics["memory_usage"] = float(value)
                    case "DISK":
                        metrics["disk_usage"] = float(value)
        
        return metrics

    async def _get_network_metrics(self, worker: dict[str, Any]) -> dict[str, Any]:
        """Get network latency and connectivity metrics."""
        node_name = worker["name"]
        
        # Ping test for latency
        ping_query = f"ping -c 1 {worker['ip']} | grep 'time=' | awk -F'time=' '{{print $2}}' | awk '{{print $1}}'"
        
        success, stdout, stderr = await asyncio.get_event_loop().run_in_executor(
            self._executor,
            self.cluster.execute_on_worker,
            node_name, ping_query, 5
        )
        
        if success and stdout.strip():
            try:
                latency = float(stdout.strip())
                return {"latency": latency}
            except ValueError:
                pass
        
        return {"latency": 999.0}  # High latency indicates issues

    async def _get_process_metrics(self, worker: dict[str, Any]) -> dict[str, Any]:
        """Check training process status."""
        node_name = worker["name"]
        
        # Check for Python training processes
        process_query = (
            "ps aux | grep -E '(python|torch|cuda)' | grep -v grep | "
            "awk '{print $2, $11, $12}' | head -10"
        )
        
        success, stdout, stderr = await asyncio.get_event_loop().run_in_executor(
            self._executor,
            self.cluster.execute_on_worker,
            node_name, process_query, 5
        )
        
        process_status = {}
        if success:
            for line in stdout.strip().split("\n"):
                if line.strip():
                    parts = line.split()
                    if len(parts) >= 2:
                        pid = parts[0]
                        cmd = " ".join(parts[1:])
                        process_status[pid] = "training" in cmd.lower()
        
        return process_status

    def _calculate_health_score(
        self,
        gpu_metrics: dict[str, Any],
        system_metrics: dict[str, Any],
        network_metrics: dict[str, Any],
        process_metrics: dict[str, Any],
    ) -> HealthScore:
        """Calculate overall health score (0-100) using weighted metrics."""
        score = 100.0
        
        # GPU health (40% weight)
        gpu_temp = gpu_metrics.get("temperature", 0)
        gpu_mem_usage = (
            gpu_metrics.get("memory_used", 0) / max(gpu_metrics.get("memory_total", 1), 1)
        )
        
        match gpu_temp:
            case temp if temp > 85:
                score -= 30
            case temp if temp > 80:
                score -= 15
        
        match gpu_mem_usage:
            case usage if usage > 0.95:
                score -= 20
            case usage if usage > 0.90:
                score -= 10
        
        # System health (30% weight)
        mem_usage = system_metrics.get("memory_usage", 0)
        disk_usage = system_metrics.get("disk_usage", 0)
        
        match mem_usage:
            case usage if usage > 95:
                score -= 25
            case usage if usage > 90:
                score -= 10
        
        match disk_usage:
            case usage if usage > 95:
                score -= 20
            case usage if usage > 90:
                score -= 10
        
        # Network health (20% weight)
        latency = network_metrics.get("latency", 0)
        match latency:
            case l if l > 100:
                score -= 15
            case l if l > 50:
                score -= 5
        
        # Process health (10% weight)
        if not process_metrics and self._active_jobs:
            score -= 10  # Should have processes during active jobs
        
        return max(0, score)

    def _generate_alerts(
        self,
        node_name: str,
        health_score: HealthScore,
        gpu_metrics: dict[str, Any],
        system_metrics: dict[str, Any],
        process_metrics: dict[str, Any],
    ) -> list[AlertMessage]:
        """Generate alerts based on metrics using structural pattern matching."""
        alerts = []
        
        # Health score alerts
        match health_score:
            case score if score < self.config.critical_threshold:
                alerts.append(f"{AlertLevel.CRITICAL}: {node_name} health score {score:.1f}")
            case score if score < self.config.warning_threshold:
                alerts.append(f"{AlertLevel.WARNING}: {node_name} health score {score:.1f}")
        
        # GPU alerts
        gpu_temp = gpu_metrics.get("temperature", 0)
        match gpu_temp:
            case temp if temp > 85:
                alerts.append(f"{AlertLevel.CRITICAL}: {node_name} GPU temperature {temp}°C")
            case temp if temp > 80:
                alerts.append(f"{AlertLevel.WARNING}: {node_name} GPU temperature {temp}°C")
        
        # Memory alerts
        mem_usage = system_metrics.get("memory_usage", 0)
        match mem_usage:
            case usage if usage > 95:
                alerts.append(f"{AlertLevel.CRITICAL}: {node_name} memory usage {usage:.1f}%")
            case usage if usage > 90:
                alerts.append(f"{AlertLevel.WARNING}: {node_name} memory usage {usage:.1f}%")
        
        # Process alerts for active jobs
        if self._active_jobs and not process_metrics:
            alerts.append(
                f"{AlertLevel.WARNING}: {node_name} no training processes detected during active job"
            )
        
        return alerts

    def _handle_node_failure(self, node_name: NodeName, error: Exception) -> None:
        """Handle node failures with circuit breaker logic."""
        breaker = self._get_circuit_breaker(node_name)
        breaker.record_failure()
        
        self.logger.warning(f"Node {node_name} failure #{breaker.failure_count}: {error}")
        
        if breaker.is_open():
            self.logger.error(f"Circuit breaker opened for {node_name}")
            
            # Update node status in state manager
            try:
                self.state_manager.update_node_status(
                    node_name, 
                    status=NodeStatus.UNHEALTHY.value,
                    last_check=datetime.now().isoformat()
                )
            except KeyError:
                pass

    def _get_circuit_breaker(self, node_name: NodeName) -> CircuitBreaker:
        """Get or create circuit breaker for a node."""
        if node_name not in self._circuit_breakers:
            self._circuit_breakers[node_name] = CircuitBreaker()
        return self._circuit_breakers[node_name]

    def _reset_circuit_breaker(self, node_name: NodeName) -> None:
        """Reset circuit breaker after successful operation."""
        breaker = self._get_circuit_breaker(node_name)
        breaker.record_success()

    def _is_circuit_breaker_open(self, node_name: NodeName) -> bool:
        """Check if circuit breaker is open for a node."""
        return self._get_circuit_breaker(node_name).is_open()

    async def _process_metrics(self, metrics_list: list[HealthMetrics]) -> None:
        """Process collected metrics and update state."""
        for metrics in metrics_list:
            node_name = metrics.node_name
            
            # Store metrics history with size limit
            if node_name not in self._metrics_history:
                self._metrics_history[node_name] = []
            
            self._metrics_history[node_name].append(metrics)
            
            # Maintain history size limit
            if len(self._metrics_history[node_name]) > self.config.metrics_history_size:
                self._metrics_history[node_name] = self._metrics_history[node_name][-self.config.metrics_history_size:]
            
            # Update health score
            self._health_scores[node_name] = metrics.health_score
            
            # Update node status in state manager
            try:
                self.state_manager.update_node_status(
                    node_name,
                    status=metrics.status.value,
                    health_score=metrics.health_score,
                    last_check=metrics.timestamp.isoformat()
                )
            except KeyError:
                pass

    async def _check_critical_conditions(self, metrics_list: list[HealthMetrics]) -> None:
        """Check for critical cluster conditions and send alerts."""
        critical_alerts = []
        
        for metrics in metrics_list:
            critical_alerts.extend(metrics.critical_alerts)
            self._stats["critical_alerts"] += len(metrics.critical_alerts)
            self._stats["warning_alerts"] += len(metrics.warning_alerts)
        
        # Check for cluster-wide issues
        unhealthy_nodes = sum(1 for m in metrics_list if m.health_score < self.config.critical_threshold)
        total_nodes = len(metrics_list)
        
        if total_nodes > 0 and (unhealthy_nodes / total_nodes) > 0.5:
            critical_alerts.append(
                f"{AlertLevel.CRITICAL}: {unhealthy_nodes}/{total_nodes} nodes unhealthy"
            )
        
        # Send alerts if any
        if critical_alerts:
            await self._send_alerts(critical_alerts)

    async def _send_alerts(self, alerts: list[AlertMessage]) -> None:
        """Send alerts to configured channels."""
        for alert in alerts:
            self.logger.critical(alert)
            # TODO: Add webhook, Slack, email notifications

    def _cleanup_old_metrics(self) -> None:
        """Clean up old metrics to prevent memory leaks."""
        cutoff_time = datetime.now() - timedelta(hours=self.config.cleanup_interval_hours)
        
        for node_name in list(self._metrics_history.keys()):
            self._metrics_history[node_name] = [
                m for m in self._metrics_history[node_name] 
                if m.timestamp > cutoff_time
            ]
            
            if not self._metrics_history[node_name]:
                del self._metrics_history[node_name]

    def register_job_activity(self, job_name: str) -> None:
        """Register job activity to adjust monitoring frequency."""
        self._active_jobs.add(job_name)
        self._last_job_activity = datetime.now()
        self.logger.info(f"Job activity registered: {job_name}")

    def unregister_job_activity(self, job_name: str) -> None:
        """Unregister job activity."""
        self._active_jobs.discard(job_name)
        self._last_job_activity = datetime.now()
        self.logger.info(f"Job activity unregistered: {job_name}")

    def get_statistics(self) -> dict[str, Any]:
        """Get monitoring statistics."""
        return {
            **self._stats,
            "active_jobs": len(self._active_jobs),
            "monitored_nodes": len(self._health_scores),
            "circuit_breakers_open": sum(
                1 for cb in self._circuit_breakers.values() if cb.is_open()
            ),
            "average_health_score": (
                sum(self._health_scores.values()) / len(self._health_scores)
                if self._health_scores else 0
            ),
        }

    async def stop(self) -> None:
        """Stop the health monitor gracefully."""
        self._running = False
        self._executor.shutdown(wait=True)
        self.logger.info("Health monitor stopped")


class TrainingMetricsCollector:
    """Real-time training metrics collector for distributed training jobs.
    
    This class collects, stores, and provides access to training-specific metrics
    from distributed ranks, complementing the system health monitoring.
    """
    
    def __init__(
        self,
        cluster: Cluster,
        config: MetricsCollectorConfig | None = None,
    ) -> None:
        """Initialize the training metrics collector.
        
        Args:
            cluster: UGRO cluster instance for accessing training nodes
            config: Metrics collection configuration, defaults to sensible values
        """
        self.cluster = cluster
        self.config = config or MetricsCollectorConfig()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        self._result_aggregator = ResultAggregator()
        
        # Storage for training metrics by job_id and rank
        self._metrics_storage: dict[JobID, dict[RankID, list[TrainingMetrics]]] = {}
        
        # Circuit breakers for each node/rank
        self._circuit_breakers: dict[str, CircuitBreaker] = {}
        
        # Performance optimization
        self._executor = ThreadPoolExecutor(max_workers=8)
        
        # State tracking
        self._running = False
        self._collection_tasks: dict[JobID, asyncio.Task] = {}
        
        # Statistics
        self._stats = {
            "total_collections": 0,
            "failed_collections": 0,
            "metrics_collected": 0,
            "active_jobs": 0,
        }

    async def start_collection(
        self, 
        job_id: JobID, 
        ranks: list[RankID],
        workers: list[dict[str, Any]] | None = None,
        ssh_clients: dict[str, SSHClient] | None = None
    ) -> None:
        """Start collecting metrics for a specific training job.
        
        Args:
            job_id: Unique identifier for the training job
            ranks: List of distributed training ranks to monitor
            workers: Optional list of workers for SSH syncing
            ssh_clients: Optional SSH clients for sync
        """
        if job_id in self._collection_tasks:
            self.logger.warning(f"Metrics collection already active for job {job_id}")
            return
        
        # Initialize storage for this job
        self._metrics_storage[job_id] = {rank: [] for rank in ranks}
        
        # Start collection task
        self._collection_tasks[job_id] = asyncio.create_task(
            self._collect_metrics_loop(job_id, ranks, workers, ssh_clients),
            name=f"metrics_{job_id}"
        )
        
        self._stats["active_jobs"] += 1
        self.logger.info(f"Started metrics collection for job {job_id} with {len(ranks)} ranks")

    async def stop_collection(self, job_id: JobID) -> None:
        """Stop collecting metrics for a specific training job.
        
        Args:
            job_id: Unique identifier for the training job
        """
        if job_id not in self._collection_tasks:
            self.logger.warning(f"No active collection for job {job_id}")
            return
        
        # Cancel collection task
        task = self._collection_tasks.pop(job_id)
        task.cancel()
        
        try:
            await task
        except asyncio.CancelledError:
            pass
        
        self._stats["active_jobs"] -= 1
        self.logger.info(f"Stopped metrics collection for job {job_id}")

    async def _collect_metrics_loop(
        self, 
        job_id: JobID, 
        ranks: list[RankID],
        workers: list[dict[str, Any]] | None = None,
        ssh_clients: dict[str, SSHClient] | None = None
    ) -> NoReturn:
        """Main collection loop for a specific job.
        
        Args:
            job_id: Unique identifier for the training job
            ranks: List of distributed training ranks to monitor
            workers: Optional list of workers for SSH syncing
            ssh_clients: Optional SSH clients for sync
        """
        self._running = True
        
        while self._running:
            start_time = time.time()
            
            try:
                # Sync metrics from workers if clients available
                if workers and ssh_clients:
                    await self._result_aggregator.sync_rank_metrics(job_id, workers, ssh_clients)

                # Collect metrics from all ranks concurrently (reads synced per-rank files)
                metrics = await self._collect_from_all_ranks(job_id, ranks)
                
                # Store metrics in history and consolidate into main metrics.jsonl
                self._store_metrics(job_id, metrics)
                
                # Check for performance issues
                await self._check_performance_alerts(job_id, metrics)
                
                # Clean up old metrics periodically
                if self._stats["total_collections"] % 100 == 0:
                    self._cleanup_old_metrics(job_id)
                
                # Calculate precise sleep time
                execution_time = time.time() - start_time
                sleep_time = max(0, self.config.collection_interval - execution_time)
                
                await asyncio.sleep(sleep_time)
                
                # Update statistics
                self._stats["total_collections"] += 1
                
            except* Exception as eg:
                self.logger.error(f"Metrics collection errors for job {job_id}: {eg}")
                self._stats["failed_collections"] += 1
                await asyncio.sleep(self.config.collection_interval)

    async def _collect_from_all_ranks(self, job_id: JobID, ranks: list[RankID]) -> list[TrainingMetrics]:
        """Collect metrics from all ranks concurrently.
        
        Args:
            job_id: Unique identifier for the training job
            ranks: List of distributed training ranks to monitor
            
        Returns:
            List of training metrics from responsive ranks
        """
        # Create concurrent tasks for each rank
        tasks = {
            rank: asyncio.create_task(
                self._collect_rank_metrics(job_id, rank),
                name=f"collect_{job_id}_rank_{rank}"
            )
            for rank in ranks
            if not self._is_circuit_breaker_open(f"{job_id}_{rank}")
        }
        
        # Wait for all tasks with comprehensive error handling
        results = []
        exceptions = []
        
        for rank, task in tasks.items():
            try:
                metrics = await task
                if metrics:
                    results.append(metrics)
                    # Reset circuit breaker on success
                    self._reset_circuit_breaker(f"{job_id}_{rank}")
            except Exception as e:
                exceptions.append(e)
                self._handle_rank_failure(job_id, rank, e)
        
        # Log exceptions if any
        if exceptions:
            self.logger.warning(f"Failed to collect metrics from {len(exceptions)} ranks for job {job_id}")
        
        return results

    async def _collect_rank_metrics(self, job_id: JobID, rank: RankID) -> TrainingMetrics | None:
        """Collect metrics from a specific training rank.
        
        Args:
            job_id: Unique identifier for the training job
            rank: Distributed training rank
            
        Returns:
            TrainingMetrics object or None if collection failed
        """
        try:
            # Get worker node for this rank (assuming rank maps to node)
            workers = self.cluster.get_all_workers()
            worker = workers[rank % len(workers)]  # Simple round-robin mapping
            
            node_name = worker["name"]
            
            # Try to read metrics from file first
            metrics = await self._read_metrics_from_file(job_id, rank, node_name)
            if metrics:
                return metrics
            
            # Fallback to parsing log files
            metrics = await self._parse_metrics_from_logs(job_id, rank, node_name)
            if metrics:
                return metrics
            
            # Fallback to GPU metrics estimation
            metrics = await self._estimate_metrics_from_gpu(job_id, rank, node_name)
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to collect metrics from rank {rank}: {e}")
            raise

    async def _read_metrics_from_file(self, job_id: JobID, rank: RankID, node_name: str) -> TrainingMetrics | None:
        """Read metrics from rank-specific metrics JSONL file."""
        try:
            # We look for the local rank-specific file pulled by sync_rank_metrics
            paths = self._result_aggregator.ensure_job_layout(job_id)
            rank_file = paths.job_dir / f"metrics_rank{rank}.jsonl"
            
            if not rank_file.exists():
                return None
            
            import json
            with open(rank_file, "r", encoding="utf-8") as f:
                line = f.readline().strip()
                if not line:
                    return None
                    
                metrics_data = json.loads(line)
                
                # Safety check
                if str(metrics_data.get("job_id")) != str(job_id) or int(metrics_data.get("rank", -1)) != int(rank):
                    return None

                return TrainingMetrics(
                    timestamp=datetime.now(),
                    job_id=job_id,
                    rank=rank,
                    gpu_util=float(metrics_data.get("gpu_util", 0)),
                    gpu_mem_used_gb=float(metrics_data.get("gpu_mem_used_gb", 0)),
                    training_loss=float(metrics_data.get("training_loss", 0)),
                    throughput_tokens_sec=float(metrics_data.get("throughput_tokens_sec", 0)),
                    gradient_norm=float(metrics_data.get("gradient_norm", 0)),
                    learning_rate=float(metrics_data.get("learning_rate", 0)),
                )

        except (json.JSONDecodeError, KeyError, ValueError, OSError) as e:
            self.logger.debug(f"Failed to read rank file for rank {rank}: {e}")
            return None

    async def _parse_metrics_from_logs(self, job_id: JobID, rank: RankID, node_name: str) -> TrainingMetrics | None:
        """Parse metrics from training log files."""
        try:
            log_path = self._result_aggregator.rank_log_path(job_id, rank)
            if not log_path.exists():
                return None

            with open(log_path, "r", encoding="utf-8") as f:
                lines = f.readlines()[-50:]

            # Parse log lines for metrics (simplified pattern matching)
            metrics = {}
            for line in [line.strip() for line in lines if line.strip()]:
                if "loss:" in line.lower():
                    try:
                        # Extract loss value
                        loss_str = line.split("loss:")[-1].split()[0]
                        metrics["training_loss"] = float(loss_str)
                    except (IndexError, ValueError):
                        pass
                elif "lr:" in line.lower():
                    try:
                        # Extract learning rate
                        lr_str = line.split("lr:")[-1].split()[0]
                        metrics["learning_rate"] = float(lr_str)
                    except (IndexError, ValueError):
                        pass
            
            # If we found some metrics, estimate others
            if metrics:
                return await self._estimate_metrics_from_gpu(job_id, rank, node_name, metrics)
            
            return None
            
        except Exception as e:
            self.logger.debug(f"Failed to parse logs for rank {rank}: {e}")
            return None

    async def _estimate_metrics_from_gpu(self, job_id: JobID, rank: RankID, node_name: str, base_metrics: dict[str, float] | None = None) -> TrainingMetrics | None:
        """Estimate metrics from GPU utilization."""
        try:
            # Get GPU metrics
            gpu_query = "nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits"
            
            success, stdout, stderr = await asyncio.get_event_loop().run_in_executor(
                self._executor,
                self.cluster.execute_on_worker,
                node_name, gpu_query, self.config.collection_timeout
            )
            
            if not success:
                return None
            
            # Parse GPU metrics
            parts = stdout.strip().split(",")
            if len(parts) >= 2:
                gpu_util = float(parts[0])
                gpu_mem_used = float(parts[1]) / 1024  # Convert MB to GB
                
                # Use provided metrics or defaults
                metrics = base_metrics or {}
                
                return TrainingMetrics(
                    timestamp=datetime.now(),
                    job_id=job_id,
                    rank=rank,
                    gpu_util=gpu_util,
                    gpu_mem_used_gb=gpu_mem_used,
                    training_loss=metrics.get("training_loss", 1.0),
                    throughput_tokens_sec=metrics.get("throughput_tokens_sec", 100.0),
                    gradient_norm=metrics.get("gradient_norm", 1.0),
                    learning_rate=metrics.get("learning_rate", 0.001)
                )
            
            return None
            
        except Exception as e:
            self.logger.debug(f"Failed to estimate GPU metrics for rank {rank}: {e}")
            return None

    def _store_metrics(self, job_id: JobID, metrics_list: list[TrainingMetrics]) -> None:
        """Store collected metrics with history limits."""
        if job_id not in self._metrics_storage:
            return
        
        for metrics in metrics_list:
            rank = metrics.rank
            if rank not in self._metrics_storage[job_id]:
                continue
            
            # Add metrics to history
            self._metrics_storage[job_id][rank].append(metrics)
            
            # Maintain history size limit
            if len(self._metrics_storage[job_id][rank]) > self.config.max_history_size:
                self._metrics_storage[job_id][rank] = self._metrics_storage[job_id][rank][-self.config.max_history_size:]
            
            # Update statistics
            self._stats["metrics_collected"] += 1

        # Batch consolidate into central metrics.jsonl
        if metrics_list:
            payloads = [
                {
                    "timestamp": m.timestamp.isoformat(),
                    "job_id": m.job_id,
                    "rank": m.rank,
                    "gpu_util": m.gpu_util,
                    "gpu_mem_used_gb": m.gpu_mem_used_gb,
                    "training_loss": m.training_loss,
                    "throughput_tokens_sec": m.throughput_tokens_sec,
                    "gradient_norm": m.gradient_norm,
                    "learning_rate": m.learning_rate,
                }
                for m in metrics_list
            ]
            self._result_aggregator._consolidate_rank_metrics(job_id, payloads)

    async def _check_performance_alerts(self, job_id: JobID, metrics_list: list[TrainingMetrics]) -> None:
        """Check for performance issues and generate alerts."""
        for metrics in metrics_list:
            alerts = []
            
            # Check throughput
            if metrics.throughput_tokens_sec < self.config.low_throughput_threshold:
                alerts.append(f"WARNING: Low throughput {metrics.throughput_tokens_sec:.1f} tokens/sec")
            
            # Check loss
            if metrics.training_loss > self.config.high_loss_threshold:
                alerts.append(f"WARNING: High training loss {metrics.training_loss:.3f}")
            
            # Check GPU utilization
            if metrics.gpu_util < self.config.low_gpu_util_threshold:
                alerts.append(f"WARNING: Low GPU utilization {metrics.gpu_util:.1f}%")
            
            # Log alerts
            for alert in alerts:
                self.logger.warning(f"Job {job_id} Rank {metrics.rank}: {alert}")

    def _cleanup_old_metrics(self, job_id: JobID) -> None:
        """Clean up old metrics to prevent memory leaks."""
        if job_id not in self._metrics_storage:
            return
        
        cutoff_time = datetime.now() - timedelta(hours=self.config.cleanup_interval_hours)
        
        for rank in list(self._metrics_storage[job_id].keys()):
            # Filter old metrics
            self._metrics_storage[job_id][rank] = [
                m for m in self._metrics_storage[job_id][rank] 
                if m.timestamp > cutoff_time
            ]
            
            # Remove empty rank histories
            if not self._metrics_storage[job_id][rank]:
                del self._metrics_storage[job_id][rank]

    def _handle_rank_failure(self, job_id: JobID, rank: RankID, error: Exception) -> None:
        """Handle rank failures with circuit breaker logic."""
        breaker_key = f"{job_id}_{rank}"
        breaker = self._get_circuit_breaker(breaker_key)
        breaker.record_failure()
        
        self.logger.warning(f"Job {job_id} Rank {rank} failure #{breaker.failure_count}: {error}")

    def _get_circuit_breaker(self, key: str) -> CircuitBreaker:
        """Get or create circuit breaker for a rank."""
        if key not in self._circuit_breakers:
            self._circuit_breakers[key] = CircuitBreaker()
        return self._circuit_breakers[key]

    def _reset_circuit_breaker(self, key: str) -> None:
        """Reset circuit breaker after successful operation."""
        breaker = self._get_circuit_breaker(key)
        breaker.record_success()

    def _is_circuit_breaker_open(self, key: str) -> bool:
        """Check if circuit breaker is open for a rank."""
        return self._get_circuit_breaker(key).is_open()

    def get_latest_metrics(self, job_id: JobID, rank: RankID | None = None) -> TrainingMetrics | list[TrainingMetrics] | None:
        """Get latest metrics for a job and optional rank.
        
        Args:
            job_id: Unique identifier for the training job
            rank: Optional specific rank, if None returns metrics for all ranks
            
        Returns:
            Latest TrainingMetrics or list of metrics for all ranks
        """
        if job_id not in self._metrics_storage:
            return None
        
        if rank is not None:
            # Return latest metrics for specific rank
            rank_metrics = self._metrics_storage[job_id].get(rank, [])
            return rank_metrics[-1] if rank_metrics else None
        else:
            # Return latest metrics for all ranks
            latest_metrics = []
            for rank, metrics_list in self._metrics_storage[job_id].items():
                if metrics_list:
                    latest_metrics.append(metrics_list[-1])
            return latest_metrics

    def get_metrics_history(self, job_id: JobID, rank: RankID, limit: int | None = None) -> list[TrainingMetrics]:
        """Get historical metrics for a specific rank.
        
        Args:
            job_id: Unique identifier for the training job
            rank: Distributed training rank
            limit: Optional limit on number of metrics returned
            
        Returns:
            List of historical TrainingMetrics
        """
        if job_id not in self._metrics_storage or rank not in self._metrics_storage[job_id]:
            return []
        
        metrics = self._metrics_storage[job_id][rank]
        return metrics[-limit:] if limit else metrics

    def get_job_statistics(self, job_id: JobID) -> dict[str, Any]:
        """Get statistics for a specific job.
        
        Args:
            job_id: Unique identifier for the training job
            
        Returns:
            Dictionary with job statistics
        """
        if job_id not in self._metrics_storage:
            return {"status": "not_found"}
        
        total_metrics = sum(len(metrics) for metrics in self._metrics_storage[job_id].values())
        ranks = list(self._metrics_storage[job_id].keys())
        
        # Calculate averages
        all_metrics = []
        for rank_metrics in self._metrics_storage[job_id].values():
            all_metrics.extend(rank_metrics)
        
        if not all_metrics:
            return {"status": "no_metrics", "ranks": ranks}
        
        avg_gpu_util = sum(m.gpu_util for m in all_metrics) / len(all_metrics)
        avg_throughput = sum(m.throughput_tokens_sec for m in all_metrics) / len(all_metrics)
        avg_loss = sum(m.training_loss for m in all_metrics) / len(all_metrics)
        avg_efficiency = sum(m.efficiency_score for m in all_metrics) / len(all_metrics)
        
        return {
            "status": "active",
            "ranks": len(ranks),
            "total_metrics": total_metrics,
            "avg_gpu_util": avg_gpu_util,
            "avg_throughput": avg_throughput,
            "avg_loss": avg_loss,
            "avg_efficiency": avg_efficiency,
            "latest_timestamp": max(m.timestamp for m in all_metrics).isoformat()
        }

    def get_statistics(self) -> dict[str, Any]:
        """Get collector statistics."""
        return {
            **self._stats,
            "active_jobs": len(self._collection_tasks),
            "circuit_breakers_open": sum(
                1 for cb in self._circuit_breakers.values() if cb.is_open()
            ),
            "stored_jobs": len(self._metrics_storage),
        }

    async def stop(self) -> None:
        """Stop the metrics collector gracefully."""
        self._running = False
        
        # Cancel all collection tasks
        for job_id, task in list(self._collection_tasks.items()):
            task.cancel()
        
        # Wait for all tasks to complete
        await asyncio.gather(*self._collection_tasks.values(), return_exceptions=True)
        
        self._collection_tasks.clear()
        self._executor.shutdown(wait=True)
        self.logger.info("Training metrics collector stopped")


# Factory function for easy instantiation
def create_health_monitor(
    cluster: Cluster,
    state_manager: ClusterStateManager,
    config: MonitoringConfig | None = None,
) -> AdaptiveHealthMonitor:
    """Factory function to create health monitor with dependency injection."""
    return AdaptiveHealthMonitor(cluster, state_manager, config)


def create_metrics_collector(
    cluster: Cluster,
    config: MetricsCollectorConfig | None = None,
) -> TrainingMetricsCollector:
    """Factory function to create training metrics collector with dependency injection."""
    return TrainingMetricsCollector(cluster, config)
