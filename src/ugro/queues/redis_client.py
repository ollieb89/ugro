"""
Redis Client Factory for UGRO.

Provides abstracted Redis client creation supporting:
- Standalone: Single Redis node
- Sentinel: Master-slave with automatic failover  
- Cluster: Sharded across multiple masters

Usage:
    from ugro.queues.redis_client import RedisClientFactory, RedisConfig, RedisMode
    
    config = RedisConfig(mode=RedisMode.SENTINEL, sentinels=[...])
    client = RedisClientFactory.create(config)
"""

import logging
import time
from enum import Enum
from functools import wraps
from typing import Any, Callable, List, Optional, TypeVar, Union

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

try:
    import redis
    from redis.sentinel import Sentinel
    from redis.cluster import RedisCluster
    from redis.exceptions import ConnectionError, TimeoutError, RedisError
    REDIS_AVAILABLE = True
except ImportError:
    redis = None  # type: ignore
    Sentinel = None  # type: ignore
    RedisCluster = None  # type: ignore
    REDIS_AVAILABLE = False


class RedisMode(str, Enum):
    """Redis deployment mode."""
    STANDALONE = "standalone"
    SENTINEL = "sentinel"
    CLUSTER = "cluster"


class SentinelNode(BaseModel):
    """Sentinel node configuration."""
    host: str
    port: int = 26379


class ClusterNode(BaseModel):
    """Cluster startup node configuration."""
    host: str
    port: int = 6379


class RedisConfig(BaseModel):
    """Redis connection configuration."""
    
    mode: RedisMode = Field(default=RedisMode.STANDALONE, description="Deployment mode")
    
    # Standalone / common settings
    host: str = Field(default="localhost", description="Redis host for standalone mode")
    port: int = Field(default=6379, description="Redis port for standalone mode")
    db: int = Field(default=0, description="Database number (standalone/sentinel only)")
    password: Optional[str] = Field(default=None, description="Redis password")
    
    # Sentinel settings
    sentinels: List[SentinelNode] = Field(default_factory=list, description="Sentinel nodes")
    master_name: str = Field(default="mymaster", description="Sentinel master name")
    
    # Cluster settings
    startup_nodes: List[ClusterNode] = Field(default_factory=list, description="Cluster startup nodes")
    
    # Connection pool settings
    pool_size: int = Field(default=10, description="Connection pool max size")
    socket_timeout: float = Field(default=5.0, description="Socket timeout in seconds")
    socket_connect_timeout: float = Field(default=2.0, description="Connection timeout")
    
    # Retry settings
    retry_attempts: int = Field(default=3, description="Number of retry attempts")
    retry_base_delay: float = Field(default=0.1, description="Base delay for exponential backoff")
    retry_max_delay: float = Field(default=2.0, description="Maximum retry delay")


F = TypeVar("F", bound=Callable[..., Any])


def with_retry(
    attempts: int = 3,
    base_delay: float = 0.1,
    max_delay: float = 2.0,
    exceptions: tuple = (ConnectionError, TimeoutError),
) -> Callable[[F], F]:
    """
    Decorator for retrying Redis operations with exponential backoff.
    
    Args:
        attempts: Maximum retry attempts
        base_delay: Initial delay between retries (seconds)
        max_delay: Maximum delay cap (seconds)
        exceptions: Tuple of exceptions to catch and retry
    """
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception = None
            for attempt in range(attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < attempts - 1:
                        delay = min(base_delay * (2 ** attempt), max_delay)
                        logger.warning(
                            f"Redis operation failed (attempt {attempt + 1}/{attempts}): {e}. "
                            f"Retrying in {delay:.2f}s..."
                        )
                        time.sleep(delay)
                    else:
                        logger.error(f"Redis operation failed after {attempts} attempts: {e}")
            raise last_exception  # type: ignore
        return wrapper  # type: ignore
    return decorator


class RedisClientFactory:
    """
    Factory for creating Redis clients based on configuration.
    
    Supports standalone, sentinel, and cluster modes with connection pooling.
    """
    
    @classmethod
    def create(cls, config: RedisConfig) -> "redis.Redis":
        """
        Create a Redis client based on the provided configuration.
        
        Args:
            config: RedisConfig instance with connection parameters
            
        Returns:
            Redis client (Redis, SentinelManagedConnection, or RedisCluster)
            
        Raises:
            ImportError: If redis-py is not installed
            ValueError: If configuration is invalid for the selected mode
        """
        if not REDIS_AVAILABLE:
            raise ImportError(
                "redis-py is required for Redis queue. Install with: pip install redis"
            )
        
        mode = config.mode
        logger.info(f"Creating Redis client in {mode.value} mode")
        
        if mode == RedisMode.STANDALONE:
            return cls._create_standalone(config)
        elif mode == RedisMode.SENTINEL:
            return cls._create_sentinel(config)
        elif mode == RedisMode.CLUSTER:
            return cls._create_cluster(config)
        else:
            raise ValueError(f"Unsupported Redis mode: {mode}")
    
    @classmethod
    def _create_standalone(cls, config: RedisConfig) -> "redis.Redis":
        """Create standalone Redis client with connection pool."""
        pool = redis.ConnectionPool(
            host=config.host,
            port=config.port,
            db=config.db,
            password=config.password,
            max_connections=config.pool_size,
            socket_timeout=config.socket_timeout,
            socket_connect_timeout=config.socket_connect_timeout,
            decode_responses=True,
        )
        
        client = redis.Redis(connection_pool=pool)
        
        # Test connection
        try:
            client.ping()
            logger.info(f"Connected to Redis at {config.host}:{config.port}")
        except redis.ConnectionError as e:
            logger.warning(f"Could not connect to Redis: {e}")
            # Don't fail here - let caller handle connection issues
        
        return client
    
    @classmethod
    def _create_sentinel(cls, config: RedisConfig) -> "redis.Redis":
        """Create Sentinel-managed Redis client."""
        if not config.sentinels:
            raise ValueError("Sentinel mode requires at least one sentinel node")
        
        sentinel_nodes = [(s.host, s.port) for s in config.sentinels]
        
        sentinel = Sentinel(
            sentinels=sentinel_nodes,
            password=config.password,
            socket_timeout=config.socket_timeout,
        )
        
        # Get master connection - this returns a proxy that auto-discovers master
        client = sentinel.master_for(
            config.master_name,
            socket_timeout=config.socket_timeout,
            password=config.password,
            db=config.db,
            decode_responses=True,
        )
        
        logger.info(f"Connected to Redis Sentinel master '{config.master_name}'")
        return client
    
    @classmethod
    def _create_cluster(cls, config: RedisConfig) -> "RedisCluster":
        """Create Redis Cluster client."""
        if not config.startup_nodes:
            raise ValueError("Cluster mode requires at least one startup node")
        
        startup_nodes = [
            redis.cluster.ClusterNode(host=n.host, port=n.port) 
            for n in config.startup_nodes
        ]
        
        client = RedisCluster(
            startup_nodes=startup_nodes,
            password=config.password,
            socket_timeout=config.socket_timeout,
            socket_connect_timeout=config.socket_connect_timeout,
            decode_responses=True,
            # Cluster-specific options
            skip_full_coverage_check=True,  # Allow partial cluster
        )
        
        logger.info(f"Connected to Redis Cluster with {len(startup_nodes)} startup nodes")
        return client
    
    @classmethod
    def create_from_queue_config(cls, queue_config: Any) -> "redis.Redis":
        """
        Create client from legacy QueueConfig for backward compatibility.
        
        Converts old-style QueueConfig (flat params) to RedisConfig.
        """
        # Check if it's already a RedisConfig
        if isinstance(queue_config, RedisConfig):
            return cls.create(queue_config)
        
        # Check if it has nested redis config
        if hasattr(queue_config, 'redis') and isinstance(queue_config.redis, RedisConfig):
            return cls.create(queue_config.redis)
        
        # Legacy flat config - assume standalone
        config = RedisConfig(
            mode=RedisMode.STANDALONE,
            host=getattr(queue_config, 'redis_host', 'localhost'),
            port=getattr(queue_config, 'redis_port', 6379),
            db=getattr(queue_config, 'redis_db', 0),
            password=getattr(queue_config, 'redis_password', None),
        )
        return cls.create(config)


# Convenience function for simple usage
def create_redis_client(
    mode: RedisMode = RedisMode.STANDALONE,
    host: str = "localhost",
    port: int = 6379,
    password: Optional[str] = None,
    **kwargs: Any,
) -> "redis.Redis":
    """
    Convenience function to create a Redis client.
    
    Args:
        mode: Redis deployment mode
        host: Redis host (for standalone)
        port: Redis port (for standalone)
        password: Redis password
        **kwargs: Additional RedisConfig parameters
        
    Returns:
        Configured Redis client
    """
    config = RedisConfig(
        mode=mode,
        host=host,
        port=port,
        password=password,
        **kwargs,
    )
    return RedisClientFactory.create(config)
