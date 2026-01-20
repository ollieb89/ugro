"""
Tests for Redis Client Factory.

Tests cover:
- Mode detection (standalone, sentinel, cluster)
- Connection pooling
- Retry decorator
- Config validation
"""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from ugro.queues.redis_client import (
    RedisClientFactory,
    RedisConfig,
    RedisMode,
    SentinelNode,
    ClusterNode,
    with_retry,
    REDIS_AVAILABLE,
)


# Skip all tests if redis not available
pytestmark = pytest.mark.skipif(not REDIS_AVAILABLE, reason="redis-py not installed")


class TestRedisConfig:
    """Tests for RedisConfig model."""
    
    def test_default_mode_is_standalone(self):
        config = RedisConfig()
        assert config.mode == RedisMode.STANDALONE
        
    def test_standalone_config(self):
        config = RedisConfig(
            mode=RedisMode.STANDALONE,
            host="redis.local",
            port=6380,
            db=1,
            password="secret"
        )
        assert config.host == "redis.local"
        assert config.port == 6380
        assert config.db == 1
        assert config.password == "secret"
        
    def test_sentinel_config(self):
        config = RedisConfig(
            mode=RedisMode.SENTINEL,
            sentinels=[
                SentinelNode(host="sentinel1", port=26379),
                SentinelNode(host="sentinel2", port=26380),
            ],
            master_name="mymaster",
            password="secret"
        )
        assert config.mode == RedisMode.SENTINEL
        assert len(config.sentinels) == 2
        assert config.master_name == "mymaster"
        
    def test_cluster_config(self):
        config = RedisConfig(
            mode=RedisMode.CLUSTER,
            startup_nodes=[
                ClusterNode(host="node1", port=7000),
                ClusterNode(host="node2", port=7001),
            ],
            password="secret"
        )
        assert config.mode == RedisMode.CLUSTER
        assert len(config.startup_nodes) == 2
        
    def test_pool_settings(self):
        config = RedisConfig(
            pool_size=20,
            socket_timeout=10.0,
            retry_attempts=5
        )
        assert config.pool_size == 20
        assert config.socket_timeout == 10.0
        assert config.retry_attempts == 5


class TestRedisClientFactory:
    """Tests for RedisClientFactory."""
    
    @patch("ugro.queues.redis_client.redis")
    def test_standalone_creates_redis_client(self, mock_redis):
        """Test standalone mode creates Redis client with connection pool."""
        mock_pool = MagicMock()
        mock_redis.ConnectionPool.return_value = mock_pool
        mock_client = MagicMock()
        mock_redis.Redis.return_value = mock_client
        
        config = RedisConfig(
            mode=RedisMode.STANDALONE,
            host="localhost",
            port=6379,
            db=0
        )
        
        client = RedisClientFactory.create(config)
        
        # Verify connection pool was created
        mock_redis.ConnectionPool.assert_called_once()
        call_kwargs = mock_redis.ConnectionPool.call_args.kwargs
        assert call_kwargs["host"] == "localhost"
        assert call_kwargs["port"] == 6379
        assert call_kwargs["db"] == 0
        assert call_kwargs["decode_responses"] is True
        
        # Verify client was created with pool
        mock_redis.Redis.assert_called_once_with(connection_pool=mock_pool)
        
    @patch("ugro.queues.redis_client.Sentinel")
    def test_sentinel_creates_sentinel_client(self, mock_sentinel_class):
        """Test sentinel mode creates Sentinel-managed client."""
        mock_sentinel = MagicMock()
        mock_sentinel_class.return_value = mock_sentinel
        mock_master = MagicMock()
        mock_sentinel.master_for.return_value = mock_master
        
        config = RedisConfig(
            mode=RedisMode.SENTINEL,
            sentinels=[
                SentinelNode(host="sentinel1", port=26379),
                SentinelNode(host="sentinel2", port=26380),
            ],
            master_name="mymaster",
            password="secret"
        )
        
        client = RedisClientFactory.create(config)
        
        # Verify Sentinel was created with correct nodes
        mock_sentinel_class.assert_called_once()
        call_kwargs = mock_sentinel_class.call_args.kwargs
        assert ("sentinel1", 26379) in call_kwargs["sentinels"]
        assert ("sentinel2", 26380) in call_kwargs["sentinels"]
        
        # Verify master_for was called
        mock_sentinel.master_for.assert_called_once_with(
            "mymaster",
            socket_timeout=config.socket_timeout,
            password="secret",
            db=0,
            decode_responses=True,
        )
        
    @patch("ugro.queues.redis_client.RedisCluster")
    @patch("ugro.queues.redis_client.redis")
    def test_cluster_creates_cluster_client(self, mock_redis, mock_cluster_class):
        """Test cluster mode creates RedisCluster client."""
        mock_cluster = MagicMock()
        mock_cluster_class.return_value = mock_cluster
        
        config = RedisConfig(
            mode=RedisMode.CLUSTER,
            startup_nodes=[
                ClusterNode(host="node1", port=7000),
                ClusterNode(host="node2", port=7001),
            ],
            password="secret"
        )
        
        client = RedisClientFactory.create(config)
        
        # Verify RedisCluster was called
        mock_cluster_class.assert_called_once()
        call_kwargs = mock_cluster_class.call_args.kwargs
        assert call_kwargs["password"] == "secret"
        assert call_kwargs["decode_responses"] is True
        
    def test_sentinel_requires_nodes(self):
        """Test sentinel mode validation requires sentinel nodes."""
        config = RedisConfig(
            mode=RedisMode.SENTINEL,
            sentinels=[],  # Empty
            master_name="mymaster"
        )
        
        with pytest.raises(ValueError, match="sentinel"):
            RedisClientFactory.create(config)
            
    def test_cluster_requires_nodes(self):
        """Test cluster mode validation requires startup nodes."""
        config = RedisConfig(
            mode=RedisMode.CLUSTER,
            startup_nodes=[]  # Empty
        )
        
        with pytest.raises(ValueError, match="startup node"):
            RedisClientFactory.create(config)


class TestWithRetry:
    """Tests for retry decorator."""
    
    def test_successful_call_returns_immediately(self):
        """Test that successful calls don't retry."""
        call_count = 0
        
        @with_retry(attempts=3, base_delay=0.01)
        def success_fn():
            nonlocal call_count
            call_count += 1
            return "success"
        
        result = success_fn()
        assert result == "success"
        assert call_count == 1
        
    def test_retry_on_connection_error(self):
        """Test that connection errors trigger retry."""
        import redis.exceptions
        
        call_count = 0
        
        @with_retry(attempts=3, base_delay=0.01, exceptions=(redis.exceptions.ConnectionError,))
        def flaky_fn():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise redis.exceptions.ConnectionError("Connection failed")
            return "success"
        
        result = flaky_fn()
        assert result == "success"
        assert call_count == 3
        
    def test_raises_after_max_attempts(self):
        """Test that error is raised after max attempts."""
        import redis.exceptions
        
        @with_retry(attempts=2, base_delay=0.01, exceptions=(redis.exceptions.ConnectionError,))
        def always_fails():
            raise redis.exceptions.ConnectionError("Always fails")
        
        with pytest.raises(redis.exceptions.ConnectionError):
            always_fails()


class TestLegacyCompatibility:
    """Tests for backward compatibility with legacy config."""
    
    @patch("ugro.queues.redis_client.redis")
    def test_create_from_queue_config_legacy(self, mock_redis):
        """Test factory can create from legacy QueueConfig style object."""
        mock_pool = MagicMock()
        mock_redis.ConnectionPool.return_value = mock_pool
        mock_client = MagicMock()
        mock_redis.Redis.return_value = mock_client
        
        # Simulate legacy QueueConfig
        class LegacyQueueConfig:
            redis_host = "legacy-host"
            redis_port = 6380
            redis_db = 1
            redis_password = "legacy-pass"
        
        client = RedisClientFactory.create_from_queue_config(LegacyQueueConfig())
        
        call_kwargs = mock_redis.ConnectionPool.call_args.kwargs
        assert call_kwargs["host"] == "legacy-host"
        assert call_kwargs["port"] == 6380
        assert call_kwargs["db"] == 1
        assert call_kwargs["password"] == "legacy-pass"
