import json
import uuid
import logging
from datetime import datetime
from typing import List, Optional, Tuple, Any, Union
from .models import Job, JobStatus, JobPriority
from .base import JobQueue
from .redis_client import (
    RedisClientFactory, 
    RedisConfig, 
    RedisMode,
    with_retry,
    REDIS_AVAILABLE,
)

logger = logging.getLogger(__name__)

# Backward compat: try direct import for type checking
try:
    import redis
except ImportError:
    redis = None


class RedisJobQueue(JobQueue):
    """
    Redis-backed Job Queue for distributed scenarios.
    
    Supports Standalone, Sentinel, and Cluster modes via RedisClientFactory.
    
    Structure (Standalone/Sentinel):
    - ugro:job:{id} -> Hash of job data
    - ugro:queue:pending -> Sorted Set (score=priority, member=job_id)
    - ugro:queue:running -> Set of job_ids
    
    Structure (Cluster mode - uses hash tags for slot colocation):
    - ugro:{job_id}:data -> Hash of job data  
    - ugro:queue:pending -> Sorted Set
    - ugro:queue:running -> Set
    """
    
    def __init__(
        self, 
        host: str = "localhost", 
        port: int = 6379, 
        db: int = 0, 
        password: Optional[str] = None,
        config: Optional[RedisConfig] = None,
    ):
        """
        Initialize Redis job queue.
        
        Args:
            host: Redis host (legacy, used if config not provided)
            port: Redis port (legacy)
            db: Redis database number (legacy)
            password: Redis password (legacy)
            config: RedisConfig object (preferred, takes precedence)
        """
        if not REDIS_AVAILABLE:
            raise ImportError("redis-py is required for RedisJobQueue. Install with `pip install redis`.")
        
        # Use config if provided, otherwise create from legacy params
        if config is not None:
            self._config = config
        else:
            self._config = RedisConfig(
                mode=RedisMode.STANDALONE,
                host=host,
                port=port,
                db=db,
                password=password,
            )
        
        self.client = RedisClientFactory.create(self._config)
        self.prefix = "ugro:"
        self._is_cluster = self._config.mode == RedisMode.CLUSTER
        
        logger.info(f"RedisJobQueue initialized in {self._config.mode.value} mode")

    def _key(self, key: str) -> str:
        """Generate key with appropriate format for cluster/non-cluster."""
        return f"{self.prefix}{key}"
    
    def _job_key(self, job_id: str) -> str:
        """
        Generate job-specific key with hash tag for cluster mode.
        
        In cluster mode, uses hash tags to colocate all job data on same slot.
        """
        if self._is_cluster:
            # Hash tag: {job_id} ensures all keys with same job_id go to same slot
            return f"{self.prefix}{{{job_id}}}:data"
        return f"{self.prefix}job:{job_id}"


    def submit(self, job: Job) -> str:
        pipeline = self.client.pipeline()

        priority = int(job.priority)
        
        # 1. Store job details (using _job_key for cluster compatibility)
        pipeline.hset(self._job_key(job.id), mapping={
            "data": job.model_dump_json(),
            "status": job.status.value,
            "priority": priority,
            "created_at": job.created_at.isoformat()
        })
        
        # 2. Add to pending queue (Sorted Set)
        # Score is priority. ZREVRANGE gives highest first.
        # Redis doesn't guarantee FIFO for same score, but usually insertion order.
        pipeline.zadd(self._key("queue:pending"), {job.id: priority})
        
        pipeline.execute()
        return job.id

    def next(self) -> Optional[Job]:
        """
        Get and claim next job atomically.
        
        Uses Lua script for standalone/sentinel modes.
        Uses two-phase approach for cluster mode (cross-slot keys can't use Lua).
        """
        if self._is_cluster:
            return self._next_cluster_safe()
        return self._next_lua()
    
    def _next_cluster_safe(self) -> Optional[Job]:
        """Cluster-safe next() using two-phase approach (not fully atomic)."""
        # Phase 1: Get highest priority job ID
        jobs = self.client.zrevrange(self._key("queue:pending"), 0, 0)
        if not jobs:
            return None
        
        job_id = jobs[0]
        
        # Phase 2: Move from pending to running (separate operations)
        # Use pipeline for efficiency but not atomicity
        pipeline = self.client.pipeline()
        pipeline.zrem(self._key("queue:pending"), job_id)
        pipeline.sadd(self._key("queue:running"), job_id)
        results = pipeline.execute()
        
        # Check if we successfully removed from pending (race condition check)
        if results[0] == 0:  # zrem returns 0 if key wasn't there
            # Someone else took it, retry
            return self.next()
        
        # Get and update job
        job = self.get_job(job_id)
        if not job:
            return None
            
        job.status = JobStatus.RUNNING
        job.started_at = datetime.now()
        self.update_job(job)
        
        return job
    
    def _next_lua(self) -> Optional[Job]:
        """Atomic next() using Lua script for standalone/sentinel modes."""
        lua_pop = """
        local pending_key = KEYS[1]
        local running_key = KEYS[2]
        local job_prefix = KEYS[3]
        
        -- Get highest priority element
        local jobs = redis.call('ZREVRANGE', pending_key, 0, 0)
        if #jobs == 0 then
            return nil
        end
        local job_id = jobs[1]
        
        -- Move to running
        redis.call('ZREM', pending_key, job_id)
        redis.call('SADD', running_key, job_id)
        
        -- Get job data from hash
        local job_key = job_prefix .. job_id
        local job_data_str = redis.call('HGET', job_key, 'data')
        
        return {job_id, job_data_str}
        """
        
        cmd = self.client.register_script(lua_pop)
        res = cmd(keys=[self._key("queue:pending"), self._key("queue:running"), self._key("job:")])
        
        if not res:
            return None
            
        job_id, job_data_str = res
        job = Job.model_validate_json(job_data_str)
        
        # Update Python object and save back to Redis
        job.status = JobStatus.RUNNING
        job.started_at = datetime.now()
        self.update_job(job)
        
        return job

    def claim(self, job_id: str) -> Optional[Job]:
        """
        Claim a specific job by ID.
        
        Uses Lua for standalone/sentinel, two-phase for cluster.
        """
        if self._is_cluster:
            return self._claim_cluster_safe(job_id)
        return self._claim_lua(job_id)
    
    def _claim_cluster_safe(self, job_id: str) -> Optional[Job]:
        """Cluster-safe claim using two-phase approach."""
        # Check if in pending
        score = self.client.zscore(self._key("queue:pending"), job_id)
        if score is None:
            return None
        
        # Move from pending to running
        pipeline = self.client.pipeline()
        pipeline.zrem(self._key("queue:pending"), job_id)
        pipeline.sadd(self._key("queue:running"), job_id)
        results = pipeline.execute()
        
        if results[0] == 0:  # Already taken
            return None
        
        job = self.get_job(job_id)
        if not job:
            return None
            
        job.status = JobStatus.RUNNING
        job.started_at = datetime.now()
        self.update_job(job)
        
        return job
    
    def _claim_lua(self, job_id: str) -> Optional[Job]:
        """Atomic claim using Lua script for standalone/sentinel."""
        lua_claim = """
        local pending_key = KEYS[1]
        local running_key = KEYS[2]
        local job_prefix = KEYS[3]
        local job_id = ARGV[1]
        
        -- Check if job is in pending
        local score = redis.call('ZSCORE', pending_key, job_id)
        if not score then
            return nil
        end
        
        -- Move to running
        redis.call('ZREM', pending_key, job_id)
        redis.call('SADD', running_key, job_id)
        
        -- Get job data
        local job_key = job_prefix .. job_id
        local job_data_str = redis.call('HGET', job_key, 'data')
        
        return job_data_str
        """
        
        cmd = self.client.register_script(lua_claim)
        res = cmd(
            keys=[self._key("queue:pending"), self._key("queue:running"), self._key("job:")],
            args=[job_id]
        )
        
        if not res:
            return None
            
        job = Job.model_validate_json(res)
        job.status = JobStatus.RUNNING
        job.started_at = datetime.now()
        self.update_job(job)
        
        return job

    def peek(self) -> Optional[Job]:
        jobs = self.client.zrevrange(self._key("queue:pending"), 0, 0)
        if not jobs:
            return None
        job_id = jobs[0]
        return self.get_job(job_id)

    def get_job(self, job_id: str) -> Optional[Job]:
        data = self.client.hget(self._job_key(job_id), "data")
        if data:
            return Job.model_validate_json(data)
        return None

    def list_jobs(self, status: Optional[JobStatus] = None, limit: int = 100) -> List[Job]:
        # This is expensive in Redis ("keys *" is bad). 
        # Better to maintain sets per status if listing is frequent.
        # PROD: maintain lists/sets per status.
        # MVP: Scan jobs.
        
        # Scan all jobs or use status index?
        # Let's assume for MVP we iterate our tracking sets.
        
        found_jobs = []
        if status == JobStatus.PENDING:
            ids = self.client.zrevrange(self._key("queue:pending"), 0, limit-1)
        elif status == JobStatus.RUNNING:
            ids = list(self.client.smembers(self._key("queue:running")))[:limit]
        else:
            # For COMPLETED/FAILED/CANCELLED we don't track in sets in this MVP ZSET approach?
            # Creating sets for them is good practice.
            # Fallback: Scan keys "ugro:job:*"
            cursor = '0'
            while cursor != 0:
                cursor, keys = self.client.scan(cursor=cursor, match=self._key("job:*"), count=50)
                for key in keys:
                    # key is full key "ugro:job:UUID"
                    job_id = key.split(":")[-1]
                    job = self.get_job(job_id)
                    if job:
                        if status is None or job.status == status:
                            found_jobs.append(job)
                        if len(found_jobs) >= limit:
                            return found_jobs
            return found_jobs

        # For known sets
        for job_id in ids:
            job = self.get_job(job_id)
            if job:
                found_jobs.append(job)
        return found_jobs

    def update_job(self, job: Job) -> None:
        priority = int(job.priority)
        self.client.hset(self._job_key(job.id), mapping={
            "data": job.model_dump_json(),
            "status": job.status.value,
            "priority": priority
        })
        
        # If status changed to terminal, remove from running
        if job.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
            self.client.srem(self._key("queue:running"), job.id)
            self.client.zrem(self._key("queue:pending"), job.id)

    def cancel(self, job_id: str) -> bool:
        job = self.get_job(job_id)
        if not job:
            return False
        
        if job.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
            return False
            
        job.status = JobStatus.CANCELLED
        job.completed_at = datetime.now()
        self.update_job(job)
        return True

    def clear(self) -> None:
        keys = self.client.keys(self._key("*"))
        if keys:
            self.client.delete(*keys)
