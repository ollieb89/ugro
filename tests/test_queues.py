import pytest
import os
import time
from ugro.queues import SQLiteJobQueue, Job, JobPriority, JobStatus

@pytest.fixture
def sqlite_queue(tmp_path):
    db_path = tmp_path / "test_ugro.db"
    queue = SQLiteJobQueue(db_path=str(db_path))
    yield queue
    queue.clear()

def test_sqlite_submit_and_peek(sqlite_queue):
    job = Job(name="test_job", command="echo hello")
    job_id = sqlite_queue.submit(job)
    assert job_id == job.id
    
    peeked = sqlite_queue.peek()
    assert peeked is not None
    assert peeked.id == job_id
    assert peeked.status == JobStatus.PENDING

def test_sqlite_priority(sqlite_queue):
    job_low = Job(name="low", command="echo low", priority=JobPriority.LOW)
    job_high = Job(name="high", command="echo high", priority=JobPriority.HIGH)
    
    sqlite_queue.submit(job_low)
    sqlite_queue.submit(job_high)
    
    # Next should be high priority
    next_job = sqlite_queue.next()
    assert next_job.id == job_high.id
    assert next_job.status == JobStatus.RUNNING
    
    # Next should be low
    next_job_2 = sqlite_queue.next()
    assert next_job_2.id == job_low.id

def test_sqlite_cancel(sqlite_queue):
    job = Job(name="cancel_me", command="sleep 100")
    sqlite_queue.submit(job)
    
    success = sqlite_queue.cancel(job.id)
    assert success is True
    
    cancelled_job = sqlite_queue.get_job(job.id)
    assert cancelled_job.status == JobStatus.CANCELLED
    
    # Cannot pick up cancelled job
    assert sqlite_queue.next() is None
