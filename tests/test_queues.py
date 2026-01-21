import pytest
import os
import time
import sqlite3
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

def test_sqlite_queue_migrates_legacy_schema(tmp_path):
    db_path = tmp_path / "legacy_ugro.db"

    # Legacy schema (created by ugro.database.Database) lacks `job_data`.
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE jobs (
                id TEXT PRIMARY KEY,
                priority INTEGER DEFAULT 0,
                status TEXT DEFAULT 'pending',
                model_name TEXT,
                dataset_name TEXT,
                config TEXT,
                created_at TIMESTAMP,
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                logs_path TEXT,
                worker_nodes TEXT
            )
            """
        )

    queue = SQLiteJobQueue(db_path=str(db_path))
    job = Job(name="test_job", command="echo hello")
    job_id = queue.submit(job)
    assert job_id == job.id

def test_sqlite_list_jobs_skips_null_job_data(tmp_path):
    db_path = tmp_path / "legacy_rows.db"

    # Create a legacy jobs table (no job_data), then run queue init which adds the column.
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE jobs (
                id TEXT PRIMARY KEY,
                priority INTEGER DEFAULT 0,
                status TEXT DEFAULT 'pending',
                model_name TEXT,
                dataset_name TEXT,
                config TEXT,
                created_at TIMESTAMP,
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                logs_path TEXT,
                worker_nodes TEXT
            )
            """
        )

        # Insert a legacy row (will have NULL job_data after migration)
        conn.execute(
            "INSERT INTO jobs (id, priority, status, created_at) VALUES (?, ?, ?, CURRENT_TIMESTAMP)",
            ("legacy-1", 0, "pending"),
        )

    queue = SQLiteJobQueue(db_path=str(db_path))
    queue.submit(Job(name="new", command="echo new"))

    jobs = queue.list_jobs(limit=10)
    assert all(j is not None for j in jobs)
    assert all(j.name for j in jobs)

def test_sqlite_list_jobs_filters_by_enum_status(sqlite_queue):
    job = Job(name="enum-status", command="echo ok")
    sqlite_queue.submit(job)

    jobs = sqlite_queue.list_jobs(status=JobStatus.PENDING, limit=10)
    assert any(j.id == job.id for j in jobs)
