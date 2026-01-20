import sqlite3
import json
from datetime import datetime
from typing import List, Optional, Any
from pathlib import Path
from .models import Job, JobStatus
from .base import JobQueue

class SQLiteJobQueue(JobQueue):
    """
    SQLite-backed Job Queue.
    Stores jobs in a local SQLite database.
    """
    def __init__(self, db_path: str = "ugro.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize the database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    priority INTEGER,
                    status TEXT,
                    created_at TIMESTAMP,
                    job_data JSON
                )
            """)
            # Index for faster retrieval of pending jobs by priority
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_jobs_pending_priority 
                ON jobs(priority DESC, created_at ASC) 
                WHERE status = 'PENDING'
            """)

    def submit(self, job: Job) -> str:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO jobs (id, priority, status, created_at, job_data) VALUES (?, ?, ?, ?, ?)",
                (
                    job.id,
                    job.priority,
                    job.status,
                    job.created_at,
                    job.model_dump_json(),
                )
            )
        return job.id

    def next(self) -> Optional[Job]:
        """
        Get the next PENDING job with highest priority.
        Marks it as RUNNING immediately within a transaction.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Find candidate
            cursor.execute("""
                SELECT id, job_data FROM jobs 
                WHERE status = ? 
                ORDER BY priority DESC, created_at ASC 
                LIMIT 1
            """, (JobStatus.PENDING.value,))
            
            row = cursor.fetchone()
            if not row:
                return None
            
            job_id = row['id']
            job_data_str = row['job_data']
            
            # Use Pydantic to parse
            job = Job.model_validate_json(job_data_str)
            
            # Update state to RUNNING
            job.status = JobStatus.RUNNING
            job.started_at = datetime.now()
            
            # Update DB
            cursor.execute(
                "UPDATE jobs SET status = ?, job_data = ? WHERE id = ?",
                (JobStatus.RUNNING, job.model_dump_json(), job_id)
            )
            return job

    def claim(self, job_id: str) -> Optional[Job]:
        """
        Claim a specific job by ID.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Find candidate - strictly check ID and PENDING status
            cursor.execute("""
                SELECT id, job_data FROM jobs 
                WHERE id = ? AND status = ?
            """, (job_id, JobStatus.PENDING.value))
            
            row = cursor.fetchone()
            if not row:
                return None
            
            job_data_str = row['job_data']
            job = Job.model_validate_json(job_data_str)
            
            # Update state to RUNNING
            job.status = JobStatus.RUNNING
            job.started_at = datetime.now()
            
            # Update DB
            cursor.execute(
                "UPDATE jobs SET status = ?, job_data = ? WHERE id = ?",
                (JobStatus.RUNNING, job.model_dump_json(), job_id)
            )
            return job

    def peek(self) -> Optional[Job]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""
                SELECT job_data FROM jobs 
                WHERE status = ? 
                ORDER BY priority DESC, created_at ASC 
                LIMIT 1
            """, (JobStatus.PENDING.value,))
            row = cursor.fetchone()
            if row:
                return Job.model_validate_json(row['job_data'])
        return None

    def get_job(self, job_id: str) -> Optional[Job]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT job_data FROM jobs WHERE id = ?", (job_id,))
            row = cursor.fetchone()
            if row:
                return Job.model_validate_json(row['job_data'])
        return None

    def list_jobs(self, status: Optional[JobStatus] = None, limit: int = 100) -> List[Job]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            if status:
                cursor = conn.execute(
                    "SELECT job_data FROM jobs WHERE status = ? ORDER BY created_at DESC LIMIT ?",
                    (status, limit)
                )
            else:
                 cursor = conn.execute(
                    "SELECT job_data FROM jobs ORDER BY created_at DESC LIMIT ?",
                    (limit,)
                )
            
            jobs = []
            for row in cursor:
                jobs.append(Job.model_validate_json(row['job_data']))
            return jobs

    def update_job(self, job: Job) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE jobs SET status = ?, priority = ?, job_data = ? WHERE id = ?",
                (job.status, job.priority, job.model_dump_json(), job.id)
            )

    def cancel(self, job_id: str) -> bool:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Check current status
            cursor.execute("SELECT status, job_data FROM jobs WHERE id = ?", (job_id,))
            row = cursor.fetchone()
            if not row:
                return False
            
            current_status = row['status']
            if current_status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
                return False
            
            # Update to cancelled
            job = Job.model_validate_json(row['job_data'])
            job.status = JobStatus.CANCELLED
            job.completed_at = datetime.now() # effective completion
            
            conn.execute(
                "UPDATE jobs SET status = ?, job_data = ? WHERE id = ?",
                (JobStatus.CANCELLED, job.model_dump_json(), job_id)
            )
            return True

    def clear(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM jobs")
