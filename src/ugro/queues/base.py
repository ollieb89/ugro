from abc import ABC, abstractmethod
from typing import List, Optional
from .models import Job, JobStatus

class JobQueue(ABC):
    """Abstract base class for job queues."""

    @abstractmethod
    def submit(self, job: Job) -> str:
        """
        Submit a job to the queue.
        Returns the job ID.
        """
        pass

    @abstractmethod
    def next(self) -> Optional[Job]:
        """
        Get the next pending job from the queue, obeying priority.
        Returns None if queue is empty or no suitable job is found.
        
        The implementation should mark the job as 'RUNNING' or lock it
        to prevent other workers from picking it up immediately, 
        or rely on an explicit claim mechanism.
        For simplicity in Phase 2d, we assume `next()` acts as a claim.
        """
        pass

    @abstractmethod
    def claim(self, job_id: str) -> Optional[Job]:
        """
        Claim a specific job by ID (used for backfilling/scheduling specific jobs).
        Atomically transitions status from PENDING to RUNNING.
        Returns the Job object if successful, None if already claimed/not found.
        """
        pass
        
    @abstractmethod
    def peek(self) -> Optional[Job]:
        """
        Look at the next job without claiming it.
        """
        pass

    @abstractmethod
    def get_job(self, job_id: str) -> Optional[Job]:
        """Get a specific job by ID."""
        pass

    @abstractmethod
    def list_jobs(self, status: Optional[JobStatus] = None, limit: int = 100) -> List[Job]:
        """List jobs, optionally filtered by status."""
        pass

    @abstractmethod
    def update_job(self, job: Job) -> None:
        """Update a job's state (status, timestamps, etc.) in the storage."""
        pass
    
    @abstractmethod
    def cancel(self, job_id: str) -> bool:
        """
        Cancel a pending job. 
        Returns True if cancelled, False if not found or already running/completed.
        """
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all jobs from the queue (mostly for testing)."""
        pass
