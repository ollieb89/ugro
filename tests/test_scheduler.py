import pytest
from unittest.mock import MagicMock
from ugro.scheduler import Scheduler, ResourceTracker
from ugro.queues import Job, JobResources, JobStatus, JobPriority

class MockQueue:
    def __init__(self):
        self.jobs = []
    
    def peek(self):
        return self.jobs[0] if self.jobs else None
    
    def next(self):
        if self.jobs:
            job = self.jobs.pop(0)
            job.status = JobStatus.RUNNING
            return job
        return None
    
    def list_jobs(self, status=None, limit=100):
        """Return list of jobs filtered by status."""
        if status == JobStatus.PENDING:
            return [j for j in self.jobs if j.status == JobStatus.PENDING][:limit]
        elif status is None:
            return self.jobs[:limit]
        return [j for j in self.jobs if j.status == status][:limit]
    
    def claim(self, job_id):
        """Claim a specific job by ID."""
        for i, job in enumerate(self.jobs):
            if job.id == job_id and job.status == JobStatus.PENDING:
                job.status = JobStatus.RUNNING
                return self.jobs.pop(i)
        return None
    
    def update_job(self, job):
        pass

@pytest.fixture
def mock_tracker():
    tracker = ResourceTracker()
    tracker.can_fit = MagicMock(return_value=["node1"])
    tracker.update_usage = MagicMock()
    return tracker

def test_scheduler_schedule_success(mock_tracker):
    queue = MockQueue()
    job = Job(name="job1", command="echo 1", resources=JobResources(gpu_count=1))
    queue.jobs.append(job)
    
    scheduler = Scheduler(queue, mock_tracker)
    scheduled_job = scheduler.schedule_next()
    
    assert scheduled_job is not None
    assert scheduled_job.id == job.id
    assert scheduled_job.worker_id == "node1"
    mock_tracker.update_usage.assert_called_once()

def test_scheduler_no_resources():
    queue = MockQueue()
    job = Job(name="job1", command="echo 1")
    queue.jobs.append(job)
    
    tracker = ResourceTracker()
    tracker.can_fit = MagicMock(return_value=[]) # No resources
    
    scheduler = Scheduler(queue, tracker)
    scheduled_job = scheduler.schedule_next()
    
    assert scheduled_job is None
    # Job should remain in queue (peeked but not popped)
    assert queue.peek().id == job.id
