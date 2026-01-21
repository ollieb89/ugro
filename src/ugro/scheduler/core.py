import time
import logging
from datetime import datetime
from typing import Optional
from ..queues.base import JobQueue
from ..queues.models import Job, JobStatus
from .resources import ResourceTracker

logger = logging.getLogger(__name__)

class Scheduler:
    """
    Core Scheduler Loop.
    Checks queue, checks resources, schedules jobs.
    """
    def __init__(self, queue: JobQueue, resource_tracker: ResourceTracker, poll_interval: int = 5):
        self.queue = queue
        self.tracker = resource_tracker
        self.poll_interval = poll_interval
        self.running = False

    def schedule_next(self) -> Optional[Job]:
        """
        Try to schedule the next job.
        
        Optimized Strategy (Backfilling + Gang Scheduling):
        1. List pending jobs (ordered by priority DESC, created_at ASC).
        2. Iterate through candidates.
        3. For multi-node jobs: Use can_fit_gang() for atomic N-node allocation.
        4. For single-node jobs: Use can_fit() for single node.
        5. If resources fit, CLAIM job and allocate all nodes.
        6. If no, continue to next candidate (Backfilling).
        """
        pending_jobs = self.queue.list_jobs(status=JobStatus.PENDING, limit=50)
        
        if not pending_jobs:
            return None
        
        for job in pending_jobs:
            nnodes = getattr(job.resources, 'nnodes', 1)
            
            if nnodes > 1:
                # Gang scheduling path: need all N nodes or nothing
                candidate_nodes = self.tracker.can_fit_gang(job.resources)
                
                if not candidate_nodes:
                    # Can't satisfy gang requirement, try next job (backfilling)
                    logger.debug(f"Job {job.id} pending: Need {nnodes} nodes, insufficient resources.")
                    continue
                
                # We have all nodes! Claim the job
                claimed_job = self.queue.claim(job.id)
                
                if claimed_job:
                    # Assign all nodes for gang scheduling
                    claimed_job.worker_ids = candidate_nodes
                    claimed_job.worker_id = candidate_nodes[0]  # Primary node
                    
                    # Reserve resources on ALL gang nodes
                    for node in candidate_nodes:
                        self.tracker.update_usage(
                            node,
                            used_vram=job.resources.min_vram_gb,
                            used_gpu_count=job.resources.gpu_count
                        )
                    
                    self.queue.update_job(claimed_job)
                    logger.info(f"Gang-scheduled job {claimed_job.id} on {len(candidate_nodes)} nodes: {candidate_nodes}")
                    return claimed_job
                else:
                    logger.debug(f"Failed to claim gang job {job.id}, already taken.")
                    continue
                    
            else:
                # Single-node scheduling path (existing logic)
                candidate_nodes = self.tracker.can_fit(job.resources)
                
                if not candidate_nodes:
                    continue
                    
                claimed_job = self.queue.claim(job.id)
                
                if claimed_job:
                    claimed_job.worker_id = candidate_nodes[0]
                    self.queue.update_job(claimed_job)
                    
                    self.tracker.update_usage(
                        candidate_nodes[0],
                        used_vram=job.resources.min_vram_gb,
                        used_gpu_count=job.resources.gpu_count
                    )
                    
                    logger.info(f"Scheduled job {claimed_job.id} on {claimed_job.worker_id} (BFD)")
                    return claimed_job
                else:
                    logger.debug(f"Failed to claim job {job.id}, already taken.")
                    continue
            
        return None

    def loop(self):
        """
        Main scheduling loop.
        """
        self.running = True
        logger.info("Scheduler started.")
        try:
            while self.running:
                # Sync resources if possible
                if hasattr(self.tracker, 'sync_from_prometheus'):
                     self.tracker.sync_from_prometheus()
                     
                job = self.schedule_next()
                if job:
                    # Execute job? 
                    # The Scheduler usually just assigns. An Agent/Worker executes.
                    # For `ugro run-worker`, we might be both Scheduler AND Worker.
                    # Let's assume this loop is running in `ugro run-worker`.
                    self.execute_job(job)
                else:
                    print(f"⏳ No schedulable jobs. Sleeping {self.poll_interval}s...", flush=True)
                    time.sleep(self.poll_interval)
        except KeyboardInterrupt:
            logger.info("Scheduler stopping...")
            self.running = False

    def execute_job(self, job: Job):
        """
        Execute the job.
        For Phase 2d, this uses subprocess to run the command on the assigned node.
        """
        import subprocess
        
        logger.info(f"Executing job {job.id}: {job.command}")
        
        # Determine if we need SSH
        # For MVP let's run locally if node is localhost matches
        # or simplified: just run subprocess.
        
        try:
            # If command is "ugro launch ...", we might want to run it directly?
            # Or just shell out.
            
            # Update status to running (already done by queue.next() but good to confirm)
            
            # Run
            # Blocking call for simplicity in `run-worker` MVP.
            # Real production would be async or thread pool.
            result = subprocess.run(job.command, shell=True, capture_output=True, text=True)
            
            # Update status
            job.exit_code = result.returncode
            job.completed_at = datetime.now()
            
            if result.returncode == 0:
                job.status = JobStatus.COMPLETED
            else:
                job.status = JobStatus.FAILED
                job.error_message = result.stderr
                
            self.queue.update_job(job)
            
            # Release resources
            # Ideally we know exactly what was released.
            # self.tracker.release(...) 
            # Re-syncing tracker is safer.
            # self.tracker.sync()
            
        except Exception as e:
            logger.error(f"Job {job.id} failed execution: {e}")
            job.status = JobStatus.FAILED
            job.error_message = str(e)
            self.queue.update_job(job)
