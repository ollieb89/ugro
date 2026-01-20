from typing import List
from fastapi import APIRouter, HTTPException, Query

from apps.api.schemas.job import JobSummary, JobMetricsResponse, MetricPoint
from apps.api.services.job_service import JobService

router = APIRouter(prefix="/jobs", tags=["jobs"])
service = JobService()

@router.get("/", response_model=List[JobSummary])
async def list_jobs(
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0)
):
    """List all training jobs."""
    return await service.list_jobs(limit, offset)

@router.get("/{job_id}", response_model=JobSummary)
async def get_job(job_id: str):
    """Get details for a specific job."""
    job = await service.get_job_details(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job

@router.get("/{job_id}/metrics", response_model=List[MetricPoint])
async def get_job_metrics(job_id: str):
    """Get time-series metrics for a job."""
    metrics = await service.get_job_metrics(job_id)
    # Return empty list instead of 404 if job exists but no metrics yet
    # But check if job exists first
    job = await service.get_job_details(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return metrics
