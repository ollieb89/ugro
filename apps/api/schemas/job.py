from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class GpuStats(BaseModel):
    utilization: float = Field(..., description="GPU utilization percentage")
    memory_used_gb: float = Field(..., description="VRAM used in GB")
    memory_reserved_gb: Optional[float] = Field(None, description="VRAM reserved in GB")
    temperature: Optional[float] = Field(None, description="GPU temperature in Celsius")

class MetricPoint(BaseModel):
    timestamp: float
    step: int
    loss: Optional[float] = None
    learning_rate: Optional[float] = None
    throughput: Optional[float] = None
    gradient_norm: Optional[float] = None
    gpu_stats: Optional[Dict[str, float]] = None

class JobConfig(BaseModel):
    model_name: str
    dataset_name: str
    num_epochs: int
    batch_size: Optional[int] = None
    learning_rate: Optional[float] = None

class JobValidationStatus(BaseModel):
    is_valid: bool
    errors: List[str] = []

# Output Models

class JobSummary(BaseModel):
    job_id: str
    status: str
    model: str
    dataset: str
    started_at: datetime
    duration_seconds: Optional[float]
    total_steps: int
    final_loss: Optional[float]
    avg_throughput: Optional[float]
    gpu_nodes: List[str]
    
    # We can add a list of key metrics for the summary view
    metrics_summary: Optional[Dict[str, Any]] = None

class JobListResponse(BaseModel):
    jobs: List[JobSummary]
    total: int

class JobMetricsResponse(BaseModel):
    job_id: str
    metrics: List[MetricPoint]
