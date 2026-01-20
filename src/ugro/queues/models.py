from enum import Enum, IntEnum
from typing import Dict, Optional, Any, List
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict
import uuid

class JobStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"

class JobPriority(IntEnum):
    LOW = 0
    NORMAL = 10
    HIGH = 20
    URGENT = 30

class JobResources(BaseModel):
    gpu_count: int = Field(default=1, description="Number of GPUs required per node")
    min_vram_gb: int = Field(default=8, description="Minimum VRAM per GPU in GB")
    cpu_cores: int = Field(default=1, description="Number of CPU cores required per node")
    nnodes: int = Field(default=1, description="Number of nodes required for gang scheduling")

class Job(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    command: str = Field(description="Command to execute")
    priority: JobPriority = Field(default=JobPriority.NORMAL)
    status: JobStatus = Field(default=JobStatus.PENDING)
    
    resources: JobResources = Field(default_factory=JobResources)
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Runtime info
    worker_id: Optional[str] = Field(default=None, description="Primary node for single-node jobs")
    worker_ids: Optional[List[str]] = Field(default=None, description="List of assigned nodes for gang scheduling")
    exit_code: Optional[int] = None
    error_message: Optional[str] = None
    
    # Arbitrary tags or metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)

