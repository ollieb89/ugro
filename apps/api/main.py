from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from apps.api.routers.v1 import jobs, cluster

app = FastAPI(
    title="UGRO API",
    description="Unified GPU Resource Orchestrator API",
    version="1.0.0",
)

# CORS configuration
origins = [
    "http://localhost:3000",
    "http://localhost:5173",  # Vite default
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(jobs.router, prefix="/api/v1")
app.include_router(cluster.router, prefix="/api/v1")

@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "ugro-api"}
