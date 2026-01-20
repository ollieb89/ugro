from fastapi import APIRouter
from ugro.cluster_state import ClusterStateManager
from typing import Dict, Any

router = APIRouter(prefix="/cluster", tags=["cluster"])
state_manager = ClusterStateManager()

@router.get("/health")
async def get_cluster_health() -> Dict[str, Any]:
    """Get current health status of all nodes."""
    state = state_manager.refresh()
    
    # Calculate aggregate health
    nodes = state.nodes
    total_nodes = len(nodes)
    healthy_nodes = sum(1 for n in nodes.values() if n.status == "healthy")
    
    return {
        "status": "healthy" if total_nodes > 0 and healthy_nodes == total_nodes else "degraded",
        "total_nodes": total_nodes,
        "healthy_nodes": healthy_nodes,
        "nodes": {
            name: {
                "status": node.status,
                "health_score": node.health_score,
                "last_check": node.last_check,
                "gpu": node.gpu
            }
            for name, node in nodes.items()
        }
    }

@router.get("/state")
async def get_cluster_state() -> Dict[str, Any]:
    """Get full cluster state."""
    state = state_manager.refresh()
    return state.to_dict()
