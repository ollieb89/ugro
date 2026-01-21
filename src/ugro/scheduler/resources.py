from typing import Dict, Any, List, Optional
import requests
import logging

logger = logging.getLogger(__name__)
# We'll need access to the cluster monitor potentially
# For Phase 2d, we can mock it or integrate with the existing monitoring code if importable

class ResourceTracker:
    """
    Tracks available resources in the cluster to help Scheduler make decisions.
    
    In a full implementation, this would query the ClusterStateManager or Prometheus.
    For this phase, we will assume a static cluster config or a simple state.
    """
    def __init__(self, cluster_config: Dict[str, Any] = None, prometheus_url: str = None):
        self.cluster_config = cluster_config or {}
        # Default to localhost:8000 for our daemon if not specified
        self.prometheus_url = prometheus_url 
        # In-memory "mock" usage for now until we hook up real monitoring
        self.node_usage: Dict[str, Dict[str, float]] = {} 
        
    def sync_from_prometheus(self):
        """Polls prometheus for real-time metrics and updates internal state."""
        if not self.prometheus_url:
            return

        try:
            # We assume standard Prometheus /api/v1/query format, but our daemon 
            # exposes /metrics (text format).
            # If the user has a full Prometheus server scraping our daemon, we query that server.
            # If we point directly to our daemon (port 8000), we get raw text.
            # For "Real-Time Prometheus Metrics", we should assume a Prometheus Server is the source of truth.
            
            # Example query: ugro_gpu_utilization_percent
            response = requests.get(f"{self.prometheus_url}/api/v1/query", params={
                "query": "ugro_gpu_mem_used_bytes"
            }, timeout=2)
            
            if response.status_code == 200:
                data = response.json()
                if data["status"] == "success":
                    for result in data["data"]["result"]:
                        node = result["metric"].get("node")
                        # gpu_idx = result["metric"].get("gpu_index")
                        val_bytes = float(result["value"][1])
                        
                        if node:
                             if node not in self.node_usage:
                                 self.node_usage[node] = {"vram_gb": 0, "gpu_count": 0}
                             
                             # Convert bytes to GB and update
                             # This overwrites our "in-flight" tracking which might be dangerous 
                             # if the scheduler relies on looking ahead.
                             # But for observability, this is "current reality".
                             # Ideally we mix "reserved" vs "actual". 
                             # For Phase 2e, we just update "actual usage".
                             self.node_usage[node]["vram_gb"] = val_bytes / (1024**3)
                             
            # Also get GPU count? Or assume config is static? Config is static.
            
        except Exception as e:
            logger.warning(f"Failed to sync with Prometheus at {self.prometheus_url}: {e}")

    def update_usage(self, node_name: str, used_vram: int, used_gpu_count: int):
        """Update known usage for a node (adds to existing usage)."""
        if node_name not in self.node_usage:
            self.node_usage[node_name] = {"vram_gb": 0, "gpu_count": 0}
        
        self.node_usage[node_name]["vram_gb"] += used_vram
        self.node_usage[node_name]["gpu_count"] += used_gpu_count
    
    def release_usage(self, node_name: str, vram_gb: int, gpu_count: int):
        """Release resources after job completion."""
        if node_name in self.node_usage:
            self.node_usage[node_name]["vram_gb"] = max(0, self.node_usage[node_name]["vram_gb"] - vram_gb)
            self.node_usage[node_name]["gpu_count"] = max(0, self.node_usage[node_name]["gpu_count"] - gpu_count)

    def can_fit(self, job_resources) -> List[str]:
        """
        Check which nodes can fit the job.
        Returns list of node names sorted by Best Fit (least remaining resources after placement).
        This implements a Best Fit Decreasing (BFD) strategy when used with a sorted queue.
        """
        node_scores = [] # Tuple (node_name, waste_score)
        
        # Logic to check self.cluster_config vs job_resources
        # For MVP, fall back to localhost if config doesn't define schedulable nodes.
        if not self.cluster_config:
            return ["localhost"]

        nodes_cfg = self.cluster_config.get('nodes', {})
        if not isinstance(nodes_cfg, dict) or not nodes_cfg:
            return ["localhost"]

        for node_name, node_info in nodes_cfg.items():
            # Check GPU count
            total_gpus = 1 
            if 'gpu_count' in node_info:
                total_gpus = node_info['gpu_count']
            
            # Check VRAM
            total_vram = node_info.get('vram_gb', 8)
            
            # Check usage
            current_usage = self.node_usage.get(node_name, {"vram_gb": 0, "gpu_count": 0})
            
            # Resources available *before* placement
            free_gpus = total_gpus - current_usage.get('gpu_count', 0)
            free_vram = total_vram - current_usage.get('vram_gb', 0.0)
            
            # Check strict constraints
            if free_gpus >= job_resources.gpu_count:
                # Check if the node has enough capacity
                # Simplified: Node must have enough TOTAL vram for the job.
                if free_vram >= job_resources.min_vram_gb:
                    
                    # Calculate "Waste" / "Fit Score"
                    # Best Fit: Choose node where remaining space is SMALLEST (tightest fit)
                    # to leave large spaces for large jobs.
                    # Score = Remaining Free VRAM after placement
                    remaining_vram = free_vram - job_resources.min_vram_gb
                    
                    # Let's simple score: VRAM waste
                    node_scores.append((node_name, remaining_vram))

        # Sort by score ascending (Smallest remaining space -> Best Fit)
        node_scores.sort(key=lambda x: x[1])
        
        return [n[0] for n in node_scores]

    def can_fit_gang(self, job_resources) -> Optional[List[str]]:
        """
        Check if we can allocate N nodes for a gang-scheduled job.
        
        Returns list of node names if we can satisfy the entire gang requirement,
        None if insufficient resources. Uses Best-Fit Decreasing strategy.
        
        Args:
            job_resources: JobResources with nnodes indicating nodes required
            
        Returns:
            List of N best-fit node names, or None if gang can't be scheduled
        """
        nnodes_required = getattr(job_resources, 'nnodes', 1)
        
        if nnodes_required <= 1:
            # Fallback to single-node scheduling
            capable = self.can_fit(job_resources)
            return capable[:1] if capable else None
        
        # Get all capable nodes sorted by BFD
        capable = self.can_fit(job_resources)
        
        if len(capable) >= nnodes_required:
            # Return exactly N nodes (best-fit sorted)
            logger.info(f"Gang scheduling: {nnodes_required} nodes required, {len(capable)} available")
            return capable[:nnodes_required]
        
        logger.debug(f"Gang scheduling failed: need {nnodes_required}, only {len(capable)} capable nodes")
        return None

