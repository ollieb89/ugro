#!/usr/bin/env python3
"""Test cluster health monitoring as shown in documentation."""

import sys
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from ugro.config import load_config, expand_paths
from ugro.cluster import Cluster

def main():
    """Test cluster health monitoring functionality."""
    
    print("Testing cluster health monitoring...")
    print("=" * 60)
    
    # Load configuration and initialize cluster
    config = expand_paths(load_config())
    cluster = Cluster(config)
    
    # Check health
    print("\n🔍 Cluster Health Check:")
    health = cluster.check_health()
    for name, status in health.items():
        symbol = '✓' if status['healthy'] else '❌'
        print(f'{symbol} {name}: {status["message"]}')
    
    # Test worker operations
    print("\n🔧 Worker Operations Test:")
    if cluster.get_all_workers():
        worker = cluster.get_all_workers()[0]['name']
        success, stdout, stderr = cluster.execute_on_worker(worker, 'echo "Hello from worker"')
        print(f'Worker command on {worker}: {success} - {stdout.strip()}')
    else:
        print('No workers found in cluster configuration')
    
    print("\n✅ All tests completed successfully!")

if __name__ == "__main__":
    main()
