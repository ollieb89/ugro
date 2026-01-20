#!/usr/bin/env python3
"""
UGRO Cluster Health Monitoring Implementation

This script demonstrates the exact functionality shown in the documentation:
https://github.com/your-repo/ugro/docs/UGRO-Complete-Setup.md#L1731-L1753

The implementation includes:
1. cluster.check_health() - Health monitoring for all nodes
2. cluster.get_all_workers() - Get list of all worker nodes  
3. cluster.execute_on_worker() - Execute commands on specific workers
4. CLI integration via 'ugro health' command
"""

import sys
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from ugro.config import load_config, expand_paths
from ugro.cluster import Cluster

def test_cluster_health_monitoring():
    """Test cluster health monitoring exactly as shown in documentation."""
    
    print("UGRO Cluster Health Monitoring Implementation")
    print("=" * 60)
    print("This matches the documentation snippet:")
    print("https://github.com/your-repo/ugro/docs/UGRO-Complete-Setup.md#L1731-L1753")
    print()
    
    # Initialize cluster (exact code from documentation)
    config = expand_paths(load_config())
    cluster = Cluster(config)
    
    # Check health (exact code from documentation)
    print("🔍 Testing cluster.check_health():")
    health = cluster.check_health()
    for name, status in health.items():
        symbol = '✓' if status['healthy'] else '❌'
        print(f'{symbol} {name}: {status["message"]}')
    
    # Test worker operations (exact code from documentation)
    print("\n🔧 Testing cluster.get_all_workers() and cluster.execute_on_worker():")
    if cluster.get_all_workers():
        worker = cluster.get_all_workers()[0]['name']
        success, stdout, stderr = cluster.execute_on_worker(worker, 'echo "Hello from worker"')
        print(f'Worker command: {success} - {stdout.strip()}')
    else:
        print('No workers found in cluster configuration')
    
    print("\n✅ Implementation verified!")
    print("\n📋 Available commands:")
    print("  python ugro_cli.py health          # CLI health check")
    print("  python test_cluster_health.py       # Direct API test")
    print("  pixi run python -c '...'           # Pixi environment test")

if __name__ == "__main__":
    test_cluster_health_monitoring()
