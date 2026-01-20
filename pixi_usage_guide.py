#!/usr/bin/env python3
"""
UGRO Cluster Health Monitoring - Pixi Environment Usage

This script demonstrates the cluster health monitoring functionality
using pixi tasks for proper environment management.
"""

def main():
    print("UGRO Cluster Health Monitoring - Pixi Environment")
    print("=" * 60)
    print()
    
    print("🚀 Available Pixi Commands:")
    print()
    
    print("1. CLI Health Check:")
    print("   pixi run ugro-health")
    print("   # Runs: python -m ugro.cli health")
    print()
    
    print("2. Direct API Test:")
    print("   pixi run ugro-test-health")
    print("   # Runs: python src/test_cluster_health.py")
    print()
    
    print("3. General CLI Access:")
    print("   pixi run ugro --help")
    print("   # Shows all available UGRO commands")
    print()
    
    print("4. Inline Python (exact documentation snippet):")
    print('   pixi run python -c "')
    print("   from ugro.config import load_config, expand_paths")
    print("   from ugro.cluster import Cluster")
    print()
    print("   config = expand_paths(load_config())")
    print("   cluster = Cluster(config)")
    print()
    print("   # Check health")
    print("   health = cluster.check_health()")
    print("   for name, status in health.items():")
    print('       symbol = "✓" if status["healthy"] else "❌"')
    print('       print(f"{symbol} {name}: {status[\"message\"]}")')
    print()
    print("   # Test worker operations")
    print("   if cluster.get_all_workers():")
    print("       worker = cluster.get_all_workers()[0][\"name\"]")
    print('       success, stdout, stderr = cluster.execute_on_worker(worker, "echo Hello from worker")')
    print('       print(f"Worker command: {success} - {stdout.strip()}")')
    print('   "')
    print()
    
    print("✅ All commands use pixi environment management!")
    print("📦 Dependencies are automatically managed via pixi.toml")

if __name__ == "__main__":
    main()
