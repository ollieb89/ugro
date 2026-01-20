#!/usr/bin/env python3
"""
UGRO Monitor Daemon - Continuous health monitoring service.

This daemon runs the AdaptiveHealthMonitor in the background,
continuously checking cluster node health and updating state.
"""

import asyncio
import logging
import signal
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ugro.cluster import Cluster
from ugro.cluster_state import ClusterStateManager
from ugro.health_monitor import create_health_monitor


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("/tmp/ugro-monitor.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


class MonitorDaemon:
    """Daemon wrapper for AdaptiveHealthMonitor."""
    
    def __init__(self):
        self.cluster = None
        self.state_manager = None
        self.health_monitor = None
        self.running = False
        
    async def setup(self):
        """Initialize cluster, state manager, and health monitor."""
        logger.info("Initializing UGRO monitor daemon...")
        
        # Load cluster config
        from ugro.config import get_config_dir
        import yaml
        
        config_dir = get_config_dir()
        cluster_config_path = config_dir / "cluster.yaml"
        
        if not cluster_config_path.exists():
            raise FileNotFoundError(f"Cluster config not found: {cluster_config_path}")
            
        with open(cluster_config_path) as f:
            cluster_config = yaml.safe_load(f)
        
        # Initialize cluster
        self.cluster = Cluster(cluster_config)
        logger.info(f"Loaded cluster with {len(self.cluster.workers)} workers")
        
        # Initialize state manager
        self.state_manager = ClusterStateManager()
        logger.info("Initialized cluster state manager")
        
        # Create health monitor
        self.health_monitor = create_health_monitor(
            self.cluster,
            self.state_manager
        )
        logger.info("Created health monitor")
        
    async def run(self):
        """Run the monitoring loop."""
        self.running = True
        logger.info("Starting health monitoring...")
        
        try:
            await self.health_monitor.start_monitoring()
        except Exception as e:
            logger.error(f"Error in monitoring loop: {e}", exc_info=True)
            raise
            
    async def shutdown(self):
        """Gracefully shutdown the daemon."""
        logger.info("Shutting down monitor daemon...")
        self.running = False
        
        if self.health_monitor:
            await self.health_monitor.stop()
            
        logger.info("Monitor daemon stopped")


# Global daemon instance
daemon = None


def signal_handler(signum, frame):
    """Handle shutdown signals."""
    logger.info(f"Received signal {signum}, initiating shutdown...")
    if daemon:
        asyncio.create_task(daemon.shutdown())


async def main():
    """Main entry point."""
    global daemon
    
    # Set up signal handlers
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    # Create and run daemon
    daemon = MonitorDaemon()
    
    try:
        await daemon.setup()
        await daemon.run()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        return 1
    finally:
        await daemon.shutdown()
        
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
