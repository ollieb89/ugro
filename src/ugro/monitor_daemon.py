#!/usr/bin/env python3
"""UGRO Health Monitoring Daemon.

Runs the AdaptiveHealthMonitor in a background process to provide
real-time cluster health and metrics collection.
"""

import asyncio
import logging
import os
import signal
import sys
from pathlib import Path

from .agent import UGROAgent
from .health_monitor import AdaptiveHealthMonitor, MonitoringConfig

try:
    from prometheus_client import start_http_server
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

# Configure logging
LOG_DIR = Path(f"{os.getenv('HOME')}/Development/Tools/ugro/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "monitor_daemon.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("ugro.monitor_daemon")

class MonitorDaemon:
    def __init__(self):
        self.agent = UGROAgent()
        self.config = MonitoringConfig(
            base_interval=10.0,
            max_interval=60.0,
            min_interval=5.0
        )
        self.monitor = AdaptiveHealthMonitor(
            cluster=self.agent.cluster,
            state_manager=self.agent.cluster_state_manager,
            config=self.config
        )
        self._stop_event = asyncio.Event()

    def handle_exit(self, sig, frame):
        logger.info(f"Received signal {sig}, shutting down...")
        self._stop_event.set()

    async def run(self):
        logger.info("UGRO Monitor Daemon starting...")
        
        # Register signal handlers
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, self._stop_event.set)

        # Start Prometheus metrics server
        if PROMETHEUS_AVAILABLE:
            try:
                # Todo: make port configurable under config.monitoring.prometheus_port
                start_http_server(8000)
                logger.info("Prometheus metrics server started on port 8000")
            except Exception as e:
                logger.error(f"Failed to start metrics server: {e}")
        else:
            logger.warning("prometheus_client not installed, metrics server disabled")

        # Start the monitor loop as a background task

        # Start the monitor loop as a background task
        monitor_task = asyncio.create_task(self.monitor.start_monitoring())
        
        logger.info("Monitor loop active.")
        
        # Wait for stop event
        await self._stop_event.wait()
        
        logger.info("Shutting down monitor...")
        self.monitor._running = False
        await self.monitor.stop()
        
        # Wait for task to finish
        try:
            await asyncio.wait_for(monitor_task, timeout=10)
        except asyncio.TimeoutError:
            logger.warning("Monitor task shutdown timed out.")
        except Exception as e:
            logger.error(f"Error during monitor shutdown: {e}")

        logger.info("UGRO Monitor Daemon stopped.")

def main():
    daemon = MonitorDaemon()
    try:
        asyncio.run(daemon.run())
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logger.exception(f"Fatal error in monitor daemon: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
