#!/usr/bin/env python3
"""
UGRO Worker Synchronization Script

Synchronizes the project repository from master to all worker nodes using rsync.
"""

import logging
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, List
import yaml


class WorkerSynchronizer:
    """Handles synchronization to worker nodes."""
    
    def __init__(self, config_path: str = "config/cluster.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.logger = self._setup_logging()
        
        # Get paths from config
        self.master_project_path = Path(self.config['paths']['project_root'])
        self.workers = self.config['workers']
        
    def _load_config(self) -> Dict[str, Any]:
        """Load cluster configuration."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        log_config = self.config.get('logging', {})
        log_file = log_config.get('file', 'logs/sync.log')
        
        # Ensure log directory exists
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=getattr(logging, log_config.get('level', 'INFO')),
            format=log_config.get('format', '[%(asctime)s] %(name)s - %(levelname)s - %(message)s'),
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        return logging.getLogger(__name__)
    
    def _run_rsync(self, worker: Dict[str, Any]) -> bool:
        """Run rsync to a specific worker."""
        worker_name = worker['name']
        worker_host = worker['hostname']
        worker_user = worker['user']
        worker_project_path = worker['paths']['project']
        
        self.logger.info(f"Syncing to {worker_name} ({worker_user}@{worker_host})")
        
        # Build rsync command
        rsync_cmd = [
            'rsync',
            '-avz',                    # archive, verbose, compress
            '--delete',                # delete files that don't exist on master
            '--exclude', '.git',       # exclude git directory
            '--exclude', '__pycache__', # exclude Python cache
            '--exclude', '*.pyc',      # exclude compiled Python files
            '--exclude', '.pytest_cache', # exclude pytest cache
            '--exclude', 'logs/',      # exclude logs directory
            '--exclude', '.venv',      # exclude virtual environments
            '--exclude', 'node_modules', # exclude node modules
            '--exclude', '.DS_Store',  # exclude macOS files
            '--exclude', '*.log',      # exclude log files
            f"{self.master_project_path}/",  # source (trailing slash important!)
            f"{worker_user}@{worker_host}:{worker_project_path}/"  # destination
        ]
        
        try:
            self.logger.debug(f"Running rsync command: {' '.join(rsync_cmd)}")
            
            result = subprocess.run(
                rsync_cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout per worker
            )
            
            if result.returncode == 0:
                self.logger.info(f"Successfully synced to {worker_name}")
                if result.stdout:
                    self.logger.debug(f"rsync output for {worker_name}: {result.stdout}")
                return True
            else:
                self.logger.error(f"rsync failed for {worker_name}: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            self.logger.error(f"rsync timed out for {worker_name}")
            return False
        except Exception as e:
            self.logger.error(f"Error running rsync for {worker_name}: {e}")
            return False
    
    def _test_ssh_connection(self, worker: Dict[str, Any]) -> bool:
        """Test SSH connection to a worker."""
        worker_name = worker['name']
        worker_host = worker['hostname']
        worker_user = worker['user']
        
        self.logger.info(f"Testing SSH connection to {worker_name}")
        
        try:
            result = subprocess.run(
                ['ssh', f'{worker_user}@{worker_host}', 'echo "SSH connection test successful"'],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                self.logger.info(f"SSH connection to {worker_name} successful")
                return True
            else:
                self.logger.error(f"SSH connection to {worker_name} failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            self.logger.error(f"SSH connection to {worker_name} timed out")
            return False
        except Exception as e:
            self.logger.error(f"Error testing SSH connection to {worker_name}: {e}")
            return False
    
    def sync_all_workers(self) -> bool:
        """Sync to all workers."""
        self.logger.info(f"Starting synchronization to {len(self.workers)} workers")
        
        if not self.master_project_path.exists():
            self.logger.error(f"Master project path does not exist: {self.master_project_path}")
            return False
        
        success_count = 0
        total_workers = len(self.workers)
        
        for worker in self.workers:
            worker_name = worker['name']
            
            # Test SSH connection first
            if not self._test_ssh_connection(worker):
                self.logger.warning(f"Skipping {worker_name} due to SSH connection failure")
                continue
            
            # Run rsync
            if self._run_rsync(worker):
                success_count += 1
            else:
                self.logger.error(f"Failed to sync to {worker_name}")
        
        self.logger.info(f"Synchronization completed: {success_count}/{total_workers} workers successful")
        return success_count == total_workers
    
    def sync_worker(self, worker_name: str) -> bool:
        """Sync to a specific worker."""
        worker = next((w for w in self.workers if w['name'] == worker_name), None)
        
        if not worker:
            self.logger.error(f"Worker {worker_name} not found in configuration")
            return False
        
        self.logger.info(f"Syncing to specific worker: {worker_name}")
        
        # Test SSH connection first
        if not self._test_ssh_connection(worker):
            self.logger.error(f"Cannot sync to {worker_name}: SSH connection failed")
            return False
        
        # Run rsync
        return self._run_rsync(worker)
    
    def list_workers(self) -> List[str]:
        """List all configured workers."""
        return [worker['name'] for worker in self.workers]
    
    def verify_worker_paths(self) -> Dict[str, bool]:
        """Verify that worker project paths exist."""
        results = {}
        
        for worker in self.workers:
            worker_name = worker['name']
            worker_host = worker['hostname']
            worker_user = worker['user']
            worker_project_path = worker['paths']['project']
            
            try:
                result = subprocess.run(
                    ['ssh', f'{worker_user}@{worker_host}', f'test -d {worker_project_path}'],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                results[worker_name] = result.returncode == 0
                
                if result.returncode == 0:
                    self.logger.info(f"Worker path exists on {worker_name}: {worker_project_path}")
                else:
                    self.logger.warning(f"Worker path missing on {worker_name}: {worker_project_path}")
                    
            except Exception as e:
                self.logger.error(f"Error checking path on {worker_name}: {e}")
                results[worker_name] = False
        
        return results


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='UGRO Worker Synchronization')
    parser.add_argument('--config', default='config/cluster.yaml', help='Configuration file path')
    parser.add_argument('--worker', help='Sync to specific worker only')
    parser.add_argument('--list', action='store_true', help='List all workers')
    parser.add_argument('--verify', action='store_true', help='Verify worker paths exist')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be synced without actually syncing')
    
    args = parser.parse_args()
    
    try:
        synchronizer = WorkerSynchronizer(args.config)
        
        if args.list:
            workers = synchronizer.list_workers()
            print("Configured workers:")
            for worker in workers:
                print(f"  - {worker}")
            return 0
        
        if args.verify:
            results = synchronizer.verify_worker_paths()
            print("Worker path verification:")
            for worker, exists in results.items():
                status = "✓" if exists else "✗"
                print(f"  {status} {worker}")
            return 0
        
        if args.dry_run:
            print("Dry run mode - showing what would be synced:")
            print(f"Source: {synchronizer.master_project_path}")
            for worker in synchronizer.workers:
                print(f"  -> {worker['user']}@{worker['hostname']}:{worker['paths']['project']}")
            return 0
        
        if args.worker:
            success = synchronizer.sync_worker(args.worker)
        else:
            success = synchronizer.sync_all_workers()
        
        return 0 if success else 1
        
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == '__main__':
    exit(main())