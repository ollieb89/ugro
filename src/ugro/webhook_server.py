#!/usr/bin/env python3
"""
GitHub Webhook Server for UGRO Cluster Synchronization

Receives GitHub push events and triggers synchronization to worker nodes.
"""

import hmac
import hashlib
import json
import logging
import subprocess
import threading
from pathlib import Path
from typing import Dict, Any, Optional

import yaml
from flask import Flask, request, jsonify


class WebhookServer:
    """GitHub webhook server for cluster synchronization."""
    
    def __init__(self, config_path: str = "config/cluster.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.logger = self._setup_logging()
        self.app = Flask(__name__)
        self._setup_routes()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load cluster configuration."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        log_config = self.config.get('logging', {})
        log_file = log_config.get('file', 'logs/webhook.log')
        
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
    
    def _setup_routes(self):
        """Setup Flask routes."""
        
        @self.app.route('/webhook/github', methods=['POST'])
        def handle_github_webhook():
            """Handle GitHub webhook events."""
            return self._handle_webhook()
        
        @self.app.route('/health', methods=['GET'])
        def health_check():
            """Health check endpoint."""
            return jsonify({'status': 'healthy', 'service': 'ugro-webhook'})
        
        @self.app.route('/', methods=['GET'])
        def index():
            """Index endpoint."""
            return jsonify({
                'service': 'UGRO Webhook Server',
                'version': '1.0.0',
                'endpoints': ['/webhook/github', '/health']
            })
    
    def _verify_signature(self, payload: bytes, signature: str) -> bool:
        """Verify GitHub webhook signature."""
        webhook_secret = self.config.get('webhook', {}).get('secret')
        if not webhook_secret:
            self.logger.warning("No webhook secret configured - skipping signature verification")
            return True
        
        expected_signature = hmac.new(
            webhook_secret.encode('utf-8'),
            payload,
            hashlib.sha256
        ).hexdigest()
        
        return hmac.compare_digest(f"sha256={expected_signature}", signature)
    
    def _handle_webhook(self):
        """Handle incoming webhook requests."""
        try:
            # Get signature from headers
            signature = request.headers.get('X-Hub-Signature-256')
            if not signature:
                self.logger.warning("No signature provided in webhook request")
                return jsonify({'error': 'No signature provided'}), 400
            
            # Get payload
            payload = request.data
            if not payload:
                self.logger.warning("No payload provided in webhook request")
                return jsonify({'error': 'No payload provided'}), 400
            
            # Verify signature
            if not self._verify_signature(payload, signature):
                self.logger.warning("Invalid webhook signature")
                return jsonify({'error': 'Invalid signature'}), 401
            
            # Parse payload
            event_data = json.loads(payload.decode('utf-8'))
            event_type = request.headers.get('X-GitHub-Event')
            
            self.logger.info(f"Received {event_type} event from GitHub")
            
            # Handle push events
            if event_type == 'push':
                return self._handle_push_event(event_data)
            else:
                self.logger.info(f"Ignoring {event_type} event")
                return jsonify({'status': 'ignored', 'event': event_type})
                
        except Exception as e:
            self.logger.error(f"Error handling webhook: {e}")
            return jsonify({'error': str(e)}), 500
    
    def _handle_push_event(self, event_data: Dict[str, Any]) -> tuple:
        """Handle push events and trigger synchronization."""
        try:
            repo_name = event_data.get('repository', {}).get('full_name')
            branch = event_data.get('ref', '').replace('refs/heads/', '')
            commits = event_data.get('commits', [])
            
            self.logger.info(f"Push to {repo_name}:{branch} with {len(commits)} commits")
            
            # Check if this is the configured repository
            webhook_config = self.config.get('webhook', {})
            expected_repo = webhook_config.get('repository')
            expected_branch = webhook_config.get('branch', 'main')
            
            if expected_repo and repo_name != expected_repo:
                self.logger.info(f"Ignoring push to {repo_name} (expected {expected_repo})")
                return jsonify({'status': 'ignored', 'reason': 'wrong_repository'})
            
            if branch != expected_branch:
                self.logger.info(f"Ignoring push to {branch} (expected {expected_branch})")
                return jsonify({'status': 'ignored', 'reason': 'wrong_branch'})
            
            # Trigger synchronization in background thread
            sync_thread = threading.Thread(
                target=self._trigger_synchronization,
                args=(event_data,),
                daemon=True
            )
            sync_thread.start()
            
            self.logger.info("Synchronization triggered in background")
            return jsonify({'status': 'triggered', 'message': 'Synchronization started'})
            
        except Exception as e:
            self.logger.error(f"Error handling push event: {e}")
            return jsonify({'error': str(e)}), 500
    
    def _trigger_synchronization(self, event_data: Dict[str, Any]):
        """Trigger synchronization to worker nodes."""
        try:
            # Get sync script path
            sync_script = self.config.get('paths', {}).get('scripts', 'scripts') + '/sync_to_workers.py'
            sync_script_path = Path(sync_script)
            
            if not sync_script_path.exists():
                self.logger.error(f"Sync script not found: {sync_script_path}")
                return
            
            # Run synchronization script
            self.logger.info("Running synchronization script")
            result = subprocess.run(
                ['python3', str(sync_script_path)],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode == 0:
                self.logger.info(f"Synchronization completed successfully: {result.stdout}")
            else:
                self.logger.error(f"Synchronization failed: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            self.logger.error("Synchronization timed out")
        except Exception as e:
            self.logger.error(f"Error during synchronization: {e}")
    
    def run(self, host: str = '0.0.0.0', port: int = 8080, debug: bool = False):
        """Run the webhook server."""
        self.logger.info(f"Starting webhook server on {host}:{port}")
        self.app.run(host=host, port=port, debug=debug)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='UGRO Webhook Server')
    parser.add_argument('--config', default='config/cluster.yaml', help='Configuration file path')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8080, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    try:
        server = WebhookServer(args.config)
        server.run(host=args.host, port=args.port, debug=args.debug)
    except Exception as e:
        print(f"Error starting webhook server: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())