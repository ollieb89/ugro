#!/bin/bash
"""
Setup script for UGRO webhook systemd service
"""

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SERVICE_FILE="$PROJECT_ROOT/systemd/ugro-webhook.service"
SYSTEMD_SERVICE="/etc/systemd/system/ugro-webhook.service"

echo "Setting up UGRO webhook service..."

# Check if running as root for systemd operations
if [[ $EUID -eq 0 ]]; then
    echo "Running as root - installing systemd service..."
    
    # Copy service file
    cp "$SERVICE_FILE" "$SYSTEMD_SERVICE"
    
    # Reload systemd
    systemctl daemon-reload
    
    # Enable and start service
    systemctl enable ugro-webhook.service
    systemctl start ugro-webhook.service
    
    echo "Service installed and started!"
    echo "Check status with: systemctl status ugro-webhook.service"
    echo "View logs with: journalctl -u ugro-webhook.service -f"
    
else
    echo "Not running as root - showing manual installation steps:"
    echo ""
    echo "To install the systemd service, run:"
    echo "sudo cp '$SERVICE_FILE' '$SYSTEMD_SERVICE'"
    echo "sudo systemctl daemon-reload"
    echo "sudo systemctl enable ugro-webhook.service"
    echo "sudo systemctl start ugro-webhook.service"
    echo ""
    echo "To check status:"
    echo "systemctl status ugro-webhook.service"
    echo ""
    echo "To view logs:"
    echo "journalctl -u ugro-webhook.service -f"
fi

echo ""
echo "Configuration required:"
echo "1. Edit config/cluster.yaml and set:"
echo "   webhook.repository: 'your-username/ugro'"
echo "   webhook.secret: 'your-secure-random-string'"
echo ""
echo "2. Set up GitHub webhook:"
echo "   - URL: http://<master-ip>:8080/webhook/github"
echo "   - Content type: application/json"
echo "   - Secret: same as webhook.secret above"
echo "   - Events: Push"