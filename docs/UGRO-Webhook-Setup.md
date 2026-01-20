# UGRO Webhook Setup Guide

This guide explains how to set up automatic synchronization from the GPU master to worker nodes when changes are pushed to GitHub.

## Overview

The webhook system provides immediate synchronization when you push changes to GitHub:

1. **GitHub Webhook**: Sends push events to the master node
2. **Webhook Server**: Receives events and triggers synchronization
3. **Sync Script**: Uses rsync to efficiently update worker nodes

## Quick Setup

### 1. Configure Repository Settings

Edit `config/cluster.yaml` and set your GitHub repository:

```yaml
webhook:
  repository: "your-username/ugro"  # Your GitHub repository
  branch: "main"
  secret: "generate-secure-random-string-here"  # See step 2
```

### 2. Generate Webhook Secret

Generate a secure random string for the webhook secret:

```bash
# Generate a secure random string
openssl rand -hex 32
# Or use python
python3 -c "import secrets; print(secrets.token_hex(32))"
```

Set this as `webhook.secret` in your configuration.

### 3. Install Webhook Service

Run the setup script:

```bash
# Install systemd service
sudo ./scripts/setup_webhook_service.sh
```

Or manually install:

```bash
sudo cp systemd/ugro-webhook.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable ugro-webhook.service
sudo systemctl start ugro-webhook.service
```

### 4. Configure GitHub Webhook

1. Go to your GitHub repository
2. Navigate to **Settings** → **Webhooks**
3. Click **Add webhook**
4. Configure:
   - **Payload URL**: `http://<master-ip>:8099/webhook/github`
   - **Content type**: `application/json`
   - **Secret**: Same string you set in step 2
   - **Events**: Select "Just the `push` event"
5. Click **Add webhook**

### 5. Test the Setup

#### Test Webhook Server

```bash
# Check if webhook server is running
curl http://localhost:8099/health

# Should return:
# {"status": "healthy", "service": "ugro-webhook"}
```

#### Test Synchronization

```bash
# Test sync script manually
python3 scripts/sync_to_workers.py --dry-run

# List workers
python3 scripts/sync_to_workers.py --list

# Verify worker paths
python3 scripts/sync_to_workers.py --verify

# Sync to all workers
python3 scripts/sync_to_workers.py

# Sync to specific worker
python3 scripts/sync_to_workers.py --worker gpu1
```

## Detailed Configuration

### Webhook Configuration Options

```yaml
webhook:
  # GitHub repository to monitor (format: owner/repo)
  repository: "username/ugro"
  
  # Branch to monitor
  branch: "main"
  
  # Webhook secret for security
  secret: "your-secure-random-string"
  
  # Webhook server configuration
  server:
    host: "0.0.0.0"  # Listen on all interfaces
    port: 8099       # Port for webhook endpoint
    
  # Synchronization settings
  sync:
    method: "rsync"  # "rsync" (master pushes) or "pull" (workers pull)
    timeout: 600     # Timeout in seconds
    
    # Files/directories to exclude
    exclude:
      - ".git"
      - "__pycache__"
      - "*.pyc"
      - ".pytest_cache"
      - "logs/"
      - ".venv"
      - "node_modules"
      - ".DS_Store"
      - "*.log"
```

### Worker Configuration

Each worker in `config/cluster.yaml` should have:

```yaml
workers:
  - name: "gpu1"
    hostname: "gpu1"        # SSH hostname
    ip: "192.168.1.101"     # IP address
    user: "ob"              # SSH username
    ssh_port: 22            # SSH port
    
    paths:
      project: "/home/ob/Development/Tools/ugro"  # Project path on worker
```

## How It Works

### Webhook Flow

1. **Push to GitHub**: You push changes to your repository
2. **GitHub Notification**: GitHub sends a POST request to your webhook endpoint
3. **Signature Verification**: The webhook server verifies the request signature
4. **Event Processing**: The server validates the repository and branch
5. **Sync Trigger**: The sync script is executed in a background thread

### Synchronization Process

The sync script (`scripts/sync_to_workers.py`) performs:

1. **SSH Connection Test**: Verifies connectivity to each worker
2. **Path Verification**: Ensures project directories exist on workers
3. **Rsync Execution**: Efficiently transfers only changed files
4. **Error Handling**: Logs any failures and continues with other workers

### Rsync Benefits

- **Incremental**: Only transfers changed files
- **Compression**: Reduces network usage
- **Deletion**: Removes files deleted from master
- **Exclusions**: Skips unnecessary files (cache, logs, etc.)

## Security Considerations

### Webhook Security

- **Secret Verification**: Always use a strong webhook secret
- **HTTPS**: Use HTTPS for webhook URLs in production
- **IP Whitelisting**: Consider restricting webhook source IPs

### SSH Security

- **Key Authentication**: Use SSH keys instead of passwords
- **Limited Users**: Use dedicated service accounts where possible
- **Network Security**: Ensure proper firewall rules

## Troubleshooting

### Common Issues

#### Webhook Server Not Starting

```bash
# Check service status
sudo systemctl status ugro-webhook.service

# View logs
sudo journalctl -u ugro-webhook.service -f

# Check configuration
python3 -c "from src.ugro.webhook_server import WebhookServer; WebhookServer()"
```

#### SSH Connection Failures

```bash
# Test SSH connection manually
ssh ob@gpu1 "echo 'Connection test'"

# Check SSH keys
ssh-add -l

# Debug SSH connection
ssh -v ob@gpu1 "echo 'Debug connection'"
```

#### Rsync Failures

```bash
# Test rsync manually
rsync -avz --exclude '.git' /home/ollie/Development/Tools/ugro/ ob@gpu1:/home/ob/Development/Tools/ugro/

# Check permissions on workers
ssh ob@gpu1 "ls -la /home/ob/Development/Tools/"
```

#### Webhook Not Triggering

1. **Check GitHub webhook status** in repository settings
2. **Verify webhook URL** is accessible from GitHub
3. **Check webhook secret** matches configuration
4. **Review webhook delivery logs** in GitHub

### Debug Mode

Run the webhook server in debug mode:

```bash
# Stop systemd service
sudo systemctl stop ugro-webhook.service

# Run manually in debug mode
python3 -m ugro.webhook_server --config config/cluster.yaml --debug

# Or use the script directly
python3 src/ugro/webhook_server.py --config config/cluster.yaml --debug
```

## Monitoring

### Service Status

```bash
# Check if service is running
sudo systemctl is-active ugro-webhook.service

# Check service status
sudo systemctl status ugro-webhook.service

# View recent logs
sudo journalctl -u ugro-webhook.service --since "1 hour ago"
```

### Synchronization Logs

```bash
# View sync logs
tail -f logs/sync.log

# View webhook logs
tail -f logs/webhook.log
```

### Health Checks

```bash
# Webhook server health
curl http://localhost:8099/health

# Test webhook endpoint
curl -X POST http://localhost:8099/webhook/github \
  -H "Content-Type: application/json" \
  -H "X-GitHub-Event: ping" \
  -d '{"zen": "non-blocking"}'
```

## Advanced Configuration

### Custom Webhook Endpoint

You can customize the webhook endpoint by modifying the Flask routes in `src/ugro/webhook_server.py`.

### Alternative Sync Methods

#### Pull Method (Workers Pull from GitHub)

Change the sync method in configuration:

```yaml
webhook:
  sync:
    method: "pull"  # Workers pull from GitHub instead of master pushing
```

This requires workers to have GitHub access and the sync script would be modified to pull updates on each worker.

#### GitHub Actions Integration

You could also use GitHub Actions to trigger synchronization:

```yaml
# .github/workflows/sync-workers.yml
name: Sync to Workers
on:
  push:
    branches: [main]
jobs:
  sync:
    runs-on: ubuntu-latest
    steps:
      - name: Sync to workers
        run: |
          # SSH to workers and pull changes
          ssh ob@gpu1 "cd /path/to/ugro && git pull"
          ssh ollie@gpu2 "cd /path/to/ugro && git pull"
```

## Performance Considerations

- **Network Bandwidth**: Rsync compression reduces bandwidth usage
- **Large Files**: Consider excluding large binary files or datasets
- **Frequency**: Webhooks trigger on every push - consider batch updates for frequent changes
- **Timeouts**: Adjust timeouts based on repository size and network speed

## Maintenance

### Regular Tasks

1. **Monitor logs** for errors or performance issues
2. **Update webhook secret** periodically for security
3. **Review exclusions** to ensure unnecessary files aren't synced
4. **Test connectivity** to workers regularly
5. **Backup configuration** before making changes

### Updates

When updating UGRO:

1. **Stop webhook service**: `sudo systemctl stop ugro-webhook.service`
2. **Update code**: Pull latest changes
3. **Restart service**: `sudo systemctl start ugro-webhook.service`
4. **Verify functionality**: Test webhook and sync operations

## Support

For issues or questions:

1. Check the troubleshooting section above
2. Review logs in `logs/` directory
3. Test components individually
4. Verify configuration matches your environment