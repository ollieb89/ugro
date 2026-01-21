# W&B Integration Setup Guide

## Overview
UGRO HPO supports Weights & Biases (W&B) for experiment tracking alongside MLflow. This provides dual tracking capabilities with real-time visualization in the W&B dashboard.

## Prerequisites
1. Install W&B dependency (already included in pixi.toml):
   ```bash
   pixi add wandb
   ```

2. Authenticate with W&B:
   ```bash
   # Option 1: Login interactively
   wandb login
   
   # Option 2: Set API key environment variable
   export WANDB_API_KEY=your_api_key_here
   ```

## Usage

### Basic HPO with W&B
```bash
ugro hpo sweep \
  --study-name llama-lora-hpo \
  --search-space config/llama_lora_hpo.yaml \
  --n-trials 100 \
  --parallel-jobs 8 \
  --wandb-project my-lora-experiments \
  --model meta-llama/Llama-2-7b-hf \
  --dataset wikitext
```

### Dual Tracking (MLflow + W&B)
```bash
ugro hpo sweep \
  --study-name dual-tracking-test \
  --search-space config/llama_lora_hpo.yaml \
  --n-trials 50 \
  --tracking-uri http://localhost:5000 \
  --wandb-project dual-tracking-demo \
  --model unsloth/tinyllama-bnb-4bit
```

## What Gets Logged

### Objective Function Level
- **Parameters**: All hyperparameters from search space
- **Metrics**: eval_loss, eval_perplexity, train_loss
- **Configuration**: model_id, dataset_name, max_steps
- **Constraint Violations**: Logged if constraints are violated

### Ray Tune Level (via WandbLoggerCallback)
- **Trial-level metrics**: Real-time updates during training
- **System metrics**: GPU/CPU usage (if available)
- **Trial metadata**: Trial ID, configuration

## W&B Dashboard Features
1. **Parallel Coordinates**: Visualize parameter correlations
2. **Hyperparameter Importance**: Automatic importance analysis
3. **Trial Comparison**: Side-by-side metric comparison
4. **Real-time Updates**: Live monitoring during HPO runs

## Configuration Options
- `WANDB_PROJECT`: Project name (required)
- `WANDB_ENTITY`: Team or username (optional, defaults to personal)
- `WANDB_API_KEY`: Authentication key (alternative to wandb login)
- `WANDB_MODE`: Set to "offline" to disable sync

## Security Best Practices

### API Key Management
1. **Never hardcode API keys** in your code or configuration files
2. **Use environment variables** for API keys:
   ```bash
   export WANDB_API_KEY=your_api_key_here
   ```
3. **Use .env files** for local development (add to .gitignore):
   ```bash
   echo "WANDB_API_KEY=your_api_key_here" >> .env
   echo ".env" >> .gitignore
   ```
4. **Rotate API keys regularly** via W&B dashboard
5. **Use scoped keys** with minimal required permissions

### Project Name Security
- Project names are validated to prevent injection attacks
- Only alphanumeric characters, hyphens, and underscores allowed
- Maximum length: 128 characters
- Examples of valid names:
  - `my-lora-experiments`
  - `project_2024`
  - `llama-fine-tuning`

### Data Privacy Considerations
1. **Review what you log** - avoid logging sensitive data:
   ```python
   # Good - log only hyperparameters and metrics
   tracker.log_params({"learning_rate": 0.001, "batch_size": 32})
   
   # Bad - don't log sensitive information
   tracker.log_params({"api_key": "secret", "password": "secret"})
   ```

2. **Use private W&B projects** for sensitive experiments
3. **Enable data retention policies** in W&B settings
4. **Consider on-premise W&B** for highly sensitive data

### Network Security
1. **Use HTTPS** - W&B automatically encrypts data in transit
2. **VPNs/Firewalls** - Ensure W&B domains are accessible:
   - `api.wandb.ai`
   - `wandb.ai`
   - `cdn.wandb.ai`

3. **Proxy support** - Configure if behind corporate firewall:
   ```bash
   export HTTPS_PROXY=http://proxy.company.com:8080
   export HTTP_PROXY=http://proxy.company.com:8080
   ```

### Audit and Compliance
1. **Track who has access** to your W&B projects
2. **Enable audit logs** in W&B enterprise settings
3. **Regular security reviews** of logged data
4. **GDPR/CCPA compliance** - W&B supports data deletion requests

### Validation Features
UGRO includes automatic validation for:
- API key format (20+ alphanumeric characters)
- Project names (prevents injection attacks)
- Input sanitization before sending to W&B

Example of validation in action:
```bash
# This will be rejected
ugro hpo sweep --wandb-project "invalid project name!"

# Output: ValueError: Invalid project name: invalid project name!
```

## Troubleshooting

### Issue: "Failed to initialize W&B"
- Ensure you're authenticated: `wandb login`
- Check API key: `echo $WANDB_API_KEY`
- Verify project name is valid

### Issue: No metrics appearing in W&B
- Check that `--wandb-project` is set
- Verify internet connection for W&B sync
- Check W&B dashboard for correct project

### Issue: Ray Tune W&B integration not available
- Ensure ray[tune] is installed with W&B extras
- Update Ray: `pixi update ray`

## Example W&B Dashboard URL
After running HPO with `--wandb-project my-experiments`, view results at:
```
https://wandb.ai/<your-username>/my-experiments
```

## Notes
- W&B runs are automatically tagged with trial IDs
- Each HPO study creates a W&B group for organization
- Constraint violations are logged as tags and metrics
- W&B integration is optional - HPO works without it
