# UGRO HPO: Quick Reference & Examples

## Command Reference

### 1. Basic Sweep (Default TPE + Sequential)
```bash
ugro sweep \
  --study-name basic-hpo \
  --search-space config/search.yaml \
  --n-trials 50
```

### 2. Parallel Async Sweep (Ray + Multivariate TPE)
```bash
ugro sweep \
  --study-name parallel-tpe \
  --search-space config/search.yaml \
  --n-trials 100 \
  --parallel-jobs 8 \
  --backend ray
```

### 3. Early Stopping with ASHA Scheduler
```bash
ugro sweep \
  --study-name asha-sweep \
  --search-space config/search.yaml \
  --n-trials 200 \
  --parallel-jobs 4 \
  --algorithm asha \
  --grace-period 3
```

### 4. Multi-GPU with Fractional Allocation
```bash
ugro sweep \
  --study-name multi-gpu \
  --search-space config/search.yaml \
  --n-trials 150 \
  --parallel-jobs 8 \
  --ray-gpu-per-trial 0.5  # 2 trials per GPU
  --ray-address auto
```

### 5. MLflow Tracking + W&B Monitoring
```bash
ugro sweep \
  --study-name tracked-hpo \
  --search-space config/search.yaml \
  --n-trials 100 \
  --tracking-uri mlflow://localhost:5000 \
  --wandb-project my-project \
  --export-best config/best.yaml
```

### 6. Resume Interrupted Study
```bash
ugro sweep \
  --study-name resumed-hpo \
  --search-space config/search.yaml \
  --n-trials 200 \
  --storage-backend sqlite:///studies/resumed-hpo.db \
  --resume
```

### 7. Multi-Objective Optimization
```bash
ugro sweep \
  --study-name multi-objective \
  --search-space config/multi_objective.yaml \
  --n-trials 100 \
  --algorithm tpe \
  --objectives accuracy,latency \
  --directions maximize,minimize \
  --weights 0.7,0.3
```

### 8. Population Based Training (Transfer Learning)
```bash
ugro sweep \
  --study-name pbt-transfer \
  --search-space config/search.yaml \
  --n-trials 100 \
  --algorithm pbt \
  --parallel-jobs 16 \
  --perturbation-interval 5
```

---

## Environment Variables

```bash
# Ray Cluster
export RAY_MEMORY=32000000000  # 32GB
export RAY_OBJECT_STORE_MEMORY=5000000000  # 5GB

# MLflow
export MLFLOW_TRACKING_URI="http://localhost:5000"
export MLFLOW_EXPERIMENT_NAME="hpo"

# Weights & Biases
export WANDB_API_KEY="your-key-here"
export WANDB_PROJECT="your-project"

# Logging
export PYTHONUNBUFFERED=1
```

---

## Config File Examples

### Example 1: LoRA Fine-tuning Search Space
```yaml
# config/llama_lora_hpo.yaml
parameters:
  lora_r:
    type: int
    min: 4
    max: 64
    step: 4
    default: 16

  lora_alpha:
    type: float
    min: 8.0
    max: 128.0
    log: true
    default: 32.0

  lora_dropout:
    type: float
    min: 0.0
    max: 0.3
    default: 0.1

  learning_rate:
    type: float
    min: 1e-5
    max: 1e-3
    log: true
    default: 5e-4

  batch_size:
    type: categorical
    choices: [8, 16, 32, 64]
    default: 16

  warmup_ratio:
    type: float
    min: 0.0
    max: 0.2
    default: 0.05

objectives:
  - name: eval_loss
    direction: minimize
    weight: 1.0

constraints:
  - "lora_alpha >= lora_r"
  - "learning_rate <= 0.001"
```

### Example 2: Vision Transformer Search Space
```yaml
# config/vit_hpo.yaml
parameters:
  learning_rate:
    type: float
    min: 1e-5
    max: 1e-2
    log: true

  weight_decay:
    type: float
    min: 1e-6
    max: 1e-2
    log: true

  dropout:
    type: float
    min: 0.0
    max: 0.5

  patch_size:
    type: categorical
    choices: [4, 8, 16, 32]

  attention_heads:
    type: categorical
    choices: [4, 8, 12, 16]

  hidden_dim:
    type: categorical
    choices: [384, 576, 768, 1024]

objectives:
  - name: top1_accuracy
    direction: maximize
    weight: 0.8
  - name: inference_time_ms
    direction: minimize
    weight: 0.2
```

### Example 3: NLP Model Search Space
```yaml
# config/nlp_hpo.yaml
parameters:
  learning_rate:
    type: float
    min: 2e-5
    max: 5e-4
    log: true

  num_train_epochs:
    type: int
    min: 2
    max: 5

  per_device_train_batch_size:
    type: categorical
    choices: [8, 16, 32]

  warmup_steps:
    type: int
    min: 0
    max: 1000
    step: 100

  weight_decay:
    type: float
    min: 0.0
    max: 0.1

objectives:
  - name: eval_f1
    direction: maximize

constraints:
  - "per_device_train_batch_size <= 32"
```

---

## Integration Patterns

### Pattern 1: With Hugging Face Trainer
```python
from ugro.hpo.objective import create_lora_training_objective
from ugro.hpo.optimizer import UGROOptimizer
from ugro.hpo.config import HPOConfig
import yaml

# Load config
with open("config/llama_lora_hpo.yaml") as f:
    search_space_config = yaml.safe_load(f)

# Create objective
objective = create_lora_training_objective(
    model_init_fn=lambda **kwargs: load_model_with_lora(**kwargs),
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    base_training_args={
        "output_dir": "./outputs",
        "num_train_epochs": 3,
    }
)

# Setup config
hpo_config = HPOConfig(
    study_name="llama-lora",
    search_space=search_space_config["parameters"],
    n_trials=100,
    parallel_jobs=4,
    algorithm="asha",
    storage_backend="sqlite:///studies/llama-lora.db",
)

# Optimize
optimizer = UGROOptimizer(hpo_config, objective)
results = optimizer.optimize()
```

### Pattern 2: With Custom Training Loop
```python
from ugro.hpo.optimizer import UGROOptimizer
from ugro.hpo.config import HPOConfig
from ray.air import session

def custom_objective(params):
    """Custom training loop objective"""
    model = initialize_model(**params)
    
    for epoch in range(10):
        train_loss = train_epoch(model, train_loader, params)
        val_loss = validate(model, val_loader)
        
        # Report intermediate results
        session.report({
            "eval_loss": val_loss,
            "train_loss": train_loss,
            "epoch": epoch,
        })
        
        if early_stop_check(val_loss):
            break
    
    return {"eval_loss": val_loss}

# Setup and run
config = HPOConfig(
    study_name="custom-loop",
    search_space=search_space,
    n_trials=50,
    parallel_jobs=4,
)

optimizer = UGROOptimizer(config, custom_objective)
results = optimizer.optimize()
```

### Pattern 3: Analysis & Export
```python
from ugro.hpo.analysis_advanced import analyze_hpo_results

# Run analysis
results = analyze_hpo_results(
    storage_backend="sqlite:///studies/llama-lora.db",
    study_name="llama-lora"
)

# Export best config
best_config = results["best_params"]
with open("config/best_lora.yaml", "w") as f:
    yaml.dump(best_config, f)

# Load and use
with open("config/best_lora.yaml") as f:
    best_params = yaml.safe_load(f)

model = load_model_with_lora(**best_params)
```

---

## Troubleshooting

### Issue: "Ray init timeout"
```bash
# Solution: Increase Ray startup timeout
export RAY_INIT_TIMEOUT=600
ugro sweep --ray-address auto ...
```

### Issue: "CUDA out of memory"
```bash
# Solution: Reduce batch size or use fractional GPU
ugro sweep \
  --ray-gpu-per-trial 0.25 \
  --search-space config/small_batch.yaml \
  ...
```

### Issue: "MLflow tracking slow"
```bash
# Solution: Use local SQLite storage + batch logging
ugro sweep \
  --storage-backend "sqlite:///studies/local.db" \
  --tracking-uri "file://./mlruns" \
  ...
```

### Issue: "Study not resuming"
```bash
# Solution: Use exact same database path
ugro sweep \
  --storage-backend "sqlite:///studies/my_study.db" \
  --resume \
  ...
```

---

## Performance Optimization Tips

1. **Reduce Sample Size** (default: 10)
   ```bash
   ugro sweep --sample-size 3 ...  # Trust TPE earlier
   ```

2. **Use ASHA Early Stopping**
   ```bash
   ugro sweep --algorithm asha --grace-period 2 ...
   ```

3. **Increase Parallel Jobs**
   ```bash
   ugro sweep --parallel-jobs 16 ...  # If GPU available
   ```

4. **Profile Objective Function**
   ```python
   import cProfile
   cProfile.run("objective(test_params)", sort="cumtime")
   ```

5. **Cache Expensive Computations**
   ```python
   @lru_cache(maxsize=1)
   def load_model():
       return AutoModel.from_pretrained(...)
   ```

---

## Monitoring During Sweep

### Via MLflow UI
```bash
mlflow ui --host 0.0.0.0 --port 5000
# Navigate: http://localhost:5000
```

### Via W&B Dashboard
```bash
# Automatic: view at https://wandb.ai/username/project
```

### Via Ray Dashboard
```bash
# Automatic: http://localhost:8265 (if Ray head on localhost)
```

### Via Logs
```bash
tail -f ray_results/llama-lora*/worker_*.log
```

---

## Export & Deployment

### Export Best Config
```bash
ugro sweep \
  --export-best config/best.yaml \
  --save-trials-csv results/trials.csv \
  ...
```

### Use Best Config for Production
```python
import yaml

with open("config/best.yaml") as f:
    best_params = yaml.safe_load(f)

# Load model with best hyperparameters
model = load_model_with_lora(**best_params)
```

### Share Results
```bash
# Archive for team
tar -czf hpo_results.tar.gz \
  config/best.yaml \
  results/ \
  hpo_analysis_*.png

# Upload to S3
aws s3 cp hpo_results.tar.gz s3://my-bucket/
```

---

## Typical HPO Workflows

### Workflow 1: Fast Baseline (30 min)
```bash
ugro sweep \
  --study-name baseline \
  --search-space config/quick_search.yaml \
  --n-trials 20 \
  --parallel-jobs 4
```

### Workflow 2: Production HPO (4-8 hours)
```bash
ugro sweep \
  --study-name production \
  --search-space config/full_search.yaml \
  --n-trials 100 \
  --parallel-jobs 8 \
  --algorithm asha \
  --tracking-uri mlflow://localhost:5000 \
  --wandb-project prod-hpo
```

### Workflow 3: Research Deep Dive (24-48 hours)
```bash
ugro sweep \
  --study-name research \
  --search-space config/extensive_search.yaml \
  --n-trials 500 \
  --parallel-jobs 16 \
  --algorithm bohb \
  --storage-backend postgresql://user:pass@host/ugro_db
```

---

## Contact & Support

For issues:
- Check logs: `ray_results/*/worker_*.log`
- MLflow: `mlflow://localhost:5000`
- Community: GitHub Discussions
