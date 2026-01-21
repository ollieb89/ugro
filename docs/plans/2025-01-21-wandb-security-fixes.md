# W&B Security and Resource Management Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement security fixes and resource management improvements for the W&B integration in UGRO HPO system

**Architecture:** Phase-based approach addressing security vulnerabilities, resource cleanup, architectural refactoring, and comprehensive testing

**Tech Stack:** Python, W&B SDK, Ray Tune, MLflow, pytest, mocking frameworks

---

## Phase 1: Security Fixes (Priority: High)

### Task 1.1: Create Security Utilities Module

**Files:**
- Create: `src/ugro/hpo/security.py`

**Step 1: Write the failing test**

```python
# tests/hpo/test_security.py
import pytest
from ugro.hpo.security import validate_wandb_api_key, mask_api_key, validate_project_name

def test_validate_wandb_api_key_valid():
    # Valid W&B API key format
    valid_key = "wandb_api_key_1234567890abcdef"
    assert validate_wandb_api_key(valid_key) == True

def test_validate_wandb_api_key_invalid():
    # Invalid formats
    assert validate_wandb_api_key("") == False
    assert validate_wandb_api_key("short") == False
    assert validate_wandb_api_key(None) == False

def test_mask_api_key():
    key = "wandb_api_key_1234567890abcdef"
    masked = mask_api_key(key)
    assert masked == "wandb_api_key_****cdef"
    assert len(masked) == len(key)

def test_validate_project_name():
    assert validate_project_name("valid-project") == True
    assert validate_project_name("invalid project!") == False
    assert validate_project_name("") == False
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/hpo/test_security.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'ugro.hpo.security'"

**Step 3: Write minimal implementation**

```python
# src/ugro/hpo/security.py
import re
from typing import Optional

def validate_wandb_api_key(api_key: Optional[str]) -> bool:
    """Validate W&B API key format."""
    if not api_key or not isinstance(api_key, str):
        return False
    # W&B API keys are typically 40 characters long and alphanumeric
    return bool(re.match(r'^[a-zA-Z0-9_]{20,}$', api_key))

def mask_api_key(api_key: str) -> str:
    """Mask API key for logging, showing first and last 4 chars."""
    if not api_key or len(api_key) < 8:
        return "***"
    return f"{api_key[:12]}{'*' * (len(api_key) - 16)}{api_key[-4:]}"

def validate_project_name(project_name: Optional[str]) -> bool:
    """Validate W&B project name."""
    if not project_name or not isinstance(project_name, str):
        return False
    # Project names should be alphanumeric with hyphens/underscores
    return bool(re.match(r'^[a-zA-Z0-9_-]+$', project_name)) and 1 <= len(project_name) <= 128
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/hpo/test_security.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/ugro/hpo/security.py tests/hpo/test_security.py
git commit -m "feat: add security utilities for W&B API key and project validation"
```

### Task 1.2: Update CLI with Security Validation

**Files:**
- Modify: `src/ugro/cli.py:527-727`

**Step 1: Write the failing test**

```python
# tests/test_cli_wandb_security.py
import pytest
from click.testing import CliRunner
from ugro.cli import hpo

def test_wandb_project_validation():
    runner = CliRunner()
    
    # Test invalid project name
    result = runner.invoke(hpo, [
        'sweep',
        '--wandb-project', 'invalid project!',
        '--dry-run'
    ])
    assert result.exit_code != 0
    assert 'Invalid project name' in result.output
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_cli_wandb_security.py -v`
Expected: FAIL with validation not implemented

**Step 3: Update CLI implementation**

```python
# In src/ugro/cli.py, add import near top
from .hpo.security import validate_wandb_api_key, mask_api_key, validate_project_name

# Update the hpo sweep command (around line 560-580)
@sweep.command()
@click.option('--wandb-project', help='W&B project name')
# ... other options ...
def sweep_cmd(search_space, study_name, wandb_project, **kwargs):
    """Run hyperparameter optimization sweep."""
    
    # Validate W&B project name if provided
    if wandb_project and not validate_project_name(wandb_project):
        raise click.BadParameter(
            f"Invalid W&B project name '{wandb_project}'. "
            "Project names must be alphanumeric with hyphens/underscores, "
            "1-128 characters long."
        )
    
    # Mask API key in logs if present
    api_key = os.environ.get('WANDB_API_KEY')
    if api_key and wandb_project:
        logger.info(f"Using W&B project: {wandb_project}, API key: {mask_api_key(api_key)}")
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_cli_wandb_security.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/ugro/cli.py tests/test_cli_wandb_security.py
git commit -m "feat: add W&B project name validation to CLI"
```

---

## Phase 2: Resource Management (Priority: High)

### Task 2.1: Implement Proper Cleanup in Objective Function

**Files:**
- Modify: `src/ugro/hpo/objective.py`

**Step 1: Write the failing test**

```python
# tests/hpo/test_objective_cleanup.py
import pytest
from unittest.mock import patch, MagicMock
from ugro.hpo.objective import LoRAFinetuningObjective

def test_wandb_cleanup_on_exception():
    objective = LoRAFinetuningObjective(
        model_name="test/model",
        dataset_path="test.csv",
        use_wandb=True,
        wandb_project="test-project"
    )
    
    with patch('wandb.init') as mock_init, \
         patch('wandb.finish') as mock_finish, \
         patch('ugro.hpo.objective.LoraFinetuningObjective._train_model', side_effect=Exception("Training failed")):
        
        with pytest.raises(Exception):
            objective({})
        
        # Verify wandb.finish() is called even on exception
        mock_finish.assert_called_once()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/hpo/test_objective_cleanup.py -v`
Expected: FAIL - wandb.finish() not called in exception path

**Step 3: Update objective function with proper cleanup**

```python
# In src/ugro/hpo/objective.py, update the __call__ method
def __call__(self, config: Dict[str, Any]) -> Dict[str, float]:
    """Execute training with given config."""
    
    wandb_run = None
    try:
        # Existing setup code...
        
        # Initialize W&B if requested
        if self.use_wandb and self.wandb_project:
            wandb_run = self._get_wandb_context()
        
        # Existing training code...
        
    except Exception as e:
        # Log the error
        logger.error(f"Training failed: {e}")
        raise
    finally:
        # Ensure W&B run is properly finished
        if wandb_run is not None:
            try:
                import wandb
                if wandb.run is not None:
                    wandb.finish()
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup W&B run: {cleanup_error}")
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/hpo/test_objective_cleanup.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/ugro/hpo/objective.py tests/hpo/test_objective_cleanup.py
git commit -m "fix: ensure proper W&B cleanup in objective function"
```

### Task 2.2: Create W&B Context Manager

**Files:**
- Create: `src/ugro/hpo/tracking/wandb_context.py`

**Step 1: Write the failing test**

```python
# tests/hpo/test_wandb_context.py
import pytest
from ugro.hpo.tracking.wandb_context import WandbContextManager

def test_wandb_context_manager_cleanup():
    with patch('wandb.init') as mock_init, \
         patch('wandb.finish') as mock_finish:
        
        with WandbContextManager(project="test", config={"key": "value"}):
            pass
        
        mock_init.assert_called_once()
        mock_finish.assert_called_once()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/hpo/test_wandb_context.py -v`
Expected: FAIL - module doesn't exist

**Step 3: Implement context manager**

```python
# src/ugro/hpo/tracking/wandb_context.py
import wandb
from typing import Dict, Any, Optional
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)

@contextmanager
def WandbContextManager(project: str, config: Optional[Dict[str, Any]] = None, **kwargs):
    """Context manager for W&B runs with guaranteed cleanup."""
    run = None
    try:
        if project:
            run = wandb.init(project=project, config=config or {}, **kwargs)
        yield run
    except Exception as e:
        logger.error(f"W&B context error: {e}")
        raise
    finally:
        if run is not None or wandb.run is not None:
            try:
                wandb.finish()
            except Exception as e:
                logger.warning(f"Failed to finish W&B run: {e}")
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/hpo/test_wandb_context.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/ugro/hpo/tracking/wandb_context.py tests/hpo/test_wandb_context.py
git commit -m "feat: add W&B context manager for proper resource cleanup"
```

---

## Phase 3: Architecture - Abstraction Layer (Priority: Medium)

### Task 3.1: Create Tracking Interface

**Files:**
- Create: `src/ugro/hpo/tracking/__init__.py`
- Create: `src/ugro/hpo/tracking/interface.py`

**Step 1: Write the failing test**

```python
# tests/hpo/test_tracking_interface.py
import pytest
from ugro.hpo.tracking.interface import TrackingInterface, TrackingSystem

def test_tracking_interface():
    # Test that interface defines required methods
    assert hasattr(TrackingInterface, 'log_params'))
    assert hasattr(TrackingInterface, 'log_metrics'))
    assert hasattr(TrackingInterface, 'finish'))
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/hpo/test_tracking_interface.py -v`
Expected: FAIL - interface doesn't exist

**Step 3: Create interface**

```python
# src/ugro/hpo/tracking/interface.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from enum import Enum

class TrackingSystem(Enum):
    MLFLOW = "mlflow"
    WANDB = "wandb"
    BOTH = "both"

class TrackingInterface(ABC):
    """Abstract interface for tracking systems."""
    
    @abstractmethod
    def log_params(self, params: Dict[str, Any]) -> None:
        """Log parameters."""
        pass
    
    @abstractmethod
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics."""
        pass
    
    @abstractmethod
    def set_tags(self, tags: Dict[str, str]) -> None:
        """Set tags."""
        pass
    
    @abstractmethod
    def finish(self) -> None:
        """Finish tracking session."""
        pass
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/hpo/test_tracking_interface.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/ugro/hpo/tracking/__init__.py src/ugro/hpo/tracking/interface.py tests/hpo/test_tracking_interface.py
git commit -m "feat: create tracking interface abstraction"
```

### Task 3.2: Implement W&B Tracker

**Files:**
- Create: `src/ugro/hpo/tracking/wandb_tracker.py`

**Step 1: Write the failing test**

```python
# tests/hpo/test_wandb_tracker.py
import pytest
from unittest.mock import patch
from ugro.hpo.tracking.wandb_tracker import WandbTracker

def test_wandb_tracker_logs():
    with patch('wandb.init'), patch('wandb.log') as mock_log:
        tracker = WandbTracker(project="test")
        tracker.log_metrics({"loss": 0.5})
        mock_log.assert_called_once()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/hpo/test_wandb_tracker.py -v`
Expected: FAIL - tracker doesn't exist

**Step 3: Implement W&B tracker**

```python
# src/ugro/hpo/tracking/wandb_tracker.py
import wandb
from typing import Dict, Any, Optional
from .interface import TrackingInterface
from ..security import validate_wandb_api_key, mask_api_key
import logging

logger = logging.getLogger(__name__)

class WandbTracker(TrackingInterface):
    """W&B implementation of tracking interface."""
    
    def __init__(self, project: str, config: Optional[Dict[str, Any]] = None, **kwargs):
        # Validate inputs
        if not validate_wandb_api_key(wandb.api.api_key):
            raise ValueError("Invalid W&B API key")
        
        self.project = project
        self.config = config or {}
        self.kwargs = kwargs
        self._run = None
    
    def _ensure_run(self):
        """Ensure W&B run is initialized."""
        if self._run is None and wandb.run is None:
            self._run = wandb.init(
                project=self.project,
                config=self.config,
                **self.kwargs
            )
    
    def log_params(self, params: Dict[str, Any]) -> None:
        """Log parameters to W&B."""
        self._ensure_run()
        wandb.config.update(params)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics to W&B."""
        self._ensure_run()
        wandb.log(metrics, step=step)
    
    def set_tags(self, tags: Dict[str, str]) -> None:
        """Set tags in W&B."""
        self._ensure_run()
        wandb.run.tags = tags
    
    def finish(self) -> None:
        """Finish W&B run."""
        if wandb.run is not None:
            wandb.finish()
        self._run = None
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/hpo/test_wandb_tracker.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/ugro/hpo/tracking/wandb_tracker.py tests/hpo/test_wandb_tracker.py
git commit -m "feat: implement W&B tracker with security validation"
```

### Task 3.3: Update Objective to Use Abstraction

**Files:**
- Modify: `src/ugro/hpo/objective.py`

**Step 1: Write the failing test**

```python
# tests/hpo/test_objective_abstraction.py
import pytest
from unittest.mock import patch
from ugro.hpo.objective import LoRAFinetuningObjective

def test_objective_uses_tracking_interface():
    objective = LoRAFinetuningObjective(
        model_name="test/model",
        dataset_path="test.csv",
        use_wandb=True,
        wandb_project="test-project"
    )
    
    # Should have tracking interface, not direct wandb
    assert hasattr(objective, '_tracker')
    assert not hasattr(objective, '_wandb_run')
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/hpo/test_objective_abstraction.py -v`
Expected: FAIL - still using direct wandb

**Step 3: Refactor objective function**

```python
# In src/ugro/hpo/objective.py, add imports
from .tracking.interface import TrackingInterface
from .tracking.wandb_tracker import WandbTracker
from .tracking.mlflow_tracker import MlflowTracker  # Create if needed

# Update LoRAFinetuningObjective class
class LoRAFinetuningObjective:
    def __init__(self, 
                 model_name: str,
                 dataset_path: str,
                 use_wandb: bool = False,
                 wandb_project: Optional[str] = None,
                 **kwargs):
        # Existing init code...
        
        # Initialize tracking
        self._trackers = []
        if use_wandb and wandb_project:
            self._trackers.append(WandbTracker(project=wandb_project))
        
        # Always use MLflow
        self._trackers.append(MlflowTracker())
    
    def _log_to_all_trackers(self, method_name: str, *args, **kwargs):
        """Call method on all trackers."""
        for tracker in self._trackers:
            getattr(tracker, method_name)(*args, **kwargs)
    
    def __call__(self, config: Dict[str, Any]) -> Dict[str, float]:
        """Execute training with given config."""
        try:
            # Log parameters
            self._log_to_all_trackers('log_params', config)
            
            # Training logic...
            
            # Log metrics
            metrics = {"eval_loss": loss, "eval_perplexity": perplexity}
            self._log_to_all_trackers('log_metrics', metrics)
            
            return metrics
            
        finally:
            # Cleanup all trackers
            self._log_to_all_trackers('finish')
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/hpo/test_objective_abstraction.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/ugro/hpo/objective.py tests/hpo/test_objective_abstraction.py
git commit -m "refactor: update objective to use tracking abstraction layer"
```

---

## Phase 4: Testing and Verification (Priority: Medium)

### Task 4.1: Fix Test Imports

**Files:**
- Modify: `test_wandb_integration.py`

**Step 1: Identify the import issue**

Run: `python -c "from test_wandb_integration import main"` to see the error

**Step 2: Fix the import**

```python
# In test_wandb_integration.py, fix the import
# From:
from ugro.hpo.search_space import ParameterDefinition
# To:
from ugro.hpo.search_space import Parameter  # or correct class name
```

**Step 3: Verify the fix**

Run: `python test_wandb_integration.py`

**Step 4: Commit**

```bash
git add test_wandb_integration.py
git commit -m "fix: correct import in W&B integration test"
```

### Task 4.2: Add Comprehensive W&B Tests

**Files:**
- Create: `tests/hpo/test_wandb_integration_comprehensive.py`

**Step 1: Write comprehensive tests**

```python
# tests/hpo/test_wandb_integration_comprehensive.py
import pytest
from unittest.mock import patch, MagicMock
import os

from ugro.hpo.objective import LoRAFinetuningObjective
from ugro.hpo.security import validate_wandb_api_key, mask_api_key

def test_wandb_integration_with_valid_api_key():
    """Test W&B integration works with valid API key."""
    with patch.dict(os.environ, {'WANDB_API_KEY': 'wandb_test_key_1234567890abcdef'}):
        with patch('wandb.init') as mock_init, patch('wandb.log') as mock_log:
            objective = LoRAFinetuningObjective(
                model_name="test/model",
                dataset_path="test.csv",
                use_wandb=True,
                wandb_project="test-project"
            )
            
            result = objective({"learning_rate": 0.001})
            
            assert "eval_loss" in result
            mock_init.assert_called_once()
            mock_log.assert_called()

def test_wandb_api_key_masking():
    """Test API key is properly masked in logs."""
    api_key = "wandb_api_key_1234567890abcdef"
    masked = mask_api_key(api_key)
    assert "1234567890abcd" not in masked
    assert masked.endswith("cdef")

def test_wandb_fails_without_api_key():
    """Test W&B integration fails gracefully without API key."""
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError, match="Invalid W&B API key"):
            LoRAFinetuningObjective(
                model_name="test/model",
                dataset_path="test.csv",
                use_wandb=True,
                wandb_project="test-project"
            )
```

**Step 2: Run tests**

Run: `pytest tests/hpo/test_wandb_integration_comprehensive.py -v`

**Step 3: Commit**

```bash
git add tests/hpo/test_wandb_integration_comprehensive.py
git commit -m "test: add comprehensive W&B integration tests"
```

### Task 4.3: Integration Test with Ray Tune

**Files:**
- Create: `tests/hpo/test_ray_tune_wandb.py`

**Step 1: Write Ray Tune integration test**

```python
# tests/hpo/test_ray_tune_wandb.py
import pytest
from unittest.mock import patch
from ray import tune
from ugro.hpo.optimizer import HPOOptimizer

def test_ray_tune_wandb_logger_callback():
    """Test WandbLoggerCallback is properly configured."""
    config = {
        "search_space": {
            "learning_rate": {"type": "float", "min": 1e-5, "max": 1e-3}
        },
        "objectives": [{"name": "eval_loss", "direction": "minimize"}]
    }
    
    with patch('ray.tune.Tuner') as mock_tuner:
        optimizer = HPOOptimizer(config)
        optimizer.optimize(
            model_name="test/model",
            dataset_path="test.csv",
            wandb_project="test-project"
        )
        
        # Verify WandbLoggerCallback was added
        call_args = mock_tuner.call_args
        callbacks = call_args[1].get('callbacks', [])
        assert any('WandbLogger' in str(cb) for cb in callbacks)
```

**Step 2: Run test**

Run: `pytest tests/hpo/test_ray_tune_wandb.py -v`

**Step 3: Commit**

```bash
git add tests/hpo/test_ray_tune_wandb.py
git commit -m "test: add Ray Tune W&B integration test"
```

---

## Final Verification

### Task 5.1: End-to-End Test

**Step 1: Run full integration test**

```bash
# Set up test environment
export WANDB_API_KEY="test_key_1234567890abcdef"

# Run minimal HPO with W&B
ugro hpo sweep \
  --study-name security-test \
  --search-space config/llama_lora_hpo.yaml \
  --n-trials 2 \
  --wandb-project ugro-security-test \
  --model unsloth/tinyllama-bnb-4bit \
  --dry-run
```

**Step 2: Verify security measures**
- Check that API key is masked in output
- Verify project name validation works
- Ensure no API key leakage in logs

**Step 3: Clean up test data**

```bash
# Remove test W&B project if needed
wandb projects delete ugro-security-test
```

### Task 5.2: Documentation Update

**Files:**
- Modify: `docs/hyperparameter/W&B_Integration_Setup.md`

**Step 1: Add security section**

```markdown
## Security Considerations

### API Key Management
- API keys are validated on initialization
- Keys are automatically masked in logs and outputs
- Invalid keys prevent W&B initialization

### Project Name Validation
- Project names must be alphanumeric with hyphens/underscores
- Length限制: 1-128 characters
- Invalid names are rejected at CLI level

### Resource Cleanup
- All W&B runs are properly closed using context managers
- Cleanup occurs even on training failures
- No resource leaks from abandoned runs
```

**Step 2: Commit**

```bash
git add docs/hyperparameter/W&B_Integration_Setup.md
git commit -m "docs: add security considerations to W&B integration guide"
```

---

## Summary

This implementation plan addresses all critical and medium security issues identified in the W&B integration:

1. **Security Fixes**: API key validation, masking, and project name validation
2. **Resource Management**: Proper cleanup in finally blocks and context managers
3. **Architecture**: Abstraction layer to reduce coupling and improve testability
4. **Testing**: Comprehensive test suite with fixed imports and integration tests

Estimated completion time: 4-6 hours
All changes maintain backward compatibility while improving security and maintainability.
