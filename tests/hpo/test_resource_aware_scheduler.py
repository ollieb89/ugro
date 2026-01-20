import pytest
from ugro.hpo.schedulers import create_scheduler, MaxTokensPerTrialStopper, TimeoutStopper
from ray.tune.stopper import CombinedStopper

def test_create_scheduler_with_resource_limits():
    scheduler, stopper = create_scheduler(
        scheduler_type="asha",
        max_tokens=1000,
        timeout_seconds=1,
    )
    assert scheduler is not None
    # Stopper should be a CombinedStopper instance
    assert isinstance(stopper, CombinedStopper)
    # Token stop condition should trigger
    assert stopper("trial1", {"tokens_used": 2000}) is True
    # Timeout stop condition: simulate by directly using TimeoutStopper
    timeout = TimeoutStopper(timeout_seconds=0)  # immediate timeout
    assert timeout("t2", {}) is True

def test_create_scheduler_without_limits():
    scheduler, stopper = create_scheduler(scheduler_type="asha")
    assert scheduler is not None
    assert stopper is None
