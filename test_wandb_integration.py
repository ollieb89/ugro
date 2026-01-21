#!/usr/bin/env python3
"""Test W&B integration with UGRO HPO.

This script runs a small HPO sweep to verify W&B integration is working.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_wandb_integration():
    """Test W&B integration with a small HPO run."""
    
    # Set W&B project for testing
    os.environ["WANDB_PROJECT"] = "ugro-hpo-test"
    
    # Import required modules
    from ugro.hpo.objective import LoRAFinetuningObjective
    from ugro.hpo.config import HPOConfig, OptimizerAlgorithm, ParameterBound
    
    print("🧪 Testing W&B Integration...")
    print(f"   W&B Project: {os.environ.get('WANDB_PROJECT')}")
    
    # Create a minimal search space
    search_space = [
        ParameterBound(
            name="learning_rate",
            type="float",
            min=1e-5,
            max=1e-3,
            default=5e-4,
            log=True,
        ),
        ParameterBound(
            name="lora_r",
            type="int",
            min=4,
            max=16,
            default=8,
        ),
    ]
    
    # Create HPO config
    hpo_config = HPOConfig(
        study_name="wandb-test-study",
        search_space=search_space,
        algorithm=OptimizerAlgorithm.RANDOM,  # Use random for speed
        n_trials=2,  # Just 2 trials for testing
        parallel_jobs=1,
        max_steps_per_trial=10,  # Very short for testing
        wandb_project="ugro-hpo-test",
    )
    
    # Create objective with W&B enabled
    objective = LoRAFinetuningObjective(
        model_id="unsloth/tinyllama-bnb-4bit",  # Small model for testing
        dataset_name="wikitext",
        max_steps=10,  # Very short
        use_mlflow=False,  # Disable MLflow for this test
        use_wandb=True,
    )
    
    print("\n📊 Running 2 trials with W&B tracking...")
    
    # Run a single trial manually to test W&B logging
    print("\n🔬 Trial 1:")
    params1 = {"learning_rate": 5e-4, "lora_r": 8}
    metrics1 = objective(params1)
    print(f"   Metrics: {metrics1}")
    
    print("\n🔬 Trial 2:")
    params2 = {"learning_rate": 1e-4, "lora_r": 16}
    metrics2 = objective(params2)
    print(f"   Metrics: {metrics2}")
    
    print("\n✅ W&B Integration Test Complete!")
    print("\n📋 Check your W&B dashboard:")
    print(f"   Project: {os.environ.get('WANDB_PROJECT')}")
    print("   You should see 2 runs with logged parameters and metrics")
    
    return True


if __name__ == "__main__":
    # Check if wandb is installed
    try:
        import wandb
        print("✅ W&B is installed")
    except ImportError:
        print("❌ W&B not installed. Install with: pixi add wandb")
        sys.exit(1)
    
    # Run the test
    success = test_wandb_integration()
    
    if success:
        print("\n🎉 Test completed successfully!")
    else:
        print("\n❌ Test failed!")
        sys.exit(1)
