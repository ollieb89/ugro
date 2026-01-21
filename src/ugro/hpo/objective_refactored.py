"""Refactored objective function using tracking interface."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from .tracking_interface import TrackingInterface, NoOpTracker, MultiTracker
from .mlflow_tracker import MLflowTracker
from .wandb_tracker import WandbTracker
from .objective import LoRAFinetuningObjective as BaseObjective

logger = logging.getLogger(__name__)


class LoRAFinetuningObjectiveRefactored(BaseObjective):
    """Refactored LoRA fine-tuning objective using tracking interface.
    
    This version demonstrates the new architecture with proper abstraction
    between the objective function and tracking backends.
    """
    
    def __init__(
        self,
        model_id: str,
        dataset_name: str,
        dataset_config: Optional[str] = None,
        output_dir: str = "./checkpoints",
        max_steps: int = 100,
        max_train_samples: int = 10000,
        max_eval_samples: int = 1000,
        use_mlflow: bool = True,
        use_wandb: bool = True,
        wandb_project: Optional[str] = None,
        target_modules: Optional[list[str]] = None,
        constraints: Optional[List[str]] = None,
        objectives: Optional[List[Dict[str, Any]]] = None,
        parameter_bounds: Optional[List["ParameterBound"]] = None,
        tracking_uri: Optional[str] = None,
        mlflow_experiment: Optional[str] = None,
    ):
        """Initialize with tracking interface.
        
        Args:
            model_id: HuggingFace model identifier
            dataset_name: Dataset name from HuggingFace Hub
            wandb_project: W&B project name (validated)
            tracking_uri: MLflow tracking server URI
            mlflow_experiment: MLflow experiment name
            Other args: Same as base class
        """
        # Initialize base class without tracking
        super().__init__(
            model_id=model_id,
            dataset_name=dataset_name,
            dataset_config=dataset_config,
            output_dir=output_dir,
            max_steps=max_steps,
            max_train_samples=max_train_samples,
            max_eval_samples=max_eval_samples,
            use_mlflow=False,  # We'll handle tracking through interface
            use_wandb=False,   # We'll handle tracking through interface
            target_modules=target_modules,
            constraints=constraints,
            objectives=objectives,
            parameter_bounds=parameter_bounds,
        )
        
        # Store tracking configuration
        self.tracking_uri = tracking_uri
        self.mlflow_experiment = mlflow_experiment
        self.wandb_project = wandb_project
        
        # Initialize tracking interface
        self.tracker = self._create_tracker(use_mlflow, use_wandb)
    
    def _create_tracker(self, use_mlflow: bool, use_wandb: bool) -> TrackingInterface:
        """Create appropriate tracker based on configuration."""
        trackers = []
        
        if use_mlflow:
            mlflow_tracker = MLflowTracker(
                tracking_uri=self.tracking_uri,
                experiment_name=self.mlflow_experiment,
                tags={"model_id": self.model_id, "dataset": self.dataset_name}
            )
            trackers.append(mlflow_tracker)
        
        if use_wandb and self.wandb_project:
            wandb_tracker = WandbTracker(
                project=self.wandb_project,
                config={
                    "model_id": self.model_id,
                    "dataset_name": self.dataset_name,
                    "max_steps": self.max_steps,
                }
            )
            trackers.append(wandb_tracker)
        
        if not trackers:
            logger.info("No tracking enabled, using NoOpTracker")
            return NoOpTracker()
        
        return MultiTracker(trackers)
    
    def __call__(self, params: Dict[str, Any]) -> Dict[str, float]:
        """Run trial using tracking interface."""
        # Import here to avoid circular imports
        from .objective import validate_constraints, get_penalty_metrics
        
        # Try to get Ray session
        try:
            from ray import tune
            trial_id = tune.get_trial_id() if tune.is_session_enabled() else "local"
        except (ImportError, RuntimeError):
            trial_id = "local"
        
        run_name = f"lora-trial-{trial_id}"
        
        # Apply conditional parameters
        if self.parameter_bounds:
            from ugro.hpo.search_space import apply_conditional_parameters
            original_params = params.copy()
            params = apply_conditional_parameters(params, self.parameter_bounds)
            if params != original_params:
                logger.info(f"Applied conditional parameters: {original_params} -> {params}")
        
        # Validate constraints
        if self.constraints:
            is_valid, violated = validate_constraints(params, self.constraints)
            if not is_valid:
                logger.warning(f"Trial violates constraints: {violated}")
                
                # Return penalty metrics
                penalty_metrics = get_penalty_metrics(self.objectives)
                
                # Log violation using tracking interface
                with self.tracker.start_run(run_name + "-violation"):
                    self.tracker.log_params(params)
                    self.tracker.log_metrics(penalty_metrics)
                    self.tracker.log_tags({"constraint_violations": str(violated)})
                
                return penalty_metrics
        
        # Run trial with tracking
        with self.tracker.start_run(run_name):
            # Log parameters
            self.tracker.log_params(params)
            
            # Run the actual training using base class logic
            # but with our tracking interface
            return self._run_training_with_tracking(params)
    
    def _run_training_with_tracking(self, params: Dict[str, Any]) -> Dict[str, float]:
        """Run training using tracking interface instead of direct MLflow/W&B calls."""
        # Import heavy dependencies
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import (
            AutoModelForCausalLM,
            EarlyStoppingCallback,
            Trainer,
            TrainingArguments,
        )
        import torch
        
        logger.info(f"Starting trial with params: {params}")
        
        # 1. Load base model
        logger.info(f"Loading base model: {self.model_id}")
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        
        # 2. Configure LoRA
        lora_config = LoraConfig(
            r=params.get("lora_r", 8),
            lora_alpha=params.get("lora_alpha", 16),
            lora_dropout=params.get("lora_dropout", 0.1),
            target_modules=self.target_modules,
            task_type=TaskType.CAUSAL_LM,
        )
        
        # 3. Apply LoRA
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
        # 4. Prepare datasets
        train_dataset, eval_dataset = self.train_dataset, self.eval_dataset
        
        # 5. Configure training
        training_args = TrainingArguments(
            output_dir=str(self.output_dir / "trial"),
            learning_rate=params.get("learning_rate", 5e-4),
            per_device_train_batch_size=params.get("batch_size", 4),
            per_device_eval_batch_size=params.get("batch_size", 4),
            gradient_accumulation_steps=params.get("gradient_accumulation_steps", 1),
            warmup_ratio=params.get("warmup_ratio", 0.1),
            max_steps=self.max_steps,
            logging_steps=10,
            evaluation_strategy="steps",
            eval_steps=20,
            save_steps=20,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to=[],  # Disable default reporting
        )
        
        # 6. Create trainer with custom callback for tracking
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            callbacks=[TrackingInterfaceCallback(self.tracker)],
        )
        
        # 7. Train
        trainer.train()
        
        # 8. Evaluate
        eval_results = trainer.evaluate()
        
        # 9. Calculate metrics
        eval_loss = eval_results["eval_loss"]
        eval_perplexity = torch.exp(torch.tensor(eval_loss)).item()
        train_loss = eval_results.get("train_loss", eval_loss)
        
        metrics = {
            "eval_loss": eval_loss,
            "eval_perplexity": eval_perplexity,
            "train_loss": train_loss,
        }
        
        # Log final metrics
        self.tracker.log_metrics(metrics)
        
        logger.info(f"Trial completed with metrics: {metrics}")
        return metrics


class TrackingInterfaceCallback:
    """HuggingFace Trainer callback for tracking interface."""
    
    def __init__(self, tracker: TrackingInterface):
        self.tracker = tracker
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called on each log."""
        if logs:
            # Filter out non-numeric logs
            metrics = {k: v for k, v in logs.items() if isinstance(v, (int, float))}
            if metrics:
                self.tracker.log_metrics(metrics, step=state.global_step)
