"""HPO Objective Functions.

Production-ready objective functions for hyperparameter optimization,
including LoRA fine-tuning for LLMs with Ray Tune and MLflow integration.
"""

from __future__ import annotations

import ast
import logging
import operator
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, List, Tuple

import torch

if TYPE_CHECKING:
    from datasets import Dataset
    from transformers import PreTrainedModel, PreTrainedTokenizer

# Optional W&B import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

logger = logging.getLogger(__name__)


def validate_constraints(
    params: Dict[str, Any], 
    constraints: List[str]
) -> Tuple[bool, List[str]]:
    """Validate parameter constraints.
    
    Args:
        params: Dictionary of parameter names to values
        constraints: List of constraint expressions (e.g., "lora_alpha >= lora_r")
        
    Returns:
        Tuple of (is_valid, list_of_violated_constraints)
        
    Note:
        Uses safe evaluation with AST parsing to avoid code injection.
        Only supports basic comparison operators: >=, <=, >, <, ==, !=
    """
    if not constraints:
        return True, []
    
    # Map operator strings to actual operator functions
    operators = {
        ast.Gt: operator.gt,
        ast.GtE: operator.ge,
        ast.Lt: operator.lt,
        ast.LtE: operator.le,
        ast.Eq: operator.eq,
        ast.NotEq: operator.ne,
    }
    
    violated = []
    
    for constraint in constraints:
        try:
            # Parse the constraint expression
            tree = ast.parse(constraint, mode='eval')
            
            if not isinstance(tree.body, ast.Compare):
                logger.warning(f"Invalid constraint format: {constraint}")
                violated.append(constraint)
                continue
            
            # Extract left operand, operator, and right operand
            left = tree.body.left
            ops = tree.body.ops
            comparators = tree.body.comparators
            
            # Evaluate left operand (should be a parameter name)
            if isinstance(left, ast.Name):
                left_val = params.get(left.id)
                if left_val is None:
                    logger.warning(f"Parameter '{left.id}' not found in params for constraint: {constraint}")
                    violated.append(constraint)
                    continue
            else:
                logger.warning(f"Invalid left operand in constraint: {constraint}")
                violated.append(constraint)
                continue
            
            # Evaluate each comparison
            for op, comp in zip(ops, comparators):
                # Get the operator function
                op_type = type(op)
                if op_type not in operators:
                    logger.warning(f"Unsupported operator in constraint: {constraint}")
                    violated.append(constraint)
                    continue
                
                # Evaluate right operand
                if isinstance(comp, ast.Name):
                    right_val = params.get(comp.id)
                    if right_val is None:
                        logger.warning(f"Parameter '{comp.id}' not found in params for constraint: {constraint}")
                        violated.append(constraint)
                        continue
                elif isinstance(comp, ast.Constant):
                    right_val = comp.value
                else:
                    logger.warning(f"Invalid right operand in constraint: {constraint}")
                    violated.append(constraint)
                    continue
                
                # Apply the operator
                op_func = operators[op_type]
                if not op_func(left_val, right_val):
                    violated.append(constraint)
                    break
                # For chained comparisons, update left_val for next comparison
                left_val = right_val
                
        except Exception as e:
            logger.error(f"Error evaluating constraint '{constraint}': {e}")
            violated.append(constraint)
    
    is_valid = len(violated) == 0
    return is_valid, violated


def get_penalty_metrics(
    objectives: List[Dict[str, Any]], 
    penalty_value: float = 1e6
) -> Dict[str, float]:
    """Generate penalty metrics for constraint violations.
    
    Args:
        objectives: List of objective dictionaries with 'direction' key
        penalty_value: Large penalty value to use
        
    Returns:
        Dictionary of objective names with penalty values
    """
    metrics = {}
    for obj in objectives:
        name = obj.get("name", "eval_loss")
        direction = obj.get("direction", "minimize")
        
        if direction == "minimize":
            metrics[name] = penalty_value
        else:  # maximize
            metrics[name] = -penalty_value
    
    return metrics


class BaseObjective(ABC):
    """Abstract base class for HPO objective functions."""

    @abstractmethod
    def __call__(self, params: Dict[str, Any]) -> Dict[str, float]:
        """Execute training with given hyperparameters.

        Args:
            params: Dictionary of hyperparameter values

        Returns:
            Dictionary of metric name -> value
        """
        pass


class LoRAFinetuningObjective(BaseObjective):
    """Production objective function for LoRA hyperparameter tuning.

    Implements the training loop from UGRO_HPO_Advanced.md with:
    - Ray Tune session reporting for schedulers
    - MLflow experiment tracking
    - Early stopping support
    - Mixed precision training (bf16)

    Example:
        objective = LoRAFinetuningObjective(
            model_id="meta-llama/Llama-2-7b-hf",
            dataset_name="wikitext",
            max_steps=1000,
        )
        metrics = objective({"lora_r": 16, "lora_alpha": 32, ...})
    """

    def __init__(
        self,
        model_id: str = "meta-llama/Llama-2-7b-hf",
        dataset_name: str = "wikitext",
        dataset_config: str = "wikitext-2-raw-v1",
        output_dir: str = "./checkpoints",
        max_steps: int = 1000,
        max_train_samples: int = 10000,
        max_eval_samples: int = 1000,
        use_mlflow: bool = True,
        use_wandb: bool = True,
        target_modules: Optional[list[str]] = None,
        constraints: Optional[List[str]] = None,
        objectives: Optional[List[Dict[str, Any]]] = None,
        parameter_bounds: Optional[List["ParameterBound"]] = None,
    ):
        """Initialize the LoRA fine-tuning objective.

        Args:
            model_id: HuggingFace model identifier
            dataset_name: Dataset name from HuggingFace Hub
            dataset_config: Dataset configuration name
            output_dir: Base directory for checkpoints
            max_steps: Maximum training steps per trial
            max_train_samples: Limit training samples (for speed)
            max_eval_samples: Limit evaluation samples
            use_mlflow: Enable MLflow logging
            use_wandb: Enable W&B logging
            target_modules: LoRA target modules (default: q_proj, v_proj)
            constraints: List of constraint expressions to validate
            objectives: List of objective dictionaries with direction info
            parameter_bounds: List of parameter bounds for conditional logic
        """
        self.model_id = model_id
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.output_dir = Path(output_dir)
        self.max_steps = max_steps
        self.max_train_samples = max_train_samples
        self.max_eval_samples = max_eval_samples
        self.use_mlflow = use_mlflow
        self.use_wandb = use_wandb
        self.target_modules = target_modules or ["q_proj", "v_proj"]
        self.constraints = constraints or []
        self.objectives = objectives or [{"name": "eval_loss", "direction": "minimize"}]
        self.parameter_bounds = parameter_bounds or []

        # Lazy-loaded resources
        self._tokenizer: Optional[PreTrainedTokenizer] = None
        self._train_dataset: Optional[Dataset] = None
        self._eval_dataset: Optional[Dataset] = None

    @property
    def tokenizer(self) -> PreTrainedTokenizer:
        """Lazy-load tokenizer."""
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            logger.info(f"Loading tokenizer: {self.model_id}")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
        return self._tokenizer

    def _prepare_datasets(self) -> tuple[Dataset, Dataset]:
        """Tokenize and prepare datasets (cached after first call)."""
        if self._train_dataset is not None and self._eval_dataset is not None:
            return self._train_dataset, self._eval_dataset

        from datasets import load_dataset

        logger.info(f"Loading dataset: {self.dataset_name}/{self.dataset_config}")
        dataset = load_dataset(self.dataset_name, self.dataset_config)

        def tokenize_fn(batch: Dict[str, Any]) -> Dict[str, Any]:
            return self.tokenizer(
                batch["text"],
                max_length=512,
                truncation=True,
                padding="max_length",
            )

        # Tokenize training split
        train_split = dataset["train"].map(
            tokenize_fn,
            batched=True,
            remove_columns=["text"],
            desc="Tokenizing train",
        )
        if len(train_split) > self.max_train_samples:
            train_split = train_split.shuffle(seed=42).select(range(self.max_train_samples))

        # Tokenize validation split
        eval_split_name = "validation" if "validation" in dataset else "test"
        eval_split = dataset[eval_split_name].map(
            tokenize_fn,
            batched=True,
            remove_columns=["text"],
            desc="Tokenizing eval",
        )
        if len(eval_split) > self.max_eval_samples:
            eval_split = eval_split.shuffle(seed=42).select(range(self.max_eval_samples))

        self._train_dataset = train_split
        self._eval_dataset = eval_split

        logger.info(
            f"Prepared datasets: train={len(train_split)}, eval={len(eval_split)}"
        )
        return self._train_dataset, self._eval_dataset

    def __call__(self, params: Dict[str, Any]) -> Dict[str, float]:
        """Run LoRA fine-tuning trial with given hyperparameters.

        Args:
            params: Dictionary with keys like lora_r, lora_alpha, lora_dropout,
                   learning_rate, batch_size, warmup_ratio, etc.

        Returns:
            Dictionary with eval_loss, eval_perplexity, train_loss
        """
        # Import heavy dependencies only when needed
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import (
            AutoModelForCausalLM,
            EarlyStoppingCallback,
            Trainer,
            TrainingArguments,
        )
        from ugro.hpo.search_space import apply_conditional_parameters

        # Try to get Ray session (may be None if running locally)
        try:
            from ray import tune

            trial_id = tune.get_trial_id() if tune.is_session_enabled() else "local"
            session = tune
        except (ImportError, RuntimeError):
            session = None
            trial_id = "local"

        run_name = f"lora-trial-{trial_id}"

        # Apply conditional parameter logic
        if self.parameter_bounds:
            original_params = params.copy()
            params = apply_conditional_parameters(params, self.parameter_bounds)
            if params != original_params:
                logger.info(f"Applied conditional parameters: {original_params} -> {params}")

        # Validate constraints before training
        if self.constraints:
            is_valid, violated = validate_constraints(params, self.constraints)
            if not is_valid:
                logger.warning(f"Trial violates constraints: {violated}")
                logger.warning(f"Trial parameters: {params}")
                
                # Return penalty metrics for constraint violations
                penalty_metrics = get_penalty_metrics(self.objectives)
                
                # Log to MLflow if enabled
                if self.use_mlflow:
                    import mlflow
                    mlflow.set_tag("constraint_violations", str(violated))
                    mlflow.log_metrics(penalty_metrics)
                
                # Log to W&B if enabled
                if self.use_wandb and WANDB_AVAILABLE and os.getenv("WANDB_PROJECT"):
                    try:
                        import wandb
                        wandb.log({"constraint_violations": str(violated)})
                        wandb.log(penalty_metrics)
                    except Exception as e:
                        logger.warning(f"Failed to log to W&B: {e}")
                
                # Cleanup W&B if it was initialized
                if self.use_wandb and os.getenv("WANDB_PROJECT"):
                    try:
                        import wandb
                        if wandb.run is not None:
                            wandb.finish()
                    except Exception as cleanup_error:
                        logger.warning(f"Failed to cleanup W&B run: {cleanup_error}")
                
                return penalty_metrics

        # Optional MLflow and W&B tracking
        mlflow_context = self._get_mlflow_context(run_name)
        wandb_run = self._get_wandb_context(run_name)

        with mlflow_context:
            try:
                if self.use_mlflow:
                    import mlflow
                    mlflow.log_params(params)
                
                # Log parameters to W&B
                if self.use_wandb and wandb_run:
                    wandb_run.config.update(params)

                logger.info(f"Starting trial with params: {params}")

                # 1. Load base model
                logger.info(f"Loading base model: {self.model_id}")
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    torch_dtype=torch.float16,
                    device_map="auto",
                )

                # 2. Apply LoRA configuration
                # Allow target_modules from params for HPO sweeping
                target_modules = params.get("target_modules", self.target_modules)
                
                lora_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=int(params.get("lora_r", 16)),
                    lora_alpha=float(params.get("lora_alpha", 32)),
                    lora_dropout=float(params.get("lora_dropout", 0.1)),
                    bias="none",
                    target_modules=target_modules,
                )

                model = get_peft_model(model, lora_config)
                model.print_trainable_parameters()

                # 3. Prepare datasets
                train_dataset, eval_dataset = self._prepare_datasets()

                # 4. Training arguments
                training_args = TrainingArguments(
                    output_dir=str(self.output_dir / run_name),
                    learning_rate=float(params.get("learning_rate", 5e-4)),
                    per_device_train_batch_size=int(params.get("batch_size", 16)),
                    per_device_eval_batch_size=32,
                    num_train_epochs=1,
                    max_steps=self.max_steps,
                    warmup_ratio=float(params.get("warmup_ratio", 0.05)),
                    weight_decay=float(params.get("weight_decay", 0.01)),
                    logging_steps=50,
                    eval_strategy="steps",
                    eval_steps=100,
                    save_strategy="steps",
                    save_steps=100,
                    load_best_model_at_end=True,
                    metric_for_best_model="eval_loss",
                    gradient_accumulation_steps=4,
                    bf16=torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
                    fp16=not torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
                    seed=42,
                    report_to="none",  # We handle tracking ourselves
                )

                # 5. Build callbacks
                callbacks = [
                    EarlyStoppingCallback(
                        early_stopping_patience=int(
                            params.get("early_stopping_patience", 5)
                        )
                    ),
                ]

                # Add Ray reporting callback if available
                if session is not None:
                    callbacks.append(RayTuneReportCallback())

                # 6. Create trainer
                trainer = Trainer(
                    model=model,
                    args=training_args,
                    train_dataset=train_dataset,
                    eval_dataset=eval_dataset,
                    callbacks=callbacks,
                )

                # 7. Train
                logger.info("Starting training...")
                result = trainer.train()

                # 8. Final evaluation
                eval_result = trainer.evaluate()

                # 9. Compute metrics
                eval_loss = eval_result["eval_loss"]
                eval_perplexity = torch.exp(torch.tensor(eval_loss)).item()

                metrics = {
                    "eval_loss": eval_loss,
                    "eval_perplexity": eval_perplexity,
                    "train_loss": result.training_loss,
                }

                # 10. Log to MLflow and W&B
                if self.use_mlflow:
                    import mlflow
                    mlflow.log_metrics(metrics)
                
                if self.use_wandb and wandb_run:
                    wandb_run.log(metrics)

                logger.info(f"Trial complete: {metrics}")
                
                # Cleanup model to free VRAM for next trial
                del model
                del trainer
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
                return metrics

            except Exception as e:
                logger.error(f"Trial failed: {e}", exc_info=True)
                raise
            finally:
                # Ensure W&B run is properly finished
                if self.use_wandb and wandb_run:
                    try:
                        import wandb
                        if wandb.run is not None:
                            wandb.finish()
                    except Exception as cleanup_error:
                        logger.warning(f"Failed to cleanup W&B run: {cleanup_error}")

    def _get_mlflow_context(self, run_name: str) -> Any:
        """Get MLflow run context or null context.

        Args:
            run_name: Name for the MLflow run

        Returns:
            Context manager for the run
        """
        if self.use_mlflow:
            try:
                import mlflow

                return mlflow.start_run(run_name=run_name, nested=True)
            except ImportError:
                logger.warning("MLflow not installed, disabling tracking")
                self.use_mlflow = False

        # Return null context
        from contextlib import nullcontext

        return nullcontext()

    def _get_wandb_context(self, run_name: str):
        """Get W&B run context or null context.

        Args:
            run_name: Name for the W&B run

        Returns:
            Context manager for the run
        """
        if self.use_wandb and WANDB_AVAILABLE and os.getenv("WANDB_PROJECT"):
            try:
                # Initialize wandb run
                wandb.init(
                    project=os.getenv("WANDB_PROJECT"),
                    name=run_name,
                    reinit=True,
                    config={
                        "model_id": self.model_id,
                        "dataset_name": self.dataset_name,
                        "max_steps": self.max_steps,
                    }
                )
                return wandb
            except Exception as e:
                logger.warning(f"Failed to initialize W&B: {e}")
                self.use_wandb = False

        # Return null context if W&B not available
        from contextlib import nullcontext
        return nullcontext()


class RayTuneReportCallback:
    """HuggingFace Trainer callback for Ray Tune session reporting.

    Reports evaluation metrics to Ray Tune during training, enabling
    schedulers like ASHA to make early stopping decisions.
    """

    def __init__(self):
        """Initialize callback."""
        try:
            from ray import tune

            self._session = tune
        except ImportError:
            self._session = None
            logger.warning("Ray not available, RayTuneReportCallback will be no-op")

    def on_evaluate(
        self,
        args: Any,
        state: Any,
        control: Any,
        metrics: Optional[Dict[str, float]] = None,
        **kwargs: Any,
    ) -> None:
        """Called after evaluation step.

        Args:
            args: TrainingArguments
            state: TrainerState
            control: TrainerControl
            metrics: Evaluation metrics dictionary
        """
        if self._session is None or metrics is None:
            return

        # Report intermediate results to Ray
        report_metrics = {
            "eval_loss": metrics.get("eval_loss", float("inf")),
            "global_step": state.global_step,
        }

        # Add perplexity if loss is available
        if "eval_loss" in metrics:
            report_metrics["eval_perplexity"] = torch.exp(
                torch.tensor(metrics["eval_loss"])
            ).item()

        try:
            self._session.report(report_metrics)
        except (AttributeError, RuntimeError):
            # Ray Tune API changed or not in session context
            pass


def create_objective_factory(
    model_id: str,
    dataset_name: str,
    constraints: Optional[List[str]] = None,
    objectives: Optional[List[Dict[str, Any]]] = None,
    **kwargs: Any,
) -> Callable[[Dict[str, Any]], Dict[str, float]]:
    """Factory function to create objective callable.

    This is the entry point for the CLI and programmatic usage.

    Args:
        model_id: HuggingFace model identifier
        dataset_name: Dataset name from HuggingFace Hub
        constraints: List of constraint expressions to validate
        objectives: List of objective dictionaries with direction info
        **kwargs: Additional arguments passed to LoRAFinetuningObjective

    Returns:
        Callable objective function for HPO
    """
    objective = LoRAFinetuningObjective(
        model_id=model_id,
        dataset_name=dataset_name,
        constraints=constraints,
        objectives=objectives,
        **kwargs,
    )
    return objective
