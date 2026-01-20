"""HPO Objective Functions.

Production-ready objective functions for hyperparameter optimization,
including LoRA fine-tuning for LLMs with Ray Tune and MLflow integration.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

import torch

if TYPE_CHECKING:
    from datasets import Dataset
    from transformers import PreTrainedModel, PreTrainedTokenizer

logger = logging.getLogger(__name__)


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
        target_modules: Optional[list[str]] = None,
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
            target_modules: LoRA target modules (default: q_proj, v_proj)
        """
        self.model_id = model_id
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.output_dir = Path(output_dir)
        self.max_steps = max_steps
        self.max_train_samples = max_train_samples
        self.max_eval_samples = max_eval_samples
        self.use_mlflow = use_mlflow
        self.target_modules = target_modules or ["q_proj", "v_proj"]

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

        # Try to get Ray session (may be None if running locally)
        try:
            from ray.air import session

            trial_id = session.get_trial_id() if session else "local"
        except ImportError:
            session = None
            trial_id = "local"

        run_name = f"lora-trial-{trial_id}"

        # Optional MLflow tracking
        mlflow_context = self._get_mlflow_context(run_name)

        with mlflow_context:
            try:
                if self.use_mlflow:
                    import mlflow

                    mlflow.log_params(params)

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

                # 10. Log to MLflow
                if self.use_mlflow:
                    import mlflow

                    mlflow.log_metrics(metrics)

                # 11. Report to Ray session
                if session is not None:
                    session.report(metrics)

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

                # Report failure metrics to Ray
                if session is not None:
                    session.report(
                        {"eval_loss": float("inf"), "eval_perplexity": float("inf")}
                    )
                raise

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


class RayTuneReportCallback:
    """HuggingFace Trainer callback for Ray Tune session reporting.

    Reports evaluation metrics to Ray Tune during training, enabling
    schedulers like ASHA to make early stopping decisions.
    """

    def __init__(self):
        """Initialize callback."""
        try:
            from ray.air import session

            self._session = session
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

        self._session.report(report_metrics)


def create_objective_factory(
    model_id: str,
    dataset_name: str,
    **kwargs: Any,
) -> Callable[[Dict[str, Any]], Dict[str, float]]:
    """Factory function to create objective callable.

    This is the entry point for the CLI and programmatic usage.

    Args:
        model_id: HuggingFace model identifier
        dataset_name: Dataset name from HuggingFace Hub
        **kwargs: Additional arguments passed to LoRAFinetuningObjective

    Returns:
        Callable objective function for HPO
    """
    objective = LoRAFinetuningObjective(
        model_id=model_id,
        dataset_name=dataset_name,
        **kwargs,
    )
    return objective
