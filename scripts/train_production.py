"""
Production-grade Multi-Node Distributed LLM Fine-tuning
Optimized for heterogeneous GPU cluster (RTX 5070 Ti, 4070, 3070 Ti)
Uses Unsloth + PyTorch DDP + LoRA
"""

from unsloth import FastLanguageModel
import os
import sys
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from transformers import (
    get_linear_schedule_with_warmup,
)
from datasets import load_dataset
from datetime import datetime
import argparse
import logging
from pathlib import Path


# Setup logging
def setup_logging(rank: int, log_dir: str) -> logging.Logger:
    """Configure logging for distributed training."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    log_file = Path(log_dir) / f"training_rank{rank}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO if rank == 0 else logging.WARNING,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler() if rank == 0 else logging.NullHandler(),
        ],
    )
    
    return logging.getLogger(__name__)


def setup_distributed(log_dir: str) -> tuple[int, int, int, logging.Logger]:
    """Initialize distributed training environment."""
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        print("Not a distributed environment. Use torchrun to launch.")
        sys.exit(1)
    
    dist.init_process_group("nccl")
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    # Set device
    torch.cuda.set_device(local_rank)
    
    logger = setup_logging(rank, log_dir=log_dir)
    
    if rank == 0:
        logger.info("=" * 60)
        logger.info("DISTRIBUTED TRAINING INITIALIZED")
        logger.info("=" * 60)
        logger.info(f"Total Nodes (World Size): {world_size}")
        logger.info(f"Current Rank: {rank}")
        logger.info(f"Local Rank: {local_rank}")
        logger.info(f"GPU: {torch.cuda.get_device_name(local_rank)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(local_rank).total_memory / 1e9:.2f} GB")
        logger.info("=" * 60)
    
    return rank, world_size, local_rank, logger


def cleanup_distributed():
    """Cleanup distributed training."""
    dist.destroy_process_group()


def load_model_with_lora(model_name: str, rank: int, logger):
    """
    Load model with LoRA using Unsloth optimization.
    
    Optimizations for your hardware:
    - 4-bit quantization (fits on all GPUs)
    - LoRA with conservative settings (fits on all GPUs)
    - Gradient checkpointing enabled
    """
    
    if rank == 0:
        logger.info(f"Loading model: {model_name}")
    
    max_seq_length = 2048
    
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            dtype=torch.float16,
            load_in_4bit=True,  # Critical: 4-bit quantization for VRAM efficiency
        )
    except Exception as e:
        if rank == 0:
            logger.error(f"Failed to load model: {e}")
        raise
    
    # Apply LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,                          # LoRA rank
        lora_alpha=32,                 # LoRA alpha
        target_modules=["q_proj", "v_proj"],  # Target projection layers
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",  # Unsloth's efficient gradient checkpointing
        random_state=42,
    )
    
    if rank == 0:
        logger.info(f"Model loaded successfully")
        model.print_trainable_parameters()
    
    return model, tokenizer, max_seq_length


def prepare_dataset(
    dataset_name: str,
    tokenizer,
    max_seq_length: int,
    rank: int,
    world_size: int,
    logger,
    split: str = "train",
):
    """
    Load and prepare dataset with DistributedSampler.
    Ensures each GPU gets unique data batches.
    """
    
    if rank == 0:
        logger.info(f"Loading dataset: {dataset_name}")
    
    # Load dataset
    if dataset_name == "wikitext":
        dataset = load_dataset("wikitext", "wikitext-2-v1", split=split)
    elif dataset_name == "openwebtext":
        # Note: This is a large dataset, consider subsetting
        dataset = load_dataset("openwebtext", split=split).select(range(1000))
    else:
        dataset = load_dataset(dataset_name, split=split)
    
    # Filter empty examples
    dataset = dataset.filter(lambda x: len(x.get("text", "")) > 0)
    
    # Tokenization function
    def tokenize_function(examples):
        texts = examples.get("text", [""] * len(examples.get("input_ids", [])))
        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=max_seq_length,
            padding="max_length",
            return_tensors=None,
        )
        # Set labels (same as input_ids for causal language modeling)
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized
    
    # Apply tokenization
    if rank == 0:
        logger.info("Tokenizing dataset...")
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names if hasattr(dataset, 'column_names') else [],
        desc="Tokenizing",
    )
    
    # Create DistributedSampler
    sampler = DistributedSampler(
        tokenized_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=42,
    )
    
    # DataLoader
    # Per-GPU batch size = 1 (recommended for Unsloth DDP)
    # Gradient accumulation provides effective batch size
    dataloader = DataLoader(
        tokenized_dataset,
        sampler=sampler,
        batch_size=1,
        pin_memory=True,
        num_workers=2,  # Parallel data loading
    )
    
    if rank == 0:
        logger.info(f"Dataset size: {len(tokenized_dataset)}")
        logger.info(f"Batches per epoch: {len(dataloader)}")
    
    return dataloader, sampler


def train_epoch(
    model,
    dataloader,
    optimizer,
    scheduler,
    epoch: int,
    total_epochs: int,
    rank: int,
    world_size: int,
    logger,
    gradient_accumulation_steps: int = 8,
):
    """Train for one epoch with gradient accumulation."""
    
    model.train()
    total_loss = 0
    accumulated_steps = 0
    
    for step, batch in enumerate(dataloader):
        # Move batch to GPU
        for key in batch.keys():
            batch[key] = batch[key].to(torch.cuda.current_device())
        
        # Forward pass
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            labels=batch["labels"],
        )
        loss = outputs.loss / gradient_accumulation_steps
        
        # Backward pass
        loss.backward()
        accumulated_steps += 1
        total_loss += loss.item() * gradient_accumulation_steps
        
        # Optimizer step every N accumulation steps
        if (step + 1) % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        # Logging (rank 0 only)
        if rank == 0 and (step + 1) % 10 == 0:
            avg_loss = total_loss / (step + 1)
            lr = scheduler.get_last_lr()[0]
            logger.info(
                f"Epoch [{epoch+1}/{total_epochs}] "
                f"Step [{step+1}/{len(dataloader)}] "
                f"Loss: {avg_loss:.4f} "
                f"LR: {lr:.2e}"
            )
    
    # Synchronize loss across all ranks
    loss_tensor = torch.tensor(total_loss, device=torch.cuda.current_device())
    dist.all_reduce(loss_tensor)
    avg_loss = loss_tensor.item() / world_size
    
    if rank == 0:
        logger.info(f"Epoch {epoch+1} completed - Avg Loss: {avg_loss:.4f}")
    
    return avg_loss


def save_checkpoint(
    model,
    optimizer,
    epoch: int,
    loss: float,
    rank: int,
    checkpoint_dir: str,
):
    """Save training checkpoint (rank 0 only)."""
    
    if rank != 0:
        return
    
    checkpoint_dir_path = Path(checkpoint_dir)
    checkpoint_dir_path.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.module.state_dict() if isinstance(model, DDP) else model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "timestamp": datetime.now().isoformat(),
    }
    
    checkpoint_path = checkpoint_dir_path / f"epoch_{epoch+1}_loss_{loss:.4f}.pt"
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")


def main(args):
    """Main training function."""
    
    # Setup
    rank, world_size, local_rank, logger = setup_distributed(log_dir=args.log_dir)
    
    # Load model
    model, tokenizer, max_seq_length = load_model_with_lora(
        args.model_name, rank, logger
    )
    
    # Move to GPU
    model = model.to(local_rank)
    
    # Wrap with DDP
    model = DDP(
        model,
        device_ids=[local_rank],
        find_unused_parameters=False,
        gradient_as_bucket_view=True,  # Optimization for NCCL
    )
    
    # Prepare data
    train_dataloader, train_sampler = prepare_dataset(
        args.dataset_name,
        tokenizer,
        max_seq_length,
        rank,
        world_size,
        logger,
        split="train",
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.01,
        betas=(0.9, 0.95),
    )
    
    # Learning rate scheduler
    num_training_steps = len(train_dataloader) * args.num_epochs
    num_warmup_steps = int(0.1 * num_training_steps)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    
    if rank == 0:
        logger.info(f"Total training steps: {num_training_steps}")
        logger.info(f"Warmup steps: {num_warmup_steps}")
    
    # Training loop
    best_loss = float("inf")
    
    for epoch in range(args.num_epochs):
        train_sampler.set_epoch(epoch)  # Shuffle differently each epoch
        
        epoch_loss = train_epoch(
            model,
            train_dataloader,
            optimizer,
            scheduler,
            epoch,
            args.num_epochs,
            rank,
            world_size,
            logger,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
        )
        
        # Save checkpoint
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            save_checkpoint(
                model,
                optimizer,
                epoch,
                epoch_loss,
                rank,
                checkpoint_dir=args.checkpoint_dir,
            )
        
        # Synchronization barrier
        dist.barrier()
    
    if rank == 0:
        logger.info("Training completed successfully")
    
    cleanup_distributed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multi-Node Distributed LLM Fine-tuning"
    )
    parser.add_argument(
        "--model-name",
        default="unsloth/tinyllama-bnb-4bit",
        type=str,
        help="Model name or path",
    )
    parser.add_argument(
        "--dataset-name",
        default="wikitext",
        type=str,
        choices=["wikitext", "openwebtext", "custom"],
        help="Dataset name",
    )
    parser.add_argument(
        "--learning-rate",
        default=2e-4,
        type=float,
        help="Learning rate",
    )
    parser.add_argument(
        "--num-epochs",
        default=3,
        type=int,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        default=8,
        type=int,
        help="Gradient accumulation steps (for larger effective batch size)",
    )
    parser.add_argument(
        "--log-dir",
        default="../logs",
        type=str,
        help="Directory for logs",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="../checkpoints",
        type=str,
        help="Directory for checkpoints",
    )
    
    args = parser.parse_args()
    main(args)
