#!/usr/bin/env python3
"""Batch job submission helper for UGRO.

Usage:
    python scripts/batch_submit.py --model llama-7b --epochs 1 --count 5
    python scripts/batch_submit.py --config configs/batch.yaml
"""

import argparse
import itertools
from pathlib import Path
import sys

# Add src to path so we can import ugro
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ugro.cli import app
from ugro.queues import Job, JobPriority, JobResources
import typer
from rich.console import Console

console = Console()


def generate_job_variants(params: dict, count: int = None) -> list[dict]:
    """Generate cartesian product of parameter combinations."""
    # Filter out None values and create ranges
    clean_params = {k: v for k, v in params.items() if v is not None}
    
    # Convert single values to lists for cartesian product
    for k, v in clean_params.items():
        if isinstance(v, (list, tuple)):
            continue
        clean_params[k] = [v]
    
    # Generate all combinations
    keys = list(clean_params.keys())
    values = list(clean_params.values())
    combinations = list(itertools.product(*values))
    
    # Convert to dicts
    result = [dict(zip(keys, combo)) for combo in combinations]
    
    # Limit by count if specified
    if count and len(result) > count:
        result = result[:count]
    
    return result


def submit_jobs(job_configs: list[dict], dry_run: bool = False):
    """Submit multiple jobs to the queue."""
    from ugro.config import load_config
    from ugro.queues import SQLiteJobQueue, RedisJobQueue
    
    # Load config and initialize queue like CLI does
    config = load_config()
    cluster_conf = config.get("cluster", {})
    q_conf = cluster_conf.get("queue", {})
    q_type = q_conf.get("type", "sqlite")
    
    if q_type == "redis":
        host = q_conf.get("redis_host", "localhost")
        port = q_conf.get("redis_port", 6379)
        db = q_conf.get("redis_db", 0)
        password = q_conf.get("redis_password", None)
        queue = RedisJobQueue(host=host, port=port, db=db, password=password)
    else:
        from ugro.database import Database
        queue = SQLiteJobQueue(db_path=str(Database().db_path))
    
    console.print(f"Submitting {len(job_configs)} jobs to {q_type} queue...", style="bold blue")
    
    submitted = []
    for i, job_cfg in enumerate(job_configs, 1):
        # Generate job name
        name = job_cfg.pop("name", f"batch_job_{i}")
        model = job_cfg.pop("model", "llama-7b")
        dataset = job_cfg.pop("dataset", "wikitext")
        epochs = job_cfg.pop("epochs", 1)
        lr = job_cfg.pop("lr", 0.0002)
        verbose = job_cfg.pop("verbose", False)
        
        # Build command
        cmd_parts = [
            "ugro", "launch", "--now",
            "--name", name,
            "--model", model,
            "--dataset", dataset,
            "--epochs", str(epochs),
            "--lr", str(lr),
        ]
        if verbose:
            cmd_parts.append("--verbose")
        
        job = Job(
            name=name,
            command=" ".join(cmd_parts),
            priority=JobPriority.NORMAL,
            resources=JobResources(gpu_count=1),
            metadata={
                "model": model,
                "dataset": dataset,
                "epochs": epochs,
                "learning_rate": lr,
                "verbose": verbose,
                **job_cfg  # Include any extra params
            }
        )
        
        if dry_run:
            console.print(f"[DRY RUN] Would submit: {name} ({model})", style="yellow")
            submitted.append(job.id)
        else:
            job_id = queue.submit(job)
            submitted.append(job_id)
            console.print(f"✅ Submitted: {name} -> {job_id[:8]}...", style="green")
    
    console.print(f"\nSubmitted {len(submitted)} jobs total.", style="bold green")
    return submitted


def main():
    parser = argparse.ArgumentParser(description="Batch submit UGRO jobs")
    parser.add_argument("--model", default="llama-7b", help="Model name")
    parser.add_argument("--dataset", default="wikitext", help="Dataset name")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.0002, help="Learning rate")
    parser.add_argument("--count", type=int, help="Number of jobs to submit")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be submitted without actually submitting")
    parser.add_argument("--config", help="YAML config file with batch parameters")
    
    args = parser.parse_args()
    
    if args.config:
        import yaml
        with open(args.config) as f:
            config = yaml.safe_load(f)
        job_configs = generate_job_variants(config, args.count)
    else:
        # Single config repeated count times
        base_config = {
            "model": args.model,
            "dataset": args.dataset,
            "epochs": args.epochs,
            "lr": args.lr,
            "verbose": args.verbose,
        }
        if args.count:
            job_configs = [base_config.copy() for _ in range(args.count)]
            # Add unique names
            for i, cfg in enumerate(job_configs, 1):
                cfg["name"] = f"{args.model.replace('/', '_')}_batch_{i}"
        else:
            base_config["name"] = f"{args.model.replace('/', '_')}_single"
            job_configs = [base_config]
    
    submitted = submit_jobs(job_configs, dry_run=args.dry_run)
    
    if not args.dry_run:
        console.print("\nNext steps:")
        console.print("1. Run worker: pixi run ugro run-worker")
        console.print("2. Monitor queue: pixi run python -m ugro.cli queue list")
        console.print("3. View logs: pixi run ugro logs <job_name>")


if __name__ == "__main__":
    main()
