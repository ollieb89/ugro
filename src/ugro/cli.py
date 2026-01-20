"""UGRO CLI: Main command interface.

Provides command-line interface for GPU cluster orchestration and distributed training.

Usage:
    ugro health                          # Check cluster health
    ugro launch --name exp1 --epochs 3   # Launch training job
    ugro logs exp1                       # View job logs
    ugro results exp1                    # Show job results
    ugro status                          # Display cluster status
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import click

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Any

# Import from same package using relative imports
from .agent import UGROAgent
from .config import load_config


@click.group()
@click.pass_context
def cli(ctx: click.Context) -> None:
    """UGRO: Unified GPU Resource Orchestrator.
    
    Personal-scale GPU cluster orchestration tool for distributed training.
    
    Quick Start:
        ugro health          # Check cluster health
        ugro launch          # Start training job
        ugro logs <name>     # View training logs
        ugro results <name>  # Display results
    """
    ctx.ensure_object(dict)
    ctx.obj["agent"] = UGROAgent()
    ctx.obj["config"] = load_config()


@cli.command()
@click.pass_context
def health(ctx: click.Context) -> None:
    """Check cluster health status."""
    agent: UGROAgent = ctx.obj["agent"]
    
    click.echo("\n🔍 Cluster Health Check")
    click.echo("=" * 60)
    
    health_status: Mapping[str, Any] = agent.check_cluster_health()
    
    for node_name, status in health_status.items():
        symbol = "✓" if status["healthy"] else "❌"
        click.echo(f"{symbol} {node_name:15} {status['message']}")
    
    click.echo()


@cli.command()
@click.option("--name", required=True, help="Job name")
@click.option("--model", default="unsloth/tinyllama-bnb-4bit", help="Model name")
@click.option("--dataset", default="wikitext", help="Dataset name")
@click.option("--epochs", default=1, type=int, help="Number of epochs")
@click.option("--lr", default=0.0002, type=float, help="Learning rate")
@click.option("--verbose", is_flag=True, help="Verbose output")
@click.pass_context
def launch(
    ctx: click.Context,
    name: str,
    model: str,
    dataset: str,
    epochs: int,
    lr: float,
    verbose: bool,
) -> None:
    """Launch distributed training across cluster."""
    agent: UGROAgent = ctx.obj["agent"]
    
    success = agent.launch_training(
        job_name=name,
        model=model,
        dataset=dataset,
        epochs=epochs,
        learning_rate=lr,
        verbose=verbose,
    )
    
    sys.exit(0 if success else 1)


@cli.command()
@click.argument("job_name")
@click.option("--rank", default=None, type=int, help="Specific rank to view")
@click.pass_context
def logs(ctx: click.Context, job_name: str, rank: int | None) -> None:
    """View training logs for a job."""
    agent: UGROAgent = ctx.obj["agent"]
    agent.display_logs(job_name, rank)


@cli.command()
@click.argument("job_name")
@click.pass_context
def results(ctx: click.Context, job_name: str) -> None:
    """Show results summary for a job."""
    agent: UGROAgent = ctx.obj["agent"]
    agent.display_results(job_name)


@cli.command()
@click.pass_context
def status(ctx: click.Context) -> None:
    """Show current cluster status."""
    agent: UGROAgent = ctx.obj["agent"]
    agent.display_status()


def main() -> None:
    """Main entry point for CLI."""
    cli(obj={})


if __name__ == "__main__":
    main()
